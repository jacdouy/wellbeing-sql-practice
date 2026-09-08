### @Q1 Daily flagged volume by category (v1)
-- Query: 
SELECT
  CASE
    WHEN c.created_at < '2026-07-01' THEN '1 June'
    WHEN c.created_at < '2026-07-16' THEN '2 Jul 1-15'
    WHEN c.created_at < '2026-08-01' THEN '3 Jul 16-31'
    ELSE '4 Aug'
  END AS period,
  e.predicted_category,
  ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT c.created_at::DATE), 1) AS flags_per_day
FROM classifier_events e
JOIN conversations c ON e.conversation_id = c.conversation_id
WHERE e.classifier_version = 'v1' AND e.flagged
GROUP BY 1, 2
ORDER BY 2, 1;
--
-- Finding: Emotional_dependence and sycophancy_concern rise through early
-- July, then fall below their June levels after July 16. Self_harm,
-- suicide, and disordered_eating drift upward across the whole window.
-- July 16 is the model_version switch (day 45). Hypothesis: the new model
-- is less sycophantic and less dependence-inviting, which lowers flags in
-- the two model-behavior categories; user-state categories are unaffected.
-- To test in Q11.

### Q2 v1 Precision and recall at 0.5 / 0.7 / 0.9
--Query:
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
scored AS (
  SELECT e.conversation_id,
         e.score,
         (fr.reviewed_severity >= 2) AS is_positive
  FROM classifier_events e
  JOIN first_reviews fr USING (conversation_id)
  WHERE e.classifier_version = 'v1'
)
SELECT threshold, tp, fp, fn,
       ROUND(tp * 1.0 / NULLIF(tp + fp, 0), 3) AS precision,
       ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 3) AS recall
FROM (
  SELECT t.threshold,
         SUM(CASE WHEN s.score >= t.threshold AND s.is_positive     THEN 1 ELSE 0 END) AS tp,
         SUM(CASE WHEN s.score >= t.threshold AND NOT s.is_positive THEN 1 ELSE 0 END) AS fp,
         SUM(CASE WHEN s.score <  t.threshold AND s.is_positive     THEN 1 ELSE 0 END) AS fn
  FROM scored s
  CROSS JOIN (VALUES (0.5), (0.7), (0.9)) AS t(threshold)
  GROUP BY 1
)
ORDER BY threshold;
--
-- FINDINGS: at the reviewed sample level
-- Sample level 0.9 is too restrictive only flagging 1062 true positives and sacrifices too many actionable cases. It is not usable
-- sample level 0.5 is permissive and the sample is enriched. Queues would baloon unmanageably 
-- .7 is a sensible operating point at .76 of recall and .724 precision. 

### Q3 — Same for v2, then pick a threshold
-- Query
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
scored AS (
  SELECT e.conversation_id,
         e.score,
         (fr.reviewed_severity >= 2) AS is_positive
  FROM classifier_events e
  JOIN first_reviews fr USING (conversation_id)
  WHERE e.classifier_version = 'v2'
)
SELECT threshold, tp, fp, fn,
       ROUND(tp * 1.0 / NULLIF(tp + fp, 0), 3) AS precision,
       ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 3) AS recall
FROM (
  SELECT t.threshold,
         SUM(CASE WHEN s.score >= t.threshold AND s.is_positive     THEN 1 ELSE 0 END) AS tp,
         SUM(CASE WHEN s.score >= t.threshold AND NOT s.is_positive THEN 1 ELSE 0 END) AS fp,
         SUM(CASE WHEN s.score <  t.threshold AND s.is_positive     THEN 1 ELSE 0 END) AS fn
  FROM scored s
  CROSS JOIN (VALUES (0.5), (0.6), (0.7), (0.8), (0.9)) AS t(threshold)
  GROUP BY 1
)
ORDER BY threshold;

-- FINDINGS:
-- Included 0.6 and 0.8 for additional data points. v2 is smaller sample size and only from days 60-90. 
-- At .8, precision is a 0.098 higher than 0.7 with ~ half FP and .633 recall. 
-- Hypothesis: v2 precision gain comes from concentrating on higher-severity. Pending verification with tier 3+ run.
-- Recommend v2 with variable per category thresholds pending completion of per-category analysis

### Q4 — Find the locale miscalibration
Compute v1 flag rate and precision (at 0.7) **by locale**, and the same for v2.
Which locales are anomalous, in which classifier version, and how do you know it
is a classifier problem rather than a genuine prevalence difference?

--EXPECTATION: finding significant outliers based on locale and language

--Query:
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
scored AS (
  SELECT e.conversation_id,
         e.classifier_version,
         e.score,
         c.locale,
         (fr.reviewed_severity >= 2) AS is_positive
  FROM classifier_events e
  JOIN first_reviews fr USING (conversation_id)
  JOIN conversations c USING (conversation_id)
)
SELECT classifier_version, locale, n, tp, fp, fn,
       ROUND(tp * 1.0 / NULLIF(tp + fp, 0), 3) AS precision,
       ROUND(tp * 1.0 / NULLIF(tp + fn, 0), 3) AS recall
FROM (
  SELECT classifier_version,
         locale,
         COUNT(*) AS n,
         SUM(CASE WHEN score >= 0.7 AND is_positive     THEN 1 ELSE 0 END) AS tp,
         SUM(CASE WHEN score >= 0.7 AND NOT is_positive THEN 1 ELSE 0 END) AS fp,
         SUM(CASE WHEN score <  0.7 AND is_positive     THEN 1 ELSE 0 END) AS fn
  FROM scored
  GROUP BY 1, 2
)
ORDER BY classifier_version, locale;

FINDINGS: 
--in classifier_version v1, hi-IN and pt-BR exhibit 0.471 in precision relative to 0.8-0.85 in all other locales. Recall unaffected.
--in classifier_version v2, this over-flagging is corrected. No substantive change in population from v1 to v2. TP/N is ~0.24 in both locales and in line with ~0.26 elsewhere. Excess is false positives
--under v1, reviewers would see a substantial backlog for IN and BR and end users would experience increased anomalous interventions. 
--Recommendation: model team to review training data coverages for Brazilian Portuguese and Hindi. Implement locale specific stop-gap in v1 by raising threshold for hi-IN and pt-BR until v2 fully implemented.

### Q5 — Reviewer strictness ranking
Rank the 12 reviewers from strictest to most lenient. A raw mean of
`reviewed_severity` is confounded by queue mix (some reviewers get harder
queues), so adjust for it: compare each reviewer to what *other* reviewers
assign to conversations with similar v1 scores (deciles work).

--EXPECTATION: I expect a small correlation between strictness and queue mix. Expecting smaller variations when comparative to similar v1 scores.
--Querry;
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
)
SELECT fr.reviewer_id,
       COUNT(*) AS n,
       ROUND(AVG(fr.reviewed_severity), 2)                                  AS mean_severity,
       ROUND(AVG(CASE WHEN fr.reviewed_severity >= 2 THEN 1.0 ELSE 0 END), 3) AS actionable_rate,
       ROUND(AVG(e.score), 3)                                               AS mean_queue_score
FROM first_reviews fr
JOIN classifier_events e USING (conversation_id)
WHERE e.classifier_version = 'v1'
GROUP BY 1
ORDER BY actionable_rate DESC;

--Query;
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
scored AS (
  SELECT fr.reviewer_id,
         fr.reviewed_severity,
         NTILE(10) OVER (ORDER BY e.score) AS score_decile
  FROM first_reviews fr
  JOIN classifier_events e USING (conversation_id)
  WHERE e.classifier_version = 'v1'
),
decile_baseline AS (
  SELECT score_decile, AVG(reviewed_severity) AS baseline_severity
  FROM scored
  GROUP BY 1
)
SELECT s.reviewer_id,
       COUNT(*) AS n,
       ROUND(AVG(s.reviewed_severity - d.baseline_severity), 3) AS strictness_vs_peers
FROM scored s
JOIN decile_baseline d USING (score_decile)
GROUP BY 1
ORDER BY strictness_vs_peers DESC;

--Findings
--1st query reveals queues were dealt flat between 0.42 and 0.44. Reviewer appears to be the variable factor in severity.
--Two clusters appear between 0.38-0.41 and 0.27-0.32. Clustering suggests two operative understandings of tier cutoffs.
--2nd query reveals substantial deviation spread of 0.64 and confirms clusters.
--Recommendation: confirm reviewer calibration is consistent. Human review will always introduce a certain level of variability. Consistent training will work to keep that variability within one cluster instead of two.
--Recommendation: Focus reviewer recalibration on more lenient group. Both groups introduce cost in different ways. Leniency has the potential for significant human cost versus potential over exposure on review time and interventions.
--Note: R11 is an outlier in paucity of reviews and in rate stability.

### Q6 — Inter-rater agreement
~8% of reviewed conversations were reviewed twice by different reviewers.
Compute simple agreement, within-one-tier agreement, and **Cohens kappa** on
`reviewed_severity`. Interpret the kappa.
--EXPECTATION: either reveal agreement between reviewers or divergence. Cross cluster pairs from Q5 should disagree with higher frequency.

--Query

WITH pairs AS (
  SELECT a.conversation_id,
         a.reviewer_id        AS reviewer_a,
         b.reviewer_id        AS reviewer_b,
         a.reviewed_severity  AS sev_a,
         b.reviewed_severity  AS sev_b
  FROM human_reviews a
  JOIN human_reviews b
    ON a.conversation_id = b.conversation_id
   AND a.review_id  <> b.review_id
   AND a.reviewed_at < b.reviewed_at
),
agreement AS (
  SELECT COUNT(*) AS n_pairs,
         AVG(CASE WHEN sev_a = sev_b THEN 1.0 ELSE 0 END)           AS exact_agreement,
         AVG(CASE WHEN ABS(sev_a - sev_b) <= 1 THEN 1.0 ELSE 0 END) AS within_one
  FROM pairs
),
marg_a AS (
  SELECT sev_a AS tier, COUNT(*) * 1.0 / (SELECT COUNT(*) FROM pairs) AS pa
  FROM pairs GROUP BY 1
),
marg_b AS (
  SELECT sev_b AS tier, COUNT(*) * 1.0 / (SELECT COUNT(*) FROM pairs) AS pb
  FROM pairs GROUP BY 1
),
expected AS (
  SELECT SUM(pa * pb) AS p_expected
  FROM marg_a JOIN marg_b USING (tier)
)
SELECT n_pairs,
       ROUND(exact_agreement, 3) AS exact_agreement,
       ROUND(within_one, 3)      AS within_one,
       ROUND(p_expected, 3)      AS p_expected,
       ROUND((exact_agreement - p_expected) / (1 - p_expected), 3) AS kappa
FROM agreement, expected;

--Query

WITH pairs AS (
  SELECT a.conversation_id,
         a.reviewer_id        AS reviewer_a,
         b.reviewer_id        AS reviewer_b,
         a.reviewed_severity  AS sev_a,
         b.reviewed_severity  AS sev_b
  FROM human_reviews a
  JOIN human_reviews b
    ON a.conversation_id = b.conversation_id
   AND a.review_id  <> b.review_id
   AND a.reviewed_at < b.reviewed_at
)
SELECT CASE
         WHEN (reviewer_a IN ('R01','R03','R05','R07','R10','R11'))
            = (reviewer_b IN ('R01','R03','R05','R07','R10','R11'))
         THEN 'same_cluster' ELSE 'cross_cluster' END AS pair_type,
       COUNT(*) AS n_pairs,
       ROUND(AVG(CASE WHEN sev_a = sev_b THEN 1.0 ELSE 0 END), 3) AS exact_agreement,
       ROUND(AVG(sev_b - sev_a), 3) AS mean_b_minus_a
FROM pairs
GROUP BY 1;

WITH pairs AS (
  SELECT a.conversation_id,
         a.reviewer_id        AS reviewer_a,
         b.reviewer_id        AS reviewer_b,
         a.reviewed_severity  AS sev_a,
         b.reviewed_severity  AS sev_b
  FROM human_reviews a
  JOIN human_reviews b
    ON a.conversation_id = b.conversation_id
   AND a.review_id  <> b.review_id
   AND a.reviewed_at < b.reviewed_at
)
SELECT COUNT(*) AS n_cross_pairs,
       ROUND(AVG(
         CASE WHEN reviewer_a IN ('R01','R03','R05','R07','R10','R11')
              THEN sev_a - sev_b
              ELSE sev_b - sev_a END
       ), 3) AS strict_minus_lenient
FROM pairs
WHERE (reviewer_a IN ('R01','R03','R05','R07','R10','R11'))
   <> (reviewer_b IN ('R01','R03','R05','R07','R10','R11'));

   --FINDINGS:
   -- Two reviewers agree at 0.54. They are within one tier at 0.959
   -- There is some correlation between cross-cluster reviewers but the difference is a modest 5 points. Recommendation from Q5 stands, but not a singular fix.
   -- Kappa value of 0.367 is below tolerance for this subject matter.
   --Strict vs. lenient is 0.357. Strict sits a third of a tier higher and straddles a tier boundary. Further evidence of two operative calibrations.
   -- Recommend raising rate for double review to 12% for a period of 60 days. There is a cluster off-set but not enough data at 8% to confirm recalibration.

   ### Q7 — Queue latency and the backlog
For first reviews, compute p50 and p95 of (reviewed_at − created_at) in hours,
by week. Identify the backlog week and quantify it against the other weeks.

--EXPECTATION: Finding when latency occurred and pinpointing how long it persists in the workflow.

--Query
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
latency AS (
  SELECT date_diff('day', DATE '2026-06-01', c.created_at::DATE) // 7 + 1 AS week,
         date_diff('minute', c.created_at, fr.reviewed_at) / 60.0 AS hours
  FROM first_reviews fr
  JOIN conversations c USING (conversation_id)
)
SELECT week,
       COUNT(*) AS reviews,
       ROUND(quantile_cont(hours, 0.5), 1)  AS p50_hours,
       ROUND(quantile_cont(hours, 0.95), 1) AS p95_hours
FROM latency
GROUP BY 1
ORDER BY 1;

-- FINDINGS:
-- Latency remains remarkably stable throughout the surveyed period with the exception of week 7. All weeks sans week 7 were between 3.8 and 4.1 at p50 hrs and between 16 and 18.5 at p95 hrs.
-- Week 7 witnessed a spike of 18.3 at p50 and 83 at p95. The backlog does not persist beyond week 7. Notably, total number of reviews decreased from 1376 in wk 6 to 1275 in wk7. 10 of 12 reviewers saw decreased volume of ~20% week over week. Consistent with slower handling or reduced hours. The cost: 5% of flagged conversations waited 3.5 days before a human touched the file.
-- In Weeks 1-8, volume remains steady at between 1105 and 1376. In weeks 9-13, volume increases to 1472-1881. Latency remains stable.
-- Hypothesis: week 7 contained the model switch on day 45. It is possible (but not testable) that changes in conversational style resulted in reviewers needing additional time to complete reviews.
-- Hypothesis: Week 9 is when v2 launched. Increased volume of ~40% with no effect on latency indicates capacity headroom for review team
-- NOTE: In querying reviews completed by reviewer_id, a few things worth noting became evident: R03 is completing a substantial portion on their own. This represents a possible stress point to the rest of the team if they are out of commission during a week like wk7. R11 is completing about a fourth of the volume as coworker median. Future datasets ought to record handling time alongside FTE status


### Q8 — A/B recurrence, naive
For every conversation with an intervention, compute the 7-day severity
recurrence rate by `variant`, plus `resource_link_clicked` rate. Does B look
like it works?

--EXPECTATION: Evidence that B works.

--Query
SELECT i.variant,
       COUNT(*) AS n,
       ROUND(AVG(CASE WHEN f.returned_within_7d AND f.followup_severity >= 2 THEN 1.0 ELSE 0 END), 3) AS recurrence_rate,
       ROUND(AVG(CASE WHEN f.resource_link_clicked THEN 1.0 ELSE 0 END), 3) AS click_rate
FROM interventions i
JOIN followup_signals f USING (conversation_id)
GROUP BY 1
ORDER BY 1;

-- FINDINGS:
-- B lowers recurrence by 1.7 points while raising click rate by 7.2 points.
-- B is a sample size that is roughly 60% the size of A. Note that B testing began 30 days after A.
-- On its face, B Works

## Q9 — A/B stratified by severity and decision
Restrict to interventions shown **on or after day 30** (before that only A
existed). Stratify by `reviewed_severity` (first review) and first-review
`decision`, and compute B − A recurrence within each stratum with enough data
(say ≥ 30 per arm). Now what does B do, for which tiers, and why did Q8 hide it?
Name the two confounders.

--EXPECTATION: I was wrong on Q8 and this is going to show me why.

--Query 1
SELECT i.variant,
       COUNT(*) AS n,
       ROUND(AVG(CASE WHEN f.returned_within_7d AND f.followup_severity >= 2 THEN 1.0 ELSE 0 END), 3) AS recurrence_rate
FROM interventions i
JOIN followup_signals f USING (conversation_id)
JOIN conversations c USING (conversation_id)
WHERE c.created_at >= DATE '2026-07-01'
GROUP BY 1
ORDER BY 1;

--Query 1 Results
variant	n	recurrence_rate
A	4763	0.308
B	4646	0.272

--Query 2
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
base AS (
  SELECT i.variant,
         fr.reviewed_severity,
         fr.decision,
         CASE WHEN f.returned_within_7d AND f.followup_severity >= 2 THEN 1.0 ELSE 0 END AS recurred
  FROM interventions i
  JOIN followup_signals f USING (conversation_id)
  JOIN conversations c    USING (conversation_id)
  JOIN first_reviews fr   USING (conversation_id)
  WHERE c.created_at >= DATE '2026-07-01'
),
per_arm AS (
  SELECT reviewed_severity, decision, variant,
         COUNT(*) AS n,
         AVG(recurred) AS rate
  FROM base
  GROUP BY 1, 2, 3
)
SELECT a.reviewed_severity,
       a.decision,
       a.n AS n_a, b.n AS n_b,
       ROUND(a.rate, 3) AS rate_a,
       ROUND(b.rate, 3) AS rate_b,
       ROUND(b.rate - a.rate, 3) AS b_minus_a
FROM per_arm a
JOIN per_arm b
  ON a.reviewed_severity = b.reviewed_severity
 AND a.decision = b.decision
 AND a.variant = 'A' AND b.variant = 'B'
WHERE a.n >= 30 AND b.n >= 30
ORDER BY 1, 2;

--Query 2 Results
reviewed_severity	decision	n_a	n_b	rate_a	rate_b	b_minus_a
0	no_action	327	345	0.083	0.148	0.065
1	no_action	280	272	0.168	0.129	-0.039
1	resource_shown	423	422	0.227	0.173	-0.054
2	no_action	56	180	0.304	0.317	0.013
2	resource_shown	516	655	0.326	0.305	-0.02
2	steer_response	345	91	0.275	0.154	-0.122
3	escalate	225	111	0.204	0.153	-0.051
3	resource_shown	124	249	0.516	0.47	-0.046
3	steer_response	453	390	0.371	0.249	-0.122
4	escalate	392	392	0.342	0.286	-0.056
4	steer_response	76	57	0.539	0.386	-0.154

-- FINDINGS:
-- Confounder 1: time period. As a holdover from pre model switch, A over-represents a lower recurrence era. Filtering out that earlier data shows a much wider gap between A and B.
-- Confounder 2: changed reviewer behavior.Reviewers assigned steer_response to 345 A conversations vs 91 B, and no_action to 56 A vs 180 B, at tier 2 — B shifted reviewers toward lighter decisions.
-- Q8 hid this because the two confounders stacked. The additional time in the A dataset pulled the rate down and the decision shift moved B into strata where it does little.
-- Within stratum, B cuts recurrence by ~0.12 for steer_response at tiers 2–3 and ~0.15 at tier 4; ~0.05 for escalate and resource_shown at tiers 3–4; mixed at tiers 0–1, and tier 0 no_action moves the wrong way (+0.065).
-- Decision is set after the variant is shown, so stratifying on it conditions on a post-treatment variable; the within-stratum numbers explain the mechanism, but the clean overall estimate is the day-30+ figure of −0.036.
-- Recommendation: Ship B for tiers 2+ and maintain experiment blind on reviewers. Hold tiers 0-1 pending further investigation of no_action signal in tier 0.

## Q10 — Reviewer drift over time
Split the window into two-week blocks and recompute the Q5 residual per
reviewer per block. Which reviewers drift, in which direction, and by roughly
how many tiers over the period? Be careful: the queue mix itself changes over
time (v2 launch at day 60), so compute the baseline within block.

--Expectation: The stricter reviewers will remain steady and the lenient reviewers will drfit.

--Query
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
scored AS (
  SELECT fr.reviewer_id,
         fr.reviewed_severity,
         date_diff('day', DATE '2026-06-01', fr.reviewed_at::DATE) // 14 + 1 AS block,
         e.score
  FROM first_reviews fr
  JOIN classifier_events e USING (conversation_id)
  WHERE e.classifier_version = 'v1'
    AND date_diff('day', DATE '2026-06-01', fr.reviewed_at::DATE) < 84
),
deciled AS (
  SELECT *,
         NTILE(10) OVER (PARTITION BY block ORDER BY score) AS score_decile
  FROM scored
),
baseline AS (
  SELECT block, score_decile, AVG(reviewed_severity) AS baseline_severity
  FROM deciled
  GROUP BY 1, 2
),
residuals AS (
  SELECT d.reviewer_id, d.block,
         ROUND(AVG(d.reviewed_severity - b.baseline_severity), 2) AS resid
  FROM deciled d
  JOIN baseline b USING (block, score_decile)
  GROUP BY 1, 2
)
PIVOT residuals
ON block
USING FIRST(resid)
GROUP BY reviewer_id
ORDER BY reviewer_id;

-- FINDINGS
-- R05 drifts gradually about .6 from lenient to strictest over the 12 weeks.
-- R09 drifts in a step (coinciding with the model switch) about .5 from mildly strict to most lenient.
-- q5 classified R05 as strict and R09 as lenient, but this was only true for ~half of the reporting period.
-- My expectation was wrong: five of the strict reviewers remained steady and five of the lenient reviewers remained steady. Two reviewers crossed over in opposite directions midway through.
-- Recommendation: set this report to run on a weekly schedule to monitor drift accurately and make adjustments as needed.

### Q11 — Model-version vs classifier-version effect on flagged rate
Flagged volume changes around day 45 and again around day 60. Separate the two:
(a) hold the classifier fixed at v1 and compare flag rate for days 30–44 vs 45–59;
(b) on days 60–89, compare v1 and v2 flag rates on the *same* conversations.
Which change is a change in what users bring, and which is a change in the detector?

-- Expectation: the detector change is the main driver, followed by what users bring in their interactions with the model.

-- Query 1
SELECT CASE WHEN date_diff('day', DATE '2026-06-01', c.created_at::DATE) < 45
            THEN 'days 30-44' ELSE 'days 45-59' END AS period,
       e.predicted_category,
       COUNT(*) AS conversations,
       ROUND(AVG(CASE WHEN e.flagged THEN 1.0 ELSE 0 END), 4) AS flag_rate
FROM classifier_events e
JOIN conversations c USING (conversation_id)
WHERE e.classifier_version = 'v1'
  AND date_diff('day', DATE '2026-06-01', c.created_at::DATE) BETWEEN 30 AND 59
GROUP BY 1, 2
ORDER BY 2, 1;

-- Query 1 Response
period	predicted_category	conversations	flag_rate
days 30-44	benign	33698	0.0
days 45-59	benign	36337	0.0
days 30-44	disordered_eating	1219	0.2199
days 45-59	disordered_eating	1209	0.2142
days 30-44	emotional_dependence	1568	0.28
days 45-59	emotional_dependence	1372	0.2296
days 30-44	self_harm	1617	0.2913
days 45-59	self_harm	1725	0.2893
days 30-44	suicide	1107	0.1915
days 45-59	suicide	1132	0.1926
days 30-44	sycophancy_concern	1331	0.2795
days 45-59	sycophancy_concern	1190	0.1958

-- Query 2

SELECT v1.predicted_category,
       COUNT(*) AS conversations,
       ROUND(AVG(CASE WHEN v1.flagged THEN 1.0 ELSE 0 END), 4) AS v1_flag_rate,
       ROUND(AVG(CASE WHEN v2.flagged THEN 1.0 ELSE 0 END), 4) AS v2_flag_rate
FROM classifier_events v1
JOIN classifier_events v2
  ON v1.conversation_id = v2.conversation_id
 AND v1.classifier_version = 'v1'
 AND v2.classifier_version = 'v2'
GROUP BY 1
ORDER BY 1;

-- Query 2 Response
predicted_category	conversations	v1_flag_rate	v2_flag_rate
benign	78137	0.0	0.0205
disordered_eating	2692	0.234	0.2403
emotional_dependence	2948	0.2208	0.2137
self_harm	3756	0.2673	0.2849
suicide	2546	0.1866	0.1917
sycophancy_concern	2388	0.1801	0.1905

--Q uery 3
WITH first_reviews AS (
  SELECT *
  FROM (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews r
  )
  WHERE rn = 1
),
new_catches AS (
  SELECT v1.conversation_id
  FROM classifier_events v1
  JOIN classifier_events v2
    ON v1.conversation_id = v2.conversation_id
   AND v1.classifier_version = 'v1'
   AND v2.classifier_version = 'v2'
  WHERE v1.flagged = FALSE
    AND v2.flagged = TRUE
)
SELECT COUNT(*)                                        AS new_catches_total,
       COUNT(fr.review_id)                             AS reviewed,
       ROUND(AVG(CASE WHEN fr.reviewed_severity >= 2 THEN 1.0 ELSE 0 END), 3) AS actionable_rate,
       ROUND(AVG(CASE WHEN fr.reviewed_severity >= 3 THEN 1.0 ELSE 0 END), 3) AS tier3plus_rate
FROM new_catches nc
LEFT JOIN first_reviews fr USING (conversation_id);

-- Query 3 Response

new_catches_total	reviewed	actionable_rate	tier3plus_rate
2933	2071	0.408	0.189


-- FINDINGS:
-- Query 1 confirms hypothesis from Q1. New model is less sycophantic and less reliance-building.
-- Query 2 reveals that v2 flags 2.05% of what v1 flagged as benign, 2933 additional conversations. Within the categories, flag rates remain stable. 
-- Hypothesis: v2 casts a wider net than v1 and flags signals v1 misses.
-- Query 3 reveals that 41% of new flags were actionable. A little under half of the actionable conversations were classified at tier 3+. Crucially, these would have been entirely missed by v1. Hypothesis confirmed.
-- Recommendation: maintain wider net of v2. Given demonstrated queue headroom in Q7, cost to headroom is worth the price of a 3+ tier conversation going unreviewed.

### Q12 — Recommendation
In one paragraph, using only results from Q2–Q11, recommend: which classifier
version and threshold to run, what to do about the locale issue, whether to
roll out variant B and to whom, what to do about reviewer calibration, and how
the day-45 and day-60 changes should be reported to leadership.

-- Recommendations:
-- The day 45 model change coupled with v1 shows evidence of a model that is less sycophantic
-- and invites fewer instances of user dependence. The day 60 classifier change to v2 shows
-- v2 flags ~2,900 conversations a month that v1 passed; 41% of those are actionable and ~550 
-- are tier 3+. Recommend full implementation of v2 at 0.7 threshold. This increases recall, but Q7 shows 
-- evidence of headroom to accommodate the increase. While this costs something in capacity, it is worth
-- the price of 550 monthly interventions that would have otherwise been missed.
-- Full implementation of v2 addresses locale specific issues in v1 and eliminates need for stop gap.
-- Variant B shows evidence of effectiveness at higher tiers. Recommend rolling this out to
-- tiers 2+ and continuing with Variant A for 0-1 until further review is complete on B. Recommend a 
-- blind on reviewers to counter behavior changes evidenced when reviewer is aware of variant.
-- This dataset shows evidence of drift and clusters with reviewers. Two reviewers drifted by a 
-- half tier each in the opposite direction. Two clusters formed with one being more lenient and 
-- the other more strict. The gap in between indicates two very different operative understandings of
-- review procedures. Recommend recalibration starting with the more lenient group, raising the 
-- double-review rate to 12%, and running the drift report weekly.
-- Finally, this dataset revealed one potential stress point on the review team. R03 contributes
-- ~20% of the queue. Should R03 need to be out of the office for a family emergency or vacation,
-- a high volume week could be costly for latency. Recommend distributing load and logging handling time 
-- and FTE status for each reviewer_id in order to better diagnose future spikes in latency.
  