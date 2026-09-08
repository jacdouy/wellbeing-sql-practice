-- ============================================================================
-- solutions.sql  --  reference answers for exercises.md (DuckDB dialect)
--
-- Conventions used throughout:
--   * "Positive" for precision/recall = human-reviewed severity >= 2 (actionable).
--   * When a conversation has two reviews, the FIRST review (by reviewed_at)
--     is the operative one; second reviews are used only for agreement (Q6).
--   * Day index = days since the first conversation (2026-06-01).
--   * ground_truth / ground_truth_reviewers are NOT used in any answer here;
--     they exist only so validate.py can check that the answers recover the
--     planted effects.
--
-- validate.py splits this file on the "-- @Q<n>" markers and runs each block.
-- ============================================================================

-- @Q1  Daily flagged volume by predicted category (classifier v1, the one that runs all 90 days)
SELECT
    CAST(c.created_at AS DATE)        AS day,
    e.predicted_category,
    COUNT(*)                          AS flagged_conversations
FROM classifier_events e
JOIN conversations c USING (conversation_id)
WHERE e.classifier_version = 'v1' AND e.flagged
GROUP BY 1, 2
ORDER BY 1, 2;

-- @Q2  Precision / recall of v1 at 0.5 / 0.7 / 0.9 against the first human review
WITH first_review AS (
    SELECT conversation_id, reviewed_severity,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews
),
labelled AS (
    SELECT e.score, (r.reviewed_severity >= 2) AS positive
    FROM classifier_events e
    JOIN first_review r ON r.conversation_id = e.conversation_id AND r.rn = 1
    WHERE e.classifier_version = 'v1'
),
thresholds AS (SELECT UNNEST([0.5, 0.7, 0.9]) AS threshold)
SELECT
    t.threshold,
    SUM(CASE WHEN l.score >= t.threshold AND l.positive THEN 1 ELSE 0 END)                        AS tp,
    SUM(CASE WHEN l.score >= t.threshold AND NOT l.positive THEN 1 ELSE 0 END)                    AS fp,
    SUM(CASE WHEN l.score <  t.threshold AND l.positive THEN 1 ELSE 0 END)                        AS fn,
    ROUND(SUM(CASE WHEN l.score >= t.threshold AND l.positive THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN l.score >= t.threshold THEN 1 ELSE 0 END), 0), 3)                 AS precision,
    ROUND(SUM(CASE WHEN l.score >= t.threshold AND l.positive THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN l.positive THEN 1 ELSE 0 END), 0), 3)                              AS recall
FROM thresholds t CROSS JOIN labelled l
GROUP BY 1 ORDER BY 1;

-- @Q3  Same for v2 at 0.5/0.7/0.8/0.9 (only conversations scored by both, so the comparison is apples-to-apples),
--      plus recall on the highest tiers, which is what v2 was built to improve.
WITH first_review AS (
    SELECT conversation_id, reviewed_severity,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews
),
both_scored AS (
    SELECT conversation_id FROM classifier_events GROUP BY 1 HAVING COUNT(DISTINCT classifier_version) = 2
),
labelled AS (
    SELECT e.classifier_version, e.score,
           (r.reviewed_severity >= 2) AS positive,
           (r.reviewed_severity >= 3) AS high_tier
    FROM classifier_events e
    JOIN both_scored b USING (conversation_id)
    JOIN first_review r ON r.conversation_id = e.conversation_id AND r.rn = 1
),
thresholds AS (SELECT UNNEST([0.5, 0.7, 0.8, 0.9]) AS threshold)
SELECT
    l.classifier_version, t.threshold,
    ROUND(SUM(CASE WHEN score >= threshold AND positive THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN score >= threshold THEN 1 ELSE 0 END), 0), 3)      AS precision,
    ROUND(SUM(CASE WHEN score >= threshold AND positive THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN positive THEN 1 ELSE 0 END), 0), 3)                 AS recall,
    ROUND(SUM(CASE WHEN score >= threshold AND high_tier THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN high_tier THEN 1 ELSE 0 END), 0), 3)                AS recall_tier3plus,
    SUM(CASE WHEN score >= threshold THEN 1 ELSE 0 END)                             AS flagged_n
FROM thresholds t CROSS JOIN labelled l
GROUP BY 1, 2 ORDER BY 1, 2;
-- Pick: v2 @ 0.8. At roughly the same queue volume as v1 @ 0.7 it has better precision
-- AND better tier-3+ recall. v2 @ 0.7 buys a little more tier-3+ recall at ~50% more
-- queue volume, mostly tier 0-1 false positives; v2 @ 0.9 gives up too much recall.

-- @Q4  Locale miscalibration: v1 flag rate and precision by locale, side by side with v2
WITH first_review AS (
    SELECT conversation_id, reviewed_severity,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews
)
SELECT
    c.locale,
    e.classifier_version,
    COUNT(*)                                                             AS scored,
    ROUND(AVG(CASE WHEN e.flagged THEN 1 ELSE 0 END), 3)                 AS flag_rate,
    ROUND(AVG(CASE WHEN e.flagged AND r.reviewed_severity >= 2 THEN 1
                   WHEN e.flagged THEN 0 END), 3)                        AS precision_at_0_7,
    ROUND(AVG(CASE WHEN e.flagged AND r.reviewed_severity <= 1 THEN 1
                   WHEN e.flagged THEN 0 END), 3)                        AS false_positive_share
FROM classifier_events e
JOIN conversations c USING (conversation_id)
LEFT JOIN first_review r ON r.conversation_id = e.conversation_id AND r.rn = 1
GROUP BY 1, 2
ORDER BY 2, 5;
-- pt-BR and hi-IN show a v1 flag rate roughly double the other locales with much
-- lower precision; the gap disappears under v2 -> v1-specific miscalibration.

-- @Q5  Reviewer strictness: mean reviewed severity relative to what other reviewers
--      give conversations with the same v1 score decile (adjusts for queue mix)
WITH scored AS (
    SELECT r.review_id, r.reviewer_id, r.reviewed_severity,
           NTILE(10) OVER (ORDER BY e.score) AS score_decile
    FROM human_reviews r
    JOIN classifier_events e ON e.conversation_id = r.conversation_id AND e.classifier_version = 'v1'
),
decile_mean AS (
    SELECT score_decile, AVG(reviewed_severity) AS expected FROM scored GROUP BY 1
)
SELECT
    s.reviewer_id,
    COUNT(*)                                          AS reviews,
    ROUND(AVG(s.reviewed_severity), 3)                AS mean_severity,
    ROUND(AVG(s.reviewed_severity - d.expected), 3)   AS strictness_residual
FROM scored s JOIN decile_mean d USING (score_decile)
GROUP BY 1
ORDER BY strictness_residual DESC;

-- @Q6  Inter-rater agreement on double-reviewed conversations: simple agreement + Cohen's kappa on severity
WITH pairs AS (
    SELECT conversation_id,
           MIN(CASE WHEN rn = 1 THEN reviewed_severity END) AS s1,
           MIN(CASE WHEN rn = 2 THEN reviewed_severity END) AS s2
    FROM (SELECT conversation_id, reviewed_severity,
                 ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
          FROM human_reviews)
    GROUP BY 1 HAVING COUNT(*) = 2
),
n AS (SELECT COUNT(*) AS total FROM pairs),
po AS (SELECT AVG(CASE WHEN s1 = s2 THEN 1.0 ELSE 0 END) AS p_obs FROM pairs),
marg AS (
    SELECT k,
           SUM(CASE WHEN s1 = k THEN 1.0 ELSE 0 END) / (SELECT total FROM n) AS p1,
           SUM(CASE WHEN s2 = k THEN 1.0 ELSE 0 END) / (SELECT total FROM n) AS p2
    FROM pairs CROSS JOIN (SELECT UNNEST([0,1,2,3,4]) AS k)
    GROUP BY k
),
pe AS (SELECT SUM(p1 * p2) AS p_exp FROM marg)
SELECT
    (SELECT total FROM n)                                     AS double_reviewed,
    ROUND(p_obs, 3)                                           AS simple_agreement,
    ROUND(AVG(CASE WHEN ABS(s1 - s2) <= 1 THEN 1.0 ELSE 0 END), 3) AS within_one_tier,
    ROUND((p_obs - p_exp) / (1 - p_exp), 3)                   AS cohens_kappa
FROM pairs, po, pe
GROUP BY p_obs, p_exp;

-- @Q7  Queue latency p50 / p95 by week (first reviews only) -> backlog spike in week 7
WITH first_review AS (
    SELECT r.conversation_id, r.reviewed_at, c.created_at,
           ROW_NUMBER() OVER (PARTITION BY r.conversation_id ORDER BY r.reviewed_at) AS rn
    FROM human_reviews r JOIN conversations c USING (conversation_id)
)
SELECT
    1 + DATE_DIFF('day', DATE '2026-06-01', CAST(created_at AS DATE)) // 7        AS week,
    COUNT(*)                                                                      AS reviews,
    ROUND(QUANTILE_CONT(EPOCH(reviewed_at - created_at) / 3600.0, 0.50), 1)       AS p50_hours,
    ROUND(QUANTILE_CONT(EPOCH(reviewed_at - created_at) / 3600.0, 0.95), 1)       AS p95_hours
FROM first_review
WHERE rn = 1
GROUP BY 1 ORDER BY 1;

-- @Q8  Naive A/B: 7-day severity recurrence by variant (recurrence = returned with follow-up severity >= 2)
SELECT
    i.variant,
    COUNT(*)                                                              AS n,
    ROUND(AVG(CASE WHEN f.returned_within_7d AND f.followup_severity >= 2 THEN 1.0 ELSE 0 END), 3) AS recurrence_rate,
    ROUND(AVG(CASE WHEN f.resource_link_clicked THEN 1.0 ELSE 0 END), 3)  AS click_rate
FROM interventions i
JOIN followup_signals f USING (conversation_id)
GROUP BY 1 ORDER BY 1;

-- @Q9  A/B stratified: post-launch traffic only, by reviewed severity x first-review decision
WITH first_review AS (
    SELECT conversation_id, reviewed_severity, decision,
           ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) AS rn
    FROM human_reviews
),
base AS (
    SELECT i.variant, r.reviewed_severity, r.decision,
           CASE WHEN f.returned_within_7d AND f.followup_severity >= 2 THEN 1.0 ELSE 0 END AS recurred
    FROM interventions i
    JOIN followup_signals f USING (conversation_id)
    JOIN first_review r ON r.conversation_id = i.conversation_id AND r.rn = 1
    WHERE i.shown_at >= DATE '2026-06-01' + INTERVAL 30 DAY
)
SELECT
    reviewed_severity, decision,
    SUM(CASE WHEN variant = 'A' THEN 1 ELSE 0 END)                    AS n_a,
    SUM(CASE WHEN variant = 'B' THEN 1 ELSE 0 END)                    AS n_b,
    ROUND(AVG(CASE WHEN variant = 'A' THEN recurred END), 3)          AS recur_a,
    ROUND(AVG(CASE WHEN variant = 'B' THEN recurred END), 3)          AS recur_b,
    ROUND(AVG(CASE WHEN variant = 'B' THEN recurred END)
        - AVG(CASE WHEN variant = 'A' THEN recurred END), 3)          AS b_minus_a
FROM base
WHERE reviewed_severity BETWEEN 1 AND 4
GROUP BY 1, 2
HAVING n_a >= 30 AND n_b >= 30
ORDER BY 1, 2;
-- Read-out: within (severity, decision) strata B is consistently lower for tiers 2-3
-- and flat for tier 4. The naive comparison hides this because B shifts reviewers
-- toward lighter decisions (which recur more) and A over-represents the early period.

-- @Q10 Reviewer drift: strictness residual per reviewer per two-week block.
--      The baseline is computed per (score decile, block) so that changes in the
--      queue mix over time (v2 launch, model switch) do not masquerade as drift.
WITH scored AS (
    SELECT r.reviewer_id, r.reviewed_severity,
           DATE_DIFF('day', DATE '2026-06-01', CAST(c.created_at AS DATE)) // 14 AS block,
           NTILE(10) OVER (ORDER BY e.score) AS score_decile
    FROM human_reviews r
    JOIN conversations c USING (conversation_id)
    JOIN classifier_events e ON e.conversation_id = r.conversation_id AND e.classifier_version = 'v1'
),
decile_mean AS (SELECT score_decile, block, AVG(reviewed_severity) AS expected FROM scored GROUP BY 1, 2),
resid AS (
    SELECT s.reviewer_id, s.block, AVG(s.reviewed_severity - d.expected) AS residual, COUNT(*) AS n
    FROM scored s JOIN decile_mean d USING (score_decile, block)
    GROUP BY 1, 2
)
SELECT
    reviewer_id,
    ROUND(MIN(CASE WHEN block = 0 THEN residual END), 3)          AS block0,
    ROUND(MIN(CASE WHEN block = 2 THEN residual END), 3)          AS block2,
    ROUND(MIN(CASE WHEN block = 4 THEN residual END), 3)          AS block4,
    ROUND(MIN(CASE WHEN block = 6 THEN residual END), 3)          AS block6,
    ROUND(REGR_SLOPE(residual, block) * 6, 3)                     AS drift_over_period
FROM resid
GROUP BY 1
ORDER BY drift_over_period DESC;

-- @Q11 Model-version change vs classifier-version change on flagged rate.
--      Hold the classifier fixed (v1 runs all 90 days) to isolate the model switch at day 45;
--      then compare v1 vs v2 on the same conversations (day >= 60) to isolate the classifier.
WITH by_day AS (
    SELECT DATE_DIFF('day', DATE '2026-06-01', CAST(c.created_at AS DATE)) AS day,
           c.model_version, e.classifier_version, e.flagged
    FROM classifier_events e JOIN conversations c USING (conversation_id)
)
SELECT 'model switch (classifier v1 held fixed)' AS comparison,
       CASE WHEN day BETWEEN 30 AND 44 THEN 'days 30-44 / ' || model_version
            WHEN day BETWEEN 45 AND 59 THEN 'days 45-59 / ' || model_version END AS window,
       COUNT(*) AS n, ROUND(AVG(CASE WHEN flagged THEN 1.0 ELSE 0 END), 4) AS flag_rate
FROM by_day WHERE classifier_version = 'v1' AND day BETWEEN 30 AND 59
GROUP BY 1, 2
UNION ALL
SELECT 'classifier switch (same conversations, days 60-89)',
       'classifier ' || classifier_version,
       COUNT(*), ROUND(AVG(CASE WHEN flagged THEN 1.0 ELSE 0 END), 4)
FROM by_day WHERE day >= 60
GROUP BY 1, 2
ORDER BY 1, 2;

-- @Q12 Recommendation (prose; no query)
-- Keep classifier v1 in production only as a shadow signal and cut over to v2, but
-- not at the current 0.7 default: at 0.7 v2 adds roughly half again as much review
-- volume and the extra flags are mostly tier 0-1, whereas at 0.8 v2 beats v1@0.7 on
-- both precision and tier-3+ recall at essentially the same queue size. Before
-- cutover, fix the v1 locale problem: pt-BR and hi-IN are over-flagged by v1 with
-- roughly half the precision of other locales, and those users have been absorbing
-- false-positive interventions for 90 days; v2 does not show the gap. Reviewer
-- calibration needs attention independent of the model: strictness spread is about
-- one full tier between the most lenient and strictest reviewers, kappa on
-- double-reviews is moderate, and two reviewers (R05 tightening, R09 loosening)
-- have drifted steadily across the period, so any precision figure that leans on
-- a single reviewer is soft. The week-7 backlog (p95 latency several days) should
-- be treated as a capacity incident, not a model signal. On the resource-card test,
-- ship variant B for tiers 2-3 only: the naive comparison is flat because B nudges
-- reviewers toward no_action, but stratified by decision it lowers 7-day recurrence
-- by several points; it does nothing for tier 4, where escalation remains the lever.
-- Finally, the day-45 model_version change lowered the v1 flag rate on its own
-- (fewer dependence/sycophancy conversations), so the day-60 jump in flagged volume
-- is a classifier artifact, not a change in user risk, and should not be reported as one.
