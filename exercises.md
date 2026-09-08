# Exercises: wellbeing enforcement analysis in SQL

Work against `wellbeing.duckdb` (DuckDB dialect). Reference answers live in
`solutions.sql`; try each question before reading them. Do **not** use
`ground_truth` or `ground_truth_reviewers` in your answers — those tables exist
only so `validate.py` can check that the planted effects are recoverable.

Conventions the solutions use (you may choose differently, but say so):

* **Positive class** for precision/recall = human-reviewed severity **≥ 2**
  (an "actionable" conversation). Predicted positive = `score >= threshold`.
* When a conversation has two reviews, the **first** (by `reviewed_at`) is the
  operative decision. Second reviews are for agreement analysis only.
* **Day index** = days since `2026-06-01` (day 0). Week *n* = days 7(n−1) to 7n−1.
* **7-day severity recurrence** = `returned_within_7d AND followup_severity >= 2`.

Timeline you should know about (it is also in `data/planted_effects.json`):

| Day | Event |
|-----|-------|
| 0   | Start of window (2026-06-01) |
| 30  | Resource-card **variant B** launched for 50% of eligible traffic |
| 45  | Assistant **model_version** switches `m-2026.05 → m-2026.07` |
| 60  | **Classifier v2** starts scoring alongside v1 (v1 keeps running) |
| 42–48 (week 7) | Review-queue **backlog** |

---

## Warm-up

### Q1 — Daily flagged volume by category
For classifier **v1** (the only version present all 90 days), count flagged
conversations per calendar day and `predicted_category`. Plot it mentally:
where does volume change, and does the change look like user behaviour or
like a system change? (Q11 will settle this.)

### Q2 — v1 precision and recall at 0.5 / 0.7 / 0.9
Using first human reviews as labels, compute TP / FP / FN, precision and recall
of v1 at each threshold. Note the sample is enriched for flagged conversations
(70% of flagged are reviewed vs 4% of unflagged) and say how that biases each metric.

## Core

### Q3 — Same for v2, then pick a threshold
Repeat Q2 for **v2**, restricted to conversations scored by *both* versions so
the comparison is fair. Add a column for recall on tier ≥ 3 (the tier v2 was
built to catch) and the number of conversations flagged at each threshold
(review-queue cost). Choose an operating point for v2 and defend it in two
sentences. Hint: 0.5 / 0.7 / 0.9 may not be the only thresholds worth looking at.

### Q4 — Find the locale miscalibration
Compute v1 flag rate and precision (at 0.7) **by locale**, and the same for v2.
Which locales are anomalous, in which classifier version, and how do you know it
is a classifier problem rather than a genuine prevalence difference?

### Q5 — Reviewer strictness ranking
Rank the 12 reviewers from strictest to most lenient. A raw mean of
`reviewed_severity` is confounded by queue mix (some reviewers get harder
queues), so adjust for it: compare each reviewer to what *other* reviewers
assign to conversations with similar v1 scores (deciles work).

### Q6 — Inter-rater agreement
~8% of reviewed conversations were reviewed twice by different reviewers.
Compute simple agreement, within-one-tier agreement, and **Cohen's kappa** on
`reviewed_severity`. Interpret the kappa.

### Q7 — Queue latency and the backlog
For first reviews, compute p50 and p95 of (reviewed_at − created_at) in hours,
by week. Identify the backlog week and quantify it against the other weeks.

## Experiment analysis

### Q8 — A/B recurrence, naive
For every conversation with an intervention, compute the 7-day severity
recurrence rate by `variant`, plus `resource_link_clicked` rate. Does B look
like it works?

### Q9 — A/B stratified by severity and decision
Restrict to interventions shown **on or after day 30** (before that only A
existed). Stratify by `reviewed_severity` (first review) and first-review
`decision`, and compute B − A recurrence within each stratum with enough data
(say ≥ 30 per arm). Now what does B do, for which tiers, and why did Q8 hide it?
Name the two confounders.

## Monitoring

### Q10 — Reviewer drift over time
Split the window into two-week blocks and recompute the Q5 residual per
reviewer per block. Which reviewers drift, in which direction, and by roughly
how many tiers over the period? Be careful: the queue mix itself changes over
time (v2 launch at day 60), so compute the baseline within block.

### Q11 — Model-version vs classifier-version effect on flagged rate
Flagged volume changes around day 45 and again around day 60. Separate the two:
(a) hold the classifier fixed at v1 and compare flag rate for days 30–44 vs 45–59;
(b) on days 60–89, compare v1 and v2 flag rates on the *same* conversations.
Which change is a change in what users bring, and which is a change in the detector?

## Synthesis

### Q12 — Recommendation
In one paragraph, using only results from Q2–Q11, recommend: which classifier
version and threshold to run, what to do about the locale issue, whether to
roll out variant B and to whom, what to do about reviewer calibration, and how
the day-45 and day-60 changes should be reported to leadership.
