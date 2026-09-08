#Notes from Jacob

This project is the result of an application submitted to Anthropic. That application
called for intervention effectiveness analysis in SQL, which I have not done before.
Claude helped me build a generator to help me learn and to demonstrate my ability to
work in this medium. The generator is built by Claude Code, to my spec. The queries and 
notes in my_answers.sql are mine with Claude tutoring. Start with Q9 and Q12.


# Wellbeing enforcement analysis: synthetic SQL practice set

A seeded, deterministic generator for a **synthetic** trust-and-safety /
user-wellbeing dataset, plus 12 SQL exercises of increasing difficulty and a
validator that proves the planted effects are recoverable.

Everything here is fabricated. There are no real users, conversations, or
reviewers. Nothing in any table is or resembles a person's words.

## Content constraint

No table contains simulated user message text, quotations, or anything
resembling what a person might say about self-harm, suicide, eating, or crisis.
Content is represented only by:

* categorical labels (`category`, `severity_tier`, `decision`, …),
* numeric scores (`score`, `reviewed_severity`, …),
* abstract, rubric-level descriptors in `codebook` (e.g. tier 3 =
  "active concern with intent indicators, no specifics").

The only free-text column is `human_reviews.review_note`, drawn from a fixed
list of 15 clinical-register templates (e.g. "no risk indicators present; no
action taken"). None describe methods, plans, or specifics. `generate.py` and
`validate.py` both assert this on every text column before writing / after loading.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate.py        # ~10s: writes data/*.csv, wellbeing.duckdb, then runs validate.py
```

Then open `exercises.md` and query the database, e.g.

```bash
duckdb wellbeing.duckdb            # or: .venv/bin/python -c "import duckdb; ..."
```

Regenerating is idempotent: same seed (42) → byte-identical CSVs.

## Files

| Path | What |
|------|------|
| `generate.py` | Generator. All planted parameters are constants at the top. |
| `validate.py` | Runs every block of `solutions.sql` and asserts the planted effects are visible (35 checks). |
| `exercises.md` | 12 questions, conventions, and the timeline of system changes. |
| `solutions.sql` | Reference answers (DuckDB SQL). Blocks are delimited by `-- @Q<n>` markers. Spoilers. |
| `data/*.csv` | One CSV per table. |
| `data/planted_effects.json` | Every hidden parameter used to simulate the data. Spoilers. |
| `wellbeing.duckdb` | All tables loaded. |

## Tables

Analyst view (use these for the exercises):

| Table | Rows | Grain | Notes |
|-------|-----:|-------|-------|
| `codebook` | 26 | (category, severity_tier) | Screening-rubric descriptors. Tiers 0–4; benign is tier 0 only. |
| `conversations` | 250,000 | conversation | `user_id` is a hash. 90 days from 2026-06-01, diurnal + weekly pattern, mild growth. `model_version` switches at day 45. |
| `classifier_events` | 342,467 | (conversation, classifier_version) | v1 for all 90 days; v2 from day 60. `flagged = score >= 0.70`. |
| `reviewers` | 12 | reviewer | Handles, team, shift. Nothing here reveals strictness. |
| `human_reviews` | 19,812 | review | 70% of flagged + 4% of unflagged conversations; ~8% of those reviewed twice by a different reviewer. |
| `interventions` | 12,445 | conversation | `resource_card` or `both` (card + response steer). Variant B from day 30, 50% of eligible traffic by id-hash. |
| `followup_signals` | 30,155 | conversation | For every conversation with an intervention or true tier ≥ 1. `followup_severity` is NULL when the user did not return. |

Hidden (answer-checking only):

| Table | Notes |
|-------|-------|
| `ground_truth` | `true_category`, `true_severity` per conversation. Everything else was simulated from these. |
| `ground_truth_reviewers` | Planted `strictness` offset, `drift_per_day`, `workload_share` per reviewer. |

## What is planted (spoilers)

Read `exercises.md` first if you want to discover these yourself.

* **Classifier scores** are `sigmoid(tier_mean[true_severity] + noise)`. v2 has
  much higher means for tiers 3–4 (better high-tier recall) but also a higher
  tier-1 mean (worse precision at low thresholds). At 0.8, v2 beats v1@0.7 on
  both precision and tier-3+ recall at about the same queue volume.
* **Locale miscalibration**: v1 adds +1.1 logit to tier 0–1 conversations in
  `pt-BR` and `hi-IN`, roughly doubling their flag rate and halving precision.
  v2 has no such term.
* **Reviewers**: each has a fixed strictness offset in [−0.6, 0.6] tiers plus
  N(0, 0.55) noise. `R05` drifts +1.1 tiers stricter over the 90 days; `R09`
  drifts −1.0 tiers more lenient. Reviewed category is confused 10% of the time.
* **Queue latency** is lognormal (median ≈ 4h); week 7 (days 42–48) is
  multiplied by 3.5–6×. Second reviews land 1.2–2.5× later than first reviews.
* **Variant B** directly lowers 7-day severity recurrence by 25% for true tiers
  2–3 and does nothing for tiers 1 and 4. Two things mask it in a naive
  comparison: (1) when B is shown, reviewers choose lighter decisions
  (`no_action` / `resource_shown` instead of `steer_response` / `escalate`),
  and those decisions recur more; (2) recurrence is 1.4× higher after the day-45
  model switch, and arm A over-represents the early period because it was the
  only arm for days 0–29. Stratifying by decision and severity on post-launch
  traffic recovers the effect.
* **Model-version switch** (day 45) lowers the prevalence of
  `emotional_dependence` and `sycophancy_concern`, which drops the v1 flag rate
  by ~18% with the classifier held fixed. Classifier v2 (day 60) flags ~50%
  more than v1 on the same conversations. Both look like "flagged volume
  changed" on a naive time series.

## Validation

`validate.py` splits `solutions.sql` on the `-- @Q<n>` markers, runs each
block read-only against `wellbeing.duckdb`, and asserts 35 conditions
(monotone precision/recall, the locale gap present in v1 and absent in v2,
Spearman ≥ 0.8 between recovered and planted reviewer strictness, drifters
identified with everyone else stable, backlog week is the worst week by > 2.5×,
naive A/B difference small while the stratified effect is clearly negative for
tiers 2–3 and null for tier 4, model-switch and classifier-switch effects
separable, and the content-safety token scan). It exits non-zero on any failure.

```bash
.venv/bin/python validate.py
```

## Dependencies

Python 3 with `numpy`, `pandas`, `duckdb`, `faker` only (`requirements.txt`).
Faker is used solely for reviewer handles and onboarding dates.
