#!/usr/bin/env python3
"""
Runs every block of solutions.sql against wellbeing.duckdb and asserts that the
planted effects are recoverable from the analyst-facing tables. Ground-truth
tables are used only for a few cross-checks, clearly marked below.
"""
from __future__ import annotations

import json
import os
import re
import sys

import duckdb
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_blocks(path: str) -> dict[str, str]:
    text = open(path).read()
    parts = re.split(r"^-- @(Q\d+)", text, flags=re.M)
    blocks = {}
    for i in range(1, len(parts), 2):
        body = parts[i + 1].split("\n", 1)   # drop the title text on the marker line
        blocks[parts[i]] = body[1] if len(body) > 1 else ""
    return blocks


def run(con, sql: str) -> pd.DataFrame | None:
    code = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    stmts = [s for s in code.split(";") if s.strip()]
    out = None
    for s in stmts:
        out = con.execute(s).df()
    return out


RESULTS: list[tuple[bool, str]] = []


def check(cond: bool, msg: str) -> None:
    RESULTS.append((bool(cond), msg))
    print(("  PASS  " if cond else "  FAIL  ") + msg)


def main() -> None:
    db = os.path.join(ROOT, "wellbeing.duckdb")
    con = duckdb.connect(db, read_only=True)
    planted = json.load(open(os.path.join(ROOT, "data", "planted_effects.json")))
    blocks = load_blocks(os.path.join(ROOT, "solutions.sql"))
    res = RESULTS
    out: dict[str, pd.DataFrame] = {}

    print("\n=== running solutions.sql ===")
    for q, sql in blocks.items():
        df = run(con, sql)
        out[q] = df
        n = 0 if df is None else len(df)
        print(f"  {q}: {'(prose)' if df is None else f'{n} rows'}")

    print("\n=== assertions ===")

    # Q1 -------------------------------------------------------------------
    q1 = out["Q1"]
    check(q1["day"].nunique() == planted["n_days"], "Q1: one row-group per day for all 90 days")
    check(set(q1["predicted_category"]) <= {"self_harm", "suicide", "disordered_eating",
                                            "emotional_dependence", "sycophancy_concern"},
          "Q1: flagged rows carry only risk categories")

    # Q2 -------------------------------------------------------------------
    q2 = out["Q2"].set_index("threshold")
    check(q2.loc[0.5, "recall"] > q2.loc[0.7, "recall"] > q2.loc[0.9, "recall"], "Q2: v1 recall falls as threshold rises")
    check(q2.loc[0.5, "precision"] < q2.loc[0.7, "precision"] < q2.loc[0.9, "precision"], "Q2: v1 precision rises as threshold rises")

    # Q3 -------------------------------------------------------------------
    q3 = out["Q3"].set_index(["classifier_version", "threshold"])
    check(q3.loc[("v2", 0.7), "recall_tier3plus"] > q3.loc[("v1", 0.7), "recall_tier3plus"] + 0.05,
          "Q3: v2 has clearly better tier-3+ recall than v1 at 0.7")
    check(q3.loc[("v2", 0.7), "precision"] < q3.loc[("v1", 0.7), "precision"],
          "Q3: v2 has worse precision than v1 at 0.7 (tradeoff is real)")
    check(q3.loc[("v2", 0.8), "recall_tier3plus"] > q3.loc[("v1", 0.7), "recall_tier3plus"]
          and q3.loc[("v2", 0.8), "precision"] > q3.loc[("v1", 0.7), "precision"]
          and abs(q3.loc[("v2", 0.8), "flagged_n"] / q3.loc[("v1", 0.7), "flagged_n"] - 1) < 0.15,
          "Q3: v2@0.8 beats v1@0.7 on precision AND tier-3+ recall at ~same queue volume (recommended point)")
    check(q3.loc[("v2", 0.9), "recall_tier3plus"] < q3.loc[("v1", 0.7), "recall_tier3plus"],
          "Q3: v2@0.9 gives up tier-3+ recall vs v1@0.7 (so 0.9 is not a free lunch)")

    # Q4 -------------------------------------------------------------------
    q4 = out["Q4"]
    v1 = q4[q4.classifier_version == "v1"].set_index("locale")
    v2 = q4[q4.classifier_version == "v2"].set_index("locale")
    bad = planted["miscalibrated_locales_v1"]
    good = [l for l in v1.index if l not in bad]
    check(v1.loc[bad, "flag_rate"].min() > 1.5 * v1.loc[good, "flag_rate"].max(),
          f"Q4: v1 flag rate in {bad} > 1.5x every other locale")
    check(v1.loc[bad, "precision_at_0_7"].max() < 0.8 * v1.loc[good, "precision_at_0_7"].min(),
          f"Q4: v1 precision in {bad} < 0.8x every other locale")
    check(v2.loc[bad, "flag_rate"].max() < 1.25 * v2.loc[good, "flag_rate"].median(),
          "Q4: gap is absent in v2 (v1-specific miscalibration)")
    # ground-truth cross-check
    gt = con.execute("""
        SELECT c.locale IN ('pt-BR','hi-IN') AS bad, AVG(e.flagged::INT) AS fpr
        FROM classifier_events e JOIN conversations c USING (conversation_id)
        JOIN ground_truth g USING (conversation_id)
        WHERE e.classifier_version='v1' AND g.true_severity <= 1 GROUP BY 1
    """).df().set_index("bad")["fpr"]
    check(gt[True] > 2 * gt[False], "Q4 (ground truth): v1 false-positive rate on tier 0-1 >2x in affected locales")

    # Q5 -------------------------------------------------------------------
    q5 = out["Q5"].set_index("reviewer_id")
    # effective strictness over the period = base offset + drift at the mid-point (day 45)
    gtr = con.execute("SELECT reviewer_id, strictness + drift_per_day * 45 AS strictness FROM ground_truth_reviewers") \
             .df().set_index("reviewer_id")
    joined = q5.join(gtr)
    rho = joined["strictness_residual"].rank().corr(joined["strictness"].rank())
    check(rho > 0.8, f"Q5 (ground truth): strictness ranking recovers planted offsets (spearman={rho:.2f})")
    check(q5["strictness_residual"].max() - q5["strictness_residual"].min() > 0.6,
          "Q5: spread between strictest and most lenient reviewer > 0.6 tiers")

    # Q6 -------------------------------------------------------------------
    q6 = out["Q6"].iloc[0]
    check(q6["double_reviewed"] > 500, f"Q6: enough double reviews ({int(q6['double_reviewed'])})")
    check(0.25 < q6["cohens_kappa"] < 0.9, f"Q6: kappa in a plausible moderate range ({q6['cohens_kappa']})")
    check(q6["simple_agreement"] > q6["cohens_kappa"], "Q6: kappa below raw agreement (chance-corrected)")

    # Q7 -------------------------------------------------------------------
    q7 = out["Q7"].set_index("week")
    others = q7.drop(index=planted["backlog_week"])
    check(q7.loc[planted["backlog_week"], "p95_hours"] > 2.5 * others["p95_hours"].median(),
          "Q7: week-7 p95 latency > 2.5x the median of other weeks")
    check(q7.loc[planted["backlog_week"], "p50_hours"] > 2.5 * others["p50_hours"].median(),
          "Q7: week-7 p50 latency > 2.5x the median of other weeks")
    check(q7.loc[planted["backlog_week"], "p95_hours"] == q7["p95_hours"].max(), "Q7: week 7 is the worst week")

    # Q8 vs Q9 -------------------------------------------------------------
    q8 = out["Q8"].set_index("variant")
    naive = q8.loc["B", "recurrence_rate"] - q8.loc["A", "recurrence_rate"]
    check(abs(naive) < 0.04, f"Q8: naive A/B recurrence difference is small ({naive:+.3f})")
    q9 = out["Q9"]
    mid = q9[q9.reviewed_severity.isin([2, 3])]
    w = (mid["n_a"] * mid["n_b"]) / (mid["n_a"] + mid["n_b"])
    pooled_mid = float((mid["b_minus_a"] * w).sum() / w.sum())
    check(pooled_mid < -0.05, f"Q9: pooled within-stratum B-A for tiers 2-3 is clearly negative ({pooled_mid:+.3f})")
    check((mid["b_minus_a"] < 0).mean() >= 0.75, "Q9: B lower than A in >=75% of tier 2-3 strata")
    t4 = q9[q9.reviewed_severity == 4]
    if len(t4):
        w4 = (t4["n_a"] * t4["n_b"]) / (t4["n_a"] + t4["n_b"])
        pooled_t4 = float((t4["b_minus_a"] * w4).sum() / w4.sum())
        print(f"        (info) analyst-view reviewed-tier-4 pooled B-A = {pooled_t4:+.3f}; "
              "noisy because reviewers push some true tier-3 into tier 4")
    check(pooled_mid < naive - 0.02, "Q9: stratified effect is larger than the naive one (confounding masked it)")
    # ground-truth cross-check on the true tiers, post-launch, within first-review decision
    gt9 = con.execute("""
        WITH fr AS (SELECT conversation_id, decision,
                    ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY reviewed_at) rn FROM human_reviews)
        SELECT g.true_severity, fr.decision, i.variant,
               AVG(CASE WHEN f.returned_within_7d AND f.followup_severity >= 2 THEN 1.0 ELSE 0 END) r, COUNT(*) n
        FROM interventions i JOIN followup_signals f USING (conversation_id)
        JOIN ground_truth g USING (conversation_id) JOIN fr ON fr.conversation_id=i.conversation_id AND fr.rn=1
        WHERE i.shown_at >= DATE '2026-06-01' + INTERVAL 30 DAY AND g.true_severity IN (2,3,4)
        GROUP BY 1,2,3
    """).df()
    piv = gt9.pivot_table(index=["true_severity", "decision"], columns="variant", values="r")
    cnt = gt9.pivot_table(index=["true_severity", "decision"], columns="variant", values="n")
    ok = (cnt.min(axis=1) >= 30)
    diffs = (piv["B"] - piv["A"])[ok]
    wts = (cnt["A"] * cnt["B"] / (cnt["A"] + cnt["B"]))[ok]          # Mantel-Haenszel-style weights

    def pooled(tiers):
        m = [t in tiers for t, _ in diffs.index]
        return float((diffs[m] * wts[m]).sum() / wts[m].sum())
    check(pooled((2, 3)) < -0.04, f"Q9 (ground truth): pooled within-stratum B-A on true tiers 2-3 = {pooled((2, 3)):+.3f}")
    check(abs(pooled((4,))) < 0.05, f"Q9 (ground truth): no effect on true tier 4 = {pooled((4,)):+.3f}")

    # Q10 ------------------------------------------------------------------
    q10 = out["Q10"].set_index("reviewer_id")
    stable = q10.drop(index=["R05", "R09"])["drift_over_period"].abs().max()
    check(q10.loc["R05", "drift_over_period"] > 0.5, f"Q10: R05 drifts stricter ({q10.loc['R05','drift_over_period']:+.2f})")
    check(q10.loc["R09", "drift_over_period"] < -0.5, f"Q10: R09 drifts more lenient ({q10.loc['R09','drift_over_period']:+.2f})")
    check(stable < 0.4, f"Q10: all other reviewers stable (max |drift| = {stable:.2f})")
    check(q10.index[0] == "R05" and q10.index[-1] == "R09", "Q10: drifters are the two extremes of the ranking")

    # Q11 ------------------------------------------------------------------
    q11 = out["Q11"].set_index("window")
    pre, post = q11.loc["days 30-44 / m-2026.05", "flag_rate"], q11.loc["days 45-59 / m-2026.07", "flag_rate"]
    check(post < 0.9 * pre, f"Q11: model switch lowers v1 flag rate ({pre:.4f} -> {post:.4f}) with classifier fixed")
    c1, c2 = q11.loc["classifier v1", "flag_rate"], q11.loc["classifier v2", "flag_rate"]
    check(c2 > 1.3 * c1, f"Q11: classifier v2 flags far more than v1 on the same conversations ({c1:.4f} vs {c2:.4f})")

    # Q12 ------------------------------------------------------------------
    check(out["Q12"] is None and len(blocks["Q12"].strip()) > 500, "Q12: recommendation paragraph present")

    # content-safety guard on the built database ----------------------------
    free_text = con.execute("SELECT DISTINCT review_note FROM human_reviews").df()["review_note"]
    cols = con.execute("""
        SELECT table_name, column_name FROM information_schema.columns
        WHERE data_type = 'VARCHAR' AND table_schema = 'main'
    """).df()
    banned = re.compile(r"\b(i want|i feel|i am|i'm|kill|cut|pills|starve|purge|overdose|hang)\b", re.I)
    hits = []
    for t, c in cols.itertuples(index=False):
        vals = con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL').df().iloc[:, 0].astype(str)
        hits += [f"{t}.{c}" for v in vals if banned.search(v)]
    check(not hits, "Safety: no first-person or method-descriptive tokens in any text column")
    check(len(free_text) <= 15, f"Safety: review_note drawn from a fixed template list ({len(free_text)} distinct)")

    con.close()
    failed = [m for ok, m in res if not ok]
    print(f"\n{len(res) - len(failed)}/{len(res)} assertions passed")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
