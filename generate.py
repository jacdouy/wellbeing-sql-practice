#!/usr/bin/env python3
"""
Synthetic dataset generator for trust-and-safety / user-wellbeing enforcement
analysis practice in SQL.

Deterministic (seed=42). Dependencies: numpy, pandas, duckdb, faker.

CONTENT CONSTRAINT: no table contains simulated user text. Content is
represented only through categorical labels, numeric scores, and abstract
codebook descriptors. The only free-text field (human_reviews.review_note)
is drawn from a fixed list of clinical-register reviewer notes that never
describe methods, plans, or specifics.

Outputs:
  ./data/*.csv            one CSV per table
  ./wellbeing.duckdb      all tables loaded
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
from faker import Faker

# ----------------------------------------------------------------------------
# Configuration / planted parameters
# ----------------------------------------------------------------------------
SEED = 42
N_CONVERSATIONS = 250_000
N_USERS = 70_000
N_DAYS = 90
START = datetime(2026, 6, 1, 0, 0, 0)  # day 0

MODEL_SWITCH_DAY = 45          # model_version changes
CLASSIFIER_V2_DAY = 60         # classifier v2 starts emitting scores
VARIANT_B_LAUNCH_DAY = 30      # A/B: B launched for 50% of eligible traffic
BACKLOG_WEEK = 7               # 1-indexed week with review backlog spike (days 42-48)
FLAG_THRESHOLD = 0.70

CATEGORIES = ["benign", "self_harm", "suicide", "disordered_eating",
              "emotional_dependence", "sycophancy_concern"]
RISK_CATEGORIES = CATEGORIES[1:]

# Category prevalence by model_version (model v2 reduces dependence/sycophancy)
CATEGORY_P = {
    "m-2026.05": {"benign": 0.870, "self_harm": 0.035, "suicide": 0.015,
                  "disordered_eating": 0.020, "emotional_dependence": 0.035,
                  "sycophancy_concern": 0.025},
    "m-2026.07": {"benign": 0.895, "self_harm": 0.035, "suicide": 0.015,
                  "disordered_eating": 0.020, "emotional_dependence": 0.021,
                  "sycophancy_concern": 0.014},
}
SEVERITY_P = [0.50, 0.30, 0.15, 0.05]   # tiers 1..4 given risk category

PLATFORMS = (["web", "ios", "android", "api"], [0.45, 0.25, 0.20, 0.10])
LOCALES = (["en-US", "en-GB", "es-ES", "es-MX", "pt-BR", "de-DE", "fr-FR", "ja-JP", "hi-IN"],
           [0.42, 0.10, 0.08, 0.07, 0.09, 0.06, 0.06, 0.05, 0.07])
MISCALIBRATED_LOCALES = {"pt-BR", "hi-IN"}   # v1 over-scores low-tier content here
MISCAL_LOGIT_SHIFT = 1.1

# Classifier score model: logit = tier_mean + noise
CLF = {
    "v1": {"tier_mean": [-2.3, -0.7, 0.5, 1.15, 1.7], "sigma": 1.00},
    # v2: higher recall on tiers 3-4, but tiers 0-1 pushed up -> worse precision
    "v2": {"tier_mean": [-1.9, 0.35, 0.8, 2.6, 3.4], "sigma": 0.90},
}

REVIEW_SAMPLE_FLAGGED = 0.70
REVIEW_SAMPLE_UNFLAGGED = 0.04
DOUBLE_REVIEW_RATE = 0.08
REVIEWER_NOISE_SD = 0.55

# Follow-up model
RECUR_BASE = {0: 0.03, 1: 0.10, 2: 0.30, 3: 0.45, 4: 0.60}   # P(recurrence) by true tier
DECISION_MULT = {"not_reviewed": 1.00, "no_action": 1.00, "resource_shown": 1.00,
                 "steer_response": 0.65, "escalate": 0.45}
VARIANT_B_MULT = {0: 1.0, 1: 1.0, 2: 0.75, 3: 0.75, 4: 1.0}   # direct effect
MODEL_V2_RECUR_MULT = 1.40   # time confounder: later period has higher recurrence


def sha(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


# ----------------------------------------------------------------------------
# 1. codebook
# ----------------------------------------------------------------------------
def build_codebook() -> pd.DataFrame:
    rows = [("benign", 0, "no wellbeing-relevant indicators present")]
    tiers = {
        "self_harm": [
            "expressed distress, no self-harm risk indicators",
            "passive self-harm indicators, no intent expressed",
            "active concern with intent indicators, no specifics",
            "imminent-risk indicators, escalation required",
        ],
        "suicide": [
            "expressed distress, no suicide risk indicators",
            "passive ideation indicators, no intent expressed",
            "active concern with intent indicators, no specifics",
            "imminent-risk indicators, escalation required",
        ],
        "disordered_eating": [
            "body-image or diet concern expressed, no risk indicators",
            "restrictive-pattern indicators, no medical-risk indicators",
            "active concern with compensatory-behavior indicators, no specifics",
            "medical-risk indicators, escalation required",
        ],
        "emotional_dependence": [
            "mild reliance signals within expected use",
            "elevated reliance signals; reduced external supports referenced",
            "marked dependence indicators; distress at unavailability",
            "acute dependence with wellbeing-risk indicators, escalation required",
        ],
        "sycophancy_concern": [
            "minor agreement bias in a low-stakes context",
            "validation of a questionable premise in a moderate-stakes context",
            "reinforcement of a harmful belief pattern; correction needed",
            "sustained reinforcement in a high-stakes context, escalation required",
        ],
    }
    for cat, descs in tiers.items():
        rows.append((cat, 0, "screened; no indicators for this category"))
        for i, d in enumerate(descs, start=1):
            rows.append((cat, i, d))
    return pd.DataFrame(rows, columns=["category", "severity_tier", "descriptor"])


# ----------------------------------------------------------------------------
# 2. conversations (+ ground truth)
# ----------------------------------------------------------------------------
def build_conversations(rng: np.random.Generator) -> pd.DataFrame:
    n = N_CONVERSATIONS
    # --- timestamps: day weights (weekly pattern x gentle growth), hour weights (diurnal)
    days = np.arange(N_DAYS)
    dow = np.array([(START + timedelta(days=int(d))).weekday() for d in days])
    weekly = np.array([1.00, 1.05, 1.06, 1.03, 0.96, 0.80, 0.78])[dow]
    growth = np.linspace(0.85, 1.20, N_DAYS)
    day_w = weekly * growth
    day_w /= day_w.sum()
    hour_w = np.array([0.55, 0.40, 0.30, 0.22, 0.18, 0.20, 0.30, 0.50, 0.75, 0.95, 1.05, 1.10,
                       1.10, 1.08, 1.05, 1.05, 1.10, 1.20, 1.35, 1.50, 1.60, 1.55, 1.30, 0.90])
    hour_w /= hour_w.sum()

    day = rng.choice(N_DAYS, size=n, p=day_w)
    hour = rng.choice(24, size=n, p=hour_w)
    sec = rng.integers(0, 3600, size=n)
    created_at = np.array([START + timedelta(days=int(d), hours=int(h), seconds=int(s))
                           for d, h, s in zip(day, hour, sec)])
    order = np.argsort(created_at)
    day, created_at = day[order], created_at[order]

    # --- users: power-law-ish activity
    user_w = rng.pareto(1.6, size=N_USERS) + 1
    user_w /= user_w.sum()
    user_idx = rng.choice(N_USERS, size=n, p=user_w)
    user_id = np.array([sha(f"user:{u}") for u in user_idx])

    platform = rng.choice(PLATFORMS[0], size=n, p=PLATFORMS[1])
    locale = rng.choice(LOCALES[0], size=n, p=LOCALES[1])
    model_version = np.where(day < MODEL_SWITCH_DAY, "m-2026.05", "m-2026.07")

    # --- hidden ground truth
    true_category = np.empty(n, dtype=object)
    for mv, probs in CATEGORY_P.items():
        m = model_version == mv
        true_category[m] = rng.choice(CATEGORIES, size=m.sum(), p=[probs[c] for c in CATEGORIES])
    true_severity = np.zeros(n, dtype=int)
    risk = true_category != "benign"
    true_severity[risk] = rng.choice([1, 2, 3, 4], size=risk.sum(), p=SEVERITY_P)

    conv_id = np.array([f"c_{sha(f'conv:{i}', 12)}" for i in range(n)])
    return pd.DataFrame({
        "conversation_id": conv_id,
        "user_id": user_id,
        "created_at": created_at,
        "day_index": day,                  # dropped from analyst view
        "platform": platform,
        "locale": locale,
        "model_version": model_version,
        "true_category": true_category,
        "true_severity": true_severity,
    })


# ----------------------------------------------------------------------------
# 3. classifier_events
# ----------------------------------------------------------------------------
def score_classifier(rng, conv: pd.DataFrame, version: str) -> pd.DataFrame:
    p = CLF[version]
    tier = conv["true_severity"].to_numpy()
    logit = np.array(p["tier_mean"])[tier] + rng.normal(0, p["sigma"], len(conv))
    if version == "v1":
        miscal = conv["locale"].isin(MISCALIBRATED_LOCALES).to_numpy() & (tier <= 1)
        logit = logit + miscal * MISCAL_LOGIT_SHIFT
    score = 1 / (1 + np.exp(-logit))
    score = np.round(np.clip(score, 0.0005, 0.9995), 4)

    # predicted category: benign when low score; otherwise mostly the true category
    tc = conv["true_category"].to_numpy()
    pred = np.where(score < 0.5, "benign", tc).astype(object)
    high = score >= 0.5
    benign_high = high & (tc == "benign")
    pred[benign_high] = rng.choice(RISK_CATEGORIES, size=benign_high.sum(),
                                   p=[0.30, 0.15, 0.15, 0.25, 0.15])
    swap = high & (tc != "benign") & (rng.random(len(conv)) < 0.15)
    pred[swap] = rng.choice(RISK_CATEGORIES, size=swap.sum())
    low_risk = (~high) & (rng.random(len(conv)) < 0.08)
    pred[low_risk] = rng.choice(RISK_CATEGORIES, size=low_risk.sum())

    scored_at = conv["created_at"] + pd.to_timedelta(rng.integers(1, 6, len(conv)), unit="s")
    return pd.DataFrame({
        "event_id": [f"e_{sha(f'{version}:{c}', 12)}" for c in conv["conversation_id"]],
        "conversation_id": conv["conversation_id"].to_numpy(),
        "classifier_version": version,
        "scored_at": scored_at,
        "score": score,
        "predicted_category": pred,
        "flagged": score >= FLAG_THRESHOLD,
    })


def build_classifier_events(rng, conv: pd.DataFrame) -> pd.DataFrame:
    v1 = score_classifier(rng, conv, "v1")
    v2 = score_classifier(rng, conv[conv["day_index"] >= CLASSIFIER_V2_DAY].reset_index(drop=True), "v2")
    return pd.concat([v1, v2], ignore_index=True)


# ----------------------------------------------------------------------------
# 4. reviewers + human_reviews
# ----------------------------------------------------------------------------
def build_reviewers(rng, fake: Faker) -> pd.DataFrame:
    n = 12
    strictness = np.round(rng.uniform(-0.6, 0.6, n), 3)
    strictness[4] = -0.35   # R05 starts lenient ...
    strictness[8] = +0.45   # R09 starts strict ...
    drift_per_day = np.zeros(n)
    drift_per_day[4] = +1.1 / N_DAYS    # ... and becomes stricter over time
    drift_per_day[8] = -1.0 / N_DAYS    # ... and becomes more lenient over time
    workload = rng.dirichlet(np.ones(n) * 6)
    teams = rng.choice(["alpha", "bravo", "charlie"], size=n)
    shifts = rng.choice(["day", "swing", "night"], size=n, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({
        "reviewer_id": [f"R{i+1:02d}" for i in range(n)],
        "reviewer_handle": [fake.unique.user_name() for _ in range(n)],
        "team": teams,
        "shift": shifts,
        "onboarded_on": [fake.date_between(start_date="-3y", end_date="-6M") for _ in range(n)],
        # hidden parameters (moved to ground_truth_reviewers)
        "strictness": strictness,
        "drift_per_day": np.round(drift_per_day, 6),
        "workload_share": np.round(workload, 4),
    })


REVIEW_NOTES = {
    0: ["no risk indicators present; no action taken",
        "screened; benign on review; classifier false positive",
        "no wellbeing-relevant indicators; routine closure"],
    1: ["low-tier distress indicators; resource card displayed",
        "tier 1 indicators; no risk indicators present; resource card displayed",
        "mild indicators noted; no action beyond standard resources"],
    2: ["tier 2 indicators present; response steer applied",
        "moderate indicators; resource card displayed; monitoring recommended",
        "passive indicators without intent; steer and resources applied"],
    3: ["active concern indicators; response steer applied; escalation considered",
        "tier 3 indicators; escalated per protocol",
        "intent indicators without specifics; steer applied; flagged for follow-up"],
    4: ["imminent-risk indicators; escalated per protocol",
        "tier 4 indicators; escalation completed; crisis resources displayed",
        "escalated immediately per rubric; case handed to specialist queue"],
}


def build_human_reviews(rng, conv: pd.DataFrame, events: pd.DataFrame,
                        reviewers: pd.DataFrame, variant: pd.Series) -> pd.DataFrame:
    flagged_any = events[events["flagged"]].groupby("conversation_id").size()
    is_flagged = conv["conversation_id"].isin(flagged_any.index).to_numpy()
    u = rng.random(len(conv))
    selected = (is_flagged & (u < REVIEW_SAMPLE_FLAGGED)) | (~is_flagged & (u < REVIEW_SAMPLE_UNFLAGGED))
    base = conv[selected].reset_index(drop=True)

    # 8% get a second review by a different reviewer
    dup_mask = rng.random(len(base)) < DOUBLE_REVIEW_RATE
    second = base[dup_mask].reset_index(drop=True)
    parts = []
    for pass_no, df in ((1, base), (2, second)):
        df = df.copy()
        n = len(df)
        rid = rng.choice(len(reviewers), size=n, p=reviewers["workload_share"] / reviewers["workload_share"].sum())
        if pass_no == 2:
            # ensure a different reviewer than pass 1 for the same conversation
            first = parts[0].set_index("conversation_id")["_rid"]
            clash = rid == first.loc[df["conversation_id"]].to_numpy()
            rid[clash] = (rid[clash] + rng.integers(1, len(reviewers), clash.sum())) % len(reviewers)
        df["_rid"] = rid

        # queue latency: lognormal, backlog spike in week 7
        lat_h = rng.lognormal(mean=np.log(4.0), sigma=0.9, size=n)
        week = df["day_index"].to_numpy() // 7 + 1
        spike = week == BACKLOG_WEEK
        lat_h[spike] *= rng.uniform(3.5, 6.0, spike.sum())
        if pass_no == 2:
            lat_h *= rng.uniform(1.2, 2.5, n)    # QA re-reviews come later
        df["reviewed_at"] = df["created_at"] + pd.to_timedelta(lat_h * 3600, unit="s")

        # severity judgement: truth + strictness + drift + noise
        strict = reviewers["strictness"].to_numpy()[rid]
        drift = reviewers["drift_per_day"].to_numpy()[rid] * df["day_index"].to_numpy()
        sev = df["true_severity"].to_numpy() + strict + drift + rng.normal(0, REVIEWER_NOISE_SD, n)
        sev = np.clip(np.rint(sev), 0, 4).astype(int)
        # a benign conversation rarely gets bumped above tier 2 by a reviewer
        benign = df["true_category"].to_numpy() == "benign"
        sev[benign] = np.minimum(sev[benign], np.where(rng.random(n) < 0.9, 1, 2)[benign])
        df["reviewed_severity"] = sev

        # category: true category most of the time, benign if severity 0
        cat = df["true_category"].to_numpy().astype(object)
        confuse = (rng.random(n) < 0.10) & (cat != "benign")
        cat[confuse] = rng.choice(RISK_CATEGORIES, size=confuse.sum())
        need_cat = (sev > 0) & (cat == "benign")
        cat[need_cat] = rng.choice(RISK_CATEGORIES, size=need_cat.sum(), p=[0.30, 0.10, 0.15, 0.30, 0.15])
        cat[sev == 0] = "benign"
        df["reviewed_category"] = cat

        # decision: driven by reviewed severity; variant B nudges reviewers toward
        # lighter decisions on tiers 2-3 ("resource already displayed")
        dec_p = {
            0: [0.97, 0.03, 0.00, 0.00],
            1: [0.40, 0.58, 0.02, 0.00],
            2: [0.05, 0.50, 0.35, 0.10],
            3: [0.01, 0.14, 0.55, 0.30],
            4: [0.00, 0.02, 0.15, 0.83],
        }
        dec_p_b = {
            2: [0.17, 0.70, 0.10, 0.03],
            3: [0.05, 0.35, 0.45, 0.15],
        }
        v = variant.reindex(df["conversation_id"]).to_numpy()
        decisions = np.empty(n, dtype=object)
        opts = np.array(["no_action", "resource_shown", "steer_response", "escalate"])
        for t in range(5):
            for is_b in (False, True):
                m = (sev == t) & ((v == "B") == is_b)
                if not m.any():
                    continue
                p = dec_p_b.get(t, dec_p[t]) if is_b else dec_p[t]
                decisions[m] = rng.choice(opts, size=m.sum(), p=p)
        df["decision"] = decisions
        df["review_note"] = [REVIEW_NOTES[t][rng.integers(0, 3)] for t in sev]
        df["review_pass"] = pass_no
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out["reviewer_id"] = reviewers["reviewer_id"].to_numpy()[out["_rid"]]
    out["review_id"] = [f"r_{sha(f'{c}:{p}', 12)}" for c, p in zip(out["conversation_id"], out["review_pass"])]
    out = out.sort_values("reviewed_at").reset_index(drop=True)
    return out[["review_id", "conversation_id", "reviewer_id", "reviewed_at", "reviewed_category",
                "reviewed_severity", "decision", "review_note"]]


# ----------------------------------------------------------------------------
# 5. interventions
# ----------------------------------------------------------------------------
def build_interventions(rng, conv: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    # real-time intervention is driven by the max score across active classifiers
    mx = events.groupby("conversation_id")["score"].max()
    df = conv[["conversation_id", "created_at", "day_index"]].copy()
    df["max_score"] = mx.reindex(df["conversation_id"]).to_numpy()
    df = df[df["max_score"] >= FLAG_THRESHOLD].reset_index(drop=True)
    itype = np.where(df["max_score"] >= 0.85, "both", "resource_card").astype(object)
    df["intervention_type"] = itype
    eligible_b = (df["day_index"] >= VARIANT_B_LAUNCH_DAY).to_numpy()
    # deterministic 50/50 bucket from the conversation id
    bucket = np.array([int(c[-2:], 16) % 2 for c in df["conversation_id"]])
    df["variant"] = np.where(eligible_b & (bucket == 1), "B", "A")
    df["shown_at"] = df["created_at"] + pd.to_timedelta(rng.integers(2, 12, len(df)), unit="s")
    df["intervention_id"] = [f"i_{sha(f'int:{c}', 12)}" for c in df["conversation_id"]]
    return df[["intervention_id", "conversation_id", "intervention_type", "variant", "shown_at"]]


# ----------------------------------------------------------------------------
# 6. followup_signals
# ----------------------------------------------------------------------------
def build_followups(rng, conv: pd.DataFrame, interventions: pd.DataFrame,
                    reviews: pd.DataFrame) -> pd.DataFrame:
    has_int = conv["conversation_id"].isin(interventions["conversation_id"])
    df = conv[(conv["true_severity"] >= 1) | has_int].reset_index(drop=True)
    n = len(df)
    tier = df["true_severity"].to_numpy()

    variant = interventions.set_index("conversation_id")["variant"].reindex(df["conversation_id"]).fillna("none").to_numpy()
    itype = interventions.set_index("conversation_id")["intervention_type"].reindex(df["conversation_id"]).fillna("none").to_numpy()
    # first-pass decision (the one that actually drove handling)
    first = reviews.sort_values("reviewed_at").drop_duplicates("conversation_id").set_index("conversation_id")["decision"]
    decision = first.reindex(df["conversation_id"]).fillna("not_reviewed").to_numpy()

    p = np.array([RECUR_BASE[t] for t in tier])
    p *= np.array([DECISION_MULT[d] for d in decision])
    p *= np.where(variant == "B", np.array([VARIANT_B_MULT[t] for t in tier]), 1.0)
    p *= np.where(df["model_version"].to_numpy() == "m-2026.07", MODEL_V2_RECUR_MULT, 1.0)
    p = np.clip(p, 0, 0.95)
    recur = rng.random(n) < p

    # return probability rises with tier; recurrence implies return
    p_ret = np.array([0.35, 0.45, 0.55, 0.62, 0.70])[tier]
    returned = (rng.random(n) < p_ret) | recur

    # follow-up severity: recurrence => >=2 (near the original tier); else low
    fs = np.zeros(n, dtype=int)
    fs[recur] = np.clip(np.rint(np.maximum(tier[recur], 2) + rng.normal(-0.2, 0.6, recur.sum())), 2, 4)
    non = returned & ~recur
    fs[non] = np.clip(np.rint(rng.normal(0.4, 0.55, non.sum())), 0, 1)
    fc = df["true_category"].to_numpy().astype(object).copy()
    fc[~returned] = None
    fc[returned & (fs == 0)] = "benign"
    newcat = returned & (fs > 0) & (fc == "benign")
    fc[newcat] = rng.choice(RISK_CATEGORIES, size=newcat.sum(), p=[0.30, 0.10, 0.15, 0.30, 0.15])
    fs_out = np.where(returned, fs, np.nan)

    click_p = np.where(variant == "B", 0.19, 0.12) * (itype != "none")
    clicked = rng.random(n) < click_p

    return pd.DataFrame({
        "conversation_id": df["conversation_id"].to_numpy(),
        "returned_within_7d": returned,
        "followup_category": fc,
        "followup_severity": pd.array(fs_out, dtype="Int64"),
        "resource_link_clicked": clicked,
    })


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    Faker.seed(SEED)
    fake = Faker()

    print("building codebook ...")
    codebook = build_codebook()
    print("building conversations ...")
    conv = build_conversations(rng)
    print("building classifier_events ...")
    events = build_classifier_events(rng, conv)
    print("building reviewers ...")
    reviewers = build_reviewers(rng, fake)
    print("building interventions ...")
    interventions = build_interventions(rng, conv, events)
    variant = interventions.set_index("conversation_id")["variant"]
    print("building human_reviews ...")
    reviews = build_human_reviews(rng, conv, events, reviewers, variant)
    print("building followup_signals ...")
    followups = build_followups(rng, conv, interventions, reviews)

    conversations = conv[["conversation_id", "user_id", "created_at", "platform", "locale", "model_version"]]
    ground_truth = conv[["conversation_id", "true_category", "true_severity"]]
    reviewers_public = reviewers[["reviewer_id", "reviewer_handle", "team", "shift", "onboarded_on"]]
    ground_truth_reviewers = reviewers[["reviewer_id", "strictness", "drift_per_day", "workload_share"]]

    tables = {
        "codebook": codebook,
        "conversations": conversations,
        "classifier_events": events,
        "reviewers": reviewers_public,
        "human_reviews": reviews,
        "interventions": interventions,
        "followup_signals": followups,
        "ground_truth": ground_truth,
        "ground_truth_reviewers": ground_truth_reviewers,
    }

    # ---- content-safety guard: the only free text is review_note from the fixed list
    allowed_notes = {s for v in REVIEW_NOTES.values() for s in v}
    assert set(reviews["review_note"]).issubset(allowed_notes)
    banned = ["i want", "i feel", "i'm", "i am", "kill", "cut", "pills", "starve", "purge", "overdose", "hang"]
    for name, df in tables.items():
        for col in df.select_dtypes(include=["object", "str"]).columns:
            vals = df[col].dropna().astype(str).str.lower()
            for b in banned:
                assert not vals.str.contains(rf"\b{b}\b", regex=True).any(), f"banned token {b!r} in {name}.{col}"

    print("writing CSVs ...")
    for name, df in tables.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)

    planted = {
        "seed": SEED, "start_date": START.date().isoformat(), "n_days": N_DAYS,
        "model_switch_day": MODEL_SWITCH_DAY, "classifier_v2_day": CLASSIFIER_V2_DAY,
        "variant_b_launch_day": VARIANT_B_LAUNCH_DAY, "backlog_week": BACKLOG_WEEK,
        "flag_threshold": FLAG_THRESHOLD, "miscalibrated_locales_v1": sorted(MISCALIBRATED_LOCALES),
        "miscal_logit_shift": MISCAL_LOGIT_SHIFT, "classifier": CLF,
        "recurrence_base_by_tier": RECUR_BASE, "decision_multipliers": DECISION_MULT,
        "variant_b_multiplier_by_tier": VARIANT_B_MULT, "model_v2_recurrence_multiplier": MODEL_V2_RECUR_MULT,
        "drifting_reviewers": {"R05": "+1.1 tiers over 90 days", "R09": "-0.9 tiers over 90 days"},
        "category_prevalence_by_model_version": CATEGORY_P,
    }
    with open(os.path.join(data_dir, "planted_effects.json"), "w") as f:
        json.dump(planted, f, indent=2, default=str)

    print("loading DuckDB ...")
    db_path = os.path.join(root, "wellbeing.duckdb")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = duckdb.connect(db_path)
    for name, df in tables.items():
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
    con.close()

    print("\nrow counts:")
    for name, df in tables.items():
        print(f"  {name:24s} {len(df):>8,d}")
    print(f"\nwrote {db_path}")


if __name__ == "__main__":
    main()
    if "--no-validate" not in sys.argv:
        import validate
        validate.main()
