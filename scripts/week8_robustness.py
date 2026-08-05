"""Out-of-time validation (train pre-COVID 2018-2019, test 2020-2021 + 2022-2024), yearly drift
table, race-stratified per-group models, and threshold/training-cap sensitivity analysis.

The out-of-time experiment answers a question random splits cannot: does a model trained entirely
before the pandemic generalize through a regime change (COVID-era care utilization shifts) and
into the recent recovery? The race-stratified models ask whether group-specific training buys
anything over the pooled model for each demographic, and the sensitivity sweeps confirm the
operating-point choice and training-cap sufficiency documented in Weeks 4-5.

Run: POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week8_robustness.py
Smoke: EQUIBIRTH_SMOKE=1 POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week8_robustness.py
"""

from __future__ import annotations

import gc
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from equibirth.eval import robustness
from equibirth.features import leakage
from equibirth.models import boosting
from equibirth.models import evaluate as ev
from equibirth.utils.config import load_config
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
DATA = ROOT / "data" / "processed" / "modeling_full_2018_2024.parquet"
WEEK4_JSON = ROOT / "reports" / "week4_boosting_fairness.json"
WEEK5_JSON = ROOT / "reports" / "week5_models.json"
JSON_OUT = ROOT / "reports" / "week8_robustness.json"
MD_OUT = ROOT / "reports" / "week8_robustness.md"
SEED = CFG.get("project.random_seed")
NON_FEATURES = {"preterm", "split", "mother_race_ethnicity", "pay", "birth_year"}

SMOKE = os.environ.get("EQUIBIRTH_SMOKE") == "1"
TRAIN_CAP = 20_000 if SMOKE else 500_000
OOT_TRAIN_CAP = 20_000 if SMOKE else 200_000
HO_CAP = 20_000 if SMOKE else 300_000
GROUP_TRAIN_CAP = 10_000 if SMOKE else 100_000
GROUP_HO_CAP = 10_000 if SMOKE else 50_000
GROUP_MIN_ROWS = 10_000

EXPECTED_SPLITS = {"dev_train": 1_050_000, "dev_val": 225_000,
                   "dev_test": 225_000, "held_out_final": 1_500_000}


def _validate(df: pl.DataFrame) -> None:
    """Fail loudly if the parquet is not the exact canonical full-data build."""
    if len(df) != 3_000_000:
        raise SystemExit(f"expected 3,000,000 rows, got {len(df):,}: wrong parquet build")
    counts = {s: n for s, n in df.group_by("split").len().iter_rows()}
    if counts != EXPECTED_SPLITS:
        raise SystemExit(f"split counts {counts} != expected {EXPECTED_SPLITS}")
    prev = df.filter(pl.col("preterm").is_not_null()).group_by("split").agg(
        pl.col("preterm").mean().alias("prev"))
    for s, p in prev.iter_rows():
        if not 0.084 <= p <= 0.088:
            raise SystemExit(f"prevalence {p:.4f} in split {s!r} outside [0.084, 0.088]")
    barred = leakage.schema.all_excluded_vars()
    present = barred & set(df.columns)
    if present:
        raise SystemExit(f"leakage columns present in the modeling parquet: {sorted(present)}")


def _cap_mask(mask: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    """Seeded subsample of a boolean mask down to cap true rows."""
    idx = np.flatnonzero(mask)
    if cap is None or idx.size <= cap:
        return mask
    drop = np.random.default_rng(seed).choice(idx, idx.size - cap, replace=False)
    mask = mask.copy()
    mask[drop] = False
    return mask


def main() -> None:
    set_seeds(SEED)
    t0 = time.time()

    # -- Load data and validate --
    df = pl.read_parquet(DATA)
    _validate(df)

    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["preterm"].fill_null(-1).to_numpy().astype(int)
    split = df["split"].to_numpy()
    race = df["mother_race_ethnicity"].to_numpy()
    birth_year = df["birth_year"].to_numpy()
    del df
    gc.collect()
    valid = y >= 0

    # Load week-4 params (xgboost selected in week 5)
    wk4 = json.loads(WEEK4_JSON.read_text())
    xgb_params = wk4["params"]["xgboost"]
    threshold = wk4["threshold"]

    print(f"smoke={SMOKE} features={len(feats)} threshold={threshold:.4f}", flush=True)

    # ======================================================================
    # STEP 2: Out-of-time split
    # ======================================================================
    print(f"\n[{time.time() - t0:7.1f}s] === Out-of-time validation ===", flush=True)

    era_masks = robustness.out_of_time_masks(birth_year)
    train_precovid = era_masks["train_precovid"]
    test_covid = era_masks["test_covid"]
    test_recent = era_masks["test_recent"]

    # Intersect era masks with split masks
    dev_train_val = (split == "dev_train") | (split == "dev_val")
    dev_test_mask = split == "dev_test"
    ho_mask = split == "held_out_final"

    oot_train = _cap_mask(train_precovid & dev_train_val & valid, OOT_TRAIN_CAP, SEED)
    oot_cal = train_precovid & dev_test_mask & valid  # use all available
    oot_test_covid = test_covid & ho_mask & valid
    oot_test_recent = test_recent & ho_mask & valid

    print(f"  OOT train (pre-COVID, dev_train+dev_val): {oot_train.sum():,}", flush=True)
    print(f"  OOT cal (pre-COVID, dev_test): {oot_cal.sum():,}", flush=True)
    print(f"  OOT test COVID (2020-2021, held_out): {oot_test_covid.sum():,}", flush=True)
    print(f"  OOT test recent (2022-2024, held_out): {oot_test_recent.sum():,}", flush=True)

    # Fit OOT model
    oot_est = boosting.fit_calibrated("xgboost", X, y, oot_train, oot_cal, xgb_params, seed=SEED)
    p_covid = oot_est.predict_proba(X[oot_test_covid])[:, 1]
    p_recent = oot_est.predict_proba(X[oot_test_recent])[:, 1]

    oot_results = {
        "train_era": "2018-2019 (pre-COVID)",
        "train_n": int(oot_train.sum()),
        "cal_n": int(oot_cal.sum()),
        "covid_era": {
            "years": "2020-2021",
            "n": int(oot_test_covid.sum()),
            "auc_pr": ev.auc_pr(y[oot_test_covid], p_covid),
            "auc_roc": ev.auc_roc(y[oot_test_covid], p_covid),
            "ece": ev.ece(y[oot_test_covid], p_covid),
        },
        "recent_era": {
            "years": "2022-2024",
            "n": int(oot_test_recent.sum()),
            "auc_pr": ev.auc_pr(y[oot_test_recent], p_recent),
            "auc_roc": ev.auc_roc(y[oot_test_recent], p_recent),
            "ece": ev.ece(y[oot_test_recent], p_recent),
        },
    }
    print(f"  COVID era: AUC-PR={oot_results['covid_era']['auc_pr']:.4f} "
          f"AUC-ROC={oot_results['covid_era']['auc_roc']:.4f} "
          f"ECE={oot_results['covid_era']['ece']:.4f}", flush=True)
    print(f"  Recent era: AUC-PR={oot_results['recent_era']['auc_pr']:.4f} "
          f"AUC-ROC={oot_results['recent_era']['auc_roc']:.4f} "
          f"ECE={oot_results['recent_era']['ece']:.4f}", flush=True)
    del oot_est
    gc.collect()

    # ======================================================================
    # STEP 3: Drift table (full selected model scored on held-out)
    # ======================================================================
    print(f"\n[{time.time() - t0:7.1f}s] === Yearly drift table ===", flush=True)

    # Fit the full-years model (same as week5/6: 500K-capped dev_train, calibrated on dev_test)
    tr_full = _cap_mask((split == "dev_train") & valid, TRAIN_CAP, SEED)
    cal_full = (split == "dev_test") & valid
    full_est = boosting.fit_calibrated("xgboost", X, y, tr_full, cal_full, xgb_params, seed=SEED)

    # Score held-out partition (capped to 300K seeded draw)
    ho_full = _cap_mask((split == "held_out_final") & valid, HO_CAP, SEED)
    p_ho = full_est.predict_proba(X[ho_full])[:, 1]
    y_ho = y[ho_full]
    birth_year_ho = birth_year[ho_full]

    drift_rows = robustness.yearly_drift(y_ho, p_ho, birth_year_ho)
    print(f"  Drift table ({len(drift_rows)} years):", flush=True)
    for row in drift_rows:
        print(f"    {row['year']}: n={row['n']:,} prev={row['prevalence']:.4f} "
              f"AUC-ROC={row['auc_roc']:.4f} AUC-PR={row['auc_pr']:.4f} "
              f"ECE={row['ece']:.4f}", flush=True)

    # ======================================================================
    # STEP 4: Race-stratified models
    # ======================================================================
    print(f"\n[{time.time() - t0:7.1f}s] === Race-stratified models ===", flush=True)

    # Identify groups with >= GROUP_MIN_ROWS in dev_train
    dev_train_mask = (split == "dev_train") & valid
    # race may contain None; filter those out before finding unique groups
    race_in_train = race[dev_train_mask]
    unique_groups = sorted(set(str(g) for g in race_in_train if g is not None and str(g) != "nan"))
    eligible_groups = []
    for g_str in unique_groups:
        n_in_train = int(((race == g_str) & dev_train_mask).sum())
        if n_in_train >= GROUP_MIN_ROWS:
            eligible_groups.append(g_str)

    print(f"  Eligible groups (>={GROUP_MIN_ROWS:,} in dev_train): "
          f"{[str(g) for g in eligible_groups]}", flush=True)

    stratified_results = {}
    for g_str in eligible_groups:
        g_mask = race == g_str
        g_tr = _cap_mask(g_mask & dev_train_mask & valid, GROUP_TRAIN_CAP, SEED)
        g_cal = g_mask & (split == "dev_test") & valid
        g_ho = _cap_mask(g_mask & (split == "held_out_final") & valid, GROUP_HO_CAP, SEED)

        if g_tr.sum() < 100 or g_cal.sum() < 50 or g_ho.sum() < 50:
            continue
        if len(np.unique(y[g_tr])) < 2 or len(np.unique(y[g_cal])) < 2:
            continue

        g_est = boosting.fit_calibrated("xgboost", X, y, g_tr, g_cal, xgb_params, seed=SEED)
        g_proba = g_est.predict_proba(X[g_ho])[:, 1]

        # Top-10 feature importances from the base booster
        # CalibratedClassifierCV -> calibrated_classifiers_[0].estimator (FrozenEstimator)
        base_booster = g_est.calibrated_classifiers_[0].estimator
        fi = base_booster.feature_importances_
        top_idx = np.argsort(fi)[::-1][:10]
        top_features = [(feats[i], float(fi[i])) for i in top_idx]

        stratified_results[g_str] = {
            "train_n": int(g_tr.sum()),
            "cal_n": int(g_cal.sum()),
            "held_out_n": int(g_ho.sum()),
            "prevalence": float(y[g_ho].mean()),
            "auc_pr": ev.auc_pr(y[g_ho], g_proba),
            "auc_roc": ev.auc_roc(y[g_ho], g_proba),
            "ece": ev.ece(y[g_ho], g_proba),
            "top_features": top_features,
            "hypothesis_only": True,
        }
        print(f"    {g_str}: train={int(g_tr.sum()):,} ho={int(g_ho.sum()):,} "
              f"AUC-PR={stratified_results[g_str]['auc_pr']:.4f} "
              f"AUC-ROC={stratified_results[g_str]['auc_roc']:.4f}", flush=True)
        del g_est
        gc.collect()

    # ======================================================================
    # STEP 5: Sensitivity analysis (skip in SMOKE mode)
    # ======================================================================
    sensitivity_threshold = None
    sensitivity_cap = None

    if not SMOKE:
        print(f"\n[{time.time() - t0:7.1f}s] === Sensitivity analysis ===", flush=True)

        # 5a: Threshold sensitivity
        race_ho = race[ho_full]
        thresholds = [
            threshold * 0.50,
            threshold * 0.75,
            threshold,
            threshold * 1.25,
            threshold * 1.50,
        ]
        sensitivity_threshold = []
        for thr in thresholds:
            sens, spec = ev.sensitivity_specificity(y_ho, p_ho, thr)
            # Compute TPR per group for worst EO gap
            tprs = {}
            for g_str in eligible_groups:
                g_mask_ho = race_ho == g_str
                y_g = y_ho[g_mask_ho]
                p_g = p_ho[g_mask_ho]
                pos_mask = y_g == 1
                if pos_mask.sum() > 0:
                    tprs[g_str] = float((p_g[pos_mask] >= thr).mean())
            # Worst EO gap = max TPR - min TPR across groups
            if len(tprs) >= 2:
                worst_eo_gap = max(tprs.values()) - min(tprs.values())
            else:
                worst_eo_gap = float("nan")
            sensitivity_threshold.append({
                "threshold": float(thr),
                "threshold_pct_of_base": float(thr / threshold * 100),
                "sensitivity": sens,
                "specificity": spec,
                "worst_eo_gap": worst_eo_gap,
            })
            print(f"    thr={thr:.4f} ({thr/threshold*100:.0f}%): "
                  f"sens={sens:.4f} spec={spec:.4f} worst_EO_gap={worst_eo_gap:.4f}", flush=True)

        # 5b: Train-cap sensitivity
        print(f"  Training-cap sensitivity:", flush=True)
        sensitivity_cap = []
        for cap in [100_000, 250_000, 500_000]:
            tr_cap = _cap_mask((split == "dev_train") & valid, cap, SEED)
            cap_est = boosting.fit_calibrated(
                "xgboost", X, y, tr_cap, cal_full, xgb_params, seed=SEED)
            cap_proba = cap_est.predict_proba(X[ho_full])[:, 1]
            cap_auc_pr = ev.auc_pr(y_ho, cap_proba)
            sensitivity_cap.append({
                "train_cap": cap,
                "held_out_auc_pr": cap_auc_pr,
            })
            print(f"    cap={cap:,}: held-out AUC-PR={cap_auc_pr:.4f}", flush=True)
            del cap_est
            gc.collect()

    # ======================================================================
    # STEP 6: Serialize
    # ======================================================================
    print(f"\n[{time.time() - t0:7.1f}s] === Writing outputs ===", flush=True)

    result = {
        "out_of_time": oot_results,
        "out_of_time_note": ("This establishes temporal stability WITHIN the NCHS natality system "
                             "(same birth-certificate instrument and coding across 2018-2024), not "
                             "external validity. The model is not validated on any non-NCHS "
                             "population or clinical cohort, so the small out-of-time drop should "
                             "not be read as evidence it would transport to a different data source "
                             "or care setting."),
        "drift": drift_rows,
        "stratified_models": stratified_results,
        "stratified_comparability_note": ("AUC-PR is prevalence-dependent and NOT comparable across "
                                          "groups with different preterm prevalence. A group with "
                                          "higher prevalence (e.g. NH Black) can show a higher "
                                          "AUC-PR partly as a mechanical consequence of its base "
                                          "rate, not because the model discriminates better for it. "
                                          "Use AUC-ROC, which is prevalence-robust, for cross-group "
                                          "discrimination comparison; the per-group prevalence is "
                                          "reported alongside each model for this reason."),
        "sensitivity_threshold": sensitivity_threshold,
        "sensitivity_cap": sensitivity_cap,
        "sensitivity_cap_note": ("The train-cap plateau is an XGBoost property and shows the 500K "
                                 "cap is past the diminishing-returns knee for the boosters; it does "
                                 "NOT license the same claim for the deep nets in Week 5, which are "
                                 "the archetype that keeps improving with data."),
        "caps": {
            "train_cap": TRAIN_CAP,
            "oot_train_cap": OOT_TRAIN_CAP,
            "held_out_cap": HO_CAP,
            "group_train_cap": GROUP_TRAIN_CAP,
            "group_ho_cap": GROUP_HO_CAP,
        },
        "seed": SEED,
        "smoke": SMOKE,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(result, indent=2, default=float))

    # Markdown report
    lines = [
        "# Week 8: out-of-time validation, yearly drift, race-stratified models, sensitivity",
        "",
    ]

    # OOT section
    lines += [
        "## Out-of-time validation",
        "",
        f"Model trained on pre-COVID births (2018-2019, n={oot_results['train_n']:,}), "
        f"isotonic-calibrated on a separate pre-COVID fold (n={oot_results['cal_n']:,}), "
        f"then evaluated on COVID-era and recent held-out data.",
        "",
        "| era | years | n | AUC-PR | AUC-ROC | ECE |",
        "|---|---|---|---|---|---|",
        f"| COVID | {oot_results['covid_era']['years']} | "
        f"{oot_results['covid_era']['n']:,} | "
        f"{oot_results['covid_era']['auc_pr']:.4f} | "
        f"{oot_results['covid_era']['auc_roc']:.4f} | "
        f"{oot_results['covid_era']['ece']:.4f} |",
        f"| Recent | {oot_results['recent_era']['years']} | "
        f"{oot_results['recent_era']['n']:,} | "
        f"{oot_results['recent_era']['auc_pr']:.4f} | "
        f"{oot_results['recent_era']['auc_roc']:.4f} | "
        f"{oot_results['recent_era']['ece']:.4f} |",
        "",
    ]

    # Drift section
    lines += [
        "## Yearly drift (full model, held-out partition)",
        "",
        "| year | n | prevalence | AUC-ROC | AUC-PR | ECE |",
        "|---|---|---|---|---|---|",
    ]
    for row in drift_rows:
        lines.append(
            f"| {row['year']} | {row['n']:,} | {row['prevalence']:.4f} | "
            f"{row['auc_roc']:.4f} | {row['auc_pr']:.4f} | {row['ece']:.4f} |"
        )
    lines.append("")

    # Stratified models section
    lines += [
        "## Race-stratified models (hypothesis_only)",
        "",
        "Per-group XGBoost models trained and evaluated on that group's data only.",
        "",
        "AUC-PR is prevalence-dependent and NOT comparable across groups with different preterm "
        "prevalence: a higher-prevalence group can post a higher AUC-PR mechanically, not because "
        "the model serves it better. Use the prevalence-robust AUC-ROC for cross-group comparison. "
        "Per-group prevalence is shown for this reason.",
        "",
        "| group | prevalence | train_n | held_out_n | AUC-PR | AUC-ROC | ECE |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, r in sorted(stratified_results.items()):
        lines.append(
            f"| {g} | {r['prevalence']*100:.1f}% | {r['train_n']:,} | {r['held_out_n']:,} | "
            f"{r['auc_pr']:.4f} | {r['auc_roc']:.4f} | {r['ece']:.4f} |"
        )
    lines.append("")

    # Sensitivity section
    if sensitivity_threshold:
        lines += [
            "## Threshold sensitivity",
            "",
            f"Base threshold: {threshold:.4f} (from Week 4).",
            "",
            "| threshold | % of base | sensitivity | specificity | worst EO gap |",
            "|---|---|---|---|---|",
        ]
        for s in sensitivity_threshold:
            lines.append(
                f"| {s['threshold']:.4f} | {s['threshold_pct_of_base']:.0f}% | "
                f"{s['sensitivity']:.4f} | {s['specificity']:.4f} | "
                f"{s['worst_eo_gap']:.4f} |"
            )
        lines.append("")

    if sensitivity_cap:
        lines += [
            "## Training-cap sensitivity",
            "",
            "| train_cap | held-out AUC-PR |",
            "|---|---|",
        ]
        for s in sensitivity_cap:
            lines.append(f"| {s['train_cap']:,} | {s['held_out_auc_pr']:.4f} |")
        lines.append("")

    if SMOKE:
        lines += ["", "**SMOKE artifact: subsampled data. Do not ship.**"]

    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[{time.time() - t0:7.1f}s] Done. Outputs: {JSON_OUT} and {MD_OUT}", flush=True)


if __name__ == "__main__":
    main()
