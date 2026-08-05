"""Week 7: global SHAP for the selected model, race-stratified SHAP, and gap decomposition.

TreeSHAP runs on the RAW (uncalibrated) xgboost booster because SHAP's TreeExplainer requires
direct access to the tree structure. The isotonic-calibration wrapper (used for scoring in week 5)
is a monotone post-hoc transform that does not change feature attribution rankings, so explaining
the raw model is both correct and standard. The background sample is a disclosed fixed 30K
stratified held-out subsample; the reported importances are associational, not causal.

Race-stratified SHAP reveals which features dominate for each group. If risk-factor distributions
differ (e.g. prior preterm birth dominating for NH Black mothers while access factors dominate for
others), that is itself a finding about where risk concentrates.

The Fairlie-style nonlinear decomposition quantifies what fraction of the observed NH Black minus
NH White preterm gap is statistically associated with measured covariates. THREE covariate sets are
run: full model features (minus race/interaction columns), SES-only, and medical-only, producing a
RANGE of explained shares. Bootstrap (100 resamples) gives a confidence interval on the full-set
estimate.

MANDATORY CAVEAT (must accompany any use of the decomposition): the measured covariates (education,
payment, WIC, medical risk factors) are themselves products of structural racism, so the
"explained" share is inflated and the "unexplained" residual is a LOWER BOUND on structural
contributions. This decomposition quantifies statistical association, not causal mechanism.

Run: POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week7_explain_decompose.py
Smoke: EQUIBIRTH_SMOKE=1 POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week7_explain_decompose.py
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

from equibirth.explain.shap_analysis import (
    mean_abs_importance,
    shap_values_for,
    top_features_by_group,
)
from equibirth.fairness.decomposition import decompose_gap
from equibirth.features import leakage
from equibirth.models import boosting
from equibirth.utils.config import load_config
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
DATA = ROOT / "data" / "processed" / "modeling_full_2018_2024.parquet"
WEEK4_JSON = ROOT / "reports" / "week4_boosting_fairness.json"
JSON_OUT = ROOT / "reports" / "week7_explain.json"
MD_OUT = ROOT / "reports" / "week7_explain.md"
SEED = CFG.get("project.random_seed")
NON_FEATURES = {"preterm", "split", "mother_race_ethnicity", "pay", "birth_year"}

SMOKE = os.environ.get("EQUIBIRTH_SMOKE") == "1"
TRAIN_CAP = 50_000 if SMOKE else 500_000
SHAP_N = 5_000 if SMOKE else 30_000
DEV_N = 20_000 if SMOKE else 200_000
BOOT_N = 20 if SMOKE else 100

EXPECTED_SPLITS = {"dev_train": 1_050_000, "dev_val": 225_000,
                   "dev_test": 225_000, "held_out_final": 1_500_000}

RACE_GROUPS = ["NH White", "NH Black", "Hispanic", "NH Asian", "NH AIAN",
               "NH NHOPI", "NH Multiracial"]


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
    """Seeded subsample of a boolean mask down to ``cap`` true rows."""
    idx = np.flatnonzero(mask)
    if cap is None or idx.size <= cap:
        return mask
    drop = np.random.default_rng(seed).choice(idx, idx.size - cap, replace=False)
    mask = mask.copy()
    mask[drop] = False
    return mask


def _stratified_subsample(y: np.ndarray, base_mask: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Draw a stratified subsample of size n from rows where base_mask is True."""
    idx = np.flatnonzero(base_mask)
    if idx.size <= n:
        return base_mask
    rng = np.random.default_rng(seed)
    yy = y[idx]
    pos_idx = idx[yy == 1]
    neg_idx = idx[yy == 0]
    # maintain prevalence
    n_pos = max(1, int(round(n * len(pos_idx) / len(idx))))
    n_neg = n - n_pos
    chosen_pos = rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)
    out = np.zeros(len(y), dtype=bool)
    out[chosen_pos] = True
    out[chosen_neg] = True
    return out


def main() -> None:
    set_seeds(SEED)
    t0 = time.time()

    # --- 1. Validation prologue ---
    df = pl.read_parquet(DATA)
    _validate(df)
    print(f"[{time.time() - t0:7.1f}s] Loaded parquet: {len(df):,} rows, validated.", flush=True)

    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["preterm"].fill_null(-1).to_numpy().astype(int)
    split = df["split"].to_numpy()
    race = df["mother_race_ethnicity"].to_numpy()
    del df
    gc.collect()
    valid = y >= 0

    # --- 2. Refit the RAW (uncalibrated) xgboost ---
    tr = _cap_mask((split == "dev_train") & valid, TRAIN_CAP, SEED)
    wk4 = json.loads(WEEK4_JSON.read_text())
    xgb_params = wk4["params"]["xgboost"]
    spw = boosting.scale_pos_weight(y[tr])
    print(f"[{time.time() - t0:7.1f}s] Fitting raw xgboost (train={tr.sum():,}, "
          f"spw={spw:.2f})...", flush=True)
    raw_model = boosting._make_estimator("xgboost", xgb_params, spw, SEED)
    raw_model.fit(X[tr], y[tr])
    print(f"[{time.time() - t0:7.1f}s] Raw xgboost fitted.", flush=True)

    # --- 3. Global SHAP ---
    # Stratified subsample from dev_val + dev_test (held-out from the training rows)
    shap_base = ((split == "dev_val") | (split == "dev_test")) & valid
    shap_mask = _stratified_subsample(y, shap_base, SHAP_N, SEED)
    shap_idx = np.flatnonzero(shap_mask)
    print(f"[{time.time() - t0:7.1f}s] Computing SHAP on {len(shap_idx):,} rows...", flush=True)
    shap_vals = shap_values_for(raw_model, X[shap_idx])
    global_imp = mean_abs_importance(shap_vals, feats)
    global_top_25 = global_imp[:25]
    print(f"[{time.time() - t0:7.1f}s] Global SHAP top-5: "
          + ", ".join(f"{n} ({v:.4f})" for n, v in global_top_25[:5]), flush=True)

    # --- 4. Race-stratified SHAP ---
    race_shap = race[shap_idx]
    groups_present = [g for g in RACE_GROUPS if (race_shap == g).sum() > 0]
    per_group = top_features_by_group(shap_vals, feats, race_shap,
                                      which=groups_present, top_n=10)
    # Leakage check: every reported feature must be in the feature list
    all_reported = set()
    for g, items in per_group.items():
        for fname, _ in items:
            all_reported.add(fname)
    leakage_check_pass = all_reported.issubset(set(feats))
    leakage_msg = (f"PASS: all {len(all_reported)} reported features are in the "
                   f"{len(feats)}-feature modeling set"
                   if leakage_check_pass else
                   f"FAIL: features not in modeling set: {all_reported - set(feats)}")
    print(f"  Leakage check: {leakage_msg}", flush=True)
    if not leakage_check_pass:
        raise SystemExit("Leakage check failed in race-stratified SHAP.")

    # --- 5. Dependence data for top 4 global features ---
    top4_names = [n for n, _ in global_top_25[:4]]
    dependence_data = []
    for fname in top4_names:
        fi = feats.index(fname)
        fv = X[shap_idx, fi]
        sv = shap_vals[:, fi]
        # 10 decile bins
        percentiles = np.nanpercentile(fv, np.linspace(0, 100, 11))
        bins = []
        for i in range(10):
            lo, hi = percentiles[i], percentiles[i + 1]
            if i < 9:
                mask_bin = (fv >= lo) & (fv < hi)
            else:
                mask_bin = (fv >= lo) & (fv <= hi)
            if mask_bin.sum() == 0:
                continue
            center = float(np.nanmean(fv[mask_bin]))
            mean_shap = float(np.mean(sv[mask_bin]))
            bins.append((center, mean_shap))
        dependence_data.append((fname, bins))
    print(f"[{time.time() - t0:7.1f}s] Dependence data computed for: {top4_names}", flush=True)

    # --- 6. Decomposition ---
    # Use dev_train + dev_val rows for decomposition (inference, not evaluation)
    dev_mask = ((split == "dev_train") | (split == "dev_val")) & valid
    dev_sub = _stratified_subsample(y, dev_mask, DEV_N, SEED)
    dev_idx = np.flatnonzero(dev_sub)

    # Covariate matrix: modeling features MINUS race one-hots and race-interaction columns
    cov_cols = [c for c in feats
                if not c.startswith("mother_race_ethnicity_")
                and not c.startswith("race_education_")]
    cov_col_idx = [feats.index(c) for c in cov_cols]
    X_cov_full = X[dev_idx][:, cov_col_idx]
    y_dev = y[dev_idx]
    race_dev = race[dev_idx]

    is_focal = race_dev == "NH Black"
    is_ref = race_dev == "NH White"

    print(f"[{time.time() - t0:7.1f}s] Decomposition on {len(dev_idx):,} rows "
          f"(focal={is_focal.sum():,}, ref={is_ref.sum():,})...", flush=True)

    # (a) Full covariate set
    dec_full = decompose_gap(X_cov_full, y_dev, is_focal, is_ref, seed=SEED)
    print(f"  Full: raw_gap={dec_full.raw_gap:.5f}, explained_pct={dec_full.explained_pct:.3f}",
          flush=True)

    # (b) SES-only: columns containing "meduc" or "pay_" or "wic"
    ses_idx = [i for i, c in enumerate(cov_cols)
               if "meduc" in c or "pay_" in c or "wic" in c]
    X_ses = X_cov_full[:, ses_idx]
    dec_ses = decompose_gap(X_ses, y_dev, is_focal, is_ref, seed=SEED)
    print(f"  SES-only ({len(ses_idx)} cols): explained_pct={dec_ses.explained_pct:.3f}",
          flush=True)

    # (c) Medical-only: columns containing "rf_" or "medical_risk" or "cig_" or "bmi"
    med_idx = [i for i, c in enumerate(cov_cols)
               if "rf_" in c or "medical_risk" in c or "cig_" in c or "bmi" in c]
    X_med = X_cov_full[:, med_idx]
    dec_med = decompose_gap(X_med, y_dev, is_focal, is_ref, seed=SEED)
    print(f"  Medical-only ({len(med_idx)} cols): explained_pct={dec_med.explained_pct:.3f}",
          flush=True)

    range_low = min(dec_full.explained_pct, dec_ses.explained_pct, dec_med.explained_pct)
    range_high = max(dec_full.explained_pct, dec_ses.explained_pct, dec_med.explained_pct)
    print(f"  Explained range: [{range_low:.3f}, {range_high:.3f}]", flush=True)

    # Bootstrap CI on full-set explained_pct
    print(f"[{time.time() - t0:7.1f}s] Bootstrapping full-set explained_pct "
          f"({BOOT_N} resamples)...", flush=True)
    rng = np.random.default_rng(SEED + 7)
    boot_explained = []
    combined_mask = is_focal | is_ref
    combined_idx = np.flatnonzero(combined_mask)
    for _ in range(BOOT_N):
        resample = rng.choice(combined_idx, len(combined_idx), replace=True)
        X_boot = X_cov_full[resample]
        y_boot = y_dev[resample]
        focal_boot = is_focal[resample]
        ref_boot = is_ref[resample]
        try:
            d = decompose_gap(X_boot, y_boot, focal_boot, ref_boot, seed=SEED)
            boot_explained.append(d.explained_pct)
        except Exception:
            continue
    boot_arr = np.array(boot_explained)
    ci_lo = float(np.percentile(boot_arr, 2.5))
    ci_hi = float(np.percentile(boot_arr, 97.5))
    print(f"  Bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}] (n={len(boot_explained)} valid)",
          flush=True)

    # --- Sanity checks (real runs only) ---
    if not SMOKE:
        if not 0.03 <= dec_full.raw_gap <= 0.07:
            raise SystemExit(f"Sanity: raw_gap={dec_full.raw_gap:.5f} outside [0.03, 0.07]. "
                             "Check focal/ref masks.")
        if not 0.3 <= dec_full.explained_pct <= 0.8:
            raise SystemExit(f"Sanity: explained_pct={dec_full.explained_pct:.3f} outside "
                             "[0.3, 0.8]. Literature range for admin-data decompositions.")
        expected_top = {"rf_ppterm", "medical_risk_count", "mager", "cig_0", "rf_phype"}
        actual_top_10 = {n for n, _ in global_top_25[:10]}
        overlap = expected_top & actual_top_10
        if len(overlap) < 3:
            raise SystemExit(f"Sanity: only {len(overlap)} of expected top features in top-10: "
                             f"{overlap}. Expected at least 3 of {expected_top}.")

    # --- 7. Serialize ---
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "global_top_25": global_top_25,
        "per_group_top_10": {g: items for g, items in per_group.items()},
        "dependence_data": dependence_data,
        "decomposition": {
            "raw_gap": dec_full.raw_gap,
            "adjusted_gap": dec_full.adjusted_gap,
            "explained_pct_full": dec_full.explained_pct,
            "explained_pct_full_ci": [ci_lo, ci_hi],
            "explained_pct_ses": dec_ses.explained_pct,
            "explained_pct_medical": dec_med.explained_pct,
            "range_low": range_low,
            "range_high": range_high,
            "n_focal": dec_full.n_focal,
            "n_ref": dec_full.n_ref,
        },
        "n_shap": int(len(shap_idx)),
        "leakage_check": leakage_msg,
        "caps": {
            "train_cap": TRAIN_CAP,
            "shap_n": SHAP_N,
            "dev_n": DEV_N,
            "boot_n": BOOT_N,
        },
        "seed": SEED,
        "smoke": SMOKE,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, default=float))

    # Markdown report
    lines = [
        "# Week 7: SHAP explainability and Black-White gap decomposition",
        "",
        "## Global SHAP (TreeSHAP on raw xgboost, n={:,})".format(len(shap_idx)),
        "",
        "| rank | feature | mean |SHAP| |",
        "|---|---|---|",
    ]
    for i, (fname, val) in enumerate(global_top_25, 1):
        lines.append(f"| {i} | {fname} | {val:.5f} |")
    lines += [
        "",
        "## Race-stratified top-10 features",
        "",
    ]
    for g in groups_present:
        if g not in per_group:
            continue
        lines.append(f"### {g}")
        lines.append("")
        lines.append("| rank | feature | mean |SHAP| |")
        lines.append("|---|---|---|")
        for i, (fname, val) in enumerate(per_group[g], 1):
            lines.append(f"| {i} | {fname} | {val:.5f} |")
        lines.append("")

    lines += [
        "## NH Black minus NH White gap decomposition",
        "",
        f"- Raw gap: {dec_full.raw_gap:.5f} "
        f"(NH Black {dec_full.raw_gap + dec_full.adjusted_gap:.5f} rate difference observed)",
        f"- Adjusted (unexplained) gap: {dec_full.adjusted_gap:.5f}",
        f"- Explained share (full covariates): {dec_full.explained_pct:.1%} "
        f"[95% CI: {ci_lo:.1%}, {ci_hi:.1%}]",
        f"- Explained share (SES-only: education, payment, WIC): {dec_ses.explained_pct:.1%}",
        f"- Explained share (medical-only: risk factors, smoking, BMI): "
        f"{dec_med.explained_pct:.1%}",
        f"- Range across covariate sets: [{range_low:.1%}, {range_high:.1%}]",
        f"- n focal (NH Black): {dec_full.n_focal:,}",
        f"- n reference (NH White): {dec_full.n_ref:,}",
        "",
        "## Structural-racism caveat",
        "",
        "The measured covariates (maternal education, payment source, WIC participation, "
        "medical risk factors, smoking, BMI) are themselves products of structural racism. "
        "Residential segregation restricts access to quality prenatal care; wealth gaps rooted "
        "in historical exclusion shape education and insurance; chronic stress from experienced "
        "racism elevates hypertensive risk. Consequently, the 'explained' share of the gap "
        "overstates genuine covariate-mediated pathways and the 'unexplained' residual is a "
        "LOWER BOUND on structural contributions. This decomposition quantifies statistical "
        "association between measured SES/medical factors and the preterm disparity. It does "
        "not prove mechanism, does not exonerate the system where covariates 'explain' the gap, "
        "and must not be read as evidence that individual behavior accounts for the disparity.",
        "",
    ]
    if SMOKE:
        lines += ["**SMOKE artifact: subsampled data. Do not ship.**", ""]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[{time.time() - t0:7.1f}s] Results written to {JSON_OUT} and {MD_OUT}", flush=True)


if __name__ == "__main__":
    main()
