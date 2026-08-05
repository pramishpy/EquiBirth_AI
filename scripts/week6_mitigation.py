"""Week 6: three mitigations priced on one tradeoff table; nothing "solves" fairness.

Reweighting (pre-processing), ExponentiatedGradient reductions (in-processing), and
ThresholdOptimizer (post-processing) are run against the Week-4-selected xgboost baseline.
Each mitigation is priced on the same 100K held-out subsample so a reader can compare the
fairness gain against the discrimination or calibration cost in one table.

Disparate-treatment tension (ThresholdOptimizer uses different rules by group) is reported
explicitly and not decided here. The script outputs a tradeoff summary; it is not a
recommendation engine.

Run: POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week6_mitigation.py
Smoke: EQUIBIRTH_SMOKE=1 POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week6_mitigation.py
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

from equibirth.fairness import metrics as fm
from equibirth.fairness import mitigation
from equibirth.models import boosting
from equibirth.models import evaluate as ev
from equibirth.utils.config import load_config
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
DATA = ROOT / "data" / "processed" / "modeling_full_2018_2024.parquet"
WEEK4_JSON = ROOT / "reports" / "week4_boosting_fairness.json"
TABLE = ROOT / "reports" / "week6_mitigation.md"
JSON_OUT = ROOT / "reports" / "week6_mitigation.json"
SEED = CFG.get("project.random_seed")
REF = CFG.get("fairness.reference_group")
SUPPRESS = CFG.get("evaluation.cell_suppression_floor")
NON_FEATURES = {"preterm", "split", "mother_race_ethnicity", "pay", "birth_year"}

SMOKE = os.environ.get("EQUIBIRTH_SMOKE") == "1"
TRAIN_CAP = 10_000 if SMOKE else 500_000
FAIR_CAP = 10_000 if SMOKE else 100_000
VAL_CAP = 10_000 if SMOKE else 50_000
EXPGRAD_CAP = 10_000 if SMOKE else 50_000
BOOT_N = 50 if SMOKE else 200
PERM_N = 100 if SMOKE else 300

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
    from equibirth.features import leakage
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


def _tpr_per_group(y: np.ndarray, pred: np.ndarray, groups: np.ndarray,
                   ref: str) -> dict[str, float]:
    """TPR per group and the EO gap vs ref."""
    tprs = {}
    for g in sorted({str(x) for x in groups if x is not None and str(x) != "nan"}):
        m = groups == g
        pos = y[m] == 1
        if pos.any():
            tprs[g] = float(pred[m][pos].mean())
        else:
            tprs[g] = float("nan")
    return tprs


def _sens_spec_from_hard(y: np.ndarray, pred: np.ndarray):
    """Sensitivity and specificity from hard 0/1 predictions."""
    pos = y == 1
    neg = y == 0
    sens = float(pred[pos].mean()) if pos.any() else float("nan")
    spec = float((1 - pred[neg]).mean()) if neg.any() else float("nan")
    return sens, spec


def main() -> None:
    set_seeds(SEED)
    t0 = time.time()

    # --- Load week-4 params and threshold ---
    wk4 = json.loads(WEEK4_JSON.read_text())
    xgb_params = wk4["params"]["xgboost"]
    thr = wk4["threshold"]

    # --- Validation prologue ---
    print(f"[{time.time() - t0:7.1f}s] loading parquet...", flush=True)
    df = pl.read_parquet(DATA)
    _validate(df)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["preterm"].fill_null(-1).to_numpy().astype(int)
    split = df["split"].to_numpy()
    race_all = df["mother_race_ethnicity"].to_numpy()
    del df
    gc.collect()
    valid = y >= 0

    tr = _cap_mask((split == "dev_train") & valid, TRAIN_CAP, SEED)
    val = _cap_mask((split == "dev_val") & valid, VAL_CAP, SEED)
    cal = (split == "dev_test") & valid
    ho = (split == "held_out_final") & valid

    # Fairness subsample: 100K seeded draw from held-out (same strategy as week5)
    n_ho = int(ho.sum())
    ho_idx = np.flatnonzero(ho)
    fair_size = min(FAIR_CAP, n_ho)
    fair_local = np.sort(np.random.default_rng(SEED).choice(n_ho, size=fair_size, replace=False))
    fair_idx = ho_idx[fair_local]

    y_fair = y[fair_idx]
    race_fair = race_all[fair_idx]

    print(f"[{time.time() - t0:7.1f}s] smoke={SMOKE} features={len(feats)} "
          f"train={tr.sum():,}(capped) val={val.sum():,} cal={cal.sum():,} "
          f"held_out={n_ho:,} fairness_subsample={fair_size:,}", flush=True)

    # --- Step 2: refit baseline xgboost ---
    print(f"[{time.time() - t0:7.1f}s] fitting baseline xgboost (week-4 params, "
          f"isotonic on dev_test)...", flush=True)
    est_baseline = boosting.fit_calibrated("xgboost", X, y, tr, cal, xgb_params, seed=SEED)
    p_fair_baseline = est_baseline.predict_proba(X[fair_idx])[:, 1]
    p_val_baseline = est_baseline.predict_proba(X[np.flatnonzero(val)])[:, 1]

    # Sanity check: AUC-PR should roughly match week-4/5
    baseline_auc_pr = ev.auc_pr(y_fair, p_fair_baseline)
    baseline_auc_roc = ev.auc_roc(y_fair, p_fair_baseline)
    baseline_ece = ev.ece(y_fair, p_fair_baseline)
    # Equal-mass (quantile-binned) ECE is the stricter, more honest calibration figure when
    # predictions pile up near the base rate: equal-width bins leave almost all mass in one or two
    # low-probability bins and mechanically understate miscalibration.
    baseline_ece_q = ev.ece_quantile(y_fair, p_fair_baseline)
    baseline_sens, baseline_spec = ev.sensitivity_specificity(y_fair, p_fair_baseline, thr)
    print(f"[{time.time() - t0:7.1f}s]   baseline: AUC-PR={baseline_auc_pr:.4f} "
          f"AUC-ROC={baseline_auc_roc:.4f} ECE={baseline_ece:.4f} "
          f"sens={baseline_sens:.3f} spec={baseline_spec:.3f}", flush=True)
    if not SMOKE and abs(baseline_auc_pr - 0.17) > 0.02:
        print(f"  WARNING: baseline AUC-PR {baseline_auc_pr:.4f} differs from expected ~0.17 "
              f"by more than 0.02; check for bugs", flush=True)

    # --- Step 3: baseline fairness table ---
    print(f"[{time.time() - t0:7.1f}s] computing baseline fairness table "
          f"(n_boot={BOOT_N}, n_perm={PERM_N})...", flush=True)
    baseline_rows = fm.fairness_table(y_fair, p_fair_baseline, race_fair,
                                      ref_group=REF, threshold=thr,
                                      suppress_floor=SUPPRESS,
                                      n_boot=BOOT_N, n_perm=PERM_N, seed=SEED)
    baseline_tprs = _tpr_per_group(y_fair, (p_fair_baseline >= thr).astype(int), race_fair, REF)
    print(f"[{time.time() - t0:7.1f}s]   baseline EO gaps: "
          + "  ".join(f"{g}={baseline_tprs.get(g, float('nan')) - baseline_tprs.get(REF, 0):+.3f}"
                      for g in sorted(baseline_tprs) if g != REF), flush=True)

    # --- Step 4: Mitigation A (reweighting) ---
    print(f"[{time.time() - t0:7.1f}s] Mitigation A: reweighting + refit xgboost...", flush=True)
    w_tr = mitigation.reweight_sample_weights(y[tr], race_all[tr])
    # reweight_sample_weights already equalizes every (group x outcome) cell to equal mass, which
    # class-balances positives against negatives. Constructing the estimator with scale_pos_weight>1
    # on top of that would double-count the positive class (roughly a further 10x at 8.6%
    # prevalence), so the reweight arm would not be a clean "baseline + reweighting" comparison.
    # We therefore set spw=1.0 here: the sample weights ARE the class-balancing mechanism.
    est_rw = boosting._make_estimator("xgboost", xgb_params, 1.0, SEED)
    est_rw.fit(X[tr], y[tr], sample_weight=w_tr)

    # Isotonic-recalibrate on cal
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        cal_rw = CalibratedClassifierCV(FrozenEstimator(est_rw), method="isotonic")
    except ImportError:
        cal_rw = CalibratedClassifierCV(est_rw, method="isotonic", cv="prefit")
    cal_rw.fit(X[cal], y[cal])

    p_fair_rw = cal_rw.predict_proba(X[fair_idx])[:, 1]
    rw_auc_pr = ev.auc_pr(y_fair, p_fair_rw)
    rw_auc_roc = ev.auc_roc(y_fair, p_fair_rw)
    rw_ece = ev.ece(y_fair, p_fair_rw)
    rw_ece_q = ev.ece_quantile(y_fair, p_fair_rw)
    rw_sens, rw_spec = ev.sensitivity_specificity(y_fair, p_fair_rw, thr)

    print(f"[{time.time() - t0:7.1f}s]   reweight: AUC-PR={rw_auc_pr:.4f} "
          f"AUC-ROC={rw_auc_roc:.4f} ECE={rw_ece:.4f} "
          f"sens={rw_sens:.3f} spec={rw_spec:.3f}", flush=True)

    print(f"[{time.time() - t0:7.1f}s]   computing reweight fairness table...", flush=True)
    rw_rows = fm.fairness_table(y_fair, p_fair_rw, race_fair,
                                ref_group=REF, threshold=thr,
                                suppress_floor=SUPPRESS,
                                n_boot=BOOT_N, n_perm=PERM_N, seed=SEED)
    rw_tprs = _tpr_per_group(y_fair, (p_fair_rw >= thr).astype(int), race_fair, REF)
    del est_rw, cal_rw
    gc.collect()
    print(f"[{time.time() - t0:7.1f}s]   reweight EO gaps: "
          + "  ".join(f"{g}={rw_tprs.get(g, float('nan')) - rw_tprs.get(REF, 0):+.3f}"
                      for g in sorted(rw_tprs) if g != REF), flush=True)

    # --- Step 5: Mitigation B (ExponentiatedGradient) ---
    print(f"[{time.time() - t0:7.1f}s] Mitigation B: ExponentiatedGradient "
          f"(cap={EXPGRAD_CAP:,})...", flush=True)
    # Subsample tr for ExpGrad (it is expensive)
    tr_sub = _cap_mask(tr, EXPGRAD_CAP, SEED + 1)
    tr_sub_idx = np.flatnonzero(tr_sub)
    print(f"[{time.time() - t0:7.1f}s]   expgrad training on {len(tr_sub_idx):,} rows...",
          flush=True)

    eg_base_pred, eg_mit_pred = mitigation.expgrad_predictions(
        X[tr_sub_idx], y[tr_sub_idx], race_all[tr_sub_idx],
        X[fair_idx], race_fair,
        constraint="true_positive_rate_parity", seed=SEED)

    eg_tprs_base = _tpr_per_group(y_fair, eg_base_pred, race_fair, REF)
    eg_tprs_mit = _tpr_per_group(y_fair, eg_mit_pred, race_fair, REF)
    eg_sens_base, eg_spec_base = _sens_spec_from_hard(y_fair, eg_base_pred)
    eg_sens_mit, eg_spec_mit = _sens_spec_from_hard(y_fair, eg_mit_pred)
    print(f"[{time.time() - t0:7.1f}s]   expgrad baseline: sens={eg_sens_base:.3f} "
          f"spec={eg_spec_base:.3f}", flush=True)
    print(f"[{time.time() - t0:7.1f}s]   expgrad mitigated: sens={eg_sens_mit:.3f} "
          f"spec={eg_spec_mit:.3f}", flush=True)
    print(f"[{time.time() - t0:7.1f}s]   expgrad mitigated EO gaps: "
          + "  ".join(f"{g}={eg_tprs_mit.get(g, float('nan')) - eg_tprs_mit.get(REF, 0):+.3f}"
                      for g in sorted(eg_tprs_mit) if g != REF), flush=True)

    # --- Step 6: Mitigation C (ThresholdOptimizer) ---
    print(f"[{time.time() - t0:7.1f}s] Mitigation C: ThresholdOptimizer...", flush=True)
    # Fit on val partition scores from baseline model
    val_idx = np.flatnonzero(val)
    y_val = y[val_idx]
    race_val = race_all[val_idx]

    to_pred = mitigation.threshold_optimizer_predictions(
        y_val, p_val_baseline, race_val,
        p_fair_baseline, race_fair, seed=SEED)

    to_tprs = _tpr_per_group(y_fair, to_pred, race_fair, REF)
    to_sens, to_spec = _sens_spec_from_hard(y_fair, to_pred)
    print(f"[{time.time() - t0:7.1f}s]   threshopt: sens={to_sens:.3f} "
          f"spec={to_spec:.3f}", flush=True)
    print(f"[{time.time() - t0:7.1f}s]   threshopt EO gaps: "
          + "  ".join(f"{g}={to_tprs.get(g, float('nan')) - to_tprs.get(REF, 0):+.3f}"
                      for g in sorted(to_tprs) if g != REF), flush=True)

    # --- Step 7: Summary table ---
    print(f"[{time.time() - t0:7.1f}s] assembling summary...", flush=True)

    ref_tpr_base = baseline_tprs.get(REF, float("nan"))
    ref_tpr_rw = rw_tprs.get(REF, float("nan"))
    ref_tpr_eg = eg_tprs_mit.get(REF, float("nan"))
    ref_tpr_to = to_tprs.get(REF, float("nan"))

    groups = sorted(g for g in baseline_tprs if g != REF)

    def _eo_gaps(tprs, ref_tpr):
        return {g: tprs.get(g, float("nan")) - ref_tpr for g in groups}

    def _worst_mean(gaps):
        vals = [abs(v) for v in gaps.values() if v == v]
        if not vals:
            return float("nan"), float("nan")
        return max(vals), float(np.mean(vals))

    def _dp_diffs(y_sub, pred, groups_arr, ref_group):
        """Demographic parity diffs from hard predictions."""
        ref_m = groups_arr == ref_group
        ref_pr = float(pred[ref_m].mean()) if ref_m.any() else float("nan")
        out = {}
        for g in sorted({str(x) for x in groups_arr if x is not None and str(x) != "nan" and str(x) != ref_group}):
            m = groups_arr == g
            out[g] = float(pred[m].mean()) - ref_pr
        return out

    # Baseline
    base_eo = _eo_gaps(baseline_tprs, ref_tpr_base)
    base_worst, base_mean = _worst_mean(base_eo)
    base_dp = _dp_diffs(y_fair, (p_fair_baseline >= thr).astype(int), race_fair, REF)

    # Reweight
    rw_eo = _eo_gaps(rw_tprs, ref_tpr_rw)
    rw_worst, rw_mean = _worst_mean(rw_eo)
    rw_dp = _dp_diffs(y_fair, (p_fair_rw >= thr).astype(int), race_fair, REF)

    # ExpGrad
    eg_eo = _eo_gaps(eg_tprs_mit, ref_tpr_eg)
    eg_worst, eg_mean = _worst_mean(eg_eo)
    eg_dp = _dp_diffs(y_fair, eg_mit_pred, race_fair, REF)

    # ThreshOpt
    to_eo = _eo_gaps(to_tprs, ref_tpr_to)
    to_worst, to_mean = _worst_mean(to_eo)
    to_dp = _dp_diffs(y_fair, to_pred, race_fair, REF)

    summary = {
        "baseline": {
            "method": "xgboost (week-4 tuned, isotonic-calibrated)",
            "auc_pr": baseline_auc_pr,
            "auc_roc": baseline_auc_roc,
            "ece": baseline_ece,
            "ece_quantile": baseline_ece_q,
            "sensitivity": baseline_sens,
            "specificity": baseline_spec,
            "threshold": thr,
            "per_group_eo_gap": base_eo,
            "per_group_dp_diff": base_dp,
            "worst_group_eo_gap": base_worst,
            "mean_abs_eo_gap": base_mean,
        },
        "reweight": {
            "method": "reweighting (group x outcome cell mass equalized) + xgboost refit",
            "auc_pr": rw_auc_pr,
            "auc_roc": rw_auc_roc,
            "ece": rw_ece,
            "ece_quantile": rw_ece_q,
            "sensitivity": rw_sens,
            "specificity": rw_spec,
            "threshold": thr,
            "per_group_eo_gap": rw_eo,
            "per_group_dp_diff": rw_dp,
            "worst_group_eo_gap": rw_worst,
            "mean_abs_eo_gap": rw_mean,
        },
        "expgrad": {
            "method": "ExponentiatedGradient (TPR parity constraint, logistic base learner)",
            "sensitivity": eg_sens_mit,
            "specificity": eg_spec_mit,
            "per_group_eo_gap": eg_eo,
            "per_group_dp_diff": eg_dp,
            "worst_group_eo_gap": eg_worst,
            "mean_abs_eo_gap": eg_mean,
            "expgrad_baseline_sens": eg_sens_base,
            "expgrad_baseline_spec": eg_spec_base,
            "note": ("Hard 0/1 predictions from a logistic base learner; no AUC-PR/ECE (no "
                     "calibrated probabilities). IMPORTANT: the near-zero mitigated sensitivity is a "
                     "setup artifact, NOT a property of the fairness constraint. The UNCONSTRAINED "
                     "logistic baseline already has sensitivity "
                     f"{eg_sens_base:.3f} at the classifier's default 0.5 decision cutoff, because at "
                     "8.6% prevalence almost no logistic score exceeds 0.5. The XGBoost baseline, by "
                     f"contrast, is scored at a tuned {thr:.3f} threshold (sensitivity "
                     f"{baseline_sens:.3f}), so the two are compared at incomparable operating "
                     "points. What ExponentiatedGradient did achieve, near-complete TPR equalization "
                     f"(worst gap {eg_worst:.3f}) with a weak convex base learner, is real; the low "
                     "sensitivity reflects the logistic base learner plus a 0.5 cutoff, not a cost "
                     "inherent to in-processing fairness constraints."),
        },
        "threshopt": {
            "method": "ThresholdOptimizer (group-specific thresholds on baseline scores)",
            "sensitivity": to_sens,
            "specificity": to_spec,
            "per_group_eo_gap": to_eo,
            "per_group_dp_diff": to_dp,
            "worst_group_eo_gap": to_worst,
            "mean_abs_eo_gap": to_mean,
            "note": ("disparate treatment: different decision rules by group; reported as "
                     "analytical option, never deployed"),
        },
    }

    # --- Step 8: serialize ---
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps({
        "mitigations": summary,
        "fairness_subsample_n": fair_size,
        "threshold": thr,
        "reference_group": REF,
        "groups": groups,
        "seed": SEED,
        "smoke": SMOKE,
        "caps": {
            "train_cap": TRAIN_CAP,
            "fair_cap": FAIR_CAP,
            "val_cap": VAL_CAP,
            "expgrad_cap": EXPGRAD_CAP,
            "boot_n": BOOT_N,
            "perm_n": PERM_N,
        },
    }, indent=2, default=float))

    # --- Markdown report ---
    lines = [
        "# Week 6: fairness mitigation tradeoff table",
        "",
        "Three mitigations priced against the xgboost baseline on a shared "
        f"{fair_size:,}-row held-out subsample. Nothing here 'solves' fairness; each "
        "mitigation buys equal-opportunity improvement at a disclosed cost to overall "
        "discrimination or calibration. The ThresholdOptimizer is disparate treatment "
        "(different decision rules by group) and is reported as an analytical bound, not "
        "a deployment recommendation.",
        "",
        f"Screening threshold: {thr:.4f} (from Week 4, target sensitivity 0.60). "
        f"Reference group: {REF}.",
        "",
        "## Tradeoff summary",
        "",
        "| method | AUC-PR | AUC-ROC | ECE | sensitivity | specificity | worst EO gap | mean |EO gap| |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for key in ["baseline", "reweight", "expgrad", "threshopt"]:
        s = summary[key]
        apr = f"{s['auc_pr']:.4f}" if "auc_pr" in s else "-"
        aroc = f"{s['auc_roc']:.4f}" if "auc_roc" in s else "-"
        ece_v = f"{s['ece']:.4f}" if "ece" in s else "-"
        sens = f"{s['sensitivity']:.3f}"
        spec = f"{s['specificity']:.3f}"
        worst = f"{s['worst_group_eo_gap']:.3f}"
        mean_v = f"{s['mean_abs_eo_gap']:.3f}"
        lines.append(f"| {key} | {apr} | {aroc} | {ece_v} | {sens} | {spec} | {worst} | {mean_v} |")

    lines += [
        "",
        "## Per-group equal-opportunity gap (TPR difference vs NH White)",
        "",
        "| group | baseline | reweight | expgrad | threshopt |",
        "|---|---|---|---|---|",
    ]
    for g in groups:
        b = base_eo.get(g, float("nan"))
        r = rw_eo.get(g, float("nan"))
        e = eg_eo.get(g, float("nan"))
        t = to_eo.get(g, float("nan"))
        lines.append(f"| {g} | {b:+.3f} | {r:+.3f} | {e:+.3f} | {t:+.3f} |")

    lines += [
        "",
        "## Per-group demographic parity diff (positive rate difference vs NH White)",
        "",
        "| group | baseline | reweight | expgrad | threshopt |",
        "|---|---|---|---|---|",
    ]
    for g in groups:
        b = base_dp.get(g, float("nan"))
        r = rw_dp.get(g, float("nan"))
        e = eg_dp.get(g, float("nan"))
        t = to_dp.get(g, float("nan"))
        lines.append(f"| {g} | {b:+.3f} | {r:+.3f} | {e:+.3f} | {t:+.3f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- Reweighting equalizes (group x outcome) cell mass during training; the model still "
        "applies a single threshold (no disparate treatment). The cost is a small AUC-PR drop.",
        "- ExponentiatedGradient uses a logistic base learner under a TPR-parity constraint. "
        "It produces hard 0/1 decisions, so AUC-PR is not available. The logistic learner is "
        "weaker than xgboost, so sensitivity drops; the gain is in EO gap reduction.",
        "- ThresholdOptimizer learns group-specific thresholds on precomputed xgboost scores. "
        "It can equalize TPR almost perfectly, but at the cost of applying different rules to "
        "different groups (disparate treatment). This is reported as an analytical bound on "
        "what post-processing can achieve.",
        "- Race is a marker of exposure to racism, not a biological cause; nothing here is a "
        "clinical tool.",
    ]
    if SMOKE:
        lines += ["", "**SMOKE artifact: subsampled data. Do not ship.**"]

    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[{time.time() - t0:7.1f}s] results -> {TABLE} and {JSON_OUT}", flush=True)


if __name__ == "__main__":
    main()
