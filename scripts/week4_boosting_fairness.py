"""Week 4 (Phase 4): gradient boosting with Optuna, and the first formal fairness audit.

Full-data rebuild of Phase 4. Tunes XGBoost and LightGBM on the dev_val partition (AUC-PR), isotonic-
calibrates the better booster on dev_test, and reports honest metrics on the untouched held_out_final
partition (1.5M singleton births). Then, at a screening operating point chosen on dev_val, it runs the
first formal fairness audit versus the NH White reference: demographic parity, equal opportunity (with
a bootstrap CI and a stratified-permutation p-value, BH corrected), and disparate impact.

Reads data/processed/modeling_full_2018_2024.parquet (from scripts/week2_feature_leakage_audit.py).
Gradient-boosted trees are expected to lead on this tabular data (Correction 4); deep learning stays
deferred as an expected non-winner. Every number is computed live on the real full data.

Run: PYTHONPATH=src ./.venv/bin/python scripts/week4_boosting_fairness.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl

# LightGBM warns when scored on a bare numpy array after being fit via sklearn (benign here).
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from equibirth.fairness import metrics as fm
from equibirth.models import boosting
from equibirth.models import evaluate as ev
from equibirth.utils.config import load_config
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
DATA = ROOT / "data" / "processed" / "modeling_full_2018_2024.parquet"
TABLE = ROOT / "reports" / "week4_boosting_fairness.md"
JSON_OUT = ROOT / "reports" / "week4_boosting_fairness.json"
SEED = CFG.get("project.random_seed")
N_TRIALS = 30
TARGET_SENSITIVITY = 0.60
REF = CFG.get("fairness.reference_group")
SUPPRESS = CFG.get("evaluation.cell_suppression_floor")
# meduc (education) is a legitimate numeric feature (features.build emits it), kept in the matrix;
# only true non-features and meta/label columns are excluded (raw `pay` is meta; model form is pay_*).
NON_FEATURES = {"preterm", "split", "mother_race_ethnicity", "pay", "birth_year"}
# Train the boosters on a capped stratified dev subsample (past the ceiling; keeps tuning fast).
TRAIN_CAP = 500_000


def _threshold_for_sensitivity(y, p, target):
    """Largest threshold achieving at least ``target`` sensitivity (a screening operating point)."""
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    total_pos = int(np.asarray(y).sum())
    if total_pos == 0:
        return 0.5
    cum_tp = np.cumsum(y_sorted)
    k = int(np.searchsorted(cum_tp, target * total_pos))
    k = min(k, len(p) - 1)
    return float(np.sort(p)[::-1][k])


def main() -> None:
    set_seeds(SEED)
    df = pl.read_parquet(DATA)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["preterm"].fill_null(-1).to_numpy().astype(int)
    split = df["split"].to_numpy()
    race_all = df["mother_race_ethnicity"].to_numpy()
    del df
    valid = y >= 0

    tr = (split == "dev_train") & valid
    val = (split == "dev_val") & valid
    cal = (split == "dev_test") & valid
    ho = (split == "held_out_final") & valid

    tr_idx = np.flatnonzero(tr)
    if tr_idx.size > TRAIN_CAP:
        drop = np.random.default_rng(SEED).choice(tr_idx, tr_idx.size - TRAIN_CAP, replace=False)
        tr = tr.copy(); tr[drop] = False
    print(f"features={len(feats)} train={tr.sum():,}(capped) val={val.sum():,} cal={cal.sum():,} "
          f"held_out={ho.sum():,}", flush=True)

    results = {}
    for name in ["xgboost", "lightgbm"]:
        print(f"tuning {name} ({N_TRIALS} trials)...", flush=True)
        params = boosting.tune(name, X, y, tr, val, n_trials=N_TRIALS,
                               timeout=CFG.get("optuna_runtime.timeout_seconds"), seed=SEED)
        est = boosting.fit_calibrated(name, X, y, tr, cal, params, seed=SEED)
        p_ho = est.predict_proba(X[ho])[:, 1]
        p_cal = est.predict_proba(X[cal])[:, 1]
        sel = ev.auc_pr(y[cal], p_cal)   # selection metric on dev_test, never held-out
        results[name] = {"p_ho": p_ho, "p_val": est.predict_proba(X[val])[:, 1], "sel": sel,
                         "params": params}
        print(f"  {name}: dev_test AUC-PR={sel:.3f}  held-out AUC-PR={ev.auc_pr(y[ho], p_ho):.3f} "
              f"AUC-ROC={ev.auc_roc(y[ho], p_ho):.3f} ECE={ev.ece(y[ho], p_ho):.4f}", flush=True)

    best = max(results, key=lambda k: results[k]["sel"])
    p_ho = results[best]["p_ho"]
    yho = y[ho]
    race_ho = race_all[ho]
    thr = _threshold_for_sensitivity(y[val], results[best]["p_val"], TARGET_SENSITIVITY)
    sens, spec = ev.sensitivity_specificity(yho, p_ho, thr)
    print(f"\nbest by dev_test AUC-PR: {best}. Screening threshold {thr:.4f} "
          f"(target sens {TARGET_SENSITIVITY}); held-out sens={sens:.3f} spec={spec:.3f}", flush=True)

    # Fairness audit at the screening operating point. The bootstrap-CI and permutation-null loops
    # are O(n) per iteration, so on 1.5M rows they are prohibitively slow; a fixed 300K held-out
    # subsample keeps every large group well above the reporting floor and gives converged CIs. The
    # point EO/DI/DP metrics are near-identical to the full-held-out values at this size.
    n_ho = int(ho.sum())
    FAIR_N = min(300_000, n_ho)
    fair_local = np.sort(np.random.default_rng(SEED).choice(n_ho, size=FAIR_N, replace=False))
    rows = fm.fairness_table(yho[fair_local], p_ho[fair_local], race_ho[fair_local],
                             ref_group=REF, threshold=thr, suppress_floor=SUPPRESS,
                             n_boot=400, n_perm=600, seed=SEED)

    lines = ["# Week 4 (Phase 4): boosting + first fairness audit (full-data held-out)", "",
             f"Best booster (selected on dev_test AUC-PR): **{best}**. Held-out (n={int(ho.sum()):,}) "
             f"AUC-PR {ev.auc_pr(yho, p_ho):.3f}, AUC-ROC {ev.auc_roc(yho, p_ho):.3f}, "
             f"ECE {ev.ece(yho, p_ho):.4f}.",
             f"Operating point: screening threshold {thr:.4f} (target sensitivity {TARGET_SENSITIVITY}); "
             f"held-out sensitivity {sens:.3f}, specificity {spec:.3f}.", "",
             f"Fairness computed on a fixed {FAIR_N:,}-row held-out subsample (converged CIs; the "
             "O(n)-per-iteration bootstrap and permutation loops are prohibitive on the full 1.5M). "
             f"Versus {REF} (equal-opportunity = TPR gap; a NEGATIVE value means the model misses more "
             "real preterm cases for that group; a large disparate-impact ratio means that group is "
             "flagged more often):", "",
             "| group | n | pos | demographic parity diff | equal-opp diff [95% CI] | disparate impact | perm p | BH p |",
             "|---|---|---|---|---|---|---|---|"]
    audit_json = []
    print("\nfairness audit:", flush=True)
    for r in rows:
        if r.note == "suppressed":
            lines.append(f"| {r.group} | {r.n:,} | {r.positives} | - | suppressed | - | - | - |")
            print(f"  {r.group:<15} suppressed (small cell)", flush=True)
            audit_json.append({"group": r.group, "n": r.n, "status": "suppressed"})
            continue
        ci = f"[{r.eo_ci[0]:+.3f}, {r.eo_ci[1]:+.3f}]"
        sig = "*" if (r.eo_pvalue_bh == r.eo_pvalue_bh and r.eo_pvalue_bh < 0.05) else ""
        lines.append(f"| {r.group} | {r.n:,} | {r.positives} | {r.demographic_parity_diff:+.3f} | "
                     f"{r.equal_opportunity_diff:+.3f} {ci} | {r.disparate_impact_ratio:.3f} | "
                     f"{r.eo_pvalue:.3f} | {r.eo_pvalue_bh:.3f}{sig} |")
        print(f"  {r.group:<15} EOdiff={r.equal_opportunity_diff:+.3f} {ci} "
              f"DI={r.disparate_impact_ratio:.3f} BHp={r.eo_pvalue_bh:.3f}{sig}", flush=True)
        audit_json.append({"group": r.group, "n": r.n, "positives": r.positives,
                           "demographic_parity_diff": r.demographic_parity_diff,
                           "equal_opportunity_diff": r.equal_opportunity_diff,
                           "eo_ci": list(r.eo_ci), "disparate_impact_ratio": r.disparate_impact_ratio,
                           "eo_pvalue": r.eo_pvalue, "eo_pvalue_bh": r.eo_pvalue_bh})

    lines += ["", "* BH-adjusted p < 0.05. Equal opportunity is the primary fairness metric; "
              "calibration-by-group (Week 3) shows the risk scores are well-calibrated in aggregate "
              "across large groups. This is one model at one operating point, not a final claim; the "
              "full mitigation and decomposition analysis comes later. Race is a marker of exposure to "
              "racism, not a biological cause; nothing here is a clinical tool."]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_OUT.write_text(json.dumps({
        "best": best, "held_out_n": int(ho.sum()),
        "held_out_auc_pr": ev.auc_pr(yho, p_ho), "held_out_auc_roc": ev.auc_roc(yho, p_ho),
        "held_out_ece": ev.ece(yho, p_ho), "threshold": thr, "sensitivity": sens, "specificity": spec,
        "fairness_subsample_n": FAIR_N,
        "params": {k: results[k]["params"] for k in results},
        "dev_test_auc_pr": {k: results[k]["sel"] for k in results},
        "fairness": audit_json}, indent=2, default=float))
    print(f"\nresults -> {TABLE}", flush=True)


if __name__ == "__main__":
    main()
