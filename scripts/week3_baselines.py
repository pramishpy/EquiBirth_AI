"""Week 3 (Phase 3): calibration-first baselines on the FULL 2018-2024 working dataset.

Trains logistic regression (the calibration reference), a decision tree, and a random forest on the
dev_train partition of the full-data working sample, isotonic-recalibrates each on dev_test, and
reports honest metrics on the untouched held_out_final partition (1.5M singleton births): AUC-PR
(primary) with a bootstrap CI, AUC-ROC, Brier, ECE (raw vs calibrated, uniform and quantile bins),
and MCE with a minimum-bin guard. Then a race-stratified read with tiered suppression, plus a
risk-spread diagnostic that keeps the calibration claim honest (low ECE means aggregate calibration,
not individual risk stratification).

This is the full-data rebuild of Phase 3. It reads data/processed/modeling_full_2018_2024.parquet
(produced by scripts/week2_feature_leakage_audit.py). Every number is computed live on the real data.

Run: PYTHONPATH=src ./.venv/bin/python scripts/week3_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from equibirth.models import baselines
from equibirth.models import evaluate as ev
from equibirth.utils.config import load_config
from equibirth.utils.seeds import set_seeds
from equibirth.viz import eda

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
DATA = ROOT / "data" / "processed" / "modeling_full_2018_2024.parquet"
FIG = ROOT / "reports" / "figures" / "week3"
TABLE = ROOT / "reports" / "week3_baseline_metrics.md"
JSON_OUT = ROOT / "reports" / "week3_baseline_metrics.json"
SEED = CFG.get("project.random_seed")
N_BOOT = CFG.get("evaluation.bootstrap_iters")
SUPPRESS = CFG.get("evaluation.cell_suppression_floor")
FULL_CI = CFG.get("evaluation.full_ci_floor")
RACES = CFG.get("data.race_ethnicity_harmonized_categories")
# meduc (education) is a legitimate numeric feature (features.build emits it), so it is NOT excluded
# here; it stays available by name for any SES-proxy stratified reporting. Only the true non-features
# and the meta/label columns are excluded (raw `pay` is meta; its model form is the pay_* one-hots).
NON_FEATURES = {"preterm", "split", "mother_race_ethnicity", "pay", "birth_year"}


def main() -> None:
    set_seeds(SEED)
    df = pl.read_parquet(DATA)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    # float32 halves the matrix footprint (~2.4GB -> ~1.2GB at 3M x 102) with no metric cost here.
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["preterm"].fill_null(-1).to_numpy().astype(int)
    split = df["split"].to_numpy()
    race_all = df["mother_race_ethnicity"].to_numpy()
    del df  # free the Polars frame before training
    valid = y >= 0

    tr = (split == "dev_train") & valid
    cal = (split == "dev_test") & valid       # calibrate on dev_test (dev_val reserved for tuning)
    ho = (split == "held_out_final") & valid
    # Cap the training rows for the baselines. 500K stratified dev rows is well past the point where
    # these models stop improving (the ceiling is set by the weak signal, not the row count), and it
    # keeps the random forest's memory bounded. Metrics are still reported on the full 1.5M held-out.
    TRAIN_CAP = 500_000
    tr_idx = np.flatnonzero(tr)
    if tr_idx.size > TRAIN_CAP:
        rng = np.random.default_rng(SEED)
        drop = rng.choice(tr_idx, size=tr_idx.size - TRAIN_CAP, replace=False)
        tr = tr.copy()
        tr[drop] = False
    print(f"features={len(feats)}  train={tr.sum():,} (capped)  cal={cal.sum():,}  "
          f"held_out={ho.sum():,}", flush=True)

    print("fitting baselines (single fit -> raw + calibrated)...", flush=True)
    raw_preds, cal_preds = baselines.fit_predict_raw_and_calibrated(X, y, tr, cal, ho, seed=SEED)

    yho = y[ho]
    base = float(yho.mean())
    race_ho = race_all[ho]
    print(f"held-out n={ho.sum():,} prevalence={base:.4f}\n", flush=True)

    # Point metrics use the FULL held-out. Bootstrap CIs are computed on a fixed subsample: each
    # AUC-PR resample on 1.5M rows costs ~0.25s, so 2000 iters x 3 models would be ~25 min, and the CI
    # is already converged by a few hundred iters. A 250K subsample gives an honest (very slightly
    # wider, i.e. conservative) CI at a fraction of the cost. ``p`` is positionally aligned to the
    # held-out rows, so we subsample in held-out-local index space. Deterministic given the seed.
    n_ho = int(ho.sum())
    BOOT_N = min(250_000, n_ho)
    boot_local = np.sort(np.random.default_rng(SEED).choice(n_ho, size=BOOT_N, replace=False))
    yb = yho[boot_local]

    rows_json = {}
    lines = ["# Week 3 (Phase 3) baseline metrics (full-data held-out)", "",
             f"Held-out n={n_ho:,}, prevalence={base:.4f}. AUC-ROC, Brier, and ECE point estimates use "
             f"the full held-out; AUC-PR (point + {N_BOOT}-resample bootstrap CI) uses a fixed "
             f"{BOOT_N:,}-row held-out subsample (each AUC-PR resample on 1.5M rows is costly, and the "
             "estimate is converged well before this size). Primary metric AUC-PR. Calibration trained "
             "on dev_test; raw vs calibrated ECE shown. MCE uses a 30-row minimum bin. Quantile-bin ECE "
             "(equal mass per bin) is the stricter calibration summary.", "",
             "| model | AUC-PR [95% CI] | AUC-ROC | Brier | ECE raw | ECE cal | ECE quantile | MCE |",
             "|---|---|---|---|---|---|---|---|"]
    for name in cal_preds:
        p, praw = cal_preds[name], raw_preds[name]
        # AUC-PR point + CI both from the bootstrap subsample so the point sits inside its own CI;
        # AUC-ROC/Brier/ECE point estimates use the full held-out (they are cheap on 1.5M rows).
        pr, lo, hi = ev.bootstrap_ci(ev.auc_pr, yb, p[boot_local], n=N_BOOT, seed=SEED)
        roc, b = ev.auc_roc(yho, p), ev.brier(yho, p)
        e_raw, e_cal = ev.ece(yho, praw), ev.ece(yho, p)
        e_q, m = ev.ece_quantile(yho, p), ev.mce(yho, p)
        print(f"{name:<20} AUC-PR={pr:.3f} [{lo:.3f},{hi:.3f}]  AUC-ROC={roc:.3f}  "
              f"ECE raw={e_raw:.4f}->cal={e_cal:.4f}  ECEq={e_q:.4f}  MCE={m:.3f}", flush=True)
        lines.append(f"| {name} | {pr:.3f} [{lo:.3f}, {hi:.3f}] | {roc:.3f} | {b:.4f} | "
                     f"{e_raw:.4f} | {e_cal:.4f} | {e_q:.4f} | {m:.3f} |")
        rows_json[name] = {"auc_pr": pr, "auc_pr_ci": [lo, hi], "auc_roc": roc, "brier": b,
                           "ece_raw": e_raw, "ece_cal": e_cal, "ece_quantile": e_q, "mce": m}

    near = ev.fraction_near_base_rate(cal_preds["random_forest"], base)
    # Overall random-forest reliability curve, serialized as non-empty bins so the notebook can render
    # the calibration diagram inline from the JSON (reports/figures/ is gitignored and not distributed).
    mp_rf, obs_rf, cnt_rf = ev.calibration_curve(yho, cal_preds["random_forest"])
    reliability_rf = [{"mean_pred": float(a), "observed": float(b), "count": int(c)}
                      for a, b, c in zip(mp_rf, obs_rf, cnt_rf) if c > 0]
    lines += ["", f"Risk-spread diagnostic: {near*100:.0f}% of random-forest predictions fall within "
              f"+/-5pp of the {base*100:.1f}% base rate, so low ECE reflects aggregate calibration, "
              "not individual risk stratification.", ""]

    ref = "logistic_regression"
    pref = cal_preds[ref]
    lines += [f"## Race-stratified ({ref}, held-out)", "",
              "| group | n | positives | rate | AUC-ROC | AUC-PR | ECE | status |",
              "|---|---|---|---|---|---|---|---|"]
    print(f"\nrace-stratified ({ref}):", flush=True)
    strat_json = []
    for r in ev.stratified_report(yho, pref, race_ho, suppress_floor=SUPPRESS, full_ci_floor=FULL_CI):
        if r.status == "suppressed":
            lines.append(f"| {r.group} | {r.n:,} | {r.positives} | {r.rate:.3f} | - | - | - | suppressed |")
            print(f"  {r.group:<15} n={r.n:>8,} pos={r.positives:>6}  suppressed", flush=True)
        else:
            lines.append(f"| {r.group} | {r.n:,} | {r.positives} | {r.rate:.3f} | {r.auc_roc:.3f} | "
                         f"{r.auc_pr:.3f} | {r.ece:.4f} | {r.status} |")
            print(f"  {r.group:<15} n={r.n:>8,} pos={r.positives:>6} rate={r.rate:.3f} "
                  f"AUC-ROC={r.auc_roc:.3f} ECE={r.ece:.4f} [{r.status}]", flush=True)
        strat_json.append({"group": r.group, "n": r.n, "positives": r.positives, "rate": r.rate,
                           "auc_roc": r.auc_roc, "auc_pr": r.auc_pr, "ece": r.ece, "status": r.status})

    # reliability curves overall + by race
    eda.plot_reliability(
        {n: ev.calibration_curve(yho, p) for n, p in cal_preds.items()},
        FIG / "reliability_overall.png", title="Calibration: baseline models (full-data held-out)")
    by_race = {}
    for g in RACES:
        m = race_ho == g
        if m.sum() > FULL_CI and len(np.unique(yho[m])) > 1:
            by_race[g] = ev.calibration_curve(yho[m], pref[m])
    eda.plot_reliability(by_race, FIG / "reliability_by_race.png",
                         title=f"Calibration by race/ethnicity ({ref}, full-data held-out)")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_OUT.write_text(json.dumps(
        {"held_out_n": int(ho.sum()), "prevalence": base, "n_features": len(feats),
         "models": rows_json, "fraction_near_base_rate": near, "stratified": strat_json,
         "reliability_rf": reliability_rf}, indent=2))
    print(f"\nresults -> {TABLE}\nfigures -> {FIG}", flush=True)


if __name__ == "__main__":
    main()
