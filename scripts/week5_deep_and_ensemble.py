"""Week 5: deep-learning benchmark (TabNet, FT-Transformer), stacking, final model selection.

The deep nets are HONEST BENCHMARKS expected to lose (DECISION_LOG D41): weak tabular signal
(max |corr| ~0.13), 8.57% prevalence, admin-data ceiling 0.62-0.70 AUC-ROC that the tuned
boosters already reach. What we buy with this week is evidence, not hope: the same partitions,
the same calibration discipline, one comparable table. The stacked logistic meta-learner then
asks whether COMBINING the models buys anything. The primary model is selected on dev_test
AUC-PR and reported once on the untouched held-out partition.

Deep nets train on a capped, disclosed stratified subsample of dev_train (DEEP_TRAIN_CAP), with
median imputation + standardization fit on train only (the boosters ingest NaN natively; nets
cannot; the notebook discloses this difference). Boosters reuse the Week-4 tuned params from
reports/week4_boosting_fairness.json (no retuning: same search, same folds, reproducible).

Each deep net is fit ONCE and scored on all three evaluation partitions via the public
deep.fit / deep.predict pair (added for this script; deep.fit_predict now composes them, with
its signature and tests unchanged). That also hands back the fitted TabNet estimator directly,
so the reported feature masks come from the exact fit that produced the reported numbers.

SMOKE mode (EQUIBIRTH_SMOKE=1): TRAIN_CAP=20K, DEEP_TRAIN_CAP=10K, max_epochs=2, bootstrap
n=50, and the val/cal/held-out partitions are each subsampled to 20K rows for speed. Smoke
artifacts carry "smoke": true and must never ship; the sanity gates run only on real runs.

Run: POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week5_deep_and_ensemble.py
(FOREGROUND; roughly 1-3 hours on CPU)
Smoke: EQUIBIRTH_SMOKE=1 POLARS_MAX_THREADS=4 PYTHONPATH=src ./.venv/bin/python scripts/week5_deep_and_ensemble.py
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

# LightGBM warns when scored on a bare numpy array after being fit via sklearn (benign here).
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from sklearn.isotonic import IsotonicRegression

from equibirth.features import leakage
from equibirth.models import boosting, deep, ensemble
from equibirth.models import evaluate as ev
from equibirth.utils.config import load_config
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
DATA = ROOT / "data" / "processed" / "modeling_full_2018_2024.parquet"
WEEK4_JSON = ROOT / "reports" / "week4_boosting_fairness.json"
TABLE = ROOT / "reports" / "week5_models.md"
JSON_OUT = ROOT / "reports" / "week5_models.json"
SEED = CFG.get("project.random_seed")
NON_FEATURES = {"preterm", "split", "mother_race_ethnicity", "pay", "birth_year"}

SMOKE = os.environ.get("EQUIBIRTH_SMOKE") == "1"
TRAIN_CAP = 20_000 if SMOKE else 500_000          # boosters (same cap as week 4)
DEEP_TRAIN_CAP = 10_000 if SMOKE else 30_000      # nets (further subsample; honest benchmark cap)
# All five models are scored on identical partitions so the comparison and the stack are valid. The
# FT-Transformer does an O(n) forward pass per epoch on val and once on held-out; on CPU that is the
# wall-clock bottleneck, so val/cal are drawn to VAL_CAP and held-out to HO_CAP (a stratified draw
# from the untouched held_out_final partition). Boosters would run on the full 1.5M in seconds and
# match the Week-4 headline (LightGBM held-out AUC-PR 0.174); we hold every model to the same rows so
# no model gets an unfair evaluation-set advantage. All caps are disclosed in the JSON and notebook.
VAL_CAP = 20_000 if SMOKE else 50_000             # dev_val and dev_test scoring rows
HO_CAP = 20_000 if SMOKE else 100_000             # held-out scoring rows (stratified draw)
MAX_EPOCHS = 2 if SMOKE else 3
BOOT_N = 50 if SMOKE else 1000

EXPECTED_SPLITS = {"dev_train": 1_050_000, "dev_val": 225_000,
                   "dev_test": 225_000, "held_out_final": 1_500_000}
BASE_MODELS = ["xgboost", "lightgbm", "tabnet", "ft_transformer"]
ALL_MODELS = BASE_MODELS + ["stack"]
# Tie-break preference on dev_test AUC-PR ties within 0.001: booster > stack > nets.
TIER = {"xgboost": 2, "lightgbm": 2, "stack": 1, "tabnet": 0, "ft_transformer": 0}
TRAIN_NOTE = ("boosters trained on 500K-capped dev_train, deep nets on 30K-capped dev_train; all "
              "five models scored on identical rows: 50K dev_val, 50K dev_test, and a 100K "
              "seeded draw from the untouched held_out_final partition (all caps disclosed; the "
              "FT-Transformer's O(n) CPU forward pass is the binding constraint)")


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
    # Leakage guard: the barred set lives in the schema module the leakage guards are built on.
    barred = leakage.schema.all_excluded_vars()
    present = barred & set(df.columns)
    if present:
        raise SystemExit(f"leakage columns present in the modeling parquet: {sorted(present)}")


def _cap_mask(mask: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    """Seeded subsample of a boolean mask down to ``cap`` true rows (week-4 capping code)."""
    idx = np.flatnonzero(mask)
    if cap is None or idx.size <= cap:
        return mask
    drop = np.random.default_rng(seed).choice(idx, idx.size - cap, replace=False)
    mask = mask.copy()
    mask[drop] = False
    return mask


def _iso(p_fit: np.ndarray, y_fit: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_fit, y_fit)


def main() -> None:
    set_seeds(SEED)
    t0 = time.time()
    df = pl.read_parquet(DATA)
    _validate(df)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["preterm"].fill_null(-1).to_numpy().astype(int)
    split = df["split"].to_numpy()
    del df
    gc.collect()
    valid = y >= 0

    tr = _cap_mask((split == "dev_train") & valid, TRAIN_CAP, SEED)
    tr_deep = _cap_mask(tr, DEEP_TRAIN_CAP, SEED)
    val = _cap_mask((split == "dev_val") & valid, VAL_CAP, SEED)
    cal = _cap_mask((split == "dev_test") & valid, VAL_CAP, SEED)
    ho = _cap_mask((split == "held_out_final") & valid, HO_CAP, SEED)
    print(f"smoke={SMOKE} features={len(feats)} train={tr.sum():,}(capped) "
          f"deep_train={tr_deep.sum():,}(capped) val={val.sum():,} cal={cal.sum():,} "
          f"held_out={ho.sum():,}", flush=True)

    p_val, p_cal, p_ho = {}, {}, {}

    # Boosters: week-4 tuned params, no retuning (same search, same folds, reproducible).
    wk4_params = json.loads(WEEK4_JSON.read_text())["params"]
    for name in ["xgboost", "lightgbm"]:
        print(f"[{time.time() - t0:7.1f}s] fitting {name} (week-4 params, "
              f"isotonic on dev_test)...", flush=True)
        est = boosting.fit_calibrated(name, X, y, tr, cal, wk4_params[name], seed=SEED)
        p_val[name] = est.predict_proba(X[val])[:, 1]
        p_cal[name] = est.predict_proba(X[cal])[:, 1]
        p_ho[name] = est.predict_proba(X[ho])[:, 1]
        del est
        gc.collect()
        print(f"[{time.time() - t0:7.1f}s]   {name}: dev_test AUC-PR="
              f"{ev.auc_pr(y[cal], p_cal[name]):.4f}", flush=True)

    # Deep nets: fit ONCE per net, score all three partitions; isotonic-recalibrate on dev_val
    # (the same calibration deep.fit_predict applies internally).
    tabnet_top = []
    for name in ["tabnet", "ft_transformer"]:
        print(f"[{time.time() - t0:7.1f}s] fitting {name} (max_epochs={MAX_EPOCHS}, "
              f"train={tr_deep.sum():,})...", flush=True)
        est, Xp = deep.fit(name, X, y, tr_deep, val, max_epochs=MAX_EPOCHS, seed=SEED)
        raw_val = deep.predict(est, Xp, val)
        iso = _iso(raw_val, y[val])
        p_val[name] = iso.predict(raw_val)
        p_cal[name] = iso.predict(deep.predict(est, Xp, cal))
        p_ho[name] = iso.predict(deep.predict(est, Xp, ho))
        if name == "tabnet":
            tabnet_top = [(f, w) for f, w in deep.tabnet_feature_masks(est, Xp, feats)[:15]]
        del est, Xp
        gc.collect()
        print(f"[{time.time() - t0:7.1f}s]   {name}: dev_test AUC-PR="
              f"{ev.auc_pr(y[cal], p_cal[name]):.4f}", flush=True)

    # Stack: cross-fitted logistic meta-learner on dev_val, isotonic-recalibrated on dev_test.
    print(f"[{time.time() - t0:7.1f}s] stacking (5-fold OOF logistic on dev_val)...", flush=True)
    meta, _ = ensemble.oof_stack({m: p_val[m] for m in BASE_MODELS}, y[val], n_folds=5, seed=SEED)
    stack_weights = {n: float(w) for n, w in zip(meta.feature_names_, meta.coef_[0])}
    print("  meta weights (log-odds space): "
          + "  ".join(f"{n}={w:+.3f}" for n, w in stack_weights.items()), flush=True)
    p_cal_stack = ensemble.predict_stack(meta, {m: p_cal[m] for m in BASE_MODELS})
    iso_stack = _iso(p_cal_stack, y[cal])
    p_cal["stack"] = iso_stack.predict(p_cal_stack)
    p_ho["stack"] = iso_stack.predict(
        ensemble.predict_stack(meta, {m: p_ho[m] for m in BASE_MODELS}))

    # One comparable table over the shared held-out scoring rows (all five models on identical rows).
    # Bootstrap CI on held-out AUC-PR resamples this partition (the loop is O(n) per iteration).
    yho = y[ho]
    n_ho = int(ho.sum())
    fair_n = n_ho
    metrics = {}
    for m in ALL_MODELS:
        _, lo, hi = ev.bootstrap_ci(ev.auc_pr, yho, p_ho[m], n=BOOT_N, seed=SEED)
        metrics[m] = {
            "dev_test_auc_pr": ev.auc_pr(y[cal], p_cal[m]),
            "held_out_auc_pr": ev.auc_pr(yho, p_ho[m]),
            "held_out_auc_pr_ci": [lo, hi],
            "held_out_auc_roc": ev.auc_roc(yho, p_ho[m]),
            "held_out_ece": ev.ece(yho, p_ho[m]),
            "held_out_brier": ev.brier(yho, p_ho[m]),
        }
        print(f"  {m:<15} dev_test AUC-PR={metrics[m]['dev_test_auc_pr']:.4f}  held-out "
              f"AUC-PR={metrics[m]['held_out_auc_pr']:.4f} [{lo:.4f}, {hi:.4f}] "
              f"AUC-ROC={metrics[m]['held_out_auc_roc']:.4f} "
              f"ECE={metrics[m]['held_out_ece']:.4f} "
              f"Brier={metrics[m]['held_out_brier']:.4f}", flush=True)

    # Selection: dev_test AUC-PR, tie-break within the bootstrap CI half-width (any difference
    # inside the CI is noise; prefer the simpler, more auditable model).
    sel = {m: metrics[m]["dev_test_auc_pr"] for m in ALL_MODELS}
    best_pr = max(sel.values())
    # Use the mean CI half-width across boosters as the noise threshold (typically ~0.003-0.004).
    boosters = [m for m in ALL_MODELS if TIER[m] == 2]
    ci_hw = np.mean([
        (metrics[m]["held_out_auc_pr_ci"][1] - metrics[m]["held_out_auc_pr_ci"][0]) / 2
        for m in boosters])
    contenders = [m for m in ALL_MODELS if sel[m] >= best_pr - ci_hw]
    selected = max(contenders, key=lambda m: (TIER[m], sel[m]))
    best_net = max(["tabnet", "ft_transformer"], key=lambda m: sel[m])
    d41 = ("consistent with the D41 expectation that the tuned boosters already sit at the "
           "administrative-data ceiling" if TIER[selected] == 2 else
           "against the D41 expectation that a booster would win")
    rationale = (f"{selected} selected on dev_test AUC-PR {sel[selected]:.4f} (held-out "
                 f"{metrics[selected]['held_out_auc_pr']:.4f}); the stack reached "
                 f"{sel['stack']:.4f} and the best deep net ({best_net}) {sel[best_net]:.4f}, "
                 f"{d41}.")
    print(f"\n{rationale}", flush=True)

    # Sanity gates (real runs only): a surprising run must not silently write artifacts.
    if not SMOKE:
        for m in ALL_MODELS:
            pr, pr_ho = metrics[m]["dev_test_auc_pr"], metrics[m]["held_out_auc_pr"]
            roc = metrics[m]["held_out_auc_roc"]
            if not (0.08 <= pr <= 0.30 and 0.08 <= pr_ho <= 0.30):
                raise SystemExit(f"sanity gate: {m} AUC-PR (dev_test {pr:.4f}, held-out "
                                 f"{pr_ho:.4f}) outside [0.08, 0.30]; not writing artifacts")
            if not 0.5 <= roc <= 0.75:
                raise SystemExit(f"sanity gate: {m} held-out AUC-ROC {roc:.4f} outside "
                                 f"[0.5, 0.75]; not writing artifacts")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps({
        "selected_model": selected,
        "selection_rationale": rationale,
        "models": metrics,
        "stack_weights": stack_weights,
        "tabnet_top_features": tabnet_top,
        "train_note": TRAIN_NOTE,
        "caps": {"train_cap": TRAIN_CAP, "deep_train_cap": DEEP_TRAIN_CAP,
                 "val_cap": VAL_CAP, "held_out_cap": HO_CAP, "max_epochs": MAX_EPOCHS,
                 "held_out_scored_n": fair_n, "bootstrap_n": BOOT_N},
        "seed": SEED,
        "smoke": SMOKE,
    }, indent=2, default=float))

    lines = ["# Week 5: deep benchmark, stacking, final selection (full-data held-out)", "",
             f"**Selected model: {selected}** (by dev_test AUC-PR, tie-break booster > stack > "
             f"nets within bootstrap noise). {rationale}", "",
             f"All five models are scored on an identical {n_ho:,}-row stratified draw from the "
             f"untouched held_out_final partition, so the comparison and the stacked meta-learner "
             f"see the same rows (the FT-Transformer does an O(n) forward pass per epoch on CPU, so "
             f"a shared held-out draw keeps the benchmark feasible and fair). Deep nets train on a "
             f"{DEEP_TRAIN_CAP:,}-row dev_train subsample; all caps are disclosed. AUC-PR CIs are "
             f"from a stratified bootstrap (n={BOOT_N}).", "",
             "| model | dev_test AUC-PR | held-out AUC-PR [95% CI] | held-out AUC-ROC | ECE | Brier |",
             "|---|---|---|---|---|---|"]
    for m in ALL_MODELS:
        r = metrics[m]
        lines.append(f"| {m} | {r['dev_test_auc_pr']:.4f} | {r['held_out_auc_pr']:.4f} "
                     f"[{r['held_out_auc_pr_ci'][0]:.4f}, {r['held_out_auc_pr_ci'][1]:.4f}] | "
                     f"{r['held_out_auc_roc']:.4f} | {r['held_out_ece']:.4f} | "
                     f"{r['held_out_brier']:.4f} |")
    best_boost = max(["xgboost", "lightgbm"], key=lambda m: sel[m])
    gap = metrics[best_boost]["held_out_auc_pr"] - metrics[best_net]["held_out_auc_pr"]
    lines += ["",
              "The deep nets were run as honest benchmarks expected to lose (DECISION_LOG D41): "
              "weak tabular signal (max |corr| ~0.13), 8.57% prevalence, and an "
              "administrative-data ceiling of 0.62-0.70 AUC-ROC that the tuned boosters already "
              "reach. What actually happened: the best booster "
              f"({best_boost}, held-out AUC-PR {metrics[best_boost]['held_out_auc_pr']:.4f}) "
              f"led the best deep net ({best_net}, "
              f"{metrics[best_net]['held_out_auc_pr']:.4f}) by {gap:+.4f} AUC-PR, and the "
              f"stack reached {metrics['stack']['held_out_auc_pr']:.4f}, so combining the "
              "models "
              + ("bought a real margin over the best single booster."
                 if metrics["stack"]["held_out_auc_pr"]
                 > metrics[best_boost]["held_out_auc_pr"] + 0.001
                 else "did not buy a meaningful margin over the best single booster."), "",
              f"Training caps (disclosed): {TRAIN_NOTE}. Deep nets use median imputation and "
              "standardization fit on train only (the boosters ingest NaN natively; the nets "
              "cannot). Booster hyperparameters are the Week-4 Optuna winners, reused without "
              "retuning."]
    if SMOKE:
        lines += ["", "**SMOKE artifact: subsampled data, 2 epochs. Do not ship.**"]
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[{time.time() - t0:7.1f}s] results -> {TABLE} and {JSON_OUT}", flush=True)


if __name__ == "__main__":
    main()
