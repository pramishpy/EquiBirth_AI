"""SAMPLE-ERA (retained): part of the spread-sample chain that regenerates the Streamlit app's
reports/dashboard_data.json bundle. Its numbers are NOT the canonical ones and must not be cited;
the canonical full-data (2018-2024) pipeline lives in scripts/week*_*.py and reports/week*. This
chain is kept only so the app's exploration sections stay reproducible, and the app discloses on
screen that those sections use a validated representative spread sample.

Assemble the full 2019-2024 modeling dataset and confirm the performance ceiling.

Builds the combined feature matrix, assigns stratified splits, saves the processed Parquet, then
trains on dev_train and reports honest metrics on the untouched held_out_final partition, overall and
stratified by race/ethnicity. This is the multi-year confirmation of FINDINGS F5 plus the first
race-stratified performance read (a Phase 4 preview, computed only on held-out data).

Run: ./.venv/bin/python scripts/phase2_build_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from equibirth.features import dataset, splits
from equibirth.utils.config import load_config
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "modeling_2019_2024.parquet"
CFG = load_config(ROOT / "config.yaml")
YEARS = CFG.get("data.natality_years")
SEED = CFG.get("project.random_seed")
THRESHOLD = CFG.get("data.preterm_threshold_weeks")
RACE_GROUPS = CFG.get("data.race_ethnicity_harmonized_categories")
SUPPRESS_FLOOR = CFG.get("evaluation.cell_suppression_floor")
FULL_CI_FLOOR = CFG.get("evaluation.full_ci_floor")


def main() -> None:
    set_seeds(SEED)
    bundle = dataset.assemble(
        YEARS, RAW, singletons_only=True, seed=SEED, threshold_weeks=THRESHOLD,
        confirm_log=ROOT / "reports" / "format_confirmation.json",
    )
    dataset.save_processed(bundle, OUT)
    splits.save_splits(bundle["split"], ROOT / "data" / "processed" / "splits.parquet")

    X, y, split, meta = bundle["X"], bundle["y"], bundle["split"], bundle["meta"]
    print(f"assembled: rows={X.height:,} features={X.width} years={YEARS}")
    for s in ["dev_train", "dev_val", "dev_test", "held_out_final"]:
        m = (split == s)
        sub = y.filter(m)
        print(f"  {s:<15} n={m.sum():>8,}  prevalence={sub.drop_nulls().mean():.4f}")

    # fit on dev_train, evaluate on held_out_final (no leakage)
    yv = y.is_not_null()
    tr = ((split == "dev_train") & yv).to_numpy()
    ho = ((split == "held_out_final") & yv).to_numpy()
    Xall = X.to_numpy().astype(float)
    yall = y.fill_null(0).to_numpy().astype(int)

    imp = SimpleImputer(strategy="median").fit(Xall[tr])
    Xtr, Xho = imp.transform(Xall[tr]), imp.transform(Xall[ho])
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), yall[tr])
    gb = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, yall[tr])
    p_lr = lr.predict_proba(sc.transform(Xho))[:, 1]
    p_gb = gb.predict_proba(Xho)[:, 1]

    yho = yall[ho]
    print(f"\nHELD-OUT (n={ho.sum():,}, prevalence={yho.mean():.4f}):")
    print(f"  Logistic AUC-ROC={roc_auc_score(yho, p_lr):.3f}  AUC-PR={average_precision_score(yho, p_lr):.3f}")
    print(f"  HistGBM  AUC-ROC={roc_auc_score(yho, p_gb):.3f}  AUC-PR={average_precision_score(yho, p_gb):.3f}")

    race = meta["mother_race_ethnicity"].to_numpy()[ho]
    print("\nrace-stratified (HistGBM, held-out; AUC-ROC is base-rate invariant):")
    for grp in RACE_GROUPS:
        gm = race == grp
        pos = int(yho[gm].sum())
        if gm.sum() < SUPPRESS_FLOOR or pos < 10 or len(np.unique(yho[gm])) < 2:
            print(f"  {grp:<15} n={gm.sum():>7,}  pos={pos:>5}  suppressed (cell too small)")
            continue
        auc = roc_auc_score(yho[gm], p_gb[gm])
        tag = "" if pos >= FULL_CI_FLOOR else "  (wide CI, caution)"
        print(f"  {grp:<15} n={gm.sum():>7,}  pos={pos:>5}  rate={yho[gm].mean():.3f}  AUC-ROC={auc:.3f}{tag}")

    print(f"\nprocessed dataset saved to {OUT}")


if __name__ == "__main__":
    main()
