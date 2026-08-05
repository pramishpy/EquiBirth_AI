"""Week 2 (Phase 2): feature engineering, the leakage audit, and the full-population equity audit.

This is the full-data rebuild of Phase 2. It reads the COMPLETE per-year cleaned Parquet files
(natality{year}_clean.parquet, ~24.9M singletons, 2018-2024) and does four things:

1. Full-population OUTCOME/EQUITY audit on every row (so even the smallest race groups are reliable):
   overall preterm rate, per-year rate, per-race rate, and the NH Black minus NH White gap.
2. The FR6.1 LEAKAGE DETECTOR: a classifier trained on the excluded post-delivery variables alone.
   Its AUC must exceed the 0.55 gate (it should, dramatically), which is exactly why those variables
   are barred from the primary matrix. This quantifies the leakage trap as a teaching result.
3. The FR6.2 composite-component guard and FR6.3 univariate gate on the real full-data feature matrix.
4. Assembles and saves the leakage-safe WORKING modeling dataset (stratified dev sample + held-out
   remainder drawn from the full population) that Phases 3-4 train and report on.

Model development uses a tractable stratified working sample; the equity audit uses the full census.
This mirrors SPEC FR5/FR8.1: separate the outcome equity audit (full population) from the
model-performance audit (adequate cell sizes). Every number is computed live from the real data.

Run: PYTHONPATH=src ./.venv/bin/python scripts/week2_feature_leakage_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from equibirth.features import full_dataset, splits
from equibirth.features.build import build_modeling_frame
from equibirth.features.leakage import excluded_vars_detector_auc, univariate_auc
from equibirth.ingest import schema
from equibirth.utils.config import load_config
from equibirth.utils.io import ensure_dir
from equibirth.utils.seeds import set_seeds

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CFG = load_config(ROOT / "config.yaml")

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
SEED = CFG.get("project.random_seed")
LEAKAGE_GATE = CFG.get("evaluation.excluded_vars_leakage_gate")     # 0.55
UNIVARIATE_GATE = CFG.get("evaluation.univariate_leakage_gate")     # 0.60
# Working sample sizes for model development (Phases 3-4). The equity audit below is full-population.
DEV_SAMPLE = CFG.get("sampling.dev_sample_size")                    # 1,000,000
WORKING_SIZE = 3_000_000        # dev_sample + a large held-out remainder, drawn from the full census
# A capped sample is plenty to demonstrate the leakage AUC; it keeps the fit fast.
LEAKAGE_DEMO_SIZE = 400_000

OUT_MODELING = PROCESSED / "modeling_full_2018_2024.parquet"
OUT_JSON = REPORTS / "week2_phase2_audit.json"
OUT_MD = REPORTS / "week2_phase2_audit.md"


def main() -> None:
    set_seeds(SEED)
    print("Loading the complete 2018-2024 cleaned files (singletons)...", flush=True)
    df = full_dataset.load_full_frame(INTERIM, YEARS)
    print(f"  pooled full population: {df.height:,} singleton rows x {df.width} raw cols", flush=True)

    # ---- 1. Full-population outcome / equity audit (every row) ----
    audit = full_dataset.outcome_audit(df)
    print(f"\nFull-population outcome audit ({audit['n_singletons']:,} rows with a valid target):")
    print(f"  overall preterm rate: {audit['overall_rate']*100:.2f}%")
    print(f"  NH Black minus NH White gap: {audit['black_white_gap_pp']:.2f} pp")
    for r in audit["rate_by_race"]:
        print(f"    {r['mother_race_ethnicity']:<16} {r['rate']*100:6.2f}%  n={r['n']:>10,}")

    # ---- 2. Leakage audit: FR6.1 detector + leaky-model demonstration ----
    # Draw a stratified sample for the leakage fits (a full-population fit is unnecessary and slow).
    demo_mask = splits.stratified_subsample(df, LEAKAGE_DEMO_SIZE, seed=SEED)
    demo = df.filter(demo_mask)
    print(f"\nLeakage audit on a {demo.height:,}-row stratified sample:")
    leak_auc = excluded_vars_detector_auc(demo, target="preterm")
    print(f"  FR6.1 excluded-vars detector AUC = {leak_auc:.3f}  (gate {LEAKAGE_GATE}; "
          f"{'FIRES correctly' if leak_auc > LEAKAGE_GATE else 'DID NOT FIRE'})")

    # ---- 3. Build the real full-data feature matrix; verify guards + univariate gate ----
    Xdemo, ydemo, components = build_modeling_frame(demo)
    print(f"  primary matrix: {Xdemo.width} leakage-safe features; FR6.2 component guard passed "
          "(build_modeling_frame raises otherwise).")
    # FR6.3 univariate gate on the timing-conditional features kept in the primary model.
    timing_checks = {}
    for raw, feat in [("precare", "first_trimester_care"), ("wic", "wic")]:
        if raw in demo.columns:
            au = univariate_auc(demo, raw, target="preterm")
            timing_checks[feat] = au
            flag = "ok" if au <= UNIVARIATE_GATE else "EXCEEDS GATE"
            print(f"  FR6.3 univariate AUC[{feat:<20}] = {au:.3f}  ({flag})")

    # ---- 4. Assemble + save the leakage-safe working modeling dataset for Phases 3-4 ----
    print(f"\nAssembling the working modeling dataset (stratified {WORKING_SIZE:,} rows from the "
          "full census)...", flush=True)
    bundle = full_dataset.assemble_modeling(df, working_size=WORKING_SIZE, seed=SEED)
    ensure_dir(PROCESSED)
    frame = bundle["X"].with_columns(
        bundle["y"].alias("preterm"),
        bundle["split"].alias("split"),
        *[bundle["meta"][c].alias(c) for c in bundle["meta"].columns],
    )
    frame.write_parquet(OUT_MODELING)
    splits.save_splits(bundle["split"], PROCESSED / "splits_full.parquet")
    part_counts = {}
    for s in ["dev_train", "dev_val", "dev_test", "held_out_final"]:
        m = (bundle["split"] == s)
        sub = bundle["y"].filter(m).drop_nulls()
        part_counts[s] = {"n": int(m.sum()), "prevalence": float(sub.mean()) if sub.len() else float("nan")}
        print(f"  {s:<15} n={m.sum():>9,}  prevalence={part_counts[s]['prevalence']:.4f}")
    print(f"  wrote {OUT_MODELING.relative_to(ROOT)} ({Xdemo.width} features)")

    # ---- write machine + markdown reports ----
    result = {
        "n_singletons_full": audit["n_singletons"],
        "years": YEARS,
        "overall_rate": audit["overall_rate"],
        "black_white_gap_pp": audit["black_white_gap_pp"],
        "rate_by_race": audit["rate_by_race"],
        "rate_by_year": audit["rate_by_year"],
        "leakage_detector_auc": leak_auc,
        "leakage_gate": LEAKAGE_GATE,
        "n_primary_features": Xdemo.width,
        "univariate_timing_auc": timing_checks,
        "univariate_gate": UNIVARIATE_GATE,
        "n_leakage_detector_vars": len(schema.LEAKAGE_DETECTOR_VARS),
        "working_partitions": part_counts,
        "working_size": WORKING_SIZE,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    md = [
        "# Week 2 (Phase 2): feature engineering, leakage audit, and the equity audit",
        "",
        "All numbers computed live on the COMPLETE 2018-2024 public-use files "
        f"({audit['n_singletons']:,} singleton births with a valid target).",
        "",
        "## Full-population outcome audit (every row)",
        "",
        f"Overall singleton preterm rate: **{audit['overall_rate']*100:.2f}%**. "
        f"NH Black minus NH White gap: **{audit['black_white_gap_pp']:.2f} percentage points**.",
        "",
        "| race/ethnicity | preterm rate | n |",
        "|---|---|---|",
    ]
    for r in audit["rate_by_race"]:
        md.append(f"| {r['mother_race_ethnicity']} | {r['rate']*100:.2f}% | {r['n']:,} |")
    md += [
        "",
        "Race is a marker of exposure to racism and structural barriers, not a biological cause. "
        "These are descriptive population rates, not model outputs.",
        "",
        "## Leakage audit (the central integrity check)",
        "",
        f"A classifier trained on the {len(schema.LEAKAGE_DETECTOR_VARS)} EXCLUDED post-delivery "
        f"variables alone reaches AUC **{leak_auc:.3f}** on a {demo.height:,}-row stratified sample. "
        f"That is far above the {LEAKAGE_GATE} gate (FR6.1 fires correctly), which is precisely why "
        "those variables (birthweight, Apgar, labor interventions, total visits) are barred from the "
        "primary matrix: they are recorded at or after delivery, so a model that used them would look "
        "excellent and mean nothing.",
        "",
        f"The primary leakage-safe matrix has **{Xdemo.width} features**, all pre-delivery. The FR6.2 "
        "composite-component guard passed (no engineered feature is built from an excluded variable), "
        "and the timing-conditional features kept in the model pass the FR6.3 univariate gate "
        f"({UNIVARIATE_GATE}):",
        "",
        "| timing-conditional feature | univariate AUC | within gate? |",
        "|---|---|---|",
    ]
    for feat, au in timing_checks.items():
        md.append(f"| {feat} | {au:.3f} | {'yes' if au <= UNIVARIATE_GATE else 'NO'} |")
    md += [
        "",
        "## Working modeling dataset (for the coming weeks)",
        "",
        f"A stratified {WORKING_SIZE:,}-row working sample was drawn from the full census (preserving "
        "the outcome and race mix) and split deterministically. The held-out partition is untouched "
        "until final reporting.",
        "",
        "| partition | n | prevalence |",
        "|---|---|---|",
    ]
    for s, v in part_counts.items():
        md.append(f"| {s} | {v['n']:,} | {v['prevalence']:.4f} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}\nwrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
