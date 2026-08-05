"""Full-population assembly for the complete 2018-2024 rebuild (Phase 2 onward).

The earlier ``features.dataset`` path reads per-year spread-sample CSVs. This module reads the
per-year CLEANED Parquet files (``natality{year}_clean.parquet``, already de-sentinelled, singletons,
with the ``preterm`` target and ``mother_race_ethnicity`` label) produced from the COMPLETE public-use
files, and splits the work the way the SPEC prescribes (FR5/FR8.1):

- ``outcome_audit``: runs on EVERY row of the pooled population. Because it is the full census, even
  the smallest race groups (AIAN, NHOPI) are reliable here. This is the equity story.
- ``assemble_modeling``: draws a deterministic stratified WORKING sample (tractable to train on,
  ~10 GB dense at 25M rows is not), builds the leakage-safe primary matrix via the tested
  ``build_modeling_frame``, and assigns the same deterministic stratified splits. Reported model
  metrics come from the held_out_final partition of this working sample.

Nothing here re-implements cleaning, target construction, feature engineering, or leakage control;
it composes the already-tested ingest and feature layers.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from equibirth.features import splits
from equibirth.features.build import build_modeling_frame


def load_full_frame(interim_dir: str | Path, years: list[int]) -> pl.DataFrame:
    """Pool the per-year cleaned Parquet files into one frame tagged with ``birth_year``.

    Each ``natality{year}_clean.parquet`` is the output of the tested ingest pipeline on the complete
    CSV for that year (singletons, de-sentinelled, target + race label already present).
    """
    interim_dir = Path(interim_dir)
    frames = []
    for year in years:
        path = interim_dir / f"natality{year}_clean.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing cleaned file for {year}: {path}. Run scripts/week1_build_full_dataset.py."
            )
        lf = pl.scan_parquet(path)
        if "birth_year" not in lf.collect_schema().names():
            lf = lf.with_columns(pl.lit(year).alias("birth_year"))
        frames.append(lf.collect())
    return pl.concat(frames, how="diagonal_relaxed")


def outcome_audit(df: pl.DataFrame) -> dict:
    """Full-population outcome/equity audit (every row, so small groups stay reliable).

    Returns overall rate, per-year rate, per-race rate + count, and the NH Black minus NH White gap
    in percentage points. All descriptive and associational: race is a marker of exposure to racism,
    not a biological cause.
    """
    valid = df.filter(pl.col("preterm").is_not_null())
    n = valid.height
    overall = float(valid["preterm"].mean()) if n else float("nan")

    by_race = (
        valid.filter(pl.col("mother_race_ethnicity").is_not_null())
        .group_by("mother_race_ethnicity")
        .agg(rate=pl.col("preterm").mean(), n=pl.len())
        .sort("rate")
    )
    by_year = (
        valid.group_by("birth_year").agg(rate=pl.col("preterm").mean(), n=pl.len()).sort("birth_year")
        if "birth_year" in valid.columns
        else pl.DataFrame({"birth_year": [], "rate": [], "n": []})
    )

    rate_by_race = [
        {"mother_race_ethnicity": r["mother_race_ethnicity"], "rate": float(r["rate"]), "n": int(r["n"])}
        for r in by_race.iter_rows(named=True)
    ]
    race_rate = {r["mother_race_ethnicity"]: r["rate"] for r in rate_by_race}
    gap = (
        (race_rate["NH Black"] - race_rate["NH White"]) * 100
        if {"NH Black", "NH White"} <= race_rate.keys()
        else float("nan")
    )
    return {
        "n_singletons": n,
        "overall_rate": overall,
        "rate_by_race": rate_by_race,
        "rate_by_year": [
            {"birth_year": int(r["birth_year"]), "rate": float(r["rate"]), "n": int(r["n"])}
            for r in by_year.iter_rows(named=True)
        ],
        "black_white_gap_pp": gap,
    }


def assemble_modeling(
    df: pl.DataFrame,
    *,
    working_size: int,
    held_out_frac: float = 0.5,
    seed: int = 42,
) -> dict:
    """Draw a stratified working sample and build the leakage-safe modeling bundle.

    ``working_size`` bounds the number of rows carried into training/evaluation (the full population
    is too large to hold as a dense matrix). Splits are stratified and deterministic; the returned
    ``held_out_final`` partition is what final metrics and bootstrap CIs are computed on.
    """
    mask = splits.stratified_subsample(df, working_size, seed=seed)
    work = df.filter(mask)

    X, y, components = build_modeling_frame(work)
    meta = work.select(
        [c for c in ["birth_year", "mother_race_ethnicity", "meduc", "pay"] if c in work.columns]
    )
    split = splits.assign_splits(work, held_out_frac=held_out_frac, seed=seed)
    return {"X": X, "y": y, "meta": meta, "split": split, "components": components}
