"""Assemble the multi-year modeling dataset from per-year spread samples.

Clean frames are concatenated first (consistent raw columns), then the feature matrix is built once on
the combined frame so one-hot columns are identical across years. Stratified splits are assigned on the
combined frame and saved alongside the features and target, giving a single reproducible processed
dataset for modeling.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from equibirth.features import splits
from equibirth.features.build import build_modeling_frame
from equibirth.ingest import confirm
from equibirth.ingest.load import load_natality_csv
from equibirth.utils.io import ensure_dir


def load_year_clean(
    year: int, raw_dir: str | Path, *, singletons_only: bool = True,
    threshold_weeks: int = 37, confirm_log: str | Path | None = None,
) -> pl.DataFrame:
    """Load one year's spread-sample CSV as a clean frame tagged with ``birth_year``.

    Runs the FR1.1 column-confirmation gate (raises if a required column is missing) and, when
    ``confirm_log`` is given, writes a machine record of the confirmation.
    """
    path = Path(raw_dir) / f"natality{year}_spread.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing spread sample for {year}: {path}")
    header = pl.read_csv(path, n_rows=1).columns
    record = confirm.confirm_format(year, header)
    if confirm_log is not None:
        confirm.log_format_confirmation(record, confirm_log)
    df = load_natality_csv(path, singletons_only=singletons_only, threshold_weeks=threshold_weeks)
    return df.with_columns(pl.lit(year).alias("birth_year"))


def assemble(
    years: list[int], raw_dir: str | Path, *, singletons_only: bool = True, seed: int = 42,
    threshold_weeks: int = 37, confirm_log: str | Path | None = None,
) -> dict:
    """Build the combined features, target, meta, and split labels for the given years."""
    frames = [
        load_year_clean(y, raw_dir, singletons_only=singletons_only,
                        threshold_weeks=threshold_weeks, confirm_log=confirm_log)
        for y in years
    ]
    combined = pl.concat(frames, how="diagonal_relaxed")

    X, y, components = build_modeling_frame(combined)
    meta = combined.select(
        [c for c in ["birth_year", "mother_race_ethnicity", "meduc", "pay"] if c in combined.columns]
    )
    split = splits.assign_splits(combined, seed=seed)
    return {"X": X, "y": y, "meta": meta, "split": split, "components": components}


def save_processed(bundle: dict, out_path: str | Path) -> Path:
    """Write features + target + meta + split to a single processed Parquet."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    frame = bundle["X"].with_columns(
        bundle["y"].alias("preterm"),
        bundle["split"].alias("split"),
        *[bundle["meta"][c].alias(c) for c in bundle["meta"].columns],
    )
    frame.write_parquet(out_path)
    return out_path
