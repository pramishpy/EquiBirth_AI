"""Stream a natality CSV into a slim, cleaned Parquet file.

Memory-safe by design (SPEC NFR3): a lazy Polars pipeline projects to the needed columns, converts
per-variable sentinels to null, builds the target, and adds the harmonized race/ethnicity label,
then streams the result to Parquet without materializing the whole frame. One year is processed at a
time; downstream code concatenates the per-year Parquet files lazily.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from equibirth.ingest import schema
from equibirth.ingest.load import RACE_ETHNICITY_MAP
from equibirth.utils.io import ensure_dir


def _sentinel_exprs(columns: list[str]) -> list[pl.Expr]:
    """Lazy per-variable sentinel-to-null expressions for the present columns."""
    exprs: list[pl.Expr] = []
    for col, sentinels in schema.SENTINEL_MAP.items():
        if col in columns and sentinels:
            exprs.append(
                pl.when(pl.col(col).is_in(sentinels)).then(None).otherwise(pl.col(col)).alias(col)
            )
    return exprs


def build_clean_lazyframe(
    csv_path: str | Path, *, threshold_weeks: int = 37, singletons_only: bool = False
) -> pl.LazyFrame:
    """Return a lazy pipeline: project -> de-sentinel -> target -> race label -> (singletons)."""
    csv_path = Path(csv_path)
    lf = pl.scan_csv(csv_path, infer_schema_length=10000)
    available = set(lf.collect_schema().names())
    if schema.OBSTETRIC_GESTATION not in available:
        raise KeyError(
            f"Target column {schema.OBSTETRIC_GESTATION!r} missing from {csv_path.name} (FR1.1)."
        )
    keep = [c for c in schema.needed_raw_columns() if c in available]
    lf = lf.select(keep)

    sent = _sentinel_exprs(keep)
    if sent:
        lf = lf.with_columns(sent)

    oe = schema.OBSTETRIC_GESTATION
    lf = lf.with_columns(
        pl.when(pl.col(oe).is_null())
        .then(None)
        .otherwise((pl.col(oe) < threshold_weeks).cast(pl.Int8))
        .alias("preterm")
    )
    if "mracehisp" in keep:
        lf = lf.with_columns(
            pl.col("mracehisp").replace_strict(RACE_ETHNICITY_MAP, default=None)
            .alias("mother_race_ethnicity")
        )
    if singletons_only and "dplural" in keep:
        lf = lf.filter(pl.col("dplural") == 1)
    return lf


def convert_csv_to_parquet(
    csv_path: str | Path, out_path: str | Path, *,
    threshold_weeks: int = 37, singletons_only: bool = False,
) -> Path:
    """Stream a cleaned slim Parquet to ``out_path``. Returns the path."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    lf = build_clean_lazyframe(
        csv_path, threshold_weeks=threshold_weeks, singletons_only=singletons_only
    )
    try:
        lf.sink_parquet(out_path)
    except Exception:
        # Fallback for engines/versions where a given expr is not streamable.
        lf.collect().write_parquet(out_path)
    return out_path
