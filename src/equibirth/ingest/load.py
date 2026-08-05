"""Load a natality CSV into a clean, harmonized Polars frame.

Steps, in order:
1. Read the CSV (or a row subset for development).
2. Validate the obstetric-estimate target column is present (fail loudly otherwise; FR1.4).
3. Convert per-variable "unknown / not stated" sentinels to null (never a global list; FR3).
4. Build the binary preterm target from the obstetric estimate (FR4).
5. Map race/ethnicity codes to the harmonized category labels.
6. Optionally restrict to singleton births (FR5).

The loader does not engineer features or drop leakage columns; that is the feature pipeline's job
(Phase 2). It produces a faithful, de-sentinelled, harmonized table plus the target.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from equibirth.ingest import schema

# mracehisp integer code -> harmonized race/ethnicity label. Code 8 (origin unknown) -> null.
RACE_ETHNICITY_MAP: dict[int, str] = {
    1: "NH White",
    2: "NH Black",
    3: "NH AIAN",
    4: "NH Asian",
    5: "NH NHOPI",
    6: "NH Multiracial",
    7: "Hispanic",
    8: None,  # origin unknown / not stated
}


def apply_sentinels(df: pl.DataFrame, sentinel_map: dict[str, list] | None = None) -> pl.DataFrame:
    """Replace each variable's per-variable sentinel codes with null.

    Only columns present in the frame are touched. Valid values that happen to look like sentinels
    for a different variable are protected because the map is per-variable (e.g. 9 visits is not a
    sentinel for ``previs``, whose sentinel is 99).
    """
    sentinel_map = sentinel_map or schema.SENTINEL_MAP
    exprs = []
    for col, sentinels in sentinel_map.items():
        if col in df.columns and sentinels:
            exprs.append(
                pl.when(pl.col(col).is_in(sentinels)).then(None).otherwise(pl.col(col)).alias(col)
            )
    return df.with_columns(exprs) if exprs else df


def build_target(df: pl.DataFrame, threshold_weeks: int = 37) -> pl.DataFrame:
    """Add the binary ``preterm`` target from the obstetric-estimate column.

    Rows whose obstetric estimate is null (was a sentinel) get a null target and are excluded from
    target-based analysis rather than silently labeled term.
    """
    oe = schema.OBSTETRIC_GESTATION
    if oe not in df.columns:
        raise KeyError(f"Obstetric-estimate column {oe!r} not found; cannot build target (FR1.4).")
    return df.with_columns(
        pl.when(pl.col(oe).is_null())
        .then(None)
        .otherwise((pl.col(oe) < threshold_weeks).cast(pl.Int8))
        .alias("preterm")
    )


def map_race_ethnicity(df: pl.DataFrame) -> pl.DataFrame:
    """Add the harmonized ``mother_race_ethnicity`` label from the mracehisp code."""
    if "mracehisp" not in df.columns:
        return df
    return df.with_columns(
        pl.col("mracehisp").replace_strict(RACE_ETHNICITY_MAP, default=None).alias("mother_race_ethnicity")
    )


def filter_singletons(df: pl.DataFrame) -> pl.DataFrame:
    """Restrict to singleton births (plurality == 1)."""
    if "dplural" not in df.columns:
        raise KeyError("Plurality column 'dplural' not found; cannot filter singletons (FR5).")
    return df.filter(pl.col("dplural") == 1)


def load_natality_csv(
    path: str | Path,
    *,
    threshold_weeks: int = 37,
    singletons_only: bool = False,
    n_rows: int | None = None,
) -> pl.DataFrame:
    """Load, de-sentinel, target, and harmonize a natality CSV.

    ``n_rows`` reads only the first N rows for fast development. NOTE: the NBER CSV is NOT randomly
    ordered, so a head slice is not nationally representative; use it for code validation only, never
    for headline rates (see DECISION_LOG; FR4.1 rate checks run on full or randomly sampled data).
    """
    path = Path(path)
    df = pl.read_csv(path, n_rows=n_rows, infer_schema_length=10000)
    if schema.OBSTETRIC_GESTATION not in df.columns:
        raise KeyError(
            f"Target column {schema.OBSTETRIC_GESTATION!r} missing from {path.name}; "
            "confirm the year's column names (FR1.1)."
        )
    df = apply_sentinels(df)
    df = build_target(df, threshold_weeks=threshold_weeks)
    df = map_race_ethnicity(df)
    if singletons_only:
        df = filter_singletons(df)
    return df
