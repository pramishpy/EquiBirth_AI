"""Exploration analytics (Phase 8): temporal, intersectional, and unsupervised structure.

Pure, deterministic functions over already-loaded Polars frames so they are unit-testable and never
re-read raw data. The narrative built on top of these (notebooks, dashboard) cites only numbers these
functions produced. Interpretation is associational: race is a social construct and a marker of
exposure to racism, not a biological cause (SPEC FR11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

# COVID-era period labels (DECISION_LOG D13; 2020-2021 flagged atypical for care utilization).
PRE_COVID_YEARS = (2019,)
COVID_YEARS = (2020, 2021)
POST_COVID_YEARS = (2022, 2023, 2024)


def preterm_rate_by(df: pl.DataFrame, group_col: str, *, suppress_floor: int = 50) -> pl.DataFrame:
    """Preterm rate and n per group; cells with fewer than ``suppress_floor`` total births keep their
    counts but get a null rate (small-denominator suppression, do not silently blank whole groups).

    A genuine 0% rate over a large denominator is reliable and is NOT suppressed; only thin cells are.
    """
    valid = df.filter(pl.col("preterm").is_not_null() & pl.col(group_col).is_not_null())
    out = (
        valid.group_by(group_col)
        .agg(n=pl.len(), positives=pl.col("preterm").sum(), rate=pl.col("preterm").mean())
        .sort(group_col)
    )
    return out.with_columns(
        pl.when(pl.col("n") >= suppress_floor).then(pl.col("rate")).otherwise(None).alias("rate")
    )


def intersectional_rate(
    df: pl.DataFrame, row_col: str, col_col: str, *, suppress_floor: int = 50
) -> pl.DataFrame:
    """Preterm rate for every (row_col, col_col) cell with tiered suppression.

    Returns long format (row_col, col_col, n, positives, rate). The killer equity view: does the
    Black-White gap persist *within* an SES stratum (education, payment source)?
    """
    valid = df.filter(
        pl.col("preterm").is_not_null()
        & pl.col(row_col).is_not_null()
        & pl.col(col_col).is_not_null()
    )
    out = (
        valid.group_by([row_col, col_col])
        .agg(n=pl.len(), positives=pl.col("preterm").sum(), rate=pl.col("preterm").mean())
        .sort([row_col, col_col])
    )
    return out.with_columns(
        pl.when(pl.col("n") >= suppress_floor).then(pl.col("rate")).otherwise(None).alias("rate")
    )


def gap_within_strata(
    df: pl.DataFrame,
    group_col: str,
    stratum_col: str,
    *,
    focal: str,
    reference: str,
    suppress_floor: int = 50,
) -> pl.DataFrame:
    """Focal-minus-reference preterm-rate gap within each stratum.

    Tests whether SES (education, payment) explains the racial gap: if the gap survives within every
    education level, education alone cannot account for it. Returns one row per stratum with the focal
    rate, reference rate, the percentage-point gap, and both cell sizes. A null gap means a cell was
    suppressed.
    """
    inter = intersectional_rate(df, group_col, stratum_col, suppress_floor=suppress_floor)
    f = inter.filter(pl.col(group_col) == focal).select(
        [stratum_col, pl.col("rate").alias("focal_rate"), pl.col("n").alias("focal_n")]
    )
    r = inter.filter(pl.col(group_col) == reference).select(
        [stratum_col, pl.col("rate").alias("ref_rate"), pl.col("n").alias("ref_n")]
    )
    joined = f.join(r, on=stratum_col, how="full", coalesce=True).sort(stratum_col)
    return joined.with_columns(
        ((pl.col("focal_rate") - pl.col("ref_rate")) * 100).alias("gap_pp")
    )


def temporal_rates(df: pl.DataFrame, *, group_col: str | None = None) -> pl.DataFrame:
    """Per-year preterm rate, optionally within each group. Used for the COVID trend-break check."""
    keys = ["birth_year"] + ([group_col] if group_col else [])
    valid = df.filter(pl.col("preterm").is_not_null())
    if group_col:
        valid = valid.filter(pl.col(group_col).is_not_null())
    return (
        valid.group_by(keys)
        .agg(n=pl.len(), rate=pl.col("preterm").mean())
        .sort(keys)
    )


def covid_period_table(df: pl.DataFrame, *, group_col: str | None = None) -> pl.DataFrame:
    """Preterm rate aggregated into pre-COVID (2019), COVID (2020-2021), post (2022-2024).

    Returns rate per period (per group if given) plus the COVID-minus-precovid shift in percentage
    points, the quantity a real lockdown effect would move. Interpretation stays associational.
    """
    period = (
        pl.when(pl.col("birth_year").is_in(list(PRE_COVID_YEARS)))
        .then(pl.lit("pre_covid"))
        .when(pl.col("birth_year").is_in(list(COVID_YEARS)))
        .then(pl.lit("covid"))
        .otherwise(pl.lit("post_covid"))
        .alias("period")
    )
    keys = ["period"] + ([group_col] if group_col else [])
    valid = df.filter(pl.col("preterm").is_not_null()).with_columns(period)
    if group_col:
        valid = valid.filter(pl.col(group_col).is_not_null())
    agg = valid.group_by(keys).agg(n=pl.len(), rate=pl.col("preterm").mean())
    wide = agg.pivot(values="rate", index=group_col, on="period") if group_col else agg
    if group_col and {"covid", "pre_covid"}.issubset(set(wide.columns)):
        wide = wide.with_columns(
            ((pl.col("covid") - pl.col("pre_covid")) * 100).alias("covid_shift_pp")
        ).sort(group_col)
    return wide


@dataclass
class ClusterProfile:
    """Descriptive summary of one unsupervised cluster."""

    cluster: int
    n: int
    preterm_rate: float
    race_composition: dict[str, float] = field(default_factory=dict)
    feature_means: dict[str, float] = field(default_factory=dict)


def profile_clusters(
    labels: np.ndarray,
    preterm: np.ndarray,
    race: np.ndarray,
    features: pl.DataFrame,
) -> list[ClusterProfile]:
    """Per-cluster size, preterm rate, race composition, and mean of each feature column.

    ``features`` is a numeric Polars frame aligned row-wise with ``labels``. Noise points (label -1
    from HDBSCAN) are reported as their own cluster so nothing is silently dropped.
    """
    labels = np.asarray(labels)
    preterm = np.asarray(preterm, dtype=float)
    race = np.array(["Unknown" if r is None else str(r) for r in race], dtype=object)
    fcols = features.columns
    fvals = {c: features[c].to_numpy() for c in fcols}
    out: list[ClusterProfile] = []
    for c in sorted(set(int(x) for x in labels)):
        m = labels == c
        n = int(m.sum())
        races, counts = np.unique(race[m], return_counts=True)
        comp = {str(rk): float(ct) / n for rk, ct in zip(races, counts)}
        means = {col: float(np.nanmean(fvals[col][m])) for col in fcols}
        out.append(
            ClusterProfile(
                cluster=c,
                n=n,
                preterm_rate=float(np.nanmean(preterm[m])) if n else float("nan"),
                race_composition=comp,
                feature_means=means,
            )
        )
    return out
