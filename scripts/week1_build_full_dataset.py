"""Build the week-1 exploration table from the FULL NBER natality files (2018-2024).

This replaces the earlier spread-sample shortcut with the complete public-use files. It reuses the
project's existing, tested ingest pipeline (equibirth.ingest.to_parquet.convert_csv_to_parquet) to
stream each full year's CSV into a slim, de-sentinelled, harmonized, singletons-only Parquet, then
pools the years and projects to the 17 raw pre-delivery fields the exploration notebooks use.

Year window: 2018-2024. 2018 is INCLUDED. We verified directly (header + sample-row diff) that the
2018 file is schema-identical to 2019 (238 columns, identical race recodes mracehisp 1-8 and mrace6
1-6), so the "2003-revision form consistency" argument that justifies pooling 2019+ applies equally
to 2018. Including 2018 also gives two clean pre-COVID baseline years (2018, 2019).

This does NO feature engineering or modeling. It produces the raw exploration table plus the two
simple derived fields the EDA already used (married from dmar, first_trimester_care from precare),
exactly as the feature pipeline defines them, so the notebooks and dashboard are unchanged in shape.

Run: PYTHONPATH=src ./.venv/bin/python scripts/week1_build_full_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from equibirth.ingest.to_parquet import convert_csv_to_parquet

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
EXPLORE_OUT = INTERIM / "natality_exploration_2019_2024.parquet"  # kept name; notebooks read this path
FULL_OUT = PROCESSED / "natality_full_2018_2024.parquet"

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Expected file byte sizes, so we refuse to process a truncated download.
EXPECTED_BYTES = {
    2018: 1873289723, 2019: 1851708487, 2020: 1776551100, 2021: 1801003417,
    2022: 1804046181, 2023: 2212345146, 2024: 2232876130,
}

# The 17 raw pre-delivery fields the week-1 notebooks and dashboard explore.
RAW_FIELDS = [
    "birth_year", "mager", "meduc", "bmi", "cig_0", "priorlive", "priordead", "married",
    "rf_pdiab", "rf_phype", "rf_ppterm", "rf_cesar", "wic", "first_trimester_care", "pay",
    "mother_race_ethnicity", "preterm",
]


def _coerce_rf_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The rf_* risk flags arrive as Y/N strings; map to 1/0 (U already nulled by the pipeline)."""
    cols = set(lf.collect_schema().names())
    exprs = []
    for c in ["rf_pdiab", "rf_phype", "rf_ppterm", "rf_cesar"]:
        if c in cols:
            exprs.append(
                pl.when(pl.col(c) == "Y").then(1)
                .when(pl.col(c) == "N").then(0)
                .otherwise(None).cast(pl.Int8).alias(c)
            )
    # wic likewise Y/N
    if "wic" in cols:
        exprs.append(
            pl.when(pl.col("wic") == "Y").then(1)
            .when(pl.col("wic") == "N").then(0)
            .otherwise(None).cast(pl.Int8).alias("wic")
        )
    return lf.with_columns(exprs) if exprs else lf


def build_year(year: int) -> pl.LazyFrame:
    """Convert one full year's CSV to a slim singletons-only Parquet and return a projected LazyFrame."""
    csv = RAW / f"natality{year}us.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} missing. Run scripts/week1_download_full.sh first.")
    have = csv.stat().st_size
    want = EXPECTED_BYTES[year]
    if have != want:
        raise ValueError(f"{csv.name} is {have} bytes, expected {want}; download is incomplete.")

    slim = INTERIM / f"natality{year}_clean.parquet"
    if not slim.is_file():
        print(f"  {year}: converting full CSV -> {slim.name} ...", flush=True)
        convert_csv_to_parquet(csv, slim, singletons_only=True)
    lf = pl.scan_parquet(slim)
    lf = _coerce_rf_flags(lf)
    cols = set(lf.collect_schema().names())
    # derived fields, matching equibirth.features.build exactly
    derived = [pl.lit(year).alias("birth_year")]
    if "dmar" in cols:
        derived.append((pl.col("dmar") == 1).cast(pl.Int8).alias("married"))
    if "precare" in cols:
        derived.append(
            pl.when(pl.col("precare").is_null()).then(None)
            .otherwise(pl.col("precare").is_between(1, 3).cast(pl.Int8))
            .alias("first_trimester_care")
        )
    lf = lf.with_columns(derived)
    present = [c for c in RAW_FIELDS if c in set(lf.collect_schema().names())]
    return lf.select(present)


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    frames = []
    for y in YEARS:
        print(f"{y}: processing...", flush=True)
        frames.append(build_year(y).collect())
    df = pl.concat(frames, how="diagonal")  # diagonal: tolerate a year missing an optional column

    # Persist the pooled full table and the exploration projection (same columns the notebooks read).
    df.write_parquet(FULL_OUT)
    df.select([c for c in RAW_FIELDS if c in df.columns]).write_parquet(EXPLORE_OUT)

    n = df.height
    rate = df.drop_nulls("preterm")["preterm"].mean() * 100
    by_year = df.group_by("birth_year").len().sort("birth_year")
    print(f"\nPooled FULL singleton dataset: {n:,} rows x {df.width} fields")
    print(f"overall preterm rate: {rate:.2f}%")
    print("rows by year:")
    for r in by_year.iter_rows(named=True):
        print(f"  {r['birth_year']}: {r['len']:,}")
    print(f"\nwrote {FULL_OUT.relative_to(ROOT)}")
    print(f"wrote {EXPLORE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
