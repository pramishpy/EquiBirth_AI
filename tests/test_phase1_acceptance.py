"""SPEC-mandated Phase 1 acceptance tests (FR4.1, schema, sentinels).

The data-dependent tests run against the real spread samples when present and skip cleanly otherwise,
so CI without data still passes while local runs enforce the real acceptance criteria.
"""

from pathlib import Path

import polars as pl
import pytest

from equibirth.ingest import confirm, load, schema
from equibirth.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
CFG = load_config(REPO / "config.yaml")
YEARS = CFG.get("data.natality_years")
BAND = CFG.get("data.preterm_rate_sanity_band")


def _spread(year: int) -> Path:
    return REPO / "data" / "raw" / f"natality{year}_spread.csv"


def _available_years() -> list[int]:
    return [y for y in YEARS if _spread(y).is_file()]


# --- FR3: per-variable sentinels do not overlap valid ranges (parametrized over all numeric keys) ---
# Documented valid ranges for the numeric sentinel variables (string Y/N flags use 'U', no overlap).
_VALID_RANGES = {
    "oegest_comb": (17, 47), "combgest": (17, 47), "meduc": (1, 8), "precare": (0, 10),
    "previs": (0, 98), "cig_0": (0, 98), "bmi": (13.0, 69.9), "dmar": (1, 2),
    "pay": (1, 8), "priorlive": (0, 30), "priordead": (0, 30),
}


@pytest.mark.parametrize("var", sorted(_VALID_RANGES))
def test_sentinel_does_not_overlap_valid_range(var):
    lo, hi = _VALID_RANGES[var]
    for sentinel in schema.SENTINEL_MAP.get(var, []):
        assert not (lo <= sentinel <= hi), f"{var}: sentinel {sentinel} overlaps valid range {lo}-{hi}"


@pytest.mark.parametrize("year", _available_years() or [pytest.param(0, marks=pytest.mark.skip(reason="no spread data"))])
def test_schema_all_years(year):
    header = pl.read_csv(_spread(year), n_rows=1).columns
    rec = confirm.confirm_format(year, header)  # raises if a required column is missing
    assert rec["target_column"] in header


@pytest.mark.parametrize("year", _available_years() or [pytest.param(0, marks=pytest.mark.skip(reason="no spread data"))])
def test_preterm_rate_in_sanity_band(year):
    df = load.load_natality_csv(_spread(year))
    rate = df.filter(pl.col("preterm").is_not_null())["preterm"].mean()
    assert BAND[0] <= rate <= BAND[1], f"{year}: rate {rate:.4f} outside band {BAND}"


@pytest.mark.parametrize("year", _available_years() or [pytest.param(0, marks=pytest.mark.skip(reason="no spread data"))])
def test_target_range_20_45(year):
    df = load.load_natality_csv(_spread(year))
    oe = df[schema.OBSTETRIC_GESTATION].drop_nulls()
    # the vast majority of births fall in 20-45 completed weeks; allow rare <20 extreme preterm
    assert oe.min() >= 17 and oe.max() <= 47
    in_clinical = ((oe >= 20) & (oe <= 45)).mean()
    assert in_clinical > 0.99
