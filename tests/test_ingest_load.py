"""Tests for the natality loader: sentinels, target, race mapping, singleton filter."""

from pathlib import Path

import polars as pl
import pytest

from equibirth.ingest import load
from equibirth.ingest import schema

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "data" / "raw" / "natality2023_sample.csv"


def _toy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "oegest_comb": [36, 37, 40, 99],     # 99 is the unknown sentinel
            "combgest": [36, 38, 40, 41],
            "previs": [9, 12, 99, 5],            # 9 is valid (9 visits), 99 is unknown
            "dplural": [1, 1, 2, 1],
            "mracehisp": [1, 2, 7, 8],           # 8 -> unknown -> null
            "rf_pdiab": ["N", "Y", "U", "N"],    # U -> null
        }
    )


def test_apply_sentinels_is_per_variable():
    out = load.apply_sentinels(_toy())
    previs = out["previs"].to_list()
    assert previs[0] == 9        # valid value preserved
    assert previs[2] is None     # 99 -> null
    assert out["oegest_comb"].to_list()[3] is None  # 99 -> null
    assert out["rf_pdiab"].to_list()[2] is None     # 'U' -> null


def test_build_target_marks_preterm_and_nulls():
    out = load.build_target(load.apply_sentinels(_toy()))
    target = out["preterm"].to_list()
    assert target[0] == 1   # 36 weeks -> preterm
    assert target[1] == 0   # 37 weeks -> term
    assert target[3] is None  # gestation unknown -> target null, not term


def test_map_race_ethnicity_labels_and_unknown():
    out = load.map_race_ethnicity(load.apply_sentinels(_toy()))
    labels = out["mother_race_ethnicity"].to_list()
    assert labels[0] == "NH White"
    assert labels[2] == "Hispanic"
    assert labels[3] is None  # code 8 origin unknown


def test_filter_singletons_keeps_only_plurality_one():
    out = load.filter_singletons(_toy())
    assert out.height == 3
    assert set(out["dplural"].to_list()) == {1}


def test_build_target_raises_without_target_column():
    with pytest.raises(KeyError):
        load.build_target(pl.DataFrame({"x": [1]}))


@pytest.mark.skipif(not SAMPLE.is_file(), reason="real 2023 sample not present")
def test_real_sample_loads_and_targets():
    df = load.load_natality_csv(SAMPLE, n_rows=20000)
    assert "preterm" in df.columns
    assert "mother_race_ethnicity" in df.columns
    # target is binary or null only
    assert set(df["preterm"].drop_nulls().unique().to_list()) <= {0, 1}
    # obstetric estimate sentinel 99 is gone after de-sentinelling
    assert 99 not in df[schema.OBSTETRIC_GESTATION].drop_nulls().to_list()
