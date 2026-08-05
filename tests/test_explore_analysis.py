"""Tests for Phase 8 exploration analytics (toy data with known answers)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from equibirth.explore import analysis as ea


def _toy() -> pl.DataFrame:
    # 4 White (1 preterm => 0.25), 4 Black (2 preterm => 0.50), across two education levels.
    return pl.DataFrame(
        {
            "preterm": [0, 1, 0, 0, 1, 1, 0, 0],
            "mother_race_ethnicity": ["NH White"] * 4 + ["NH Black"] * 4,
            "meduc": [3, 3, 6, 6, 3, 3, 6, 6],
            "birth_year": [2019, 2020, 2021, 2022, 2019, 2020, 2021, 2022],
        }
    )


def test_preterm_rate_by_computes_group_rates():
    out = ea.preterm_rate_by(_toy(), "mother_race_ethnicity", suppress_floor=1)
    rates = dict(zip(out["mother_race_ethnicity"], out["rate"]))
    assert rates["NH White"] == pytest.approx(0.25)
    assert rates["NH Black"] == pytest.approx(0.50)


def test_preterm_rate_by_suppresses_small_cells():
    out = ea.preterm_rate_by(_toy(), "mother_race_ethnicity", suppress_floor=5)
    # each group has n=4 < floor 5 => rate suppressed to null, counts retained
    assert out["rate"].null_count() == 2
    assert out["n"].sum() == 8


def test_preterm_rate_by_keeps_reliable_zero_rate():
    # a large all-term cell is a real 0% rate, not noise; must survive suppression
    df = pl.DataFrame({"preterm": [0] * 100, "g": ["A"] * 100})
    out = ea.preterm_rate_by(df, "g", suppress_floor=50)
    assert out["rate"].item() == pytest.approx(0.0)


def test_intersectional_rate_cell_values():
    out = ea.intersectional_rate(_toy(), "mother_race_ethnicity", "meduc", suppress_floor=1)
    key = {(r, e): rate for r, e, rate in zip(out["mother_race_ethnicity"], out["meduc"], out["rate"])}
    assert key[("NH White", 3)] == pytest.approx(0.5)   # [0,1] -> 0.5
    assert key[("NH Black", 6)] == pytest.approx(0.0)    # [0,0] -> 0.0


def test_gap_within_strata_persists_across_education():
    gap = ea.gap_within_strata(
        _toy(), "mother_race_ethnicity", "meduc",
        focal="NH Black", reference="NH White", suppress_floor=1,
    )
    g = dict(zip(gap["meduc"], gap["gap_pp"]))
    # edu3: Black [1,1]=1.0 minus White [0,1]=0.5 => +50pp ; edu6: 0.0 minus 0.0 => 0pp
    assert g[3] == pytest.approx(50.0)
    assert g[6] == pytest.approx(0.0)


def test_temporal_rates_per_year():
    out = ea.temporal_rates(_toy())
    assert out["birth_year"].to_list() == [2019, 2020, 2021, 2022]
    assert out.filter(pl.col("birth_year") == 2020)["rate"].item() == pytest.approx(1.0)


def test_covid_period_table_shift_sign():
    out = ea.covid_period_table(_toy(), group_col="mother_race_ethnicity")
    assert "covid_shift_pp" in out.columns
    # White: pre(2019)=0 rate, covid(2020,2021)=[1,0]->0.5 => +50pp shift
    white = out.filter(pl.col("mother_race_ethnicity") == "NH White")
    assert white["covid_shift_pp"].item() == pytest.approx(50.0)


def test_profile_clusters_basic():
    labels = np.array([0, 0, 1, 1])
    preterm = np.array([1, 0, 1, 1])
    race = np.array(["NH White", "NH White", "NH Black", "NH Black"])
    feats = pl.DataFrame({"bmi": [20.0, 30.0, 25.0, 25.0]})
    profs = ea.profile_clusters(labels, preterm, race, feats)
    p0, p1 = profs
    assert p0.preterm_rate == pytest.approx(0.5)
    assert p1.preterm_rate == pytest.approx(1.0)
    assert p0.race_composition["NH White"] == pytest.approx(1.0)
    assert p1.feature_means["bmi"] == pytest.approx(25.0)


def test_profile_clusters_handles_null_race():
    # null race must not break np.unique sorting (regression: None vs str comparison)
    labels = np.array([0, 0])
    profs = ea.profile_clusters(labels, np.array([1, 0]), np.array([None, "NH Black"], dtype=object),
                                pl.DataFrame({"bmi": [20.0, 30.0]}))
    assert profs[0].race_composition["Unknown"] == pytest.approx(0.5)
    assert profs[0].race_composition["NH Black"] == pytest.approx(0.5)
