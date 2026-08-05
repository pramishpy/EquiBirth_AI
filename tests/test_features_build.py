"""Tests for the primary feature matrix builder (leakage-disciplined)."""

import polars as pl

from equibirth.features import build
from equibirth.ingest import schema


def _clean_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "preterm": [1, 0, 0, 1],
            "mager": [19, 27, 33, 41],
            "meduc": [2, 4, 6, 3],
            "dmar": [2, 1, 1, 2],
            "bmi": [22.0, 25.0, 30.0, 28.0],
            "cig_0": [0, 0, 5, 0],
            "priorlive": [0, 1, 2, 3],
            "priordead": [0, 0, 0, 1],
            "rf_pdiab": ["N", "N", "Y", "N"],
            "rf_phype": ["N", "N", "N", "Y"],
            "rf_ppterm": ["Y", "N", "N", "Y"],
            "rf_cesar": ["N", "N", "Y", "N"],
            "rf_gdiab": ["N", "Y", "N", "N"],
            "wic": ["Y", "N", "N", "Y"],
            "precare": [2, 1, 5, 99],
            "mracehisp": [2, 1, 7, 2],
            "pay": [1, 2, 2, 1],
            "mother_race_ethnicity": ["NH Black", "NH White", "Hispanic", "NH Black"],
            # leakage columns that must NOT appear in the matrix
            "dbwt": [2200, 3400, 3300, 2100],
            "previs": [8, 12, 13, 6],
            "dplural": [1, 1, 1, 1],
            "combgest": [35, 39, 40, 34],
        }
    )


def test_returns_features_target_components():
    X, y, comps = build.build_modeling_frame(_clean_frame())
    assert y.to_list() == [1, 0, 0, 1]
    assert X.height == 4
    assert "medical_risk_count" in X.columns
    assert "married" in X.columns
    assert "age_bracket" in X.columns


def test_no_excluded_variable_in_matrix():
    X, _, _ = build.build_modeling_frame(_clean_frame())
    excluded = schema.all_excluded_vars()
    assert not (set(X.columns) & excluded)
    for leaky in ["dbwt", "previs", "dplural", "combgest"]:
        assert leaky not in X.columns


def test_medical_risk_count_is_enumerated_sum():
    X, _, comps = build.build_modeling_frame(_clean_frame())
    # row0: ppterm=Y only -> 1 ; row3: phype+ppterm -> 2 ; row2: pdiab+cesar -> 2
    assert X["medical_risk_count"].to_list() == [1, 0, 2, 2]
    assert "rf_gdiab" not in comps["medical_risk_count"]   # gest diabetes excluded from composite


def test_first_trimester_care_recode():
    X, _, _ = build.build_modeling_frame(_clean_frame())
    # precare 2->1, 1->1, 5->0, 99(sentinel-like here treated as value)->0
    assert X["first_trimester_care"].to_list()[:3] == [1, 1, 0]


def test_components_only_reference_legitimate_sources():
    _, _, comps = build.build_modeling_frame(_clean_frame())
    excluded = schema.all_excluded_vars()
    for feature, sources in comps.items():
        assert not (set(sources) & excluded), (feature, sources)


def test_timing_conditional_can_be_excluded():
    X, _, _ = build.build_modeling_frame(_clean_frame(), include_timing_conditional=False)
    assert "first_trimester_care" not in X.columns
    assert "wic" not in X.columns
