"""Leakage-integrity tests for the ingest schema. These guard the project's core honesty claim."""

from equibirth.ingest import schema


def test_detector_and_structural_lists_are_disjoint():
    assert schema.LEAKAGE_DETECTOR_VARS.isdisjoint(schema.STRUCTURAL_EXCLUSION_VARS)


def test_dplural_is_structural_not_in_detector():
    # DPLURAL must NOT be in the FR6.1 detector (its gestation link is biology, would false-fail CI).
    assert "dplural" in schema.STRUCTURAL_EXCLUSION_VARS
    assert "dplural" not in schema.LEAKAGE_DETECTOR_VARS


def test_lmp_gestation_excluded_obstetric_is_target():
    assert schema.LMP_GESTATION in schema.STRUCTURAL_EXCLUSION_VARS
    assert schema.OBSTETRIC_GESTATION not in schema.all_excluded_vars()


def test_gdiab_not_in_medical_risk_composite():
    # FR6.2: gestational diabetes is a timing leak and must not enter the composite.
    assert "rf_gdiab" not in schema.MEDICAL_RISK_COMPOSITE_COMPONENTS


def test_key_leaky_vars_are_excluded():
    for v in ["dbwt", "ld_ster", "ld_antb", "apgar5", "previs", "attend", "bfacil"]:
        assert v in schema.all_excluded_vars(), v


def test_harmonized_features_never_map_a_leakage_detector_var():
    # No post-delivery / target-encoding leakage variable may receive a harmonized analysis name.
    for raw in schema.COLUMN_MAP:
        assert raw not in schema.LEAKAGE_DETECTOR_VARS, raw


def test_only_dplural_is_a_mapped_structural_var():
    # dplural is mapped because the loader needs it for the singleton filter; no other structural
    # var (e.g. the LMP gestation or a gestation recode) may receive a harmonized name.
    mapped_structural = set(schema.COLUMN_MAP) & schema.STRUCTURAL_EXCLUSION_VARS
    assert mapped_structural == {"dplural"}


def test_sentinels_do_not_overlap_valid_values():
    # previs sentinel is 99 (not 9); dplural has no sentinel so plurality 5 survives.
    assert 9 not in schema.SENTINEL_MAP["previs"]
    assert 99 in schema.SENTINEL_MAP["previs"]
    assert schema.SENTINEL_MAP["dplural"] == []
    assert schema.SENTINEL_MAP["mager"] == []
