"""Tests for the FR1.1 format-confirmation gate and FR1.4 config guard."""

import json

import pytest

from equibirth.ingest import confirm
from equibirth.ingest.confirm import ConfigurationError
from equibirth.utils.config import Config


def _good_columns():
    return list(confirm.REQUIRED_COLUMNS) + ["mager14", "extra"]


def test_confirm_format_accepts_complete_columns():
    rec = confirm.confirm_format(2024, _good_columns())
    assert rec["year"] == 2024
    assert rec["target_column"] == "oegest_comb"
    assert rec["lmp_excluded_column"] == "combgest"


def test_confirm_format_raises_on_missing_required_column():
    cols = [c for c in confirm.REQUIRED_COLUMNS if c != "oegest_comb"]
    with pytest.raises(ConfigurationError):
        confirm.confirm_format(2024, cols)


def test_format_gate_logged(tmp_path):
    log = tmp_path / "format_confirmation.json"
    confirm.log_format_confirmation(confirm.confirm_format(2023, _good_columns()), log)
    confirm.log_format_confirmation(confirm.confirm_format(2024, _good_columns()), log)
    data = json.loads(log.read_text())
    assert set(data.keys()) == {"2023", "2024"}
    assert data["2024"]["target_column"] == "oegest_comb"


def test_validate_gestational_maps_raises_when_empty():
    cfg = Config({"data": {"gestational_age_var_by_year": {}, "lmp_gestation_var_by_year": {}}})
    with pytest.raises(ConfigurationError):
        confirm.validate_gestational_maps(cfg)


def test_validate_gestational_maps_passes_when_populated():
    cfg = Config({"data": {
        "gestational_age_var_by_year": {2024: "oegest_comb"},
        "lmp_gestation_var_by_year": {2024: "combgest"},
    }})
    confirm.validate_gestational_maps(cfg)  # should not raise
