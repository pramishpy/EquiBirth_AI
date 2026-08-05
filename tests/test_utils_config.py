"""Tests for config loading. Written before the implementation (TDD)."""

from pathlib import Path

import pytest

from equibirth.utils.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_config_returns_config_object():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert isinstance(cfg, Config)


def test_config_exposes_random_seed():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg["project"]["random_seed"] == 42


def test_config_get_supports_dotted_path():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.get("project.random_seed") == 42
    assert cfg.get("data.preterm_threshold_weeks") == 37


def test_config_get_returns_default_for_missing_key():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.get("does.not.exist", default="fallback") == "fallback"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(REPO_ROOT / "no_such_config.yaml")
