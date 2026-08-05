"""Tests for the FR6 leakage guards."""

import numpy as np
import polars as pl
import pytest

from equibirth.features import leakage
from equibirth.ingest.schema import MEDICAL_RISK_COMPOSITE_COMPONENTS


def _frame_with_leaky_signal(n=600):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    # dbwt strongly anti-correlated with preterm (leaky); ld_ster given when preterm anticipated.
    dbwt = np.where(y == 1, rng.normal(2200, 200, n), rng.normal(3400, 200, n))
    ld_ster = np.where(y == 1, "Y", "N")
    return pl.DataFrame({"preterm": y, "dbwt": dbwt, "ld_ster": ld_ster})


def test_detector_flags_strong_leakage():
    df = _frame_with_leaky_signal()
    auc = leakage.excluded_vars_detector_auc(df)
    assert auc > 0.55
    with pytest.raises(AssertionError):
        leakage.assert_no_excluded_leakage(df, gate=0.55)


def test_detector_passes_on_noise():
    rng = np.random.default_rng(1)
    n = 600
    df = pl.DataFrame({
        "preterm": rng.integers(0, 2, n),
        "dbwt": rng.normal(3300, 300, n),   # unrelated to target
        "ld_ster": rng.choice(["Y", "N"], n),
    })
    auc = leakage.assert_no_excluded_leakage(df, gate=0.65)
    assert 0.4 < auc < 0.65


def test_component_guard_rejects_excluded_source():
    with pytest.raises(AssertionError):
        leakage.assert_no_component_is_excluded({"bad_feature": ("rf_pdiab", "dbwt")})


def test_component_guard_allows_clean_composite():
    leakage.assert_no_component_is_excluded({"medical_risk_count": MEDICAL_RISK_COMPOSITE_COMPONENTS})
