"""Tests for the calibration-first evaluation utilities."""

import numpy as np

from equibirth.models import evaluate as ev


def _rng():
    return np.random.default_rng(0)


def test_auc_roc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert ev.auc_roc(y, p) == 1.0


def test_ece_near_zero_for_well_calibrated():
    rng = _rng()
    n = 20000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)  # perfectly calibrated by construction
    assert ev.ece(y, p, n_bins=10) < 0.02


def test_ece_large_for_miscalibrated():
    rng = _rng()
    n = 20000
    p = np.full(n, 0.9)            # always predicts 0.9
    y = (rng.uniform(0, 1, n) < 0.1).astype(int)  # true rate 0.1
    assert ev.ece(y, p, n_bins=10) > 0.5


def test_sensitivity_specificity_basic():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.2, 0.8])
    sens, spec = ev.sensitivity_specificity(y, p, threshold=0.5)
    assert sens == 0.5   # 1 of 2 positives caught
    assert spec == 0.5   # 1 of 2 negatives correct


def test_bootstrap_ci_brackets_point_and_orders():
    rng = _rng()
    n = 4000
    y = rng.integers(0, 2, n)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)
    point, lo, hi = ev.bootstrap_ci(ev.auc_roc, y, p, n=300, seed=1)
    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_stratified_report_tiers():
    rng = _rng()
    n = 5000
    groups = np.where(np.arange(n) < 4900, "big", "tiny")  # tiny has 100 rows
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    res = {r.group: r for r in ev.stratified_report(y, p, groups, full_ci_floor=1000)}
    assert res["big"].status == "ok"
    assert res["big"].auc_roc is not None
    # tiny: ~50 positives -> below full_ci_floor; either wide_ci or suppressed, never "ok"
    assert res["tiny"].status in {"wide_ci", "suppressed"}


def test_mce_min_bin_guard_ignores_sparse_bins():
    # one well-populated bin near-perfect, one 1-sample bin with a big gap -> guard ignores the sparse bin
    # build a case where sparse bin would inflate MCE: 1 row predicted 0.95 but observed 0
    y2 = np.array([0] * 100 + [0])
    p2 = np.array([0.10] * 100 + [0.95])
    assert ev.mce(y2, p2, n_bins=10, min_bin_size=30) < 0.2     # sparse 0.95 bin ignored
    assert ev.mce(y2, p2, n_bins=10, min_bin_size=1) > 0.5      # without guard, dominated by it


def test_ece_quantile_higher_when_clustered():
    rng = _rng()
    n = 20000
    # predictions clustered near base rate but a miscalibrated tail
    p = np.concatenate([rng.uniform(0.05, 0.10, n - 200), rng.uniform(0.8, 0.95, 200)])
    y = (rng.uniform(0, 1, len(p)) < 0.09).astype(int)  # true rate ~0.09 regardless -> tail miscalibrated
    assert ev.ece_quantile(y, p, n_bins=10) >= ev.ece(y, p, n_bins=10) - 1e-9


def test_fraction_near_base_rate():
    p = np.array([0.09, 0.10, 0.50, 0.92])
    assert ev.fraction_near_base_rate(p, base_rate=0.09, window=0.05) == 0.5


def test_calibration_curve_shapes():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3])
    mp, obs, counts = ev.calibration_curve(y, p, n_bins=5)
    assert len(mp) == len(obs) == len(counts) == 5
    assert counts.sum() == 6
