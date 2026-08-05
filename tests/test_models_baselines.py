"""Tests for the baseline model training/prediction wrapper."""

import numpy as np

from equibirth.models import baselines
from equibirth.models import evaluate as ev


def _separable(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # two informative features correlated with y, plus noise and some missing values
    x0 = y + rng.normal(0, 0.8, n)
    x1 = y * 0.5 + rng.normal(0, 1.0, n)
    x2 = rng.normal(0, 1, n)
    X = np.column_stack([x0, x1, x2])
    X[rng.uniform(0, 1, n) < 0.05, 0] = np.nan  # missingness handled by imputer
    return X, y


def test_fit_predict_returns_probs_for_all_models():
    X, y = _separable()
    n = len(y)
    idx = np.arange(n)
    tr = idx < int(0.6 * n)
    val = (idx >= int(0.6 * n)) & (idx < int(0.75 * n))
    ho = idx >= int(0.75 * n)
    preds = baselines.fit_predict(X, y, tr, val, ho, calibrate=True, seed=1)
    assert set(preds) == {"logistic_regression", "decision_tree", "random_forest"}
    for name, p in preds.items():
        assert p.shape == (ho.sum(),)
        assert np.all((p >= 0) & (p <= 1))
        # informative features -> better than chance
        assert ev.auc_roc(y[ho], p) > 0.7, name


def test_calibration_off_still_works():
    X, y = _separable()
    n = len(y)
    idx = np.arange(n)
    tr = idx < int(0.7 * n)
    val = np.zeros(n, dtype=bool)
    ho = idx >= int(0.7 * n)
    preds = baselines.fit_predict(X, y, tr, val, ho, calibrate=False, seed=1)
    assert np.all((preds["logistic_regression"] >= 0) & (preds["logistic_regression"] <= 1))


def test_fit_predict_raw_and_calibrated_single_fit():
    """One base fit per model returns BOTH raw and calibrated held-out probabilities."""
    X, y = _separable()
    n = len(y)
    idx = np.arange(n)
    tr = idx < int(0.6 * n)
    val = (idx >= int(0.6 * n)) & (idx < int(0.75 * n))
    ho = idx >= int(0.75 * n)
    raw, cal = baselines.fit_predict_raw_and_calibrated(X, y, tr, val, ho, seed=1)
    assert set(raw) == set(cal) == {"logistic_regression", "decision_tree", "random_forest"}
    for name in raw:
        assert raw[name].shape == cal[name].shape == (ho.sum(),)
        assert np.all((raw[name] >= 0) & (raw[name] <= 1))
        assert np.all((cal[name] >= 0) & (cal[name] <= 1))
        assert ev.auc_roc(y[ho], cal[name]) > 0.7, name
