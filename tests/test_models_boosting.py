"""Tests for the gradient-boosting tune/fit_predict wrappers (small/fast)."""

import numpy as np
import pytest

from equibirth.models import boosting
from equibirth.models import evaluate as ev


def _data(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.2).astype(int)   # imbalanced like preterm
    x0 = y + rng.normal(0, 0.7, n)
    x1 = rng.normal(0, 1, n)
    X = np.column_stack([x0, x1])
    idx = np.arange(n)
    tr = idx < int(0.6 * n)
    val = (idx >= int(0.6 * n)) & (idx < int(0.8 * n))
    ho = idx >= int(0.8 * n)
    return X, y, tr, val, ho


def test_scale_pos_weight():
    y = np.array([0, 0, 0, 1])
    assert boosting.scale_pos_weight(y) == 3.0


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_tune_then_fit_predict(name):
    X, y, tr, val, ho = _data()
    params = boosting.tune(name, X, y, tr, val, n_trials=3, timeout=120, seed=1)
    assert isinstance(params, dict) and "learning_rate" in params
    p = boosting.fit_predict(name, X, y, tr, val, ho, params, calibrate=True, seed=1)
    assert p.shape == (ho.sum(),)
    assert np.all((p >= 0) & (p <= 1))
    assert ev.auc_roc(y[ho], p) > 0.75   # informative feature -> better than chance
