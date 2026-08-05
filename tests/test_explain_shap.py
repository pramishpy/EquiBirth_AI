"""Tests for SHAP helpers (small tree model)."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from equibirth.explain import shap_analysis as sa


def _model_and_data():
    rng = np.random.default_rng(0)
    n = 800
    y = rng.integers(0, 2, n)
    x0 = y + rng.normal(0, 0.5, n)   # informative
    x1 = rng.normal(0, 1, n)         # noise
    X = np.column_stack([x0, x1])
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    return model, X, y


def test_mean_abs_importance_ranks_informative_first():
    model, X, y = _model_and_data()
    vals = sa.shap_values_for(model, X)
    ranking = sa.mean_abs_importance(vals, ["x0", "x1"])
    assert ranking[0][0] == "x0"          # informative feature ranks first
    assert ranking[0][1] >= ranking[1][1]


def test_top_features_by_group():
    model, X, y = _model_and_data()
    vals = sa.shap_values_for(model, X)
    groups = np.where(np.arange(len(y)) % 2 == 0, "A", "B")
    res = sa.top_features_by_group(vals, ["x0", "x1"], groups, which=["A", "B"], top_n=1)
    assert set(res) == {"A", "B"}
    assert res["A"][0][0] in {"x0", "x1"}
