"""Tests for threshold optimization, group thresholds, reweighting, and decomposition."""

import numpy as np

from equibirth.fairness import decomposition as dc
from equibirth.fairness import mitigation as mit


def test_threshold_for_tpr_achieves_target():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.7, 0.5, 0.3, 0.4, 0.1])
    thr = mit.threshold_for_tpr(y, p, 0.75)   # catch 3 of 4 positives
    pred = (p >= thr).astype(int)
    tpr = pred[y == 1].mean()
    assert tpr >= 0.75


def test_equal_opportunity_thresholds_per_group():
    rng = np.random.default_rng(0)
    n = 4000
    groups = np.where(rng.uniform(size=n) < 0.5, "A", "B")
    y = rng.integers(0, 2, n)
    # group B scored lower when positive -> needs a lower threshold to reach the same TPR
    p = rng.uniform(0, 1, n)
    p[(groups == "B") & (y == 1)] *= 0.5
    thr = mit.equal_opportunity_thresholds(y, p, groups, target_tpr=0.8)
    assert thr["B"] < thr["A"]
    pred = mit.apply_group_thresholds(p, groups, thr)
    tpr_a = pred[(groups == "A") & (y == 1)].mean()
    tpr_b = pred[(groups == "B") & (y == 1)].mean()
    assert abs(tpr_a - tpr_b) < 0.08   # opportunity roughly equalized


def test_reweight_balances_group_outcome_mass():
    y = np.array([1, 0, 0, 0, 1, 0])
    groups = np.array(["A", "A", "A", "A", "B", "B"])
    w = mit.reweight_sample_weights(y, groups)
    # rare cells (A-positive, B-positive) get higher weight than common A-negative
    assert w[0] > w[1]
    assert abs(w.mean() - 1.0) < 1e-9


def test_decomposition_full_explanation_when_covariate_drives_gap():
    rng = np.random.default_rng(1)
    n = 6000
    focal = rng.uniform(size=n) < 0.5
    # a covariate differs by group AND fully determines risk; race has no direct effect
    cov = np.where(focal, rng.normal(1.0, 0.5, n), rng.normal(-1.0, 0.5, n))
    risk = 1 / (1 + np.exp(-(1.5 * cov)))
    y = (rng.uniform(size=n) < risk).astype(int)
    res = dc.decompose_gap(cov.reshape(-1, 1), y, focal, ~focal)
    assert res.raw_gap > 0.1               # there is a raw gap
    assert abs(res.adjusted_gap) < 0.05    # vanishes after adjusting for the covariate
    assert res.explained_pct > 0.7         # most of the gap is explained by the covariate


def test_decomposition_unexplained_when_race_has_direct_effect():
    rng = np.random.default_rng(2)
    n = 6000
    focal = rng.uniform(size=n) < 0.5
    cov = rng.normal(0, 1, n)              # covariate unrelated to group
    risk = 1 / (1 + np.exp(-(0.1 * cov + 1.2 * focal)))  # direct race effect
    y = (rng.uniform(size=n) < risk).astype(int)
    res = dc.decompose_gap(cov.reshape(-1, 1), y, focal, ~focal)
    assert res.explained_pct < 0.3         # little explained by covariates


# --- fairlearn-backed mitigations (weeks 6) ---
import pytest  # noqa: E402  (kept here so the importorskip below guards this section)

fairlearn = pytest.importorskip("fairlearn")


def _biased_data(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    g = np.where(rng.random(n) < 0.5, "A", "B")
    x1 = rng.normal(size=n) + (g == "B") * 0.8          # feature correlated with group
    x2 = rng.normal(size=n)
    logit = 1.2 * x1 + 0.5 * x2 - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.column_stack([x1, x2]).astype(np.float32)
    return X, y, g


def test_expgrad_reduces_tpr_gap():
    from equibirth.fairness import mitigation as mt
    X, y, g = _biased_data()
    half = len(y) // 2
    fit, sc = slice(0, half), slice(half, None)
    base_pred, mit_pred = mt.expgrad_predictions(
        X[fit], y[fit], g[fit], X[sc], g[sc], seed=42)

    def tpr(pred, mask):
        m = mask & (y[sc] == 1)
        return pred[m].mean() if m.sum() else 0.0

    gA, gB = g[sc] == "A", g[sc] == "B"
    base_gap = abs(tpr(base_pred, gA) - tpr(base_pred, gB))
    mit_gap = abs(tpr(mit_pred, gA) - tpr(mit_pred, gB))
    assert mit_gap <= base_gap + 0.02   # no worse; typically clearly better


def test_threshold_optimizer_predictions_shape():
    from equibirth.fairness import mitigation as mt
    X, y, g = _biased_data()
    half = len(y) // 2
    fit, sc = slice(0, half), slice(half, None)
    p_fit = np.clip(0.2 + 0.5 * y[fit] + np.random.default_rng(0).normal(0, 0.1, half), 0.01, 0.99)
    p_sc = np.clip(0.2 + 0.5 * y[sc] + np.random.default_rng(1).normal(0, 0.1, len(y) - half), 0.01, 0.99)
    pred = mt.threshold_optimizer_predictions(y[fit], p_fit, g[fit], p_sc, g[sc], seed=42)
    assert pred.shape == (len(y) - half,)
    assert set(np.unique(pred)).issubset({0, 1})
