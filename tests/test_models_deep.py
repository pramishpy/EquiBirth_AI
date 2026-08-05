"""Deep tabular wrappers: same fit/predict contract as boosting, deterministic, probs in [0,1]."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from equibirth.models import deep  # noqa: E402  (must follow the torch importorskip above)


def _toy(n=600, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    idx = np.arange(n)
    return X, y, idx < 300, (idx >= 300) & (idx < 450), idx >= 450


@pytest.mark.parametrize("name", ["tabnet", "ft_transformer"])
def test_fit_predict_returns_valid_probs(name):
    X, y, tr, val, ho = _toy()
    p = deep.fit_predict(name, X, y, tr, val, ho, max_epochs=3, seed=42)
    assert p.shape == (int(ho.sum()),)
    assert np.all((p >= 0) & (p <= 1))
    assert len(np.unique(np.round(p, 6))) > 1  # not degenerate/constant


def test_deterministic_across_runs():
    X, y, tr, val, ho = _toy()
    p1 = deep.fit_predict("tabnet", X, y, tr, val, ho, max_epochs=3, seed=42)
    p2 = deep.fit_predict("tabnet", X, y, tr, val, ho, max_epochs=3, seed=42)
    np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_unknown_name_raises():
    X, y, tr, val, ho = _toy(n=100)
    with pytest.raises(ValueError):
        deep.fit_predict("mlp9000", X, y, tr, val, ho)
