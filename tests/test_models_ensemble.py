"""Out-of-fold stacking: meta-learner recovers a dominant base model, stays calibrated."""
import numpy as np
import pytest

from equibirth.models import ensemble


def _preds(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    good = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, n), 0.001, 0.999)   # informative
    noise = np.clip(rng.random(n), 0.001, 0.999)                            # useless
    return y, {"good": good, "noise": noise}


def test_stack_weights_favor_informative_model():
    y, base = _preds()
    meta, p = ensemble.oof_stack(base, y, n_folds=5, seed=42)
    w = dict(zip(sorted(base), meta.coef_[0]))
    assert w["good"] > w["noise"]
    assert p.shape == (len(y),)
    assert np.all((p >= 0) & (p <= 1))


def test_stack_beats_or_matches_worst_base():
    from sklearn.metrics import roc_auc_score
    y, base = _preds()
    _, p = ensemble.oof_stack(base, y, n_folds=5, seed=42)
    assert roc_auc_score(y, p) > roc_auc_score(y, base["noise"])


def test_predict_stack_on_new_data():
    y, base = _preds()
    meta, _ = ensemble.oof_stack(base, y, n_folds=5, seed=42)
    y2, base2 = _preds(seed=2)
    p2 = ensemble.predict_stack(meta, base2)
    assert p2.shape == (len(y2),)
    assert np.all((p2 >= 0) & (p2 <= 1))


def test_requires_at_least_two_models():
    y, base = _preds()
    with pytest.raises(ValueError):
        ensemble.oof_stack({"only": base["good"]}, y)
