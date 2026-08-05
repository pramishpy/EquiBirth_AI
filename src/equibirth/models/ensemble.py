"""Out-of-fold stacked logistic meta-learner over base-model probabilities (roadmap item 2).

Stacking discipline: the meta-learner must never see a base prediction that was produced by a
model trained on the same row (that is meta-leakage). Each base model is fit on dev_train, so its
predictions on dev_test and held_out are cleanly out-of-sample. This function additionally fits
the logistic stack with internal K-fold cross-fitting and reports the cross-fitted probabilities,
guarding the META fit itself. Logit-transformed inputs (standard for probability stacking) keep
the meta-model linear in log-odds space. The stack is deliberately simple: with base models this
correlated and signal this weak, anything fancier overfits.

CAVEAT for the Week-5 usage: the deep nets (TabNet, FT-Transformer) early-stop and isotonic-
recalibrate on dev_val, so THEIR dev_val columns are in-sample (the boosters' dev_val columns are
not, since boosters do not touch dev_val). When the stack is fit on dev_val, the internal K-fold
cannot fully undo that global contamination of the two net columns. In the Week-5 run this is
benign: the stack's held-out AUC-PR (0.171) came in at or below XGBoost's (0.172), so the
contaminated columns did not inflate the selected headline, and XGBoost, not the stack, was
selected. A stricter design would reserve a separate fold for net early-stopping/calibration
distinct from the stacking partition; we disclose the limitation rather than overstate the
guarantee.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _design(base: dict) -> tuple[list, np.ndarray]:
    names = sorted(base)
    return names, np.column_stack([_logit(base[k]) for k in names])


def oof_stack(base: dict, y: np.ndarray, *, n_folds: int = 5, seed: int = 42):
    """Cross-fitted logistic stack.

    ``base``: {model_name: probs} on a COMMON row set (e.g. dev_val). Returns
    ``(meta_model, oof_probs)`` where ``oof_probs[i]`` comes from the fold whose training
    excluded row i, and ``meta_model`` is refit on all rows for downstream scoring.
    """
    if len(base) < 2:
        raise ValueError("stacking needs at least two base models")
    y = np.asarray(y).astype(int)
    names, Z = _design(base)
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in skf.split(Z, y):
        m = LogisticRegression(max_iter=2000, random_state=seed).fit(Z[tr_idx], y[tr_idx])
        oof[te_idx] = m.predict_proba(Z[te_idx])[:, 1]
    meta = LogisticRegression(max_iter=2000, random_state=seed).fit(Z, y)
    meta.feature_names_ = names  # so weights are reportable by name
    return meta, oof


def predict_stack(meta, base: dict) -> np.ndarray:
    """Score the fitted stack on a new partition's base probabilities."""
    names, Z = _design(base)
    if getattr(meta, "feature_names_", names) != names:
        raise ValueError(f"base model set mismatch: expected {meta.feature_names_}, got {names}")
    return meta.predict_proba(Z)[:, 1]
