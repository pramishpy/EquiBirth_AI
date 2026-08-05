"""Baseline models with calibration-first discipline (SPEC FR7, Phase 3).

Three baselines, each with balanced class weights (not resampling; Correction 3):
- logistic regression: the calibration reference and the honest performance floor;
- decision tree: an interpretable nonlinear baseline;
- random forest: a stronger nonlinear baseline.

Each model is wrapped in a median imputer and (for logistic regression) a scaler, so callers can pass
a raw feature matrix with missing values. When ``calibrate=True`` an isotonic recalibration is fit on
a held-aside fold (``val``) after the base model is fit on ``tr``; the returned probabilities for the
held-out rows (``ho``) are then well-calibrated. Calibration on a SEPARATE fold (not the training
fold) is what keeps the fairness audit's calibration-by-group trustworthy.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _make_estimators(seed: int) -> dict:
    """Construct the three baseline pipelines with balanced class weights."""
    return {
        "logistic_regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "decision_tree": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                           class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            # max_depth + a larger min_samples_leaf bound tree size so the forest stays memory-safe
            # on ~1M rows (unbounded trees on 1M rows exhaust RAM), with negligible metric cost here.
            # n_jobs is bounded (not -1): each worker copies the data, so -1 on a many-core box
            # multiplies memory and can OOM. 4 workers is a safe throughput/footprint tradeoff.
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=16, min_samples_leaf=50,
                                           n_jobs=4, class_weight="balanced", random_state=seed)),
        ]),
    }


def _prefit_calibrator(est):
    """Wrap an already-fitted estimator for isotonic calibration, across sklearn versions.

    sklearn >=1.6 prefers ``CalibratedClassifierCV(FrozenEstimator(est))``; older versions use
    ``cv="prefit"``. Prefer the frozen form when available to avoid the deprecation path.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(est), method="isotonic")
    except ImportError:  # pragma: no cover - older sklearn
        return CalibratedClassifierCV(est, method="isotonic", cv="prefit")


def _fit_one(est, X, y, tr, val, ho, *, calibrate: bool, seed: int) -> np.ndarray:
    """Fit ``est`` on ``tr``; if calibrating, isotonic-recalibrate on ``val``. Predict on ``ho``."""
    est.fit(X[tr], y[tr])
    if calibrate and val is not None and val.sum() > 0 and len(np.unique(y[val])) > 1:
        cal = _prefit_calibrator(est)
        cal.fit(X[val], y[val])
        return cal.predict_proba(X[ho])[:, 1]
    return est.predict_proba(X[ho])[:, 1]


def fit_predict(X: np.ndarray, y: np.ndarray, tr: np.ndarray, val: np.ndarray, ho: np.ndarray, *,
                calibrate: bool = True, seed: int = 42) -> dict[str, np.ndarray]:
    """Train each baseline on ``tr`` and return held-out probabilities per model.

    ``tr``, ``val``, ``ho`` are boolean masks over the rows of ``X``. ``val`` is the isotonic
    calibration fold (ignored when ``calibrate=False`` or empty). Returns a dict keyed by model name
    with a probability array aligned to ``ho``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    return {
        name: _fit_one(est, X, y, tr, val, ho, calibrate=calibrate, seed=seed)
        for name, est in _make_estimators(seed).items()
    }


def fit_predict_raw_and_calibrated(X: np.ndarray, y: np.ndarray, tr: np.ndarray, val: np.ndarray,
                                   ho: np.ndarray, *, seed: int = 42):
    """Fit each baseline ONCE and return both raw and calibrated held-out probabilities.

    Returns ``(raw, calibrated)``, each a dict keyed by model name. Fitting once (rather than the two
    separate ``fit_predict`` calls that raw-vs-calibrated comparison would otherwise need) halves the
    memory and time, which matters when the forest is trained on ~1M rows.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    raw: dict[str, np.ndarray] = {}
    calibrated: dict[str, np.ndarray] = {}
    can_calibrate = val is not None and np.asarray(val).sum() > 0 and len(np.unique(y[val])) > 1
    for name, est in _make_estimators(seed).items():
        est.fit(X[tr], y[tr])
        raw[name] = est.predict_proba(X[ho])[:, 1]
        if can_calibrate:
            cal = _prefit_calibrator(est)
            cal.fit(X[val], y[val])
            calibrated[name] = cal.predict_proba(X[ho])[:, 1]
        else:
            calibrated[name] = raw[name]
    return raw, calibrated
