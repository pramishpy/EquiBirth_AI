"""Gradient-boosted trees with Optuna tuning and post-hoc calibration (SPEC FR7, Phase 4).

XGBoost and LightGBM are the expected top performers on this tabular data (Correction 4). Both handle
missing values natively (no imputer needed) and use ``scale_pos_weight`` for the ~9% imbalance rather
than resampling (Correction 3), which protects the calibration the fairness audit depends on.

- ``scale_pos_weight``: negative/positive ratio, the standard imbalance weight for boosters.
- ``tune``: an Optuna study maximizing dev-fold AUC-PR (the primary metric), with a trial and
  wall-clock cap so it is safe in CI and on a laptop.
- ``fit_calibrated``: fit on train, then isotonic-recalibrate on a separate fold; returns the fitted
  calibrated estimator (used by the fairness script, which needs to score several partitions).
- ``fit_predict``: tune-free convenience that fits with given params and returns held-out probabilities.
"""

from __future__ import annotations

import numpy as np

_CAL_IMPORT_ERROR = "install xgboost/lightgbm (and libomp on macOS) to use the boosting module"


def scale_pos_weight(y: np.ndarray) -> float:
    """Return negative/positive ratio for class-weighted boosting. 1.0 if no positives."""
    y = np.asarray(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / pos) if pos else 1.0


def _make_estimator(name: str, params: dict, spw: float, seed: int):
    """Construct an unfitted booster of the given family with sensible fixed defaults."""
    name = name.lower()
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            min_child_weight=params.get("min_child_weight", 1.0),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=spw,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            max_depth=params.get("max_depth", -1),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            min_child_samples=params.get("min_child_samples", 20),
            reg_lambda=params.get("reg_lambda", 1.0),
            scale_pos_weight=spw,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"Unknown booster {name!r}; expected 'xgboost' or 'lightgbm'.")


def _search_space(name: str, trial) -> dict:
    """Optuna search space per booster family."""
    common = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if name.lower() == "xgboost":
        common["max_depth"] = trial.suggest_int("max_depth", 3, 8)
        common["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 10.0)
    else:
        common["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
        common["min_child_samples"] = trial.suggest_int("min_child_samples", 10, 100)
    return common


def tune(name: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, val: np.ndarray, *,
         n_trials: int = 50, timeout: int | None = 3600, seed: int = 42) -> dict:
    """Optuna-tune a booster on ``tr``, scoring AUC-PR on ``val``. Returns the best params dict.

    Tuning uses the dev validation fold only; the held-out set is never touched here (no tuning
    leakage). ``timeout`` (seconds) and ``n_trials`` bound the search.
    """
    import optuna
    from sklearn.metrics import average_precision_score

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    spw = scale_pos_weight(y[tr])
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = _search_space(name, trial)
        est = _make_estimator(name, params, spw, seed)
        est.fit(X[tr], y[tr])
        p = est.predict_proba(X[val])[:, 1]
        return average_precision_score(y[val], p)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return dict(study.best_params)


def fit_calibrated(name: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, cal: np.ndarray,
                   params: dict, *, seed: int = 42):
    """Fit a booster on ``tr`` and isotonic-recalibrate on ``cal``. Returns the calibrated estimator."""
    from sklearn.calibration import CalibratedClassifierCV

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    est = _make_estimator(name, params, scale_pos_weight(y[tr]), seed)
    est.fit(X[tr], y[tr])
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(est), method="isotonic")
    except ImportError:  # pragma: no cover - older sklearn
        calibrated = CalibratedClassifierCV(est, method="isotonic", cv="prefit")
    calibrated.fit(X[cal], y[cal])
    return calibrated


def fit_predict(name: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray, val: np.ndarray,
                ho: np.ndarray, params: dict, *, calibrate: bool = True, seed: int = 42) -> np.ndarray:
    """Fit a booster (optionally calibrated on ``val``) and return held-out probabilities on ``ho``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if calibrate and val is not None and np.asarray(val).sum() > 0 and len(np.unique(y[val])) > 1:
        est = fit_calibrated(name, X, y, tr, val, params, seed=seed)
        return est.predict_proba(X[ho])[:, 1]
    est = _make_estimator(name, params, scale_pos_weight(y[tr]), seed)
    est.fit(X[tr], y[tr])
    return est.predict_proba(X[ho])[:, 1]
