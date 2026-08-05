"""Automated leakage guards (SPEC FR6.1 and FR6.3).

Two checks protect the project's central honesty claim:

- ``excluded_vars_detector_auc``: trains a trivial classifier on the EXCLUDED variables alone. If
  those post-delivery variables can predict the target well on their own (AUC above the gate), the
  exclusion list has a hole or a leaky column slipped through, and the build should fail.
- ``univariate_auc``: single-feature discrimination, used to gate timing-conditional features that
  are otherwise kept in the primary model.

Both encode Y/N birth-certificate flags to 1/0 and median-impute, so they run on the raw retained
columns without bespoke preprocessing.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from equibirth.ingest import schema

_SEED = 42


def _to_numeric(series: pl.Series) -> np.ndarray:
    """Coerce a column to float: Y->1, N->0, numbers kept, everything else null."""
    if series.dtype == pl.String:
        mapped = series.str.to_uppercase().replace_strict(
            {"Y": 1.0, "N": 0.0}, default=None, return_dtype=pl.Float64
        )
        return mapped.to_numpy().astype(float)
    return series.cast(pl.Float64, strict=False).to_numpy().astype(float)


def _matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Build a median-imputed float matrix from the present columns."""
    present = [c for c in cols if c in df.columns]
    arrs = []
    for c in present:
        x = _to_numeric(df[c])
        med = np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0
        x = np.where(np.isnan(x), med, x)
        arrs.append(x)
    if not arrs:
        raise ValueError("No usable columns to build the leakage matrix.")
    return np.column_stack(arrs)


def _fit_auc(X: np.ndarray, y: np.ndarray) -> float:
    """Train/test logistic regression and return held-out ROC AUC."""
    if len(np.unique(y)) < 2:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=_SEED, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba))


def _target(df: pl.DataFrame, target: str) -> tuple[pl.DataFrame, np.ndarray]:
    valid = df.filter(pl.col(target).is_not_null())
    return valid, valid[target].to_numpy().astype(int)


def excluded_vars_detector_auc(df: pl.DataFrame, target: str = "preterm") -> float:
    """AUC of a classifier trained on the leakage-detector variables alone.

    Returns 0.5 (no discriminability, hence no detectable leak) when none of the leakage-detector
    columns are present, e.g. when called on an already-cleaned feature matrix rather than the raw
    ingested frame.
    """
    valid, y = _target(df, target)
    cols = sorted(schema.LEAKAGE_DETECTOR_VARS)
    if not any(c in valid.columns for c in cols):
        return 0.5
    X = _matrix(valid, cols)
    return _fit_auc(X, y)


def assert_no_excluded_leakage(df: pl.DataFrame, gate: float = 0.55, target: str = "preterm") -> float:
    """Raise if the excluded-vars detector AUC exceeds ``gate`` (FR6.1 build gate)."""
    auc = excluded_vars_detector_auc(df, target=target)
    if auc > gate:
        raise AssertionError(
            f"Excluded-vars leakage detector AUC {auc:.3f} > gate {gate} (FR6.1): the exclusion list "
            "has a hole or a leaky column entered the matrix."
        )
    return auc


def univariate_auc(df: pl.DataFrame, col: str, target: str = "preterm") -> float:
    """Single-feature held-out AUC, used to gate timing-conditional features (FR6.3)."""
    valid, y = _target(df, target)
    X = _matrix(valid, [col])
    return _fit_auc(X, y)


def assert_no_component_is_excluded(feature_components: dict[str, tuple[str, ...]]) -> None:
    """FR6.2 guard: no engineered feature may be built from an excluded source variable."""
    excluded = schema.all_excluded_vars()
    for feature, components in feature_components.items():
        bad = set(components) & excluded
        if bad:
            raise AssertionError(f"Feature {feature!r} is built from excluded variable(s): {sorted(bad)}")
