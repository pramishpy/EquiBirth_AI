"""SHAP explainability for the tree model (SPEC FR10).

Global feature importance plus race-stratified rankings: if the top predictors differ across groups
(for example previous preterm birth dominating for one group while access factors dominate for
another), that is itself a finding about where risk is concentrated. Interpretation is associational.
"""

from __future__ import annotations

import numpy as np


def shap_values_for(model, X: np.ndarray) -> np.ndarray:
    """Return a (n_samples, n_features) SHAP value matrix for a fitted tree model."""
    import shap

    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    if isinstance(vals, list):           # older API: per-class list
        vals = vals[1] if len(vals) > 1 else vals[0]
    vals = np.asarray(vals)
    if vals.ndim == 3:                   # newer API: (n_samples, n_features, n_classes)
        vals = vals[:, :, -1]            # positive-class contributions
    return vals


def mean_abs_importance(shap_vals: np.ndarray, feature_names: list[str]) -> list[tuple[str, float]]:
    """Rank features by mean absolute SHAP value (descending)."""
    imp = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(imp)[::-1]
    return [(feature_names[i], float(imp[i])) for i in order]


def top_features_by_group(
    shap_vals: np.ndarray, feature_names: list[str], groups: np.ndarray, *,
    which: list[str], top_n: int = 8,
) -> dict[str, list[tuple[str, float]]]:
    """Return the top-N features (by mean |SHAP|) within each requested group."""
    out: dict[str, list[tuple[str, float]]] = {}
    for g in which:
        m = groups == g
        if m.sum() == 0:
            continue
        out[g] = mean_abs_importance(shap_vals[m], feature_names)[:top_n]
    return out
