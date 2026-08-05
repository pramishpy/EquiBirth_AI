"""Decompose the Black-White preterm gap (SPEC FR10, research question 4).

Uses g-computation (standardization) over a logistic model: fit preterm on race plus measured
covariates, then compare the counterfactual mean risk if everyone were assigned the focal race versus
the reference race, holding each person's covariates fixed. The difference is the covariate-adjusted
("unexplained") gap; the raw observed gap minus that is the share statistically associated with the
measured covariates.

Heavy causal caveat (must accompany any use): the measured covariates (education, payment, WIC, etc.)
are themselves products of structural racism, so the "explained" share is inflated and the
"unexplained" residual is a LOWER bound on structural contributions. This is association, not proof of
mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class Decomposition:
    raw_gap: float
    adjusted_gap: float
    explained_abs: float
    explained_pct: float
    n_focal: int
    n_ref: int


def decompose_gap(
    X_cov: np.ndarray, y: np.ndarray, is_focal: np.ndarray, is_ref: np.ndarray, *, seed: int = 42,
) -> Decomposition:
    """g-computation decomposition.

    ``X_cov`` are the measured covariates (NOT including race). ``is_focal`` / ``is_ref`` are boolean
    group masks (e.g. NH Black, NH White). Returns raw vs adjusted gap and the explained share.
    """
    mask = is_focal | is_ref
    Xc = X_cov[mask]
    yy = y[mask]
    race = is_focal[mask].astype(float)  # 1 = focal, 0 = reference

    raw_gap = float(yy[race == 1].mean() - yy[race == 0].mean())

    imp = SimpleImputer(strategy="median").fit(Xc)
    Xi = imp.transform(Xc)
    sc = StandardScaler().fit(Xi)
    Xs = sc.transform(Xi)
    design = np.column_stack([race, Xs])

    model = LogisticRegression(max_iter=2000, random_state=seed).fit(design, yy)

    # counterfactual standardization over the COMBINED covariate distribution
    as_focal = np.column_stack([np.ones(len(Xs)), Xs])
    as_ref = np.column_stack([np.zeros(len(Xs)), Xs])
    adjusted_gap = float(model.predict_proba(as_focal)[:, 1].mean()
                         - model.predict_proba(as_ref)[:, 1].mean())

    explained_abs = raw_gap - adjusted_gap
    explained_pct = (explained_abs / raw_gap) if raw_gap else float("nan")
    return Decomposition(raw_gap, adjusted_gap, explained_abs, explained_pct,
                         int(is_focal[mask].sum()), int(is_ref[mask].sum()))
