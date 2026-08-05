"""Bias-mitigation experiments (SPEC FR9.2): post-processing threshold optimization and reweighting.

Post-processing group-specific thresholds can equalize true-positive rate (equal opportunity) across
groups. This is transparent and effective, but it is also disparate TREATMENT (different decision
rules by race), which is in legal and ethical tension with disparate-impact goals. The runner reports
that tradeoff explicitly; nothing here decides it.
"""

from __future__ import annotations

import numpy as np


def threshold_for_tpr(y: np.ndarray, p: np.ndarray, target_tpr: float) -> float:
    """Smallest score threshold whose true-positive rate is at least ``target_tpr``.

    Lower thresholds flag more cases and raise TPR, so we scan positives' scores from high to low and
    take the cutoff where the cumulative share of positives caught reaches the target.
    """
    pos = p[y == 1]
    if pos.size == 0:
        return 0.5
    cutoffs = np.sort(pos)[::-1]
    k = int(np.ceil(target_tpr * len(cutoffs))) - 1
    k = min(max(k, 0), len(cutoffs) - 1)
    return float(cutoffs[k])


def equal_opportunity_thresholds(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, *, target_tpr: float,
    min_positives: int = 50,
) -> dict[str, float]:
    """Per-group thresholds that each achieve ``target_tpr`` (equalized opportunity).

    Groups with too few positives fall back to the global threshold rather than fitting noise.
    """
    global_thr = threshold_for_tpr(y, p, target_tpr)
    out: dict[str, float] = {}
    for g in sorted({str(x) for x in groups if x is not None and str(x) != "nan"}):
        m = groups == g
        if int((y[m] == 1).sum()) < min_positives:
            out[g] = global_thr
        else:
            out[g] = threshold_for_tpr(y[m], p[m], target_tpr)
    return out


def apply_group_thresholds(p: np.ndarray, groups: np.ndarray, thresholds: dict[str, float],
                           default: float = 0.5) -> np.ndarray:
    """Binarize predictions using a per-group threshold map."""
    thr = np.array([thresholds.get(str(g), default) for g in groups])
    return (p >= thr).astype(int)


def reweight_sample_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Pre-processing reweighting: weight so each group x outcome cell has equal total mass.

    Upweights underrepresented (group, outcome) combinations, e.g. preterm births in smaller groups,
    so the model "sees" them more. Returns a weight per row, mean-normalized to 1.
    """
    w = np.ones(len(y), dtype=float)
    labels = np.array([str(g) for g in groups])
    n = len(y)
    for g in np.unique(labels):
        for cls in (0, 1):
            cell = (labels == g) & (y == cls)
            c = int(cell.sum())
            if c:
                w[cell] = n / (len(np.unique(labels)) * 2 * c)
    return w / w.mean()


def expgrad_predictions(X_fit: np.ndarray, y_fit: np.ndarray, g_fit: np.ndarray,
                        X_score: np.ndarray, g_score: np.ndarray, *,
                        constraint: str = "equalized_odds", seed: int = 42,
                        max_iter: int = 30):
    """In-processing mitigation via Fairlearn ExponentiatedGradient reductions.

    Fits a logistic base learner under a fairness constraint ("equalized_odds" or
    "true_positive_rate_parity" for equal opportunity). Returns ``(baseline_pred, mitigated_pred)``
    hard 0/1 predictions on the score partition, so callers can price the tradeoff. The reduction
    yields a randomized classifier; we report its realized decisions (seeded, reproducible).
    Logistic base learner keeps the reduction convex and fast; the point is the constraint's
    cost, not squeezing the last AUC point.

    ``g_score`` is accepted for a symmetric fit/score signature but is intentionally unused: the
    fitted ExponentiatedGradient randomized classifier predicts from features alone and needs no
    sensitive attribute at scoring time (unlike ThresholdOptimizer, which does).
    """
    del g_score  # documented above: not needed at scoring time for this mitigator
    from fairlearn.reductions import (EqualizedOdds, ExponentiatedGradient,
                                      TruePositiveRateParity)
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def _logit_pipeline():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed))

    class _WeightedLogit(BaseEstimator, ClassifierMixin):
        """Impute+scale+logistic pipeline that forwards ``sample_weight`` to the logistic step.

        ExponentiatedGradient reweights the training data and calls ``fit(X, y, sample_weight=w)``;
        a bare sklearn Pipeline rejects that keyword, so this wrapper routes it explicitly.
        """
        def fit(self, X, y, sample_weight=None):
            self.pipe_ = _logit_pipeline()
            self.pipe_.fit(X, y, logisticregression__sample_weight=sample_weight)
            self.classes_ = self.pipe_.classes_
            return self

        def predict(self, X):
            return self.pipe_.predict(X)

    cons = {"equalized_odds": EqualizedOdds,
            "true_positive_rate_parity": TruePositiveRateParity}[constraint]()
    base = _logit_pipeline()
    base.fit(X_fit, y_fit)
    baseline_pred = base.predict(X_score)

    mitigator = ExponentiatedGradient(_WeightedLogit(), constraints=cons, max_iter=max_iter)
    np.random.seed(seed)  # ExponentiatedGradient randomizes over predictors via np.random
    mitigator.fit(X_fit, y_fit, sensitive_features=np.asarray(g_fit, dtype=object))
    mitigated_pred = np.asarray(mitigator.predict(X_score)).astype(int)
    return baseline_pred.astype(int), mitigated_pred


def threshold_optimizer_predictions(y_fit: np.ndarray, p_fit: np.ndarray, g_fit: np.ndarray,
                                    p_score: np.ndarray, g_score: np.ndarray, *,
                                    constraint: str = "true_positive_rate_parity",
                                    seed: int = 42) -> np.ndarray:
    """Post-processing mitigation via Fairlearn ThresholdOptimizer on precomputed scores.

    Learns group-specific thresholds on the fit partition and applies them to the score
    partition. This IS disparate treatment (different rules by group); it is reported as an
    analytical option and never exposed in any deployed artifact.
    """
    from fairlearn.postprocessing import ThresholdOptimizer
    from sklearn.base import BaseEstimator, ClassifierMixin

    class _Precomputed(BaseEstimator, ClassifierMixin):
        """Identity estimator: 'features' are the precomputed probability column."""
        def fit(self, X, y):
            self.classes_ = np.array([0, 1])
            return self
        def predict_proba(self, X):
            p = np.asarray(X).ravel()
            return np.column_stack([1 - p, p])

    opt = ThresholdOptimizer(estimator=_Precomputed().fit(p_fit.reshape(-1, 1), y_fit),
                             constraints=constraint, objective="balanced_accuracy_score",
                             prefit=True, predict_method="predict_proba")
    opt.fit(p_fit.reshape(-1, 1), y_fit, sensitive_features=np.asarray(g_fit, dtype=object))
    return np.asarray(opt.predict(p_score.reshape(-1, 1),
                                  sensitive_features=np.asarray(g_score, dtype=object),
                                  random_state=seed)).astype(int)
