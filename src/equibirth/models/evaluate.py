"""Calibration-first evaluation utilities (SPEC FR8).

The project leads with AUC-PR and calibration, not accuracy, because the outcome is rare (~9%) and
the whole fairness argument rests on calibration-by-group being trustworthy. This module provides:

- discrimination: ``auc_roc`` (base-rate invariant, the cross-group comparator), ``auc_pr`` (primary);
- calibration: ``brier``, ``ece`` (uniform bins), ``ece_quantile`` (equal-mass bins, the stricter
  summary when predictions cluster near the base rate), ``mce`` (with a minimum-bin-size guard so a
  sparse bin cannot dominate), and ``calibration_curve`` for reliability plots;
- operating point: ``sensitivity_specificity`` at a threshold;
- uncertainty: ``bootstrap_ci`` (stratified resampling of a metric);
- reporting: ``stratified_report`` with tiered suppression (SPEC FR8.1) so small race groups are not
  silenced but are flagged as wide-CI or suppressed instead of reported as reliable.

All functions take numpy arrays of the held-out truth ``y`` (0/1) and predicted probability ``p``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auc_roc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC. Base-rate invariant, so it is the metric compared across race groups."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def auc_pr(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve). The primary metric under imbalance."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score (mean squared error of the probability). Lower is better; a proper scoring rule."""
    return float(brier_score_loss(np.asarray(y), np.asarray(p)))


def _bin_edges_uniform(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_edges_quantile(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-mass edges; deduplicated so clustered scores do not create empty bins."""
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(p, q))
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def _calibration_error(y, p, edges, *, worst: bool, min_bin_size: int) -> float:
    """Weighted mean (ECE) or guarded max (MCE) gap between predicted and observed rate per bin."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n = len(y)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    total, worst_gap = 0.0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0 or cnt < min_bin_size:
            continue
        gap = abs(p[m].mean() - y[m].mean())
        if worst:
            worst_gap = max(worst_gap, gap)
        else:
            total += (cnt / n) * gap
    return float(worst_gap if worst else total)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10, *, min_bin_size: int = 1) -> float:
    """Expected calibration error over ``n_bins`` equal-width probability bins."""
    return _calibration_error(y, p, _bin_edges_uniform(n_bins), worst=False, min_bin_size=min_bin_size)


def ece_quantile(y: np.ndarray, p: np.ndarray, n_bins: int = 10, *, min_bin_size: int = 1) -> float:
    """ECE over equal-MASS bins. Stricter when predictions pile up near the base rate."""
    return _calibration_error(y, p, _bin_edges_quantile(np.asarray(p, dtype=float), n_bins),
                              worst=False, min_bin_size=min_bin_size)


def mce(y: np.ndarray, p: np.ndarray, n_bins: int = 10, *, min_bin_size: int = 30) -> float:
    """Maximum calibration error, ignoring bins with fewer than ``min_bin_size`` rows.

    The guard stops a single sparse bin (a handful of high-probability rows) from inflating MCE into
    a binning artifact, which is what the raw MCE did in the sample-era analysis (DECISION_LOG D31).
    """
    return _calibration_error(y, p, _bin_edges_uniform(n_bins), worst=True, min_bin_size=min_bin_size)


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Return (mean_predicted, observed_rate, counts) per equal-width bin for a reliability diagram."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = _bin_edges_uniform(n_bins)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b]:
            mean_pred[b] = p[m].mean()
            observed[b] = y[m].mean()
    return mean_pred, observed, counts


def sensitivity_specificity(y: np.ndarray, p: np.ndarray, threshold: float = 0.5):
    """Return (sensitivity, specificity) at a decision ``threshold``."""
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    sens = float(pred[pos].mean()) if pos.any() else float("nan")
    spec = float((1 - pred[neg]).mean()) if neg.any() else float("nan")
    return sens, spec


def fraction_near_base_rate(p: np.ndarray, base_rate: float, window: float = 0.05) -> float:
    """Share of predictions within ``window`` of the base rate.

    A high value means the model rarely moves far from the prevalence, so low ECE reflects aggregate
    calibration rather than individual risk stratification (the honest qualification from D31).
    """
    p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(p - base_rate) <= window)) if len(p) else float("nan")


def bootstrap_ci(metric_fn, y: np.ndarray, p: np.ndarray, *, n: int = 2000, seed: int = 42,
                 alpha: float = 0.05):
    """Stratified bootstrap CI for a metric. Returns (point, lo, hi).

    Resampling is stratified by outcome class so each resample keeps the prevalence, which matters at
    the ~9% base rate. Degenerate resamples (a single class) are skipped.
    """
    y, p = np.asarray(y), np.asarray(p)
    point = metric_fn(y, p)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float(point), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        bi = np.concatenate([rng.choice(pos, len(pos), replace=True),
                             rng.choice(neg, len(neg), replace=True)])
        val = metric_fn(y[bi], p[bi])
        if val == val:  # skip NaN
            stats.append(val)
    if not stats:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


@dataclass
class GroupResult:
    group: str
    n: int
    positives: int
    rate: float
    status: str                       # "ok" | "wide_ci" | "suppressed"
    auc_roc: float | None = None
    auc_pr: float | None = None
    ece: float | None = None


def stratified_report(y: np.ndarray, p: np.ndarray, groups: np.ndarray, *,
                      suppress_floor: int = 50, full_ci_floor: int = 1000) -> list[GroupResult]:
    """Per-group metrics with tiered suppression (SPEC FR8.1).

    - fewer than ``suppress_floor`` positives (or negatives): suppressed (privacy + noise);
    - ``suppress_floor`` up to ``full_ci_floor`` positives: reported but flagged ``wide_ci``;
    - at least ``full_ci_floor`` positives: ``ok``.

    Small race groups are flagged, never silently dropped.
    """
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    out: list[GroupResult] = []
    for g in sorted({str(v) for v in groups if v is not None and str(v) != "nan"}):
        m = groups == g
        yy, pp = y[m], p[m]
        n_g = int(m.sum())
        pos = int((yy == 1).sum())
        neg = int((yy == 0).sum())
        rate = float(yy.mean()) if n_g else float("nan")
        if pos < suppress_floor or neg < suppress_floor:
            out.append(GroupResult(g, n_g, pos, rate, "suppressed"))
            continue
        status = "ok" if pos >= full_ci_floor else "wide_ci"
        out.append(GroupResult(g, n_g, pos, rate, status,
                               auc_roc=auc_roc(yy, pp), auc_pr=auc_pr(yy, pp), ece=ece(yy, pp)))
    return out
