"""Group fairness metrics with confidence intervals and significance testing (SPEC FR9).

Anchored on equal opportunity (gap in true-positive rate; a model that misses more real preterm
cases for one group is directly harmful) and calibration-by-group (in models.evaluate). Disparate
impact uses the EEOC four-fifths heuristic as a caveated reference point, not the headline.

Significance uses a stratified permutation null (group labels shuffled within the y=1 and within the
y=0 classes, Ojala and Garriga 2010) so the test isolates the disparity from base-rate differences.
P-values across the comparison family are corrected with Benjamini-Hochberg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _binarize(p: np.ndarray, threshold: float) -> np.ndarray:
    return (p >= threshold).astype(int)


def positive_rate(pred: np.ndarray) -> float:
    return float(pred.mean()) if len(pred) else float("nan")


def true_positive_rate(y: np.ndarray, pred: np.ndarray) -> float:
    pos = y == 1
    return float(pred[pos].mean()) if pos.any() else float("nan")


def _diff(metric_fn, y, pred, groups, group, ref):
    a = groups == group
    r = groups == ref
    return metric_fn(y[a], pred[a]) - metric_fn(y[r], pred[r])


def _pos_rate_m(y, pred):  # signature parity with tpr
    return positive_rate(pred)


def _tpr_m(y, pred):
    return true_positive_rate(y, pred)


def _stratified_permutation_pvalue(y, pred, groups, group, ref, metric_fn, observed, *,
                                   n: int, seed: int) -> float:
    """Two-sided p-value under a within-class label shuffle between ``group`` and ``ref``."""
    mask = (groups == group) | (groups == ref)
    ym, pm = y[mask], pred[mask]
    is_grp = (groups[mask] == group)
    rng = np.random.default_rng(seed)
    count = 0
    n_eff = 0
    for _ in range(n):
        shuffled = is_grp.copy()
        for cls in (0, 1):
            idx = np.flatnonzero(ym == cls)
            shuffled[idx] = rng.permutation(shuffled[idx])
        ga, gr = shuffled, ~shuffled
        if not ga.any() or not gr.any():
            continue
        stat = metric_fn(ym[ga], pm[ga]) - metric_fn(ym[gr], pm[gr])
        n_eff += 1
        if abs(stat) >= abs(observed):
            count += 1
    if n_eff == 0:
        return float("nan")
    return (count + 1) / (n_eff + 1)


def _bootstrap_diff_ci(y, pred, groups, group, ref, metric_fn, *, n: int, seed: int):
    a = np.flatnonzero(groups == group)
    r = np.flatnonzero(groups == ref)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        ai = rng.choice(a, len(a), replace=True)
        ri = rng.choice(r, len(r), replace=True)
        stats.append(metric_fn(y[ai], pred[ai]) - metric_fn(y[ri], pred[ri]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Return BH-adjusted p-values (same order as input). NaNs pass through."""
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return out.tolist()
    vals = p[idx]
    order = np.argsort(vals)
    m = len(vals)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, vals[i] * m / (rank + 1))
        adj[i] = prev
    out[idx] = adj
    return out.tolist()


@dataclass
class FairnessRow:
    group: str
    n: int
    positives: int
    demographic_parity_diff: float
    equal_opportunity_diff: float
    disparate_impact_ratio: float
    eo_ci: tuple[float, float]
    eo_pvalue: float
    eo_pvalue_bh: float = float("nan")
    note: str = ""


def fairness_table(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, *, ref_group: str, threshold: float = 0.5,
    suppress_floor: int = 50, n_boot: int = 1000, n_perm: int = 1000, seed: int = 42,
) -> list[FairnessRow]:
    """Per non-reference group: demographic parity, equal opportunity, disparate impact.

    Equal-opportunity differences carry a bootstrap CI and a stratified-permutation p-value, BH
    corrected across the group family. Groups below the suppression floor are reported without
    inferential statistics.
    """
    pred = _binarize(p, threshold)
    ref = groups == ref_group
    ref_pr = positive_rate(pred[ref])
    rows: list[FairnessRow] = []
    others = sorted({str(g) for g in groups if g is not None and str(g) not in {"nan", ref_group}})

    raw_p: list[float] = []
    for g in others:
        a = groups == g
        n_g, pos = int(a.sum()), int((y[a] == 1).sum())
        dp = positive_rate(pred[a]) - ref_pr
        eo = _diff(_tpr_m, y, pred, groups, g, ref_group)
        di = positive_rate(pred[a]) / ref_pr if ref_pr else float("nan")
        if pos < suppress_floor or (n_g - pos) < suppress_floor:
            rows.append(FairnessRow(g, n_g, pos, dp, eo, di, (float("nan"), float("nan")),
                                    float("nan"), note="suppressed"))
            raw_p.append(float("nan"))
            continue
        ci = _bootstrap_diff_ci(y, pred, groups, g, ref_group, _tpr_m, n=n_boot, seed=seed)
        pv = _stratified_permutation_pvalue(y, pred, groups, g, ref_group, _tpr_m, eo,
                                            n=n_perm, seed=seed)
        rows.append(FairnessRow(g, n_g, pos, dp, eo, di, ci, pv))
        raw_p.append(pv)

    for row, adj in zip(rows, benjamini_hochberg(raw_p)):
        row.eo_pvalue_bh = adj
    return rows
