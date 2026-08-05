"""Out-of-time validation and drift summaries (roadmap item 7).

A model trained on pre-COVID births faces 2020-2021 as a natural distribution shock (care
utilization changed; Week 1 documented per-race preterm-rate deltas). Splitting time this way
answers the deployment question a random split cannot: does performance and calibration survive
a regime change? Era boundaries are fixed by the pandemic timeline, not tuned.
"""

from __future__ import annotations

import numpy as np

PRECOVID_YEARS = (2018, 2019)
COVID_YEARS = (2020, 2021)
RECENT_YEARS = (2022, 2023, 2024)


def out_of_time_masks(years: np.ndarray) -> dict:
    """Boolean era masks keyed on birth year. Disjoint by construction."""
    years = np.asarray(years)
    return {
        "train_precovid": np.isin(years, PRECOVID_YEARS),
        "test_covid": np.isin(years, COVID_YEARS),
        "test_recent": np.isin(years, RECENT_YEARS),
    }


def yearly_drift(y: np.ndarray, p: np.ndarray, years: np.ndarray) -> list:
    """Per-year n, prevalence, AUC-ROC, AUC-PR, ECE. Degenerate years get NaN AUCs."""
    from equibirth.models import evaluate as ev

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    years = np.asarray(years)
    rows = []
    for yr in sorted(np.unique(years).tolist()):
        m = years == yr
        yy, pp = y[m], p[m]
        degenerate = len(np.unique(yy)) < 2
        rows.append({
            "year": int(yr), "n": int(m.sum()), "prevalence": float(yy.mean()),
            "auc_roc": float("nan") if degenerate else ev.auc_roc(yy, pp),
            "auc_pr": float("nan") if degenerate else ev.auc_pr(yy, pp),
            "ece": ev.ece(yy, pp),
        })
    return rows
