"""Fairlearn import smoke test and cross-check of our hand-rolled equal-opportunity metric.

SPEC FR9 named Fairlearn/Aequitas. We keep the transparent hand-rolled metrics as primary (DECISION_LOG
D33) and validate them against Fairlearn here, so the two implementations must agree.
"""

import numpy as np

from equibirth.fairness import metrics as fm


def test_fairlearn_imports_with_version_floor():
    import fairlearn

    major, minor = (int(x) for x in fairlearn.__version__.split(".")[:2])
    assert (major, minor) >= (0, 10)


def test_equal_opportunity_matches_fairlearn():
    from fairlearn.metrics import MetricFrame, true_positive_rate

    rng = np.random.default_rng(0)
    n = 4000
    groups = np.where(rng.uniform(size=n) < 0.5, "ref", "B")
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    p[(groups == "B") & (y == 1)] *= 0.5
    pred = (p >= 0.5).astype(int)

    # Fairlearn per-group TPR
    mf = MetricFrame(metrics=true_positive_rate, y_true=y, y_pred=pred, sensitive_features=groups)
    fl_gap = mf.by_group["B"] - mf.by_group["ref"]

    # Our hand-rolled equal-opportunity difference (B vs ref) at the same threshold
    rows = {r.group: r for r in fm.fairness_table(y, p, groups, ref_group="ref", threshold=0.5,
                                                  n_boot=50, n_perm=50, seed=1)}
    assert abs(rows["B"].equal_opportunity_diff - fl_gap) < 1e-9
