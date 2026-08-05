"""Tests for the fairness metrics, permutation test, and BH correction."""

import numpy as np

from equibirth.fairness import metrics as fm


def test_benjamini_hochberg_monotone_and_bounds():
    adj = fm.benjamini_hochberg([0.01, 0.02, 0.03, 0.04])
    assert all(0 <= a <= 1 for a in adj)
    # adjusted >= raw for BH
    assert adj[0] >= 0.01


def test_bh_passes_through_nan():
    adj = fm.benjamini_hochberg([0.01, float("nan"), 0.04])
    assert np.isnan(adj[1])
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


def test_equal_opportunity_difference_detects_real_gap():
    rng = np.random.default_rng(0)
    n = 8000
    groups = np.where(rng.uniform(size=n) < 0.5, "ref", "B")
    y = rng.integers(0, 2, n)
    # ref gets high scores when positive; B gets low scores when positive -> worse TPR for B
    p = rng.uniform(0, 0.5, n)
    ref_pos = (groups == "ref") & (y == 1)
    p[ref_pos] = rng.uniform(0.6, 1.0, ref_pos.sum())
    b_pos = (groups == "B") & (y == 1)
    p[b_pos] = rng.uniform(0.0, 0.4, b_pos.sum())

    rows = {r.group: r for r in fm.fairness_table(y, p, groups, ref_group="ref", threshold=0.5,
                                                  n_boot=200, n_perm=300, seed=1)}
    b = rows["B"]
    assert b.equal_opportunity_diff < -0.2          # B has much lower TPR
    assert b.eo_pvalue < 0.05                        # detectable
    assert b.eo_ci[1] < 0                            # CI entirely below 0


def test_no_gap_is_not_significant():
    rng = np.random.default_rng(2)
    n = 6000
    groups = np.where(rng.uniform(size=n) < 0.5, "ref", "B")
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)  # same distribution for both groups
    rows = {r.group: r for r in fm.fairness_table(y, p, groups, ref_group="ref", threshold=0.5,
                                                  n_boot=200, n_perm=300, seed=1)}
    assert rows["B"].eo_pvalue > 0.05


def test_small_group_is_suppressed():
    rng = np.random.default_rng(3)
    n = 5000
    groups = np.where(np.arange(n) < 30, "tiny", np.where(rng.uniform(size=n) < 0.5, "ref", "B"))
    y = rng.integers(0, 2, n)
    p = rng.uniform(0, 1, n)
    rows = {r.group: r for r in fm.fairness_table(y, p, groups, ref_group="ref",
                                                  suppress_floor=50, n_boot=100, n_perm=100)}
    assert rows["tiny"].note == "suppressed"
    assert np.isnan(rows["tiny"].eo_pvalue)
