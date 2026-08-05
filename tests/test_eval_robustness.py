"""Out-of-time split masks: disjoint, year-correct; drift table shape sane."""
import numpy as np

from equibirth.eval import robustness


def test_out_of_time_masks_disjoint_and_correct():
    years = np.array([2018, 2019, 2020, 2021, 2022, 2023, 2024] * 10)
    m = robustness.out_of_time_masks(years)
    assert set(np.unique(years[m["train_precovid"]])) == {2018, 2019}
    assert set(np.unique(years[m["test_covid"]])) == {2020, 2021}
    assert set(np.unique(years[m["test_recent"]])) == {2022, 2023, 2024}
    stacked = m["train_precovid"].astype(int) + m["test_covid"].astype(int) + m["test_recent"].astype(int)
    assert stacked.max() == 1  # disjoint


def test_yearly_drift_table():
    rng = np.random.default_rng(0)
    years = np.repeat([2018, 2019, 2020], 200)
    y = (rng.random(600) < 0.1).astype(int)
    p = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.2, 600), 0.001, 0.999)
    rows = robustness.yearly_drift(y, p, years)
    assert [r["year"] for r in rows] == [2018, 2019, 2020]
    for r in rows:
        assert set(r) == {"year", "n", "prevalence", "auc_roc", "auc_pr", "ece"}
        assert 0 <= r["prevalence"] <= 1


def test_yearly_drift_skips_degenerate_years():
    years = np.array([2018] * 50 + [2019] * 50)
    y = np.array([0] * 50 + [1] * 25 + [0] * 25)   # 2018 has no positives
    p = np.linspace(0.01, 0.99, 100)
    rows = robustness.yearly_drift(y, p, years)
    y18 = [r for r in rows if r["year"] == 2018][0]
    assert np.isnan(y18["auc_roc"])
