"""Smoke tests for the EDA plotting helpers (headless Agg backend)."""

import polars as pl

from equibirth.viz import eda


def _toy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "preterm": [1, 0, 0, 1, 0, None],
            "mother_race_ethnicity": ["NH Black", "NH White", "NH White", "NH Black", "Hispanic", "NH White"],
            "meduc": [3, 6, 6, 2, 4, 6],
        }
    )


def test_preterm_rate_by_group_computes_expected(tmp_path):
    summary = eda._preterm_rate_by(_toy(), "mother_race_ethnicity")
    rates = dict(zip(summary["mother_race_ethnicity"].to_list(), summary["rate"].to_list()))
    assert rates["NH Black"] == 1.0       # both Black rows preterm
    assert rates["NH White"] == 0.0       # null target row excluded


def test_plot_preterm_rate_by_group_writes_file(tmp_path):
    out = eda.plot_preterm_rate_by_group(
        _toy(), "mother_race_ethnicity", tmp_path / "r.png", title="t"
    )
    assert out.is_file() and out.stat().st_size > 0


def test_plot_missingness_writes_file(tmp_path):
    out = eda.plot_missingness(_toy(), ["meduc", "preterm"], tmp_path / "m.png")
    assert out.is_file() and out.stat().st_size > 0
