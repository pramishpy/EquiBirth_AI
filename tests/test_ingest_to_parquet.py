"""Tests for the streaming CSV-to-Parquet converter."""

import polars as pl

from equibirth.ingest import to_parquet


def _write_csv(path):
    pl.DataFrame(
        {
            "oegest_comb": [36, 40, 99],
            "combgest": [36, 40, 41],
            "previs": [9, 99, 12],
            "dplural": [1, 2, 1],
            "mracehisp": [2, 1, 7],
            "mager": [30, 25, 40],
            "dbwt": [2500, 3500, 3000],  # leakage var retained in slim parquet, not as feature
        }
    ).write_csv(path)


def test_convert_builds_target_and_race_and_desentinels(tmp_path):
    csv = tmp_path / "natality2099us.csv"
    _write_csv(csv)
    out = to_parquet.convert_csv_to_parquet(csv, tmp_path / "y.parquet")
    df = pl.read_parquet(out)

    assert "preterm" in df.columns and "mother_race_ethnicity" in df.columns
    assert df["preterm"].to_list() == [1, 0, None]            # 36->1, 40->0, 99->null
    assert df["previs"].to_list()[1] is None                  # 99 sentinel -> null
    assert df["mother_race_ethnicity"].to_list() == ["NH Black", "NH White", "Hispanic"]
    # leakage var kept in interim parquet (for the Phase 2 leakage demo) but it is not a feature name
    assert "dbwt" in df.columns


def test_singletons_only_filter(tmp_path):
    csv = tmp_path / "natality2099us.csv"
    _write_csv(csv)
    out = to_parquet.convert_csv_to_parquet(csv, tmp_path / "s.parquet", singletons_only=True)
    df = pl.read_parquet(out)
    assert set(df["dplural"].to_list()) == {1}
