"""End-to-end test of multi-year assembly on tiny synthetic CSVs (real ingest path)."""

import polars as pl

from equibirth.features import dataset


def _write_year_csv(path, n_rows):
    # Minimal NBER-like columns the loader/builder need.
    base = {
        "oegest_comb": [36, 40] * (n_rows // 2),
        "combgest": [36, 40] * (n_rows // 2),
        "dplural": [1] * n_rows,
        "mracehisp": [2, 1] * (n_rows // 2),
        "mager": [25, 33] * (n_rows // 2),
        "meduc": [3, 6] * (n_rows // 2),
        "dmar": [2, 1] * (n_rows // 2),
        "pay": [1, 2] * (n_rows // 2),
        "wic": ["Y", "N"] * (n_rows // 2),
        "rf_pdiab": ["N", "N"] * (n_rows // 2),
        "rf_phype": ["N", "Y"] * (n_rows // 2),
        "rf_ppterm": ["Y", "N"] * (n_rows // 2),
        "rf_cesar": ["N", "N"] * (n_rows // 2),
        "rf_gdiab": ["N", "N"] * (n_rows // 2),
        "bmi": [22.0, 28.0] * (n_rows // 2),
        "cig_0": [0, 0] * (n_rows // 2),
        "precare": [2, 5] * (n_rows // 2),
        "dbwt": [2200, 3400] * (n_rows // 2),
        "previs": [8, 12] * (n_rows // 2),
    }
    pl.DataFrame(base).write_csv(path)


def test_assemble_multi_year(tmp_path):
    for year in (2023, 2024):
        _write_year_csv(tmp_path / f"natality{year}_spread.csv", 200)

    bundle = dataset.assemble([2023, 2024], tmp_path)
    assert bundle["X"].height == bundle["y"].len() == bundle["split"].len()
    assert set(bundle["split"].unique().to_list()) <= {
        "held_out_final", "dev_train", "dev_val", "dev_test"
    }
    # no leakage columns leaked into features
    for leaky in ("dbwt", "previs", "dplural", "combgest"):
        assert leaky not in bundle["X"].columns
    # birth_year captured in meta for stratification/audit
    assert set(bundle["meta"]["birth_year"].unique().to_list()) == {2023, 2024}


def test_save_processed_roundtrip(tmp_path):
    for year in (2023, 2024):
        _write_year_csv(tmp_path / f"natality{year}_spread.csv", 100)
    bundle = dataset.assemble([2023, 2024], tmp_path)
    out = dataset.save_processed(bundle, tmp_path / "proc.parquet")
    back = pl.read_parquet(out)
    assert "preterm" in back.columns and "split" in back.columns and "birth_year" in back.columns
