"""Tests for deterministic stratified splits."""

import numpy as np
import polars as pl

from equibirth.features import splits


def _frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "preterm": rng.integers(0, 2, n),
            "mother_race_ethnicity": rng.choice(["NH White", "NH Black", "Hispanic"], n),
            "birth_year": rng.choice([2023, 2024], n),
        }
    )


def test_all_rows_labeled_with_four_partitions():
    lab = splits.assign_splits(_frame())
    assert lab.len() == 4000
    assert set(lab.unique().to_list()) == {"held_out_final", "dev_train", "dev_val", "dev_test"}


def test_held_out_fraction_is_approximately_right():
    lab = splits.assign_splits(_frame(), held_out_frac=0.4)
    frac = (lab == "held_out_final").sum() / lab.len()
    assert 0.37 < frac < 0.43


def test_is_deterministic():
    df = _frame()
    a = splits.assign_splits(df, seed=42).to_list()
    b = splits.assign_splits(df, seed=42).to_list()
    assert a == b


def test_stratification_preserves_outcome_rate_in_held_out():
    df = _frame()
    lab = splits.assign_splits(df)
    overall = df["preterm"].mean()
    held = df.filter(lab == "held_out_final")["preterm"].mean()
    assert abs(overall - held) < 0.03


def test_save_splits_roundtrip(tmp_path):
    lab = splits.assign_splits(_frame())
    p = splits.save_splits(lab, tmp_path / "splits.parquet")
    back = pl.read_parquet(p)["split"]
    assert back.to_list() == lab.to_list()
