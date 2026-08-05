"""Tests for the full-population assembly path (Phase 2 full-data rebuild).

These cover the additive machinery that lets Phase 2 onward run on the COMPLETE 2018-2024
public-use files rather than the earlier spread sample:
- a deterministic stratified subsample (draw a tractable working sample from ~25M rows while
  preserving the outcome and race mix), and
- ``full_dataset`` orchestration: read the per-year cleaned Parquet files, run the full-population
  outcome/equity audit (all rows, so small race groups stay reliable), and assemble a leakage-safe
  modeling bundle from a stratified working sample.
"""

import numpy as np
import polars as pl

from equibirth.features import full_dataset, splits


def _frame(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    # preterm ~9%, three race groups of different sizes, two years.
    return pl.DataFrame(
        {
            "preterm": (rng.uniform(size=n) < 0.09).astype(int),
            "mother_race_ethnicity": rng.choice(
                ["NH White", "NH Black", "Hispanic"], n, p=[0.6, 0.25, 0.15]
            ),
            "birth_year": rng.choice([2023, 2024], n),
        }
    )


# --------------------------------------------------------------------------- stratified_subsample
def test_stratified_subsample_size_and_determinism():
    df = _frame()
    m1 = splits.stratified_subsample(df, 5000, seed=42)
    m2 = splits.stratified_subsample(df, 5000, seed=42)
    assert m1.sum() == 5000
    assert m1.to_list() == m2.to_list()          # deterministic
    assert m1.len() == df.height


def test_stratified_subsample_preserves_prevalence_and_race_mix():
    df = _frame()
    m = splits.stratified_subsample(df, 5000, seed=1)
    sub = df.filter(m)
    assert abs(sub["preterm"].mean() - df["preterm"].mean()) < 0.01
    # each race group's share is preserved within a small tolerance
    full_share = df["mother_race_ethnicity"].value_counts(normalize=True).sort("mother_race_ethnicity")
    sub_share = sub["mother_race_ethnicity"].value_counts(normalize=True).sort("mother_race_ethnicity")
    for f, s in zip(full_share["proportion"], sub_share["proportion"]):
        assert abs(f - s) < 0.02


def test_stratified_subsample_n_ge_population_returns_all():
    df = _frame(1000)
    m = splits.stratified_subsample(df, 5000, seed=0)
    assert m.sum() == 1000 and m.all()


# --------------------------------------------------------------------------- outcome audit
def _clean_frame(n=4000, seed=0):
    """A frame shaped like a per-year cleaned Parquet (raw NBER cols + target + race label)."""
    rng = np.random.default_rng(seed)
    race = rng.choice(["NH White", "NH Black", "Hispanic"], n, p=[0.6, 0.25, 0.15])
    base = np.where(race == "NH Black", 0.12, np.where(race == "Hispanic", 0.09, 0.07))
    return pl.DataFrame(
        {
            "preterm": (rng.uniform(size=n) < base).astype(int),
            "mother_race_ethnicity": race,
            "meduc": rng.integers(1, 9, n),
            "birth_year": rng.choice([2023, 2024], n),
            "mager": rng.integers(15, 45, n),
            "dmar": rng.choice([1, 2], n),
            "bmi": rng.normal(27, 5, n),
            "cig_0": np.zeros(n),
            "priorlive": rng.integers(0, 4, n),
            "priordead": np.zeros(n),
            "rf_pdiab": rng.choice(["Y", "N"], n),
            "rf_phype": rng.choice(["Y", "N"], n),
            "rf_ppterm": rng.choice(["Y", "N"], n),
            "rf_cesar": rng.choice(["Y", "N"], n),
            "wic": rng.choice(["Y", "N"], n),
            "precare": rng.integers(1, 6, n),
            "mracehisp": rng.integers(1, 8, n),
            "pay": rng.integers(1, 3, n),
            # leakage columns present in the raw frame; must never enter the matrix
            "dbwt": rng.integers(1500, 4000, n),
            "previs": rng.integers(0, 15, n),
            "dplural": np.ones(n),
            "combgest": rng.integers(30, 42, n),
        }
    )


def test_outcome_audit_rate_by_race_uses_all_rows():
    df = _clean_frame()
    audit = full_dataset.outcome_audit(df)
    by_race = {r["mother_race_ethnicity"]: r for r in audit["rate_by_race"]}
    # every row counted (full-population audit), not a subsample
    assert sum(r["n"] for r in audit["rate_by_race"]) == df.drop_nulls("preterm").height
    # NH Black rate exceeds NH White in this construction
    assert by_race["NH Black"]["rate"] > by_race["NH White"]["rate"]
    assert "overall_rate" in audit and "n_singletons" in audit


def test_outcome_audit_black_white_gap_present():
    df = _clean_frame()
    audit = full_dataset.outcome_audit(df)
    assert audit["black_white_gap_pp"] > 0    # the disparity the project measures


# --------------------------------------------------------------------------- assemble_modeling
def test_assemble_modeling_bundle_is_leakage_safe():
    df = _clean_frame(6000)
    bundle = full_dataset.assemble_modeling(df, working_size=3000, seed=42)
    assert bundle["X"].height == bundle["y"].len() == bundle["split"].len()
    assert bundle["X"].height <= 3000
    for leaky in ("dbwt", "previs", "dplural", "combgest"):
        assert leaky not in bundle["X"].columns
    assert set(bundle["split"].unique().to_list()) <= {
        "held_out_final", "dev_train", "dev_val", "dev_test"
    }
    # meta carries the audit columns for stratified reporting
    assert "mother_race_ethnicity" in bundle["meta"].columns


def test_assemble_modeling_is_deterministic():
    df = _clean_frame(6000)
    a = full_dataset.assemble_modeling(df, working_size=3000, seed=7)
    b = full_dataset.assemble_modeling(df, working_size=3000, seed=7)
    assert a["split"].to_list() == b["split"].to_list()
    assert a["y"].to_list() == b["y"].to_list()


# --------------------------------------------------------------------------- load_full_frame (IO)
def test_load_full_frame_pools_years(tmp_path):
    for year in (2023, 2024):
        _clean_frame(500, seed=year).write_parquet(tmp_path / f"natality{year}_clean.parquet")
    frame = full_dataset.load_full_frame(tmp_path, [2023, 2024])
    assert frame.height == 1000
    assert set(frame["birth_year"].unique().to_list()) == {2023, 2024}
    assert "preterm" in frame.columns
