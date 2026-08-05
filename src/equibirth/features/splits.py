"""Deterministic, stratified train/validation/test splits (SPEC FR5).

The naming avoids the tuning-vs-final "validation" collision (DECISION_LOG D18):
- ``held_out_final``: untouched until final reporting; all headline metrics and bootstrap CIs use it.
- ``dev_train`` / ``dev_val`` / ``dev_test``: the development partition. ``dev_val`` (the SPEC's
  "dev_internal_val", config ``sampling.dev_internal_val_fraction``) scores Optuna trials and early
  stopping; ``dev_test`` is used for probability calibration and a quick in-development smoke check.

Splits are stratified by a composite key (outcome x race/ethnicity x year by default) so every
subgroup is represented in each partition, and are reproducible from the seed. The labels are written
to disk so the exact partition is fixed across runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

DEFAULT_STRATIFY = ["preterm", "mother_race_ethnicity", "birth_year"]


def _strata_key(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    present = [c for c in cols if c in df.columns]
    if not present:
        return np.zeros(df.height, dtype=object)
    key = df.select(
        pl.concat_str([pl.col(c).cast(pl.String).fill_null("NA") for c in present], separator="|")
    ).to_series()
    return key.to_numpy()


def _safe_stratify(key: np.ndarray) -> np.ndarray | None:
    """Return a stratify array (or None) where every class has at least 2 members."""
    if len(key) == 0:
        return None
    can = len(set(key)) > 1 and np.min(np.unique(key, return_counts=True)[1]) >= 2
    return key if can else None


def assign_splits(
    df: pl.DataFrame,
    *,
    stratify_cols: list[str] | None = None,
    held_out_frac: float = 0.4,
    dev_val_frac_of_dev: float = 0.15,
    dev_test_frac_of_dev: float = 0.15,
    seed: int = 42,
) -> pl.Series:
    """Return a ``split`` label Series aligned to ``df`` rows.

    ``held_out_frac`` is a fraction of the WHOLE dataset. ``dev_val_frac_of_dev`` and
    ``dev_test_frac_of_dev`` are fractions of the DEVELOPMENT partition (what remains after holding
    out), not of the whole dataset. Example: held_out_frac=0.4, dev_*=0.15 gives roughly 40% held
    out, and within the remaining 60% dev rows, 15% val and 15% test, i.e. 40 / 42 / 9 / 9 overall.

    Rows in a rare stratum (fewer than 2 members) cannot be stratified; they are assigned to
    ``dev_train`` up front so no row is dropped and the split stays deterministic.
    """
    stratify_cols = stratify_cols or DEFAULT_STRATIFY
    n = df.height
    key = _strata_key(df, stratify_cols)
    labels = np.array(["dev_train"] * n, dtype=object)

    # Route rare-stratum rows straight to dev_train; stratify only the rest.
    unique, counts = np.unique(key, return_counts=True)
    rare = set(unique[counts < 2])
    is_rare = np.array([k in rare for k in key])
    pool = np.flatnonzero(~is_rare)
    if pool.size == 0:
        return pl.Series("split", labels)

    dev_idx, held_idx = train_test_split(
        pool, test_size=held_out_frac, random_state=seed, stratify=_safe_stratify(key[pool])
    )
    labels[held_idx] = "held_out_final"

    vt_frac = dev_val_frac_of_dev + dev_test_frac_of_dev
    _, valtest_idx = train_test_split(
        dev_idx, test_size=vt_frac, random_state=seed, stratify=_safe_stratify(key[dev_idx])
    )
    val_share = dev_val_frac_of_dev / vt_frac
    val_idx, test_idx = train_test_split(
        valtest_idx, train_size=val_share, random_state=seed,
        stratify=_safe_stratify(key[valtest_idx]),
    )
    labels[val_idx] = "dev_val"
    labels[test_idx] = "dev_test"
    return pl.Series("split", labels)


def stratified_subsample(
    df: pl.DataFrame,
    n: int,
    *,
    stratify_cols: list[str] | None = None,
    seed: int = 42,
) -> pl.Series:
    """Return a boolean mask selecting ~``n`` rows, stratified to preserve the outcome/race mix.

    Used to draw a tractable working sample from the complete ~25M-row population for model
    development, while the full-population outcome audit still runs on every row (SPEC FR5/FR8.1:
    separate the outcome equity audit from the model-performance audit). Selection is proportional
    within each stratum (outcome x race/ethnicity x year by default), so the sample's marginal
    distributions match the population. Deterministic given the seed. If ``n`` is at least the
    population size, every row is selected.
    """
    total = df.height
    if n >= total or total == 0:
        return pl.Series("selected", [True] * total)

    stratify_cols = stratify_cols or DEFAULT_STRATIFY
    key = _strata_key(df, stratify_cols)
    rng = np.random.default_rng(seed)
    frac = n / total
    selected = np.zeros(total, dtype=bool)

    # Proportional allocation within each stratum, with largest-remainder rounding so the total
    # lands exactly on ``n`` and no stratum is silently dropped.
    order = np.argsort(key, kind="stable")
    unique, starts = np.unique(key[order], return_index=True)
    bounds = list(starts) + [total]
    floors, remainders = [], []
    for i in range(len(unique)):
        idx = order[bounds[i]:bounds[i + 1]]
        exact = len(idx) * frac
        take = int(np.floor(exact))
        floors.append((idx, take))
        remainders.append(exact - take)
    allocated = sum(t for _, t in floors)
    leftover = n - allocated
    # hand out the leftover to the strata with the largest fractional remainder
    for j in np.argsort(remainders)[::-1][:leftover]:
        idx, take = floors[j]
        floors[j] = (idx, min(take + 1, len(idx)))
    for idx, take in floors:
        if take <= 0:
            continue
        chosen = rng.choice(idx, size=min(take, len(idx)), replace=False)
        selected[chosen] = True
    return pl.Series("selected", selected)


def save_splits(labels: pl.Series, path: str | Path) -> Path:
    """Persist the split labels to Parquet so the partition is fixed across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_frame().write_parquet(path)
    return path
