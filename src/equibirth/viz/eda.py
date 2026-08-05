"""Phase 1 exploratory plots for the natality data.

Every figure carries a caption that states the takeaway, not just the axes (SPEC writing style).
These functions take an already-loaded, de-sentinelled Polars frame (with the ``preterm`` target and
``mother_race_ethnicity`` label) so they never re-read raw data or duplicate cleaning logic.

IMPORTANT: rates computed on the development head-sample are NOT nationally representative (the NBER
CSV is not randomly ordered). Captions say so. Headline national rates require full or random samples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, file-only output
import matplotlib.pyplot as plt
import polars as pl

from equibirth.ingest import schema
from equibirth.utils.io import ensure_dir

EDUCATION_LABELS = {
    1: "<8th grade", 2: "9-12 no diploma", 3: "HS grad/GED", 4: "Some college",
    5: "Associate", 6: "Bachelor", 7: "Master", 8: "Doctorate/Prof",
}
PAYMENT_LABELS = {1: "Medicaid", 2: "Private", 3: "Self-pay", 4: "Indian Health Svc",
                  5: "CHAMPUS/TRICARE", 6: "Other govt", 8: "Other"}

_SAMPLE_NOTE = "Development head-sample, NOT nationally representative; for code validation only."


def _preterm_rate_by(df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Return preterm rate and n per group, ignoring null target/group rows."""
    valid = df.filter(pl.col("preterm").is_not_null() & pl.col(group_col).is_not_null())
    return (
        valid.group_by(group_col)
        .agg(rate=pl.col("preterm").mean(), n=pl.len())
        .sort(group_col)
    )


def plot_preterm_rate_by_group(
    df: pl.DataFrame, group_col: str, out_path: str | Path, *, title: str,
    label_map: dict | None = None, sample_note: bool = True,
) -> Path:
    """Bar chart of preterm rate by a grouping column. Returns the saved path."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    summary = _preterm_rate_by(df, group_col)
    labels = summary[group_col].to_list()
    if label_map:
        labels = [label_map.get(v, str(v)) for v in labels]
    rates = [r * 100 for r in summary["rate"].to_list()]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(labels)), rates, color="#3b6ea5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Preterm rate (%)")
    ax.set_title(title)
    caption = title + ("\n" + _SAMPLE_NOTE if sample_note else "")
    fig.text(0.01, -0.02, caption, fontsize=8, color="#555555", wrap=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _age_bracket_labels() -> list[str]:
    """String labels derived from schema.AGE_BRACKET_BREAKS (single source of truth)."""
    breaks = schema.AGE_BRACKET_BREAKS
    labels = [f"<{breaks[0]}"]
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        labels.append(f"{lo}-{hi - 1}")
    labels.append(f"{breaks[-1]}+")
    return labels


def add_age_bracket(df: pl.DataFrame) -> pl.DataFrame:
    """Add clinical maternal-age brackets (relationship with preterm is U-shaped, not linear)."""
    if "mager" not in df.columns:
        return df
    labels = _age_bracket_labels()
    expr = pl.when(pl.col("mager") < schema.AGE_BRACKET_BREAKS[0]).then(pl.lit(labels[0]))
    for i, brk in enumerate(schema.AGE_BRACKET_BREAKS[1:], start=1):
        expr = expr.when(pl.col("mager") < brk).then(pl.lit(labels[i]))
    return df.with_columns(expr.otherwise(pl.lit(labels[-1])).alias("age_bracket"))


def singleton_race_diagnostic(df_all: pl.DataFrame) -> pl.DataFrame:
    """Preterm rate by race for all births vs singletons (D17 singleton-bias check).

    Returns a frame with race, rate_all, rate_singleton, and the Black-White gap each way so we can
    flag if restricting to singletons materially shifts the disparity the project measures.
    """
    def rates(d: pl.DataFrame, name: str) -> pl.DataFrame:
        v = d.filter(pl.col("preterm").is_not_null() & pl.col("mother_race_ethnicity").is_not_null())
        return v.group_by("mother_race_ethnicity").agg(pl.col("preterm").mean().alias(name))

    all_r = rates(df_all, "rate_all")
    singl = rates(df_all.filter(pl.col("dplural") == 1), "rate_singleton")
    return all_r.join(singl, on="mother_race_ethnicity", how="outer_coalesce").sort("mother_race_ethnicity")


def plot_distribution(df: pl.DataFrame, col: str, out_path: str | Path, *, title: str,
                      sample_note: bool = False) -> Path:
    """Histogram of a numeric column (e.g. obstetric gestation in weeks)."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    vals = df[col].drop_nulls().to_list()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals, bins=40, color="#3b6ea5")
    ax.axvline(37, color="#a5673b", linestyle="--", label="37 wks (preterm cutoff)")
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    if sample_note:
        fig.text(0.01, -0.02, _SAMPLE_NOTE, fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reliability(curves: dict, out_path: str | Path, *, title: str) -> Path:
    """Reliability diagram. ``curves`` maps a label to (mean_pred, observed_rate, counts) arrays.

    A well-calibrated model tracks the diagonal: a predicted 30% risk should match a 30% observed
    rate. Curves stratified by race make calibration-by-group visible (the fairness-critical view).
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="perfect calibration")
    for label, (mp, obs, counts) in curves.items():
        valid = counts > 0
        ax.plot(mp[valid], obs[valid], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed preterm rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_missingness(df: pl.DataFrame, cols: list[str], out_path: str | Path) -> Path:
    """Bar chart of percent-null per column after de-sentinelling (silent-bug check)."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    present = [c for c in cols if c in df.columns]
    pct = [df[c].null_count() / df.height * 100 for c in present]
    order = sorted(range(len(present)), key=lambda i: pct[i], reverse=True)
    present = [present[i] for i in order]
    pct = [pct[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(present)), pct, color="#a5673b")
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    ax.set_xlabel("Percent missing after de-sentinelling (%)")
    ax.set_title("Missingness by variable (converted 'unknown' codes to null)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
