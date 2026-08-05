"""SAMPLE-ERA (retained): part of the spread-sample chain that regenerates the Streamlit app's
reports/dashboard_data.json bundle. Its numbers are NOT the canonical ones and must not be cited;
the canonical full-data (2018-2024) pipeline lives in scripts/week*_*.py and reports/week*. This
chain is kept only so the app's exploration sections stay reproducible, and the app discloses on
screen that those sections use a validated representative spread sample.

Phase 8 genuine exploration: temporal/COVID, intersectional equity, unsupervised structure,
and a ceiling-probe ablation. Scripts-first: every number here feeds the notebooks and dashboard.

Run: PYTHONPATH=src ./.venv/bin/python scripts/phase8_exploration.py

Outputs (figures gitignored; JSON + markdown are committed evidence):
  reports/phase8_exploration.json   machine-readable results
  reports/phase8_exploration.md     human-readable summary with takeaways
  reports/figures/phase8_*.png      figures

Interpretation is associational. Race is a social construct and a marker of exposure to racism,
not a biological cause (SPEC FR11).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from equibirth.explore import analysis as ea
from equibirth.utils.config import load_config
from equibirth.utils.io import ensure_dir
from equibirth.utils.seeds import set_seeds

# Tasteful, colorblind-aware palette (Okabe-Ito derived) used across deliverables.
PALETTE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999",
}
RACE_ORDER = ["NH White", "NH Asian", "Hispanic", "NH Multiracial", "NH NHOPI", "NH AIAN", "NH Black"]
EDUCATION_LABELS = {
    1: "<8th grade", 2: "9-12, no diploma", 3: "HS grad/GED", 4: "Some college",
    5: "Associate", 6: "Bachelor", 7: "Master", 8: "Doctorate/Prof",
}
PAYMENT_LABELS = {1: "Medicaid", 2: "Private", 3: "Self-pay", 4: "IHS", 5: "TRICARE",
                  6: "Other govt", 8: "Other"}

NUMERIC_FEATURES = ["mager", "meduc", "bmi", "cig_0", "priorlive", "priordead", "medical_risk_count"]
BINARY_FEATURES = ["married", "rf_pdiab", "rf_phype", "rf_ppterm", "rf_cesar", "wic",
                   "first_trimester_care"]
CLUSTER_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _impute(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns([pl.col(c).fill_null(pl.col(c).median()) for c in cols])


# --------------------------------------------------------------------------- temporal / COVID
def run_temporal(df: pl.DataFrame, figdir: Path) -> dict:
    overall = ea.temporal_rates(df)
    by_race = ea.temporal_rates(df, group_col="mother_race_ethnicity")
    covid = ea.covid_period_table(df, group_col="mother_race_ethnicity")

    fig, ax = plt.subplots(figsize=(9, 5))
    for race in RACE_ORDER:
        sub = by_race.filter(pl.col("mother_race_ethnicity") == race).sort("birth_year")
        if sub.height == 0:
            continue
        color = PALETTE["red"] if race == "NH Black" else (
            PALETTE["blue"] if race == "NH White" else PALETTE["grey"])
        lw = 2.5 if race in ("NH Black", "NH White") else 1.2
        ax.plot(sub["birth_year"], [r * 100 for r in sub["rate"]], marker="o",
                color=color, linewidth=lw, label=race, alpha=0.9 if lw > 2 else 0.6)
    ax.axvspan(2019.5, 2021.5, color=PALETTE["yellow"], alpha=0.18)
    ax.text(2020.5, ax.get_ylim()[1] * 0.97, "COVID-era", ha="center", fontsize=9, color="#777700")
    _style(ax, "Singleton preterm rate by race, 2019-2024",
           "Birth year", "Preterm rate (%)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "phase8_temporal_by_race.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "overall_by_year": overall.to_dicts(),
        "by_race_year": by_race.to_dicts(),
        "covid_period_shift": covid.to_dicts(),
    }


# --------------------------------------------------------------------------- intersectional equity
def run_intersectional(df: pl.DataFrame, figdir: Path) -> dict:
    gap_edu = ea.gap_within_strata(
        df, "mother_race_ethnicity", "meduc", focal="NH Black", reference="NH White")
    gap_pay = ea.gap_within_strata(
        df, "mother_race_ethnicity", "pay", focal="NH Black", reference="NH White")
    inter_edu = ea.intersectional_rate(df, "mother_race_ethnicity", "meduc")

    # Key comparison: college-educated (Bachelor+) Black vs White HS-or-less.
    def rate(race, edus):
        sub = df.filter(
            (pl.col("mother_race_ethnicity") == race) & pl.col("meduc").is_in(edus)
            & pl.col("preterm").is_not_null())
        return float(sub["preterm"].mean()), sub.height

    black_college, n_bc = rate("NH Black", [6, 7, 8])
    white_hsless, n_wh = rate("NH White", [1, 2, 3])

    fig, ax = plt.subplots(figsize=(9, 5))
    edus = sorted(EDUCATION_LABELS)
    for race, color in [("NH White", PALETTE["blue"]), ("NH Black", PALETTE["red"]),
                        ("Hispanic", PALETTE["green"]), ("NH Asian", PALETTE["orange"])]:
        ys = []
        for e in edus:
            row = inter_edu.filter((pl.col("mother_race_ethnicity") == race) & (pl.col("meduc") == e))
            ys.append(row["rate"].item() * 100 if row.height and row["rate"].item() is not None else np.nan)
        ax.plot(edus, ys, marker="o", color=color, linewidth=2, label=race)
    ax.set_xticks(edus)
    ax.set_xticklabels([EDUCATION_LABELS[e] for e in edus], rotation=35, ha="right", fontsize=8)
    _style(ax, "Preterm rate by maternal education, within race",
           "Maternal education", "Preterm rate (%)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "phase8_intersectional_education.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "gap_black_white_by_education": gap_edu.to_dicts(),
        "gap_black_white_by_payment": gap_pay.to_dicts(),
        "black_college_rate": black_college, "black_college_n": n_bc,
        "white_hs_or_less_rate": white_hsless, "white_hs_or_less_n": n_wh,
    }


# --------------------------------------------------------------------------- unsupervised structure
def run_unsupervised(df: pl.DataFrame, figdir: Path, seed: int, sample_n: int = 60000) -> dict:
    dev = df.filter(pl.col("split") == "dev_train")
    dev = _impute(dev, CLUSTER_FEATURES)
    if dev.height > sample_n:
        dev = dev.sample(sample_n, seed=seed)
    X = dev.select(CLUSTER_FEATURES).to_numpy().astype(float)
    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=10, random_state=seed).fit(Xs)
    evr = pca.explained_variance_ratio_

    sub_idx = np.random.default_rng(seed).choice(len(Xs), size=min(8000, len(Xs)), replace=False)
    best_k, best_sil = None, -1.0
    sil_by_k = {}
    for k in range(3, 8):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs[sub_idx])
        sil = float(silhouette_score(Xs[sub_idx], km.labels_))
        sil_by_k[k] = sil
        if sil > best_sil:
            best_k, best_sil = k, sil

    km = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit(Xs)
    profiles = ea.profile_clusters(
        km.labels_, dev["preterm"].to_numpy(), dev["mother_race_ethnicity"].to_numpy(),
        dev.select(CLUSTER_FEATURES))

    coords = pca.transform(Xs)[:, :2]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=km.labels_, cmap="tab10", s=4, alpha=0.4)
    _style(ax, f"Feature-space clusters (PCA projection, k={best_k})",
           "PC1", "PC2")
    ax.legend(*sc.legend_elements(), title="cluster", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "phase8_clusters_pca.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "pca_explained_variance_ratio": [float(v) for v in evr],
        "pca_cumulative_top5": float(evr[:5].sum()),
        "silhouette_by_k": sil_by_k, "chosen_k": best_k,
        "cluster_profiles": [
            {"cluster": p.cluster, "n": p.n, "preterm_rate": p.preterm_rate,
             "race_composition": p.race_composition, "feature_means": p.feature_means}
            for p in profiles],
    }


# --------------------------------------------------------------------------- ceiling probe (ablation)
def run_ceiling_probe(df: pl.DataFrame, seed: int, train_n: int = 250000,
                      test_n: int = 200000) -> dict:
    race_cols = [c for c in df.columns if c.startswith("mother_race_ethnicity_")
                 and c != "mother_race_ethnicity_null"]
    pay_cols = [c for c in df.columns if c.startswith("pay_") and c != "pay_null"]
    groups = {
        "race_only": race_cols,
        "demographic": race_cols + ["mager", "age_bracket", "married"],
        "demographic+SES": race_cols + ["mager", "age_bracket", "married", "meduc", "wic",
                                        "first_trimester_care"] + pay_cols,
        "full_clinical": race_cols + ["mager", "age_bracket", "married", "meduc", "wic",
                                      "first_trimester_care"] + pay_cols
        + ["bmi", "cig_0", "priorlive", "priordead", "rf_pdiab", "rf_phype", "rf_ppterm",
           "rf_cesar", "medical_risk_count"],
    }
    impute_cols = ["mager", "meduc", "bmi", "cig_0", "priorlive", "priordead", "age_bracket",
                   "married", "wic", "first_trimester_care", "rf_pdiab", "rf_phype", "rf_ppterm",
                   "rf_cesar", "medical_risk_count"]
    train = df.filter((pl.col("split") == "dev_train") & pl.col("preterm").is_not_null())
    test = df.filter((pl.col("split") == "held_out_final") & pl.col("preterm").is_not_null())
    train = _impute(train, impute_cols)
    test = _impute(test, impute_cols)
    if train.height > train_n:
        train = train.sample(train_n, seed=seed)
    if test.height > test_n:
        test = test.sample(test_n, seed=seed)
    ytr, yte = train["preterm"].to_numpy(), test["preterm"].to_numpy()

    results = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        Xtr = train.select(cols).to_numpy().astype(float)
        Xte = test.select(cols).to_numpy().astype(float)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(Xtr), ytr)
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        results[name] = {"n_features": len(cols), "auc_roc": float(roc_auc_score(yte, p))}
    return {"test_prevalence": float(yte.mean()), "ablation": results}


def _md(results: dict) -> str:
    t = results["temporal"]["covid_period_shift"]
    inter = results["intersectional"]
    ab = results["ceiling_probe"]["ablation"]
    lines = ["# Phase 8 Exploration Findings\n",
             "All numbers regenerated from data/processed/modeling_2019_2024.parquet. Associational.\n",
             "## 1. Temporal / COVID-era trend\n",
             "COVID-minus-precovid shift in singleton preterm rate (percentage points), by race:\n"]
    for row in t:
        shift = row.get("covid_shift_pp")
        if shift is not None:
            lines.append(f"- {row['mother_race_ethnicity']}: {shift:+.2f} pp")
    lines.append("\n## 2. Intersectional equity (race x SES)\n")
    lines.append(f"- College-educated (Bachelor+) NH Black preterm rate: "
                 f"{inter['black_college_rate']*100:.2f}% (n={inter['black_college_n']:,})")
    lines.append(f"- HS-or-less NH White preterm rate: "
                 f"{inter['white_hs_or_less_rate']*100:.2f}% (n={inter['white_hs_or_less_n']:,})")
    lines.append("\nBlack-White gap (pp) within each education level:\n")
    for row in inter["gap_black_white_by_education"]:
        if row.get("gap_pp") is not None:
            lines.append(f"- edu {row['meduc']}: {row['gap_pp']:+.2f} pp "
                         f"(Black n={row.get('focal_n')}, White n={row.get('ref_n')})")
    lines.append("\n## 3. Unsupervised structure\n")
    u = results["unsupervised"]
    lines.append(f"- PCA: top 5 PCs explain {u['pca_cumulative_top5']*100:.1f}% of variance.")
    lines.append(f"- KMeans chosen k={u['chosen_k']} (silhouette {u['silhouette_by_k']}).")
    lines.append("- Cluster preterm rates: " + ", ".join(
        f"c{p['cluster']}={p['preterm_rate']*100:.1f}% (n={p['n']:,})"
        for p in u["cluster_profiles"]))
    lines.append("\n## 4. Ceiling probe (logistic, held-out AUC-ROC)\n")
    for name, r in ab.items():
        lines.append(f"- {name} ({r['n_features']} feats): AUC-ROC {r['auc_roc']:.4f}")
    lines.append(f"\nTest prevalence {results['ceiling_probe']['test_prevalence']*100:.2f}%.\n")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["random_seed"]
    set_seeds(seed)
    reports = Path(cfg["paths"]["reports"])
    figdir = Path(cfg["paths"]["figures"])
    ensure_dir(reports)
    ensure_dir(figdir)

    df = pl.read_parquet("data/processed/modeling_2019_2024.parquet")
    print(f"loaded {df.height:,} rows")

    results = {
        "temporal": run_temporal(df, figdir),
        "intersectional": run_intersectional(df, figdir),
        "unsupervised": run_unsupervised(df, figdir, seed),
        "ceiling_probe": run_ceiling_probe(df, seed),
    }
    (reports / "phase8_exploration.json").write_text(json.dumps(results, indent=2))
    (reports / "phase8_exploration.md").write_text(_md(results))
    print("wrote reports/phase8_exploration.{json,md} and figures")
    print(_md(results))


if __name__ == "__main__":
    main()
