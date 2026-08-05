"""SAMPLE-ERA (retained): part of the spread-sample chain that regenerates the Streamlit app's
reports/dashboard_data.json bundle. Its numbers are NOT the canonical ones and must not be cited;
the canonical full-data (2018-2024) pipeline lives in scripts/week*_*.py and reports/week*. This
chain is kept only so the app's exploration sections stay reproducible, and the app discloses on
screen that those sections use a validated representative spread sample.

Precompute the dashboard's data bundle so the Streamlit app loads verified JSON (and runs on a
clone where the gitignored parquet is absent). All descriptive numbers are regenerated here from the
modeling parquet; model-performance scalars are carried from the committed phase reports with their
source noted. No raw rows are shipped, only aggregates.

Run: PYTHONPATH=src ./.venv/bin/python scripts/build_dashboard_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from equibirth.explore import analysis as ea
from equibirth.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
DATA = ROOT / "data" / "processed" / "modeling_2019_2024.parquet"
OUT = ROOT / "reports" / "dashboard_data.json"
PHASE8 = ROOT / "reports" / "phase8_exploration.json"

EDUCATION_LABELS = {
    1: "<8th grade", 2: "9-12, no diploma", 3: "HS grad/GED", 4: "Some college",
    5: "Associate", 6: "Bachelor", 7: "Master", 8: "Doctorate/Prof",
}
RACE_ORDER = ["NH White", "NH Asian", "Hispanic", "NH Multiracial", "NH NHOPI", "NH AIAN", "NH Black"]


def compute_eda() -> dict:
    """Precompute the dashboard's EDA aggregates from the parquet so the deployed app needs no raw data.

    This mirrors what the Streamlit app would otherwise compute live; shipping it as committed JSON lets
    the app run on Streamlit Community Cloud, where the gitignored parquet is absent. Only small
    aggregates leave the machine, never raw rows.
    """
    df = pd.read_parquet(DATA, columns=[
        "birth_year", "mager", "meduc", "bmi", "cig_0", "priorlive", "medical_risk_count",
        "rf_ppterm", "rf_phype", "rf_pdiab", "rf_cesar", "wic", "first_trimester_care", "married",
        "priordead", "mother_race_ethnicity", "preterm"])
    d = df.dropna(subset=["preterm"])

    def rate(col, frame=d):
        g = frame.groupby(col)["preterm"].agg(rate="mean", n="size")
        g["rate"] = g["rate"] * 100
        return g.reset_index().to_dict("records")

    num = ["mager", "meduc", "bmi", "cig_0", "priorlive", "priordead", "medical_risk_count",
           "rf_ppterm", "rf_phype", "rf_pdiab", "rf_cesar", "wic", "first_trimester_care", "married"]
    edu_race = (d[d.mother_race_ethnicity.isin(RACE_ORDER)].dropna(subset=["meduc"])
                .assign(meduc=lambda x: x.meduc.astype(int))
                .groupby(["mother_race_ethnicity", "meduc"])["preterm"].mean().mul(100)
                .rename("rate").reset_index().to_dict("records"))
    age_c, age_e = np.histogram(d["mager"].dropna(), bins=range(12, 51))
    bmi_c, bmi_e = np.histogram(d["bmi"][d.bmi < 70].dropna(), bins=40)
    return {
        "n_rows": int(len(df)), "n_cols": 108, "overall_rate": float(d.preterm.mean() * 100),
        "n_preterm": int(d.preterm.sum()), "n_missing_target": int(df.preterm.isna().sum()),
        "rate_by_race": rate("mother_race_ethnicity"), "by_year": rate("birth_year"),
        "prior_preterm": rate("rf_ppterm"), "medical_risk": rate("medical_risk_count"),
        "smoking": {k: float(v) for k, v in
                    d.assign(s=np.where(d.cig_0.between(1, 89), "Smoker", "Non-smoker"))
                     .groupby("s")["preterm"].mean().mul(100).items()},
        "corr": {k: float(v) for k, v in
                 d[num + ["preterm"]].corr()["preterm"].drop("preterm").sort_values().items()},
        "age_hist": {"counts": age_c.tolist(), "edges": list(age_e)},
        "bmi_hist": {"counts": bmi_c.tolist(), "edges": [float(x) for x in bmi_e]},
        "edu_dist": {int(k): int(v) for k, v in
                     d["meduc"].dropna().astype(int).value_counts().sort_index().items()},
        "edu_race": edu_race,
    }


def main() -> None:
    df = pl.read_parquet(DATA)

    rate_by_race = ea.preterm_rate_by(df, "mother_race_ethnicity").to_dicts()
    inter_edu = ea.intersectional_rate(df, "mother_race_ethnicity", "meduc").to_dicts()
    gap_edu = ea.gap_within_strata(
        df, "mother_race_ethnicity", "meduc", focal="NH Black", reference="NH White").to_dicts()
    temporal = ea.temporal_rates(df, group_col="mother_race_ethnicity").to_dicts()

    phase8 = json.loads(PHASE8.read_text()) if PHASE8.exists() else {}

    bundle = {
        "meta": {
            "n_rows": df.height,
            "years": sorted(df["birth_year"].unique().to_list()),
            "education_labels": EDUCATION_LABELS,
            "source": "data/processed/modeling_2019_2024.parquet (singleton 2019-2024)",
        },
        "rate_by_race": rate_by_race,
        "intersectional_education": inter_edu,
        "gap_black_white_by_education": gap_edu,
        "temporal_by_race": temporal,
        "clusters": phase8.get("unsupervised", {}).get("cluster_profiles", []),
        "ceiling_probe": phase8.get("ceiling_probe", {}),
        "covid_period_shift": phase8.get("temporal", {}).get("covid_period_shift", []),
        "intersectional_scalars": {
            "black_college_rate": phase8.get("intersectional", {}).get("black_college_rate"),
            "black_college_n": phase8.get("intersectional", {}).get("black_college_n"),
            "white_hs_or_less_rate": phase8.get("intersectional", {}).get("white_hs_or_less_rate"),
            "white_hs_or_less_n": phase8.get("intersectional", {}).get("white_hs_or_less_n"),
        },
        # Verified model-performance headlines (sources noted; see FINDINGS F9-F11, reports/*.md).
        "model_findings": {
            "best_model": "XGBoost (Optuna)",
            "auc_roc": 0.657, "auc_pr": 0.182, "ece": 0.001,
            "leaky_auc": 0.913,
            "fraction_near_base_rate": 0.85,
            "raw_black_white_gap_pp": 5.1,
            "adjusted_gap_pp": 3.4,
            "explained_share_primary": 0.33,
            "unexplained_share_primary": 0.67,
            "disparate_impact_black": 2.86,
            "source": "FINDINGS F9-F11; reports/phase4_boosting_fairness.md, model_card.md",
        },
        # Precomputed EDA aggregates so the deployed dashboard runs without the gitignored parquet.
        "eda": compute_eda(),
    }

    def _jsonable(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")

    OUT.write_text(json.dumps(bundle, indent=2, default=_jsonable))
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
