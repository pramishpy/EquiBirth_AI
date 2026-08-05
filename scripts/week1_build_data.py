"""Week 1 (Data Acquisition + Exploration + EDA) dashboard-data step.

This is the small, honest preparation step that sits between the exploration table and the
standalone HTML dashboard. It does NOT do feature engineering or modeling. It simply:

  1. Reads the week-1 exploration table (17 raw pre-delivery fields, singletons, 2018-2024) that
     scripts/week1_build_full_dataset.py produced from the COMPLETE public-use files.
     (Raw values untouched: NCHS sentinel codes like BMI 99.9 are KEPT on purpose so the EDA
     notebook can hunt them down, which is the point of exploration.)
  2. Precomputes the aggregate bundle the standalone HTML dashboard renders, so the dashboard
     ships no raw rows, only small summaries plus eight illustrative sample rows.

Pipeline order: week1_download_full.sh -> week1_build_full_dataset.py -> this script -> the
dashboard/notebooks.

Run: PYTHONPATH=src ./.venv/bin/python scripts/week1_build_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The week-1 exploration table built by scripts/week1_build_full_dataset.py from the COMPLETE
# 2018-2024 public-use files (17 raw pre-delivery fields, singletons only).
EXPLORE_OUT = ROOT / "data" / "interim" / "natality_exploration_2019_2024.parquet"
JSON_OUT = ROOT / "reports" / "week1_dashboard_data.json"

# The raw, pre-delivery certificate fields we explore this week. Deliberately NO engineered
# composites, NO one-hot dummies, NO train/test split column: those belong to later weeks.
RAW_FIELDS = [
    "birth_year", "mager", "meduc", "bmi", "cig_0", "priorlive", "priordead", "married",
    "rf_pdiab", "rf_phype", "rf_ppterm", "rf_cesar", "wic", "first_trimester_care", "pay",
    "mother_race_ethnicity", "preterm",
]

EDU = {1: "<8th grade", 2: "9-12, no diploma", 3: "HS grad/GED", 4: "Some college",
       5: "Associate", 6: "Bachelor", 7: "Master", 8: "Doctorate/Prof"}
PAY = {1: "Private", 2: "Medicaid", 3: "Self-pay", 4: "Indian Health Svc",
       5: "CHAMPUS/TRICARE", 6: "Other govt", 8: "Other"}
RACE_ORDER = ["NH White", "NH Asian", "Hispanic", "NH Multiracial", "NH NHOPI", "NH AIAN", "NH Black"]

# Published total U.S. births (NCHS final; 2024 provisional). Singletons are ~96.8% of births.
PUBLISHED_BIRTHS = {2018: 3791712, 2019: 3747540, 2020: 3613647, 2021: 3664292, 2022: 3667758,
                    2023: 3596017, 2024: 3600000}


def load_exploration_table() -> pd.DataFrame:
    """Return the tidy week-1 exploration table (built from the complete files)."""
    if EXPLORE_OUT.is_file():
        return pd.read_parquet(EXPLORE_OUT)
    raise FileNotFoundError(
        f"{EXPLORE_OUT.name} is not present. Run scripts/week1_download_full.sh then "
        "scripts/week1_build_full_dataset.py first."
    )


def _rate(frame: pd.DataFrame, col: str) -> list[dict]:
    g = frame.groupby(col)["preterm"].agg(rate="mean", n="size")
    g["rate"] = g["rate"] * 100
    return g.reset_index().to_dict("records")


def build_bundle(df: pd.DataFrame) -> dict:
    """Precompute every small aggregate the standalone HTML dashboard needs (no raw rows shipped)."""
    d = df.dropna(subset=["preterm"]).copy()

    # Per-year sample fraction vs published singleton totals (the acquisition audit).
    yr = df.groupby("birth_year").size()
    frac = []
    for y in sorted(PUBLISHED_BIRTHS):
        singletons = PUBLISHED_BIRTHS[y] * 0.968
        frac.append({"year": int(y), "our_sample": int(yr.get(y, 0)),
                     "published_singletons": int(singletons),
                     "fraction_pct": round(yr.get(y, 0) / singletons * 100, 1)})

    # Distributions for the dashboard charts.
    age_c, age_e = np.histogram(d["mager"].dropna(), bins=range(12, 51))
    bmi_c, bmi_e = np.histogram(d["bmi"][d.bmi < 70].dropna(), bins=40)

    # Education x race intersection (the headline EDA finding).
    edu_race = (d[d.mother_race_ethnicity.isin(RACE_ORDER)].dropna(subset=["meduc"])
                .assign(meduc=lambda x: x.meduc.astype(int))
                .groupby(["mother_race_ethnicity", "meduc"])["preterm"].mean().mul(100)
                .rename("rate").reset_index().to_dict("records"))

    # Correlations of numeric fields with the target.
    num = ["mager", "meduc", "bmi", "cig_0", "priorlive", "priordead", "rf_ppterm",
           "rf_phype", "rf_pdiab", "rf_cesar", "wic", "first_trimester_care", "married"]
    corr = {k: round(float(v), 4) for k, v in
            d[num + ["preterm"]].corr()["preterm"].drop("preterm").sort_values().items()}

    # An exploratory medical-risk count (sum of the four pre-pregnancy flags), computed live for
    # the dose-response chart. This is exploration, not a committed engineered feature.
    risk_n = d[["rf_pdiab", "rf_phype", "rf_ppterm", "rf_cesar"]].sum(axis=1)
    dose = (d.assign(risk_n=risk_n).groupby("risk_n")["preterm"].mean().mul(100)
            .reset_index().to_dict("records"))

    smk = d.assign(s=np.where(d.cig_0.between(1, 89), "Smoker", "Non-smoker"))
    smoking = {k: round(float(v), 2) for k, v in
               smk.groupby("s")["preterm"].mean().mul(100).items()}

    # Eight readable real rows for the dashboard's "sample of the data" table.
    samp = df.sample(8, random_state=7).copy()
    samp["education"] = samp["meduc"].map(EDU)
    samp["payment"] = samp["pay"].map(PAY)
    sample_rows = samp[["birth_year", "mager", "education", "bmi", "cig_0", "priorlive",
                        "payment", "mother_race_ethnicity", "preterm"]].round(1).to_dict("records")

    # Sentinel / outlier audit.
    sentinels = {
        "bmi_ge_70": {"n": int((df.bmi >= 70).sum()), "pct": round((df.bmi >= 70).mean() * 100, 2)},
        "cig_ge_90": {"n": int((df.cig_0 >= 90).sum()), "pct": round((df.cig_0 >= 90).mean() * 100, 2)},
        "age_le_15": int((df.mager <= 15).sum()),
        "age_ge_45": int((df.mager >= 45).sum()),
        "priorlive_ge_10": int((df.priorlive >= 10).sum()),
    }

    miss = (df.isna().mean() * 100).round(2)
    missingness = {k: float(v) for k, v in miss[miss > 0].sort_values(ascending=False).items()}

    return {
        "meta": {
            "n_rows": int(len(df)),
            "n_fields": int(df.shape[1]),
            "years": [int(y) for y in sorted(df.birth_year.unique())],
            "overall_rate": round(float(d.preterm.mean() * 100), 2),
            "n_preterm": int(d.preterm.sum()),
            "n_missing_target": int(df.preterm.isna().sum()),
            "n_race_groups": int(df.mother_race_ethnicity.nunique(dropna=True)),
            "education_labels": EDU,
            "payment_labels": PAY,
            "race_order": RACE_ORDER,
            "source": "CDC/NCHS Natality public-use files (NBER mirror), complete files 2018-2024, singletons",
        },
        "acquisition_fraction": frac,
        "sample_rows": sample_rows,
        "rate_by_race": _rate(d, "mother_race_ethnicity"),
        "rate_by_year": _rate(d, "birth_year"),
        "prior_preterm": _rate(d, "rf_ppterm"),
        "risk_dose_response": dose,
        "smoking": smoking,
        "edu_race": edu_race,
        "corr": corr,
        "age_hist": {"counts": age_c.tolist(), "edges": [int(x) for x in age_e]},
        "bmi_hist": {"counts": bmi_c.tolist(), "edges": [round(float(x), 1) for x in bmi_e]},
        "edu_dist": {int(k): int(v) for k, v in
                     d["meduc"].dropna().astype(int).value_counts().sort_index().items()},
        "pay_dist": {int(k): int(v) for k, v in
                     d["pay"].dropna().astype(int).value_counts().sort_index().items()},
        "sentinels": sentinels,
        "missingness": missingness,
    }


def main() -> None:
    df = load_exploration_table()
    print(f"exploration table: {df.shape[0]:,} rows x {df.shape[1]} raw fields -> {EXPLORE_OUT.relative_to(ROOT)}")
    bundle = build_bundle(df)
    JSON_OUT.write_text(json.dumps(bundle, indent=2))
    print(f"wrote {JSON_OUT.relative_to(ROOT)} ({JSON_OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
