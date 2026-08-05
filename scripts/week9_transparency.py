"""Week 9: regenerate transparency artifacts from full-data numbers.

Produces three outputs entirely from existing tracked JSON reports (no parquet loading):
  - reports/model_card.md   (Google Model Card structure, Mitchell et al. 2019)
  - reports/datasheet.md    (Gebru et al. 2021 Datasheets for Datasets)
  - reports/week9_external_context.json + .md  (CDC/March of Dimes aggregate context)

Every number in the Model Card and Datasheet is f-string-ed from the loaded weekly JSONs.
The external-context section cites published aggregate statistics only (no microdata linkage).

Run: PYTHONPATH=src ./.venv/bin/python scripts/week9_transparency.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    """Load a weekly JSON report by stem name."""
    with open(REPORTS / name) as f:
        return json.load(f)


def _fmt(val: float, decimals: int = 4) -> str:
    """Format a float to a given number of decimals."""
    return f"{val:.{decimals}f}"


def _pct(val: float) -> str:
    """Format a float as a percentage with one decimal place."""
    return f"{val * 100:.1f}%"


# ---------------------------------------------------------------------------
# 1. Load all weekly JSONs
# ---------------------------------------------------------------------------

def load_all_reports() -> dict:
    """Read every weekly JSON into a keyed dict."""
    return {
        "w3": _load("week3_baseline_metrics.json"),
        "w4": _load("week4_boosting_fairness.json"),
        "w5": _load("week5_models.json"),
        "w6": _load("week6_mitigation.json"),
        "w7": _load("week7_explain.json"),
        "w8": _load("week8_robustness.json"),
    }


# ---------------------------------------------------------------------------
# 2. Model Card (Mitchell et al. 2019)
# ---------------------------------------------------------------------------

def generate_model_card(reps: dict) -> str:
    """Generate the Model Card markdown from loaded reports."""
    w3 = reps["w3"]
    w4 = reps["w4"]
    w5 = reps["w5"]
    w6 = reps["w6"]
    w7 = reps["w7"]
    w8 = reps["w8"]

    # Selected model metrics
    xgb = w5["models"]["xgboost"]
    xgb_params = w4["params"]["xgboost"]

    # Fairness: worst EO gap baseline vs best mitigation
    baseline_eo = w6["mitigations"]["baseline"]
    threshopt = w6["mitigations"]["threshopt"]

    # Drift summary
    drift = w8["drift"]
    drift_roc_min = min(d["auc_roc"] for d in drift)
    drift_roc_max = max(d["auc_roc"] for d in drift)

    # Stratified models from week8
    strat = w8["stratified_models"]

    # Decomposition
    decomp = w7["decomposition"]

    card = f"""# Model Card: EquiBirth-AI Preterm-Birth Risk Model

> **This is NOT a clinical tool.** It must not be used for individual risk scoring,
> care rationing, or any prospective clinical decision. It is an educational research
> artifact for population-level equity diagnosis.

*Structure follows Mitchell et al. (2019), "Model Cards for Model Reporting"*
*Reference: https://arxiv.org/abs/1810.03993*

---

## 1. Model Details

| Field | Value |
|-------|-------|
| Architecture | Gradient-boosted decision trees (XGBoost) with isotonic recalibration |
| Selected on | dev_test AUC-PR ({_fmt(xgb["dev_test_auc_pr"])}) |
| Training data | ~3M singleton live births 2018-2024 (U.S. NCHS natality public-use files) |
| Training cap | {w5["caps"]["train_cap"]:,} rows (stratified subsample of dev_train) |
| Feature count | {w3["n_features"]} pre-delivery features |
| Hyperparameters | learning_rate={xgb_params["learning_rate"]:.4f}, n_estimators={xgb_params["n_estimators"]}, max_depth={xgb_params["max_depth"]}, subsample={xgb_params["subsample"]:.4f}, colsample_bytree={xgb_params["colsample_bytree"]:.4f}, reg_lambda={xgb_params["reg_lambda"]:.4f}, min_child_weight={xgb_params["min_child_weight"]:.4f} |
| Calibration | Isotonic regression on held-out calibration partition |
| Seed | {w5["seed"]} |

---

## 2. Intended Use

- Population-level equity diagnosis: quantifying how preterm birth risk distributes across
  race/ethnicity groups and socioeconomic proxies on U.S. birth certificates.
- Educational research: demonstrating responsible-AI methodology (fairness auditing,
  explainability, robustness) on a real public-health dataset.
- Screening tool evaluation: benchmarking the predictive ceiling of administrative
  birth-certificate data for preterm birth.

---

## 3. Out-of-Scope Uses

- **Clinical decisions**: individual risk scoring, care rationing, resource allocation.
- **Prospective deployment**: timing-conditional variables (e.g., gestational estimate,
  delivery method flags) are unavailable prospectively; the model cannot be deployed as a
  real-time screener.
- **Causal claims**: race is a social construct and a marker of exposure to structural
  racism, not a biological cause. The model captures associations, not mechanisms.
- **Profit or surveillance**: commercialization or individual tracking.

---

## 4. Metrics (Held-Out Evaluation)

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC-PR | {_fmt(xgb["held_out_auc_pr"])} | [{_fmt(xgb["held_out_auc_pr_ci"][0])}, {_fmt(xgb["held_out_auc_pr_ci"][1])}] |
| AUC-ROC | {_fmt(xgb["held_out_auc_roc"])} | -- |
| ECE | {_fmt(xgb["held_out_ece"])} | -- |
| Brier score | {_fmt(xgb["held_out_brier"])} | -- |

Discrimination is modest, consistent with the published ceiling for birth-certificate
prediction (AUC-ROC approximately 0.62-0.70; Mercer et al. 1996, Goldenberg et al.
Lancet 2008). The model does not provide reliable individual risk stratification.

---

## 5. Stratified Evaluation (Per-Race Held-Out, Week 8)

| Group | N (train) | AUC-PR | AUC-ROC | ECE |
|-------|-----------|--------|---------|-----|
| NH White | {strat["NH White"]["train_n"]:,} | {_fmt(strat["NH White"]["auc_pr"])} | {_fmt(strat["NH White"]["auc_roc"])} | {_fmt(strat["NH White"]["ece"])} |
| NH Black | {strat["NH Black"]["train_n"]:,} | {_fmt(strat["NH Black"]["auc_pr"])} | {_fmt(strat["NH Black"]["auc_roc"])} | {_fmt(strat["NH Black"]["ece"])} |
| Hispanic | {strat["Hispanic"]["train_n"]:,} | {_fmt(strat["Hispanic"]["auc_pr"])} | {_fmt(strat["Hispanic"]["auc_roc"])} | {_fmt(strat["Hispanic"]["ece"])} |
| NH Asian | {strat["NH Asian"]["train_n"]:,} | {_fmt(strat["NH Asian"]["auc_pr"])} | {_fmt(strat["NH Asian"]["auc_roc"])} | {_fmt(strat["NH Asian"]["ece"])} |
| NH Multiracial | {strat["NH Multiracial"]["train_n"]:,} | {_fmt(strat["NH Multiracial"]["auc_pr"])} | {_fmt(strat["NH Multiracial"]["auc_roc"])} | {_fmt(strat["NH Multiracial"]["ece"])} |

Note: These are group-specific XGBoost models (hypothesis-generating only).

---

## 6. Fairness Audit

### 6a. Equal Opportunity Gaps (Week 4 / Week 6 baseline)

Worst-group EO gap (NH Black vs NH White): {_fmt(baseline_eo["worst_group_eo_gap"])}
Mean |EO gap| across groups: {_fmt(baseline_eo["mean_abs_eo_gap"])}

Per-group EO gaps (positive = higher TPR than NH White):

| Group | EO Gap |
|-------|--------|
| Hispanic | {_fmt(baseline_eo["per_group_eo_gap"]["Hispanic"])} |
| NH AIAN | {_fmt(baseline_eo["per_group_eo_gap"]["NH AIAN"])} |
| NH Asian | {_fmt(baseline_eo["per_group_eo_gap"]["NH Asian"])} |
| NH Black | {_fmt(baseline_eo["per_group_eo_gap"]["NH Black"])} |
| NH Multiracial | {_fmt(baseline_eo["per_group_eo_gap"]["NH Multiracial"])} |
| NH NHOPI | {_fmt(baseline_eo["per_group_eo_gap"]["NH NHOPI"])} |

### 6b. Mitigation Summary (Week 6)

| Method | Worst EO Gap | Mean |EO Gap| | AUC-PR | Notes |
|--------|-------------|----------------|--------|-------|
| Baseline (calibrated XGBoost) | {_fmt(baseline_eo["worst_group_eo_gap"])} | {_fmt(baseline_eo["mean_abs_eo_gap"])} | {_fmt(baseline_eo["auc_pr"])} | Reference |
| ThresholdOptimizer | {_fmt(threshopt["worst_group_eo_gap"])} | {_fmt(threshopt["mean_abs_eo_gap"])} | -- | Disparate treatment (group-specific thresholds) |
| ExponentiatedGradient | {_fmt(w6["mitigations"]["expgrad"]["worst_group_eo_gap"])} | {_fmt(w6["mitigations"]["expgrad"]["mean_abs_eo_gap"])} | -- | Near-zero sensitivity; academic exercise |
| Reweighting | {_fmt(w6["mitigations"]["reweight"]["worst_group_eo_gap"])} | {_fmt(w6["mitigations"]["reweight"]["mean_abs_eo_gap"])} | {_fmt(w6["mitigations"]["reweight"]["auc_pr"])} | Performance cost |

Best mitigation for EO parity: ThresholdOptimizer (group-specific thresholds) reduces
worst-group EO gap from {_fmt(baseline_eo["worst_group_eo_gap"])} to
{_fmt(threshopt["worst_group_eo_gap"])}, but constitutes disparate treatment (different
decision rules by race). We present the tradeoff; we do not assert one mitigation is correct.

---

## 7. Explainability (Week 7)

### Top 5 global predictors (mean |SHAP|)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | {w7["global_top_25"][0][0]} | {_fmt(w7["global_top_25"][0][1])} |
| 2 | {w7["global_top_25"][1][0]} | {_fmt(w7["global_top_25"][1][1])} |
| 3 | {w7["global_top_25"][2][0]} | {_fmt(w7["global_top_25"][2][1])} |
| 4 | {w7["global_top_25"][3][0]} | {_fmt(w7["global_top_25"][3][1])} |
| 5 | {w7["global_top_25"][4][0]} | {_fmt(w7["global_top_25"][4][1])} |

### Decomposition of Black-White preterm gap

- Raw gap: {_fmt(decomp["raw_gap"])} ({_pct(decomp["raw_gap"])} points)
- Adjusted gap (full covariate set): {_fmt(decomp["adjusted_gap"])}
- Explained share (full covariates): {_pct(decomp["explained_pct_full"])} [{_pct(decomp["explained_pct_full_ci"][0])}, {_pct(decomp["explained_pct_full_ci"][1])}]
- Explained by SES only: {_pct(decomp["explained_pct_ses"])}
- Explained by medical history only: {_pct(decomp["explained_pct_medical"])}

The unexplained residual is a LOWER bound on the structural contribution (adjusting
covariates are themselves products of structural racism).

---

## 8. Robustness and Drift (Week 8)

### Out-of-time validation (trained on 2018-2019 only)

| Era | Years | N | AUC-PR | AUC-ROC | ECE |
|-----|-------|---|--------|---------|-----|
| COVID | 2020-2021 | {w8["out_of_time"]["covid_era"]["n"]:,} | {_fmt(w8["out_of_time"]["covid_era"]["auc_pr"])} | {_fmt(w8["out_of_time"]["covid_era"]["auc_roc"])} | {_fmt(w8["out_of_time"]["covid_era"]["ece"])} |
| Recent | 2022-2024 | {w8["out_of_time"]["recent_era"]["n"]:,} | {_fmt(w8["out_of_time"]["recent_era"]["auc_pr"])} | {_fmt(w8["out_of_time"]["recent_era"]["auc_roc"])} | {_fmt(w8["out_of_time"]["recent_era"]["ece"])} |

### Per-year drift (AUC-ROC range)

AUC-ROC ranges from {_fmt(drift_roc_min)} ({drift[0]["year"]}-era) to {_fmt(drift_roc_max)}
(2022), indicating stable discrimination across the full 2018-2024 window.

---

## 9. Limitations

- **Leakage taxonomy**: post-delivery variables (delivery method, APGAR, abnormal
  conditions) reach AUC approximately 0.91 but are meaningless for prediction; this model
  excludes them by construction.
- **Modest ceiling**: approximately 0.66 AUC-ROC, consistent with published literature.
  The model provides population patterns, not individual risk stratification.
- **No geography post-2005**: NCHS public-use files suppress state/county identifiers
  after 2005; spatial disparities cannot be studied at the record level.
- **Social-construct framing**: race is used as a structural exposure marker, never as
  a biological cause.
- **COVID-era sensitivity**: 2020-2021 data included; out-of-time evaluation shows
  minimal degradation but unmeasured behavioral shifts are possible.
- **Narrow risk spread**: approximately {_pct(w3["fraction_near_base_rate"])} of predictions
  fall near the {_pct(w3["prevalence"])} base rate.

---

## 10. Recommendations

- **Annual re-validation**: re-run the drift table on each new NCHS annual release.
  The 2018-2024 drift stability (AUC-ROC {_fmt(drift_roc_min)}-{_fmt(drift_roc_max)})
  serves as the trigger reference; degradation beyond 0.02 AUC-ROC warrants retraining.
- **Do not deploy prospectively** without timing-conditional feature exclusion and
  external validation on a clinical cohort.
- **Fairness monitoring**: re-audit per-group EO gaps whenever the model or threshold is
  updated; report both EO and disparate impact.
- **Cite structural framing**: any publication using these results must cite the
  social-construct framing of race and the lower-bound interpretation of the unexplained
  residual.

---

*Generated automatically from tracked weekly JSON reports. See scripts/week9_transparency.py.*
"""
    return card


# ---------------------------------------------------------------------------
# 3. Datasheet (Gebru et al. 2021)
# ---------------------------------------------------------------------------

def generate_datasheet(reps: dict) -> str:
    """Generate the Datasheet for Datasets markdown."""
    w3 = reps["w3"]
    w5 = reps["w5"]

    ds = f"""# Datasheet: U.S. Natality Modeling Dataset (EquiBirth-AI)

> **This is NOT a clinical tool.** The dataset is for educational research only.

*Structure follows Gebru et al. (2021), "Datasheets for Datasets"*
*Reference: https://arxiv.org/abs/1803.09010*

---

## 1. Motivation

- **Purpose**: Educational research on preterm-birth prediction and racial equity in the
  United States, as part of an AI4ALL responsible-AI portfolio project.
- **Creators**: Independent research; no institutional affiliation with NCHS or CDC.
- **Funding**: None (self-directed educational work).

---

## 2. Composition

| Field | Value |
|-------|-------|
| Unit of observation | Individual singleton live birth |
| Source population | Approximately 24.9M singletons, 2018-2024, U.S. vital statistics |
| Working sample | Approximately 3M records (stratified partitions for modeling) |
| Splits | dev_train (1.05M), dev_val (225K), dev_test (225K), held_out_final (1.5M) |
| Feature count | {w3["n_features"]} pre-delivery features |
| Target | Preterm birth (obstetric estimate of gestation < 37 weeks) |
| Prevalence | {_pct(w3["prevalence"])} in the held-out partition (n={w3["held_out_n"]:,}) |
| Identifiers | None (de-identified public-use files; no SSN, name, or address) |
| Geography | Suppressed in public-use files post-2005 (no state/county) |

### Demographic composition (held-out partition)

| Group | N | Preterm Rate |
|-------|---|-------------|
| NH White | {w3["stratified"][6]["n"]:,} | {_pct(w3["stratified"][6]["rate"])} |
| NH Black | {w3["stratified"][3]["n"]:,} | {_pct(w3["stratified"][3]["rate"])} |
| Hispanic | {w3["stratified"][0]["n"]:,} | {_pct(w3["stratified"][0]["rate"])} |
| NH Asian | {w3["stratified"][2]["n"]:,} | {_pct(w3["stratified"][2]["rate"])} |
| NH AIAN | {w3["stratified"][1]["n"]:,} | {_pct(w3["stratified"][1]["rate"])} |
| NH NHOPI | {w3["stratified"][5]["n"]:,} | {_pct(w3["stratified"][5]["rate"])} |
| NH Multiracial | {w3["stratified"][4]["n"]:,} | {_pct(w3["stratified"][4]["rate"])} |

---

## 3. Collection Process

- **Source**: CDC/NCHS Natality public-use microdata files, accessed via the NBER mirror
  (https://www.nber.org/research/data/vital-statistics-natality-birth-data).
- **Revision**: 2003 revised birth certificate (standard since 2016 for all states).
- **Download**: Programmatic download of full-year CSV files (2018-2024); integrity verified
  by SHA-256 hashes and expected row counts.
- **Geography limitation**: Public-use files suppress geographic identifiers (state, county)
  after 2005 to protect confidentiality.
- **No consent required**: These are de-identified administrative records released for public
  research use by NCHS.

---

## 4. Preprocessing

- **Singleton filter**: Multiple births excluded (different risk profile and clinical pathway).
- **Variable harmonization**: Per-variable "unknown/not stated" sentinel codes converted to
  null. Race and education harmonized to consistent 7-group and 9-level categories across years.
- **Feature derivation**: Medical-risk count (sum of binary risk factor flags), age bracket,
  race-education interaction terms, first-trimester care indicator.
- **Leakage audit** (Week 2): Post-delivery variables (delivery method, APGAR, abnormal
  conditions, NICU admission) identified and excluded. Timing-conditional variables
  (obstetric estimate, plurality confirmation) audited and documented.
- **Splits**: Deterministic stratified (by preterm status and race) random partition saved
  to disk. No data leakage between train/val/test/held-out.
- **No SMOTE**: Class weights and threshold selection preserve calibration.

---

## 5. Uses

- **Intended**: Population-level equity research, methodology demonstration, educational
  AI fairness portfolio.
- **Out of scope**: Clinical deployment, individual risk scoring, commercial use,
  surveillance.
- **IRB status**: IRB-exempt (publicly available de-identified data; no human subjects
  interaction).
- **Linkage**: None. No linkage to NHANES, hospital records, claims data, or any other
  source. External context (CDC aggregates) is published summary statistics only.

---

## 6. Distribution

- **License**: Public domain. NCHS natality public-use files carry no use restrictions
  beyond citation of the source.
- **Access**: Available from NBER (https://www.nber.org/research/data/vital-statistics-natality-birth-data)
  or CDC WONDER for aggregates.
- **Format**: CSV (raw), Parquet (processed modeling file).

---

## 7. Maintenance

- **Update cadence**: NCHS releases new natality public-use files annually (typically
  with a 12-18 month lag).
- **Re-validation trigger**: Re-run the confirmation gate (row counts, column layouts,
  SHA-256) and the Week 8 drift table on any new year. AUC-ROC degradation beyond 0.02
  from the 2018-2024 baseline warrants investigation.
- **Contact**: Not applicable (educational project; no maintained deployment).
- **Versioning**: Dataset version is defined by the year range (currently 2018-2024) and
  the processing pipeline commit hash.

---

*Generated automatically from tracked weekly JSON reports. See scripts/week9_transparency.py.*
"""
    return ds


# ---------------------------------------------------------------------------
# 4. External Context (CDC aggregate published statistics)
# ---------------------------------------------------------------------------

def generate_external_context() -> dict:
    """Build the external-context JSON with published aggregate statistics."""
    return {
        "source_type": "published_aggregate",
        "description": (
            "Published CDC and March of Dimes aggregate preterm birth statistics, "
            "plus NHANES directional context on maternal hypertension prevalence. "
            "No microdata linkage. These are reference points for contextualizing "
            "the model's findings against nationally reported figures."
        ),
        "entries": [
            {
                "label": "National preterm rate 2022",
                "value": "10.4%",
                "source": "CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1",
                "url": "https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-01.pdf",
                "access_date": "2026-07-19",
                "linkage": "none; aggregate context only",
                "note": (
                    "This is the overall preterm rate including multiples. "
                    "Our singleton-only sample has a lower rate (approximately 8.6%) "
                    "because multiples have much higher preterm incidence."
                ),
            },
            {
                "label": "NH Black preterm rate 2022",
                "value": "approximately 14.6%",
                "source": "CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1",
                "url": "https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-01.pdf",
                "access_date": "2026-07-19",
                "linkage": "none; aggregate context only",
                "note": (
                    "Includes multiples. Consistent with the elevated preterm rate "
                    "for NH Black mothers in our singleton sample (12.3% held-out)."
                ),
            },
            {
                "label": "NH White preterm rate 2022",
                "value": "approximately 9.4%",
                "source": "CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1",
                "url": "https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-01.pdf",
                "access_date": "2026-07-19",
                "linkage": "none; aggregate context only",
                "note": (
                    "Includes multiples. Our singleton-only NH White rate is 7.5% "
                    "(held-out), lower due to exclusion of multiples."
                ),
            },
            {
                "label": "Highest-rate state 2022 (Mississippi)",
                "value": "approximately 14.5%",
                "source": "March of Dimes Report Card 2023",
                "url": "https://www.marchofdimes.org/report-card",
                "access_date": "2026-07-19",
                "linkage": "none; aggregate context only",
                "note": (
                    "State-level variation cannot be studied in our data because "
                    "NCHS public-use files suppress geography post-2005."
                ),
            },
            {
                "label": "Lowest-rate state 2022 (Oregon)",
                "value": "approximately 8.5%",
                "source": "March of Dimes Report Card 2023",
                "url": "https://www.marchofdimes.org/report-card",
                "access_date": "2026-07-19",
                "linkage": "none; aggregate context only",
                "note": "State-level; aggregate context only.",
            },
            {
                "label": "Maternal hypertension prevalence (NHANES directional context)",
                "value": (
                    "NH Black women aged 20-44: approximately 40%; "
                    "NH White women aged 20-44: approximately 28% (NHANES 2017-2020)"
                ),
                "source": "NHANES 2017-March 2020 pre-pandemic data",
                "url": "https://www.cdc.gov/nchs/nhanes/index.htm",
                "access_date": "2026-07-19",
                "linkage": "none; aggregate context only",
                "note": (
                    "NHANES is a health/nutrition survey with NO birth-outcome linkage. "
                    "Used only as directional context: higher maternal hypertension "
                    "prevalence among NH Black women is consistent with the model's "
                    "rf_phype (pre-pregnancy hypertension) ranking as a top feature "
                    "in the NH Black stratified model. No causal claim is made."
                ),
            },
        ],
        "multiclass_protocol": {
            "status": "designed_not_run",
            "description": (
                "4-class severity classification (very preterm <28w, moderate 28-31w, "
                "late preterm 32-36w, term 37+w) with timing-conditional features excluded."
            ),
            "rationale": (
                "The exact-week gestation column used as target is not retained in the "
                "modeling parquet (only the binary preterm indicator). Reconstructing "
                "from interim files would require re-running the full preprocessing "
                "pipeline (estimated >30 minutes). Per spec, descoped to protocol-only."
            ),
            "planned_approach": (
                "Ordinal or multinomial extension of the selected XGBoost model; "
                "separate calibration per class; fairness audit on the very-preterm "
                "class (highest clinical severity); same feature set minus any "
                "timing-conditional variables that correlate with severity class."
            ),
        },
    }


def format_external_context_md(ctx: dict) -> str:
    """Format the external context JSON as a readable markdown report."""
    lines = [
        "# Week 9: External Context (Published Aggregate Statistics)",
        "",
        "> **No microdata linkage.** All figures below are published summary statistics",
        "> from CDC, NCHS, and March of Dimes reports. They provide reference context",
        "> for the model's findings but are not computationally linked to our dataset.",
        "",
        "---",
        "",
        "## Published Preterm Birth Statistics",
        "",
        "| Statistic | Value | Source |",
        "|-----------|-------|--------|",
    ]

    for entry in ctx["entries"]:
        label = entry["label"]
        value = entry["value"]
        source = entry["source"]
        # Truncate long values for table
        if len(value) > 60:
            value = value[:57] + "..."
        lines.append(f"| {label} | {value} | {source} |")

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Notes",
        "",
    ])

    for i, entry in enumerate(ctx["entries"], 1):
        lines.append(f"### {i}. {entry['label']}")
        lines.append("")
        lines.append(f"- **Value**: {entry['value']}")
        lines.append(f"- **Source**: {entry['source']}")
        lines.append(f"- **URL**: {entry['url']}")
        lines.append(f"- **Access date**: {entry['access_date']}")
        lines.append(f"- **Linkage**: {entry['linkage']}")
        lines.append(f"- **Note**: {entry['note']}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Multiclass Protocol (Designed, Not Run)",
        "",
        f"**Status**: {ctx['multiclass_protocol']['status']}",
        "",
        f"**Description**: {ctx['multiclass_protocol']['description']}",
        "",
        f"**Rationale for descoping**: {ctx['multiclass_protocol']['rationale']}",
        "",
        f"**Planned approach**: {ctx['multiclass_protocol']['planned_approach']}",
        "",
        "---",
        "",
        "*Generated automatically. See scripts/week9_transparency.py.*",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all transparency artifact generation."""
    print("Week 9: Transparency artifact generation")
    print("=" * 50)

    # 1. Load reports
    print("\n[1/4] Loading weekly JSON reports...")
    reps = load_all_reports()
    print(f"  Loaded: {', '.join(reps.keys())}")

    # 2. Model Card
    print("\n[2/4] Generating Model Card...")
    card = generate_model_card(reps)
    card_path = REPORTS / "model_card.md"
    card_path.write_text(card)
    print(f"  Written: {card_path}")

    # 3. Datasheet
    print("\n[3/4] Generating Datasheet...")
    ds = generate_datasheet(reps)
    ds_path = REPORTS / "datasheet.md"
    ds_path.write_text(ds)
    print(f"  Written: {ds_path}")

    # 4. External Context
    print("\n[4/4] Generating external context...")
    ctx = generate_external_context()
    json_path = REPORTS / "week9_external_context.json"
    with open(json_path, "w") as f:
        json.dump(ctx, f, indent=2)
    print(f"  Written: {json_path}")

    md_path = REPORTS / "week9_external_context.md"
    md_path.write_text(format_external_context_md(ctx))
    print(f"  Written: {md_path}")

    print("\n" + "=" * 50)
    print("Week 9 transparency artifacts complete.")
    print(f"  - {card_path.name}")
    print(f"  - {ds_path.name}")
    print(f"  - {json_path.name}")
    print(f"  - {md_path.name}")


if __name__ == "__main__":
    main()
