# Model Card: EquiBirth-AI Preterm-Birth Risk Model

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
| Selected on | dev_test AUC-PR (0.1738) |
| Training data | ~3M singleton live births 2018-2024 (U.S. NCHS natality public-use files) |
| Training cap | 500,000 rows (stratified subsample of dev_train) |
| Feature count | 103 pre-delivery features |
| Hyperparameters | learning_rate=0.0203, n_estimators=600, max_depth=5, subsample=0.7945, colsample_bytree=0.9991, reg_lambda=1.5882, min_child_weight=4.8153 |
| Calibration | Isotonic regression on held-out calibration partition |
| Seed | 42 |

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
| AUC-PR | 0.1720 | [0.1661, 0.1780] |
| AUC-ROC | 0.6561 | -- |
| ECE | 0.0015 | -- |
| Brier score | 0.0752 | -- |

Discrimination is modest, consistent with the published ceiling for birth-certificate
prediction (AUC-ROC approximately 0.62-0.70; Mercer et al. 1996, Goldenberg et al.
Lancet 2008). The model does not provide reliable individual risk stratification.

---

## 5. Stratified Evaluation (Per-Race Held-Out, Week 8)

| Group | N (train) | AUC-PR | AUC-ROC | ECE |
|-------|-----------|--------|---------|-----|
| NH White | 100,000 | 0.1579 | 0.6539 | 0.0015 |
| NH Black | 100,000 | 0.2127 | 0.6364 | 0.0044 |
| Hispanic | 100,000 | 0.1494 | 0.6200 | 0.0021 |
| NH Asian | 64,692 | 0.1270 | 0.6069 | 0.0035 |
| NH Multiracial | 24,756 | 0.1500 | 0.6132 | 0.0103 |

Note: These are group-specific XGBoost models (hypothesis-generating only).

---

## 6. Fairness Audit

### 6a. Equal Opportunity Gaps (Week 4 / Week 6 baseline)

Worst-group EO gap (NH Black vs NH White): 0.3893
Mean |EO gap| across groups: 0.1391

Per-group EO gaps (positive = higher TPR than NH White):

| Group | EO Gap |
|-------|--------|
| Hispanic | 0.0843 |
| NH AIAN | 0.2504 |
| NH Asian | -0.0145 |
| NH Black | 0.3893 |
| NH Multiracial | 0.0464 |
| NH NHOPI | 0.0496 |

### 6b. Mitigation Summary (Week 6)

| Method | Worst EO Gap | Mean |EO Gap| | AUC-PR | Notes |
|--------|-------------|----------------|--------|-------|
| Baseline (calibrated XGBoost) | 0.3893 | 0.1391 | 0.1723 | Reference |
| ThresholdOptimizer | 0.1729 | 0.0557 | -- | Disparate treatment (group-specific thresholds) |
| ExponentiatedGradient | 0.0720 | 0.0178 | -- | Near-zero sensitivity; academic exercise |
| Reweighting | 0.5042 | 0.1147 | 0.1656 | Performance cost |

Best mitigation for EO parity: ThresholdOptimizer (group-specific thresholds) reduces
worst-group EO gap from 0.3893 to
0.1729, but constitutes disparate treatment (different
decision rules by race). We present the tradeoff; we do not assert one mitigation is correct.

---

## 7. Explainability (Week 7)

### Top 5 global predictors (mean |SHAP|)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | priorlive | 0.1608 |
| 2 | medical_risk_count | 0.1338 |
| 3 | mager | 0.1149 |
| 4 | rf_ppterm | 0.1076 |
| 5 | bmi | 0.1039 |

### Decomposition of Black-White preterm gap

- Raw gap: 0.0491 (4.9% points)
- Adjusted gap (full covariate set): 0.0278
- Explained share (full covariates): 43.3% [40.2%, 47.3%]
- Explained by SES only: 21.6%
- Explained by medical history only: 19.2%

The unexplained residual is a LOWER bound on the structural contribution (adjusting
covariates are themselves products of structural racism).

---

## 8. Robustness and Drift (Week 8)

### Out-of-time validation (trained on 2018-2019 only)

| Era | Years | N | AUC-PR | AUC-ROC | ECE |
|-----|-------|---|--------|---------|-----|
| COVID | 2020-2021 | 424,074 | 0.1688 | 0.6527 | 0.0016 |
| Recent | 2022-2024 | 635,489 | 0.1721 | 0.6545 | 0.0017 |

### Per-year drift (AUC-ROC range)

AUC-ROC ranges from 0.6556 (2018-era) to 0.6693
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
- **Narrow risk spread**: approximately 83.0% of predictions
  fall near the 8.6% base rate.

---

## 10. Recommendations

- **Annual re-validation**: re-run the drift table on each new NCHS annual release.
  The 2018-2024 drift stability (AUC-ROC 0.6556-0.6693)
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
