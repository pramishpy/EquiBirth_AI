# Week 7: SHAP explainability and Black-White gap decomposition

## Global SHAP (TreeSHAP on raw xgboost, n=30,000)

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | priorlive | 0.16080 |
| 2 | medical_risk_count | 0.13381 |
| 3 | mager | 0.11489 |
| 4 | rf_ppterm | 0.10758 |
| 5 | bmi | 0.10391 |
| 6 | married | 0.09050 |
| 7 | meduc | 0.08782 |
| 8 | mother_race_ethnicity_NH White | 0.05694 |
| 9 | mother_race_ethnicity_NH Black | 0.05178 |
| 10 | wic | 0.04446 |
| 11 | pay_1 | 0.03002 |
| 12 | rf_phype | 0.02939 |
| 13 | first_trimester_care | 0.02881 |
| 14 | cig_0 | 0.02598 |
| 15 | rf_cesar | 0.01673 |
| 16 | pay_3 | 0.01497 |
| 17 | rf_pdiab | 0.01411 |
| 18 | priordead | 0.00776 |
| 19 | race_education_NH White_edu7 | 0.00684 |
| 20 | pay_2 | 0.00530 |
| 21 | race_education_NH Black_edu3 | 0.00282 |
| 22 | race_education_NH White_edu6 | 0.00231 |
| 23 | pay_5 | 0.00189 |
| 24 | mother_race_ethnicity_Hispanic | 0.00182 |
| 25 | race_education_NH White_edu5 | 0.00139 |

## Race-stratified top-10 features

### NH White

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | priorlive | 0.17999 |
| 2 | medical_risk_count | 0.13864 |
| 3 | bmi | 0.11503 |
| 4 | rf_ppterm | 0.10749 |
| 5 | mager | 0.10707 |
| 6 | meduc | 0.09440 |
| 7 | married | 0.08725 |
| 8 | mother_race_ethnicity_NH White | 0.05894 |
| 9 | mother_race_ethnicity_NH Black | 0.03370 |
| 10 | pay_1 | 0.03297 |

### NH Black

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | mother_race_ethnicity_NH Black | 0.15942 |
| 2 | priorlive | 0.13128 |
| 3 | medical_risk_count | 0.12169 |
| 4 | mager | 0.11942 |
| 5 | rf_ppterm | 0.11824 |
| 6 | married | 0.11089 |
| 7 | bmi | 0.08817 |
| 8 | meduc | 0.07130 |
| 9 | wic | 0.06600 |
| 10 | mother_race_ethnicity_NH White | 0.04791 |

### Hispanic

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | priorlive | 0.13995 |
| 2 | medical_risk_count | 0.13347 |
| 3 | mager | 0.12458 |
| 4 | rf_ppterm | 0.10403 |
| 5 | bmi | 0.08906 |
| 6 | married | 0.08751 |
| 7 | meduc | 0.07779 |
| 8 | wic | 0.05880 |
| 9 | mother_race_ethnicity_NH White | 0.05552 |
| 10 | mother_race_ethnicity_NH Black | 0.03471 |

### NH Asian

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | priorlive | 0.17015 |
| 2 | medical_risk_count | 0.12749 |
| 3 | mager | 0.12472 |
| 4 | meduc | 0.11445 |
| 5 | bmi | 0.11038 |
| 6 | rf_ppterm | 0.09731 |
| 7 | married | 0.08310 |
| 8 | mother_race_ethnicity_NH White | 0.06899 |
| 9 | wic | 0.03501 |
| 10 | mother_race_ethnicity_NH Black | 0.03295 |

### NH AIAN

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | medical_risk_count | 0.13855 |
| 2 | priorlive | 0.13000 |
| 3 | mager | 0.12731 |
| 4 | rf_ppterm | 0.11900 |
| 5 | bmi | 0.08561 |
| 6 | married | 0.08371 |
| 7 | meduc | 0.06871 |
| 8 | wic | 0.06067 |
| 9 | mother_race_ethnicity_NH White | 0.04674 |
| 10 | cig_0 | 0.04020 |

### NH NHOPI

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | priorlive | 0.13731 |
| 2 | medical_risk_count | 0.13121 |
| 3 | rf_ppterm | 0.12737 |
| 4 | mager | 0.12031 |
| 5 | bmi | 0.09468 |
| 6 | married | 0.09076 |
| 7 | meduc | 0.06556 |
| 8 | wic | 0.06126 |
| 9 | mother_race_ethnicity_NH White | 0.05320 |
| 10 | first_trimester_care | 0.04245 |

### NH Multiracial

| rank | feature | mean |SHAP| |
|---|---|---|
| 1 | priorlive | 0.14190 |
| 2 | medical_risk_count | 0.12340 |
| 3 | mager | 0.12007 |
| 4 | rf_ppterm | 0.10266 |
| 5 | bmi | 0.09974 |
| 6 | married | 0.08530 |
| 7 | meduc | 0.07399 |
| 8 | mother_race_ethnicity_NH White | 0.05290 |
| 9 | wic | 0.04931 |
| 10 | mother_race_ethnicity_NH Black | 0.03626 |

## NH Black minus NH White gap decomposition

- Raw gap: 0.04912 (NH Black 0.07694 rate difference observed)
- Adjusted (unexplained) gap: 0.02783
- Explained share (full covariates): 43.3% [95% CI: 40.2%, 47.3%]
- Explained share (SES-only: education, payment, WIC): 21.6%
- Explained share (medical-only: risk factors, smoking, BMI): 19.2%
- Range across covariate sets: [19.2%, 43.3%]
- n focal (NH Black): 27,802
- n reference (NH White): 101,119

## Structural-racism caveat

The measured covariates (maternal education, payment source, WIC participation, medical risk factors, smoking, BMI) are themselves products of structural racism. Residential segregation restricts access to quality prenatal care; wealth gaps rooted in historical exclusion shape education and insurance; chronic stress from experienced racism elevates hypertensive risk. Consequently, the 'explained' share of the gap overstates genuine covariate-mediated pathways and the 'unexplained' residual is a LOWER BOUND on structural contributions. This decomposition quantifies statistical association between measured SES/medical factors and the preterm disparity. It does not prove mechanism, does not exonerate the system where covariates 'explain' the gap, and must not be read as evidence that individual behavior accounts for the disparity.
