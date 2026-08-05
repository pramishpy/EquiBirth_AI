# Week 2 (Phase 2): feature engineering, leakage audit, and the equity audit

All numbers computed live on the COMPLETE 2018-2024 public-use files (24,920,407 singleton births with a valid target).

## Full-population outcome audit (every row)

Overall singleton preterm rate: **8.57%**. NH Black minus NH White gap: **4.74 percentage points**.

| race/ethnicity | preterm rate | n |
|---|---|---|
| NH White | 7.54% | 12,575,254 |
| NH Asian | 7.58% | 1,535,378 |
| Hispanic | 8.64% | 6,274,954 |
| NH Multiracial | 8.95% | 587,517 |
| NH AIAN | 10.68% | 179,698 |
| NH NHOPI | 10.80% | 67,182 |
| NH Black | 12.28% | 3,471,014 |

Race is a marker of exposure to racism and structural barriers, not a biological cause. These are descriptive population rates, not model outputs.

## Leakage audit (the central integrity check)

A classifier trained on the 50 EXCLUDED post-delivery variables alone reaches AUC **0.917** on a 400,000-row stratified sample. That is far above the 0.55 gate (FR6.1 fires correctly), which is precisely why those variables (birthweight, Apgar, labor interventions, total visits) are barred from the primary matrix: they are recorded at or after delivery, so a model that used them would look excellent and mean nothing.

The primary leakage-safe matrix has **103 features**, all pre-delivery. The FR6.2 composite-component guard passed (no engineered feature is built from an excluded variable), and the timing-conditional features kept in the model pass the FR6.3 univariate gate (0.6):

| timing-conditional feature | univariate AUC | within gate? |
|---|---|---|
| first_trimester_care | 0.545 | yes |
| wic | 0.511 | yes |

## Working modeling dataset (for the coming weeks)

A stratified 3,000,000-row working sample was drawn from the full census (preserving the outcome and race mix) and split deterministically. The held-out partition is untouched until final reporting.

| partition | n | prevalence |
|---|---|---|
| dev_train | 1,050,000 | 0.0856 |
| dev_val | 225,000 | 0.0859 |
| dev_test | 225,000 | 0.0857 |
| held_out_final | 1,500,000 | 0.0857 |
