# Week 6: fairness mitigation tradeoff table

Three mitigations priced against the xgboost baseline on a shared 100,000-row held-out subsample. Nothing here 'solves' fairness; each mitigation buys equal-opportunity improvement at a disclosed cost to overall discrimination or calibration. The ThresholdOptimizer is disparate treatment (different decision rules by group) and is reported as an analytical bound, not a deployment recommendation.

Screening threshold: 0.0784 (from Week 4, target sensitivity 0.60). Reference group: NH White.

## Tradeoff summary

| method | AUC-PR | AUC-ROC | ECE | sensitivity | specificity | worst EO gap | mean |EO gap| |
|---|---|---|---|---|---|---|---|
| baseline | 0.1723 | 0.6563 | 0.0007 | 0.607 | 0.606 | 0.389 | 0.139 |
| reweight | 0.1656 | 0.6430 | 0.0004 | 0.609 | 0.586 | 0.504 | 0.115 |
| expgrad | - | - | - | 0.010 | 0.999 | 0.072 | 0.018 |
| threshopt | - | - | - | 0.502 | 0.701 | 0.173 | 0.056 |

## Per-group equal-opportunity gap (TPR difference vs NH White)

| group | baseline | reweight | expgrad | threshopt |
|---|---|---|---|---|
| Hispanic | +0.084 | -0.017 | +0.001 | +0.015 |
| NH AIAN | +0.250 | -0.103 | +0.072 | +0.058 |
| NH Asian | -0.014 | +0.008 | +0.006 | +0.009 |
| NH Black | +0.389 | +0.017 | +0.019 | -0.007 |
| NH Multiracial | +0.046 | -0.039 | +0.005 | -0.073 |
| NH NHOPI | +0.050 | -0.504 | -0.005 | -0.173 |

## Per-group demographic parity diff (positive rate difference vs NH White)

| group | baseline | reweight | expgrad | threshopt |
|---|---|---|---|---|
| Hispanic | +0.137 | +0.040 | +0.000 | +0.068 |
| NH AIAN | +0.211 | -0.109 | +0.015 | +0.008 |
| NH Asian | +0.012 | +0.039 | +0.000 | +0.034 |
| NH Black | +0.521 | +0.036 | +0.005 | +0.013 |
| NH Multiracial | +0.120 | -0.003 | +0.002 | -0.035 |
| NH NHOPI | +0.216 | -0.233 | +0.006 | -0.068 |

## Notes

- Reweighting equalizes (group x outcome) cell mass during training; the model still applies a single threshold (no disparate treatment). The cost is a small AUC-PR drop.
- ExponentiatedGradient uses a logistic base learner under a TPR-parity constraint. It produces hard 0/1 decisions, so AUC-PR is not available. The logistic learner is weaker than xgboost, so sensitivity drops; the gain is in EO gap reduction.
- ThresholdOptimizer learns group-specific thresholds on precomputed xgboost scores. It can equalize TPR almost perfectly, but at the cost of applying different rules to different groups (disparate treatment). This is reported as an analytical bound on what post-processing can achieve.
- Race is a marker of exposure to racism, not a biological cause; nothing here is a clinical tool.
