# Week 3 (Phase 3) baseline metrics (full-data held-out)

Held-out n=1,498,760, prevalence=0.0857. AUC-ROC, Brier, and ECE point estimates use the full held-out; AUC-PR (point + 2000-resample bootstrap CI) uses a fixed 250,000-row held-out subsample (each AUC-PR resample on 1.5M rows is costly, and the estimate is converged well before this size). Primary metric AUC-PR. Calibration trained on dev_test; raw vs calibrated ECE shown. MCE uses a 30-row minimum bin. Quantile-bin ECE (equal mass per bin) is the stricter calibration summary.

| model | AUC-PR [95% CI] | AUC-ROC | Brier | ECE raw | ECE cal | ECE quantile | MCE |
|---|---|---|---|---|---|---|---|
| logistic_regression | 0.159 [0.155, 0.162] | 0.634 | 0.0759 | 0.3876 | 0.0009 | 0.0008 | 0.115 |
| decision_tree | 0.150 [0.147, 0.154] | 0.626 | 0.0760 | 0.3890 | 0.0009 | 0.0011 | 0.032 |
| random_forest | 0.166 [0.163, 0.170] | 0.651 | 0.0756 | 0.3825 | 0.0009 | 0.0012 | 0.133 |

Risk-spread diagnostic: 83% of random-forest predictions fall within +/-5pp of the 8.6% base rate, so low ECE reflects aggregate calibration, not individual risk stratification.

## Race-stratified (logistic_regression, held-out)

| group | n | positives | rate | AUC-ROC | AUC-PR | ECE | status |
|---|---|---|---|---|---|---|---|
| Hispanic | 377,388 | 32616 | 0.086 | 0.605 | 0.147 | 0.0014 | ok |
| NH AIAN | 10,806 | 1154 | 0.107 | 0.627 | 0.203 | 0.0035 | ok |
| NH Asian | 92,342 | 7001 | 0.076 | 0.599 | 0.129 | 0.0027 | ok |
| NH Black | 208,755 | 25636 | 0.123 | 0.620 | 0.209 | 0.0046 | ok |
| NH Multiracial | 35,333 | 3160 | 0.089 | 0.618 | 0.159 | 0.0038 | ok |
| NH NHOPI | 4,040 | 438 | 0.108 | 0.611 | 0.175 | 0.0098 | wide_ci |
| NH White | 756,298 | 57036 | 0.075 | 0.631 | 0.145 | 0.0009 | ok |
