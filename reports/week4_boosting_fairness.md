# Week 4 (Phase 4): boosting + first fairness audit (full-data held-out)

Best booster (selected on dev_test AUC-PR): **lightgbm**. Held-out (n=1,498,760) AUC-PR 0.174, AUC-ROC 0.657, ECE 0.0002.
Operating point: screening threshold 0.0784 (target sensitivity 0.6); held-out sensitivity 0.604, specificity 0.615.

Fairness computed on a fixed 300,000-row held-out subsample (converged CIs; the O(n)-per-iteration bootstrap and permutation loops are prohibitive on the full 1.5M). Versus NH White (equal-opportunity = TPR gap; a NEGATIVE value means the model misses more real preterm cases for that group; a large disparate-impact ratio means that group is flagged more often):

| group | n | pos | demographic parity diff | equal-opp diff [95% CI] | disparate impact | perm p | BH p |
|---|---|---|---|---|---|---|---|
| Hispanic | 75,321 | 6486 | +0.133 | +0.087 [+0.072, +0.103] | 1.455 | 0.002 | 0.002* |
| NH AIAN | 2,169 | 226 | +0.217 | +0.233 [+0.177, +0.289] | 1.743 | 0.002 | 0.002* |
| NH Asian | 18,351 | 1442 | +0.002 | -0.022 [-0.047, +0.005] | 1.009 | 0.116 | 0.116 |
| NH Black | 41,840 | 5051 | +0.508 | +0.393 [+0.381, +0.408] | 2.740 | 0.002 | 0.002* |
| NH Multiracial | 7,173 | 632 | +0.136 | +0.091 [+0.050, +0.133] | 1.466 | 0.002 | 0.002* |
| NH NHOPI | 801 | 94 | +0.267 | +0.184 [+0.095, +0.285] | 1.915 | 0.002 | 0.002* |

* BH-adjusted p < 0.05. Equal opportunity is the primary fairness metric; calibration-by-group (Week 3) shows the risk scores are well-calibrated in aggregate across large groups. This is one model at one operating point, not a final claim; the full mitigation and decomposition analysis comes later. Race is a marker of exposure to racism, not a biological cause; nothing here is a clinical tool.
