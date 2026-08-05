# Week 5: deep benchmark, stacking, final selection (full-data held-out)

**Selected model: xgboost** (by dev_test AUC-PR, tie-break booster > stack > nets within bootstrap noise). xgboost selected on dev_test AUC-PR 0.1738 (held-out 0.1720); the stack reached 0.1760 and the best deep net (tabnet) 0.1297, consistent with the D41 expectation that the tuned boosters already sit at the administrative-data ceiling.

All five models are scored on an identical 100,000-row stratified draw from the untouched held_out_final partition, so the comparison and the stacked meta-learner see the same rows (the FT-Transformer does an O(n) forward pass per epoch on CPU, so a shared held-out draw keeps the benchmark feasible and fair). Deep nets train on a 30,000-row dev_train subsample; all caps are disclosed. AUC-PR CIs are from a stratified bootstrap (n=1000).

| model | dev_test AUC-PR | held-out AUC-PR [95% CI] | held-out AUC-ROC | ECE | Brier |
|---|---|---|---|---|---|
| xgboost | 0.1738 | 0.1720 [0.1661, 0.1780] | 0.6561 | 0.0015 | 0.0752 |
| lightgbm | 0.1721 | 0.1699 [0.1643, 0.1757] | 0.6550 | 0.0008 | 0.0753 |
| tabnet | 0.1297 | 0.1299 [0.1260, 0.1344] | 0.5916 | 0.0012 | 0.0769 |
| ft_transformer | 0.1179 | 0.1201 [0.1167, 0.1234] | 0.5909 | 0.0024 | 0.0774 |
| stack | 0.1760 | 0.1713 [0.1655, 0.1770] | 0.6560 | 0.0020 | 0.0753 |

The deep nets were run as honest benchmarks expected to lose (DECISION_LOG D41): weak tabular signal (max |corr| ~0.13), 8.57% prevalence, and an administrative-data ceiling of 0.62-0.70 AUC-ROC that the tuned boosters already reach. What actually happened: the best booster (xgboost, held-out AUC-PR 0.1720) led the best deep net (tabnet, 0.1299) by +0.0421 AUC-PR, and the stack reached 0.1713, so combining the models did not buy a meaningful margin over the best single booster.

Training caps (disclosed): boosters trained on 500K-capped dev_train, deep nets on 30K-capped dev_train; all five models scored on identical rows: 50K dev_val, 50K dev_test, and a 100K seeded draw from the untouched held_out_final partition (all caps disclosed; the FT-Transformer's O(n) CPU forward pass is the binding constraint). Deep nets use median imputation and standardization fit on train only (the boosters ingest NaN natively; the nets cannot). Booster hyperparameters are the Week-4 Optuna winners, reused without retuning.
