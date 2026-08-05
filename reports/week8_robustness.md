# Week 8: out-of-time validation, yearly drift, race-stratified models, sensitivity

## Out-of-time validation

Model trained on pre-COVID births (2018-2019, n=200,000), isotonic-calibrated on a separate pre-COVID fold (n=65,764), then evaluated on COVID-era and recent held-out data.

| era | years | n | AUC-PR | AUC-ROC | ECE |
|---|---|---|---|---|---|
| COVID | 2020-2021 | 424,074 | 0.1688 | 0.6527 | 0.0016 |
| Recent | 2022-2024 | 635,489 | 0.1721 | 0.6545 | 0.0017 |

## Yearly drift (full model, held-out partition)

| year | n | prevalence | AUC-ROC | AUC-PR | ECE |
|---|---|---|---|---|---|
| 2018 | 44,163 | 0.0825 | 0.6556 | 0.1672 | 0.0014 |
| 2019 | 43,951 | 0.0842 | 0.6614 | 0.1740 | 0.0028 |
| 2020 | 42,119 | 0.0823 | 0.6610 | 0.1724 | 0.0033 |
| 2021 | 42,686 | 0.0887 | 0.6592 | 0.1800 | 0.0030 |
| 2022 | 42,788 | 0.0889 | 0.6693 | 0.1887 | 0.0023 |
| 2023 | 41,784 | 0.0886 | 0.6582 | 0.1816 | 0.0018 |
| 2024 | 42,509 | 0.0865 | 0.6586 | 0.1817 | 0.0012 |

## Race-stratified models (hypothesis_only)

Per-group XGBoost models trained and evaluated on that group's data only.

AUC-PR is prevalence-dependent and NOT comparable across groups with different preterm prevalence: a higher-prevalence group can post a higher AUC-PR mechanically, not because the model serves it better. Use the prevalence-robust AUC-ROC for cross-group comparison. Per-group prevalence is shown for this reason.

| group | prevalence | train_n | held_out_n | AUC-PR | AUC-ROC | ECE |
|---|---|---|---|---|---|---|
| Hispanic | 8.7% | 100,000 | 50,000 | 0.1494 | 0.6200 | 0.0021 |
| NH Asian | 7.6% | 64,692 | 50,000 | 0.1270 | 0.6069 | 0.0035 |
| NH Black | 12.4% | 100,000 | 50,000 | 0.2127 | 0.6364 | 0.0044 |
| NH Multiracial | 8.9% | 24,756 | 35,333 | 0.1500 | 0.6132 | 0.0103 |
| NH White | 7.6% | 100,000 | 50,000 | 0.1579 | 0.6539 | 0.0015 |

## Threshold sensitivity

Base threshold: 0.0784 (from Week 4).

| threshold | % of base | sensitivity | specificity | worst EO gap |
|---|---|---|---|---|
| 0.0392 | 50% | 0.9801 | 0.0621 | 0.0429 |
| 0.0588 | 75% | 0.8384 | 0.3146 | 0.2344 |
| 0.0784 | 100% | 0.6147 | 0.6074 | 0.4267 |
| 0.0980 | 125% | 0.4544 | 0.7711 | 0.4182 |
| 0.1176 | 150% | 0.2964 | 0.8923 | 0.2787 |

## Training-cap sensitivity

| train_cap | held-out AUC-PR |
|---|---|
| 100,000 | 0.1707 |
| 250,000 | 0.1755 |
| 500,000 | 0.1776 |

