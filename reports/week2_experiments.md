# Week 2 feature experiments (full-data compute)

All numbers computed live on the COMPLETE 2018-2024 public-use files: **24,920,407 singleton births** with a valid target. Every heavy step ran out-of-core or incrementally on the full population, with one disclosed exception: mutual information was computed on a stratified 1,000,000-row subsample. Language is associational: race is a marker of exposure to racism and structural barriers, not a biological cause, and none of this is a clinical tool.

## Dual-track LightGBM: leakage vs leakage-safe

clean/leakage tracks trained on a stratified cap of 5,000,000 dev_train rows (full dev_train was 10,466,570); held-out scoring used the full held_out_final partition.

| track | AUC-PR | AUC-ROC |
|---|---|---|
| clean (leakage-safe, 103 pre-delivery features) | 0.1810 | 0.6590 |
| leakage (adds 50 post-delivery columns) | 0.7615 | 0.9394 |

Adding the barred post-delivery variables lifts AUC-ROC by **0.2804** and AUC-PR by **0.5806**. That gap is the whole point: variables recorded at or after delivery (birthweight, Apgar, labor interventions, total visits) make a model look excellent and predict nothing usable before birth, so they are barred from the leakage-safe matrix.

## Top-5 univariate rank-sum AUC (full data, no model)

| feature | univariate AUC |
|---|---|
| medical_risk_count | 0.5704 |
| rf_ppterm | 0.5437 |
| bmi | 0.5433 |
| pay_1 | 0.5363 |
| mother_race_ethnicity_NH Black | 0.5330 |

These are exact single-feature ROC-AUCs over every row, so they measure raw marginal signal. The strongest are modest, which is why the clean model's discrimination is modest.

## Structure: PCA and clustering

IncrementalPCA over the standardized full matrix needs **more than 10** components to reach 80% of the variance and **more than 10** to reach 90%. MiniBatchKMeans (6 clusters) on the same standardized full matrix gives these per-cluster preterm rates:

| cluster | size | preterm rate |
|---|---|---|
| 0 | 11,994,818 | 9.92% |
| 1 | 8,804,796 | 7.66% |
| 2 | 196,073 | 8.69% |
| 3 | 317,500 | 9.06% |
| 4 | 175,646 | 8.45% |
| 5 | 3,431,574 | 6.12% |

Cluster preterm rates spread around the population base rate, reflecting how the unsupervised structure lines up with (but does not cleanly separate) the outcome. Per-cluster race mix and feature means are in the JSON companion.

