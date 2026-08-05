"""Model training and calibration-first evaluation.

Modules:
- ``evaluate``: discrimination (AUC-ROC, AUC-PR), calibration (Brier, ECE uniform/quantile, MCE),
  operating-point metrics, bootstrap CIs, and tiered stratified reporting.
- ``baselines``: logistic regression (calibration reference), decision tree, random forest, with
  optional isotonic recalibration on a held-aside fold.
- ``boosting``: XGBoost / LightGBM with Optuna tuning and post-hoc calibration.
- ``deep``: TabNet and FT-Transformer as honest benchmarks behind the boosting contract.
- ``ensemble``: cross-fitted out-of-fold logistic stacking over base-model probabilities.

Everything defaults to class weights and threshold selection over resampling (SPEC NFR4): SMOTE is
avoided because it degrades the calibration the fairness audit depends on.
"""
