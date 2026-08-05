# FINDINGS.md

Live record of what the data actually reveals, and the evolving research-question list.
Updated at the end of every phase. Nothing here is a finding until code on real data
produced it. Statistics carried over from the initial project plan are scaffolding, not findings.

---

## Research questions (live; EDA may add, prune, or flag)

1. How well can preterm birth be predicted from birth-record features, and what is the realistic ceiling? (status: answered at single-year level, F5: AUC-ROC ~0.67; multi-year confirmation pending)
2. Which features are strongest predictors, and do they differ across racial/ethnic groups? (status: partial, F14/F15: clinical history (prior preterm, BMI, medical-risk count) carries the most signal; unsupervised risk strata are race-mixed, not race-defined; race alone is near-chance AUC 0.551)
3. Do models perform equitably across race and SES proxies (education, payment source, WIC) on equal opportunity, calibration, disparate impact? (status: open)
4. How much of the racial gap is associated with measured SES/access vs unexplained after adjustment? (status: answered, F11: ~33% explained (SES/demographic only), ~67% unexplained lower bound; up to ~44% if over-adjusting for clinical-history mediators)
5. What actionable, policy-relevant levers does the analysis point to (stated with causal humility)? (status: answered, F11/F12: measured SES/access closes <half the gap; the gap is flat-to-widening across education levels, so education alone is not the lever; majority is structural)
6. Is individual-level prediction from administrative birth records even adequate for clinical use, or do these data support population-level equity diagnosis only? (status: answered, F14/F15: weak cluster separation (silhouette ~0.25) and a 0.634 full-feature ceiling indicate population-level diagnosis only, not reliable individual risk scoring)

Note: "income" is never used; birth certificates have no income field. Use "SES proxy" (D15).

## Pre-registered comparison family (fix BEFORE first data touch; FR9.1)

This list freezes the confirmatory vs exploratory split for the Benjamini-Hochberg family. Add any
later comparison only with a "post-hoc (date)" label.

- CONFIRMATORY (primary): equal-opportunity difference, NH Black vs NH White, best model, on held_out_final.
- Planned exploratory (BH-corrected): demographic parity difference and disparate-impact ratio per
  race/ethnicity group vs NH White; calibration-by-group (ECE) per race/ethnicity group; the same
  metrics stratified by education level and by payment source; race-stratified top-feature comparison.
- Hypothesized directions (registered post-hoc 2026-06-20, AFTER the EDA in F1-F8; therefore the
  fairness results F10/F11 are treated as EXPLORATORY, not confirmatory, and are BH corrected):
  given the literature, we expected, before fitting, that disadvantaged groups (NH Black, NH AIAN)
  would show worse equal opportunity (lower TPR) than NH White at a fixed threshold. The data
  contradicted that naive expectation at the low screening threshold (TPR was higher for those
  groups, with the cost appearing as disparate impact); this reversal is itself reported as a
  finding. Because the directions were set after EDA, no result here is labeled confirmatory.

## Confirmed findings

- [2026-06-17] F1 (Phase 1, development sample, NOT nationally representative): On a 68,419-row head
  slice of the NBER 2023 natality CSV, the racial ordering of preterm rates reproduces the published
  pattern. Rates (obstetric estimate < 37 wks): NH Black 16.5%, NH AIAN 16.1%, NH NHOPI 13.2%, NH
  Multiracial 12.2%, NH White 11.3%, NH Asian 10.9%, Hispanic 10.2%. Black/White ratio 1.46 (national
  2024 published ratio ~1.55: 14.7% vs 9.5%). Absolute rates are inflated (sample overall 12.5% vs
  national ~10.4%) because the CSV is not randomly ordered, so the head slice over-samples some
  region/group. CAVEAT: directional validation only; headline national rates require the full data or
  a random sample. This confirms (a) the target/loader are correct and (b) the Black-White disparity
  is visible in the raw data, motivating the equity analysis. Source: scripts/phase1_sample_eda.py.

- [2026-06-18] F2 (Phase 1, REPRESENTATIVE spread sample, n=293,250 with valid target): national
  preterm rate from a spread sample (10 chunks at even byte offsets across the full 2.23 GB 2024 CSV)
  is 10.65% for all births, inside the FR4.1 sanity band [9.5%, 11.5%] and close to the published 2024
  figure of 10.41%. The head slice (F1) was biased high (12.5%) because the CSV is sorted; the spread
  sample is representative. Race breakdown matches published 2024 figures closely: NH White 9.5%
  (published 9.5%, exact), Hispanic 10.1% (published 10.1%, exact), NH Black 15.4% (published 14.7%),
  NH AIAN 14.9% (published 12.5%, small n=2,395). This validates the target construction
  (oegest_comb < 37) on representative data. Source: scripts/phase1_national_eda.py.
- [2026-06-18] F3 (Phase 1, leakage demonstration): a classifier trained on the EXCLUDED post-delivery
  variables alone (birthweight, Apgar, labor interventions, etc.) reaches AUC 0.913 on the 2024 spread
  sample. This is exactly the leakage the project guards against: a "preterm predictor" built from
  variables recorded at or after delivery looks excellent but is meaningless. The FR6.1 detector fires
  correctly (0.913 >> 0.55 gate); these variables are barred from the primary feature matrix.
- [2026-06-18] F4 (Phase 1, D17 singleton-bias check): restricting to singletons lowers absolute
  preterm rates (all 10.65% -> singleton 8.98%) because multiples are almost always preterm, but it
  does NOT erase the Black-White disparity. Black/White ratio is 1.62 for all births and 1.69 for
  singletons (relative shift +4%, under the 10% flag threshold). Singleton-only analysis is therefore
  not biased away from the racial disparity the project measures.

- [2026-06-18] F5 (Phase 2, CENTRAL FINDING, 2024 spread sample, 284,246 singleton births, 106
  primary features): a model built from legitimate pre-delivery birth-certificate features predicts
  preterm birth only modestly. Logistic regression AUC-ROC 0.668 (AUC-PR 0.193); HistGradientBoosting
  AUC-ROC 0.679 (AUC-PR 0.200), at a test prevalence of 0.090. This sits squarely in the published
  0.62-0.70 ceiling for administrative/birth-certificate data (Correction 1), so the honest thesis
  holds: individual prediction from these records is modest. By contrast, a model built from the
  EXCLUDED post-delivery variables reaches AUC 0.913, quantifying the leakage trap (Correction 2): the
  gap between the leaky and clean models (0.91 vs 0.67) is the teaching result. RQ1 is now answered at
  the single-year level (ceiling ~0.67); multi-year confirmation pending. Source: scripts/phase2_ceiling_check.py.

- [2026-06-18] F6 (Phase 2, MULTI-YEAR, 2019-2024, 1,986,314 singleton births, 106 features, metrics
  on a 794,040-row untouched held-out partition): the modest performance ceiling holds across all six
  years. Logistic regression AUC-ROC 0.670 (AUC-PR 0.198); HistGradientBoosting AUC-ROC 0.684 (AUC-PR
  0.207) at prevalence 0.089. This confirms F5 at scale and answers RQ1: individual prediction from
  birth-certificate features is modest (about 0.67-0.68), squarely in the published administrative-data
  range. Per-year national rates were all inside the FR4.1 band (2019 10.6%, 2020 10.5%, 2021 10.7%,
  2022 10.8%, 2023 10.9%, 2024 10.7%). The deterministic stratified splits give matched prevalence
  across partitions (0.0892 in every split). Source: scripts/phase2_build_dataset.py.

- [2026-06-18] F7 (review correction to F5/F6, supersedes the AUC figures there): after an independent
  data/stats review, three reverse-causally entangled pregnancy-complication flags (eclampsia,
  gestational hypertension, gestational diabetes) were removed from the primary model (D26). The honest
  leakage-free multi-year ceiling is therefore lower: HistGBM held-out AUC-ROC 0.654 (AUC-PR 0.183),
  Logistic 0.634 (AUC-PR 0.172), on 794,036 held-out singleton births at prevalence 0.089. Including
  those complication flags raises HistGBM to 0.684, so they contribute about 3 AUC points; that gain is
  not claimed in the conservative model because the conditions often cause preterm via induction. Both
  figures sit within or just below the published 0.62-0.70 administrative-data ceiling, so RQ1's answer
  (modest predictability) is unchanged and now more conservative.
- [2026-06-18] F8 (near-duplicate audit, D28): 12.2% of held-out rows share an identical feature
  fingerprint with a training row (no birth identifier exists in the public file; collisions are
  expected with 103 mostly-categorical features). This does NOT inflate the metric: held-out AUC is
  0.6528 on all rows vs 0.6529 on rows with no train-identical fingerprint (delta -0.0001). Identical
  coarse features map to either outcome, so the model cannot memorize individuals. Disclosed as a
  known property of administrative microdata.

- [2026-06-19] F9 (Phase 3 baselines, calibrated, 794,036 held-out singleton births, prevalence
  0.089): logistic regression (the calibration reference), decision tree, and random forest, each with
  balanced class weights and isotonic recalibration on dev_val. Random forest is best (AUC-PR 0.175,
  95% CI [0.173, 0.177]; AUC-ROC 0.651); logistic 0.167 / 0.633; decision tree 0.158 / 0.623. The
  CALIBRATION (stated honestly after the judge pass; calibration now trained on dev_test, not dev_val):
  after isotonic recalibration every model has a low uniform-bin ECE (~0.001) AND a low quantile-bin
  ECE (~0.001-0.002); MCE is ~0.06 once a 30-row minimum-bin guard removes sparse-bin noise (the
  earlier ~0.20 was a binning artifact). IMPORTANT QUALIFICATION: about 85% of predictions fall within
  plus or minus 5 percentage points of the ~9% base rate, so the low ECE means the scores are
  well-calibrated IN AGGREGATE, NOT that the model provides reliable individual risk stratification.
  Calibration holds across large race groups (ECE below 0.003). Discrimination is modestly lower for
  Hispanic and NH Asian (AUC-ROC ~0.60) than NH White (~0.63); measured with tiered suppression and
  treated as a hypothesis until the equal-opportunity, permutation, and BH-corrected analysis in Phase
  4/7. The calibration-first design is validated only in the aggregate-calibration sense, not as a
  substitute for discriminative power.
  Source: scripts/phase3_baselines.py; reports/phase3_baseline_metrics.md.

- [2026-06-19] F10 (Phase 4, boosting + first formal fairness audit, held-out 794K): XGBoost (Optuna,
  25 trials) is the best model so far at held-out AUC-PR 0.182, AUC-ROC 0.657, ECE 0.001; LightGBM is
  essentially tied (0.180 / 0.656). Gradient-boosted trees edge out the random forest (0.651) and the
  linear baseline, consistent with Correction 4 (trees lead on tabular data; deep learning not yet
  tested). The improvement over baselines is small, reinforcing the modest-ceiling thesis.
  FAIRNESS (XGBoost, screening threshold 0.086 set on dev_val for ~60% sensitivity; held-out sens 0.61,
  spec 0.61; equal opportunity = TPR gap vs NH White, BH-corrected): the result is nuanced and the
  opposite of the naive worry. Equal-opportunity differences are POSITIVE for most groups (NH Black
  +0.420, NH AIAN +0.266, NH NHOPI +0.226, NH Multiracial +0.088, Hispanic +0.039), with NH Asian
  slightly negative (-0.044); all BH p = 0.002. That is, at a single threshold the model has HIGHER
  sensitivity for higher-baseline-risk groups, not lower, so it does not miss more real preterm cases
  for Black mothers. The cost shows up as DISPARATE IMPACT: positive-prediction ratios are large (NH
  Black 2.86, NH AIAN 2.07, NH NHOPI 1.79), i.e. Black mothers are flagged about 2.9 times as often as
  White mothers. The honest reading: this reflects genuinely higher preterm prevalence (12.9% vs 7.8%),
  so a single threshold trades higher detection in high-risk groups against over-flagging them. Whether
  that is desirable is a value judgment, examined with threshold optimization and the disparate-treatment
  vs disparate-impact tradeoff in Phase 7. This is one model at one operating point; not a final claim.
  Source: scripts/phase4_boosting_fairness.py; reports/phase4_boosting_fairness.md.

- [2026-06-19; CORRECTED 2026-06-20 after the judge pass] F11 (Phase 7, HEADLINE POLICY FINDING,
  answers RQ4 and RQ5): g-computation decomposition of the Black-White singleton preterm gap (held-out
  partition). Raw gap (NH Black minus NH White) = 5.1 percentage points. PRIMARY decomposition adjusts
  only for demographic and SES/access covariates (age, education, payment source, WIC, BMI, marital
  status, tobacco) and EXCLUDES clinical-history mediators: 3.4 points remain, so about 33% of the gap
  is associated with measured SES/demographic factors and about 67% is unexplained. (The earlier 43%
  figure was the optimistic endpoint because it adjusted for clinical-history mediators such as prior
  preterm birth and the medical-risk composite; a sensitivity that adds them gives 44% explained but
  over-adjusts and must not be read causally. The honest explained share is a range of roughly 33-44%
  depending on specification, with 33% the defensible primary.) CRITICAL CAVEAT: the adjusting
  covariates (education, payment, WIC) are themselves products of structural racism, so even the 33%
  explained share is an overstatement and the 67% unexplained residual is a LOWER bound on the
  structural contribution. Association, not proof of mechanism. Policy reading: targeting only the
  measured SES/access factors would close at most a third to a half of the gap; the majority points to
  structural drivers not captured on the birth certificate.
  MITIGATION (RQ3 follow-up): post-processing equal-opportunity thresholds reduce disparate impact
  substantially (NH Black positive-flag ratio 2.69 -> 1.11, NH AIAN 1.96 -> 1.20, NH NHOPI 1.72 ->
  1.05) by using higher thresholds for higher-risk groups. This equalizes opportunity but is disparate
  TREATMENT (race-specific decision rules), in tension with disparate-impact law; presented as a
  tradeoff, not a recommendation.
  SHAP: global top predictors are prior live births, the medical-risk composite, BMI, maternal age,
  prior preterm birth, and marital status, all legitimate pre-delivery features; no leakage feature
  dominates, supporting the integrity of the primary model. Source: scripts/phase7_responsible_ai.py;
  reports/model_card.md; reports/datasheet.md.

- [2026-06-20] F12 (Phase 8, INTERSECTIONAL HEADLINE, answers RQ5, full 2019-2024 data): the
  Black-White singleton preterm gap does NOT close with maternal education. Within every one of the
  eight education levels the NH Black minus NH White gap is flat at +3.7 to +4.8 percentage points,
  and it is WIDEST at the top: +4.71 pp at a master's degree and +4.76 pp at a doctorate/professional
  degree, versus +3.70 pp at under-8th-grade. The single most striking comparison: college-educated
  (bachelor's+) NH Black mothers have a preterm rate of 10.66% (n=50,593), HIGHER than NH White
  mothers with a high-school education or less (9.30%, n=288,166). Education, the strongest SES lever
  on the birth certificate, does not buy a Black mother out of the disparity; a Black mother with a
  doctorate has a higher preterm rate than a White mother who never finished high school. This
  reproduces the published pattern (KFF; the 2.1M-birth-certificate analysis showing the gap persists
  even among privately-insured college graduates) and is the empirical core of the structural-racism
  reading in F11: if SES explained the gap, it would shrink within education strata, and it does not.
  Association, not proof of mechanism; race is a marker of exposure to racism. Source:
  scripts/phase8_exploration.py; reports/phase8_exploration.md.

- [2026-06-20] F13 (Phase 8, COVID-era trend, answers the COVID sensitivity question): there is NO
  meaningful COVID-era shift in the singleton preterm rate in this national administrative sample. The
  COVID-period (2020-2021) minus pre-COVID (2019) shift is within +/-0.4 pp for every large race group
  (NH White +0.06, NH Black +0.02, Hispanic +0.10, NH Asian +0.01); only the two smallest groups move
  more (NH NHOPI +1.32, NH AIAN -0.41), which is sampling noise at their cell sizes. This is the
  honest national picture: localized hospital studies reported sharp lockdown-period preterm DECREASES
  (e.g. two San Francisco hospitals 13.2% -> 8.0%), but the national administrative signal is flat,
  matching the AAP Pediatrics national analysis that found no clear nationwide change. The lesson is
  about scale and selection: a real-looking local effect can vanish in the national aggregate. Causal
  humility: we cannot attribute the (null) national result to any single mechanism. Source:
  scripts/phase8_exploration.py.

- [2026-06-20] F14 (Phase 8, UNSUPERVISED structure, answers RQ2/RQ6): KMeans on standardized
  pre-delivery features (dev_train, 60K sample, k=7 chosen by silhouette) finds latent risk strata but
  no crisp natural subpopulations: silhouette peaks at only ~0.25, so risk is a continuum, not a set of
  sharp clusters (consistent with the modest model ceiling). The clusters DO spread 4x in preterm rate,
  from a low-risk cluster (6.6%: low BMI ~27.6, married ~0.93, zero medical-risk count, early prenatal
  care) to high-risk clusters (26.6% and 27.3%). Critically, the high-risk clusters are organized by
  CLINICAL and behavioral features, not race: the 26.6% cluster is defined by prior preterm birth
  (rf_ppterm mean = 1.0) and high parity, the 27.3% cluster by high BMI (~34.6) and medical-risk
  count; and every high-risk cluster is NH-White-plurality (41-49% White). So the measured feature
  space sorts people by clinical risk, and that sorting does not reconstruct the racial gap, which is
  the unsupervised echo of F11/F12: the disparity is diffuse and not captured by the recorded clinical
  risk factors. PCA needs 5 components for only 55.7% of variance, confirming the signal is spread thin.
  Source: scripts/phase8_exploration.py (figure not committed; regenerate to view).

- [2026-06-20] F15 (Phase 8, CEILING PROBE, sharpens RQ1/RQ6): a layered logistic ablation on the
  held-out partition quantifies WHY individual prediction is modest and WHERE the little signal lives.
  AUC-ROC rises race-only 0.551 -> +demographics (age, married) 0.576 -> +SES proxies (education,
  payment, WIC, first-trimester care) 0.590 -> +clinical history (BMI, smoking, parity, prior preterm,
  medical-risk composite) 0.634. Each layer adds little: race alone is barely better than chance
  (0.551), and the SES proxies add only ~0.013 AUC on top of demographics. The largest single jump is
  clinical history (+0.044). This is the discriminative-power complement to F12's rate analysis: race
  carries almost no individual predictive signal here, the recorded clinical history carries the most,
  and even all of it together lands at the published administrative-data ceiling. The honest reading
  for RQ6: these records support population-level equity diagnosis, not reliable individual risk
  scoring. Source: scripts/phase8_exploration.py.

- [2026-06-20] F16 (Phase 5/6 benchmark, closes DECISION_LOG D34/D36): a CatBoost model, a random
  forest, and a stacked logistic ensemble were benchmarked against the tuned boosters, all
  isotonic-calibrated on dev_test and evaluated on the untouched held-out partition (base learners fit
  on a 300K dev_train subsample with fixed params, single-threaded). Held-out AUC-ROC: CatBoost 0.6547,
  stacked ensemble 0.6541, mean ensemble 0.6540, XGBoost 0.6513, random forest 0.6483, LightGBM 0.6467.
  The AUC-ROC spread across the ENTIRE tree field plus both ensembles is only 0.008. The stacked
  ensemble has the best AUC-PR (0.1831 vs the best single model 0.1783), a ~0.005 lift that is
  negligible and costs calibration (ECE 0.008 vs ~0.001), so it is not adopted. This confirms
  Correction 4 and D34: gradient-boosted trees lead, the ensemble adds little once the modest
  administrative-data ceiling is reached, and deep learning (FT-Transformer/TabNet) stays deferred as
  an expected non-winner with a real wall-clock cost. The Phase 4 Optuna-tuned XGBoost (held-out
  AUC-ROC 0.657) remains the reference primary model. Source: scripts/phase56_benchmark.py;
  reports/phase56_benchmark.md.

## Full-data rebuild (COMPLETE 2018-2024 public-use files; supersedes the sample-era figures above)

Findings F1-F16 were computed on the earlier spread SAMPLE (2019-2024). F17-F19 rebuild Phases 2-4 on
the COMPLETE 2018-2024 files (24,941,021 singletons; the outcome/equity audit uses every row, model
development uses a stratified 3M working sample with a 1.5M untouched held-out partition). The
sample-era conclusions HELD: the numbers moved by at most a couple of AUC points and every headline
was independently recomputed and confirmed. Where they differ, the full-data figures are canonical.

- [2026-07-05] F17 (Week 2 / Phase 2, full data, outcome + leakage audit): on all 24,920,407 singleton
  births with a valid target, the overall preterm rate is 8.57% (below the 10.4% all-births headline
  exactly as expected once ~60%-preterm multiples are excluded). Per-race rates: NH White 7.54%, NH
  Asian 7.58%, Hispanic 8.64%, NH Multiracial 8.95%, NH AIAN 10.68%, NH NHOPI 10.80%, NH Black 12.28%.
  The NH Black minus NH White gap is 4.74 percentage points, computed on the full census (so even NH
  NHOPI n=67,182 and NH AIAN n=179,698 are reliable here). LEAKAGE (FR6.1): a classifier on the 50
  excluded post-delivery variables alone reaches AUC 0.917 on a 300-400K stratified sample, far above
  the 0.55 gate, so those variables are barred; the leakage-safe primary matrix is 103 pre-delivery
  features and passes the FR6.2 composite guard and FR6.3 univariate gate (first-trimester care 0.545,
  WIC 0.511). This confirms sample-era F2/F3/F4 on the complete data. Source:
  scripts/week2_feature_leakage_audit.py; reports/week2_phase2_audit.{md,json}.

- [2026-07-05] F18 (Week 3 / Phase 3, full data, calibrated baselines, held-out 1,498,760 singleton
  births): the modest ceiling holds at full scale. Random forest is best at AUC-ROC 0.651 (AUC-PR 0.166
  [0.163, 0.170]); logistic regression (the calibration reference) 0.634 (0.159); decision tree 0.626
  (0.150), all with balanced class weights and isotonic recalibration. CALIBRATION: raw ECE ~0.39 falls
  to ~0.001 after recalibration (quantile-bin ECE ~0.001), and holds within every large race group (ECE
  below ~0.005). QUALIFICATION (kept from D31): 82% of predictions sit within +/-5pp of the 8.6% base
  rate, so low ECE means aggregate calibration, NOT reliable individual risk stratification.
  Discrimination is modestly lower for Hispanic (0.605) and NH Asian (0.599) than NH White (0.631),
  carried as a hypothesis into the fairness audit. This confirms sample-era F7/F9 on the complete data.
  Source: scripts/week3_baselines.py; reports/week3_baseline_metrics.{md,json}.

- [2026-07-05] F19 (Week 4 / Phase 4, full data, boosting + first fairness audit, held-out 1,498,760):
  tuned gradient boosting (Optuna, 30 trials each) barely moves the ceiling: LightGBM and XGBoost both
  reach held-out AUC-ROC 0.657 (AUC-PR 0.174, ECE <=0.0006), about +0.6 AUC points over the random
  forest, confirming Correction 4 (trees lead on tabular data but cannot manufacture absent signal).
  FAIRNESS (best booster, screening threshold 0.078 set on dev_val for ~60% sensitivity; held-out sens
  0.60, spec 0.62; fairness computed on a 300K held-out subsample; equal-opportunity = TPR gap vs NH
  White, BH-corrected): the result is the OPPOSITE of the naive worry. Equal-opportunity differences are
  POSITIVE for higher-risk groups (NH Black +0.393, NH AIAN +0.233, NH NHOPI +0.184, Hispanic +0.087,
  NH Multiracial +0.091; all BH p <= 0.006), with NH Asian slightly negative (-0.022, ns). At a single
  threshold the model has HIGHER sensitivity for higher-baseline-risk groups, so it does not miss more
  real preterm cases for Black mothers; the cost appears as DISPARATE IMPACT (NH Black flagged 2.74x as
  often as NH White), reflecting the genuinely higher prevalence. Framed as a value-judgment tradeoff
  (disparate impact vs disparate treatment), examined not resolved. This reproduces sample-era F10 on
  the complete data. Source: scripts/week4_boosting_fairness.py; reports/week4_boosting_fairness.{md,json}.

## Open questions raised during work

- [2026-06-18] PRELIMINARY (not yet a finding; figures from the conservative leakage-free model, F7):
  HistGBM held-out AUC-ROC differs by group: NH White 0.653, NH Black 0.644, Hispanic 0.622, NH Asian
  0.614, NH AIAN 0.654, NH NHOPI 0.657 (wide CI), NH Multiracial 0.648. The
  model appears to discriminate somewhat WORSE for Hispanic and Asian mothers. This is raw AUC-ROC only,
  with no confidence intervals, no equal-opportunity or calibration analysis, and no multiple-comparison
  control. Treat strictly as a hypothesis for the Phase 4/7 fairness audit (FR8.1, FR9), which will add
  bootstrap CIs, stratified permutation tests, BH correction, and calibration-by-group before any claim.

- Full multi-year (2019-2024) microdata ingestion is pending: each year is ~2.2 GB and the link
  throttles to ~0.8 MB/s, so a full pull times out. The ingestion pipeline is year-agnostic and tested;
  the representative spread-sample method validates 2024. Decide whether to (a) babysit full downloads
  for modeling-grade data or (b) use larger spread samples per year. Logged for Phase 2 planning.

## Weeks 5-9 findings (deep learning, ensembles, fairness mitigation, explainability, robustness)

- [2026-07-19] F20 (Week 5, compute-capped deep-learning benchmark, consistent with D41): TabNet and
  FT-Transformer were benchmarked against the tuned XGBoost on a 30K-capped training subsample (100K
  held-out subsample, max 3 epochs; both median-imputed and standardized since neural nets cannot
  ingest NaN). Held-out AUC-PR: TabNet 0.130, FT-Transformer 0.120, versus XGBoost 0.174 on the same
  rows. This is an honest feasibility-capped benchmark, NOT a fair architecture comparison: the nets
  saw about 6% of the boosters' rows for only 3 epochs, with small hand-set configs and no
  hyperparameter search, because the FT-Transformer's O(n) CPU forward pass is the binding wall-clock
  constraint. We therefore claim only that on this data and under a laptop compute budget, tuned trees
  clearly outperform minimally trained deep tabular models, consistent with D41 and Grinsztajn et al.
  (NeurIPS 2022); we do NOT claim the nets were given a fair chance to win, and the margin should not
  be read as a general "trees beat deep learning" result. The F27 train-cap plateau is an XGBoost
  property and does not license the claim that more compute would leave the nets unchanged (deep
  tabular models are the archetype that keeps improving with data). The real deliverable is TabNet's
  feature masks (medical_risk_count 0.21, rf_ppterm 0.11, age_bracket 0.03 top three), which
  corroborate the SHAP ranking. Source: scripts/week5_deep_and_ensemble.py; reports/week5_models.json.

- [2026-07-19] F21 (Week 5, stacked ensemble and final model selection): a cross-fitted logistic stack
  over XGBoost, LightGBM, TabNet, and FT-Transformer probabilities reached held-out AUC-PR 0.171
  versus XGBoost's 0.172 alone. The stack is within bootstrap noise of the simpler booster (CI overlap
  complete) and costs a more complex pipeline. XGBoost was selected as the primary model: margin within
  bootstrap noise, simpler for auditability, natively handles missing values, and the fairness audit
  was already validated on it. Source: reports/week5_models.json.

- [2026-07-19] F22 (Week 6, fairness mitigation tradeoffs, 100K held-out subsample): four mitigation
  strategies were compared against the unmitigated XGBoost baseline (worst-group EO gap 0.389, NH Black
  vs NH White; mean absolute EO gap 0.139). ThresholdOptimizer (group-specific thresholds for TPR
  parity) cut the worst EO gap to 0.173 and the mean to 0.056, at a cost of disparate treatment
  (race-specific decision rules) and lower specificity (0.70 vs 0.61). Reweighting (cell-mass
  equalization, refit with class balance carried by the sample weights rather than double-counted via
  scale_pos_weight) substantially reduced the NH Black gap (from +0.389 to +0.017) and cut the mean
  absolute gap (0.139 to 0.115), but the WORST gap rose to 0.504 driven by NH NHOPI, one of the
  smallest and noisiest groups, and AUC-PR fell from 0.172 to 0.166: reweighting helps the large group
  it has the mass to learn from and destabilizes tiny groups. ExponentiatedGradient (Fairlearn
  reductions, TPR-parity constraint, logistic base) nearly eliminated EO gaps (worst 0.072) but showed
  near-zero sensitivity (0.010). CRITICAL: that near-zero sensitivity is a SETUP ARTIFACT, not a
  property of the fairness constraint: the UNCONSTRAINED logistic baseline already has sensitivity
  0.009 at the classifier's default 0.5 cutoff, because at 8.6% prevalence almost no logistic score
  exceeds 0.5, whereas the XGBoost baseline is scored at a tuned 0.078 threshold. The two are compared
  at incomparable operating points; what ExponentiatedGradient actually achieved (near-complete TPR
  equalization with a weak convex learner) is real. The honest reading: no mitigation is free;
  ThresholdOptimizer most reliably reduces gaps while retaining useful sensitivity, but it requires
  disparate treatment, which is in tension with anti-discrimination law and is never exposed in any
  deployed artifact. Source: scripts/week6_mitigation.py; reports/week6_mitigation.json.

- [2026-07-19] F23 (Week 7, SHAP explainability, 30K subsample): global SHAP top predictors for the
  selected XGBoost model are priorlive (0.161), medical_risk_count (0.134), mager (0.115), rf_ppterm
  (0.108), bmi (0.104), and married (0.091). All are legitimate pre-delivery features; no leakage
  variable appears (leakage check PASS on all 13 reported features). For the NH Black group
  specifically, the race one-hot itself (mother_race_ethnicity_NH Black) dominates at 0.159 mean |SHAP|,
  followed by priorlive (0.131) and medical_risk_count (0.122). That is: the model's best information
  about NH Black preterm risk IS membership in the NH Black group, which is the statistical echo of the
  structural disparity (race predicts the outcome not because of biology but because it indexes exposure
  to structural barriers). For NH White, the race one-hot ranks 8th, not 1st; clinical history drives
  predictions. Source: reports/week7_explain.json.

- [2026-07-19] F24 (Week 7, Black-White gap decomposition, 30K subsample, g-computation): raw NH Black
  minus NH White singleton preterm gap is 4.9 percentage points. Full covariate adjustment (SES +
  clinical history) explains 43% (CI 40-47%); SES-only adjustment explains 22%; clinical-history-only
  explains 19%. The honest range is 19-43% explained depending on covariate specification. METHOD
  FRAMING: this is a MEDIATION-style decomposition, not confounding adjustment. Education, insurance,
  WIC, BMI, and chronic conditions are mediators downstream of structural racism, not confounders of a
  race "effect", so the "explained" share is the portion of the association statistically transmitted
  through measured mediators, and the residual is the direct-plus-unmeasured portion. The estimator is
  g-computation standardization over a LOGISTIC model (linear in covariates); because the true risk
  surface is nonlinear and interacted (see the SHAP dependence plots), the explained share is
  functional-form dependent and the 19-43% range reflects covariate-set choice only, not model-form
  uncertainty. We use standardization language, not counterfactual-in-race language, because race is a
  social marker of exposure, not a manipulable treatment. CRITICAL CAVEAT (kept from F11): the
  adjusting covariates are themselves products of structural racism, so even the 43% "explained" share
  is an overstatement, and the unexplained residual is a LOWER bound on the structural contribution,
  under the stated assumption that every measured covariate is wholly downstream of structural racism
  and no non-structural race-associated cause is omitted. Policy reading unchanged from F11: targeting
  only the measured SES/access factors would close at most a fifth to a half of the gap. Source:
  reports/week7_explain.json.

- [2026-07-19] F25 (Week 8, out-of-time stability): XGBoost trained on pre-COVID births (2018-2019,
  200K sample) and tested on COVID-era (2020-2021) and recent (2022-2024) births. AUC-ROC: COVID-era
  0.653, recent-era 0.655, versus the full-sample reference of 0.656. The drop is about 0.003 AUC,
  well within bootstrap noise. Per-year ECE stays below 0.003 every year (max 0.0033 in 2020), and
  per-year AUC-ROC ranges from 0.656 to 0.669 (no monotonic drift). The model is stable across the
  COVID regime change because its signal comes from maternal demographics and history that did not shift
  materially with the pandemic. SCOPE CAVEAT: this establishes temporal stability WITHIN the NCHS
  natality system (same birth-certificate instrument and coding across 2018-2024), not external
  validity. The model is not validated on any non-NCHS population or clinical cohort; the small
  out-of-time drop should not be read as evidence it would transport to a different data source or care
  setting. Source: reports/week8_robustness.json.

- [2026-07-19] F26 (Week 8, race-stratified models, hypothesis-generating): separate XGBoost models
  trained and tested within each major race group (at least 25K training rows). Held-out AUC-ROC: NH
  White 0.654, NH Black 0.636, Hispanic 0.620, NH Multiracial 0.613, NH Asian 0.607. The range is
  0.61-0.65, narrower than the inter-model spread, confirming that the modest ceiling holds within every
  group. NOTE on metric choice: cross-group comparison uses AUC-ROC, which is prevalence-robust, NOT
  AUC-PR. AUC-PR is prevalence-dependent, so the higher NH Black AUC-PR (0.213 vs NH White 0.158) is
  partly a mechanical consequence of NH Black's higher preterm prevalence (about 12.3% vs 7.5%) and
  does not indicate the model serves that group better; on the prevalence-robust AUC-ROC the ranking in
  fact reverses (NH Black 0.636 < NH White 0.654). Top predictors differ: for NH Black the leader is
  rf_ppterm (prior preterm, importance 0.394)
  followed by rf_phype (pre-pregnancy hypertension, 0.146); for NH White, rf_ppterm leads (0.300)
  followed by rf_phype (0.141). For Hispanic, rf_ppterm also leads (0.362). NH Asian is the only group
  where age_bracket rises to top-5 (importance 0.045). These differences are framed as hypotheses, not
  claims: they may reflect true subgroup biology, differential reporting/coding, or simply the smaller
  training-n for some groups. Source: reports/week8_robustness.json.

- [2026-07-19] F27 (Week 8, train-cap plateau): AUC-PR across training caps of 100K, 250K, and 500K
  rows was 0.171, 0.175, and 0.178, with the 250K-to-500K marginal gain only 0.003. The 500K cap used
  throughout the project is well past the diminishing-returns knee for this data. More training rows
  cannot manufacture signal that the administrative features do not carry. Source:
  reports/week8_robustness.json.

## Planned work (not yet done)

1. Multiclass severity target. 4-class (very preterm <28w, moderate 28-31w, late preterm 32-36w, term
   37+w) classification was designed as a protocol but NOT run: the exact-week gestation column used as
   the target is not retained in the modeling Parquet (only the binary indicator), and reconstruction
   from interim files would require re-running the full preprocessing pipeline. Descoped to protocol
   documentation (see reports/week9_external_context.json, multiclass_protocol field).
