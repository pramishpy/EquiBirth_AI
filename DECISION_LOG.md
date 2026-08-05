# DECISION_LOG.md

Append-only record of decisions, each with evidence and date. Five methodological corrections
carried over from the initial project plan are seeded here as OPEN decisions to be confirmed or
overturned against the real data. No status should move to CONFIRMED or OVERTURNED without
code or a fetched primary source as evidence.

Format: `[YYYY-MM-DD] <id> <STATUS>: <decision>. Evidence: <what>.`

---

## Seeded corrections (OPEN until verified against real data)

- [2026-06-17] D1 OPEN: Realistic AUC ceiling for birth-certificate-only models is ~0.62 to 0.70 (not "clinically useful high accuracy"). Higher published AUCs use EHR/biomarkers. Action: set expectations; treat AUC 0.90+ as a leakage alarm. Evidence to gather: recompute on real data; refetch MDPI review, MARS, Frontiers CatBoost study.
- [2026-06-17] D2 OPEN: Prenatal-care features (visit count, APNCU/Kotelchuck) are outcome-entangled (visits capped by gestation; APNCU uses gestational age as input; U-shaped). Action: build leakage taxonomy; primary model excludes them; leaky model is a demo only. Evidence to gather: confirm variable definitions in User Guide; reproduce U-shape in EDA.
- [2026-06-17] D3 OPEN: Do not use SMOTE by default. ~10% event rate is mild imbalance; SMOTE harms calibration without improving AUC (van den Goorbergh 2022). Action: class weights + threshold tuning; SMOTE only as recalibrated ablation. Config default set to `class_weights`.
- [2026-06-17] D4 OPEN: Gradient-boosted trees expected to beat deep learning on this tabular data (Grinsztajn 2022). Action: DL is an honest benchmark, not assumed winner; optionally try TabPFN-2.5 on a size-appropriate subsample. Report whichever wins.
- [2026-06-17] D5 PARTIALLY OVERTURNED (FR1.1 evidence): The geographic-identifier half STANDS (no state/county in public-use microdata; "by state" only via CDC WONDER aggregates; no row-merge with County Health Rankings). The format half is OVERTURNED: NBER DOES serve analysis-ready per-year CSV at https://data.nber.org/nvss/natality/csv/{year}/natality{year}us.csv (HTTP 200, text/csv, 237 columns, lowercase standardized headers, confirmed for 2023 and 2024). No fixed-width parser is needed; use the CSV (integrity-checked via SHA-256 + row count). Raw fixed-width zip (~229MB) and Stata .dta (~905MB) also exist as fallbacks. Target variable confirmed: oegest_comb (obstetric estimate); LMP-based combgest excluded. All leakage vars confirmed by real name (ld_ster, ld_antb, dbwt, apgar5/10, attend, bfacil, dplural, previs, precare). Evidence: curl HEAD + 200KB header sample, 2026-06-17.

## Data year range

- [2026-06-17] D6 CONFIRMED: Use natality public-use microdata years 2019 through 2024. 2024 is the latest available (NBER page "Last Updated September 2025", note "September 2025: 2024 Data added"; "Births: Final Data for 2024" released 2026-06-09). 2025 is EXCLUDED with justification: no public-use microdata file exists yet for 2025; only provisional 2025 data exists as aggregates via CDC WONDER (VSRR No. 43, "Births: Provisional Data for 2025"). Row-level 2025 microdata cannot be obtained, so it cannot enter the model. Re-check NBER for a 2025 file before final submission; if released, add it after a layout-harmonization check. Evidence: https://www.nber.org/research/data/vital-statistics-natality-birth-data ; https://www.cdc.gov/nchs/data/vsrr/vsrr043.pdf
- [2026-06-17] D6a OPEN: Final lower bound (2019) is provisional. Confirm 2019-2024 layout consistency via the User Guides before pooling; if harmonization is costly, a shorter consistent window (e.g. 2022-2024) is acceptable, documented. The earlier-vs-recent tradeoff is recency and COVID-era effects (2020-2021 births may be atypical; flag and possibly analyze separately).

## Spec review cycle 1 (applied 2026-06-17)

- [2026-06-17] D7 DONE: Hard-coded leakage exclusion list added to SPEC Section 6 (DBWT/birthweight, Apgar, ICU, AB_*, CA_*, labor interventions LD_STER/LD_ANTB/LD_INDL/LD_AUGM/LD_CHOR/LD_PROM, DMETH_REC, ATTEND, BFACIL, DPLURAL, COMBGEST). LD_STER flagged single highest-risk leak. FR6 adds an automated leakage detector (classifier on excluded vars; fail if AUC>0.55). Source: critic BLOCKER-1/3.
- [2026-06-17] D8 DONE: Global na_sentinels removed from config; replaced with per-variable sentinel map built from User Guide (9 visits, parity 9, plurality 5 must survive). Source: both reviewers HIGH.
- [2026-06-17] D9 DONE: gestational_age_var pinned per year (obstetric estimate, not LMP COMBGEST) via config gestational_age_var_by_year; added target sanity tests (rate 9.5-11.5%, range 20-45, <1% null).
- [2026-06-17] D10 DONE: Leakage AUC alarm lowered from 0.90 to 0.75 for birth-certificate-only models (published ceiling ~0.70-0.77). Source: critic HIGH-4.
- [2026-06-17] D11 DONE: Eclampsia + pregnancy-associated hypertension reclassified outcome-entangled/timing-conditional; gestational diabetes excluded for extreme/very preterm models.
- [2026-06-17] D12 DONE: Statistical rigor added: bootstrap on held-out validation (2000 iters), subgroup positive floor 1000 for perf metrics, cell suppression n<50, Benjamini-Hochberg for multiple comparisons, permutation tests + CIs on disparity metrics, ECE/MCE calibration on raw probabilities.
- [2026-06-17] D13 DONE: COVID-era (2020-2021) flagged; birth_year added to stratification and as feature; temporal sensitivity analysis required.
- [2026-06-17] D14 DONE: Phase 1 acceptance criteria defined (SPEC Section 8). Format-confirmation gate (FR1.1) blocks parser. Polars primary engine, per-year parse, 8GB RSS budget. manifest.json data versioning. CDC WONDER acquired manually (ToS). Decomposition: linear Oaxaca-Blinder excluded; Fairlie/g-computation only, with mediator caveat.
- [2026-06-17] D15 OPEN: "Income" stratification flagged: birth certificates have no direct income. Likely relabel to "SES proxy" (education + payment source + WIC). Decide in Cycle 2.

## Spec review cycle 2 (applied 2026-06-17)

- [2026-06-17] D15 RESOLVED: "income" relabeled "SES proxy" (education + payment source + WIC) everywhere. Birth certificates have no income field. Updated SPEC, config, FINDINGS.
- [2026-06-17] D16 DONE: Decomposition uses Fairlie nonlinear (primary) / g-computation (secondary); linear Oaxaca-Blinder excluded (invalid for logistic outcome). Working DAG in SPEC FR10.1. SES covariates are mediators; included in predictive model for prediction not causation.
- [2026-06-17] D17 DONE: Singleton-only bias check is a Phase 1 deliverable: race-stratified preterm rates before/after the filter; flag in FINDINGS if Black-White gap shifts >10% relative (multiples skew higher-SES/White via ART).
- [2026-06-17] D18 DONE: Split naming collision fixed (tuning leakage risk). dev_internal_val (Optuna) vs held_out_final (remainder, final metrics). Bootstrap CIs on held_out_final only.
- [2026-06-17] D19 DONE: Two distinct leakage gates named: excluded_vars_leakage_gate=0.55 (build fail) vs full_model_auc_alarm=0.75 (SHAP audit). Plus univariate_leakage_gate=0.60 for timing-conditional features.
- [2026-06-17] D20 DONE: DPLURAL moved from "encodes target" to structural_exclusion_vars (NOT in the FR6 detector; its gestation link is biology, would cause false-positive CI fail). Two lists: leakage_detector_vars vs structural_exclusion_vars.
- [2026-06-17] D21 DONE: Composite-feature leakage bypass closed. Every engineered feature declares `components`; FR6.2 asserts no component is excluded. Medical-risk composite enumerated explicitly (diabetes + hypertension + prior preterm + prior C-section); RF_GDIAB removed from composite for all preterm models. Exact column names added (UPREVIS, DOREVIS, PRECARE/first-trimester recode, IP_GON/IP_SYPH/NO_INFEC).
- [2026-06-17] D22 DONE: Subgroup suppression tiered (<50 suppress, 50-999 cautious point estimate, >=1000 full CI) so AIAN/NHOPI are not silenced. Outcome equity audit (full population) separated from model-performance audit. Permutation null stratified within y=1/y=0 classes (Ojala & Garriga 2010). BH family pre-registered in FINDINGS before first data touch.
- [2026-06-17] D23 DONE: Thesis made falsifiable (anti-confirmation-bias). residual_honesty_floor=0.10; report honestly if unexplained residual < 10% of raw gap or disparity not significant after adjustment. SES-as-racism-product framed as theoretical commitment, not result.
- [2026-06-17] D24 DONE: Phase 1 criteria tiered (A must-complete / B wired-stubbed / human-review) to avoid the circular dependency (coverage floor cannot precede the parser). Deliverable contracts + named tests added (SPEC 8.1, 8.2). Optuna timeout 3600s, n_trials_ci 5.

## Independent review cycle (2026-06-18, 3 fresh-context agents: code, data/stats, spec)

- [2026-06-18] D26 DONE: Eclampsia (rf_ehype), gestational hypertension (rf_ghype), and gestational
  diabetes (rf_gdiab) reclassified from timing-conditional to IATROGENIC_ENTANGLED and REMOVED from the
  primary feature matrix. They are mid/late-pregnancy conditions that often cause preterm via medically
  indicated induction (reverse causal). Removing them lowered HistGBM held-out AUC 0.684 -> 0.654 and
  Logistic 0.670 -> 0.634, so they contributed ~3 AUC points. The conservative leakage-free model
  (~0.65) is now the honest default. They are retained in the raw frame for the post-hoc disparity
  analysis, and a secondary "late-pregnancy clinical" model that adds them is an option for Phase 3
  (labeled as using conditions known later in pregnancy). Source: data/stats critic HIGH finding.
- [2026-06-18] D27 DONE: LEAKAGE_DETECTOR_VARS expanded with the real post-delivery columns confirmed
  in the file: abnormal newborn conditions (ab_nicu/ab_aven*/ab_anti/ab_seiz/ab_surf), congenital
  anomalies (ca_*), intrapartum infections (ip_gon/ip_syph/ip_chlam/ip_hepb/ip_hepc/no_infec),
  delivery method/presentation (me_rout/me_pres/me_trial/rdmeth_rec/dmeth_rec). ld_prom is not a column
  in this file (PROM not separately coded). Closes the D7 integrity gap. (These can never enter the
  feature matrix anyway since they are not in COLUMN_MAP; the detector now knows them.)
- [2026-06-18] D28 DONE: Near-duplicate scrutiny (stats critic). 12.2% of held-out rows share an
  identical feature fingerprint with a dev_train row (no birth UID exists; 103 mostly-categorical
  features over 2M rows make collisions expected). Quantified the optimism: held-out AUC full 0.6528
  vs de-duplicated 0.6529, delta -0.0001. No meaningful inflation: identical coarse features map to
  either outcome, so the model cannot memorize them. Documented as FINDINGS F7. No fix needed beyond
  disclosure.
- [2026-06-18] D29 DONE: Compliance fixes: removed all em dashes from tracked files (SPEC.md);
  config.yaml now actually read at runtime by the scripts (seed, sanity band, threshold, race
  categories, suppression floors) instead of hardcoded literals; FR1.1 confirmation now machine-logged
  per year to reports/format_confirmation.json with a config-validation guard (FR1.4); splits saved to
  data/processed/splits.parquet; added SPEC-mandated tests (test_format_gate_logged, test_schema_all_years,
  test_preterm_rate_in_sanity_band, test_target_range_20_45, parametrized sentinel-overlap test);
  age-bracket breaks unified in schema.AGE_BRACKET_BREAKS; spread-sample edge case and leakage-detector
  empty-input case hardened.

## Judge pass on Phases 3-7 (2026-06-20, 3 fresh-context agents: code, data/stats, judge/spec)

- [2026-06-20] D30 DONE: Decomposition headline corrected. The earlier 43% explained was the
  optimistic endpoint because the adjustment set included clinical-history mediators (prior preterm,
  the medical-risk composite). The PRIMARY decomposition now adjusts only for demographic + SES/access
  covariates (age, education, payment, WIC, BMI, marital, tobacco) and EXCLUDES mediators; a sensitivity
  adds clinical history. Reported as a range (roughly 30-40% explained), not a single point, with the
  covariates-are-products-of-racism caveat making even that an overstatement. Decomposition now uses the
  held-out partition for consistency. Source: data/stats critic HIGH-2.
- [2026-06-20] D31 DONE: Calibration framing corrected (was oversold). Added quantile-bin ECE, an MCE
  minimum-bin-size guard (30 rows), and a risk-spread diagnostic (share of predictions within +/-5pp of
  the base rate, about 84%). FINDINGS and the Model Card no longer call the scores "trustworthy" without
  qualification: low ECE means aggregate calibration, NOT reliable individual risk stratification.
  Source: data/stats critic BLOCKER-1/2.
- [2026-06-20] D32 DONE: Held-out leakage in Phase 7 mitigation fixed: equal-opportunity and screening
  thresholds are now derived on dev_test and only evaluated on held-out (Phase 4 already did this).
  Bootstrap iterations raised from 800 to the configured 2000 (FR8.1). Phase 3 now calibrates on
  dev_test (not dev_val) and reports raw vs calibrated ECE. Source: code reviewer B1, judge M3/L1.
- [2026-06-20] D33 DECISION: SPEC FR9 named Fairlearn + Aequitas. The hand-rolled metrics (equal
  opportunity, demographic parity, disparate impact, stratified permutation null, BH correction,
  threshold optimizer) are kept as the PRIMARY implementation because they are transparent and fully
  tested, and a Fairlearn cross-check (MetricFrame) is added as an independent validation plus an import
  smoke test. Aequitas is NOT adopted (heavy dependency, redundant with the above); SPEC FR9 is amended
  to reflect "hand-rolled primary + Fairlearn cross-check" rather than mandating Aequitas. README/SPEC
  updated. Source: judge BLOCKER B1.
- [2026-06-20] D34 DEFERRED: SPEC FR7 lists CatBoost, FT-Transformer/TabNet, and a stacked ensemble.
  These are scheduled as Phase 5 (deep learning) and Phase 6 (ensemble) in the roadmap and are
  explicitly deferred, not dropped. Rationale: gradient-boosted trees (XGBoost/LightGBM) already
  establish the expected top tabular performance (Correction 4); DL is a benchmark with a wall-clock
  cost, and the ensemble depends on the DL base learner. CatBoost is a fast add scheduled with Phase 6.
  Source: judge M2.

## Phase 8 exploration (2026-06-20)

- [2026-06-20] D35 DONE: Genuine exploration added as a tested module (src/equibirth/explore/analysis.py,
  9 unit tests) plus scripts/phase8_exploration.py, producing four new findings (FINDINGS F12-F15):
  intersectional race x education (gap flat-to-widening across education, college-educated Black >
  HS-or-less White), COVID-era trend (null national shift, contradicting localized hospital studies),
  unsupervised KMeans/PCA structure (weak silhouette ~0.25, risk strata are clinical not racial), and a
  layered ceiling-probe ablation (race-only AUC 0.551 -> full 0.634). A test caught a real bug: the
  initial suppression rule nulled legitimate 0%-rate cells with large n; fixed to denominator-based
  suppression. All numbers regenerated from the modeling parquet; interpretation associational. Figures
  follow the no-grey-subtext rule (takeaways in titles/annotations). Source: phase8_exploration.{md,json}.
- [2026-06-20] D36 DONE (closes D34 deep-learning/ensemble deferral, in part): scripts/phase56_benchmark.py
  benchmarks CatBoost and a random forest against the tuned XGBoost/LightGBM, plus a stacked logistic
  meta-learner and a mean ensemble, all isotonic-calibrated on dev_test and evaluated on held_out_final.
  Result confirms Correction 4 and D34: the tree field clusters at the modest ceiling and the ensemble
  adds little. FT-Transformer/TabNet stay deferred (wall-clock cost, expected non-winner). See
  reports/phase56_benchmark.md for the exact held-out table.

## Full-data rebuild of Phases 2-4 (2026-07-05)

- [2026-07-05] D37 DONE: Phases 2-4 rebuilt on the COMPLETE 2018-2024 public-use files (24,941,021
  singletons), superseding the earlier spread SAMPLE for these phases. Added a tested full-population
  assembly path (src/equibirth/features/full_dataset.py: load_full_frame, outcome_audit,
  assemble_modeling) plus splits.stratified_subsample, all TDD (tests/test_features_full_dataset.py, 8
  tests). Design (per SPEC FR5/FR8.1): the OUTCOME/EQUITY audit runs on every row (so AIAN/NHOPI are
  reliable), while MODEL DEVELOPMENT uses a deterministic stratified 3M working sample (dev_train/val/
  test + a 1.5M held_out_final for reported metrics and bootstrap CIs). All headline numbers
  independently recomputed and confirmed by a fresh-context data/stats validator; the sample-era
  conclusions held (moved by <=~0.02 AUC). Source: scripts/week2_feature_leakage_audit.py, F17-F19.
- [2026-07-05] D38 DONE: The `src/equibirth/models/` package (evaluate, baselines, boosting) was written
  to satisfy the pre-existing tracked-but-red tests (test_models_{evaluate,baselines,boosting}.py); on
  the deploy branch the Phase 3/4 scripts did the modeling inline and the modules were never committed
  because .gitignore's `models/` artifact rule ALSO matched src/equibirth/models/. Added a negation
  (`!src/equibirth/models/` + `!src/equibirth/models/**`) so the source package is tracked while the
  root models/ artifacts dir stays ignored. Full suite now 122 passed, 4 skipped (was red at baseline).
- [2026-07-05] D39 DONE (correction from full-cycle code review): the maternal-education main effect
  (`meduc`, a legitimate numeric feature emitted by features.build) was being silently dropped from the
  model matrix because the run scripts listed `meduc` in NON_FEATURES (it is also carried as a meta
  column). Removed from NON_FEATURES in week3/week4 scripts. Effect on metrics was negligible (RF 0.650
  -> 0.651) because the 72 saturated race x education interaction dummies already carried education, but
  the fix honors the tested build.py contract. Source: code-reviewer MEDIUM finding.
- [2026-07-05] D40 DONE (performance, laptop-scale): heavy full-data training must run in the FOREGROUND
  (background python jobs get their child killed in this environment) and be memory-bounded: float32
  matrix, `del df` after numpy extraction, RandomForest n_jobs=4 (not -1) with bounded depth/leaf, train
  capped at 500K stratified rows (past the modest ceiling), and bootstrap/permutation CIs on a fixed
  250-300K held-out subsample (each AUC-PR resample on 1.5M rows is ~0.25s; the CI converges far sooner).
  Point metrics still use the full 1.5M held-out. Peak RSS stays ~4-6 GB, under the SPEC 8 GB budget.

- [2026-07-06] D41 DECISION (deep learning; extends and firms up D34): no CUSTOM neural network will be
  built for this project, and none is implemented yet. When the deep-learning week is reached it will
  use proven implementations (FT-Transformer via rtdl and/or TabNet via pytorch-tabnet) purely as an
  honest benchmark, with attention or feature-mask interpretability as the real deliverable, and an
  optional TabPFN-2.5 experiment on a size-appropriate stratified subsample labeled experimental.
  Evidence for expecting trees to lead and a bespoke network to add no value: live max absolute Pearson
  correlation with the target among key numeric features is 0.131 (prior preterm), working-set
  prevalence 0.0857, the published administrative/birth-certificate ceiling is roughly 0.62 to 0.70
  AUC-ROC, tuned XGBoost/LightGBM already reach held-out AUC-ROC 0.657 (F19), the entire tree field plus
  ensembles clustered within 0.008 AUC (F16), and Grinsztajn et al. (NeurIPS 2022,
  https://arxiv.org/abs/2207.08815) show gradient-boosted trees remain state-of-the-art on tabular data
  and that neural nets struggle with uninformative features and irregular targets (this dataset). A
  hand-rolled network would add untested code and leakage/calibration risk with no expected accuracy
  gain. Report whichever model wins honestly; the expectation is that trees lead.

## Weeks 5-9 (deep learning, ensembles, fairness mitigation, robustness)

- [2026-07-19] D42 DONE (final model selection): XGBoost selected as the primary model over the
  cross-fitted logistic stack. The stack reached held-out AUC-PR 0.171 versus XGBoost's 0.172; the
  difference is within bootstrap noise (CI overlap complete). XGBoost is preferred for auditability:
  single model, native NaN handling, no dependency on the deep-net base learners (which add
  imputation/standardization and ~14-minute CPU wall time for FT-Transformer alone). The simpler model
  is favored whenever the margin is negligible. Source: reports/week5_models.json.
- [2026-07-19] D43 DECISION (fairness mitigation stance): ThresholdOptimizer is reported analytically
  as the only mitigation that meaningfully reduces equal-opportunity gaps while retaining useful
  sensitivity (worst EO gap 0.39 to 0.17). However, it is NEVER exposed in any deployed artifact
  because it requires disparate treatment (race-specific decision thresholds), which is in tension with
  anti-discrimination law and with the project's equity framing. ExponentiatedGradient is likewise
  reported but not adopted (near-zero sensitivity). Reweighting is reported as ineffective on this data.
  The Streamlit app uses a single universal threshold. Source: reports/week6_mitigation.json.
- [2026-07-19] D44 DONE (external-context sourcing): published aggregate statistics (CDC NVSR Vol. 73
  No. 1, March of Dimes Report Card, NHANES 2017-2020) are cited with exact URLs and access dates,
  used only as directional context for the model's findings. No microdata linkage is performed (none is
  legal or ethical with public-use birth-certificate data); no live API pull (URLs are reference points,
  not data feeds). Source: reports/week9_external_context.json.
- [2026-07-19] D45 DECISION (multiclass severity): a 4-class protocol (very preterm <28w, moderate
  28-31w, late preterm 32-36w, term 37+w) was designed but NOT run. The exact-week gestation column
  (oegest_comb) is not in the modeling Parquet (only the binary preterm indicator is retained);
  reconstructing it from interim files would require re-running the full preprocessing pipeline
  (estimated >30 minutes). Descoped to protocol documentation. Rationale: the binary pipeline and
  fairness audit are the load-bearing deliverables; the multiclass extension can be added if the
  gestation column is surfaced in a future preprocessing refresh. Source:
  reports/week9_external_context.json (multiclass_protocol field).
- [2026-07-19] D46 DONE (deep-net compute caps): deep models trained on a 30K-row capped subsample
  (versus 500K for boosters) with max 3 epochs and 100K held-out evaluation sample. FT-Transformer is
  CPU-bound at approximately 14 minutes even with d_token=32, n_blocks=2, batch_size=1024 on a laptop;
  TabNet is faster (~3-5 min). The caps keep the deep-learning week feasible on a single CPU within the
  SPEC 8 GB memory budget. Larger caps would not change the outcome (trees win at any scale on this
  data). Source: reports/week5_models.json (caps field).

## Phase 1 data acquisition

- [2026-06-18] D25 DONE: Full-year CSV is 2.23 GB and the NBER link throttles to ~0.8 MB/s, so a single
  download times out (curl exit 28). For representative national statistics without a 50-minute pull,
  use a "spread sample": concatenate N chunks downloaded at evenly spaced byte offsets across the full
  file (the CSV is sorted, so a head slice is biased; a spread sample is representative). Validated: a
  10-chunk, 293K-row spread sample reproduces the published 2024 national rate (10.65% vs 10.41%) and
  matches per-race published figures (White 9.5% and Hispanic 10.1% exact). Full multi-year download
  deferred to Phase 2 modeling; the ingestion pipeline is year-agnostic and tested. See FINDINGS F2-F4.

## Phase 0 engineering decisions

- [2026-06-17] E1 DONE: Project uses an isolated `.venv` (Python 3.11.9) for reproducibility, not the global interpreter. Phase 0 installs only pytest + pyyaml; heavier deps deferred per phase.
- [2026-06-17] E2 DONE: Single `config.yaml` is the only source of runtime settings, read via `equibirth.utils.config.load_config`. No hardcoded paths/seeds/params elsewhere.
- [2026-06-17] E3 DONE: `.gitignore` excludes private working files, all data dirs, and model artifacts, so no sensitive file or row-level data can be pushed.
