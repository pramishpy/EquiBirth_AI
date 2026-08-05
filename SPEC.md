# SPEC.md - EquiBirth AI (v0.3, FINAL - two review cycles applied)

> Two full cycles of independent spec review and criticism are applied. All BLOCKER and HIGH
> findings are resolved here. Every requirement remains provisional against the real data: the
> FR1.1 format/variable-confirmation gate is the first executable task, and anything it
> contradicts is corrected in `DECISION_LOG.md`.

## 1. Purpose and scope

Build a reproducible, calibration-first pipeline that (a) predicts preterm birth (<37 weeks,
obstetric estimate) from U.S. birth-certificate features and (b) audits the equity of both the
models and the underlying outcomes across race/ethnicity and SES proxies. The narrative is honest
and falsifiable (Section 11): modest individual predictability, and a racial disparity whose
unexplained share is reported as the data show it, not assumed. Responsible AI is the spine.

Out of scope: any clinical decision tool; any row-level geographic join; any causal claim without
explicit caveats; race as a modifiable risk input in the demo; "income" framing (no income field
exists on birth certificates; use "SES proxy").

## 2. Data year range (decided, DECISION_LOG D6)

Use natality public-use microdata **2019 through 2024**. 2024 is the latest available (NBER added it
September 2025). **2025 excluded with justification**: no public-use microdata file exists; 2025 is
provisional aggregates only via CDC WONDER. If 2019-2024 harmonization needs remapping on >20% of
columns, fall back to **2022-2024** and log it. 2020-2021 are flagged COVID-era for sensitivity.

## 3. Functional requirements

- **FR1 Ingestion.**
  - FR1.1 **Format + variable confirmation gate (BLOCKER, first executable task):** download one year,
    inspect raw bytes, confirm fixed-width vs CSV and that widths match the User Guide. Confirm exact
    column names for the obstetric estimate (target, expected `OEGEST_COMB`) and the LMP-based estimate
    (excluded, expected `COMBGEST`), and the singleton/plurality, race, education, payment, WIC, prior-
    preterm variables. Populate `gestational_age_var_by_year` and `lmp_gestation_var_by_year` in config.
    Log everything in DECISION_LOG. No parser/feature code merges before this gate passes.
  - FR1.2 Layout-driven fixed-width parser keyed off the User Guide + NBER dictionary (no hardcoded
    offsets). Each year parsed independently to `data/interim/natality_{year}.parquet`.
  - FR1.3 Integrity: per-file SHA-256, row count (within +/-5% of `expected_row_counts_by_year`), and
    source URL in `data/raw/manifest.json`, checked in CI. Pipeline aborts on mismatch.
  - FR1.4 A startup config-validation guard raises `ConfigurationError` if the gestational-age maps are
    empty when the pipeline runs.
- **FR2 Data dictionary.** Built programmatically from the User Guide. Per variable: name, type, recode
  logic, **per-variable** sentinel codes, valid range, and **leakage class** (pre-conception /
  early-pregnancy / timing-conditional / outcome-entangled / structural-filter).
  - FR2.1 Layout-drift test diffs column specs across years; CI fails on unresolved discrepancy.
    Race/ethnicity and education encodings harmonized as a named, tested step to the categories in
    `config.data.race_ethnicity_harmonized_categories`, validated against an NVSR published total.
- **FR3 Missing-value handling.** "Unknown" codes to NaN via a **per-variable** map (no global list).
  A test per variable asserts its sentinel codes do not overlap its valid range (9 visits, parity 9,
  plurality 5 must survive).
- **FR4 Target.** Binary `preterm = obstetric_estimate < 37`, variable pinned per year (config). The
  LMP-based variable is excluded (Section 6). Sanity tests (FR4.1): target non-null >99%; range 20-45
  weeks; full-dataset preterm rate within 9.5-11.5%; build fails otherwise. Optional 4-class severity
  target only after binary is solid; timing-conditional features excluded from extreme/very models.
- **FR5 Sampling/splits.** Filter to singletons (verify plurality encoding; report pre/post counts AND
  race-stratified preterm rates before/after; flag in FINDINGS if the Black-White gap shifts >10%
  relative). Stratified `dev_sample` by `[preterm, race/ethnicity, birth_year]`; oversample rare race
  groups (plain, not SMOTE) for training only, with weights for population estimates. `held_out_final`
  = remainder (final metrics come from here, never the Optuna split). Deterministic splits saved.
  Split strategy logged; temporal sensitivity check (train pre-COVID, test 2020-2021).
- **FR6 Feature pipeline + leakage control (BLOCKER artifact).** Each engineered feature declares its
  `components` (source variables) in the data dictionary. The primary matrix uses pre-delivery features
  only. Two enforced tests:
  - FR6.1 **Excluded-vars detector:** a trivial classifier on `leakage_detector_vars` alone; if its
    AUC > `excluded_vars_leakage_gate` (0.55) the build fails. `structural_exclusion_vars` (DPLURAL,
    alternative gestational-age columns) are NOT in this detector (their gestation association is the
    filter rationale, not post-delivery leakage).
  - FR6.2 **Composite-component guard:** for every feature in the primary matrix, assert no declared
    component is in the exclusion lists. This closes the laundering path where a composite wraps a
    leaky variable. The medical-risk composite is enumerated explicitly (no open-ended "..."): it is
    the count of {pre-pregnancy diabetes, pre-pregnancy hypertension, previous preterm birth, previous
    C-section}. `RF_GDIAB` (gestational diabetes) is NOT in the composite for any preterm model
    (diagnosis postdates early/moderate preterm).
  - FR6.3 Timing-conditional features kept in the primary model pass a univariate gate: if a feature's
    univariate AUC > `univariate_leakage_gate` (0.60) it is excluded. "Month prenatal care began" is
    recoded to binary first-trimester care (months 1-3) and is subject to this gate.
- **FR7 Models.** Logistic regression (calibration reference), decision tree, random forest, XGBoost,
  LightGBM, CatBoost, FT-Transformer and/or TabNet, stacked ensemble. Optuna tunes on `dev_internal_val`
  for trial scoring and early stopping; `held_out_final` is untouched until final reporting (no tuning
  leakage). Race-stratified models only for groups with >=10K training records. DL on the dev sample
  with a wall-clock cap (`optuna_runtime`/timeouts). TabPFN-2.5 optional, experimental, subsampled.
- **FR8 Evaluation.** AUC-PR (primary), AUC-ROC (cross-group comparator), F1, sensitivity, specificity,
  Brier, reliability curves with **ECE and MCE** (12 bins), overall AND stratified by race/ethnicity and
  **SES proxies** (education, payment source, WIC; never labeled "income"). Calibration on **raw
  probabilities** before any threshold; threshold selection reported separately. Post-hoc recalibration
  (Platt/isotonic on a held-out fold) is a named step; pre/post metrics separate.
  - FR8.1 Bootstrap CIs (2000 stratified iterations) on the **`held_out_final` remainder**, never the
    Optuna split. Tiered subgroup reporting (do not hard-blank vulnerable groups): suppress at <50
    positives; point estimate with "wide CI, interpret with caution" at 50-999; full bootstrap CI at
    >=1000. Always report subgroup n and event counts. Separate the **outcome equity audit** (full
    population counts, reliable for AIAN/NHOPI) from the **model-performance audit** (needs adequate
    cell sizes). Cross-group comparisons use AUC-ROC; AUC-PR not compared directly across prevalences.
- **FR9 Fairness.** Demographic parity difference, equal-opportunity difference, disparate-impact ratio,
  calibration-by-group. Implemented as transparent, fully-tested hand-rolled metrics (PRIMARY) with a
  Fairlearn MetricFrame cross-check and an import smoke test (DECISION_LOG D33); Aequitas is not adopted
  (heavy, redundant). Versions pinned, imports smoke-tested, docs via context7
  first). Each disparity metric carries a 95% bootstrap CI and a **stratified permutation-test** p-value
  (race labels shuffled within y=1 and within y=0 independently; Ojala & Garriga 2010), so the test
  isolates disparity from base-rate differences. Metrics whose CI includes parity are labeled "not
  distinguishable from parity." Four-fifths rule is a caveated reference, not the headline.
  - FR9.1 Multiple comparisons: the full comparison family is **pre-registered** in FINDINGS.md (every
    planned comparison with hypothesized direction, dated) before first data touch. The primary
    comparison (race/ethnicity, equal-opportunity difference, best model) is confirmatory; all others
    are exploratory and Benjamini-Hochberg corrected, with uncorrected p-values shown alongside. Any
    comparison added after EDA begins is labeled "post-hoc (date)."
  - FR9.2 Mitigation experiments (reweighting, Fairlearn reductions, ThresholdOptimizer) reported with
    accuracy/fairness/calibration tradeoffs. The disparate-treatment vs disparate-impact tension of
    race-specific thresholds is discussed; no single mitigation presented as correct.
- **FR10 Explainability + decomposition.** Global and race-stratified SHAP; DL attention maps; PDP
  stated associationally. Decomposition of the Black-White gap: linear Oaxaca-Blinder **excluded**; use
  Fairlie's nonlinear decomposition (primary) or g-computation (secondary) with a stated DAG.
  - FR10.1 **Working DAG (logged as D16):** race/ethnicity acts on preterm risk through (1) an SES
    pathway (education -> insurance/WIC -> prenatal-care access -> preterm) and (2) a direct/unmeasured
    structural pathway. Maternal age and parity are pre-SES confounders adjusted in all models. SES
    covariates are mediators in the decomposition but are included in the predictive model for
    prediction (not causal) purposes, with that distinction stated.
  - FR10.2 **Mandatory caveat (body text, not footnote):** SES covariates are themselves products of
    structural racism, so the "explained" share is inflated and the "unexplained" residual is a lower
    bound on structural contributions, labeled explicitly as a theoretical commitment, not proven by
    this analysis.
- **FR11 Deliverables.** Streamlit demo, per-module notebooks compiled last, Model Card, Datasheet,
  poster, demo script, clean public repo.
  - FR11.1 Streamlit must NOT expose race-specific thresholds in the calculator, and the what-if panel
    must NOT present race as a modifiable input. Any race display triggers an interstitial: race is a
    social construct and marker of exposure to racism, not a biological cause; it illustrates measured
    disparities, not a clinical intervention. Prominent "not a clinical tool" disclaimer throughout.
  - FR11.2 CDC WONDER state aggregates acquired manually on first run, committed to `data/external/`
    with a manifest SHA-256 (WONDER ToS restricts automated querying; Playwright is a stretch goal,
    acknowledged in the Datasheet). Choropleth caption template (required): "State-level variation is
    descriptive. Rates reflect birth-certificate records; do not infer state-policy or causal effects
    without controlling for population composition." The WONDER map is a human-review gate, not part of
    the automated >=20-plot CI gate.
  - FR11.3 Model Card documents intended/out-of-scope use, stratified evaluation, the leakage caveat,
    the modest AUC ceiling, that timing-conditional variables are unavailable prospectively, expected
    degradation under distribution shift (incl. COVID-era), and a recommended re-validation schedule.

## 4. Non-functional requirements

- NFR1 Reproducibility: fixed seeds; one `config.yaml`; deterministic saved splits; pinned env;
  `make reproduce` completes offline after initial download to a stored checksum manifest within
  tolerance; expensive steps cached with integrity checks.
- NFR2 Engineering quality: typed signatures, docstrings, structured logging, small deep modules, no
  duplication; mirrored unit/integration/e2e suite; CI runs `pytest --cov`, 80% floor on
  `src/equibirth/` (applies to code that exists at each phase, not unwritten modules).
- NFR3 Performance: Polars primary (lazy), per-year parse then lazy concat; peak RSS profiled per year;
  a single-year parse exceeding ~8 GB peak RSS is rejected.
- NFR4 Calibration-first: default class weights + threshold tuning; no uncaveated SMOTE.
- NFR5 Leakage control: enforced by FR6.1 and FR6.2.
- NFR6 Data versioning: `manifest.json` checked in CI; data change requires a DECISION_LOG entry.
- NFR7 Ethics: race-as-social-construct framing; no-clinical-tool disclaimer; causal humility; cell
  suppression for n<50; no em dashes; no sensitive or private files committed.

## 5. NHANES (secondary, contextualization only)

Context and disparity sanity-checks only; NOT row-linkable to natality. Population estimates use complex
survey weights (Taylor-series variance via statsmodels survey tools or samplics), clearly labeled.

## 6. Leakage classification (enforced by FR6 tests; confirm exact names per year via FR1.1)

`leakage_detector_vars` (post-delivery / target-encoding; tested by FR6.1 AUC>0.55 gate):

| Concept | Expected column(s) | Why excluded |
| --- | --- | --- |
| Birthweight | DBWT, BWTR4, BWTR12 | consequence of gestation duration |
| Apgar | APGAR5, APGAR10 | scored after birth |
| NICU admission | AB_NICU / ICU | post-delivery |
| Assisted ventilation | AB_AVEN1, AB_AVEN6 | post-delivery intervention |
| Abnormal newborn conditions | AB_SEIZ, AB_SURF, AB_ANTI, ... | post-delivery diagnoses |
| Congenital anomalies | CA_* | recorded at delivery |
| Antenatal steroids | LD_STER | given because preterm anticipated (highest-risk leak) |
| Labor antibiotics | LD_ANTB | given because preterm anticipated |
| Induction / augmentation | LD_INDL, LD_AUGM | co-determined with preterm |
| Chorioamnionitis | LD_CHOR | diagnosed at delivery |
| Meconium / PROM | LD_MECO, LD_PROM | recorded at delivery |
| Intrapartum infection/treatment | IP_GON, IP_SYPH, NO_INFEC | noted during labor |
| This delivery's method | DMETH_REC, ME_ROUT | caused partly by preterm |
| Attendant / facility | ATTEND, BFACIL | recorded at delivery |
| Total prenatal visits | UPREVIS | mechanically capped by gestation length |
| APNCU adequacy recode | DOREVIS (APNCU/Kotelchuck) | uses gestational age as input; U-shaped |

`structural_exclusion_vars` (excluded by logic, NOT in the FR6.1 detector):
- DPLURAL (analysis is singleton-only; its gestation link is biology, not post-delivery leakage).
- The LMP-based gestational-age column (COMBGEST) and any recomputed gestational-age field (would
  encode the target).

`timing-conditional` (flagged; associational interpretation only; univariate gate FR6.3; Model Card
notes they are unavailable prospectively): eclampsia, pregnancy-associated hypertension, gestational
diabetes RF_GDIAB (excluded from extreme/very models and from the composite), WIC, binary
first-trimester care (recoded from "month care began").

`legitimate pre-delivery` (composite enumerated, no "..."): maternal age (bracketed), race/ethnicity,
education, marital status, nativity, pre-pregnancy diabetes, pre-pregnancy hypertension, previous
preterm birth, previous C-section, parity/birth order, pre-pregnancy BMI, tobacco before/during,
payment source, the enumerated medical-risk composite (FR6.2), race x education interaction.
Borderline: infant SEX (weak, non-actionable; include only with a note).

## 7. Process requirements

Every task runs the full cycle (plan, analyze, read, failing-test-first, code review, critic,
security/perf/architecture review, validate on real data, judge), with an independent reviewer: a
task's reviewer/critic is never its author. `DECISION_LOG.md` and `FINDINGS.md` are the durable record.

## 8. Phase 1 acceptance criteria (tiered, to avoid circular blocking)

**Tier A (must be fully complete to pass Phase 1):**
1. FR1.1 gate done; gestational-age maps populated; format logged.
2. Per-year Parquet files exist and pass schema validation (exact columns, dtypes, non-null rates).
3. `manifest.json` with per-file SHA-256, URL, date, row count (within +/-5%); verified in CI.
4. Per-variable missing-value report produced; no global sentinel used.
5. Target plausibility: full-dataset preterm rate 9.5-11.5%; range 20-45; <1% null.
6. Singleton filter applied with pre/post counts AND race-stratified rates before/after (FR5).
7. Race/ethnicity + education harmonized and validated against a published total.
8. >=20 captioned EDA plots producible from the parsed microdata (automated gate).
9. >=1 confirmed finding in FINDINGS.md from real data (not scaffolding).

**Tier B (wired/stubbed in Phase 1, fully functional in Phase 2):** FR6 leakage detector + composite
guard, bootstrap machinery, fairness library import smoke-tests, decomposition, permutation tests.

**Human-review gate (not automated):** CDC WONDER state map present with a manifest SHA-256.

### 8.1 Phase 1 deliverable contracts

- Path convention: `data/raw/natality{year}.dat` (raw), `data/interim/natality_{year}.parquet`.
- Parser signature: `ingest.parse_year(year: int, raw_path: Path, layout: Layout, out_path: Path) -> pl.DataFrame`.
- Minimum harmonized columns: preterm (target), obstetric_gestation_weeks, birth_year, mother_age,
  mother_race_ethnicity, education_level, marital_status, payment_source, wic_participation,
  prior_preterm, parity, prepreg_bmi, prepreg_diabetes, prepreg_hypertension, tobacco, plurality.

### 8.2 Required Phase 1 tests (names + assertion)

- `test_format_gate_logged` - DECISION_LOG records confirmed format + variable names.
- `test_schema_all_years` - each parquet matches the declared schema.
- `test_preterm_rate_in_sanity_band` - full-data rate in [0.095, 0.115].
- `test_target_range_20_45` - gestation weeks within [20, 45].
- `test_sentinel_does_not_overlap_valid_range` (parametrized per variable, e.g. UPREVIS, parity, DPLURAL).
- `test_singleton_filter_counts` - expected ~3-4% reduction; race-stratified rates emitted.
- `test_manifest_checksums` - manifest SHA-256/row counts validate.
- `test_race_harmonized_totals_match_published_spot_check`.
- `test_leakage_vars_absent_from_feature_matrix` - UPREVIS, DOREVIS, DBWT, LD_STER, etc. absent (Tier B wiring).
- `test_no_composite_component_is_excluded` - FR6.2 guard (Tier B wiring).

## 9. Final acceptance criteria

The 10-item definition-of-done checklist, applied by an independent reviewer.

## 10. Open questions (resolved in Cycle 2)

- Obstetric-estimate variable name: confirmed via FR1.1 before any code (working assumption
  OEGEST_COMB = target, COMBGEST = excluded LMP).
- NBER format: operationalized by the FR1.1 gate.
- "Income" -> relabeled **SES proxy** everywhere (D15 resolved).
- Decomposition: Fairlie primary, g-computation secondary; DAG in FR10.1 (D16).
- Singleton bias: race-stratified before/after diagnostic is a Phase 1 deliverable (D17).

## 11. Falsifiability of the thesis (anti-confirmation-bias guard)

The "modest predictability + large unexplained disparity" framing must be able to fail:
- If BH-corrected disparity metrics are not significant after SES adjustment, report that and discuss
  measurement limits, rather than asserting disparity.
- If the unexplained residual after full adjustment is below `falsifiability.residual_honesty_floor`
  (10% of the raw gap), report that honestly instead of defaulting to the structural framing.
- The SES-as-racism-product caveat is labeled a theoretical commitment, not a result.

---

## Review status

- [x] Cycle 1: independent spec review - 32 findings, applied
- [x] Cycle 1: independent critique - 21 findings, applied
- [x] Cycle 2: independent spec review - verdict reached, BLOCKER/HIGH applied
- [x] Cycle 2: independent critique - verdict reached, BLOCKER/HIGH applied

SPEC v0.3 is the Phase 1 baseline. Residual MEDIUM/LOW items are tracked in DECISION_LOG and addressed
in-phase. The FR1.1 confirmation gate is the first executable task.
