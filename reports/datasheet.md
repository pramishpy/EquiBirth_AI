# Datasheet: U.S. Natality Modeling Dataset (EquiBirth-AI)

> **This is NOT a clinical tool.** The dataset is for educational research only.

*Structure follows Gebru et al. (2021), "Datasheets for Datasets"*
*Reference: https://arxiv.org/abs/1803.09010*

---

## 1. Motivation

- **Purpose**: Educational research on preterm-birth prediction and racial equity in the
  United States, as part of an AI4ALL responsible-AI portfolio project.
- **Creators**: Independent research; no institutional affiliation with NCHS or CDC.
- **Funding**: None (self-directed educational work).

---

## 2. Composition

| Field | Value |
|-------|-------|
| Unit of observation | Individual singleton live birth |
| Source population | Approximately 24.9M singletons, 2018-2024, U.S. vital statistics |
| Working sample | Approximately 3M records (stratified partitions for modeling) |
| Splits | dev_train (1.05M), dev_val (225K), dev_test (225K), held_out_final (1.5M) |
| Feature count | 103 pre-delivery features |
| Target | Preterm birth (obstetric estimate of gestation < 37 weeks) |
| Prevalence | 8.6% in the held-out partition (n=1,498,760) |
| Identifiers | None (de-identified public-use files; no SSN, name, or address) |
| Geography | Suppressed in public-use files post-2005 (no state/county) |

### Demographic composition (held-out partition)

| Group | N | Preterm Rate |
|-------|---|-------------|
| NH White | 756,298 | 7.5% |
| NH Black | 208,755 | 12.3% |
| Hispanic | 377,388 | 8.6% |
| NH Asian | 92,342 | 7.6% |
| NH AIAN | 10,806 | 10.7% |
| NH NHOPI | 4,040 | 10.8% |
| NH Multiracial | 35,333 | 8.9% |

---

## 3. Collection Process

- **Source**: CDC/NCHS Natality public-use microdata files, accessed via the NBER mirror
  (https://www.nber.org/research/data/vital-statistics-natality-birth-data).
- **Revision**: 2003 revised birth certificate (standard since 2016 for all states).
- **Download**: Programmatic download of full-year CSV files (2018-2024); integrity verified
  by SHA-256 hashes and expected row counts.
- **Geography limitation**: Public-use files suppress geographic identifiers (state, county)
  after 2005 to protect confidentiality.
- **No consent required**: These are de-identified administrative records released for public
  research use by NCHS.

---

## 4. Preprocessing

- **Singleton filter**: Multiple births excluded (different risk profile and clinical pathway).
- **Variable harmonization**: Per-variable "unknown/not stated" sentinel codes converted to
  null. Race and education harmonized to consistent 7-group and 9-level categories across years.
- **Feature derivation**: Medical-risk count (sum of binary risk factor flags), age bracket,
  race-education interaction terms, first-trimester care indicator.
- **Leakage audit** (Week 2): Post-delivery variables (delivery method, APGAR, abnormal
  conditions, NICU admission) identified and excluded. Timing-conditional variables
  (obstetric estimate, plurality confirmation) audited and documented.
- **Splits**: Deterministic stratified (by preterm status and race) random partition saved
  to disk. No data leakage between train/val/test/held-out.
- **No SMOTE**: Class weights and threshold selection preserve calibration.

---

## 5. Uses

- **Intended**: Population-level equity research, methodology demonstration, educational
  AI fairness portfolio.
- **Out of scope**: Clinical deployment, individual risk scoring, commercial use,
  surveillance.
- **IRB status**: IRB-exempt (publicly available de-identified data; no human subjects
  interaction).
- **Linkage**: None. No linkage to NHANES, hospital records, claims data, or any other
  source. External context (CDC aggregates) is published summary statistics only.

---

## 6. Distribution

- **License**: Public domain. NCHS natality public-use files carry no use restrictions
  beyond citation of the source.
- **Access**: Available from NBER (https://www.nber.org/research/data/vital-statistics-natality-birth-data)
  or CDC WONDER for aggregates.
- **Format**: CSV (raw), Parquet (processed modeling file).

---

## 7. Maintenance

- **Update cadence**: NCHS releases new natality public-use files annually (typically
  with a 12-18 month lag).
- **Re-validation trigger**: Re-run the confirmation gate (row counts, column layouts,
  SHA-256) and the Week 8 drift table on any new year. AUC-ROC degradation beyond 0.02
  from the 2018-2024 baseline warrants investigation.
- **Contact**: Not applicable (educational project; no maintained deployment).
- **Versioning**: Dataset version is defined by the year range (currently 2018-2024) and
  the processing pipeline commit hash.

---

*Generated automatically from tracked weekly JSON reports. See scripts/week9_transparency.py.*
