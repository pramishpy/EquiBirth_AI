# Week 9: External Context (Published Aggregate Statistics)

> **No microdata linkage.** All figures below are published summary statistics
> from CDC, NCHS, and March of Dimes reports. They provide reference context
> for the model's findings but are not computationally linked to our dataset.

---

## Published Preterm Birth Statistics

| Statistic | Value | Source |
|-----------|-------|--------|
| National preterm rate 2022 | 10.4% | CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1 |
| NH Black preterm rate 2022 | approximately 14.6% | CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1 |
| NH White preterm rate 2022 | approximately 9.4% | CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1 |
| Highest-rate state 2022 (Mississippi) | approximately 14.5% | March of Dimes Report Card 2023 |
| Lowest-rate state 2022 (Oregon) | approximately 8.5% | March of Dimes Report Card 2023 |
| Maternal hypertension prevalence (NHANES directional context) | NH Black women aged 20-44: approximately 40%; NH White wo... | NHANES 2017-March 2020 pre-pandemic data |

---

## Detailed Notes

### 1. National preterm rate 2022

- **Value**: 10.4%
- **Source**: CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1
- **URL**: https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-01.pdf
- **Access date**: 2026-07-19
- **Linkage**: none; aggregate context only
- **Note**: This is the overall preterm rate including multiples. Our singleton-only sample has a lower rate (approximately 8.6%) because multiples have much higher preterm incidence.

### 2. NH Black preterm rate 2022

- **Value**: approximately 14.6%
- **Source**: CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1
- **URL**: https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-01.pdf
- **Access date**: 2026-07-19
- **Linkage**: none; aggregate context only
- **Note**: Includes multiples. Consistent with the elevated preterm rate for NH Black mothers in our singleton sample (12.3% held-out).

### 3. NH White preterm rate 2022

- **Value**: approximately 9.4%
- **Source**: CDC NCHS National Vital Statistics Reports, Vol. 73, No. 1
- **URL**: https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-01.pdf
- **Access date**: 2026-07-19
- **Linkage**: none; aggregate context only
- **Note**: Includes multiples. Our singleton-only NH White rate is 7.5% (held-out), lower due to exclusion of multiples.

### 4. Highest-rate state 2022 (Mississippi)

- **Value**: approximately 14.5%
- **Source**: March of Dimes Report Card 2023
- **URL**: https://www.marchofdimes.org/report-card
- **Access date**: 2026-07-19
- **Linkage**: none; aggregate context only
- **Note**: State-level variation cannot be studied in our data because NCHS public-use files suppress geography post-2005.

### 5. Lowest-rate state 2022 (Oregon)

- **Value**: approximately 8.5%
- **Source**: March of Dimes Report Card 2023
- **URL**: https://www.marchofdimes.org/report-card
- **Access date**: 2026-07-19
- **Linkage**: none; aggregate context only
- **Note**: State-level; aggregate context only.

### 6. Maternal hypertension prevalence (NHANES directional context)

- **Value**: NH Black women aged 20-44: approximately 40%; NH White women aged 20-44: approximately 28% (NHANES 2017-2020)
- **Source**: NHANES 2017-March 2020 pre-pandemic data
- **URL**: https://www.cdc.gov/nchs/nhanes/index.htm
- **Access date**: 2026-07-19
- **Linkage**: none; aggregate context only
- **Note**: NHANES is a health/nutrition survey with NO birth-outcome linkage. Used only as directional context: higher maternal hypertension prevalence among NH Black women is consistent with the model's rf_phype (pre-pregnancy hypertension) ranking as a top feature in the NH Black stratified model. No causal claim is made.

---

## Multiclass Protocol (Designed, Not Run)

**Status**: designed_not_run

**Description**: 4-class severity classification (very preterm <28w, moderate 28-31w, late preterm 32-36w, term 37+w) with timing-conditional features excluded.

**Rationale for descoping**: The exact-week gestation column used as target is not retained in the modeling parquet (only the binary preterm indicator). Reconstructing from interim files would require re-running the full preprocessing pipeline (estimated >30 minutes). Per spec, descoped to protocol-only.

**Planned approach**: Ordinal or multinomial extension of the selected XGBoost model; separate calibration per class; fairness audit on the very-preterm class (highest clinical severity); same feature set minus any timing-conditional variables that correlate with severity class.

---

*Generated automatically. See scripts/week9_transparency.py.*