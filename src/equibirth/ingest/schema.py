"""Schema, column mapping, and leakage classification for the NBER natality CSV.

All raw column names are the lowercase names served in the NBER per-year CSV (confirmed against the
2023 and 2024 files via the FR1.1 gate; see DECISION_LOG D5). Names are standardized across years by
NBER, but the loader validates presence per year and fails loudly on a mismatch.

The leakage classification is the enforceable core of the project's integrity (SPEC Section 6). It is
expressed as sets of raw column names so tests and the FR6 detector can enumerate columns
deterministically rather than matching concept descriptions.
"""

from __future__ import annotations

# --- Target and its forbidden sibling --------------------------------------------------------------

# Obstetric estimate of gestation in completed weeks. preterm = OBSTETRIC_GESTATION < 37.
OBSTETRIC_GESTATION = "oegest_comb"
# LMP-based gestation. Correlated with the target; EXCLUDED so it cannot encode the outcome.
LMP_GESTATION = "combgest"

# --- Harmonized feature columns (raw name -> stable analysis name) ----------------------------------
# Only legitimate pre-delivery and timing-conditional variables are mapped here. Leakage variables are
# deliberately NOT given harmonized names so they cannot drift into the feature matrix by accident.
COLUMN_MAP: dict[str, str] = {
    # demographics (pre-conception, legitimate)
    "mager": "mother_age",
    "mrace6": "mother_race6",
    "mracehisp": "mother_race_ethnicity",
    "meduc": "education_level",
    "dmar": "marital_status",
    # medical history (pre/early-pregnancy, legitimate)
    "rf_pdiab": "prepreg_diabetes",
    "rf_phype": "prepreg_hypertension",
    "rf_ppterm": "prior_preterm",
    "rf_cesar": "prior_cesarean",
    "bmi": "prepreg_bmi",
    "priorlive": "prior_live_births",
    "priordead": "prior_dead_births",
    # behavioral (legitimate)
    "cig_0": "cigs_prepreg",
    # social determinants on the record (legitimate / timing-conditional)
    "pay": "payment_source",
    "wic": "wic_participation",
    # timing-conditional (flagged; associational only; univariate-gated)
    "rf_gdiab": "gest_diabetes",
    "rf_ghype": "gest_hypertension",
    "rf_ehype": "eclampsia",
    "precare": "month_prenatal_care_began",
    # structural
    "dplural": "plurality",
    "sex": "infant_sex",
}

# --- Leakage classification (SPEC Section 6) -------------------------------------------------------

# Recorded at or after delivery, or given because preterm was anticipated. These feed the FR6.1
# excluded-vars detector: a classifier on these alone must not exceed the leakage gate.
LEAKAGE_DETECTOR_VARS: frozenset[str] = frozenset({
    # newborn measurements / outcomes (post-delivery)
    "dbwt", "bwtr12", "bwtr4", "apgar5", "apgar5r", "apgar10", "apgar10r",
    # abnormal conditions of the newborn, incl. NICU admission (post-delivery, confirmed in real file)
    "ab_nicu", "ab_aven1", "ab_aven6", "ab_anti", "ab_seiz", "ab_surf",
    # congenital anomalies (recorded at delivery; some cause iatrogenic preterm)
    "ca_anen", "ca_mnsb", "ca_cchd", "ca_cdh", "ca_omph", "ca_gast", "ca_limb",
    "ca_cleft", "ca_clpal", "ca_down", "ca_disor", "ca_hypo",
    # labor / delivery interventions (ld_ster = single highest-risk leak)
    "ld_ster", "ld_antb", "ld_indl", "ld_augm", "ld_chor", "ld_anes",
    # intrapartum infection / treatment (noted during labor)
    "ip_gon", "ip_syph", "ip_chlam", "ip_hepb", "ip_hepc", "no_infec",
    "rf_inftr", "rf_fedrg", "rf_artec",
    # delivery method / presentation / attendant / facility
    "me_rout", "me_pres", "me_trial", "rdmeth_rec", "dmeth_rec", "attend", "bfacil", "bfacil3",
    # care-intensity entangled with gestation length
    "previs", "previs_rec",
})

# Excluded by logic, NOT part of the FR6.1 detector (their gestation association is biology / target
# encoding, not post-delivery leakage; including them in the detector causes false-positive CI fails).
STRUCTURAL_EXCLUSION_VARS: frozenset[str] = frozenset({
    "dplural",          # analysis is singleton-only
    LMP_GESTATION,      # combgest: would encode the target
    "gestrec10", "gestrec3", "oegest_r10", "oegest_r3", "obgest_flg",  # gestation recodes/flags
})

# Clinical maternal-age brackets (single source of truth; the relationship with preterm is
# U-shaped). build.py derives integer codes and eda.py derives string labels from these breaks.
AGE_BRACKET_BREAKS: tuple[int, ...] = (20, 25, 30, 35, 40)

# Iatrogenically outcome-entangled: pregnancy conditions that frequently CAUSE preterm delivery via
# induction (the delivery is preterm because of the condition). Independent review (2026-06-18) showed
# their univariate AUC is low (0.51-0.56) so the inflation is small, but the mechanism is reverse
# causal. They are EXCLUDED from the primary pre-diagnosis model and kept only for the post-hoc
# disparity analysis on the raw frame. See DECISION_LOG D26.
IATROGENIC_ENTANGLED_VARS: frozenset[str] = frozenset({
    "rf_gdiab",   # gestational diabetes (dx ~24-28 wks)
    "rf_ghype",   # pregnancy-associated (gestational) hypertension
    "rf_ehype",   # eclampsia: third-trimester emergency, common cause of iatrogenic preterm
})

# Kept in the primary model only with an associational caveat and a univariate leakage gate. These are
# weakly predictive alone (univariate AUC ~0.51) and not reverse-causal at delivery.
TIMING_CONDITIONAL_VARS: frozenset[str] = frozenset({
    "precare",    # month care began -> recoded to binary first-trimester, univariate-gated
    "wic",        # enrollment window shortened by early delivery
})

# The enumerated medical-risk composite (SPEC FR6.2; no open-ended members). RF_GDIAB deliberately
# absent (timing leak). Each engineered feature declares its components for the FR6.2 guard.
MEDICAL_RISK_COMPOSITE_COMPONENTS: tuple[str, ...] = (
    "rf_pdiab", "rf_phype", "rf_ppterm", "rf_cesar",
)

# Birth-certificate "unknown / not stated" sentinel codes, PER VARIABLE (never a global list).
# Verified against the real 2023 sample (see tests). Values here must NOT overlap a variable's valid
# range (e.g. 9 prenatal visits, parity 9, plurality 5 must survive).
SENTINEL_MAP: dict[str, list] = {
    "oegest_comb": [99],          # valid 17-47
    "combgest": [99],
    "meduc": [9],                 # valid 1-8
    "precare": [99],              # month care began; valid 0-10
    "previs": [99],               # valid 0-98
    "cig_0": [99],                # valid 0-98
    "bmi": [99.9],                # valid ~13-69.9
    "mrace6": [],                 # imputed, no unknown code
    "mracehisp": [],              # code 8 (unknown origin) handled by map_race_ethnicity, not here
    "mager": [],                  # always present, valid 12-50
    "dplural": [],                # valid 1-5 (5 must survive)
    # Y/N/U coded flags: 'U' (and blank) are unknown -> null. Handled as strings.
    "rf_pdiab": ["U"], "rf_phype": ["U"], "rf_ppterm": ["U"], "rf_cesar": ["U"],
    "rf_gdiab": ["U"], "rf_ghype": ["U"], "rf_ehype": ["U"],
    "wic": ["U"], "dmar": [9], "pay": [9], "sex": [],
    "priorlive": [99], "priordead": [99],
}


def needed_raw_columns() -> list[str]:
    """Union of every raw column the pipeline keeps in the slim interim Parquet.

    Includes the target, the legitimate/timing/structural features, and the leakage variables (the
    leakage vars are retained ONLY so the Phase 2 leakage-demonstration model and the FR6.1 detector
    can use them; they never enter the primary feature matrix). Extra tobacco recodes are kept.
    """
    extra = {OBSTETRIC_GESTATION, "cig_1", "cig_2", "cig_3"}
    cols = (
        set(COLUMN_MAP)
        | LEAKAGE_DETECTOR_VARS
        | STRUCTURAL_EXCLUSION_VARS
        | TIMING_CONDITIONAL_VARS
        | IATROGENIC_ENTANGLED_VARS   # retained for post-hoc disparity analysis, not as features
        | extra
    )
    return sorted(cols)


def harmonized_name(raw: str) -> str:
    """Return the analysis name for a raw column, or the raw name if unmapped."""
    return COLUMN_MAP.get(raw, raw)


def all_excluded_vars() -> frozenset[str]:
    """Every variable that must never enter the primary feature matrix."""
    return LEAKAGE_DETECTOR_VARS | STRUCTURAL_EXCLUSION_VARS
