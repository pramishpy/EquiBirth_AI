"""Build the primary pre-delivery feature matrix (SPEC FR6).

The primary matrix uses ONLY pre-conception / early-pregnancy variables plus a small set of
timing-conditional features that pass the univariate leakage gate. Every engineered feature declares
its source components so the FR6.2 guard can prove no leakage variable was laundered into a composite.
Post-delivery variables (birthweight, Apgar, labor interventions, total visits, the LMP gestation,
plurality) are never selected here.

Returns a model-ready Polars frame of features, the target series, and the component map. Encoding is
deliberately simple and transparent: ordinal where it is genuinely ordered (education), one-hot for
nominal categoricals (race, payment, marital), Y/N flags to 0/1, numerics kept as-is.
"""

from __future__ import annotations

import polars as pl

from equibirth.features.leakage import assert_no_component_is_excluded
from equibirth.ingest import schema

# Raw Y/N flag columns that are legitimate pre-pregnancy predictors.
_YN_LEGIT = ["rf_pdiab", "rf_phype", "rf_ppterm", "rf_cesar"]
# Timing-conditional Y/N flags kept with an associational caveat. Eclampsia and gestational
# hypertension/diabetes are NOT here: they are iatrogenically outcome-entangled (they cause preterm
# via induction) and are excluded from the primary model, kept only for post-hoc disparity analysis
# on the raw frame (schema.IATROGENIC_ENTANGLED_VARS; DECISION_LOG D26).
_YN_TIMING = ["wic"]


def _yn_to_int(col: str) -> pl.Expr:
    """Map a Y/N string flag to 1/0 (anything else, incl. null, stays null)."""
    return (
        pl.col(col).str.to_uppercase().replace_strict({"Y": 1, "N": 0}, default=None,
                                                       return_dtype=pl.Int8).alias(col)
    )


def build_modeling_frame(
    df: pl.DataFrame, *, include_timing_conditional: bool = True
) -> tuple[pl.DataFrame, pl.Series, dict[str, tuple[str, ...]]]:
    """Return (features, target, component_map) for the primary pre-delivery model.

    ``df`` is a cleaned frame from the ingest layer (raw NBER column names, de-sentinelled, with the
    ``preterm`` target and ``mother_race_ethnicity`` label already present).
    """
    components: dict[str, tuple[str, ...]] = {}
    exprs: list[pl.Expr] = []

    # --- numeric legitimate features ---
    for col in ["mager", "meduc", "bmi", "cig_0", "priorlive", "priordead"]:
        if col in df.columns:
            exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))
            components[col] = (col,)

    # married flag from dmar (1=married, 2=unmarried)
    if "dmar" in df.columns:
        exprs.append((pl.col("dmar") == 1).cast(pl.Int8).alias("married"))
        components["married"] = ("dmar",)

    # Y/N legitimate flags
    yn_present = [c for c in _YN_LEGIT if c in df.columns]
    for col in yn_present:
        exprs.append(_yn_to_int(col))
        components[col] = (col,)

    # --- engineered features (declare components for FR6.2) ---
    # medical-risk composite: enumerated, no leakage member (RF_GDIAB intentionally absent)
    comp_members = [c for c in schema.MEDICAL_RISK_COMPOSITE_COMPONENTS if c in df.columns]
    if comp_members:
        # Note: sum_horizontal treats a null (unrecorded 'U') component as 0, so the count is a
        # lower bound when some flags are missing. Acceptable here because 'U' is rare (<0.2%) and a
        # tree model can use the separate flags; revisit if missingness rises.
        exprs.append(
            pl.sum_horizontal([_yn_to_int(c) for c in comp_members]).alias("medical_risk_count")
        )
        components["medical_risk_count"] = tuple(comp_members)

    # age bracket (U-shaped relationship; keep alongside raw age). Breaks are the single source of
    # truth in schema.AGE_BRACKET_BREAKS so build and eda cannot drift.
    if "mager" in df.columns:
        expr = pl.when(pl.col("mager") < schema.AGE_BRACKET_BREAKS[0]).then(0)
        for i, brk in enumerate(schema.AGE_BRACKET_BREAKS[1:], start=1):
            expr = expr.when(pl.col("mager") < brk).then(i)
        exprs.append(expr.otherwise(len(schema.AGE_BRACKET_BREAKS)).cast(pl.Int8).alias("age_bracket"))
        components["age_bracket"] = ("mager",)

    # --- timing-conditional (optional) ---
    if include_timing_conditional:
        for col in [c for c in _YN_TIMING if c in df.columns]:
            exprs.append(_yn_to_int(col))
            components[col] = (col,)
        # first-trimester care (binary recode of month care began; less entangled than raw count)
        if "precare" in df.columns:
            exprs.append(
                pl.when(pl.col("precare").is_null()).then(None)
                .otherwise((pl.col("precare").is_between(1, 3)).cast(pl.Int8))
                .alias("first_trimester_care")
            )
            components["first_trimester_care"] = ("precare",)

    feats = df.select(exprs)

    # one-hot nominal categoricals
    cat_frames = []
    if "mother_race_ethnicity" in df.columns:
        d = df.select("mother_race_ethnicity").to_dummies(drop_first=False)
        cat_frames.append(d)
        for c in d.columns:
            components[c] = ("mracehisp",)
    if "pay" in df.columns:
        d = df.select(pl.col("pay").cast(pl.Int64).cast(pl.String)).to_dummies()
        cat_frames.append(d)
        for c in d.columns:
            components[c] = ("pay",)
    if cat_frames:
        feats = pl.concat([feats] + cat_frames, how="horizontal")

    # race x education interaction (intersectional; not explained by education alone)
    if "mother_race_ethnicity" in df.columns and "meduc" in df.columns:
        inter = df.select(
            (pl.col("mother_race_ethnicity").fill_null("UNK") + "_edu" +
             pl.col("meduc").cast(pl.String).fill_null("UNK")).alias("race_education")
        ).to_dummies()
        feats = pl.concat([feats, inter], how="horizontal")
        for c in inter.columns:
            components[c] = ("mracehisp", "meduc")

    # FR6.2: no engineered feature may be built from an excluded variable.
    assert_no_component_is_excluded(components)
    # FR6.1-adjacent: no excluded raw column name may appear in the feature matrix.
    excluded = schema.all_excluded_vars()
    leaked = [c for c in feats.columns if c in excluded]
    if leaked:
        raise AssertionError(f"Excluded variables present in primary matrix: {leaked}")

    target = df["preterm"]
    return feats, target, components
