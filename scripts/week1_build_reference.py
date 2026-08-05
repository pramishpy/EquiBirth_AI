"""Compile the project code-reference notebook: the tested foundation the weekly notebooks build on.

The weekly deliverable notebooks import their load-bearing logic from ``src/equibirth/`` so the test
suite guards it against drift, and inline the rest in per-notebook SETUP blocks. A reader of any
single weekly notebook cannot see that imported logic. This reference surfaces and explains every
piece of the code in one place.

The mechanism that keeps this notebook honest is ``inspect.getsource``: every code listing prints the
LIVE source of the ``equibirth`` package, which the SETUP cell imports directly from the working-tree
``src/`` (added to ``sys.path`` in setup), so the shown code is exactly what the tests run against and
cannot drift from it.

Run (PYTHONPATH is only needed to run this generator):
  PYTHONPATH=src ./.venv/bin/python scripts/week1_build_reference.py
Then execute (no PYTHONPATH needed; the notebook SETUP does its own sys.path.insert):
  ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=600 notebooks/week1_00_project_code_reference.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week1_00_project_code_reference"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def anchor(a: str, title: str) -> str:
    return f'<a id="{a}"></a>\n## {title}'


def contents(items: list[tuple[str, str]]) -> str:
    lines = ["### Contents", ""]
    lines += [f"{i}. [{label}](#{a})" for i, (a, label) in enumerate(items, 1)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- setup
SETUP = '''\
import sys, inspect, warnings
from pathlib import Path
sys.path.insert(0, str(Path("..") / "src"))
import numpy as np, polars as pl
warnings.filterwarnings("ignore")

from equibirth.ingest import schema
from equibirth.features import build as fbuild, leakage as fleak, splits as fsplits, full_dataset as ffull
from equibirth.models import evaluate as mev, baselines as mbase, boosting as mboost
from equibirth.models import deep as mdeep, ensemble as mens
from equibirth.eval import robustness as erob
from equibirth.fairness import mitigation as fmit, decomposition as fdecomp
from equibirth.explain import shap_analysis as mshap

def show(obj):
    """Print the LIVE source of a function/class so this notebook cannot drift from the tested code."""
    print(inspect.getsource(obj))

INTERIM = Path("..") / "data" / "interim"
def load_slice(n=200_000, year=2024):
    """A small single-year slice for fast, memory-safe live examples."""
    return pl.scan_parquet(INTERIM / f"natality{year}_clean.parquet").head(n).collect()
'''


# ----------------------------------------------------------------------------- pasted notebook helpers
# These live inside the weekly notebooks' SETUP blocks and are NOT importable from the package. They
# are pasted here verbatim from the existing generators, purely for DISPLAY, and documented in one
# place. To keep this reference safe to execute (no heavy calls, no plotting side effects), each block
# is stored as a triple-quoted string and printed rather than executed.

# From scripts/week2_build_notebook.py (SETUP_PALETTE and SETUP_LOAD blocks): the Okabe-Ito palette,
# the fixed race ordering, the matplotlib house style, and the full-data year loader.
HELPERS_STYLE = r'''# The same Okabe-Ito palette and house style shared across every weekly notebook.
PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
           "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999",
           "ink": "#1a1a2e"}
RACE_ORDER = ["NH White", "NH Asian", "Hispanic", "NH Multiracial", "NH NHOPI", "NH AIAN", "NH Black"]
RACE_COLOR = {r: PALETTE["grey"] for r in RACE_ORDER}
RACE_COLOR.update({"NH White": PALETTE["blue"], "NH Black": PALETTE["red"],
                   "Hispanic": PALETTE["green"], "NH Asian": PALETTE["orange"]})

plt.rcParams.update({"figure.figsize": (8.6, 4.8), "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 11,
                     "axes.titlepad": 10, "font.family": "DejaVu Sans"})

def style(ax, title, xlabel, ylabel):
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return ax
'''

HELPERS_LOAD = r'''# The full-data year loader (Weeks 2+). Reads the complete per-year cleaned files lazily and pools
# them, tagging each row with its birth_year. Do NOT call this on all years in this reference: it
# materializes the ~24.9M-row, ~3GB full frame.
INTERIM = Path("..") / "data" / "interim"
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

def load_years(years):
    frames = [pl.scan_parquet(INTERIM / f"natality{y}_clean.parquet").with_columns(pl.lit(y).alias("birth_year"))
              for y in years]
    return pl.concat([f.collect() for f in frames], how="diagonal_relaxed")
'''

# From scripts/week1_build_notebooks.py (SETUP string): the pandas EDA workhorses. rate_by is the
# grouping workhorse of the whole Week 1 EDA; risk_ratio_ci computes a risk ratio with a log-normal CI.
HELPERS_RATE = r'''def rate_by(frame, col, dropna_target=True):
    """Preterm rate (%) and count for any grouping column. The workhorse of the whole EDA."""
    d = frame.dropna(subset=["preterm"]) if dropna_target else frame
    g = d.groupby(col)["preterm"].agg(rate="mean", n="size")
    g["rate"] = g["rate"] * 100
    return g

def risk_ratio_ci(frame, flag_col, target="preterm"):
    """Risk ratio (flag=1 vs flag=0) with a 95% log-normal CI. Returns (rr, lo, hi, r1, r0)."""
    d = frame.dropna(subset=[target, flag_col])
    g = d.groupby(flag_col)[target].agg(["mean", "sum", "size"])
    r1, r0 = g.loc[1.0, "mean"], g.loc[0.0, "mean"]
    a, n1 = g.loc[1.0, "sum"], g.loc[1.0, "size"]
    c, n0 = g.loc[0.0, "sum"], g.loc[0.0, "size"]
    rr = r1 / r0
    se = np.sqrt(1/a - 1/n1 + 1/c - 1/n0)
    return rr, rr*np.exp(-1.96*se), rr*np.exp(1.96*se), r1*100, r0*100
'''


def build() -> list:
    toc = [
        ("read", "How to read this notebook"),
        ("schema", "The schema and leakage classification"),
        ("build", "The leakage-safe feature builder"),
        ("leakage", "The leakage guards"),
        ("splits", "Deterministic splits"),
        ("full", "Full-population assembly"),
        ("evaluate", "Calibration-first evaluation"),
        ("baselines", "Baseline models"),
        ("boosting", "Gradient boosting"),
        ("deep", "Deep tabular benchmarks"),
        ("ensemble", "Stacked ensemble"),
        ("robustness", "Out-of-time robustness"),
        ("shap", "SHAP explainability"),
        ("decomposition", "Black-White gap decomposition"),
        ("mitigation", "Fairness mitigation"),
        ("helpers", "The inlined notebook helpers"),
        ("map", "How it all fits, and the test map"),
    ]
    cells = [
        md("# Project Code Reference: the tested foundation the weekly notebooks build on\n\n"
           "The weekly notebooks import this logic from `src/equibirth/` so the test suite guards it "
           "against drift, and inline the rest in per-notebook SETUP blocks. A reader of any single "
           "weekly notebook sees the story but not the machinery. This notebook shows and explains every "
           "piece in one place.\n\n"
           "The code below covers the full pipeline: the raw-file schema and leakage classification, "
           "the leakage-safe feature builder and its guards, the deterministic splits, the full-population "
           "assembly, the calibration-first evaluation metrics, the baseline and gradient-boosting model "
           "factories, the deep tabular benchmarks (TabNet and FT-Transformer), the stacked ensemble, "
           "the out-of-time robustness helpers, the SHAP explainability layer, the Black-White gap "
           "decomposition, and the fairness-mitigation wrappers."),
        md(contents(toc)),
        # ---- read ----
        md(anchor("read", "How to read this notebook") + "\n\n"
           "The project follows one principle above all: **import, do not re-implement**. The weekly "
           "notebooks never rewrite the cleaning, feature engineering, leakage gates, splitting, or "
           "scoring logic in a cell. They call the same code the test suite covers, so a notebook cannot "
           "quietly diverge from the tested pipeline. That is what makes the honesty checkable by "
           "someone who does not trust me.\n\n"
           "This reference doubles down on that principle. Every code listing you see below prints the "
           "**live source** of the `equibirth` package via `inspect.getsource`. The setup cell imports "
           "the package directly from the working-tree `src/` (added to `sys.path` in setup), so what "
           "prints is exactly the code the tests run against. If a function changes, this notebook "
           "changes with it the next time it is executed: there is no copy to fall out of date.\n\n"
           "A few reading notes. Where a piece of logic is a module-level constant (a set of column "
           "names, a dictionary) rather than a function or class, `inspect.getsource` cannot fetch it, so "
           "those are shown by printing the constant's value directly or by printing the whole module. A "
           "handful of listings are display-only: the model factories and the full-population loaders "
           "are never *run* here because they would train on or materialize far too much data for a "
           "reference. Those are called out explicitly. And the final section is the inlined notebook "
           "SETUP helpers, which are not importable and so are pasted (as read-only text) from the "
           "generators rather than shown via `inspect`.\n\n"
           "First, the setup cell. It imports the fourteen modules, defines `show(obj)` (the thin "
           "wrapper around `inspect.getsource`), and defines `load_slice`, which reads a small "
           "single-year slice of the cleaned data so the live examples run fast and stay memory-safe."),
        code(SETUP),
        code(
            "print('Modules surfaced in this reference:')\n"
            "for m in [schema, fbuild, fleak, fsplits, ffull, mev, mbase, mboost, mdeep, mens, erob, mshap, fdecomp, fmit]:\n"
            "    print(f'  {m.__name__:<40} {m.__doc__.strip().splitlines()[0]}')"),
        # ---- schema ----
        md(anchor("schema", "The schema and leakage classification") + "\n\n"
           "`equibirth.ingest.schema` is the enforceable core of the project's integrity. It expresses "
           "the leakage classification as **sets of raw column names** rather than prose, precisely so "
           "that the tests and the automated detector can enumerate columns deterministically instead of "
           "matching fuzzy concept descriptions. Here is the whole module."),
        code("print(inspect.getsource(schema))"),
        md("The classification splits the birth certificate by one property: **timing**. A field is a "
           "fair predictor only if it is known before delivery. The key structures are:\n\n"
           "- **`COLUMN_MAP`**: the raw-name to stable-analysis-name mapping, populated only for "
           "legitimate pre-delivery and (gated) timing-conditional variables. Leakage variables are "
           "deliberately left out of it so they cannot drift into the feature matrix by accident.\n"
           "- **`LEAKAGE_DETECTOR_VARS`**: fields recorded at or after delivery, or given *because* "
           "preterm was anticipated (birthweight, Apgar, NICU, antenatal steroids, delivery route, total "
           "prenatal visits, and more). These feed the automated leakage detector: a classifier trained "
           "on these alone must not beat the gate.\n"
           "- **`STRUCTURAL_EXCLUSION_VARS`**: barred by logic rather than by post-delivery leakage. The "
           "LMP-based gestation (`combgest`, which would encode the target), plurality (the analysis is "
           "singleton-only), and the gestation recodes/flags. These are excluded but kept OUT of the "
           "detector because their association is biology or target-encoding, not leakage, and including "
           "them causes false-positive gate failures.\n"
           "- **`IATROGENIC_ENTANGLED_VARS`**: pregnancy conditions that frequently *cause* a preterm "
           "delivery via induction (gestational diabetes, gestational hypertension, eclampsia). Their "
           "univariate signal is small, but the mechanism is reverse-causal, so they are excluded from "
           "the primary model and kept only for post-hoc disparity analysis on the raw frame.\n"
           "- **`TIMING_CONDITIONAL_VARS`**: kept in the primary model only with an associational caveat "
           "and a univariate gate (month prenatal care began, WIC enrollment). Weakly predictive alone, "
           "and not reverse-causal at delivery.\n"
           "- **`MEDICAL_RISK_COMPOSITE_COMPONENTS`**: the exact enumerated members of the medical-risk "
           "count feature (pre-pregnancy diabetes, pre-pregnancy hypertension, prior preterm, prior "
           "cesarean). Gestational diabetes is deliberately absent because it is a timing leak.\n"
           "- **`SENTINEL_MAP`**: the per-variable 'unknown / not stated' codes, never a single global "
           "value, so a legitimate 9 prenatal visits or parity 9 survives de-sentinelling.\n"
           "- **`AGE_BRACKET_BREAKS`**: the single source of truth for the maternal-age brackets (the "
           "age relationship with preterm is U-shaped), shared so `build.py` and the EDA cannot drift."),
        code(
            "print('LEAKAGE_DETECTOR_VARS:', len(schema.LEAKAGE_DETECTOR_VARS), 'columns')\n"
            "print(sorted(schema.LEAKAGE_DETECTOR_VARS))\n"
            "print('\\nTotal variables that must never enter the primary matrix '\n"
            "      '(all_excluded_vars):', len(schema.all_excluded_vars()))"),
        md("The three helper functions in the module are worth showing on their own: "
           "`needed_raw_columns` (the union of every raw column the pipeline retains in the slim interim "
           "Parquet), `harmonized_name` (raw name to analysis name, or the raw name if unmapped), and "
           "`all_excluded_vars` (leakage-detector plus structural exclusions: everything barred from the "
           "primary matrix)."),
        code("show(schema.needed_raw_columns)"),
        code("show(schema.harmonized_name)"),
        code("show(schema.all_excluded_vars)"),
        md("Why is the classification code rather than prose? Because a set of column names is "
           "enumerable and testable. A test can assert that `dbwt` is in `LEAKAGE_DETECTOR_VARS`, that "
           "no engineered feature draws from `all_excluded_vars()`, and that the detector trained on the "
           "barred set fails the build if it discriminates. A paragraph describing 'fields known only at "
           "delivery' cannot be checked automatically; a `frozenset` can."),
        # ---- build ----
        md(anchor("build", "The leakage-safe feature builder") + "\n\n"
           "`equibirth.features.build.build_modeling_frame` turns a cleaned raw frame into the primary "
           "pre-delivery feature matrix. It keeps only legitimate pre-conception / early-pregnancy "
           "variables plus the gated timing-conditional fields, and engineers a few features on top: the "
           "medical-risk count, the U-shaped age bracket, one-hot encodings for the nominal categoricals "
           "(race, payment), and a race-by-education interaction. Crucially, **every engineered feature "
           "declares the raw columns it is built from**, which is what lets the guard prove nothing leaky "
           "was laundered into a composite. The encoding is deliberately simple and transparent: ordinal "
           "where genuinely ordered (education), one-hot for nominal categoricals, Y/N flags mapped to "
           "1/0, numerics kept as-is."),
        code("show(fbuild.build_modeling_frame)"),
        md("The little `_yn_to_int` helper is worth its own look: it maps a Y/N string flag to 1/0 and "
           "leaves anything else (including the 'U' unknown code, and nulls) as null rather than "
           "silently coercing it to 0. That care matters, because a silent coercion to 0 would turn "
           "'unknown' into 'no' and bias every flag toward the negative."),
        code("show(fbuild._yn_to_int)"),
        md("A live example on a small single-year slice. `build_modeling_frame` returns the feature "
           "matrix `X`, the target `y`, and the component map (which raw columns each feature is built "
           "from). Note the medical-risk composite is an explicit, enumerated tuple, not an open-ended "
           "sum, so no leaky field can hide in it."),
        code(
            "df = load_slice()\n"
            "X, y, comps = fbuild.build_modeling_frame(df)\n"
            "print('feature matrix shape:', X.shape)\n"
            "print('first 12 features:', X.columns[:12])\n"
            "print('medical_risk_count is built from exactly:', comps['medical_risk_count'])"),
        # ---- leakage ----
        md(anchor("leakage", "The leakage guards") + "\n\n"
           "`equibirth.features.leakage` holds the automated guards that keep the honesty claim "
           "checkable. `excluded_vars_detector_auc` trains a trivial classifier on the barred "
           "post-delivery variables alone: if those fields can predict the target well on their own "
           "(AUC above the gate), the exclusion list has a hole or a leaky column slipped in, and the "
           "build should fail. It returns a clean 0.5 (no detectable leak) when none of the barred "
           "columns are present, which is what happens when it is called on an already-clean feature "
           "matrix rather than the raw frame."),
        code("show(fleak.excluded_vars_detector_auc)"),
        md("`univariate_auc` measures single-feature discrimination. It is used to gate the "
           "timing-conditional features we do keep: each must be weak on its own or it is dropped."),
        code("show(fleak.univariate_auc)"),
        md("`assert_no_component_is_excluded` is the composite guard. Because every engineered feature "
           "declares its source columns, this check can prove that none of those sources is a barred "
           "variable. `build_modeling_frame` calls it, so a leaky composite cannot be built at all."),
        code("show(fleak.assert_no_component_is_excluded)"),
        md("Live: on the raw single-year slice, the detector should register the leak (the barred "
           "fields are present and highly predictive), and a single barred field like birthweight "
           "(`dbwt`) should score very high on its own. This is the trap the whole leakage discipline "
           "exists to prevent."),
        code(
            "slc = load_slice()\n"
            "print('excluded-vars detector AUC on the raw slice:', round(fleak.excluded_vars_detector_auc(slc), 3))\n"
            "print('univariate AUC of birthweight (dbwt) alone :', round(fleak.univariate_auc(slc, 'dbwt'), 3))"),
        # ---- splits ----
        md(anchor("splits", "Deterministic splits") + "\n\n"
           "`equibirth.features.splits.assign_splits` produces the deterministic, stratified "
           "train/validation/test/held-out partition. The naming is deliberate and avoids a common "
           "trap: instead of a single ambiguous 'validation' set (is that for tuning or for final "
           "reporting?), the partitions are `held_out_final` (untouched until the very end; all headline "
           "metrics and bootstrap CIs use it), `dev_train`, `dev_val` (scores Optuna trials and early "
           "stopping), and `dev_test` (calibration and a quick in-development smoke check). Splits are "
           "stratified by a composite key (outcome by race/ethnicity by year) so every subgroup is "
           "represented in each partition, and are reproducible from the seed."),
        code("show(fsplits.assign_splits)"),
        md("`stratified_subsample` draws a tractable working sample from the complete ~25M-row "
           "population for model development (the full population is too large to hold as a dense "
           "matrix), while the full-population outcome audit still runs on every row. Selection is "
           "proportional within each stratum with largest-remainder rounding, so the sample's marginal "
           "distributions match the population and the total lands exactly on the requested size."),
        code("show(fsplits.stratified_subsample)"),
        md("Live: assign splits on the slice and look at the partition sizes, then the preterm rate "
           "within each partition. Because the split is stratified on the outcome, the rate should be "
           "close to constant across partitions, which is the point: no partition is accidentally "
           "easier or harder than another."),
        code(
            "d = load_slice()\n"
            "s = fsplits.assign_splits(d)\n"
            "print('partition sizes:')\n"
            "print(s.value_counts())\n"
            "labelled = d.select(pl.col('preterm')).with_columns(s.alias('split'))\n"
            "rates = (labelled.filter(pl.col('preterm').is_not_null())\n"
            "         .group_by('split').agg(preterm_rate=pl.col('preterm').mean(), n=pl.len())\n"
            "         .sort('split'))\n"
            "print('\\npreterm rate by partition (should be nearly constant):')\n"
            "print(rates)"),
        # ---- full ----
        md(anchor("full", "Full-population assembly") + "\n\n"
           "`equibirth.features.full_dataset` composes the tested ingest and feature layers for the "
           "complete 2018-2024 rebuild. It re-implements nothing: it reads the per-year cleaned Parquet "
           "files and splits the work exactly as the SPEC prescribes into two separate jobs. "
           "`load_full_frame` pools the per-year cleaned files into one frame tagged with `birth_year`."),
        code("show(ffull.load_full_frame)"),
        md("`outcome_audit` runs on EVERY row of the pooled population. Because it is the full census, "
           "even the smallest race groups are reliable here, so this is the equity story: overall rate, "
           "per-year rate, per-race rate and count, and the NH Black minus NH White gap in percentage "
           "points. It is strictly descriptive and associational: race is a marker of exposure to racism "
           "and structural barriers, not a biological cause."),
        code("show(ffull.outcome_audit)"),
        md("`assemble_modeling` is the other half: it draws the deterministic stratified working sample, "
           "builds the leakage-safe matrix via the tested `build_modeling_frame`, and assigns the same "
           "deterministic stratified splits, returning the bundle the models consume. This separation, "
           "the full-census **outcome audit** kept apart from the sampled **modeling** job, is the key "
           "design point: how unequal the world is (measured on everyone) is a different question from "
           "how well we can predict (measured on a tractable sample with an untouched held-out fold).\n\n"
           "These functions are shown for reference only and are **not run here**: `load_full_frame` and "
           "`assemble_modeling` would read and materialize the full ~24.9M-row population, which belongs "
           "in a script, not a reference cell."),
        code("show(ffull.assemble_modeling)"),
        # ---- evaluate ----
        md(anchor("evaluate", "Calibration-first evaluation") + "\n\n"
           "`equibirth.models.evaluate` is where the calibration-first discipline lives. Because the "
           "outcome is rare (~9%) and the whole fairness argument rests on calibration-by-group being "
           "trustworthy, the project leads with rate-aware metrics and calibration, never accuracy. Here "
           "are the metric functions, each with a note on why it earns its place.\n\n"
           "`auc_roc` is ROC AUC. It is base-rate invariant (its floor is 0.5 regardless of prevalence), "
           "which is exactly why it is the fair thing to compare ACROSS race groups whose prevalence "
           "differs."),
        code("show(mev.auc_roc)"),
        md("`auc_pr` is average precision (area under the precision-recall curve). Its floor equals the "
           "prevalence, so under this much imbalance a useless model cannot hide behind a high number: "
           "that makes it the honest primary metric."),
        code("show(mev.auc_pr)"),
        md("`brier` is the Brier score, a proper scoring rule (the mean squared error of the predicted "
           "probability). Lower is better, and it rewards being both discriminating and calibrated."),
        code("show(mev.brier)"),
        md("`ece` is the expected calibration error over equal-width probability bins: the "
           "count-weighted mean gap between predicted and observed rate. It is the headline calibration "
           "number, the one that answers 'when the model says 10%, is the real rate about 10%?'."),
        code("show(mev.ece)"),
        md("`ece_quantile` is the same idea over equal-MASS bins. It is the stricter summary when "
           "predictions pile up near the base rate, because equal-width bins can leave the crowded "
           "low-risk region under-resolved."),
        code("show(mev.ece_quantile)"),
        md("`mce` is the maximum calibration error, with a minimum-bin-size guard so a single sparse "
           "high-probability bin (a handful of rows) cannot inflate it into a binning artifact."),
        code("show(mev.mce)"),
        md("`calibration_curve` returns the per-bin mean-predicted, observed-rate, and count arrays "
           "used to draw a reliability diagram."),
        code("show(mev.calibration_curve)"),
        md("`fraction_near_base_rate` is the honest qualification in code: the share of predictions "
           "within a small window of the base rate. A high value means the model rarely moves far from "
           "the prevalence, so a low ECE reflects aggregate calibration rather than individual risk "
           "stratification. Both things can be true at once, and this function makes the distinction "
           "measurable."),
        code("show(mev.fraction_near_base_rate)"),
        md("`bootstrap_ci` gives a stratified bootstrap confidence interval for any metric. Resampling "
           "is stratified by outcome class so each resample keeps the ~9% prevalence, and degenerate "
           "single-class resamples are skipped."),
        code("show(mev.bootstrap_ci)"),
        md("`stratified_report` is the per-group reporting with tiered suppression: groups with too few "
           "positives are suppressed, mid-sized groups are reported but flagged wide-CI, and only "
           "well-populated groups are reported as reliable. Small race groups are flagged, never "
           "silently dropped."),
        code("show(mev.stratified_report)"),
        md("A tiny live example so the metrics are concrete. On a toy label/probability pair, "
           "`auc_roc`, `auc_pr`, and `ece` return real numbers."),
        code(
            "y = np.array([0, 0, 1, 0, 1, 1])\n"
            "p = np.array([.1, .2, .7, .3, .6, .9])\n"
            "print('auc_roc:', round(mev.auc_roc(y, p), 3))\n"
            "print('auc_pr :', round(mev.auc_pr(y, p), 3))\n"
            "print('ece    :', round(mev.ece(y, p), 3))"),
        # ---- baselines ----
        md(anchor("baselines", "Baseline models") + "\n\n"
           "`equibirth.models.baselines` builds the three Week 3 baselines: logistic regression (the "
           "calibration reference and honest floor), a decision tree, and a random forest. `"
           "_make_estimators` constructs the three pipelines, each with a median imputer (and a scaler "
           "for logistic regression) so callers can pass a raw matrix with missing values."),
        code("show(mbase._make_estimators)"),
        md("`fit_predict_raw_and_calibrated` fits each baseline once and returns both the raw and the "
           "isotonic-calibrated held-out probabilities. Fitting once (rather than twice) halves the "
           "memory and time, which matters when the forest trains on roughly a million rows.\n\n"
           "Two design choices are worth stating. First, **class weights, not SMOTE**: the imbalance is "
           "handled with balanced class weights rather than synthetic resampling, because resampling "
           "distorts the base rate and therefore corrupts the very calibration the fairness audit "
           "depends on. Second, **isotonic recalibration on a SEPARATE fold**: the calibrator is fit on "
           "a held-aside fold the base model never trained on, so the calibrated probabilities are "
           "honest out-of-sample and calibration-by-group stays trustworthy.\n\n"
           "These are shown for reference only and are **not run here**: fitting the baselines is a "
           "training job, not a reference cell."),
        code("show(mbase.fit_predict_raw_and_calibrated)"),
        # ---- boosting ----
        md(anchor("boosting", "Gradient boosting") + "\n\n"
           "`equibirth.models.boosting` holds the Week 4 gradient-boosting logic: XGBoost and LightGBM, "
           "the expected top performers on this tabular data. Both handle missing values natively (no "
           "imputer) and use `scale_pos_weight` for the imbalance rather than resampling, again to "
           "protect calibration. `_make_estimator` constructs an unfitted booster of the requested "
           "family with sensible fixed defaults."),
        code("show(mboost._make_estimator)"),
        md("`scale_pos_weight` is the standard imbalance weight for boosters: the negative/positive "
           "ratio (1.0 when there are no positives)."),
        code("show(mboost.scale_pos_weight)"),
        md("`tune` runs an Optuna study that maximizes AUC-PR (the primary metric) on the dev "
           "validation fold only, so the held-out set is never touched during tuning. A trial cap and a "
           "wall-clock timeout keep it safe in CI and on a laptop."),
        code("show(mboost.tune)"),
        md("`fit_calibrated` fits a booster on the training fold and isotonic-recalibrates it on a "
           "separate fold, returning the fitted calibrated estimator (the fairness script needs it to "
           "score several partitions).\n\n"
           "These are shown for reference only and are **not run here**: tuning and fitting boosters is "
           "a training job."),
        code("show(mboost.fit_calibrated)"),
        # ---- deep ----
        md(anchor("deep", "Deep tabular benchmarks") + "\n\n"
           "`equibirth.models.deep` provides TabNet (pytorch-tabnet) and FT-Transformer (rtdl) as "
           "honest benchmarks behind the same contract as the boosters (D41). The signal in this "
           "administrative data is too weak for neural nets to exploit (max |corr| ~0.13, prevalence "
           "~8.6%), so trees are expected to win and did win (F20). The real deliverable is TabNet's "
           "attention-based feature masks, which give an independent importance ranking without needing "
           "TreeSHAP.\n\n"
           "Unlike boosters, neural nets cannot handle NaN, so `_prep` median-imputes (fit on train) "
           "and standardizes. The rest follows the same discipline: fit on train, early-stop on val, "
           "isotonic-recalibrate on val, report on held-out."),
        code("show(mdeep.fit)"),
        md("`predict` scores a partition using the fitted estimator and the already-imputed matrix."),
        code("show(mdeep.predict)"),
        md("`fit_predict` composes fitting, calibration, and scoring in one call, matching the "
           "contract of the boosting module."),
        code("show(mdeep.fit_predict)"),
        md("`tabnet_feature_masks` extracts the global feature importance from TabNet's built-in "
           "explain mechanism. The resulting ranking is associational (same caveats as SHAP) and is "
           "checked against the leakage taxonomy before reporting."),
        code("show(mdeep.tabnet_feature_masks)"),
        # ---- ensemble ----
        md(anchor("ensemble", "Stacked ensemble") + "\n\n"
           "`equibirth.models.ensemble` implements the cross-fitted logistic meta-learner (roadmap "
           "item 2). It stacks base-model probabilities in logit space and uses stratified K-fold "
           "cross-fitting to prevent meta-leakage. The stack is deliberately simple: with base models "
           "this correlated and signal this weak, anything fancier overfits.\n\n"
           "`oof_stack` fits the cross-fitted stack and returns the meta-model plus out-of-fold "
           "probabilities."),
        code("show(mens.oof_stack)"),
        md("`predict_stack` scores a new partition using the fitted meta-model and the base "
           "probabilities from that partition."),
        code("show(mens.predict_stack)"),
        # ---- robustness ----
        md(anchor("robustness", "Out-of-time robustness") + "\n\n"
           "`equibirth.eval.robustness` answers the deployment question a random split cannot: does "
           "performance and calibration survive a regime change? Era boundaries are fixed by the "
           "pandemic timeline (pre-COVID 2018-2019, COVID 2020-2021, recent 2022-2024), not tuned. "
           "A model trained on pre-COVID births is tested on COVID-era and recent-era births.\n\n"
           "`out_of_time_masks` returns disjoint boolean masks keyed by era."),
        code("show(erob.out_of_time_masks)"),
        md("`yearly_drift` computes per-year n, prevalence, AUC-ROC, AUC-PR, and ECE so drift can "
           "be read as a time series rather than a single pre/post comparison."),
        code("show(erob.yearly_drift)"),
        # ---- shap ----
        md(anchor("shap", "SHAP explainability") + "\n\n"
           "`equibirth.explain.shap_analysis` is the explainability layer Week 7 leans on (SPEC FR10). "
           "It turns the fitted tree model into a global feature-importance ranking and, more "
           "importantly, race-stratified rankings: if the top predictors differ across groups (say prior "
           "preterm birth dominating for one group while access factors dominate for another), that "
           "difference is itself a finding about where risk concentrates. Every reading here is "
           "associational, never causal.\n\n"
           "`shap_values_for` returns the `(n_samples, n_features)` SHAP matrix for a fitted tree model. "
           "It normalizes across SHAP API versions: the older per-class list and the newer 3-D "
           "`(samples, features, classes)` array both collapse to the positive-class contributions."),
        code("show(mshap.shap_values_for)"),
        md("`mean_abs_importance` ranks features by mean absolute SHAP value (descending). Mean |SHAP| "
           "is the standard global-importance summary: it measures how much each feature moves the "
           "prediction on average, regardless of direction."),
        code("show(mshap.mean_abs_importance)"),
        md("`top_features_by_group` computes the mean-|SHAP| ranking WITHIN each requested race group, "
           "returning the top-N per group. This is the mechanism behind the stratified-importance finding: "
           "empty groups are skipped rather than erroring, so small groups do not break the call."),
        code("show(mshap.top_features_by_group)"),
        md("A tiny live example. On a toy two-feature problem where a random forest learns that the first "
           "feature carries the signal, `mean_abs_importance` should rank it first. This runs fast and "
           "needs no fitted project model."),
        code(
            "from sklearn.ensemble import RandomForestClassifier\n"
            "rng = np.random.default_rng(0)\n"
            "Xt = rng.normal(size=(300, 2))\n"
            "yt = (Xt[:, 0] > 0).astype(int)\n"
            "rf = RandomForestClassifier(n_estimators=40, random_state=0).fit(Xt, yt)\n"
            "sv = mshap.shap_values_for(rf, Xt)\n"
            "print('ranking:', mshap.mean_abs_importance(sv, ['signal_x0', 'noise_x1']))"),
        # ---- decomposition ----
        md(anchor("decomposition", "Black-White gap decomposition") + "\n\n"
           "`equibirth.fairness.decomposition` answers research question 4 (SPEC FR10): how much of the "
           "observed Black-White preterm gap is statistically associated with measured covariates, and how "
           "much is left unexplained? It uses **g-computation (standardization)** over a logistic model: "
           "fit preterm on race plus measured covariates, then compare the counterfactual mean risk if "
           "everyone were assigned the focal race versus the reference race, holding each person's "
           "covariates fixed. The difference is the covariate-adjusted (unexplained) gap; the raw gap "
           "minus that adjusted gap is the share associated with the measured covariates.\n\n"
           "The heavy caveat is stated in the module docstring and must travel with any use: the "
           "covariates (education, payment, WIC) are themselves products of structural racism, so the "
           "explained share is inflated and the unexplained residual is a LOWER bound on structural "
           "contributions. This is association, not proof of mechanism.\n\n"
           "`Decomposition` is the result dataclass: raw gap, adjusted gap, explained absolute and "
           "percentage, and the two group sample sizes."),
        code("show(fdecomp.Decomposition)"),
        md("`decompose_gap` does the work. It takes the measured covariates (explicitly NOT including "
           "race), the target, and the two boolean group masks, median-imputes and standardizes the "
           "covariates, fits the logistic model with race plus covariates, and standardizes over the "
           "COMBINED covariate distribution so the adjusted gap is not confounded by differing covariate "
           "distributions between the two groups."),
        code("show(fdecomp.decompose_gap)"),
        md("A live example. Construct a case where a single covariate fully drives the gap (the focal "
           "group has a higher covariate value, and the covariate raises risk identically for both "
           "groups). The decomposition should attribute nearly all of the raw gap to the covariate, "
           "leaving an adjusted (unexplained) gap near zero."),
        code(
            "rng = np.random.default_rng(1)\n"
            "n = 4000\n"
            "focal = np.array([True] * (n // 2) + [False] * (n // 2))\n"
            "cov = np.where(focal, rng.normal(1.0, 1, n), rng.normal(-1.0, 1, n))\n"
            "risk = 1 / (1 + np.exp(-(cov)))\n"
            "y = (rng.uniform(size=n) < risk).astype(int)\n"
            "res = fdecomp.decompose_gap(cov.reshape(-1, 1), y, focal, ~focal)\n"
            "print(f'raw gap      : {res.raw_gap:.4f}')\n"
            "print(f'adjusted gap : {res.adjusted_gap:.4f}')\n"
            "print(f'explained pct: {res.explained_pct:.1%}')"),
        # ---- mitigation ----
        md(anchor("mitigation", "Fairness mitigation") + "\n\n"
           "`equibirth.fairness.mitigation` holds the bias-mitigation experiments (SPEC FR9.2). The "
           "module reports the accuracy-fairness tradeoff of each approach; nothing here decides which "
           "mitigation to adopt (that is a value judgment, examined in F22/D43).\n\n"
           "`expgrad_predictions` wraps Fairlearn's ExponentiatedGradient reductions. It fits a "
           "logistic base learner under a TPR-parity or equalized-odds constraint and returns both "
           "baseline and mitigated hard predictions, so callers can price the fairness/accuracy "
           "tradeoff directly."),
        code("show(fmit.expgrad_predictions)"),
        md("`threshold_optimizer_predictions` wraps Fairlearn's ThresholdOptimizer: it learns "
           "group-specific thresholds on precomputed scores to equalize TPR across groups. This IS "
           "disparate treatment (different decision rules by race) and is reported analytically, "
           "never deployed."),
        code("show(fmit.threshold_optimizer_predictions)"),
        # ---- helpers ----
        md(anchor("helpers", "The inlined notebook helpers") + "\n\n"
           "Not all of the weekly-notebook logic lives in the package. Some small helpers are defined "
           "inline in each weekly notebook's SETUP block: the shared plotting style, the full-data "
           "loader, and two pandas EDA workhorses. Because they live in the generators rather than in "
           "`src/equibirth/`, they are not importable and `inspect.getsource` cannot reach them, so the "
           "cells below are **pasted copies of those SETUP helpers**, documented here in one place. To "
           "keep this reference safe to execute, each block is stored as a string and printed rather "
           "than run (in particular `load_years` is never called on all years, which would materialize "
           "the full ~24.9M-row frame).\n\n"
           "First, the shared house style, from the Week 2 generator's SETUP: the Okabe-Ito `PALETTE`, "
           "the fixed `RACE_ORDER`, the matplotlib `rcParams` block, and the `style(ax, ...)` axis "
           "helper. The same palette and style run through every weekly notebook so the whole project "
           "reads as one piece."),
        code("print(" + repr(HELPERS_STYLE) + ")"),
        md("Next, the full-data year loader `load_years`, also from the Week 2 generator's SETUP. It "
           "scans the complete per-year cleaned files lazily and pools them, tagging each row with its "
           "`birth_year`. Weeks 2 and later use it for the full-population computations."),
        code("print(" + repr(HELPERS_LOAD) + ")"),
        md("Finally, the two pandas EDA workhorses from the Week 1 generator's SETUP. `rate_by` is the "
           "grouping workhorse of the whole Week 1 exploration: preterm rate (%) and count for any "
           "grouping column. `risk_ratio_ci` computes the risk ratio of a binary flag (flag=1 vs "
           "flag=0) with a 95% log-normal confidence interval.\n\n"
           "The CI derivation is worth spelling out, because it is the standard log-risk-ratio "
           "approach. With `a` preterm births among the `n1` flagged mothers and `c` among the `n0` "
           "unflagged, the two risks are `r1 = a/n1` and `r0 = c/n0`, and the risk ratio is `RR = "
           "r1/r0`. The sampling distribution of `RR` is skewed, so the interval is built on the log "
           "scale where it is approximately normal. The standard error of `log(RR)` is\n\n"
           "`SE(log RR) = sqrt(1/a - 1/n1 + 1/c - 1/n0)`,\n\n"
           "and the 95% interval is `RR * exp(-1.96 * SE)` to `RR * exp(+1.96 * SE)`, which is exactly "
           "what the function returns (alongside the two raw rates as percentages)."),
        code("print(" + repr(HELPERS_RATE) + ")"),
        # ---- map ----
        md(anchor("map", "How it all fits, and the test map") + "\n\n"
           "Put together, the call graph runs from raw certificate to scored prediction to fairness "
           "audit:\n\n"
           "1. **`ingest/schema.py`** defines the leakage classification (which raw columns are "
           "legitimate, which are barred, and why) and the per-variable sentinel codes. The ingest layer "
           "uses it to write the de-sentinelled per-year cleaned Parquet files.\n"
           "2. **`features/build.py`** (`build_modeling_frame`) consumes a cleaned frame and produces the "
           "leakage-safe feature matrix `X`, target `y`, and component map, using `schema` for the "
           "classification and calling the **`features/leakage.py`** guard (`assert_no_component_is_"
           "excluded`) so no barred field can be laundered into a composite.\n"
           "3. **`features/splits.py`** (`assign_splits`, `stratified_subsample`) partitions the rows "
           "deterministically and stratified, and draws the tractable working sample.\n"
           "4. **`features/full_dataset.py`** composes the above: `outcome_audit` on the full census, "
           "and `assemble_modeling` (subsample -> `build_modeling_frame` -> `assign_splits`) into the "
           "modeling bundle.\n"
           "5. **`models/baselines.py`** and **`models/boosting.py`** consume `X`, `y`, and the split "
           "masks to fit calibrated models and return held-out probabilities.\n"
           "6. **`models/deep.py`** benchmarks TabNet and FT-Transformer on the same data.\n"
           "7. **`models/ensemble.py`** stacks the base-model probabilities into a logistic meta-learner.\n"
           "8. **`models/evaluate.py`** scores those probabilities with the calibration-first metrics "
           "and the tiered stratified report.\n"
           "9. **`eval/robustness.py`** tests temporal stability via pre-COVID/post-COVID splits.\n"
           "10. **`explain/shap_analysis.py`** turns the fitted tree model into global and "
           "race-stratified SHAP importance rankings.\n"
           "11. **`fairness/decomposition.py`** g-computes how much of the Black-White gap is associated "
           "with measured covariates versus left unexplained.\n"
           "12. **`fairness/mitigation.py`** runs ExponentiatedGradient and ThresholdOptimizer to "
           "price the fairness/accuracy tradeoff.\n\n"
           "Every one of these modules is guarded by a test file, which is what makes the weekly "
           "notebooks safe to build on:\n\n"
           "| Module | Test file |\n"
           "| --- | --- |\n"
           "| `ingest/schema.py` | `tests/test_ingest_schema.py` |\n"
           "| `features/build.py` | `tests/test_features_build.py` |\n"
           "| `features/leakage.py` | `tests/test_features_leakage.py` |\n"
           "| `features/splits.py` | `tests/test_features_splits.py` |\n"
           "| `features/full_dataset.py` | `tests/test_features_full_dataset.py` |\n"
           "| `models/evaluate.py` | `tests/test_models_evaluate.py` |\n"
           "| `models/baselines.py` | `tests/test_models_baselines.py` |\n"
           "| `models/boosting.py` | `tests/test_models_boosting.py` |\n"
           "| `models/deep.py` | `tests/test_models_deep.py` |\n"
           "| `models/ensemble.py` | `tests/test_models_ensemble.py` |\n"
           "| `eval/robustness.py` | `tests/test_eval_robustness.py` |\n"
           "| `explain/shap_analysis.py` | `tests/test_explain_shap.py` |\n"
           "| `fairness/decomposition.py` | `tests/test_fairness_mitigation.py` |\n"
           "| `fairness/mitigation.py` | `tests/test_fairness_mitigation.py` |\n\n"
           "That is the whole point of importing rather than re-implementing: the code you have just "
           "read is the code the tests cover, and the code the weekly notebooks run. Nothing in this "
           "reference can drift from either." + AUTHOR),
    ]
    return cells


def main() -> None:
    NBDIR.mkdir(exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.cells = build()
    nb.metadata = {"kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    out = NBDIR / f"{SLUG}.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
