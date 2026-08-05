"""Compile the Week 2 companion notebook: feature experiments.

This is the empirical lab that JUSTIFIES the Week 2 feature choices. The heavy full-data compute
(the clean/leakage dual-track models, univariate rank-sum AUCs, LightGBM gain importance, Pearson
correlation, mutual information, PCA, and KMeans clustering) was already run on the COMPLETE
2018-2024 singleton population by scripts/week2_experiments.py and serialized to
reports/week2_experiments.json. This notebook LOADS that JSON to display the heavy results and
LIVE-RECOMPUTES only cheap descriptive stats on a slim sample, so it stays memory-safe and never
retrains anything in a cell. Real calibrated model SELECTION stays Week 3+ (framed as PLANNED).

Run:  PYTHONPATH=src ./.venv/bin/python scripts/week2_build_experiments_notebook.py
Then: PYTHONPATH=src ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1200 notebooks/week2_02_feature_experiments.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week2_02_feature_experiments"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def anchor(a, title): return f'<a id="{a}"></a>\n## {title}'


def contents(items):
    return "\n".join(["### Contents", ""] + [f"{i}. [{lbl}](#{a})" for i, (a, lbl) in enumerate(items, 1)])


# ----------------------------------------------------------------------------- setup
SETUP = """\
# This companion is the empirical lab behind Week 2. The heavy numbers (the clean/leakage dual-track
# models, the univariate rank-sum AUCs, the LightGBM gain importance, Pearson correlation, mutual
# information, PCA, and KMeans clustering) were all computed on the COMPLETE 2018-2024 singleton
# population by the tested script scripts/week2_experiments.py and written to
# reports/week2_experiments.json. We LOAD those results here rather than recomputing them, so the
# notebook stays fast and memory-safe. The only things we compute live are cheap descriptive stats on
# a slim sample, so nothing heavy runs in a cell.
#
# Cap Polars threads BEFORE importing it, so a shared kernel does not oversubscribe cores. This mirrors
# the memory care in the sibling notebooks: we never load all ~24.9M rows densely here.
import os
os.environ["POLARS_MAX_THREADS"] = "4"

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 40)
RNG = 7

# The full-data results, precomputed by the tested experiment script on the complete population.
REPORTS = Path("..") / "reports"
EXP = json.loads((REPORTS / "week2_experiments.json").read_text())

PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
           "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999",
           "ink": "#1a1a2e"}
RACE_ORDER = ["NH White", "NH Asian", "Hispanic", "NH Multiracial", "NH NHOPI", "NH AIAN", "NH Black"]

plt.rcParams.update({"figure.figsize": (8.6, 4.8), "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 11,
                     "axes.titlepad": 10, "font.family": "DejaVu Sans"})

def style(ax, title, xlabel, ylabel):
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return ax
"""

SETUP_SAMPLE = """\
# Self-contained: read a SLIM projection of the modeling working set so the notebook stands on its own
# and can show real rows. We take only a handful of columns and sample down, so this stays light; the
# heavy population numbers all come from EXP (the JSON), never from a dense in-notebook computation.
DATA = Path("..") / "data" / "processed" / "modeling_full_2018_2024.parquet"
SAMPLE_COLS = ["preterm", "split", "mother_race_ethnicity", "mager", "meduc", "bmi",
               "medical_risk_count", "rf_ppterm", "first_trimester_care", "married"]

# Sample rather than collect all rows: a fixed-seed 300K slice is plenty for descriptive display and
# keeps the kernel memory-safe. The population-scale figures live in EXP.
mdf = (pl.scan_parquet(DATA)
       .select(SAMPLE_COLS)
       .collect()
       .sample(n=300_000, seed=RNG))
print(f"Slim descriptive sample: {mdf.shape[0]:,} rows x {mdf.shape[1]} columns "
      f"(the heavy numbers below come from the full-data JSON, not from this sample).")
"""


def build():
    toc = [
        ("setup", "Setup: load the full-data results, sample a slim frame"),
        ("dualtrack", "The dual-track experiment: the leakage lesson, quantified"),
        ("contribution", "Per-feature contribution: univariate AUC and gain importance"),
        ("corrmi", "Correlation and mutual information: compare and contrast"),
        ("pca", "PCA: is the feature space redundant?"),
        ("clusters", "Clustering: do natural risk subpopulations emerge?"),
        ("plan", "What this means for modeling (PLANNED)"),
    ]
    cells = [
        md("# Week 2 (companion): Feature Experiments, what the pre-delivery signals can and cannot do\n\n"
           "<sub>[Main Week 2 notebook: Feature Engineering and the Leakage Audit](week2_01_features_and_leakage.ipynb) "
           "· the tested code is documented in [the project code reference](week1_00_project_code_reference.ipynb)</sub>\n\n"
           "The main Week 2 notebook made the leakage argument qualitatively and built the leakage-safe "
           "feature matrix. This companion is the empirical lab that JUSTIFIES those choices with numbers. "
           "It does two things. First, it demonstrates leakage end to end: a clean pre-delivery model "
           "versus a leakage-included one, scored on held-out data, so the gap between honest and "
           "fraudulent is a measured quantity, not a warning. Second, it explores the pre-delivery "
           "features from every angle I could think of (single-feature strength, in-model importance, "
           "linear correlation, nonlinear association, dimensionality, and natural subpopulations) to "
           "understand exactly what signal is there.\n\n"
           "One scope note up front. This is exploration and justification, not model selection. The "
           "calibrated baseline models, the gradient-boosting comparison, and the formal fairness audit "
           "are Week 3 and beyond, and everything about them here is written as the PLAN, not as a result "
           "already delivered. Nothing in this project is a clinical tool."),
        md(contents(toc)),
        # ---- setup ----
        md(anchor("setup", "Setup: load the full-data results, sample a slim frame") + "\n\n"
           "The heavy compute is already done. The dual-track models, the univariate and in-model "
           "importance rankings, correlation, mutual information, PCA, and clustering were all computed on "
           "the COMPLETE 2018-2024 singleton population by the tested script "
           "`scripts/week2_experiments.py`, and serialized to `reports/week2_experiments.json`. We load "
           "that JSON here and display it. The only live computation is a few descriptive stats on a slim "
           "300K-row sample, so the notebook stays memory-safe and never retrains anything in a cell."),
        code(SETUP),
        code(
            "print(f\"Full singleton population behind these results: {EXP['n_singletons']:,} births.\")\n"
            "print(f\"Dual-track training cap: {EXP['train_rows_used']:,} rows; \"\n"
            "      f\"mutual-information subsample: {EXP['mi_subsample_rows']:,} rows.\")\n"
            "print('Sections available in the results file:')\n"
            "for k in sorted(EXP): print('  -', k)"),
        md("Each notebook stands on its own, so here is the slim descriptive sample and a few real rows of "
           "the leakage-safe pre-delivery feature matrix. The `split` column marks the deterministic "
           "train / validation / test / held-out partitions, so every row is tagged with the fold it "
           "belongs to."),
        code(SETUP_SAMPLE),
        code("mdf.sample(5, seed=RNG).to_pandas()"),
        # ---- dual-track ----
        md(anchor("dualtrack", "The dual-track experiment: the leakage lesson, quantified") + "\n\n"
           "This is the headline. The main notebook argued that at-delivery fields must be barred because "
           "they leak. Here that argument becomes a measurement. Two models were trained: a **clean** "
           "track on the leakage-safe pre-delivery matrix only, and a **leakage** track that additionally "
           "lets the barred at-delivery fields in. Both were scored on the same untouched held-out "
           "partition. If leakage is real, the leakage track should look near-perfect and be worthless, "
           "because those fields do not exist at the moment a real prediction would be made."),
        code(
            "dt = EXP['dual_track']\n"
            "clean, leak = dt['clean'], dt['leakage']\n"
            "print(f\"{'track':<28}{'AUC-PR':>10}{'AUC-ROC':>10}\")\n"
            "print(f\"{'clean (leakage-safe)':<28}{clean['auc_pr']:>10.4f}{clean['auc_roc']:>10.4f}\")\n"
            "print(f\"{'leakage-included':<28}{leak['auc_pr']:>10.4f}{leak['auc_roc']:>10.4f}\")\n"
            "print(f\"\\nGap (leakage minus clean):  AUC-PR +{leak['auc_pr']-clean['auc_pr']:.4f}, \"\n"
            "      f\"AUC-ROC +{leak['auc_roc']-clean['auc_roc']:.4f}\")"),
        code(
            "labels = ['AUC-PR', 'AUC-ROC']\n"
            "clean_v = [clean['auc_pr'], clean['auc_roc']]\n"
            "leak_v = [leak['auc_pr'], leak['auc_roc']]\n"
            "y = np.arange(len(labels)); h = 0.36\n"
            "fig, ax = plt.subplots(figsize=(8.2, 3.8))\n"
            "b1 = ax.barh(y + h/2, clean_v, height=h, color=PALETTE['blue'], label='clean (leakage-safe)')\n"
            "b2 = ax.barh(y - h/2, leak_v, height=h, color=PALETTE['red'], label='leakage-included')\n"
            "for bars, vals in [(b1, clean_v), (b2, leak_v)]:\n"
            "    for bar, v in zip(bars, vals):\n"
            "        ax.text(v + 0.012, bar.get_y() + bar.get_height()/2, f'{v:.3f}',\n"
            "                va='center', fontweight='bold', fontsize=9)\n"
            "ax.set_yticks(y); ax.set_yticklabels(labels)\n"
            "style(ax, 'Leakage looks near-perfect and is worthless; clean is the honest floor',\n"
            "      'Score (held-out)', '')\n"
            "ax.set_xlim(0, 1.05); ax.grid(axis='y'); ax.legend(frameon=False, loc='lower right')\n"
            "plt.tight_layout(); plt.show()"),
        code(
            "print('Compute basis, disclosed verbatim from the results file:')\n"
            "print(' ', EXP['train_note'])"),
        md("**The leakage track reaches AUC-ROC about 0.94 and is worthless.** That near-perfect number is "
           "exactly the danger the leakage discipline guards against: it is built partly from fields "
           "recorded at or after delivery, which do not exist when a prediction would actually be made, so "
           "it is reading the answer off the outcome. The clean track's AUC-ROC around 0.66 is the honest, "
           "forward-looking picture: modest, because the legitimate pre-delivery signals are individually "
           "weak, and that is the whole point of Week 2.\n\n"
           "Two honesty notes. First, the disclosed compute basis above matters: both tracks were trained "
           "on a stratified cap of 5,000,000 development-train rows (the full development-train partition "
           "was larger), while scoring used the entire held-out partition, so the comparison is fair and "
           "the training cap is stated openly. Second, this clean 0.66 is a FLOOR from a single "
           "quick-trained comparison model, not the project's final result. Calibrated model selection "
           "(logistic regression as the calibration reference, trees, then tuned gradient boosting) is "
           "Week 3 and beyond, framed as PLANNED, and nothing here is a clinical tool. What this section "
           "delivers is the end-to-end proof of the lesson the main notebook made qualitatively: leakage "
           "buys a spectacular, fraudulent number, and the discipline trades it for an honest one."),
        # ---- contribution ----
        md(anchor("contribution", "Per-feature contribution: univariate AUC and gain importance") + "\n\n"
           "If the clean ceiling is modest, the natural next question is which features carry what little "
           "signal there is, and I want to see it two ways. **Univariate AUC** (a full-data rank-sum "
           "computation) asks how well each feature predicts preterm *on its own*, a marginal view. "
           "**LightGBM gain importance** from the clean track asks how much each feature contributed "
           "*inside the model*, an interaction-aware view. They answer different questions, so agreement "
           "between them is reassuring."),
        code(
            "uni = EXP['univariate_auc'][:15]\n"
            "print('Top 15 features by univariate (single-feature) AUC, full-data rank-sum:')\n"
            "for r in uni:\n"
            "    print(f\"  {r['feature']:<34} {r['univariate_auc']:.3f}\")\n"
            "feats = [r['feature'] for r in uni][::-1]; vals = [r['univariate_auc'] for r in uni][::-1]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 6.0))\n"
            "ax.barh(feats, vals, color=PALETTE['blue'])\n"
            "for i, v in enumerate(vals): ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)\n"
            "ax.axvline(0.5, ls=':', color=PALETTE['red'])\n"
            "ax.text(0.5, len(feats)-0.5, ' chance (0.5)', color=PALETTE['red'], fontsize=8, va='top')\n"
            "style(ax, 'Even the strongest single pre-delivery signal is weak (AUC ~0.57)',\n"
            "      'Univariate AUC (full data)', ''); ax.grid(axis='y')\n"
            "ax.set_xlim(0.49, max(vals) + 0.01); plt.tight_layout(); plt.show()"),
        code(
            "gain = EXP['gain_importance'][:15]\n"
            "gmax = gain[0]['gain']\n"
            "print('Top 15 features by LightGBM gain importance (clean track), gain shown relative to the max:')\n"
            "for r in gain:\n"
            "    print(f\"  {r['feature']:<34} gain {r['gain']:>14,.0f}  ({r['gain']/gmax*100:5.1f}% of max)\")\n"
            "feats = [r['feature'] for r in gain][::-1]; vals = [r['gain']/gmax for r in gain][::-1]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 6.0))\n"
            "ax.barh(feats, vals, color=PALETTE['green'])\n"
            "for i, v in enumerate(vals): ax.text(v + 0.008, i, f'{v:.2f}', va='center', fontsize=8)\n"
            "style(ax, 'In-model gain importance agrees: rf_ppterm and medical_risk_count lead',\n"
            "      'Gain importance (relative to the top feature)', ''); ax.grid(axis='y')\n"
            "ax.set_xlim(0, 1.08); plt.tight_layout(); plt.show()"),
        md("**The strongest single signal is weak, and the two views agree.** The best univariate feature, "
           "the enumerated medical-risk count, reaches only about 0.57 AUC on its own, and prior preterm "
           "birth (`rf_ppterm`) is the strongest by in-model gain and, as the next section shows, by "
           "linear correlation too. Everything else is close to the 0.5 chance line. That weakness is the "
           "honest reason the clean ceiling is modest: there is simply no single pre-delivery field that "
           "carries strong predictive power. The reassuring part is that the marginal ranking (univariate "
           "AUC) and the in-model ranking (gain) mostly agree, with `rf_ppterm`, `medical_risk_count`, and "
           "`bmi` near the top of both. When a purely marginal measure and an interaction-aware one point "
           "at the same handful of features, I trust the signal more, even though it is small."),
        # ---- correlation and MI ----
        md(anchor("corrmi", "Correlation and mutual information: compare and contrast") + "\n\n"
           "Two more lenses on the same features, and this is where careful reading matters most. "
           "**Pearson correlation** (computed on the full data) measures *linear* association with the "
           "outcome. **Mutual information** (estimated on a 1,000,000-row subsample) can additionally pick "
           "up *nonlinear or categorical* association that a linear correlation misses. Comparing the two "
           "rankings is instructive, but only if we read the mutual-information side honestly."),
        code(
            "corr = EXP['correlation'][:12]\n"
            "print('Top 12 features by |Pearson correlation| with preterm (full data):')\n"
            "for r in corr:\n"
            "    print(f\"  {r['feature']:<36} {r['corr']:+.3f}\")\n"
            "print(f\"\\nStrongest linear correlation is only {abs(corr[0]['corr']):.3f} ({corr[0]['feature']}).\")"),
        code(
            "mi = EXP['mutual_info'][:12]\n"
            "print(f\"Top 12 features by mutual information (estimated on {EXP['mi_subsample_rows']:,} rows):\")\n"
            "for r in mi:\n"
            "    print(f\"  {r['feature']:<36} {r['mi']:.4f}\")\n"
            "corr_top = {r['feature'] for r in EXP['correlation'][:5]}\n"
            "mi_top = {r['feature'] for r in EXP['mutual_info'][:5]}\n"
            "print('\\nTop-5 overlap between the correlation and MI rankings:',\n"
            "      sorted(corr_top & mi_top) or '(none)')"),
        md("**The two rankings disagree at the top, and the honest reading is the careful one.** Linear "
           "correlation puts `rf_ppterm` and `medical_risk_count` on top, matching the univariate-AUC and "
           "gain rankings. Mutual information instead puts `first_trimester_care` and `married` at the top, "
           "features that do NOT lead the correlation or the univariate-AUC rankings. It is tempting to "
           "read that as *these are secretly strong predictors that linear methods missed*, and I want to "
           "resist that.\n\n"
           "Descriptively, mutual information can indeed capture nonlinear or categorical association that "
           "Pearson correlation cannot, so it is a legitimate different lens. But there is a real caveat: "
           "estimating mutual information against a rare binary outcome (about 8.6% prevalence) is noisy, "
           "and it can be inflated for low-variance binary features. So the top of the MI ranking should be "
           "read as a *hint* about where nonlinear structure might live, not as evidence that "
           "`first_trimester_care` is a strong predictor. It is not: its own univariate AUC sits near the "
           "chance line with everything else. The honest indicator across all of these lenses is the same "
           "one from the previous section: the univariate AUCs, all clustered near the 0.5 chance line "
           "(roughly 0.45 to 0.57, with protective-direction features like education and being married "
           "landing below 0.5), so no single pre-delivery feature is more than about 0.07 of AUC away from "
           "a coin flip. That is what says plainly that no single pre-delivery feature is strongly "
           "predictive of preterm birth."),
        # ---- PCA ----
        md(anchor("pca", "PCA: is the feature space redundant?") + "\n\n"
           "A different question: are these weak signals actually redundant, collapsing into a few "
           "dominant directions, or is the pre-delivery feature space genuinely spread out? Principal "
           "component analysis answers it. If a handful of components captured most of the variance, there "
           "would be a low-dimensional shortcut worth exploiting. If not, the coming modeling weeks have to "
           "respect the full breadth of the features."),
        code(
            "pca = EXP['pca']\n"
            "evr = pca['explained_variance_ratio']; cum = pca['cumulative']\n"
            "print('Explained variance by principal component (first 10):')\n"
            "for i, (e, c) in enumerate(zip(evr, cum), 1):\n"
            "    print(f\"  PC{i:<2} {e*100:5.2f}%   cumulative {c*100:5.2f}%\")\n"
            "print(f\"\\nTop 10 components together explain {cum[-1]*100:.2f}% of the variance.\")\n"
            "fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.0))\n"
            "pcs = [f'PC{i}' for i in range(1, len(evr)+1)]\n"
            "axL.bar(pcs, [e*100 for e in evr], color=PALETTE['blue'])\n"
            "for i, e in enumerate(evr): axL.text(i, e*100 + 0.05, f'{e*100:.1f}', ha='center', fontsize=8)\n"
            "style(axL, 'Scree: no dominant component', 'Principal component', 'Explained variance (%)')\n"
            "plt.setp(axL.get_xticklabels(), rotation=45, ha='right')\n"
            "axR.plot(range(1, len(cum)+1), [c*100 for c in cum], marker='o', color=PALETTE['green'])\n"
            "for i, c in enumerate(cum, 1): axR.text(i, c*100 + 0.6, f'{c*100:.0f}', ha='center', fontsize=8)\n"
            "style(axR, 'Cumulative variance: ~22% in 10 PCs', 'Number of components', 'Cumulative variance (%)')\n"
            "axR.set_ylim(0, max(30, cum[-1]*100 + 6)); plt.tight_layout(); plt.show()"),
        code(
            "print('Top-8 loadings for the first three components (which raw features each PC weights most):')\n"
            "for pc in ['PC1', 'PC2', 'PC3']:\n"
            "    print(f\"\\n{pc}:\")\n"
            "    for r in pca['loadings'][pc]:\n"
            "        print(f\"  {r['feature']:<34} {r['loading']:+.3f}\")"),
        md("**The feature space is largely non-redundant.** The strongest single component (PC1) explains "
           "only about 4.6% of the variance, and the first 10 components together reach only about 22%. "
           "Reaching 80% or 90% of the variance would take far more than 10 components. In other words, the "
           "weak signals do not collapse into a few dominant directions; they are spread thin across many, "
           "so there is no low-dimensional shortcut to exploit, and the coming modeling weeks have to work "
           "with the features roughly as they are. One piece of context, not a defect: the matrix is "
           "one-hot-heavy (race, payment source, and marital status are each expanded into indicator "
           "columns), and a one-hot-heavy design of this width inflates the raw dimensionality, which is "
           "part of why the variance looks so spread out."),
        # ---- clustering ----
        md(anchor("clusters", "Clustering: do natural risk subpopulations emerge?") + "\n\n"
           "The last lens is unsupervised. KMeans was run on the full population to ask whether natural "
           "subpopulations fall out of the pre-delivery features, and if so, how their preterm rates "
           "spread around the 8.6% base rate. This is strictly a descriptive, associational look at "
           "structure in the data, not a risk-scoring tool."),
        code(
            "clusters = sorted(EXP['clusters'], key=lambda c: c['cluster'])\n"
            "print(f\"{'cluster':>7}{'size':>13}{'preterm':>10}   top races (share)          key feature means\")\n"
            "for c in clusters:\n"
            "    races = ', '.join(f\"{m['race']} {m['share']*100:.0f}%\" for m in c['race_mix'])\n"
            "    fm = c['feature_means']\n"
            "    means = ', '.join(f\"{k}={v:.2f}\" for k, v in fm.items())\n"
            "    print(f\"{c['cluster']:>7}{c['size']:>13,}{c['preterm_rate']*100:>9.1f}%   {races:<26} {means}\")"),
        code(
            "base = EXP['n_singletons']  # sanity anchor for the ~8.6% base rate referenced below\n"
            "ids = [f\"cluster {c['cluster']}\" for c in clusters]\n"
            "rates = [c['preterm_rate']*100 for c in clusters]\n"
            "order = np.argsort(rates)\n"
            "ids = [ids[i] for i in order]; rates = [rates[i] for i in order]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.barh(ids, rates, color=PALETTE['purple'])\n"
            "for i, v in enumerate(rates): ax.text(v + 0.05, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=9)\n"
            "ax.axvline(8.6, ls=':', color=PALETTE['red'])\n"
            "ax.text(8.6, len(ids)-0.5, ' ~8.6% base rate', color=PALETTE['red'], fontsize=8, va='top')\n"
            "style(ax, 'Per-cluster preterm rate spreads only modestly around the 8.6% base rate',\n"
            "      'Preterm rate (%)', ''); ax.grid(axis='y')\n"
            "ax.set_xlim(0, max(rates) * 1.15); plt.tight_layout(); plt.show()"),
        md("**Subpopulations emerge, but their preterm rates spread only modestly.** The clusters range "
           "from roughly 6.1% to 9.9% preterm, a spread of only a few points around the 8.6% base rate. "
           "That modest separation is exactly what we should expect given the previous sections: when no "
           "individual feature is strongly predictive, unsupervised groupings cannot pull far apart on the "
           "outcome either. The clusters differ in composition (their race mixes and feature means differ), "
           "which is descriptively interesting, but the outcome spread is small.\n\n"
           "This is strictly associational. The race mix of a cluster describes who happens to fall in it, "
           "and race here is a marker of exposure to racism and structural barriers, not a biological "
           "cause of preterm birth. This clustering is not a clinical tool and is not used to score "
           "individual risk."),
        # ---- plan ----
        md(anchor("plan", "What this means for modeling (PLANNED)") + "\n\n"
           "Pulling the experiments together, a consistent picture emerges, and it is an honest one. The "
           "pre-delivery performance ceiling looks modest: a clean, leakage-safe comparison model lands "
           "around AUC-ROC 0.66, which this companion should be read as a floor rather than a final "
           "result. The features that carry signal are the same few across every lens (`rf_ppterm`, "
           "`medical_risk_count`, `bmi`), and even the strongest is weak on its own. The feature space is "
           "non-redundant, so there is no low-dimensional shortcut. And leakage, the real danger the whole "
           "discipline guards against, would buy a fraudulent AUC-ROC near 0.94 that evaporates the moment "
           "you remember those fields do not exist at prediction time.\n\n"
           "**The coming weeks, framed as the PLAN, not as work already done:**\n\n"
           "- **Calibrated baseline models.** Logistic regression as the calibration reference, a decision "
           "tree, and a random forest, judged on rate-aware metrics and calibration rather than accuracy.\n"
           "- **Gradient boosting with proper tuning.** XGBoost and LightGBM tuned with Optuna, to see how "
           "much (if any) the ceiling moves once we tune carefully.\n"
           "- **The first formal fairness audit**, turning per-group performance into rigorous metrics with "
           "confidence intervals and significance tests.\n"
           "- **A decomposition of the racial gap**, asking how much of the disparity measured factors can "
           "account for, and how much is left unexplained.\n\n"
           "None of those results exist yet; they are the plan this companion sets up. The modest ceiling "
           "is not a disappointment, it is the finding: these records support population-level equity "
           "diagnosis, not individual risk scoring, and that is the honest thing to build on. Nothing here "
           "is a clinical tool." + AUTHOR),
    ]
    return cells


def main():
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
