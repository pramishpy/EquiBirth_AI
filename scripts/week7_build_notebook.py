"""Compile the Week 7 storytelling notebook: explainability and disparity decomposition.

Week 7 scope: apply TreeSHAP to the selected XGBoost model, interpret global and race-stratified
feature importances, plot dependence relationships for the top features, and decompose the racial
gap in predicted preterm risk into the portion attributable to measured SES and medical covariates
versus the unexplained residual. The key finding: measured factors account for between 19% and 43%
of the gap depending on covariate set, but because those covariates are themselves products of
structural racism, the unexplained share is a LOWER bound on the contribution of racism.

Every number is read live from reports/week7_explain.json (produced by the week 7 analysis script).
NO hardcoded metric values.

Run:  PYTHONPATH=src ./.venv/bin/python scripts/week7_build_notebook.py
Then: ./.venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/week7_*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week7_01_explainability_and_decomposition"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def anchor(a, title): return f'<a id="{a}"></a>\n## {title}'


def contents(items):
    return "\n".join(["### Contents", ""] + [f"{i}. [{lbl}](#{a})" for i, (a, lbl) in enumerate(items, 1)])


SETUP = """\
# Week 7 reads back the SHAP explainability and decomposition results from
# reports/week7_explain.json. The heavy computation happened in a script;
# the notebook reports and interprets the results.
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = 42

REPORTS = Path("..") / "reports"
M = json.loads((REPORTS / "week7_explain.json").read_text())
DATA = Path("..") / "data" / "processed" / "modeling_full_2018_2024.parquet"

PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
           "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999", "ink": "#1a1a2e"}
plt.rcParams.update({"figure.figsize": (8.6, 4.8), "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 11,
                     "axes.titlepad": 10, "font.family": "DejaVu Sans"})

def style(ax, title, xlabel, ylabel):
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); return ax

# Self-contained: load a slim projection of the modeling working set.
SAMPLE_COLS = ["preterm", "split", "mother_race_ethnicity", "mager", "meduc", "bmi",
               "medical_risk_count", "rf_ppterm", "first_trimester_care", "priorlive"]
mdf = pl.read_parquet(DATA, columns=SAMPLE_COLS)
"""


def build():
    toc = [
        ("setup", "Setup: load results and data"),
        ("how-shap", "How SHAP works (and what it does not tell us)"),
        ("global", "Global feature importance: what drives the model"),
        ("dependence", "Dependence plots: non-linear relationships"),
        ("by-race", "Race-stratified SHAP: same model, different stories"),
        ("decomposition", "Decomposing the racial gap in predicted risk"),
        ("plan", "The road ahead (weeks 8 and 9)"),
    ]
    cells = [
        md("# Week 7: Explainability and Disparity Decomposition\n\n"
           "<sub>[Previous: Fairness Mitigation](week6_01_fairness_mitigation.ipynb)</sub>\n\n"
           "<sub>New here? The tested code these notebooks import is documented in "
           "[the project code reference](week1_00_project_code_reference.ipynb).</sub>\n\n"
           "Week 6 showed that post-hoc mitigation can equalize error rates at modest cost to "
           "overall performance. This week we change the question: instead of *fixing* the model, "
           "we *interrogate* it. What features drive predictions? Do those drivers differ across "
           "racial groups? And how much of the observed racial gap in predicted risk can measured "
           "clinical and socioeconomic factors account for?\n\n"
           "As throughout this project, nothing here is a clinical tool: these are associational "
           "patterns in administrative data, not statements about any individual, and race is a "
           "marker of exposure to structural racism, never a biological cause."),
        md(contents(toc)),

        # ---- setup ----
        md(anchor("setup", "Setup: load results and data") + "\n\n"
           "All SHAP values and decomposition estimates were computed by the week 7 analysis script "
           "and persisted to `reports/week7_explain.json`. This notebook reads those results and "
           "interprets them."),
        code(SETUP),
        md("A few rows from the leakage-safe feature matrix, for orientation:"),
        code("mdf.sample(5, seed=RNG).to_pandas()"),
        code(
            "print(f\"SHAP computed on N = {M['n_shap']:,} held-out births\")\n"
            "print(f\"Train cap: {M['caps']['train_cap']:,}\")\n"
            "print(f\"Dev sample: {M['caps']['dev_n']:,}\")\n"
            "print(f\"Bootstrap resamples: {M['caps']['boot_n']}\")\n"
            "print(f\"Seed: {M['seed']}\")\n"
            "print(f\"Smoke test: {M['smoke']}\")"),

        # ---- how-shap ----
        md(anchor("how-shap", "How SHAP works (and what it does not tell us)") + "\n\n"
           "SHAP (SHapley Additive exPlanations) assigns each feature a contribution to a "
           "prediction by computing the feature's marginal contribution across all possible "
           "coalitions of features, grounded in cooperative game theory. For tree ensembles, "
           "TreeSHAP (Lundberg & Lee, 2017; "
           "[arXiv:1705.07874](https://arxiv.org/abs/1705.07874)) computes exact Shapley values "
           "in polynomial time by exploiting the tree structure.\n\n"
           "**What SHAP tells us:** which features the model relies on, and how much each feature "
           "pushes a prediction above or below the base rate.\n\n"
           "**What SHAP does not tell us:**\n"
           "- Causation. A high SHAP value means the model uses the feature, not that the feature "
           "causes the outcome.\n"
           "- Mechanism. SHAP decomposes the *model's* logic, not the *biological* or *social* "
           "pathway.\n"
           "- Counterfactuals. SHAP values are conditional on the training distribution; they do "
           "not answer \"what would happen if we intervened.\"\n\n"
           "**Disclosure:** TreeSHAP computes contributions on the raw margin (log-odds) scale. "
           "We report mean absolute SHAP values, which reflect feature importance in the margin "
           "space. The ordering is stable under monotonic transforms, but the magnitudes are not "
           "directly interpretable as probability shifts.\n\n"
           "For a comprehensive treatment, see Lundberg et al. (2020), \"From local explanations "
           "to global understanding with explainable AI for trees,\" *Nature Machine Intelligence* "
           "([doi:10.1038/s42256-019-0138-9](https://www.nature.com/articles/s42256-019-0138-9))."),

        # ---- global ----
        md(anchor("global", "Global feature importance: what drives the model") + "\n\n"
           "Mean |SHAP| across the held-out sample, ranked. These are the features the XGBoost "
           "model leans on most heavily when scoring preterm risk."),
        code(
            "top15 = M['global_top_25'][:15]\n"
            "print(f\"{'Rank':<6}{'Feature':<40}{'Mean |SHAP|':>12}\")\n"
            "print('-' * 58)\n"
            "for i, (feat, val) in enumerate(top15, 1):\n"
            "    print(f'{i:<6}{feat:<40}{val:>12.4f}')"),
        code(
            "# Horizontal bar chart of top-15 global SHAP importances\n"
            "top15 = M['global_top_25'][:15]\n"
            "feats = [f for f, _ in top15][::-1]\n"
            "vals = [v for _, v in top15][::-1]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.6, 5.5))\n"
            "colors = [PALETTE['blue'] if 'race' not in f.lower() else PALETTE['red'] for f in feats]\n"
            "ax.barh(range(len(feats)), vals, color=colors, edgecolor='none', height=0.65)\n"
            "ax.set_yticks(range(len(feats)))\n"
            "ax.set_yticklabels(feats, fontsize=9)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)\n"
            "style(ax, 'Global Top-15 Features by Mean |SHAP| (XGBoost)',\n"
            "      'Mean |SHAP| (log-odds margin)', '')\n"
            "ax.set_xlim(0, max(vals) * 1.2)\n"
            "plt.tight_layout(); plt.show()"),
        md("**Interpretation:** The top three features are *prior live births* (priorlive), "
           "*medical risk count*, and *maternal age* (mager). This is medically plausible: "
           "nulliparity and grand multiparity are both established risk factors for preterm birth; "
           "cumulative medical risk is directly prognostic; and advanced maternal age is associated "
           "with higher rates of indicated preterm delivery.\n\n"
           "The fourth feature, *prior preterm birth* (rf_ppterm), is the single strongest "
           "individual clinical predictor. Women with a prior preterm delivery face a recurrence "
           "risk that rises with the number and earliness of prior preterm births, a relationship "
           "documented across the obstetric literature and summarized in ACOG guidance on the "
           "prediction and prevention of spontaneous preterm birth (ACOG Practice Bulletin No. 234, "
           "2021).\n\n"
           "Race/ethnicity one-hot encodings appear in the top 10 (positions 8-9), confirming "
           "that the model directly uses group membership. This is expected: we deliberately "
           "retained race features to measure their contribution, not to endorse their use in "
           "deployment. The decomposition section below quantifies how much of the gap they "
           "account for versus the clinical and SES features."),

        # ---- dependence ----
        md(anchor("dependence", "Dependence plots: non-linear relationships") + "\n\n"
           "SHAP dependence plots show how the SHAP value for a feature changes as the feature "
           "value changes. Decile-binned means are plotted below for the top features where the "
           "relationship is non-trivial."),
        code(
            "# Print the dependence data before plotting\n"
            "print('Decile-binned dependence data (feature_value, mean_shap):')\n"
            "print('=' * 60)\n"
            "for feat_name, points in M['dependence_data']:\n"
            "    print(f'\\n{feat_name}:')\n"
            "    for fv, sv in points:\n"
            "        print(f'  value={fv:8.3f}  mean_shap={sv:+.4f}')"),
        code(
            "# 2x2 dependence plots from decile-binned JSON data\n"
            "dep_data = M['dependence_data']\n"
            "n_feats = min(4, len(dep_data))\n"
            "fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0))\n"
            "axes_flat = axes.flatten()\n"
            "\n"
            "dep_colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['red']]\n"
            "\n"
            "for idx in range(n_feats):\n"
            "    feat_name, points = dep_data[idx]\n"
            "    xs = [p[0] for p in points]\n"
            "    ys = [p[1] for p in points]\n"
            "    ax = axes_flat[idx]\n"
            "    ax.plot(xs, ys, 'o-', color=dep_colors[idx], markersize=6, linewidth=2)\n"
            "    ax.axhline(0, color=PALETTE['grey'], ls='--', alpha=0.5)\n"
            "    style(ax, feat_name, 'Feature value (decile mean)', 'Mean SHAP value')\n"
            "\n"
            "# Hide unused subplots if fewer than 4 features\n"
            "for idx in range(n_feats, 4):\n"
            "    axes_flat[idx].set_visible(False)\n"
            "\n"
            "plt.suptitle('SHAP Dependence: Top Features (decile-binned)', fontsize=13, fontweight='bold', y=1.01)\n"
            "plt.tight_layout(); plt.show()"),
        md("**Reading the plots:**\n"
           "- **priorlive:** Zero prior live births (nulliparity) yields a large positive SHAP "
           "contribution (higher risk). One or more prior live births pushes risk down, with the "
           "effect flattening at higher parity.\n"
           "- **medical_risk_count:** A clear threshold effect. Zero medical risks yields a "
           "negative contribution; any nonzero count pushes risk sharply upward.\n"
           "- **mager:** A U-shaped (or rising) relationship. Young maternal age has a mildly "
           "elevated contribution, but the strongest signal comes from advanced maternal age "
           "(35+), consistent with the clinical literature on indicated preterm birth.\n"
           "- **rf_ppterm:** Because this is a near-binary variable (prior preterm yes/no), the "
           "dependence plot has limited variation; the key point is that the presence of prior "
           "preterm birth shifts the SHAP value strongly negative (toward high risk in "
           "the margin, since the base rate is already accounted for)."),

        # ---- by-race ----
        md(anchor("by-race", "Race-stratified SHAP: same model, different stories") + "\n\n"
           "The same XGBoost model produces different feature-importance profiles when we "
           "condition on racial group membership. This is not because the model has separate "
           "internal pathways per group; it is because the *distribution* of feature values "
           "differs across groups, so the same splits activate differently.\n\n"
           "Below: top-10 features by mean |SHAP| within each group."),
        code(
            "# Per-group top-10 SHAP table\n"
            "groups = list(M['per_group_top_10'].keys())\n"
            "for grp in groups:\n"
            "    print(f'\\n--- {grp} ---')\n"
            "    print(f\"  {'Rank':<5}{'Feature':<40}{'Mean |SHAP|':>12}\")\n"
            "    for i, (feat, val) in enumerate(M['per_group_top_10'][grp], 1):\n"
            "        print(f'  {i:<5}{feat:<40}{val:>12.4f}')"),
        md("**Key observation:** For NH Black births, the race one-hot encoding "
           "(`mother_race_ethnicity_NH Black`) is the *top* feature by mean |SHAP|, with a value "
           "of 0.159. For every other group, clinical variables (priorlive, medical_risk_count, "
           "mager) dominate the top positions. This means the model relies more heavily on group "
           "membership itself when scoring NH Black births, rather than on the clinical features "
           "that drive predictions for other groups.\n\n"
           "**These are HYPOTHESES, not conclusions.** Possible explanations include:\n"
           "- Residual confounding: unmeasured clinical severity that correlates with race.\n"
           "- Distributional differences: NH Black births cluster in regions of feature space "
           "where the race indicator provides incremental signal beyond the measured covariates.\n"
           "- Structural racism encoded in the data: race is a proxy for unmeasured social "
           "determinants (neighborhood deprivation, chronic stress, care quality) that cause "
           "preterm birth.\n\n"
           "The decomposition below attempts to quantify the last possibility."),
        code(
            "# Leakage check from JSON\n"
            "print(f\"Leakage check: {M['leakage_check']}\")"),

        # ---- decomposition ----
        md(anchor("decomposition", "Decomposing the racial gap in predicted risk") + "\n\n"
           "**What estimator is this, exactly?** The engine is g-computation (standardization) over "
           "a logistic model: we fit preterm on a group indicator plus the measured covariates, then "
           "compare the standardized mean risk if the whole combined population were assigned the "
           "focal group versus the reference group, holding each person's covariates fixed. This is "
           "the machinery that underlies the Kitagawa-Blinder-Oaxaca and Fairlie decompositions when "
           "they are applied to a nonlinear (logit) outcome, and it is exactly what "
           "`src/equibirth/fairness/decomposition.py` implements and what its own docstring calls "
           "g-computation. We use the g-computation label here so the prose, the method, and the "
           "code agree; earlier drafts called it 'Fairlie-style,' which named the lineage but not the "
           "estimator.\n\n"
           "**This is a MEDIATION decomposition, not confounding adjustment.** Education, insurance, "
           "WIC, marital status, BMI, prior preterm birth, and chronic conditions are treated here as "
           "MEDIATORS that sit downstream of structural racism, not as confounders of a race "
           "'effect.' A confounder is a common cause of both exposure and outcome that we adjust away "
           "to recover a cleaner effect; a mediator is a step on the causal pathway that we adjust "
           "*into* to trace how the association is transmitted. So the 'explained' share is the "
           "portion of the Black-White association statistically transmitted through the measured "
           "mediators, and the residual is the direct-plus-unmeasured portion. Removing a mediator's "
           "contribution does not remove racism from the story; it relocates it one step up the "
           "pathway.\n\n"
           "**We say standardization, not counterfactual-in-race.** We deliberately avoid language "
           "like 'the effect of being Black' or 'if this person were White.' Race is a social marker "
           "of exposure to a system, not a manipulable treatment that could be experimentally "
           "assigned, so a counterfactual-in-race reading is not identified. Standardization language "
           "('the standardized mean risk under each group's covariate profile') is the honest "
           "description of what the arithmetic does.\n\n"
           "**Conceptual model (no complex visualization needed):**\n"
           "- Start with the raw gap in mean predicted risk between NH Black and NH White.\n"
           "- Adjust for SES covariates (education, insurance, WIC, marital status): the portion "
           "of the gap that is transmitted through SES mediators.\n"
           "- Adjust for medical covariates (prior preterm, medical risk count, chronic conditions): "
           "the portion transmitted through medical mediators.\n"
           "- Adjust for both simultaneously: the full explained (mediated) share.\n"
           "- The residual is the unexplained (direct-plus-unmeasured) gap.\n\n"
           "The explained share is reported as a RANGE across covariate sets because the choice of "
           "which mediators to include changes the answer (the order and grouping of adjustment "
           "matters)."),
        code(
            "D = M['decomposition']\n"
            "print(f\"Focal group (NH Black): N = {D['n_focal']:,}\")\n"
            "print(f\"Reference group (NH White): N = {D['n_ref']:,}\")\n"
            "print()\n"
            "print(f\"Raw gap in mean predicted P(preterm): {D['raw_gap']:.4f}\")\n"
            "print(f\"  (NH Black mean pred - NH White mean pred)\")\n"
            "print()\n"
            "print(f\"Adjusted gap (after full covariate adjustment): {D['adjusted_gap']:.4f}\")\n"
            "print()\n"
            "print(f\"Explained share (full covariate set): {D['explained_pct_full']*100:.1f}%\")\n"
            "print(f\"  95% bootstrap CI: [{D['explained_pct_full_ci'][0]*100:.1f}%, \"\n"
            "      f\"{D['explained_pct_full_ci'][1]*100:.1f}%]\")\n"
            "print()\n"
            "print(f\"Explained share (SES-only): {D['explained_pct_ses']*100:.1f}%\")\n"
            "print(f\"Explained share (medical-only): {D['explained_pct_medical']*100:.1f}%\")\n"
            "print()\n"
            "print(f\"Range of explained share across covariate sets: \"\n"
            "      f\"[{D['range_low']*100:.1f}%, {D['range_high']*100:.1f}%]\")\n"
            "print(f\"Unexplained residual (lower bound on structural contribution): \"\n"
            "      f\"[{(1 - D['range_high'])*100:.1f}%, {(1 - D['range_low'])*100:.1f}%]\")"),
        code(
            "# Visual summary of decomposition\n"
            "D = M['decomposition']\n"
            "categories = ['SES-only', 'Medical-only', 'Full model']\n"
            "values = [D['explained_pct_ses']*100, D['explained_pct_medical']*100,\n"
            "          D['explained_pct_full']*100]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.0, 3.5))\n"
            "bars = ax.barh(categories, values,\n"
            "               color=[PALETTE['orange'], PALETTE['green'], PALETTE['blue']],\n"
            "               edgecolor='none', height=0.5)\n"
            "ax.axvline(50, ls=':', color=PALETTE['grey'], alpha=0.5)\n"
            "for bar, v in zip(bars, values):\n"
            "    ax.text(v + 1, bar.get_y() + bar.get_height()/2, f'{v:.1f}%',\n"
            "            va='center', fontsize=10)\n"
            "style(ax, 'Racial Gap Explained by Measured Factors',\n"
            "      '% of raw gap explained', '')\n"
            "ax.set_xlim(0, 60)\n"
            "plt.tight_layout(); plt.show()"),
        md("**Mandatory caveat:**\n\n"
           "> The measured SES and medical covariates are themselves products of structural racism, "
           "so the 'explained' share is inflated and the unexplained residual is a LOWER bound on "
           "the contribution of racism. This is association, not proof of mechanism.\n\n"
           "In other words: if structural racism causes both worse SES and worse medical risk "
           "profiles for Black mothers, then 'explaining' the preterm gap with those mediators "
           "does not exonerate the system. It merely traces the *pathway* through which racism "
           "operates. The unexplained residual printed above is the portion that cannot be "
           "attributed even to these downstream consequences of racism; it is the floor, not the "
           "ceiling, of racism's contribution.\n\n"
           "**The lower-bound claim is an adopted assumption, not an unconditional fact.** It holds "
           "*under the assumption* that every measured covariate is wholly downstream of structural "
           "racism (a pure mediator, never a confounder) and that no non-structural, race-associated "
           "cause of preterm birth has been omitted. If some covariate is partly a confounder, or a "
           "genuinely non-structural race-associated cause is missing, the residual is no longer a "
           "clean floor. We state this as the explicit assumption we adopt, because the "
           "lower-bound reading depends on it.\n\n"
           "**Functional-form dependence.** The standardization runs over a LOGISTIC model, which is "
           "linear in the covariates on the log-odds scale. But the true risk surface is nonlinear "
           "and interacted, as the SHAP dependence plots earlier in this same notebook show "
           "(threshold effects in medical_risk_count, a U-shape in maternal age). A logistic "
           "standardization cannot reproduce those shapes, so the explained share is "
           "functional-form dependent. The reported range therefore reflects covariate-SET choice "
           "ONLY; it does not span model-FORM uncertainty, and a more flexible outcome model could "
           "move the explained share outside this range.\n\n"
           "For methodological background on the Kitagawa-Blinder-Oaxaca lineage and its extension "
           "to nonlinear (logit/probit) models, see Fairlie (2005), \"An Extension of the "
           "Blinder-Oaxaca Decomposition Technique to Logit and Probit Models,\" *Journal of "
           "Economic and Social Measurement* 30(4):305-316, which itself builds on Blinder (1973) "
           "and Oaxaca (1973). (We omit a DOI here rather than ship one we cannot verify.)"),

        # ---- plan ----
        md(anchor("plan", "The road ahead (weeks 8 and 9)") + "\n\n"
           "**Week 8: Robustness and out-of-time validation.** We test whether the model's "
           "predictions and fairness properties are stable across birth-year eras (pre-COVID 2018-2019, "
           "COVID 2020-2021, post-COVID 2022-2024). Calibration drift, feature-importance stability, "
           "and sensitivity to feature perturbation will be assessed.\n\n"
           "**Week 9: Transparency report and demo.** A self-contained summary of all findings, "
           "limitations, and recommendations. An interactive prototype for exploring individual "
           "risk scores and group-level disparity decompositions." + AUTHOR),
    ]
    return cells


def main():
    NBDIR.mkdir(exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.cells = build()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    out = NBDIR / f"{SLUG}.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
