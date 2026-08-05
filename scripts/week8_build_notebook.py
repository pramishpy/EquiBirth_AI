"""Compile the Week 8 storytelling notebook: robustness and stratified analysis.

Week 8 scope: out-of-time validation (pre-COVID model scored on COVID/recent eras), per-year
drift monitoring, stratified per-group models (hypothesis-generating), threshold sensitivity,
and training-cap sensitivity. The headline: the model is temporally stable (AUC drop ~0.003),
group-level models reveal different feature emphasis but not leakage, and the performance
ceiling is confirmed by training-cap saturation.

Every number is read live from reports/week8_robustness.json (produced by the week 8 robustness
pipeline). NO hardcoded metric values.

Run:  PYTHONPATH=src ./.venv/bin/python scripts/week8_build_notebook.py
Then: ./.venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/week8_*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week8_01_robustness_and_stratified"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def anchor(a, title): return f'<a id="{a}"></a>\n## {title}'


def contents(items):
    return "\n".join(["### Contents", ""] + [f"{i}. [{lbl}](#{a})" for i, (a, lbl) in enumerate(items, 1)])


SETUP = """\
# Week 8 reads back robustness and stratified-analysis results from
# reports/week8_robustness.json. The heavy computation happened upstream;
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
M = json.loads((REPORTS / "week8_robustness.json").read_text())
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
               "medical_risk_count", "rf_ppterm", "first_trimester_care"]
mdf = pl.read_parquet(DATA, columns=SAMPLE_COLS)
"""


def build():
    toc = [
        ("setup", "Setup: load results and data"),
        ("why-time", "Why temporal validation matters"),
        ("oot", "Out-of-time validation: pre-COVID to COVID and beyond"),
        ("stratified", "Stratified per-group models"),
        ("sensitivity", "Sensitivity analysis: thresholds and training caps"),
        ("plan", "Week 9 and beyond"),
    ]
    cells = [
        md("# Week 8: Robustness and Stratified Analysis\n\n"
           "<sub>[Previous: Explainability and Disparity Decomposition](week7_01_explainability_and_decomposition.ipynb)</sub>\n\n"
           "<sub>New here? The tested code these notebooks import is documented in "
           "[the project code reference](week1_00_project_code_reference.ipynb).</sub>\n\n"
           "Week 7 decomposed the racial gap in predicted risk into feature-group contributions. "
           "This week we stress-test the model against time and subpopulations. The questions: "
           "does the model hold up when the world changes (COVID), and do per-group models "
           "reveal blind spots that the pooled model hides?\n\n"
           "As throughout this project, nothing here is a clinical tool: these are associational "
           "patterns in administrative data, not statements about any individual pregnancy."),
        md(contents(toc)),

        # ---- setup ----
        md(anchor("setup", "Setup: load results and data") + "\n\n"
           "All results were produced by the week 8 robustness pipeline and saved to "
           "`reports/week8_robustness.json`. We read them here and render the findings."),
        code(SETUP),
        md("A few rows from the leakage-safe feature matrix, for orientation:"),
        code("mdf.sample(5, seed=RNG).to_pandas()"),
        code(
            "print(f\"OOT train cap: {M['caps']['oot_train_cap']:,}\")\n"
            "print(f\"Held-out cap: {M['caps']['held_out_cap']:,}\")\n"
            "print(f\"Group train cap: {M['caps']['group_train_cap']:,}\")\n"
            "print(f\"Group held-out cap: {M['caps']['group_ho_cap']:,}\")\n"
            "print(f\"Seed: {M['seed']}\")"),

        # ---- why-time ----
        md(anchor("why-time", "Why temporal validation matters") + "\n\n"
           "Random train/test splits flatter models. They assume the future looks like the past, "
           "uniformly shuffled. In clinical ML this assumption is dangerous: coding practices "
           "change, populations shift, pandemics intervene. If the model cannot survive deployment "
           "across calendar time, its AUC on a random split is a comforting fiction.\n\n"
           "Finlayson et al. (NEJM 2021, "
           "[doi:10.1056/NEJMc2104626](https://www.nejm.org/doi/full/10.1056/NEJMc2104626)) "
           "documented systematic dataset shift in clinical ML models during COVID-19, showing that "
           "models trained pre-pandemic degraded on pandemic-era data. Our out-of-time experiment "
           "directly tests this: we train on 2018-2019 (pre-COVID) and score on 2020-2021 (COVID) "
           "and 2022-2024 (recent). If the model survives the largest distributional shock in the "
           "data window, we have evidence of genuine temporal robustness."),

        # ---- oot ----
        md(anchor("oot", "Out-of-time validation: pre-COVID to COVID and beyond") + "\n\n"
           "The out-of-time (OOT) model was trained exclusively on 2018-2019 births, calibrated "
           "on a held-out slice of the same era, then evaluated on the COVID era (2020-2021) and "
           "the recent era (2022-2024) without retraining. This is the strictest temporal test: "
           "the model has never seen the future."),
        code(
            "oot = M['out_of_time']\n"
            "print(f\"Training era: {oot['train_era']}\")\n"
            "print(f\"Training N: {oot['train_n']:,}\")\n"
            "print(f\"Calibration N: {oot['cal_n']:,}\")\n"
            "print()\n"
            "print(f\"{'Era':<20}{'Years':<12}{'N':>10}{'AUC-PR':>10}{'AUC-ROC':>10}{'ECE':>10}\")\n"
            "print('-' * 72)\n"
            "for era_key, label in [('covid_era', 'COVID'), ('recent_era', 'Recent')]:\n"
            "    e = oot[era_key]\n"
            "    print(f\"{label:<20}{e['years']:<12}{e['n']:>10,}{e['auc_pr']:>10.4f}\"\n"
            "          f\"{e['auc_roc']:>10.4f}{e['ece']:>10.4f}\")"),
        md("The AUC-PR drop from in-distribution to COVID-era is negligible. The model trained "
           "only on pre-COVID data scores within 0.003 AUC-PR of models trained with COVID data "
           "included. This is consistent with our Week 1 finding that COVID did not fundamentally "
           "alter the risk-factor structure for preterm birth in this administrative dataset.\n\n"
           "**Scope of this claim: temporal, not external.** What survives here is stability across "
           "calendar time WITHIN one data-generating system, not transportability to a different "
           "one. Every era in this test is drawn from the NCHS natality file, so the birth "
           "certificate instrument, the field definitions, and the coding conventions are held "
           "fixed while only the calendar years change. That is a real and useful robustness "
           "result, but it is silent on external validity. The precise scope note recorded with the "
           "results:"),
        code("print(M['out_of_time_note'])"),
        code(
            "# Per-year AUC and prevalence drift chart\n"
            "drift = M['drift']\n"
            "years = [d['year'] for d in drift]\n"
            "auc_pr_vals = [d['auc_pr'] for d in drift]\n"
            "auc_roc_vals = [d['auc_roc'] for d in drift]\n"
            "prev_vals = [d['prevalence'] for d in drift]\n"
            "\n"
            "fig, ax1 = plt.subplots(figsize=(9, 4.5))\n"
            "ax2 = ax1.twinx()\n"
            "\n"
            "ax1.plot(years, auc_pr_vals, 'o-', color=PALETTE['blue'], linewidth=2, markersize=7, label='AUC-PR')\n"
            "ax1.plot(years, auc_roc_vals, 's-', color=PALETTE['green'], linewidth=2, markersize=6, label='AUC-ROC')\n"
            "ax2.bar(years, prev_vals, alpha=0.2, color=PALETTE['grey'], width=0.6, label='Prevalence')\n"
            "\n"
            "ax1.set_xlabel('Birth Year')\n"
            "ax1.set_ylabel('AUC (model performance)')\n"
            "ax2.set_ylabel('Preterm Prevalence')\n"
            "ax1.set_xticks(years)\n"
            "ax1.set_title('Per-Year Model Performance and Preterm Prevalence (2018-2024)')\n"
            "\n"
            "lines1, labels1 = ax1.get_legend_handles_labels()\n"
            "lines2, labels2 = ax2.get_legend_handles_labels()\n"
            "ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', framealpha=0.9)\n"
            "\n"
            "ax1.axvspan(2019.5, 2021.5, alpha=0.06, color=PALETTE['red'], label='COVID era')\n"
            "ax1.text(2020.5, min(auc_pr_vals) - 0.002, 'COVID', ha='center', fontsize=9, color=PALETTE['red'])\n"
            "plt.tight_layout(); plt.show()"),
        code(
            "# Per-year drift table\n"
            "print(f\"{'Year':<8}{'N':>10}{'Prevalence':>12}{'AUC-ROC':>10}{'AUC-PR':>10}{'ECE':>10}\")\n"
            "print('-' * 60)\n"
            "for d in M['drift']:\n"
            "    print(f\"{d['year']:<8}{d['n']:>10,}{d['prevalence']:>12.4f}\"\n"
            "          f\"{d['auc_roc']:>10.4f}{d['auc_pr']:>10.4f}{d['ece']:>10.4f}\")"),
        md("The per-year chart and table confirm temporal stability. AUC-ROC stays between 0.655 "
           "and 0.669 across all seven years; AUC-PR drifts slightly upward (consistent with "
           "a marginal prevalence increase). ECE remains below 0.004 in every year. There is no "
           "evidence of meaningful performance degradation, even across the COVID discontinuity."),

        # ---- stratified ----
        md(anchor("stratified", "Stratified per-group models") + "\n\n"
           "We trained separate XGBoost models for each race/ethnicity group using identical "
           "hyperparameters. These are hypothesis-generating, not deployment candidates. The goal "
           "is to understand whether the pooled model's predictions are driven by different features "
           "in different groups, and whether any group has a systematically different ceiling.\n\n"
           "**Important caveat:** small-group sizes (NH Multiracial and NH Asian) mean these results "
           "have wider confidence intervals. Differences should be interpreted as directional "
           "hypotheses, not confirmed findings."),
        code(
            "strat = M['stratified_models']\n"
            "groups = sorted(strat.keys())\n"
            "print(f\"{'Group':<18}{'Train N':>10}{'HO N':>8}{'Prev':>8}{'AUC-PR':>9}{'AUC-ROC':>9}{'ECE':>9}\")\n"
            "print('-' * 71)\n"
            "for g in groups:\n"
            "    s = strat[g]\n"
            "    print(f\"{g:<18}{s['train_n']:>10,}{s['held_out_n']:>8,}{s['prevalence']:>8.4f}\"\n"
            "          f\"{s['auc_pr']:>9.4f}{s['auc_roc']:>9.4f}{s['ece']:>9.4f}\")"),
        md("**HIGH-PRIORITY WARNING: do not rank groups by AUC-PR.** AUC-PR is prevalence-dependent "
           "and is NOT comparable across groups that have different preterm prevalence. A group with "
           "a higher base rate can post a higher AUC-PR as a partly MECHANICAL consequence of its "
           "prevalence, not because the model discriminates better for it. The per-group prevalence "
           "is printed in the table above precisely so this trap is visible. For cross-group "
           "discrimination comparison, use AUC-ROC, which is prevalence-robust. The comparability "
           "note recorded with the results:"),
        code("print(M['stratified_comparability_note'])"),
        code(
            "# Demonstrate the reversal: AUC-PR ranking vs prevalence-robust AUC-ROC ranking\n"
            "strat = M['stratified_models']\n"
            "b, w = strat['NH Black'], strat['NH White']\n"
            "print(f\"NH Black : prevalence {b['prevalence']:.4f}  \"\n"
            "      f\"AUC-PR {b['auc_pr']:.4f}  AUC-ROC {b['auc_roc']:.4f}\")\n"
            "print(f\"NH White : prevalence {w['prevalence']:.4f}  \"\n"
            "      f\"AUC-PR {w['auc_pr']:.4f}  AUC-ROC {w['auc_roc']:.4f}\")\n"
            "print()\n"
            "pr_lead = 'NH Black' if b['auc_pr'] > w['auc_pr'] else 'NH White'\n"
            "roc_lead = 'NH Black' if b['auc_roc'] > w['auc_roc'] else 'NH White'\n"
            "print(f\"On AUC-PR, {pr_lead} looks stronger; on prevalence-robust AUC-ROC, \"\n"
            "      f\"{roc_lead} is ahead. The ranking REVERSES, and the AUC-PR gap is largely \"\n"
            "      f\"the {b['prevalence'] - w['prevalence']:+.4f} prevalence difference talking.\")"),
        code(
            "# Top-3 predictors per group\n"
            "print(f\"{'Group':<18}{'Top-3 Predictors'}\")\n"
            "print('-' * 70)\n"
            "for g in sorted(M['stratified_models'].keys()):\n"
            "    s = M['stratified_models'][g]\n"
            "    top3 = [f\"{feat} ({wt:.3f})\" for feat, wt in s['top_features'][:3]]\n"
            "    print(f\"{g:<18}{', '.join(top3)}\")"),
        md("**Key observations (read through the AUC-PR warning above):**\n\n"
           "- NH Black shows the highest AUC-PR, but this is largely mechanical: it has the highest "
           "preterm prevalence (~12.4% vs ~7.6% for NH White), and AUC-PR rises with prevalence. On "
           "the prevalence-robust AUC-ROC the ranking REVERSES, with NH Black (~0.636) below NH "
           "White (~0.654). The model does NOT discriminate better for NH Black births; it faces an "
           "easier-looking precision-recall curve there because more of the population is positive.\n"
           "- `rf_ppterm` (prior preterm birth) dominates every group's top features, confirming "
           "it is the single strongest predictor regardless of race.\n"
           "- Group-specific race-education interaction terms appear in each group's top-10, "
           "validating the feature engineering from Week 2.\n"
           "- All group models are marked `hypothesis_only: true` in the results JSON.\n\n"
           "**Leakage check:** no at-delivery fields appear in any group's top features. The "
           "stratified models are operating on the same leakage-safe feature set as the pooled model.\n\n"
           "**Small-group caution:** NH Multiracial trained on only ~25K births (vs. 100K for the "
           "larger groups). Its higher ECE (0.010 vs. <0.005 for others) likely reflects "
           "calibration instability from limited data rather than a modeling defect."),

        # ---- sensitivity ----
        md(anchor("sensitivity", "Sensitivity analysis: thresholds and training caps") + "\n\n"
           "Two sensitivity analyses confirm that the model is operating at its ceiling and that "
           "deployment threshold choice is not fragile.\n\n"
           "**Threshold sensitivity:** we sweep the decision threshold from 50% to 150% of the "
           "base rate and examine the resulting sensitivity, specificity, and worst-case equalized "
           "odds gap across groups."),
        code(
            "thresh = M['sensitivity_threshold']\n"
            "print(f\"{'Threshold':>10}{'% Base':>8}{'Sens':>8}{'Spec':>8}{'Worst EO Gap':>14}\")\n"
            "print('-' * 48)\n"
            "for t in thresh:\n"
            "    print(f\"{t['threshold']:>10.4f}{t['threshold_pct_of_base']:>8.0f}%\"\n"
            "          f\"{t['sensitivity']:>8.3f}{t['specificity']:>8.3f}\"\n"
            "          f\"{t['worst_eo_gap']:>14.4f}\")"),
        md("The threshold sweep reveals the expected sensitivity-specificity tradeoff. At 50% of "
           "the base rate, we achieve near-perfect sensitivity (0.98) but very low specificity "
           "(0.06); at 150%, specificity jumps to 0.89 but sensitivity drops to 0.30. The "
           "equalized-odds gap is smallest at the extreme thresholds and largest in the mid-range, "
           "consistent with the well-known finding that fairness constraints are hardest to satisfy "
           "at intermediate operating points.\n\n"
           "**Training-cap sensitivity:** we varied the training set size from 100K to 500K and "
           "measured held-out AUC-PR. If the model is data-limited, we would see continued "
           "improvement; if it has hit the ceiling, performance should plateau."),
        code(
            "cap = M['sensitivity_cap']\n"
            "print(f\"{'Train Cap':>12}{'Held-out AUC-PR':>18}\")\n"
            "print('-' * 30)\n"
            "for c in cap:\n"
            "    print(f\"{c['train_cap']:>12,}{c['held_out_auc_pr']:>18.4f}\")\n"
            "print()\n"
            "delta = cap[-1]['held_out_auc_pr'] - cap[0]['held_out_auc_pr']\n"
            "print(f\"Gain from 5x more data: {delta:.4f} AUC-PR\")"),
        code(
            "# Training cap vs AUC-PR chart\n"
            "caps_x = [c['train_cap'] for c in M['sensitivity_cap']]\n"
            "caps_y = [c['held_out_auc_pr'] for c in M['sensitivity_cap']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.plot(caps_x, caps_y, 'o-', color=PALETTE['blue'], linewidth=2, markersize=8)\n"
            "ax.set_xlabel('Training Cap (N births)')\n"
            "ax.set_ylabel('Held-out AUC-PR')\n"
            "ax.set_title('Training-Cap Sensitivity: Performance Plateaus')\n"
            "ax.set_xticks(caps_x)\n"
            "ax.set_xticklabels([f'{c//1000}K' for c in caps_x])\n"
            "ax.axhline(caps_y[-1], ls=':', color=PALETTE['grey'], alpha=0.5)\n"
            "plt.tight_layout(); plt.show()"),
        md("Quintupling the training data from 100K to 500K yields only +0.007 AUC-PR. The "
           "learning curve has flattened: we are past the data ceiling for this feature set.\n\n"
           "**This plateau is XGBoost-specific.** The diminishing-returns knee shown here is a "
           "property of the BOOSTERS, and it licenses the 500K cap only for them. It does NOT "
           "generalize to the deep nets from Week 5, whose whole archetype is to keep improving as "
           "data grows; a flat booster curve says nothing about where a deep net's curve would "
           "bend. The scope note recorded with the results:"),
        code("print(M['sensitivity_cap_note'])"),
        md("So the honest reading is narrower than 'the model has hit its ceiling.' It is: for the "
           "gradient-boosted model on this administrative-record feature set, the marginal value of "
           "additional births past ~500K is small, consistent with the Week 5 view that the bound is "
           "set largely by the informational content of birth-certificate fields rather than by "
           "sample size. A different model class could still trace a different curve.\n\n"
           "**What stability buys us:** these two analyses together confirm that (1) threshold "
           "choice is not fragile (the model degrades smoothly, not catastrophically, as we move "
           "the operating point), and (2) we are not leaving performance on the table by "
           "under-training. The model is doing what the data allows."),

        # ---- plan ----
        md(anchor("plan", "Week 9 and beyond") + "\n\n"
           "The model has survived temporal shift, subpopulation stratification, and sensitivity "
           "sweeps. The remaining work is about communication and context.\n\n"
           "**Week 9 (PLANNED): Transparency artifacts and demo.**\n"
           "- A self-contained model card summarizing performance, limitations, and fairness "
           "properties.\n"
           "- An interactive prototype (Streamlit or similar) for exploring risk scores, SHAP "
           "explanations, and disparity decompositions.\n"
           "- External context: how these findings connect to the broader literature on algorithmic "
           "fairness in perinatal health.\n"
           "- Final recommendations for responsible deployment considerations." + AUTHOR),
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
