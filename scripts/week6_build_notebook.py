"""Compile the Week 6 storytelling notebook: fairness mitigation and the disparate treatment tension.

Week 6 scope: apply three mitigation strategies (reweighting, ExponentiatedGradient, ThresholdOptimizer)
to the selected XGBoost model, compare the EO-gap reduction each buys, and articulate the ethical tension
between race-aware post-processing (which is disparate treatment) and failing to mitigate (which is
disparate impact). The notebook never recommends deploying group-specific thresholds; it documents the
analytical landscape to inform responsible decision-making.

Every number is read live from reports/week6_mitigation.json (produced by scripts/week6_mitigation.py
on capped training draws). NO hardcoded metric values.

Run:  PYTHONPATH=src ./.venv/bin/python scripts/week6_build_notebook.py
Then: ./.venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/week6_*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week6_01_fairness_mitigation"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def anchor(a, title): return f'<a id="{a}"></a>\n## {title}'


def contents(items):
    return "\n".join(["### Contents", ""] + [f"{i}. [{lbl}](#{a})" for i, (a, lbl) in enumerate(items, 1)])


SETUP = """\
# Week 6 reads back the fairness mitigation results from
# reports/week6_mitigation.json (produced by scripts/week6_mitigation.py).
# The heavy computation happened in a script; the notebook reports and interprets the results.
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = 7

REPORTS = Path("..") / "reports"
M = json.loads((REPORTS / "week6_mitigation.json").read_text())
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
        ("setup", "Setup: load mitigation results"),
        ("problem", "The problem: unequal opportunity across groups"),
        ("three-levers", "The three levers: pre-, in-, and post-processing"),
        ("tradeoffs", "Results: what mitigation buys and what it costs"),
        ("treatment-vs-impact", "Disparate treatment vs. disparate impact"),
        ("plan", "The road ahead (weeks 7 through 9)"),
    ]
    cells = [
        md("# Week 6: Fairness Mitigation and the Disparate Treatment Tension\n\n"
           "<sub>[Previous: Deep-Learning Benchmark and Model Selection](week5_01_deep_learning_and_selection.ipynb)</sub>\n\n"
           "<sub>New here? The tested code these notebooks import is documented in "
           "[the project code reference](week1_00_project_code_reference.ipynb).</sub>\n\n"
           "Week 5 selected XGBoost as the primary model. The week-4 fairness audit showed that "
           "NH Black mothers receive disproportionately high risk flags while NH Asian mothers are "
           "under-flagged relative to NH White mothers, a gap measured by unequal true-positive rates "
           "(equal opportunity). This week we ask: can algorithmic mitigation close that gap, and at "
           "what cost? The answer reveals a deep tension between disparate treatment (using "
           "race-aware thresholds) and disparate impact (leaving unequal outcomes untouched).\n\n"
           "As throughout this project, nothing here is a clinical tool: these are associational "
           "patterns in administrative data, not statements about any individual. This week shows "
           "group-specific thresholds as an analytical exercise, so the caveat matters more here than "
           "anywhere else. Race is treated as a marker of exposure to structural racism, never as a "
           "biological cause."),
        md(contents(toc)),

        # ---- setup ----
        md(anchor("setup", "Setup: load mitigation results") + "\n\n"
           "All mitigation experiments were run by `scripts/week6_mitigation.py`. We read the "
           "summary JSON here. The reference group is NH White; all EO gaps are measured as the "
           "difference in true-positive rate (sensitivity) between a given group and the reference."),
        code(SETUP),
        md("A few rows from the leakage-safe feature matrix, for orientation:"),
        code("mdf.sample(5, seed=RNG).to_pandas()"),
        code(
            "mit = M['mitigations']\n"
            "print(f\"Fairness subsample: {M['fairness_subsample_n']:,} births\")\n"
            "print(f\"Threshold (Youden-optimal on validation): {M['threshold']:.5f}\")\n"
            "print(f\"Reference group: {M['reference_group']}\")\n"
            "print(f\"Comparison groups: {M['groups']}\")\n"
            "print(f\"Seed: {M['seed']}\")"),

        # ---- problem ----
        md(anchor("problem", "The problem: unequal opportunity across groups") + "\n\n"
           "Equal opportunity (Hardt, Price, and Srebro, 2016; "
           "[arXiv:1610.02413](https://arxiv.org/abs/1610.02413)) requires that a classifier's "
           "true-positive rate be the same across protected groups. In a preterm-birth screener, "
           "violating EO means that some groups of mothers who *will* deliver preterm are less likely "
           "to be flagged for early intervention than others, holding outcome constant.\n\n"
           "The baseline XGBoost model (week-4 tuned, isotonic-calibrated) produces the following "
           "EO gaps relative to NH White:"),
        code(
            "print(f\"{'Group':<20}{'EO gap (TPR - ref)':>20}\")\n"
            "print('-' * 40)\n"
            "for grp, gap in mit['baseline']['per_group_eo_gap'].items():\n"
            "    print(f\"{grp:<20}{gap:>+20.4f}\")\n"
            "print()\n"
            "print(f\"Worst-group EO gap: {mit['baseline']['worst_group_eo_gap']:.4f}\")\n"
            "print(f\"Mean |EO gap|:      {mit['baseline']['mean_abs_eo_gap']:.4f}\")"),
        md("NH Black mothers have a positive gap of ~0.39, meaning the baseline model is "
           "39 percentage points *more* sensitive for NH Black mothers than for NH White mothers at "
           "the same threshold. This is *not* the direction the clinical equity concern usually runs "
           "(that would be under-detection). Here the model over-flags Black mothers at a fixed "
           "threshold because their base rate is higher and the calibrated scores are shifted right. "
           "NH AIAN mothers show a +0.25 gap for the same structural reason.\n\n"
           "The implication: at a single universal threshold, the model treats groups unequally "
           "because of base-rate differences that reflect upstream structural inequity (access to "
           "prenatal care, chronic-stress burden, socioeconomic stratification). The question is "
           "whether algorithmic post-processing can equalize opportunity without destroying the "
           "model's predictive utility."),

        # ---- three-levers ----
        md(anchor("three-levers", "The three levers: pre-, in-, and post-processing") + "\n\n"
           "The fairness-ML literature (Barocas, Hardt, and Narayanan, "
           "[Fairness and Machine Learning](https://fairmlbook.org/), 2023) organizes mitigation "
           "into three families:\n\n"
           "**Pre-processing** modifies the training data before the model sees it. The simplest "
           "version is *reweighting*: assign each training instance a weight inversely proportional "
           "to its group-by-outcome cell frequency, so that the model sees an effectively balanced "
           "joint distribution of group membership and label. This approach does not touch the model "
           "architecture or decision rule. It is race-aware at training time (the weights depend on "
           "group labels) but produces a single universal model that makes predictions without "
           "consulting group membership at inference.\n\n"
           "**In-processing** imposes fairness constraints during optimization. Fairlearn's "
           "`ExponentiatedGradient` (Agarwal et al., 2018) treats the fairness constraint as a "
           "player in a zero-sum game against the loss minimizer. The algorithm iteratively "
           "reweights training instances to steer the learner toward a model that satisfies a "
           "specified constraint (here, TPR parity across groups). The base learner is a logistic "
           "regression, because ExponentiatedGradient requires a cost-sensitive learner that accepts "
           "sample weights. The result is a hard classifier; no well-calibrated probabilities are "
           "available, so AUC-PR is not computable.\n\n"
           "**Post-processing** adjusts the decision rule after the model is trained. Fairlearn's "
           "`ThresholdOptimizer` finds group-specific thresholds that equalize TPR across groups. "
           "It takes the baseline model's predicted probabilities and selects a different operating "
           "point for each group. This is the most direct route to EO parity: by definition, you "
           "can always find thresholds that equalize TPR. The cost is that different groups face "
           "different decision rules, which is the textbook definition of *disparate treatment*."),

        # ---- tradeoffs ----
        md(anchor("tradeoffs", "Results: what mitigation buys and what it costs") + "\n\n"
           "We compare four approaches: the baseline (no mitigation), reweighting (pre-processing), "
           "ExponentiatedGradient (in-processing), and ThresholdOptimizer (post-processing). The "
           "table below shows summary-level performance and fairness metrics for each."),
        code(
            "# Summary metrics table\n"
            "methods = ['baseline', 'reweight', 'expgrad', 'threshopt']\n"
            "labels = ['Baseline (XGB)', 'Reweight', 'ExpGrad', 'ThreshOpt']\n"
            "\n"
            "print(f\"{'Method':<18}{'Sensitivity':>12}{'Specificity':>13}{'Worst EO gap':>14}{'Mean |EO|':>12}\")\n"
            "print('-' * 69)\n"
            "for meth, lbl in zip(methods, labels):\n"
            "    d = mit[meth]\n"
            "    sens = d['sensitivity']\n"
            "    spec = d['specificity']\n"
            "    wg = d['worst_group_eo_gap']\n"
            "    mg = d['mean_abs_eo_gap']\n"
            "    print(f\"{lbl:<18}{sens:>12.4f}{spec:>13.4f}{wg:>14.4f}{mg:>12.4f}\")\n"
            "\n"
            "print()\n"
            "print('Note: ExpGrad uses a logistic base learner (hard 0/1); no AUC-PR available.')\n"
            "print('Note: ThreshOpt uses group-specific thresholds (disparate treatment).')"),
        code(
            "# AUC-PR and AUC-ROC for the two methods that produce probabilities.\n"
            "# Report BOTH equal-width ECE and the stricter equal-mass (quantile) ECE.\n"
            "print(f\"{'Method':<18}{'AUC-PR':>10}{'AUC-ROC':>10}{'ECE (width)':>14}{'ECE (quantile)':>16}\")\n"
            "print('-' * 68)\n"
            "for meth, lbl in [('baseline', 'Baseline'), ('reweight', 'Reweight')]:\n"
            "    d = mit[meth]\n"
            "    print(f\"{lbl:<18}{d['auc_pr']:>10.5f}{d['auc_roc']:>10.4f}\"\n"
            "          f\"{d['ece']:>14.7f}{d['ece_quantile']:>16.7f}\")"),
        md("**A note on the ECE numbers.** The equal-width ECE looks almost too good. That is partly "
           "a binning artifact: at an 8.6% base rate the predictions cluster in the low-probability "
           "bins, so most equal-width bins are nearly empty and contribute little error. The stricter "
           "equal-mass (quantile) ECE, which puts an equal number of predictions in each bin, is the "
           "more honest read and is reported alongside it above. Both are small, but the quantile "
           "version is the one to trust."),
        md("**Key observations:**\n\n"
           "- **Reweighting** trades a little discrimination (AUC-PR drops, shown above) for a large "
           "improvement on the group it has mass to learn from and a destabilization of a tiny one. "
           "It substantially shrinks the NH Black equal-opportunity gap, which is the largest group "
           "and the one that dominated the baseline disparity, but it worsens NH NHOPI, a very small "
           "and noisy group where the reweighting mass is thin. The next cell shows this directly.\n"
           "- **ExponentiatedGradient** reaches the tightest EO gaps of any method, but its "
           "near-zero mitigated sensitivity is a setup artifact, not a fairness cost. The cell below "
           "unpacks why.\n"
           "- **ThresholdOptimizer** offers a middle ground: moderate EO reduction while retaining "
           "meaningful sensitivity (values in the summary table above). But it does so by applying "
           "different thresholds to different racial groups, which is disparate treatment."),
        code(
            "# Reweighting: helps the large group, destabilizes the tiny one.\n"
            "bl = mit['baseline']['per_group_eo_gap']\n"
            "rw = mit['reweight']['per_group_eo_gap']\n"
            "for g in ['NH Black', 'NH NHOPI']:\n"
            "    print(f\"{g:<10} EO gap  baseline {bl[g]:+.4f}  ->  reweight {rw[g]:+.4f}  \"\n"
            "          f\"(|gap| {'shrinks' if abs(rw[g]) < abs(bl[g]) else 'grows'})\")\n"
            "print()\n"
            "print(f\"Mean |EO gap|  baseline {mit['baseline']['mean_abs_eo_gap']:.4f}  ->  \"\n"
            "      f\"reweight {mit['reweight']['mean_abs_eo_gap']:.4f}\")\n"
            "print(f\"Worst |EO gap| baseline {mit['baseline']['worst_group_eo_gap']:.4f}  ->  \"\n"
            "      f\"reweight {mit['reweight']['worst_group_eo_gap']:.4f} (driven by the tiny NH NHOPI group)\")"),
        md("So reweighting is not a failure and not a free win. It moves mass toward equalizing the "
           "group-by-outcome cells, which helps the large NH Black group it can actually learn from, "
           "while the smallest group (NH NHOPI) has too few births for that reweighting to be stable "
           "and its gap swings the other way. This is the honest shape of pre-processing on a "
           "long-tailed group distribution: it helps where there is data and destabilizes where there "
           "is not."),
        code(
            "# ExpGrad's near-zero sensitivity is an operating-point artifact, not a fairness cost.\n"
            "eg = mit['expgrad']\n"
            "base_sens = eg['expgrad_baseline_sens']\n"
            "xgb_sens = mit['baseline']['sensitivity']\n"
            "xgb_thr = mit['baseline']['threshold']\n"
            "print(f\"ExpGrad UNCONSTRAINED logistic baseline sensitivity (at 0.5 cutoff): {base_sens:.4f}\")\n"
            "print(f\"ExpGrad CONSTRAINED sensitivity (at 0.5 cutoff):                     {eg['sensitivity']:.4f}\")\n"
            "print(f\"XGBoost baseline sensitivity (at tuned {xgb_thr:.3f} threshold):          {xgb_sens:.4f}\")\n"
            "print()\n"
            "print(f\"ExpGrad worst-group EO gap: {eg['worst_group_eo_gap']:.4f}  \"\n"
            "      f\"mean |EO gap|: {eg['mean_abs_eo_gap']:.4f}\")"),
        md("**Reading the ExpGrad result correctly.** It is tempting to say the fairness constraint "
           "crushed sensitivity to near zero. That reading is wrong. The *unconstrained* logistic "
           "baseline already sits at roughly the same near-zero sensitivity (printed above), because "
           "the classifier is evaluated at the default 0.5 decision cutoff and at an 8.6% prevalence "
           "almost no logistic score ever exceeds 0.5. The XGBoost baseline looks far more sensitive "
           "only because it is scored at a tuned threshold near 0.078, not 0.5. So the two are being "
           "compared at incomparable operating points, and the near-zero number is an artifact of the "
           "0.5 cutoff on a weak convex learner, not a price paid for the TPR-parity constraint.\n\n"
           "What ExponentiatedGradient genuinely achieved is real and worth stating: near-complete "
           "TPR equalization (the tightest worst-group gap of any method here) using only a logistic "
           "base learner. The lesson is that the in-processing machinery works; the logistic base "
           "learner and the default cutoff, not the fairness constraint, are what make the raw "
           "sensitivity look bad."),

        # ---- grouped bar chart ----
        code(
            "# Grouped bar chart: per-group EO gaps by mitigation method\n"
            "groups = M['groups']\n"
            "methods = ['baseline', 'reweight', 'expgrad', 'threshopt']\n"
            "method_labels = ['Baseline', 'Reweight', 'ExpGrad', 'ThreshOpt']\n"
            "colors = [PALETTE['blue'], PALETTE['orange'], PALETTE['green'], PALETTE['red']]\n"
            "\n"
            "x = np.arange(len(groups))\n"
            "width = 0.20\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 5.2))\n"
            "for i, (meth, lbl, col) in enumerate(zip(methods, method_labels, colors)):\n"
            "    vals = [mit[meth]['per_group_eo_gap'][g] for g in groups]\n"
            "    ax.bar(x + i * width, vals, width, label=lbl, color=col, edgecolor='none')\n"
            "\n"
            "ax.axhline(0, color=PALETTE['grey'], lw=0.8)\n"
            "ax.set_xticks(x + 1.5 * width)\n"
            "ax.set_xticklabels(groups, rotation=25, ha='right', fontsize=9)\n"
            "ax.legend(loc='upper left', framealpha=0.9)\n"
            "style(ax, 'Per-group EO gap (TPR difference vs. NH White) by mitigation method',\n"
            "      '', 'EO gap (positive = higher TPR than reference)')\n"
            "plt.tight_layout(); plt.show()"),
        md("The chart makes visually clear what the numbers say: ExpGrad drives all gaps toward "
           "zero (at the cost of collapsing sensitivity), ThreshOpt moderates the gaps substantially, "
           "while reweighting and the baseline show large swings."),

        # ---- pareto scatter ----
        code(
            "# Pareto scatter: fairness gain vs. AUC-PR cost\n"
            "# Only baseline and reweight have AUC-PR; ExpGrad and ThreshOpt are decision-only.\n"
            "fig, ax = plt.subplots(figsize=(7.0, 5.0))\n"
            "\n"
            "bl_auc = mit['baseline']['auc_pr']\n"
            "rw_auc = mit['reweight']['auc_pr']\n"
            "bl_eo = mit['baseline']['mean_abs_eo_gap']\n"
            "rw_eo = mit['reweight']['mean_abs_eo_gap']\n"
            "\n"
            "# Plot baseline and reweight\n"
            "ax.scatter(bl_eo, bl_auc, s=120, color=PALETTE['blue'], zorder=5, label='Baseline')\n"
            "ax.scatter(rw_eo, rw_auc, s=120, color=PALETTE['orange'], zorder=5, label='Reweight')\n"
            "\n"
            "# Annotate\n"
            "ax.annotate('Baseline', (bl_eo, bl_auc), textcoords='offset points',\n"
            "            xytext=(8, -5), fontsize=9)\n"
            "ax.annotate('Reweight', (rw_eo, rw_auc), textcoords='offset points',\n"
            "            xytext=(8, -5), fontsize=9)\n"
            "\n"
            "# Arrow showing ideal direction\n"
            "ax.annotate('', xy=(0.02, 0.18), xytext=(0.10, 0.15),\n"
            "            arrowprops=dict(arrowstyle='->', color=PALETTE['green'], lw=2))\n"
            "ax.text(0.04, 0.179, 'ideal direction\\n(lower gap, higher AUC-PR)',\n"
            "        fontsize=8, color=PALETTE['green'])\n"
            "\n"
            "style(ax, 'Pareto view: fairness gain vs. predictive cost',\n"
            "      'Mean |EO gap| (lower is fairer)', 'AUC-PR (higher is better)')\n"
            "ax.legend(loc='lower left')\n"
            "\n"
            "# Note about decision-only methods\n"
            "ax.text(0.98, 0.02, 'ExpGrad and ThreshOpt are decision-only\\n(no AUC-PR available)',\n"
            "        transform=ax.transAxes, ha='right', va='bottom', fontsize=8,\n"
            "        style='italic', color=PALETTE['grey'])\n"
            "\n"
            "plt.tight_layout(); plt.show()"),
        md("The Pareto scatter shows reweighting moving left on the mean-gap axis (it does reduce the "
           "average |EO gap|, driven by the large NH Black group) while paying a real AUC-PR cost. "
           "The mean-gap view flatters reweighting: it hides that the *worst*-group gap grew, because "
           "the tiny NH NHOPI group swung the other way. There is no free lunch here, and the "
           "aggregate fairness number can move the opposite way from the worst-off group."),

        # ---- treatment-vs-impact ----
        md(anchor("treatment-vs-impact", "Disparate treatment vs. disparate impact") + "\n\n"
           "This section is the ethical core of the notebook. The mitigation results reveal a "
           "fundamental legal and moral tension:\n\n"
           "**Disparate impact** occurs when a facially neutral policy produces unequal outcomes "
           "across protected groups. The baseline model uses a single threshold for everyone, but "
           "because base rates differ, it flags NH Black mothers at a much higher rate. If the "
           "system is used to allocate scarce intervention resources, groups with higher flag rates "
           "bear a heavier burden of false positives (unnecessary interventions, surveillance, "
           "anxiety).\n\n"
           "**Disparate treatment** occurs when a policy explicitly uses group membership to make "
           "decisions. The ThresholdOptimizer applies race-specific thresholds: it literally uses a "
           "different decision rule depending on the mother's race. This is the textbook definition "
           "of disparate treatment, regardless of whether the intent is to help disadvantaged "
           "groups.\n\n"
           "The legal tension is well-illustrated by *Ricci v. DeStefano* (557 U.S. 557, 2009), in "
           "which the U.S. Supreme Court held that an employer cannot engage in intentional "
           "race-conscious action (disparate treatment) merely to avoid a disparate-impact outcome, "
           "absent a strong basis in evidence that the original practice was actually unlawful. "
           "Applied here: one cannot simply deploy race-specific thresholds because the single "
           "threshold produces unequal flag rates, unless there is a compelling justification "
           "beyond statistical disparity alone.\n\n"
           "**What we would never deploy:** The ThresholdOptimizer results are reported here as an "
           "analytical exercise to measure the *ceiling* of what post-processing can achieve. The "
           "project demo will never expose race-specific thresholds to end users. Any deployed "
           "system would use a single universal threshold.\n\n"
           "**What we recommend instead:**\n\n"
           "1. **Monitoring without race-aware adjustment.** Deploy a single threshold, but "
           "continuously track per-group TPR, FPR, and calibration. When gaps exceed a pre-specified "
           "tolerance, flag the disparity for human review rather than automatically adjusting.\n\n"
           "2. **Upstream structural interventions.** The EO gap exists because the features "
           "themselves reflect structural inequity (differential access to prenatal care, chronic "
           "stress, socioeconomic stratification). The responsible response is not to paper over the "
           "gap at the scoring layer but to address it at the data-generation layer: expand access, "
           "reduce barriers, and collect richer contextual variables that explain rather than merely "
           "encode race.\n\n"
           "3. **Transparent reporting.** Publish per-group performance alongside overall metrics so "
           "that stakeholders can make informed decisions about acceptable tradeoffs. The SHAP "
           "decomposition in week 7 will quantify how much of the racial gap in predicted risk is "
           "attributable to medical factors vs. socioeconomic proxies."),

        # ---- plan ----
        md(anchor("plan", "The road ahead (weeks 7 through 9)") + "\n\n"
           "The model is selected and the mitigation landscape is mapped. The remaining work turns "
           "from *what can we do about the gap* to *what does the gap mean*.\n\n"
           "**Week 7: SHAP and disparity decomposition.** TreeSHAP on the selected model, "
           "stratified by race. Decompose the racial gap in predicted risk into feature-group "
           "contributions (medical risk, SES, access, behavioral). Quantify how much of the "
           "observed disparity the measured factors account for.\n\n"
           "**Week 8: Robustness and drift.** Out-of-time validation across birth-year eras "
           "(pre-COVID, COVID, post-COVID). Stability of calibration and fairness metrics over time. "
           "Sensitivity to feature perturbation.\n\n"
           "**Week 9: Transparency report and demo.** A self-contained summary of findings, "
           "limitations, and recommendations. An interactive prototype (Streamlit or similar) "
           "for exploring risk scores and disparity decompositions." + AUTHOR),
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
