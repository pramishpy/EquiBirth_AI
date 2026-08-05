"""Compile the Week 3 storytelling notebook: calibration-first baseline models.

Week 3 scope: train the first models on the leakage-safe feature matrix from Week 2, and report them
honestly. Three baselines (logistic regression as the calibration reference, a decision tree, and a
random forest), judged on rate-aware metrics and calibration rather than accuracy, because the outcome
is rare. The headline is deliberately modest: individual prediction from birth-certificate features
tops out around AUC-ROC 0.65, squarely in the published administrative-data range. That honesty is the
point, and it is what the leakage discipline of Week 2 makes trustworthy.

Every number is read live from reports/week3_baseline_metrics.json (produced by
scripts/week3_baselines.py on the full-data working set, held-out 1.5M singleton births) plus small
live computations on the modeling parquet. Gradient boosting, the fairness audit, and the gap
decomposition are written up as the PLAN for the coming weeks, not as work already done.

Run:  PYTHONPATH=src ./.venv/bin/python scripts/week3_build_notebook.py
Then: ./.venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/week3_*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week3_01_calibrated_baselines"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def anchor(a, title): return f'<a id="{a}"></a>\n## {title}'


def contents(items):
    return "\n".join(["### Contents", ""] + [f"{i}. [{lbl}](#{a})" for i, (a, lbl) in enumerate(items, 1)])


SETUP = """\
# Week 3 reports models, so we mostly read back the results our tested pipeline already computed on the
# full data, rather than retraining in the notebook (training 500K rows belongs in a script, not a
# cell). The metrics come from reports/week3_baseline_metrics.json, written by scripts/week3_baselines.py
# which trains on the leakage-safe matrix from Week 2 and evaluates on the untouched 1.5M-row held-out
# partition. We do a little live computation on the modeling parquet for the intuition plots.
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
M = json.loads((REPORTS / "week3_baseline_metrics.json").read_text())
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

# Self-contained: load a slim projection of the modeling working set so the notebook stands alone
# and can show real rows. We read only a few columns to stay light.
SAMPLE_COLS = ["preterm", "split", "mother_race_ethnicity", "mager", "meduc", "bmi",
               "medical_risk_count", "rf_ppterm", "first_trimester_care"]
mdf = pl.read_parquet(DATA, columns=SAMPLE_COLS)
"""


def build():
    toc = [
        ("setup", "Setup: read back the verified results"),
        ("accuracy", "Why accuracy is the wrong metric"),
        ("ceiling", "The honest performance ceiling"),
        ("calibration", "Calibration: the metric we actually care about"),
        ("byrace", "Does calibration hold across race groups?"),
        ("plan", "The plan from here"),
    ]
    cells = [
        md("# Week 3: Calibration-First Baseline Models\n\n"
           "<sub>[Previous: Feature Engineering and the Leakage Audit](week2_01_features_and_leakage.ipynb)</sub>\n\n"
           "<sub>New here? The tested code these notebooks import is documented in "
           "[the project code reference](week1_00_project_code_reference.ipynb).</sub>\n\n"
           "Week 2 gave us a leakage-safe feature matrix and a warning: the legitimate signals are weak, "
           "so we should expect a modest model. This week we train the first models and find out exactly "
           "how modest. The discipline of the week is to judge them the right way. With a rare outcome, "
           "accuracy is worse than useless, so we lead with rate-aware metrics and, above all, "
           "calibration, because the whole equity analysis later depends on the risk scores meaning what "
           "they say."),
        md(contents(toc)),
        md(anchor("setup", "Setup: read back the verified results") + "\n\n"
           "The models were trained by a tested script (`scripts/week3_baselines.py`) on the leakage-safe "
           "matrix, and evaluated on the untouched held-out partition. We read those results back here "
           "rather than retraining in a cell, which keeps the notebook fast and the numbers reproducible."),
        code(SETUP),
        md("Each notebook stands on its own, so here are real rows of the leakage-safe pre-delivery "
           "feature matrix (built in Week 2) that the models train on. The `split` column marks the "
           "deterministic train/val/test/held-out partitions, so every row is tagged with the fold it "
           "belongs to and nothing leaks across them."),
        code("mdf.sample(5, seed=RNG).to_pandas()"),
        code(
            "print(f\"Held-out evaluation set: {M['held_out_n']:,} singleton births, \"\n"
            "      f\"prevalence {M['prevalence']*100:.2f}% preterm, {M['n_features']} pre-delivery features.\")\n"
            "print('Models trained (each with balanced class weights, isotonic-recalibrated):')\n"
            "for name in M['models']: print('  -', name)"),
        # ---- accuracy ----
        md(anchor("accuracy", "Why accuracy is the wrong metric") + "\n\n"
           "Before any model, one number settles the choice of metric. The outcome is about 8.6% "
           "positive, so a model that simply predicts 'not preterm' for everyone is right about 91.4% of "
           "the time. Let me state that plainly, because it is the trap a lot of health-ML projects fall "
           "into."),
        code(
            "base = M['prevalence']\n"
            "print(f'Preterm prevalence: {base*100:.1f}%')\n"
            "print(f'\"Always predict not-preterm\" accuracy: {(1-base)*100:.1f}%')\n"
            "print('...which catches ZERO real preterm births. Accuracy rewards the useless model here.')"),
        md("So accuracy is off the table. We report **AUC-PR** (the area under the precision-recall "
           "curve, the right summary when positives are rare), **AUC-ROC** (which is base-rate "
           "invariant, so it is the fair thing to compare across race groups), and calibration. Not "
           "accuracy."),
        md("Here is why the two AUC metrics play different roles. AUC-PR has a floor that equals the "
           "prevalence, so under this much imbalance it is the honest primary metric: a useless model "
           "cannot hide behind a high number. AUC-ROC has a fixed 0.5 floor regardless of the base rate, "
           "so it is base-rate invariant, which is exactly what makes it the fair comparator across race "
           "groups whose prevalence differs."),
        code(
            "base = M['prevalence']\n"
            "print(f\"Prevalence (positive base rate): {base*100:.2f}%\")\n"
            "print(f\"A random classifier's AUC-PR floor equals prevalence ({base:.3f}); AUC-ROC floor is 0.5.\")\n"
            "print('AUC-PR is the honest primary metric under this imbalance; AUC-ROC is base-rate invariant, so')\n"
            "print('it is the fair thing to compare ACROSS race groups with different prevalence.')"),
        # ---- ceiling ----
        md(anchor("ceiling", "The honest performance ceiling") + "\n\n"
           "Here are the three baselines on the held-out set. I am watching AUC-ROC for the headline "
           "(how much signal is really here) and AUC-PR as the primary rare-event metric."),
        code(
            "rows = [(n, m['auc_roc'], m['auc_pr'], m['auc_pr_ci']) for n, m in M['models'].items()]\n"
            "print(f\"{'model':<22}{'AUC-ROC':>9}{'AUC-PR':>9}   95% CI (AUC-PR)\")\n"
            "for n, roc, pr, ci in rows:\n"
            "    print(f'{n:<22}{roc:>9.3f}{pr:>9.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]')\n"
            "best = max(M['models'], key=lambda k: M['models'][k]['auc_roc'])\n"
            "print(f'\\nBest baseline: {best} at AUC-ROC {M[\"models\"][best][\"auc_roc\"]:.3f}')"),
        code(
            "names = list(M['models']); rocs = [M['models'][n]['auc_roc'] for n in names]\n"
            "fig, ax = plt.subplots(figsize=(7.2, 3.6))\n"
            "ax.bar(names, rocs, color=[PALETTE['grey'], PALETTE['grey'], PALETTE['blue']])\n"
            "for i, v in enumerate(rocs): ax.text(i, v+0.004, f'{v:.3f}', ha='center', fontweight='bold')\n"
            "ax.axhspan(0.62, 0.70, color=PALETTE['green'], alpha=0.12)\n"
            "ax.axhline(0.5, ls=':', color=PALETTE['red'])\n"
            "ax.text(0.02, 0.505, 'chance (0.5)', color=PALETTE['red'], fontsize=8, transform=ax.get_yaxis_transform())\n"
            "style(ax, 'All three baselines land in the published 0.62-0.70 admin-data band', '', 'AUC-ROC (held-out)')\n"
            "ax.set_ylim(0.5, 0.72); plt.xticks(rotation=10); plt.tight_layout(); plt.show()"),
        md("**About 0.65, and I am glad it is not higher.** The best baseline (a random forest) reaches "
           "AUC-ROC around 0.65, with logistic regression close behind. The green band is the range that "
           "peer-reviewed models built from administrative or birth-certificate data typically report "
           "(roughly 0.62 to 0.70); richer clinical or biomarker data goes higher, but we do not have "
           "that, and honesty about it matters. This is the payoff of Week 2: because we barred the "
           "at-delivery fields, we get an honest 0.65 instead of a fraudulent 0.92. A number in this band "
           "is exactly what a trustworthy birth-certificate model should look like."),
        # ---- calibration ----
        md(anchor("calibration", "Calibration: the metric we actually care about") + "\n\n"
           "Discrimination (AUC) is only half the story. For an equity audit, what matters even more is "
           "**calibration**: when the model says 10% risk, does the real rate turn out to be about 10%? "
           "If the scores are not calibrated, every fairness comparison built on them is meaningless. So "
           "we recalibrate each model (isotonic regression, fit on a separate slice the model never "
           "trained on) and measure the calibration error before and after."),
        code(
            "print(f\"{'model':<22}{'ECE raw':>9}{'ECE calibrated':>16}\")\n"
            "for n, m in M['models'].items():\n"
            "    print(f'{n:<22}{m[\"ece_raw\"]:>9.4f}{m[\"ece_cal\"]:>16.4f}')\n"
            "print('\\nECE = expected calibration error (0 is perfect). Isotonic recalibration on a held-aside')\n"
            "print('slice pulls a raw ECE near 0.39 down to about 0.001.')"),
        md("The reliability diagram makes it visible: after recalibration the model tracks the diagonal "
           "(predicted risk equals observed rate). We draw it inline from the reliability bins the script "
           "serialized into the metrics JSON, so the notebook stands alone without any external image. "
           "Marker size reflects how many held-out births land in each bin, so the sparse high-risk bins "
           "(small n, in the printed table below) should not be over-read; almost all the mass sits in "
           "the low-risk bins near the base rate."),
        code(
            "cur = M[\"reliability_rf\"]\n"
            "xp = [b[\"mean_pred\"] for b in cur]; yo = [b[\"observed\"] for b in cur]\n"
            "print(\"Reliability bins (mean predicted vs observed), random forest, held-out:\")\n"
            "for b in cur:\n"
            "    print(f\"  pred {b['mean_pred']:.3f} -> observed {b['observed']:.3f}  (n={b['count']:,})\")\n"
            "cnts = np.array([b[\"count\"] for b in cur], dtype=float)\n"
            "sizes = 30 + 300 * (cnts / cnts.max())   # marker area scaled by bin mass\n"
            "fig, ax = plt.subplots(figsize=(5.6, 5.2))\n"
            "ax.plot([0,1],[0,1],'--',color=PALETTE['grey'],label='perfect calibration')\n"
            "ax.plot(xp, yo, color=PALETTE['blue'], linewidth=1.2, alpha=0.6)   # faint connecting line\n"
            "ax.scatter(xp, yo, s=sizes, color=PALETTE['blue'], zorder=3,\n"
            "           label='random forest (calibrated), marker size = bin count')\n"
            "style(ax, 'Calibrated model tracks the diagonal (predicted risk = observed rate)',\n"
            "      'Mean predicted probability', 'Observed preterm rate')\n"
            "ax.legend(frameon=False); plt.tight_layout(); plt.show()"),
        md("**One honest qualification, which I want to state loudly.** A low ECE means the scores are "
           "well-calibrated *in aggregate*. It does NOT mean the model stratifies individuals well. In "
           "fact the risk-spread diagnostic below shows most predictions sit within a few points of the "
           "base rate, so the model is essentially saying 'about average risk' for almost everyone, and "
           "it is right on average. That is useful for population-level analysis and honest about its "
           "limits for individual prediction."),
        code(
            "near = M['fraction_near_base_rate']\n"
            "print(f'{near*100:.0f}% of the best model\\'s predictions fall within +/-5 points of the '\n"
            "      f'{M[\"prevalence\"]*100:.1f}% base rate.')\n"
            "print('So low ECE = trustworthy AVERAGES, not reliable individual risk scores. Both things are true.')"),
        # ---- byrace ----
        md(anchor("byrace", "Does calibration hold across race groups?") + "\n\n"
           "This is the question that decides whether a fairness audit is even possible. If the model were "
           "well-calibrated overall but badly miscalibrated within a race group, then a predicted 10% would "
           "mean different things for different mothers, and every downstream fairness metric would be "
           "built on sand. So we check calibration *within* each race group, on the held-out data."),
        code(
            "strat = [s for s in M['stratified'] if s['status'] != 'suppressed']\n"
            "print(f\"{'group':<16}{'n':>10}{'AUC-ROC':>9}{'ECE':>9}   status\")\n"
            "for s in sorted(strat, key=lambda r: -r['n']):\n"
            "    print(f\"{s['group']:<16}{s['n']:>10,}{s['auc_roc']:>9.3f}{s['ece']:>9.4f}   {s['status']}\")"),
        md("**Calibration holds across the large groups** (ECE under about 0.005 for every group with "
           "enough data), so the risk scores mean the same thing across race, which is the green light "
           "the fairness audit needs. Two honest caveats carried from earlier weeks. First, discrimination "
           "is slightly lower for Hispanic and NH Asian mothers (AUC-ROC around 0.60 versus about 0.63 for "
           "NH White). I am flagging that only as a hypothesis for the planned Week 4 fairness audit, where "
           "it gets confidence intervals and a proper significance test; it is not a claim yet. Second, the "
           "smallest groups (NH NHOPI) get a wide-confidence-interval "
           "flag rather than a clean read, exactly as our tiered-reporting rule intends. Race here remains "
           "a marker of exposure to racism, not a biological cause, and nothing here is a clinical tool."),
        # ---- plan ----
        md(anchor("plan", "The plan from here") + "\n\n"
           "Week 3 established the honest floor: a calibrated model at about AUC-ROC 0.65, well-calibrated "
           "in aggregate and across race, and honest about its weak individual-level power. That floor is "
           "trustworthy precisely because Week 2 kept the leaky fields out.\n\n"
           "**Next weeks:**\n\n"
           "- **Gradient boosting with proper tuning.** XGBoost and LightGBM, tuned with Optuna, to see "
           "how much (if any) the ceiling moves. I expect a small gain over these baselines, consistent "
           "with the literature that trees lead on tabular data but cannot manufacture signal that is not "
           "there.\n"
           "- **The first formal fairness audit.** Turn this week's calibration-by-group result into "
           "rigorous metrics: equal opportunity, demographic parity, disparate impact, each with a "
           "confidence interval and a proper significance test.\n"
           "- **A decomposition of the racial gap**, asking how much of the disparity measured factors "
           "can account for, and how much is left unexplained.\n\n"
           "The modest ceiling is not a disappointment; it is the finding. It tells us these records "
           "support population-level equity diagnosis, not individual risk scoring, and that is the "
           "honest thing to build on." + AUTHOR),
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
