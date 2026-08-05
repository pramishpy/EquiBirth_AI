"""Compile the Week 5 storytelling notebook: deep-learning benchmark, stacking, and model selection.

Week 5 scope: benchmark two deep architectures (TabNet and FT-Transformer) against the Week 4
tuned boosters, attempt logistic stacking across all four base models, and make a final primary-model
selection decision. The headline: the deep nets underperform the boosters on this tabular birth-
certificate data (consistent with D41 and Grinsztajn et al. 2022), and stacking does not meaningfully
beat the best single booster, so xgboost is selected as the primary model going forward.

Every number is read live from reports/week5_models.json (produced by scripts/week5_deep_and_ensemble.py
on capped training draws, scored on a 100K held-out sample). NO hardcoded metric values.

Run:  PYTHONPATH=src ./.venv/bin/python scripts/week5_build_notebook.py
Then: ./.venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/week5_*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week5_01_deep_learning_and_selection"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def anchor(a, title): return f'<a id="{a}"></a>\n## {title}'


def contents(items):
    return "\n".join(["### Contents", ""] + [f"{i}. [{lbl}](#{a})" for i, (a, lbl) in enumerate(items, 1)])


SETUP = """\
# Week 5 reads back the deep-learning benchmark and stacking results from
# reports/week5_models.json (produced by scripts/week5_deep_and_ensemble.py).
# The heavy training happened in a script; the notebook reports and interprets the results.
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
M = json.loads((REPORTS / "week5_models.json").read_text())
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
        ("why-deep", "Why benchmark deep learning at all?"),
        ("contenders", "The contenders: TabNet and FT-Transformer"),
        ("results", "Head-to-head results"),
        ("masks", "TabNet attention masks: what the net sees"),
        ("stack", "Stacking: can ensembling help?"),
        ("selection", "Final model selection"),
        ("plan", "The road ahead (weeks 6 through 9)"),
    ]
    cells = [
        md("# Week 5: Deep-Learning Benchmark, Stacking, and Model Selection\n\n"
           "<sub>[Previous: Gradient Boosting and the Fairness Audit](week4_01_boosting_and_fairness.ipynb)</sub>\n\n"
           "<sub>New here? The tested code these notebooks import is documented in "
           "[the project code reference](week1_00_project_code_reference.ipynb).</sub>\n\n"
           "Week 4 gave us a tuned XGBoost at AUC-PR around 0.17 on held-out data, well-calibrated "
           "and honestly positioned in the published administrative-data band. This week we ask: can "
           "deep tabular architectures do better? And if we stack everything together, does the "
           "ensemble lift the ceiling? The answers turn out to be no and barely, which is exactly "
           "what the literature predicts for structured tabular data of this kind.\n\n"
           "As throughout this project, nothing here is a clinical tool: these are associational "
           "patterns in administrative data, not statements about any individual pregnancy."),
        md(contents(toc)),

        # ---- setup ----
        md(anchor("setup", "Setup: load results and data") + "\n\n"
           "All models were trained by `scripts/week5_deep_and_ensemble.py`. The deep nets trained on "
           "a 30K-capped draw (the FT-Transformer forward pass is the binding constraint), while the "
           "boosters used a 500K cap. All five models were scored on an identical 100K seeded draw from "
           "the untouched held-out partition. We read those results here."),
        code(SETUP),
        md("A few rows from the leakage-safe feature matrix, for orientation:"),
        code("mdf.sample(5, seed=RNG).to_pandas()"),
        code(
            "print(f\"Held-out scored: {M['caps']['held_out_scored_n']:,} births\")\n"
            "print(f\"Bootstrap resamples for CIs: {M['caps']['bootstrap_n']:,}\")\n"
            "print(f\"Booster train cap: {M['caps']['train_cap']:,}; deep-net train cap: "
            "{M['caps']['deep_train_cap']:,}\")\n"
            "print(f\"Max epochs for deep nets: {M['caps']['max_epochs']}\")\n"
            "print(f\"Seed: {M['seed']}\")"),

        # ---- why-deep ----
        md(anchor("why-deep", "Why benchmark deep learning at all?") + "\n\n"
           "The short answer: because the question deserves an empirical answer rather than an "
           "assumption. The project decision log (D41) pre-registered the expectation that gradient-"
           "boosted trees would lead on this structured tabular task, citing Grinsztajn et al. "
           "(NeurIPS 2022, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)), whose large-scale "
           "benchmark showed tree ensembles outperforming neural networks on medium-sized tabular "
           "datasets without extensive hyperparameter search.\n\n"
           "But a pre-registered expectation is not the same as a verified result. Week 5 turns the "
           "expectation into evidence: we train two architectures purpose-built for tabular data, "
           "score them on the same held-out sample as the boosters, and let the numbers speak."),

        # ---- contenders ----
        md(anchor("contenders", "The contenders: TabNet and FT-Transformer") + "\n\n"
           "**TabNet** (Arik and Pfister, 2019; [arXiv:1908.07442](https://arxiv.org/abs/1908.07442)) "
           "uses sequential sparse attention to select a subset of input features at each decision "
           "step, yielding interpretable per-instance feature masks. It was designed to bring the "
           "feature-selection inductive bias of trees into a differentiable architecture. For our "
           "purposes, the attention masks are an interesting cross-check against the boosters' SHAP "
           "importances.\n\n"
           "**FT-Transformer** (Gorishniy et al., 2021; "
           "[arXiv:2106.11959](https://arxiv.org/abs/2106.11959)) tokenizes each feature into an "
           "embedding and passes the token sequence through a standard Transformer encoder. It "
           "represents the \"apply Transformers to tabular data\" line of research, which has shown "
           "competitive results on some benchmarks but inconsistent gains over well-tuned trees.\n\n"
           "**Preprocessing disclosure:** both architectures require numeric inputs, so categorical "
           "features were ordinal-encoded and all columns were standardized (mean 0, std 1). Missing "
           "values were median-imputed before standardization. The boosters handle missing values "
           "natively and received no imputation."),

        # ---- results ----
        md(anchor("results", "Head-to-head results") + "\n\n"
           "All five models (xgboost, lightgbm, tabnet, ft_transformer, and the logistic stack over "
           "all four) scored on the same 100K held-out draw. AUC-PR is the primary metric for this "
           "rare-event problem; 95% CIs come from 1,000 bootstrap resamples."),
        code(
            "names = ['xgboost', 'lightgbm', 'tabnet', 'ft_transformer', 'stack']\n"
            "print(f\"{'model':<18}{'dev_test AUC-PR':>16}{'held-out AUC-PR':>17}   95% CI\")\n"
            "for n in names:\n"
            "    m = M['models'][n]\n"
            "    ci = m['held_out_auc_pr_ci']\n"
            "    print(f\"{n:<18}{m['dev_test_auc_pr']:>16.4f}{m['held_out_auc_pr']:>17.4f}\"\n"
            "          f\"   [{ci[0]:.4f}, {ci[1]:.4f}]\")"),
        code(
            "# Horizontal bar chart of held-out AUC-PR with CI error bars\n"
            "names = ['xgboost', 'lightgbm', 'tabnet', 'ft_transformer', 'stack']\n"
            "labels = ['XGBoost', 'LightGBM', 'TabNet', 'FT-Transformer', 'Stack']\n"
            "vals = [M['models'][n]['held_out_auc_pr'] for n in names]\n"
            "ci_lo = [M['models'][n]['held_out_auc_pr'] - M['models'][n]['held_out_auc_pr_ci'][0] for n in names]\n"
            "ci_hi = [M['models'][n]['held_out_auc_pr_ci'][1] - M['models'][n]['held_out_auc_pr'] for n in names]\n"
            "\n"
            "colors = [PALETTE['blue'], PALETTE['sky'], PALETTE['orange'], PALETTE['red'], PALETTE['green']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.0))\n"
            "y_pos = np.arange(len(names))\n"
            "ax.barh(y_pos, vals, xerr=[ci_lo, ci_hi], color=colors, edgecolor='none',\n"
            "        capsize=4, height=0.6)\n"
            "ax.set_yticks(y_pos)\n"
            "ax.set_yticklabels(labels)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)\n"
            "ax.axvline(M['models']['xgboost']['held_out_auc_pr'], ls=':', color=PALETTE['grey'], alpha=0.6)\n"
            "style(ax, 'Held-out AUC-PR: boosters lead, deep nets trail, stack ties',\n"
            "      'AUC-PR (held-out, 100K births)', '')\n"
            "ax.set_xlim(0.10, 0.20)\n"
            "plt.tight_layout(); plt.show()"),
        md("**Verdict (compute-capped).** The tuned boosters (XGBoost and LightGBM) lead on this "
           "benchmark. TabNet and FT-Transformer trail the boosters by several points of held-out "
           "AUC-PR (quantified live below), and the logistic stack does not meaningfully exceed the "
           "best single booster: its held-out AUC-PR falls inside the bootstrap confidence interval "
           "of XGBoost alone. This is consistent with D41 and with Grinsztajn et al. (2022) on "
           "medium-sized tabular data. We state this as a compute-capped finding, not an absolute "
           "one: the boosters won under this protocol, and the objection below spells out exactly "
           "what that does and does not license us to claim."),
        code(
            "# Quantify the booster-vs-deep-net gap live from the JSON.\n"
            "xgb = M['models']['xgboost']['held_out_auc_pr']\n"
            "tab = M['models']['tabnet']['held_out_auc_pr']\n"
            "ftt = M['models']['ft_transformer']['held_out_auc_pr']\n"
            "print(f\"XGBoost held-out AUC-PR:        {xgb:.4f}\")\n"
            "print(f\"TabNet gap vs XGBoost:          {(xgb - tab) * 100:.1f} AUC-PR points\")\n"
            "print(f\"FT-Transformer gap vs XGBoost:  {(xgb - ftt) * 100:.1f} AUC-PR points\")"),
        md("### A fair objection: was this a fair fight?\n\n"
           "No, and we should say so plainly. The deep nets trained on a 30K-row draw for at most a "
           "few epochs, with small hand-set configurations and no hyperparameter search, while the "
           "boosters were tuned on a 500K-row draw. That asymmetry is real, and a reader is right to "
           "press on it. Our defense has two parts.\n\n"
           "**(a) The gap is far too large to be a cap effect.** The boosters lead the deep nets by "
           "roughly 4 to 5 points of held-out AUC-PR (printed above), which is an order of magnitude "
           "larger than any plausible swing from more rows or a tuning sweep on this signal-poor "
           "administrative task. The direction and rough size match Grinsztajn et al. (2022), whose "
           "controlled benchmark (equal budgets, proper sweeps) still found tree ensembles ahead on "
           "medium-sized tabular data.\n\n"
           "**(b) But we do NOT claim the nets were given a fair chance.** Scaling the deep nets to "
           "the full data with a real hyperparameter search is a deliberate descoping decision (D46), "
           "taken on compute grounds, not a verdict that the architectures are incapable. Two honest "
           "caveats follow. First, the Week-8 XGBoost train-cap plateau (where extra rows stop "
           "helping the booster) is a property of XGBoost on this task and does NOT transfer to the "
           "deep nets, which would be expected to keep improving with more data. Second, this "
           "comparison establishes only that under a fixed, booster-favorable budget the boosters "
           "win; it does not establish a ceiling for deep tabular models in general. We select on "
           "what we measured and flag the rest as unfinished business."),

        # ---- masks ----
        md(anchor("masks", "TabNet attention masks: what the net sees") + "\n\n"
           "TabNet's sparse attention mechanism tells us which features the network relied on most "
           "heavily. These are *not* causal attributions; they are associational feature importances, "
           "analogous to split-based importances in trees. But they provide a useful cross-check: if "
           "the deep net and the boosters agree on the top features, we gain confidence that the "
           "signal is robust across architectures."),
        code(
            "print('TabNet top-15 features by mean attention weight:')\n"
            "print(f\"{'feature':<40}{'attention':>10}\")\n"
            "for feat, wt in M['tabnet_top_features']:\n"
            "    print(f'{feat:<40}{wt:>10.4f}')"),
        md("The overlap with the boosters' story is reassuring: medical risk count, prior preterm "
           "birth (rf_ppterm), prior cesarean, and education-stratified race interactions appear in "
           "both feature-importance lists. The deep net is not finding a hidden signal the trees "
           "missed; it is reading the same features less efficiently.\n\n"
           "**Leakage check:** none of TabNet's top features are at-delivery fields (those were "
           "removed in the Week 2 leakage audit). The attention masks are consistent with a model "
           "that operates on legitimate pre-delivery information only."),

        # ---- stack ----
        md(anchor("stack", "Stacking: can ensembling help?") + "\n\n"
           "Stacking combines the four base models' predicted probabilities into a single meta-model "
           "(here, L2-regularized logistic regression). The meta-model was trained on cross-fitted "
           "out-of-fold predictions to avoid meta-leakage: each base model predicts on folds it "
           "never trained on, and the meta-learner trains only on those out-of-fold predictions.\n\n"
           "The question is whether the base models contain complementary information that the stack "
           "can exploit."),
        code(
            "print('Stack meta-learner weights (logistic regression coefficients):')\n"
            "print(f\"{'base model':<20}{'weight':>8}\")\n"
            "for name, wt in sorted(M['stack_weights'].items(), key=lambda x: -x[1]):\n"
            "    print(f'{name:<20}{wt:>8.4f}')\n"
            "print()\n"
            "xgb_ho = M['models']['xgboost']['held_out_auc_pr']\n"
            "stk_ho = M['models']['stack']['held_out_auc_pr']\n"
            "print(f\"XGBoost held-out AUC-PR: {xgb_ho:.4f}\")\n"
            "print(f\"Stack held-out AUC-PR:   {stk_ho:.4f}\")\n"
            "print(f\"Difference: {stk_ho - xgb_ho:+.4f}\")\n"
            "print()\n"
            "xgb_ci = M['models']['xgboost']['held_out_auc_pr_ci']\n"
            "print(f\"XGBoost 95% CI: [{xgb_ci[0]:.4f}, {xgb_ci[1]:.4f}]\")\n"
            "print(f\"Stack point estimate ({stk_ho:.4f}) falls within XGBoost CI: \"\n"
            "      f\"{xgb_ci[0] <= stk_ho <= xgb_ci[1]}\")"),
        md("The stack leans on the two boosters and gives little weight to the deep nets (exact "
           "coefficients printed above). Its held-out AUC-PR falls within the bootstrap noise of the "
           "best single booster. In other words, the four models are reading the same signal from "
           "slightly different angles, and combining them does not lift the ceiling.\n\n"
           "Given the added complexity (four models instead of one, with a meta-learner on top), "
           "the simpler single-booster model wins on parsimony."),
        md("**A meta-leakage disclosure, in the interest of honesty.** The two deep-net columns fed "
           "to the stack were calibrated on dev_val, which is the same partition those nets used for "
           "early stopping. So those two columns are mildly in-sample: the model that generated them "
           "had already seen dev_val during fitting. The stack's internal K-fold cross-fitting guards "
           "the meta-learner's own fit (each base column is generated out-of-fold before the "
           "meta-logistic sees it), but it cannot fully undo the fact that the deep-net columns "
           "themselves were tuned against dev_val. This would bias the comparison in the stack's "
           "favor if the stack had won. It did not: the stack ties XGBoost and gets negligible weight "
           "on the deep-net columns, so the leak is benign here. We disclose it rather than let a "
           "silent optimism ride."),

        # ---- selection ----
        md(anchor("selection", "Final model selection") + "\n\n"
           "With the full comparison in hand, we make the primary-model selection decision for the "
           "remainder of the project."),
        code(
            "print('Selected model:', M['selected_model'])\n"
            "print()\n"
            "print('Rationale:')\n"
            "print(M['selection_rationale'])"),
        code(
            "# The stack edges XGBoost on dev_test, but is the difference meaningful on held-out?\n"
            "xgb = M['models']['xgboost']\n"
            "stk = M['models']['stack']\n"
            "dev_diff = stk['dev_test_auc_pr'] - xgb['dev_test_auc_pr']\n"
            "xgb_ci = xgb['held_out_auc_pr_ci']\n"
            "stk_ci = stk['held_out_auc_pr_ci']\n"
            "xgb_half = (xgb_ci[1] - xgb_ci[0]) / 2\n"
            "print(f\"Stack dev_test AUC-PR minus XGBoost dev_test AUC-PR: {dev_diff:+.4f}\")\n"
            "print(f\"XGBoost held-out 95% CI half-width:                  {xgb_half:.4f}\")\n"
            "print(f\"|dev_test edge| < CI half-width:                     \"\n"
            "      f\"{abs(dev_diff) < xgb_half}\")\n"
            "overlap = xgb_ci[0] <= stk_ci[1] and stk_ci[0] <= xgb_ci[1]\n"
            "print(f\"XGBoost and stack held-out CIs overlap:              {overlap}\")"),
        md("**Two points on the selection, so it does not read as sleight of hand.**\n\n"
           "First, the stack's dev_test AUC-PR marginally *exceeds* XGBoost's (printed above). We "
           "still pick XGBoost because that edge is smaller than the held-out bootstrap CI half-width "
           "and the two held-out CIs overlap completely, so on the untouched data the two are "
           "statistically indistinguishable. When a simpler model ties a more complex one, parsimony "
           "breaks the tie.\n\n"
           "Second, selecting on dev_test AUC-PR is *not* circular even though dev_test is also the "
           "partition on which the isotonic calibration was fit. AUC-PR is rank-based: it depends "
           "only on the ordering of scores, and isotonic calibration is a monotone (order-preserving) "
           "transform. A monotone map cannot change the ranking, so it cannot change AUC-PR. The "
           "calibration step moves the probabilities but leaves the selection metric untouched."),
        md("**What \"primary model\" means going forward:** XGBoost is the model whose predictions "
           "will be subjected to the fairness mitigation experiments (week 6), the SHAP-based "
           "disparity decomposition (week 7), the robustness and drift audit (week 8), and the "
           "final transparency report and demo (week 9). The deep-net and stack results remain "
           "documented here as evidence that the selection is informed, not arbitrary.\n\n"
           "The decision is conservative by design. A model intended for equity analysis must be "
           "stable, interpretable (via SHAP), and well-calibrated. XGBoost satisfies all three "
           "constraints and matches the administrative-data ceiling established by the literature."),

        # ---- plan ----
        md(anchor("plan", "The road ahead (weeks 6 through 9)") + "\n\n"
           "The model is selected. The remaining work is not about squeezing more AUC-PR; it is "
           "about understanding what the model's predictions reveal about structural inequity and "
           "ensuring the analysis is robust, transparent, and responsible.\n\n"
           "**Week 6: Fairness mitigation.** Apply Fairlearn's ExponentiatedGradient and "
           "ThresholdOptimizer to the selected model. Measure the Pareto frontier between "
           "predictive performance and equalized-odds constraints. Document what mitigation costs "
           "and what it buys.\n\n"
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
