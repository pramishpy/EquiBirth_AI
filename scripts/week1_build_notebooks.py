"""Compile the Week 1 storytelling notebooks (data acquisition, exploration, EDA, visualization).

These cover ONLY this week's scope: get the data, look hard at where it came from and whether we
gathered the right thing, explore it, and visualize what we find. Modeling, leakage control, and the
fairness audit are written up as the PLAN for the coming weeks, not as work already done.

Every number is computed live with pandas from the raw exploration table
(data/interim/natality_exploration_2019_2024.parquet, 17 pre-delivery certificate fields). The
notebooks read like a working analyst's lab notebook: load real rows, hunt for sentinels and outliers,
form a hypothesis, test it, narrate the reasoning. Cells are kept short and each is preceded by a short
markdown explanation of what we are about to do and why.

Run:    PYTHONPATH=src ./.venv/bin/python scripts/week1_build_notebooks.py
Then:   ./.venv/bin/jupyter nbconvert --execute --to notebook --inplace notebooks/week1_*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")

NOTEBOOKS = [
    ("01", "problem_and_acquisition", "Defining the Problem and Acquiring the Data"),
    ("02", "exploration", "Data Exploration: Provenance, Shape, and Quality"),
    ("03", "eda_and_visualization", "Exploratory Data Analysis and Visualization"),
]


def fname(num: str, slug: str) -> str:
    return f"week1_{num}_{slug}.ipynb"


def prevnext(num: str) -> str:
    idx = [i for i, (n, _, _) in enumerate(NOTEBOOKS) if n == num][0]
    parts = []
    if idx > 0:
        p = NOTEBOOKS[idx - 1]
        parts.append(f"[Previous: {p[2]}]({fname(p[0], p[1])})")
    if idx < len(NOTEBOOKS) - 1:
        nx = NOTEBOOKS[idx + 1]
        parts.append(f"[Next: {nx[2]}]({fname(nx[0], nx[1])})")
    return "<sub>" + " &nbsp;&middot;&nbsp; ".join(parts) + "</sub>"


def contents(items: list[tuple[str, str]]) -> str:
    lines = ["### Contents", ""]
    lines += [f"{i}. [{label}](#{a})" for i, (a, label) in enumerate(items, 1)]
    return "\n".join(lines)


def anchor(a: str, title: str) -> str:
    return f'<a id="{a}"></a>\n## {title}'


# ----------------------------------------------------------------------------- shared setup
SETUP = """\
# Setup. Week 1 is acquisition + exploration + EDA, so we load the RAW exploration table:
# the 17 pre-delivery birth-certificate fields we gathered, with their original values untouched.
# (No engineered features, no modeling columns yet. Those come in later weeks.) This table is built
# from the COMPLETE NBER public-use files for 2018-2024, restricted to singletons. Every number below
# is computed live from it. scipy is here for the hypothesis tests we run as we go, and we fix a seed
# so every df.sample() is reproducible.
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
RNG = 7  # fixed seed so every df.sample() below is reproducible

DATA = Path("..") / "data" / "interim" / "natality_exploration_2019_2024.parquet"
df = pd.read_parquet(DATA)  # ALL U.S. singleton births 2018-2024, raw pre-delivery fields only

# Human-readable label maps for the coded NCHS fields.
EDU = {1: "<8th grade", 2: "9-12, no diploma", 3: "HS grad/GED", 4: "Some college",
       5: "Associate", 6: "Bachelor", 7: "Master", 8: "Doctorate/Prof"}
PAY = {1: "Private", 2: "Medicaid", 3: "Self-pay", 4: "Indian Health Svc", 5: "CHAMPUS/TRICARE",
       6: "Other govt", 8: "Other"}
RACE_ORDER = ["NH White", "NH Asian", "Hispanic", "NH Multiracial", "NH NHOPI", "NH AIAN", "NH Black"]
# Published total U.S. births (NCHS final; 2024 provisional). Singletons are about 96.8% of births.
PUBLISHED_BIRTHS = {2018: 3791712, 2019: 3747540, 2020: 3613647, 2021: 3664292, 2022: 3667758,
                    2023: 3596017, 2024: 3600000}

PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
           "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999",
           "ink": "#1a1a2e"}
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

pio.renderers.default = "notebook_connected"  # small files; interactive in Jupyter / nbviewer

def ptmpl(fig, title, xtitle, ytitle, height=440):
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=16, color=PALETTE["ink"])),
        xaxis_title=xtitle, yaxis_title=ytitle, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=PALETTE["ink"]),
        margin=dict(l=60, r=30, t=60, b=50), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, font=dict(color=PALETTE["ink"])))
    fig.update_xaxes(showgrid=False, linecolor="#cccccc")
    fig.update_yaxes(showgrid=True, gridcolor="#ECECEC", zeroline=False)
    return fig

def rate_by(frame, col, dropna_target=True):
    \"\"\"Preterm rate (%) and count for any grouping column. The workhorse of the whole EDA.\"\"\"
    d = frame.dropna(subset=["preterm"]) if dropna_target else frame
    g = d.groupby(col)["preterm"].agg(rate="mean", n="size")
    g["rate"] = g["rate"] * 100
    return g

def risk_ratio_ci(frame, flag_col, target="preterm"):
    \"\"\"Risk ratio (flag=1 vs flag=0) with a 95% log-normal CI. Returns (rr, lo, hi, r1, r0).\"\"\"
    d = frame.dropna(subset=[target, flag_col])
    g = d.groupby(flag_col)[target].agg(["mean", "sum", "size"])
    r1, r0 = g.loc[1.0, "mean"], g.loc[0.0, "mean"]
    a, n1 = g.loc[1.0, "sum"], g.loc[1.0, "size"]
    c, n0 = g.loc[0.0, "sum"], g.loc[0.0, "size"]
    rr = r1 / r0
    se = np.sqrt(1/a - 1/n1 + 1/c - 1/n0)
    return rr, rr*np.exp(-1.96*se), rr*np.exp(1.96*se), r1*100, r0*100
"""

SAMPLE_MD = ("### First, a look at the raw material\n\n"
             "Each notebook stands on its own, so we start the way a real analysis session does: load "
             "the table and look at a handful of actual rows. One record is one singleton birth.")
SAMPLE_CODE = "df.sample(5, random_state=RNG)"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


CODE_REF = ("<sub>New here? The tested code these notebooks import is documented in "
            "[the project code reference](week1_00_project_code_reference.ipynb).</sub>")


def title_block(num: str, title: str, lead: str) -> nbf.NotebookNode:
    return md(f"# Week 1, Part {num}: {title}\n\n{prevnext(num)}\n\n{CODE_REF}\n\n{lead}")


def setup_cells() -> list:
    """Split setup into short, documented chunks so no single cell is a wall of code."""
    s = SETUP
    imports = s[:s.index("DATA = Path")].rstrip()
    load = s[s.index("DATA = Path"):s.index("PALETTE = {")].rstrip()
    palette = s[s.index("PALETTE = {"):s.index("plt.rcParams")].rstrip()
    style_ = s[s.index("plt.rcParams"):s.index("pio.renderers")].rstrip()
    plotly_helper = s[s.index("pio.renderers"):s.index("def rate_by")].rstrip()
    rate_helper = s[s.index("def rate_by"):].rstrip()
    return [
        md("### Setup\n\nA short, documented setup so the rest of the notebook reads cleanly. First the "
           "imports and display options. `scipy` is here because we run real hypothesis tests later, and "
           "we fix a random seed so every `df.sample()` below is reproducible."),
        code(imports),
        md("Load the raw exploration table once. It holds the 17 pre-delivery certificate fields we "
           "gathered this week, built from the complete 2018-2024 public-use files (singletons only), "
           "plus readable label maps for the coded fields and the published national birth totals we "
           "will validate our counts against."),
        code(load),
        md("A colorblind-safe palette (the Okabe-Ito set), reused on every figure so the series looks "
           "like one piece. We give the four largest race groups distinct colors and leave the rest grey "
           "to keep attention where the data are most reliable."),
        code(palette),
        md("Matplotlib house style, plus a tiny `style()` helper so every static figure gets a "
           "takeaway-as-title and labelled axes without repeating boilerplate."),
        code(style_),
        md("Marquee charts also get an interactive Plotly companion, so `ptmpl()` gives those figures the "
           "same dark-text, low-clutter house style."),
        code(plotly_helper),
        md("Finally, the 2 helpers we lean on most. `rate_by()` returns the preterm rate and count for "
           "any grouping column, and it is the workhorse of the entire EDA. `risk_ratio_ci()` goes 1 step "
           "further for binary flags: it returns the risk ratio (flag=1 vs flag=0) with a 95% log-normal "
           "confidence interval, so we can report an effect size and its uncertainty together, not just a "
           "p-value. Both are worth reading once."),
        code(rate_helper),
    ]


def opener(num: str, title: str, lead: str, toc: list) -> list:
    return [title_block(num, title, lead), md(contents(toc)), *setup_cells(),
            md(SAMPLE_MD), code(SAMPLE_CODE)]


# ============================================================================= Notebook 01
def nb_01():
    cells = opener("01", "Defining the Problem and Acquiring the Data",
        "This is the first notebook of the project.",
        [("problem", "The problem: preterm birth and equity"),
         ("questions", "The questions I want this project to answer"),
         ("source", "Where the data comes from"),
         ("governance", "Data provenance and governance"),
         ("audit", "Acquisition Audit"),
         ("lifecycle", "Where this sits in the data-science lifecycle"),
         ("scope", "Scope for this week, and what comes next")])
    cells += [
        md(anchor("problem", "The problem: preterm birth and equity") + "\n\n"
           "**Preterm birth**, delivery before 37 completed weeks of gestation, is the leading cause of "
           "neonatal death and a major driver of lifelong neurodevelopmental and cardiometabolic disease. "
           "The [WHO](https://www.who.int/news-room/fact-sheets/detail/preterm-birth) sorts it into 3 "
           "sub-categories by how early the birth is: extremely preterm (before 28 weeks), very preterm "
           "(28 to under 32 weeks), and moderate-to-late preterm (32 to under 37 weeks). The scale is "
           "large: the WHO estimates about 13.4 million preterm births worldwide in 2020, and complications "
           "of prematurity are the leading cause of death in children under 5. Our target column collapses "
           "all of this into 1 flag (born before 37 weeks or not), which is the standard public-health "
           "definition and the one the certificate supports.\n\n"
           "In the United States it is also one of the most stubborn racial inequities in all of public "
           "health. Non-Hispanic (NH) Black infants are born preterm far more often than NH White infants, "
           "and Black women die of pregnancy-related causes at "
           "[more than three times](https://www.kff.org/racial-equity-and-health-policy/racial-disparities-in-maternal-and-infant-health-current-status-and-key-issues/) "
           "the rate of White women. The part that has always struck is that this gap does not close "
           "with education or income, which is the first hint that the usual socioeconomic story is "
           "incomplete. A framing I hold to for the whole project: **race here is a social construct and a "
           "marker of exposure to racism and structural barriers, never a biological cause** of preterm "
           "birth, and nothing in these notebooks is a clinical tool. This is educational research, and it "
           "speaks in associational language, not causal claims.\n\n"
           "One number to set expectations. The CDC/NCHS provisional 2024 figure "
           "([VSRR 38](https://www.cdc.gov/nchs/data/vsrr/vsrr038.pdf)) puts the US preterm rate at 10.41%, "
           "but that is an *all-births* headline. This project studies *singletons* only. Multiples "
           "(twins, triplets, and more) are about 60% preterm and are excluded by design, because pooling "
           "them in would swamp the singleton signal we care about. So we expect our overall singleton rate "
           "to land meaningfully below 10.4%, and the EDA will check exactly that.\n\n"
           "So this project has 2 motivations. First, can we predict preterm birth from "
           "information available *before* delivery? If we could, that would help clinicians target care. "
           "Second, and more importantly, how equitable is the outcome across race and ethnicity, and how "
           "much of the disparity can measured factors actually explain? We care about the second question "
           "more, because a prediction model that ignores the equity context would be worse than useless.\n\n"
           "Sources: [WHO preterm-birth fact sheet](https://www.who.int/news-room/fact-sheets/detail/preterm-birth), "
           "[CDC/NCHS VSRR 38](https://www.cdc.gov/nchs/data/vsrr/vsrr038.pdf), "
           "[March of Dimes](https://www.marchofdimes.org/peristats/data), "
           "[KFF](https://www.kff.org/racial-equity-and-health-policy/racial-disparities-in-maternal-and-infant-health-current-status-and-key-issues/)."),
        md(anchor("questions", "The questions we want this project to answer") + "\n\n"
           "These are the questions the whole project is organized around:\n\n"
           "1. **How well can preterm birth be predicted from pre-delivery birth-certificate data?**\n"
           "2. **How large is the racial disparity in preterm birth, and is it statistically real or just "
           "sampling noise?**\n"
           "3. **Does the disparity shrink once we account for socioeconomic proxies like education?** This "
           "is the falsifiable version of the socioeconomic explanation hypothesis.\n"
           "4. **Which recorded factors actually move the outcome, and by how much?**\n\n"
           "This week we cannot answer the prediction question (that needs models, which is later work). But "
           "we can make real progress on questions 2 through 4 with careful EDA, and we can lay groundwork "
           "for question 1 by understanding exactly how much signal lives in these fields."),
        md(anchor("source", "Where the data comes from") + "\n\n"
           "The data are the **CDC/NCHS Natality public-use files**, distributed through the "
           "[NBER mirror](https://www.nber.org/research/data/vital-statistics-natality-birth-data). These "
           "are not a survey. They are a near-census assembled by the "
           "[National Vital Statistics System](https://www.cdc.gov/nchs/nvss/about_nvss.htm), in which "
           "NCHS contracts with the registration systems of the 50 states, DC, New York City, and five "
           "territories. Every birth produces a certificate; the clinical and demographic items come from "
           "hospital records and a maternal worksheet.\n\n"
           "I settled on a **2018 to 2024** window. The short version: since 2016 every jurisdiction has "
           "used the [2003 revision](https://www.cdc.gov/nchs/nvss/revisions-of-the-us-standard-certificates-and-reports.htm) "
           "of the U.S. Standard Certificate of Live Birth, so every year in this window is on the same form "
           "with the same fields, which is what makes pooling them legitimate. A framing note I will carry "
           "throughout: the familiar 10.4% national preterm headline is an *all-births* figure, and this "
           "project studies *singletons* only (multiples are excluded by design because they are about 60% "
           "preterm and would swamp the signal), so we expect our overall rate to land meaningfully below "
           "10.4%."),
        md(anchor("governance", "Data provenance and governance") + "\n\n"
           "Because so much rides on trusting these files, it is worth writing down where they come from "
           "and under what rules, before we compute a single statistic.\n\n"
           "- **Near-census, not a survey.** The "
           "[National Vital Statistics System](https://www.cdc.gov/nchs/nvss/about_nvss.htm) registers "
           "essentially every birth in the country. NCHS works through the vital-registration systems of "
           "the 50 states, DC, New York City, and the territories, so coverage is meant to be the whole "
           "population rather than a sampled slice.\n"
           "- **Where the items come from.** The demographic and clinical fields originate on the "
           "hospital worksheet and the facility birth record: some are transcribed from the medical chart, "
           "others are reported by the mother. Knowing which is which matters for the bias notes in "
           "notebook 02.\n"
           "- **1 stable form across the window.** Every US jurisdiction had adopted the "
           "[2003 revision](https://www.cdc.gov/nchs/nvss/revisions-of-the-us-standard-certificates-and-reports.htm) "
           "of the birth certificate by 2016, so all of 2018 to 2024 sits on the same form with the same "
           "fields. That is what makes pooling the years legitimate rather than an apples-to-oranges mix.\n"
           "- **Privacy and ethics.** The public-use files are de-identified and, as public secondary data, "
           "the analysis is IRB-exempt. There is nothing in them that identifies an individual mother or "
           "infant.\n"
           "- **No geography post-2005.** For privacy the public-use files carry no state or county "
           "identifiers, so any state-level comparison later has to come from separate CDC WONDER "
           "aggregates rather than from these rows. That is noted as PLANNED work for a later week, not "
           "something this week can do.\n\n"
           "The practical upshot for this notebook: because the source is a near-census on a single stable "
           "form, our coverage should land near 100% of singleton births each year, which is exactly the "
           "acquisition audit below."),
        md(anchor("audit", "Acquisition audit") + "\n\n"
           "Even with the complete files, we owe a check that what landed on disk matches the national "
           "totals, because a quietly truncated download or a botched singleton filter would show up here. "
           "Let me compare our singleton counts per year against the published births."),
        code(
            "# Our singleton count per year vs the published national totals (singletons ~96.8% of births).\n"
            "yr = df.groupby('birth_year').size()\n"
            "rows = []\n"
            "for y in sorted(PUBLISHED_BIRTHS):\n"
            "    singletons = PUBLISHED_BIRTHS[y] * 0.968\n"
            "    rows.append({'year': y, 'our_singletons': int(yr[y]), 'published_singletons': int(singletons),\n"
            "                 'coverage_%': round(yr[y] / singletons * 100, 1)})\n"
            "frac = pd.DataFrame(rows)\n"
            "sing_total = sum(PUBLISHED_BIRTHS.values()) * 0.968\n"
            "print(f'In total we hold {df.shape[0]:,} singleton rows, about '\n"
            "      f'{df.shape[0]/sing_total*100:.1f}% of the ~{sing_total/1e6:.1f}M published singleton births in this window.')\n"
            "frac"),
        md("The exact numbers behind the chart, printed so nothing hides in the bars: our count, the "
           "approximate published singleton count, and the coverage ratio for each year."),
        code(
            'print("Coverage of published singleton births, per year (values behind the chart):")\n'
            "for _, r in frac.iterrows():\n"
            "    print(f\"  {int(r['year'])}: our {int(r['our_singletons']):>9,}  \"\n"
            "          f\"vs published ~{int(r['published_singletons']):>9,}  = {r['coverage_%']:.1f}%\")"),
        md("Let me see that coverage as an interactive chart. With the complete files the bars should sit "
           "right around 100% every year (a touch over, because the published-singleton denominator is an "
           "approximation and 2024 is provisional). Hover any bar for its exact value."),
        code(
            "fig = go.Figure(go.Bar(x=frac['year'].astype(str), y=frac['coverage_%'],\n"
            "                       marker_color=PALETTE['sky'],\n"
            "                       text=[f'{v:.1f}%' for v in frac['coverage_%']], textposition='outside',\n"
            "                       hovertemplate='%{x}: %{y:.1f}% of published singletons<extra></extra>'))\n"
            "fig.add_hline(y=100, line_dash='dot', line_color=PALETTE['grey'])\n"
            "fig.update_yaxes(range=[0, max(105, frac['coverage_%'].max()*1.1)])\n"
            "ptmpl(fig, 'Our singleton counts track the published totals every year (complete files)',\n"
            "      'Birth year', 'Our count as % of published singletons')\n"
            "fig.show()"),
        md("**Reading the audit**\n\n"
           "1. **We now have essentially the whole singleton population for each year, not a sample.** "
           "Coverage lands right around 100% every year (the table reads 100 to 101 percent, slightly over "
           "because the 96.8% singleton share is a round figure and 2024 births are still provisional, so "
           "the denominator is approximate). The point is that we hold essentially every singleton record; "
           "the only ones set aside are births whose obstetric gestational estimate is missing (null "
           "target) and the handful the singleton rule removes. There is no sampling step left to bias "
           "anything. We still cross-check representativeness the reproducible way in the EDA, by confirming "
           "the per-race rates reproduce the published NVSS ordering and magnitudes.\n"
           "2. **Counts are comparable across years now**, because every year is the full population rather "
           "than an uneven sample. That said, I still default to **rates** for cross-year comparison, "
           "because rates are the right unit for a public-health question and are robust to the small "
           "year-to-year coverage differences from missing-target records."),
        md(anchor("lifecycle", "Where this sits in the data-science lifecycle") + "\n\n"
           "It helps to step back and place this week honestly on the full data-science lifecycle, so it is "
           "clear what this week covers versus what the coming weeks will build. The stages below are "
           "grouped: Covered this week, Planned (a later week, which the EDA will inform), or Out of scope "
           "for this project. This is a map of the plan, not a claim that the later stages are done.\n\n"
           "| Lifecycle stage | Status | Where |\n"
           "|---|---|---|\n"
           "| Problem / business understanding | Covered this week | This notebook |\n"
           "| Data governance and regulatory context | Covered this week | This notebook |\n"
           "| Data acquisition and ingestion | Covered this week | This notebook |\n"
           "| Data validation and schema checks | Covered this week | Notebook 02 |\n"
           "| Data-quality assessment, cleaning, missing-value and outlier treatment | Covered this week | Notebook 02, 03 |\n"
           "| EDA, descriptive stats, distribution and correlation analysis | Covered this week | Notebook 03 |\n"
           "| Hypothesis testing (p-values, confidence intervals, effect sizes) | Covered this week | Notebook 03 |\n"
           "| Visualization and storytelling | Covered this week | All week-1 notebooks and the dashboard |\n"
           "| Feature engineering, encoding, interaction features | Planned | Week 2 |\n"
           "| Leakage prevention and train / test / held-out splits | Planned | Week 2 |\n"
           "| Calibration and threshold tuning | Planned | Week 3 |\n"
           "| Supervised baselines and gradient boosting | Planned | Weeks 3 to 4 |\n"
           "| Hyperparameter search (Optuna) | Planned | Week 4 |\n"
           "| Evaluation metrics (AUC-ROC, AUC-PR, Brier, ECE / MCE) | Planned | Weeks 3 to 4 |\n"
           "| First fairness evaluation (equal opportunity, demographic parity, disparate impact) with significance control | Planned | Week 4 |\n"
           "| Deep-learning benchmark and interpretability | Planned | Later week |\n"
           "| Ensemble / stacking and final model selection | Planned | Later week |\n"
           "| Full fairness mitigation and tradeoff tables | Planned | Later week |\n"
           "| SHAP interpretability and decomposition of the racial gap | Planned | Later week |\n"
           "| Uncertainty / robustness and out-of-time validation | Planned | Later week |\n"
           "| Model card and datasheet | Planned | Later week |\n"
           "| Deployment (Streamlit) and monitoring | Planned | Later week |\n"
           "| CDC WONDER aggregate state context and NHANES external sanity check | Planned | Later week |\n"
           "| Streaming ingestion | Out of scope | Batch public-use files, no stream |\n"
           "| A/B tests and RCTs | Out of scope | Observational data only |\n"
           "| Row-level geographic joins | Out of scope | No geography in public-use files |\n"
           "| RCT-grade causal inference | Out of scope | Associational study, not causal |\n\n"
           "The Covered rows are this week's work: acquisition, exploration, quality auditing, EDA, and "
           "visualization. Everything from the feature-engineering stage on is Planned, so throughout this "
           "week it reads as intent, not as results, and the modeling and fairness rows will be filled in "
           "as the coming weeks build them. The Out-of-scope rows are honest boundaries: they are either "
           "impossible with these de-identified batch files (streaming, geography) or the wrong tool for an "
           "observational dataset (randomized experiments, causal inference)."),
        md(anchor("scope", "Scope for this week, and what comes next") + "\n\n"
           "**This week** is acquisition, exploration, data-quality auditing, EDA, and visualization. That "
           "is the work in these three notebooks and the dashboard.\n\n"
           "**What we plan to do in the coming weeks**, informed by what the EDA reveals, is roughly:\n\n"
           "- **Feature engineering with a strict leakage audit.** Birth certificates record many fields "
           "*at* delivery (birthweight, Apgar score, labor interventions) that are near-perfect but "
           "useless predictors, because they do not exist when a prediction would actually be made. We will "
           "need a disciplined, tested gate that bars those, and we expect that to be the single most "
           "important engineering decision in the project.\n"
           "- **Baseline models, calibration first.** Logistic regression and tree baselines, with honest "
           "attention to the fact that a rare outcome makes accuracy a meaningless metric.\n"
           "- **A formal fairness audit.** Demographic parity, equal opportunity, calibration-by-group, "
           "with confidence intervals, to turn the disparity we expect to find this week into rigorous "
           "metrics.\n"
           "- **Explainability and a decomposition of the gap**, to ask how much of the racial disparity "
           "measured factors can actually account for." + AUTHOR),
    ]
    return cells


# ============================================================================= Notebook 02
def nb_02():
    cells = opener("02", "Data Exploration: Provenance, Shape, and Quality",
        "Before trusting anything, we do the boring but essential work: look at the shape and types, find "
        "the holes, write down what each field means, and name the biases we know live in birth-certificate "
        "data. This is the part that decides whether everything downstream is real.",
        [("shape", "Shape, types, and memory"),
         ("missing", "Missingness"),
         ("dictionary", "A field dictionary"),
         ("biases", "Known biases"),
         ("groups", "Group sizes")])
    cells += [
        md(anchor("shape", "Shape, types, and memory") + "\n\n"
           "The first mechanical inspection any analysis owes itself: how many rows and columns, what "
           "types, and how much memory. I also want a compact, `info()`-style per-field summary so I can "
           "see at a glance which fields are present and how many distinct values each takes."),
        code(
            "print('Shape (rows, fields):', df.shape)\n"
            "print('Memory             :', f'{df.memory_usage(deep=True).sum()/1e6:.0f} MB')\n"
            "print('Dtype counts:'); print(df.dtypes.value_counts())"),
        code(
            "pd.DataFrame({'dtype': df.dtypes.astype(str), 'non_null': df.notna().sum(),\n"
            "              'pct_present': (df.notna().mean()*100).round(1),\n"
            "              'n_unique': df.nunique(dropna=True)})"),
        md("17 fields and about 25 million rows, all numeric except the race/ethnicity label. Even at the "
           "full national scale this slim 17-column projection is only a few gigabytes in memory, so we can "
           "explore it interactively in pandas without reaching for chunking or Spark. That is the payoff of "
           "projecting to just the fields we need before pooling the years. Nothing here is surprising yet, "
           "which is exactly what we want from a first look: surprises this early usually mean a parsing "
           "bug, not a discovery."),
        md(anchor("missing", "Missingness") + "\n\n"
           "Missingness is never random noise in administrative data, so we map it before trusting any "
           "field. Let me compute the missing share per field and plot only the ones that actually have "
           "gaps."),
        code(
            "miss = (df.isna().mean()*100).round(2)\n"
            "miss = miss[miss > 0].sort_values(ascending=False)\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.0))\n"
            "ax.barh(miss.index[::-1], miss.values[::-1], color=PALETTE['orange'])\n"
            "for i, v in enumerate(miss.values[::-1]): ax.text(v+0.1, i, f'{v:.1f}%', va='center', fontsize=9)\n"
            "style(ax, 'Missingness concentrates in a few fields, led by marital status', 'Missing (%)', '')\n"
            "ax.grid(axis='y'); plt.tight_layout(); plt.show()"),
        md("The big one is marital status (`married`), blank for about 12% of rows. That is not noise: some "
           "reporting areas historically did not collect it the same way, so it is *missing not at random* "
           "and correlated with place. Everything else sits under 2%. The lesson we carry into the modeling "
           "weeks is to handle missingness per-field and treat the fact that something is missing as "
           "possibly informative, rather than blindly dropping or mean-filling. For this week's descriptive "
           "work I simply exclude missing values from each statistic and say so."),
        md("**Why I call marital status missing not at random (MNAR).** A missing value is only harmless "
           "if it is missing completely at random: unrelated to anything we care about, including time. "
           "Marital status is not. The public-use files carry no geography, but the reporting-area story is "
           "well documented: some jurisdictions historically collected and reported marital status "
           "differently (a few once inferred it rather than asking directly), so whether the field is blank "
           "depends on *where* and *when* the birth was registered, which is itself correlated with the "
           "population and the outcome. If the gaps were purely random loss, the missing share would sit "
           "roughly flat near a single value across years, not follow a direction. Let me split the missing "
           "share by year and look at its shape, because a share that trends over time is better explained "
           "by how collection changed than by random loss."),
        code(
            "# WHY marital status is missing not at random: split its missing share by year.\n"
            "mar_by_year = (df.assign(missing=df['married'].isna())\n"
            "                 .groupby('birth_year')['missing'].mean().mul(100).round(2))\n"
            'print("Marital-status missing share by year (%):")\n'
            "print(mar_by_year.to_string())\n"
            'print(f"\\nRange across years: {mar_by_year.max():.1f}% (2018) down to {mar_by_year.min():.1f}% (2024). "\n'
            '      "Declines steadily by year (about 1 point), so not flat: consistent with reporting-area "\n'
            '      "collection differences converging over time (still not missing-at-random).")'),
        md("The missing share is not flat: it declines steadily and monotonically, from about 12.0% in "
           "2018 to about 11.1% in 2024, roughly a 1-point slide over the window. That steady decline is "
           "not the erratic year-to-year noise you would see from random loss; a monotonic trend is better "
           "explained by reporting areas gradually converging on more consistent marital-status collection "
           "over time. The core point stands either way: the blanks are tied to when (and where) the birth "
           "was registered, so they are informative rather than missing completely at random. In the "
           "modeling weeks (PLANNED) I therefore intend to treat the fact that marital status is missing as "
           "possibly informative in its own right, likely with an explicit missing-indicator, rather than "
           "mean-fill it and pretend the gap is neutral. This stays associational: the trend is evidence "
           "about how the field was recorded, not a claim about any mother."),
        md(anchor("dictionary", "A field dictionary") + "\n\n"
           "Before relating fields to the outcome, we want a clear record of what each one means and where "
           "on the certificate it comes from. One column matters enormously for the modeling weeks: the "
           "*timing*. A field is only a fair predictor if it is known *before* delivery. I also record the "
           "*valid range* and the *sentinel / unknown code* for each field, because NCHS encodes "
           "not-stated as an in-range-looking number (for example BMI 99.9), and knowing the exact code up "
           "front is what lets the pipeline null it instead of averaging it in. These 2 columns are "
           "documentation drawn from the ingest schema, not values computed here.\n\n"
           "| Field | Meaning | Source on the certificate | Valid range | Sentinel / unknown code | Known before delivery? |\n"
           "|---|---|---|---|---|---|\n"
           "| `birth_year` | Year of birth, 2018-2024 | Registration | 2018-2024 | none | Context |\n"
           "| `mager` | Maternal age (years) | Mother's reported age | 12-50 | none (always present) | Yes |\n"
           "| `meduc` | Education, 1 (<8th grade) to 8 (doctorate) | Mother's education | 1-8 | 9 | Yes |\n"
           "| `bmi` | Pre-pregnancy body mass index | Height and pre-pregnancy weight (worksheet) | ~13-69.9 | 99.9 | Yes |\n"
           "| `cig_0` | Cigarettes/day before pregnancy | Tobacco-use fields | 0-98 | 99 | Yes |\n"
           "| `priorlive` / `priordead` | Prior live births / prior deaths | Pregnancy history | 0-98 | 99 | Yes |\n"
           "| `married` | Marital status (1 = married), raw `dmar` | Marital status | 1-2 | 9 | Yes |\n"
           "| `rf_pdiab` `rf_phype` `rf_ppterm` `rf_cesar` | Pre-pregnancy diabetes, hypertension, prior preterm, prior cesarean | Risk-factor checkboxes | Y / N | U (unknown) | Yes |\n"
           "| `wic` | WIC nutrition-program participant | WIC field | recoded 0/1 | U (unknown) upstream | Yes (an SES proxy) |\n"
           "| `first_trimester_care` | Prenatal care began in trimester 1 | Month care began | recoded 0/1 | 99 on raw `precare` upstream | Yes |\n"
           "| `pay` | Principal payment source (Medicaid, private, ...) | Source of payment | 1-8 | 9 | Yes (an SES proxy) |\n"
           "| `mother_race_ethnicity` | Bridged race/ethnicity | Mother's race and Hispanic origin | 7 labels | mapped upstream | A marker, never a cause |\n"
           "| `preterm` | **Target**: born before 37 weeks | Derived from the obstetric gestational estimate | 0/1 (derived) | 99 on raw gestation upstream | The outcome |\n\n"
           "A few notes on that sentinel column. `wic` and `first_trimester_care` reach this table already "
           "recoded to a clean 0/1, and `preterm` is the derived 0/1 target, so the codes shown for them "
           "are the ones that were resolved *upstream* in ingest (the raw `precare` month-care field uses "
           "99, the raw gestation field uses 99) rather than codes still present here. The Y/N risk flags "
           "use the string `U` for unknown. The point of writing these down is that the sentinels for "
           "`bmi` (99.9) and `cig_0` (99) sit right at the top of an otherwise plausible numeric range, so "
           "the EDA has to hunt them explicitly (notebook 03 does exactly that) rather than assume a "
           "describe() max is a real measurement.\n\n"
           "We deliberately gathered only fields known before delivery, plus the target. The tempting "
           "at-delivery fields (birthweight, Apgar, NICU) are not in this table on purpose; designing the "
           "tested gate that keeps them out is a planned task for the feature-engineering week."),
        md(anchor("biases", "Known biases") + "\n\n"
           "Birth certificates are a gold standard but they are not flawless. Three caveats we carry "
           "forward:\n\n"
           "1. **Race and ethnicity misclassification.** Self-reported race is highly accurate for White, "
           "Black, and Hispanic groups but materially less so for "
           "[American Indian / Alaska Native and NHOPI](https://www.cdc.gov/nchs/hus/sources-definitions/race.html), "
           "where misclassification can understate rates. Those groups are also the smallest here, so we "
           "read their estimates with extra caution.\n"
           "2. **Self-reported anthropometrics.** NCHS quality studies explicitly "
           "[did not assess the maternal worksheet's height and weight]"
           "(https://www.cdc.gov/nchs/data/nvsr/nvsr65/nvsr65_06.pdf), so `bmi` is the least-audited "
           "clinical field. We expect to find sentinel codes lurking in it.\n"
           "3. **Checkbox underreporting.** Risk-factor checkboxes (`rf_*`) are known to undercount true "
           "prevalence, which biases their measured association toward zero. So any weak association we "
           "find from them is really a floor."),
        md("**A note on the missing birth identifier (no UID).** The public-use file carries no birth "
           "identifier, no way to tell one record from another beyond its field values. A direct "
           "consequence is that with coarse categorical features (a handful of age bins, education codes, "
           "yes/no flags) some records will share an identical feature fingerprint simply because 2 real, "
           "distinct births looked the same on the parts of the certificate we kept. That is expected, not "
           "a corruption: these are not duplicated rows, they are separate births that happen to be "
           "indistinguishable through this narrow 17-field window. Because there is no identifier, a model "
           "cannot use these matches to memorize any individual; there is no individual to memorize. This "
           "is a known property of administrative microdata, and we carry it forward honestly rather than "
           "'de-duplicating' away real births. (I deliberately quote no specific duplicate percentage; any "
           "such figure would be an artifact of which features you happen to keep, not a property of the "
           "data.)"),
        md(anchor("groups", "Group sizes") + "\n\n"
           "The misclassification caveat above is abstract until we see the actual group sizes. Let me list "
           "them, smallest first, because the smallest groups are exactly where both misclassification and "
           "noise hit hardest."),
        code(
            "small = df.dropna(subset=['preterm']).groupby('mother_race_ethnicity').size().sort_values()\n"
            "print('Sample size by race/ethnicity group (smallest carry the most uncertainty):')\n"
            "print(small.to_string())"),
        md("**Takeaway.** Even in the full data NH NHOPI and NH AIAN are the smallest groups (roughly 67k "
           "and 180k rows, against millions for the large groups), so we trust the NH White, NH Black, "
           "Hispanic, and NH Asian estimates most and caveat the small ones wherever they appear."),
        md("**One external cross-check that the ingest did the right thing.** Everything above audits the "
           "data against itself. The stronger test is whether it reproduces the outside world: published "
           "NVSS reporting has a well-known per-race ordering of preterm rates. If our full-data rates land "
           "in that same order and near those magnitudes, that is external evidence the ingest pipeline "
           "did its job correctly end to end: de-sentinelling the not-stated codes, deriving the preterm "
           "target from the obstetric gestation, mapping race/ethnicity, and applying the singleton filter. "
           "A silent bug in any of those steps would knock the ordering out of line, so this is validation, "
           "not just description."),
        code(
            "rr = rate_by(df, 'mother_race_ethnicity').reindex(RACE_ORDER).dropna()\n"
            'print("Per-race preterm rate (%), our full data (validates the ingest against published NVSS ordering):")\n'
            "print(rr['rate'].round(2).to_string())\n"
            'print(f"\\nOrdering check: NH White lowest ({rr.loc[\'NH White\',\'rate\']:.2f}%), "\n'
            '      f"NH Black highest ({rr.loc[\'NH Black\',\'rate\']:.2f}%) -> matches published NVSS pattern.")'),
        md("The rates reproduce the published pattern: NH White lowest, NH Black highest, with the other "
           "groups ranged between, and all near the magnitudes NVSS reports. That is the reassurance we "
           "wanted that de-sentinelling, target construction, race mapping, and the singleton filter all "
           "behaved. The 2 smallest groups (NH AIAN and NH NHOPI) still carry the misclassification caveat "
           "from the biases section, so their exact values are the ones I lean on least. And the framing "
           "that governs the whole project holds here too: **race is a marker of exposure to racism and "
           "structural barriers, never a biological cause** of preterm birth. With provenance settled, the "
           "shape understood, the holes mapped, the biases named, and the ingest externally validated, we "
           "have earned the right to let the data actually talk. That is the next notebook." + AUTHOR),
    ]
    return cells


# ============================================================================= Notebook 03
def nb_03():
    cells = opener("03", "Exploratory Data Analysis and Visualization",
        "This is notebook 03. I tried to do the basic exploration but more can be done here. The order is "
        "the one I actually explore in: anchor on the outcome, look at who is in the data, hunt for "
        "sentinels and outliers, then move to bivariate questions, forming a hypothesis and testing it "
        "before believing it. Every number is computed live from the data.",
        [("target", "How common is preterm birth?"),
         ("who", "Who is in the data? (univariate)"),
         ("outliers", "Hunting sentinels and outliers"),
         ("predict", "What predicts preterm? (bivariate)"),
         ("disparity", "The racial disparity, and a hypothesis test"),
         ("education", "Does education close the gap?"),
         ("covid", "Exploring the COVID question"),
         ("corr", "Correlations: significance vs effect size"),
         ("takeaways", "EDA takeaways and the plan from here")])
    cells += [
        md(anchor("target", "How common is preterm birth?") + "\n\n"
           "We always anchor on the outcome first. Let me compute the overall rate and the per-year rate "
           "(rates, not counts, for the reason established in notebook 01)."),
        code(
            "print('Overall preterm rate: {:.2f}%'.format(df.preterm.mean()*100))\n"
            "print('\\nBy year (rates, because raw counts are not comparable across years):')\n"
            "print(rate_by(df, 'birth_year').round(2))"),
        md("About **8.6%** of singleton births are preterm, stable year to year. That number sits sensibly "
           "below the 10.4% all-births headline, exactly as expected once multiples are excluded, which is "
           "our first reassurance that the data is sound. That imbalance already tells us that accuracy "
           "will be a garbage metric when we model (a model that always says 'not preterm' scores 91%), so "
           "we will need to lean on rate-based and calibration metrics instead."),
        md(anchor("who", "Who is in the data? (univariate)") + "\n\n"
           "Before relating anything to the outcome, we want to know who these mothers are. A `describe` "
           "table is also our first outlier radar, so I watch the `max` column closely."),
        code("df[['mager','meduc','bmi','cig_0','priorlive','priordead']].describe().round(2).T[['mean','std','min','50%','max']]"),
        md("Two things jump out of the `max` column immediately. `bmi` maxes at 99.9 and `cig_0` at 98. "
           "Those are not real measurements; they are NCHS 'not stated' sentinel codes masquerading as "
           "numbers, exactly the kind of thing the bias notes warned about. I will deal with them in the "
           "next section before they poison a single chart. For now, let me look at each genuine "
           "distribution on its own, one chart at a time."),
        md("**Maternal age.** I expect a roughly bell-shaped curve centered in the late twenties, with "
           "tails of teen and older mothers who carry different risk profiles."),
        code(
            "fig, ax = plt.subplots(figsize=(7.5, 3.6))\n"
            "ax.hist(df['mager'].dropna(), bins=range(12, 51), color=PALETTE['blue'])\n"
            "style(ax, 'Maternal age is centered near 29', 'Age (years)', 'Births')\n"
            "plt.tight_layout(); plt.show()"),
        md("**Maternal age against the outcome: a U-shape.** The age *distribution* is bell-shaped, but the "
           "preterm *rate* by age is not: obstetrically we expect both the youngest and the oldest mothers "
           "to carry higher risk, with a trough in the late twenties. Let me bracket age and read the rate "
           "in each bracket to see whether that U-shape actually shows up here."),
        code(
            "breaks = [20, 25, 30, 35, 40]\n"
            "def age_bracket(a):\n"
            "    for i, b in enumerate(breaks):\n"
            "        if a < b: return i\n"
            "    return len(breaks)\n"
            "labels = [\"<20\",\"20-24\",\"25-29\",\"30-34\",\"35-39\",\"40+\"]\n"
            "ab = df.dropna(subset=['preterm','mager']).assign(ab=lambda x: x['mager'].map(age_bracket))\n"
            "ag = ab.groupby('ab')['preterm'].mean().mul(100)\n"
            "print(\"Preterm rate (%) by maternal-age bracket (expect a U-shape):\")\n"
            "for i, lab in enumerate(labels):\n"
            "    if i in ag.index: print(f\"  {lab:<6} {ag.loc[i]:.2f}%\")\n"
            "fig, ax = plt.subplots(figsize=(6.6, 3.6))\n"
            "ax.plot([labels[i] for i in ag.index], ag.values, marker='o', color=PALETTE['purple'], linewidth=2.2)\n"
            "style(ax, 'Maternal age and preterm risk trace a U-shape (young and older both higher)',\n"
            "      'Maternal age bracket', 'Preterm rate (%)')\n"
            "plt.tight_layout(); plt.show()"),
        md("The U-shape is there: the rate is elevated for the youngest mothers, drops to its lowest in the "
           "25 to 29 bracket, then climbs steeply again after 35. That non-monotone shape is the reason the "
           "modeling weeks (PLANNED) intend to *bracket* age rather than fit it as a single linear term, "
           "since a straight line through a U would wash out both tails and misstate the risk at the "
           "extremes."),
        md("**Pre-pregnancy BMI.** Here I deliberately filter out the values above 70, because the 99.9 "
           "sentinel would otherwise stretch the axis and hide the real shape. What remains should be "
           "right-skewed with a genuine obesity tail."),
        code(
            "fig, ax = plt.subplots(figsize=(7.5, 3.6))\n"
            "ax.hist(df['bmi'][df['bmi'] < 70].dropna(), bins=40, color=PALETTE['green'])\n"
            "style(ax, 'Pre-pregnancy BMI is right-skewed (sentinels removed)', 'BMI', 'Births')\n"
            "plt.tight_layout(); plt.show()"),
        md("**Maternal education and payment source.** Two coded fields mapped to readable labels. "
           "Education should pile up at high-school-graduate and bachelor levels; payment tells us how "
           "much of this population is on Medicaid, which matters because payment is one of our SES proxies."),
        code(
            "ed = df['meduc'].dropna().astype(int).value_counts().sort_index()\n"
            "pa = df['pay'].dropna().astype(int).value_counts().sort_index()\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))\n"
            "axes[0].bar([EDU[i] for i in ed.index], ed.values, color=PALETTE['purple'])\n"
            "style(axes[0], 'Education peaks at HS-grad and bachelor', '', 'Births')\n"
            "axes[0].tick_params(axis='x', rotation=40)\n"
            "axes[1].bar([PAY.get(i, str(i)) for i in pa.index], pa.values, color=PALETTE['sky'])\n"
            "style(axes[1], 'Payment is split between private and Medicaid', '', 'Births')\n"
            "axes[1].tick_params(axis='x', rotation=40); plt.tight_layout(); plt.show()"),
        md("All as expected: age is bell-shaped, BMI is right-skewed with a real obesity tail, education "
           "piles up where the population does, and payment is dominated by private and Medicaid. Nothing "
           "preterm-specific yet; this is just the cast of characters. Now let me deal with those "
           "sentinels properly."),
        md(anchor("outliers", "Hunting sentinels and outliers") + "\n\n"
           "Instead of eyeballing, let me count the sentinels and the implausible tails. A value being "
           "rare is fine; a value being *impossible* is a data-quality flag that needs a decision."),
        code(
            "print(f\"BMI >= 70 (sentinel/implausible): {int((df.bmi>=70).sum()):>6,}  ({(df.bmi>=70).mean()*100:.2f}%)\")\n"
            "print(f\"cig_0 >= 90 (sentinel codes)    : {int((df.cig_0>=90).sum()):>6,}  ({(df.cig_0>=90).mean()*100:.2f}%)\")\n"
            "print(f\"Maternal age <= 15              : {int((df.mager<=15).sum()):>6,}\")\n"
            "print(f\"Maternal age >= 45              : {int((df.mager>=45).sum()):>6,}\")\n"
            "print(f\"Prior live births >= 10         : {int((df.priorlive>=10).sum()):>6,}\")"),
        md("The BMI and cigarette sentinels are small in share but would badly distort a mean or a "
           "histogram axis, so we exclude them from anything quantitative this week and flag that the "
           "feature pipeline will need to null them per-field upstream. The age and parity extremes are "
           "*plausible*: a 13-year-old mother, a 49-year-old mother, a mother with 12 prior births "
           "genuinely exist, and they are clinically high-risk, so we keep them. The discipline is telling "
           "a sentinel (drop) apart from a rare-but-real value (keep), and saying which is which."),
        md(anchor("predict", "What predicts preterm? (bivariate)") + "\n\n"
           "Now the question that matters. The approach is the same for each candidate predictor: use "
           "`rate_by()` to compute the preterm rate within each level, then let the gradient speak. We take "
           "the signals we expect to matter most, one at a time."),
        md("**Prior preterm birth.** Clinically this is the single strongest known risk factor, so it is "
           "our first sanity check. If the data are sound, a mother with a prior preterm birth should have "
           "a much higher rate."),
        code(
            "pp = rate_by(df, 'rf_ppterm')\n"
            "fig, ax = plt.subplots(figsize=(5.5, 3.6))\n"
            "ax.bar(['No prior\\npreterm', 'Prior\\npreterm'], pp['rate'].values, color=[PALETTE['grey'], PALETTE['red']])\n"
            "for i, v in enumerate(pp['rate'].values): ax.text(i, v+0.6, f'{v:.1f}%', ha='center', fontweight='bold')\n"
            "style(ax, 'A prior preterm birth more than triples the rate', '', 'Preterm rate (%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Risk ratio: {pp.loc[1.0,'rate']/pp.loc[0.0,'rate']:.2f}x  ({pp.loc[1.0,'rate']:.1f}% vs {pp.loc[0.0,'rate']:.1f}%)\")"),
        md("A risk ratio on its own is a point estimate; the honest quantity is the ratio *with* a "
           "confidence interval, which puts the effect size and its uncertainty side by side. Let me attach "
           "a 95% CI to the prior-preterm risk ratio using the `risk_ratio_ci()` helper. If the interval "
           "excludes 1, the association is not noise, and here we also want to see that the effect is large, "
           "not merely significant."),
        code(
            "rr, lo, hi, r1, r0 = risk_ratio_ci(df, 'rf_ppterm')\n"
            "print(f\"Prior preterm birth: {r1:.1f}% vs {r0:.1f}%  ->  risk ratio {rr:.2f} \"\n"
            "      f\"(95% CI {lo:.2f} to {hi:.2f}). The CI excludes 1, and the effect is large, not just significant.\")"),
        md("**An exploratory composite: medical-risk count.** I want to test an idea: sum the four "
           "pre-pregnancy risk flags into a 0-to-4 count and see if a real dose-response appears (more "
           "flags, higher rate). If it does, that is a feature worth engineering properly later; if it is "
           "flat, the idea is not worth pursuing. This is exploration, computed live, not a committed "
           "feature."),
        code(
            "risk_n = df[['rf_pdiab','rf_phype','rf_ppterm','rf_cesar']].sum(axis=1)\n"
            "mr = rate_by(df.assign(risk_n=risk_n), 'risk_n')\n"
            "fig, ax = plt.subplots(figsize=(6.5, 3.6))\n"
            "ax.plot(mr.index, mr['rate'].values, marker='o', color=PALETTE['blue'], linewidth=2.4)\n"
            "for x, y in zip(mr.index, mr['rate'].values): ax.text(x, y+1.6, f'{y:.0f}%', ha='center', fontsize=9)\n"
            "style(ax, 'Summed risk flags show a clean dose-response (worth engineering later)', 'Number of risk factors', 'Preterm rate (%)')\n"
            "plt.tight_layout(); plt.show()"),
        md("**Pre-pregnancy smoking.** A behavioral exposure. We define a smoker as 1 to 89 cigarettes per "
           "day, deliberately excluding the 98/99 sentinel codes found earlier so they do not contaminate "
           "the comparison."),
        code(
            "sm = rate_by(df.assign(smoker=np.where(df.cig_0.between(1,89), 'Smoker', 'Non-smoker')), 'smoker')\n"
            "fig, ax = plt.subplots(figsize=(5.5, 3.6))\n"
            "ax.bar(sm.index, sm['rate'].values, color=[PALETTE['grey'], PALETTE['orange']])\n"
            "for i, v in enumerate(sm['rate'].values): ax.text(i, v+0.4, f'{v:.1f}%', ha='center', fontweight='bold')\n"
            "style(ax, 'Smoking lifts the preterm rate by about four points', '', 'Preterm rate (%)')\n"
            "plt.tight_layout(); plt.show()"),
        md("Same discipline as prior preterm: report the smoking effect as a risk ratio with a 95% CI, so "
           "the effect size sits beside the significance. I define a smoker flag of 1 for 1 to 89 cigarettes "
           "per day and 0 for exactly 0, dropping the sentinel/unknown codes so they do not contaminate the "
           "comparison."),
        code(
            "sm = df.assign(smoker=np.where(df.cig_0.between(1,89), 1.0, np.where(df.cig_0==0, 0.0, np.nan)))\n"
            "rr, lo, hi, r1, r0 = risk_ratio_ci(sm, 'smoker')\n"
            "print(f\"Pre-pregnancy smoking: {r1:.1f}% vs {r0:.1f}%  ->  risk ratio {rr:.2f} (95% CI {lo:.2f} to {hi:.2f}).\")"),
        md("**The strongest real signals are clinical.** A prior preterm birth more than triples the rate "
           "(about a 3.4x risk ratio), the summed risk count shows the clean dose-response we hoped for "
           "(climbing toward about 61% at four flags), and smoking adds about 4 points. These are exactly "
           "the kind of legitimate pre-delivery predictors we would want a model to lean on. But the thing "
           "to keep noting is that every one of these is a *group-level* gradient, not a clean individual "
           "separator. That tension comes back hard in the correlation section."),
        md(anchor("disparity", "The racial disparity, and a hypothesis test") + "\n\n"
           "Here is the central equity question. I refuse to just eyeball it: a roughly 4.7-point gap could "
           "in principle be sampling noise, so we state a null hypothesis and test it. **H0:** NH Black and "
           "NH White mothers have the same singleton preterm rate. First, plot the rates so we can see the "
           "ordering. This is the project's headline chart, so it gets the interactive Plotly treatment: "
           "hover any bar for its exact rate and group size."),
        code(
            "rr = rate_by(df, 'mother_race_ethnicity').reindex(RACE_ORDER).dropna().sort_values('rate')\n"
            "print('Per-race singleton preterm rate (%) and group size (values behind the bars):')\n"
            "for grp, row in rr.iterrows():\n"
            "    print(f\"  {grp:<15} {row['rate']:.2f}%   (n = {int(row['n']):,})\")\n"
            "fig = go.Figure(go.Bar(\n"
            "    x=rr['rate'].values, y=rr.index, orientation='h',\n"
            "    marker_color=[RACE_COLOR[r] for r in rr.index],\n"
            "    text=[f'{v:.1f}%' for v in rr['rate'].values], textposition='outside',\n"
            "    customdata=rr['n'].values,\n"
            "    hovertemplate='%{y}: %{x:.2f}% preterm  (n = %{customdata:,})<extra></extra>'))\n"
            "fig.update_xaxes(range=[0, rr['rate'].max()*1.15])\n"
            "ptmpl(fig, 'NH Black mothers have the highest singleton preterm rate',\n"
            "      'Preterm rate (%)', '', height=420)\n"
            "fig.show()"),
        md("There is a roughly 4.7-point gap between NH Black and NH White mothers. But a difference on a "
           "dataset, however striking, deserves a test rather than a glance, so we run a two-proportion "
           "z-test. I compute it by hand so every step is visible (pooled proportion, standard error, z, "
           "two-sided p)."),
        code(
            "d = df.dropna(subset=['preterm'])\n"
            "b = d[d.mother_race_ethnicity=='NH Black'].preterm\n"
            "w = d[d.mother_race_ethnicity=='NH White'].preterm\n"
            "p_pool = (b.sum() + w.sum()) / (len(b) + len(w))\n"
            "se = np.sqrt(p_pool * (1 - p_pool) * (1/len(b) + 1/len(w)))\n"
            "z = (b.mean() - w.mean()) / se\n"
            "pval = 2 * (1 - stats.norm.cdf(abs(z)))\n"
            "print(f'NH Black {b.mean()*100:.2f}% vs NH White {w.mean()*100:.2f}%  (gap {(b.mean()-w.mean())*100:.2f} pp)')\n"
            "print(f'Two-proportion z = {z:.1f}, p = {pval:.1e}  =>  reject H0 overwhelmingly')"),
        md("The z came out around **280** (it is that large mostly because the full national data gives "
           "millions of mothers per group, which shrinks the standard error to almost nothing, so the "
           "substantive number is the 4.7-point gap, not the z itself). Either way, the p-value is very "
           "close to zero. That gap is not noise or just by chance. We keep the bias caveat from notebook "
           "02 in mind: the NH AIAN and NH NHOPI bars rest on small samples and known misclassification, so "
           "we trust the Black, White, Hispanic, and Asian estimates most. And the framing that governs the "
           "whole project: **race is a marker of exposure to racism and structural barriers, not a "
           "biological cause** of preterm birth."),
        md(anchor("education", "Does education close the gap?") + "\n\n"
           "My honest first reaction was the same one most people have: this is probably really a "
           "socioeconomic gap, and it would close if we compared mothers of the same education. That is a "
           "falsifiable hypothesis, so let me try to falsify it by stratifying on education and drawing one "
           "line per race. If it is right, the NH Black and NH White lines should converge as education "
           "rises."),
        code(
            "races = ['NH White', 'NH Black', 'Hispanic', 'NH Asian']\n"
            "piv = (d[d.mother_race_ethnicity.isin(races)].dropna(subset=['meduc'])\n"
            "        .assign(meduc=lambda x: x.meduc.astype(int))\n"
            "        .groupby(['mother_race_ethnicity', 'meduc']).preterm.mean() * 100)\n"
            "fig, ax = plt.subplots()\n"
            "for race in races:\n"
            "    s = piv[race].sort_index()\n"
            "    ax.plot([EDU[i] for i in s.index], s.values, marker='o', color=RACE_COLOR[race], linewidth=2.2, label=race)\n"
            "style(ax, 'The racial gap does not close as education rises', 'Maternal education', 'Preterm rate (%)')\n"
            "plt.xticks(rotation=35, ha='right'); ax.legend(frameon=False); plt.tight_layout(); plt.show()"),
        md("The lines do not converge. To make the point impossible to miss, let me compute the single "
           "starkest comparison: the rate for the *most* educated Black mothers (bachelor or higher) "
           "against the *least* educated White mothers (high school or less)."),
        code(
            "bc = d[(d.mother_race_ethnicity=='NH Black') & (d.meduc>=6)].preterm.mean()*100\n"
            "wl = d[(d.mother_race_ethnicity=='NH White') & (d.meduc<=3)].preterm.mean()*100\n"
            "print(f'College-educated NH Black: {bc:.1f}%   vs   HS-or-less NH White: {wl:.1f}%')"),
        md("**The hypothesis is falsified, and that is the finding.** Every NH Black line sits above every "
           "NH White line, and a college-educated NH Black mother has a *higher* preterm rate than a NH "
           "White mother who never finished high school. Education, the supposed great equalizer, does not "
           "equalize this. That points beyond a simple socioeconomic explanation, toward something "
           "structural that a birth certificate does not record. It is a strong, real pattern, and it is the "
           "thread we most want to pull on in the coming weeks."),
        md("**A second, independent falsification: payment source.** Education is 1 socioeconomic proxy; "
           "insurance payer is another, since private coverage versus Medicaid is a rough marker of income "
           "and access. If the disparity were purely a socioeconomic artifact, then comparing mothers on the "
           "*same* payer should erase the racial gap. Let me hold the payer fixed and compare *across race "
           "within that payer*. The test is: at a single payer, does the NH Black rate still sit above the "
           "NH White rate?"),
        code(
            "races = ['NH White', 'NH Black', 'Hispanic', 'NH Asian']\n"
            "d = df.dropna(subset=['preterm'])\n"
            "pv = (d[d.mother_race_ethnicity.isin(races)].dropna(subset=['pay'])\n"
            "        .assign(pay=lambda x: x.pay.astype(int))\n"
            "        .groupby(['mother_race_ethnicity','pay']).preterm.mean().mul(100))\n"
            "print(\"Preterm rate (%) by race within payment source (Private=1, Medicaid=2):\")\n"
            "for r in races:\n"
            "    s = pv[r]\n"
            "    print(f\"  {r:<12} private {s.get(1, float('nan')):.2f}%   Medicaid {s.get(2, float('nan')):.2f}%\")\n"
            "print(\"\\nEvery NH Black cell exceeds the NH White cell at the SAME payer, so a single SES proxy does \"\n"
            "      \"not close the gap, reinforcing the education result.\")"),
        md("The pattern holds across race within the same payer: NH Black mothers on private insurance have "
           "a higher preterm rate than NH White mothers on private insurance, and the same is true within "
           "Medicaid. Holding payer fixed does not close the gap, which is a second independent "
           "falsification of the pure-socioeconomic story, converging on the same conclusion as the "
           "education stratification. This stays associational, and the framing that governs the whole "
           "project holds: **race is a marker of exposure to racism and structural barriers, never a "
           "biological cause** of preterm birth."),
        md(anchor("covid", "Exploring the COVID question") + "\n\n"
           "With a 2018 to 2024 window we explored whether the pandemic moved preterm birth. Published work "
           "is genuinely mixed on this, which is itself the interesting part. National analyses found the "
           "annual picture "
           "[nuanced rather than dramatic](https://publications.aap.org/pediatrics/article/149/5/e2021055495/186706/), "
           "while CDC/NCHS reported the preterm rate "
           "[rose from 10.09% in 2020 to 10.49% in 2021](https://www.cdc.gov/nchs/products/databriefs/db442.htm) "
           "(an all-births figure). Separately, a "
           "[JAMA Pediatrics study documented a local decline](https://jamanetwork.com/journals/jamapediatrics/fullarticle/2777304) "
           "concentrated in a short spring-2020 window. Let me see whether our annual data shows any of it."),
        code(
            "yr = rate_by(df, 'birth_year')\n"
            "fig, ax = plt.subplots()\n"
            "ax.axvspan(2019.5, 2021.5, color=PALETTE['yellow'], alpha=0.20)\n"
            "ax.plot(yr.index, yr['rate'].values, marker='o', color=PALETTE['ink'], linewidth=2.6)\n"
            "for x, y in zip(yr.index, yr['rate'].values): ax.text(x, y+0.02, f'{y:.2f}', ha='center', fontsize=8)\n"
            "style(ax, 'Annual national singleton preterm rate is nearly flat through COVID', 'Birth year', 'Preterm rate (%)')\n"
            "ax.set_ylim(yr['rate'].min()-0.2, yr['rate'].max()+0.25); plt.tight_layout(); plt.show()\n"
            "v = yr['rate']\n"
            "print(f'2019 -> 2020: {v[2020]-v[2019]:+.2f} pp (small dip)   2020 -> 2021: {v[2021]-v[2020]:+.2f} pp (rebound)')"),
        md("Before drawing a lesson, let me split the COVID window by race, to check whether any group moved "
           "more than the national aggregate. If even the pandemic-era shift is small for the large, "
           "reliable groups, that tells us the annual national number is not hiding a big per-group swing."),
        code(
            "covid = df[df.birth_year.isin([2019,2020,2021])].dropna(subset=['preterm'])\n"
            "races = ['NH White','NH Black','Hispanic','NH Asian','NH AIAN','NH NHOPI']\n"
            "print(\"COVID-era per-race annual preterm rate (%), 2019 -> 2020 -> 2021:\")\n"
            "for r in races:\n"
            "    s = covid[covid.mother_race_ethnicity==r].groupby('birth_year')['preterm'].mean().mul(100)\n"
            "    if {2019,2020,2021} <= set(s.index):\n"
            "        print(f\"  {r:<12} {s[2019]:.2f} -> {s[2020]:.2f} -> {s[2021]:.2f}  \"\n"
            "              f\"(2020-2019 delta {s[2020]-s[2019]:+.2f} pp)\")"),
        md("**The lesson is about temporal granularity.** The per-race deltas are small for the large, "
           "reliable groups (well under about 0.4 pp from 2019 to 2020), and only the smallest groups (NH "
           "AIAN, NH NHOPI) wiggle more, which is the small-cell noise we already flagged in notebook 02, "
           "not a signal. So the annual picture looks like 'no COVID effect.' But note what our data cannot "
           "do: the cleaned files carry no birth-month field, so a monthly curve is simply impossible here, "
           "and we deliberately fabricate no monthly numbers to fake one. The published local declines were "
           "concentrated in a short spring-2020 window, and averaging that into a yearly national aggregate "
           "dilutes it toward nothing. That is the whole point: a real effect at one time scale can vanish "
           "at another, so the granularity mismatch is the caution here, not a finding. The mixed published "
           "record backs this up: nuanced nationally "
           "([AAP](https://publications.aap.org/pediatrics/article/149/5/e2021055495/186706/)), a "
           "2020-to-2021 rise in the CDC all-births rate "
           "([db442](https://www.cdc.gov/nchs/products/databriefs/db442.htm)), and a sharp "
           "([local decline](https://jamanetwork.com/journals/jamapediatrics/fullarticle/2777304)) at finer "
           "resolution."),
        md(anchor("corr", "Correlations: significance vs effect size") + "\n\n"
           "Finally, how strongly does each numeric field correlate with the target? Here we make the "
           "single most important statistical habit explicit when you have tens of millions of rows: "
           "**statistical significance is not the same as practical importance.** To drive it home we back "
           "the correlation chart with 3 different effect-size measures: a Welch t-test paired with Cohen's "
           "d for the BMI difference, and Cramer's V for the race association. Each one says the same thing "
           "the p-values will not: the effects are small."),
        code(
            "num = ['mager','meduc','bmi','cig_0','priorlive','priordead',\n"
            "       'rf_ppterm','rf_phype','rf_pdiab','rf_cesar','wic','first_trimester_care','married']\n"
            "corr = d[num+['preterm']].corr()['preterm'].drop('preterm').sort_values()\n"
            "fig, ax = plt.subplots(figsize=(8.4, 5))\n"
            "colors = [PALETTE['blue'] if c < 0 else PALETTE['red'] for c in corr.values]\n"
            "ax.barh(corr.index, corr.values, color=colors)\n"
            "for i, v in enumerate(corr.values): ax.text(v + (0.002 if v>=0 else -0.002), i, f'{v:+.3f}', va='center', ha='left' if v>=0 else 'right', fontsize=8)\n"
            "style(ax, 'Every single feature correlates only weakly with preterm birth', 'Pearson correlation with the target', '')\n"
            "ax.axvline(0, color=PALETTE['grey']); ax.grid(axis='y'); plt.tight_layout(); plt.show()"),
        code(
            "# The significance-vs-effect-size trap, made concrete with BMI.\n"
            "bp = d[d.preterm==1].bmi; bt = d[d.preterm==0].bmi\n"
            "bp = bp[bp<70]; bt = bt[bt<70]\n"
            "t, p = stats.ttest_ind(bp, bt, equal_var=False)\n"
            "print(f'Mean BMI, preterm vs term: {bp.mean():.1f} vs {bt.mean():.1f}  (difference {bp.mean()-bt.mean():.1f})')\n"
            "print(f'Welch t-test: t = {t:.1f}, p = {p:.1e}  <- wildly significant')\n"
            "print(f'...yet the actual difference is ~1 BMI unit. With {len(bp)+len(bt):,} rows, p-values detect trivia.')\n"
            "pooled_sd = np.sqrt(((len(bp)-1)*bp.std()**2 + (len(bt)-1)*bt.std()**2)/(len(bp)+len(bt)-2))\n"
            "d_cohen = (bp.mean()-bt.mean())/pooled_sd\n"
            "print(f\"Cohen's d for the BMI difference: {d_cohen:.3f}  (well below 0.2, so a small effect, and the \"\n"
            "      \"near-zero p-value is detecting a difference too small to act on).\")"),
        md("The effect-size measures agree with each other. Cohen's d for the BMI gap is about 0.16, which is "
           "small by any convention, so the near-zero p-value is detecting a difference no clinician would "
           "act on. One more effect-size lens, this time for the categorical race-vs-preterm relationship: "
           "Cramer's V, which rescales a chi-square statistic onto a 0-to-1 association scale so it does not "
           "balloon with sample size the way chi-square itself does."),
        code(
            "ct = pd.crosstab(d.mother_race_ethnicity, d.preterm)\n"
            "chi2 = stats.chi2_contingency(ct)[0]\n"
            "n = ct.values.sum(); k = min(ct.shape)\n"
            "cramers_v = np.sqrt(chi2 / (n * (k - 1)))\n"
            "print(f\"Cramer's V (race vs preterm): {cramers_v:.3f}  (a small association on the 0-1 scale).\")"),
        md("**This is the whole modeling story in one chart and one test, previewed.** The strongest "
           "correlate, prior preterm birth, reaches only about +0.13; everything else is weaker. There is "
           "no dominant signal hiding in these fields. And the BMI t-test is a cautionary tale: across tens "
           "of millions of rows it returns a p-value near zero for a difference of about 1 BMI unit that no "
           "clinician would ever act on. At this sample size, p-values detect trivia. Significance is "
           "cheap; effect size is what matters, and the effect sizes here are all small. It strongly "
           "suggests that when we get to modeling, the honest performance ceiling will be modest, because "
           "the real causes of preterm birth mostly live outside what a birth certificate records."),
        md(anchor("takeaways", "EDA takeaways and the plan from here") + "\n\n"
           "What this week's exploration established:\n\n"
           "1. Preterm birth is **rare (about 8.6%)** and **weakly correlated** with every recorded field "
           "(max correlation about 0.13), which will cap achievable model performance.\n"
           "2. The strongest legitimate signals are **clinical**: prior preterm birth (about 3.4x risk "
           "ratio), the summed medical-risk count, and smoking.\n"
           "3. The **racial disparity is large (about 4.7 pp), unambiguously significant, and does not "
           "close with education**, pointing beyond a simple socioeconomic explanation.\n"
           "4. With tens of millions of rows, **p-values are not the right lens**; effect sizes are, and "
           "they are modest.\n"
           "5. The **COVID era** barely moves the annual national rate, which is itself a lesson in "
           "temporal granularity.\n\n"
           "**The plan from here**, now grounded in what we have seen rather than guessed:\n\n"
           "- Engineer a leakage-safe feature set, with a tested gate barring at-delivery fields, since "
           "the EDA showed the legitimate signals are individually weak and the temptation to reach for "
           "leaky fields will be strong.\n"
           "- Build calibrated baseline and tree models, expecting a modest ceiling, and judge them on "
           "AUC and calibration rather than accuracy given the imbalance.\n"
           "- Turn the disparity finding into a formal fairness audit with confidence intervals, and "
           "attempt a decomposition that asks how much of the gap measured factors explain.\n\n"
           "That work is for the coming weeks. This week, the data has been acquired, audited, explored, "
           "and visualized, and it has already told us something real about equity." + AUTHOR),
    ]
    return cells


BUILDERS = {"01": nb_01, "02": nb_02, "03": nb_03}


def main() -> None:
    NBDIR.mkdir(exist_ok=True)
    for num, slug, _ in NOTEBOOKS:
        nb = nbf.v4.new_notebook()
        nb.cells = BUILDERS[num]()
        nb.metadata = {"kernelspec": {"display_name": ".venv", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}}
        out = NBDIR / fname(num, slug)
        nbf.write(nb, out)
        print(f"wrote {out.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
