"""EquiBirth AI: a continuous-scroll, publication-grade equity dashboard (educational, NOT clinical).

This is a single long-form data-journalism page, not a click-through of disconnected panels. It tells one
argument from top to bottom: preterm birth is only modestly predictable from birth-certificate data, the
Black-White disparity is large and does not close with education, and careful, calibrated, transparently
audited ML still yields population-level equity insight.

Design: cohesive typography and color, dark high-contrast text on every chart (no grey-on-white labels),
a sticky section nav, key-insight callouts, interactive Plotly figures, and a handful of widgets so the
reader can explore. Guardrails (SPEC FR11): prominent not-a-clinical-tool disclaimer, race-as-a-social
-construct framing, and deliberately NO individual risk calculator, NO race-specific thresholds, and NO
race as a modifiable what-if input. No em dashes anywhere.

Run: ./.venv/bin/streamlit run src/equibirth/app/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "reports" / "dashboard_data.json"
BUNDLE_W5_9 = ROOT / "reports" / "weeks5to9_dashboard_data.json"
PARQUET = ROOT / "data" / "processed" / "modeling_2019_2024.parquet"

# ------------------------------------------------------------------------------- guardrails
# WHATIF_FEATURES explicitly excludes any race columns. Race is never a modifiable what-if input.
WHATIF_FEATURES = [
    "mager", "meduc", "bmi", "cig_0", "priorlive", "priordead",
    "medical_risk_count", "rf_ppterm", "rf_phype", "rf_pdiab", "rf_cesar",
    "wic", "first_trimester_care", "married",
]

DISCLAIMER_TEXT = (
    "This is an educational and research demonstration, not a clinical tool. "
    "Nothing here should be used to screen, ration, prioritize, or deny care to any person."
)

# ----------------------------------------------------------------------------- color and type system
INK = "#16203A"        # near-black navy: ALL text uses this or white, never grey
INK_SOFT = "#3D4663"   # secondary text, still high contrast
PAPER = "#FBFBFD"
BAND = "#F1F3F8"
LINE = "#8893A8"
C = {"blue": "#0B6FB8", "orange": "#E07B00", "green": "#0A8754", "red": "#D7263D",
     "purple": "#8E44AD", "sky": "#3FA7D6", "yellow": "#F2C14E", "teal": "#117A65"}
RACE_ORDER = ["NH White", "NH Asian", "Hispanic", "NH Multiracial", "NH NHOPI", "NH AIAN", "NH Black"]
RACE_COLOR = {r: "#9AA3B2" for r in RACE_ORDER}
RACE_COLOR.update({"NH White": C["blue"], "NH Black": C["red"], "Hispanic": C["green"],
                   "NH Asian": C["orange"]})
EDU = {1: "<8th grade", 2: "9-12, no diploma", 3: "HS grad/GED", 4: "Some college",
       5: "Associate", 6: "Bachelor", 7: "Master", 8: "Doctorate/Prof"}


# ----------------------------------------------------------------------------------------- data load
@st.cache_data
def load_bundle() -> dict:
    return json.loads(BUNDLE.read_text())


@st.cache_data
def load_weeks5to9() -> dict | None:
    if BUNDLE_W5_9.exists():
        return json.loads(BUNDLE_W5_9.read_text())
    return None


def _restore_eda(e: dict) -> dict:
    """Rehydrate JSON-stored EDA aggregates into the shapes the chart functions expect."""
    e = dict(e)
    e["age_hist"] = (np.array(e["age_hist"]["counts"]), np.array(e["age_hist"]["edges"]))
    e["bmi_hist"] = (np.array(e["bmi_hist"]["counts"]), np.array(e["bmi_hist"]["edges"]))
    e["edu_dist"] = {int(k): v for k, v in e["edu_dist"].items()}
    e["corr"] = {k: float(v) for k, v in e["corr"].items()}
    return e


@st.cache_data
def load_eda() -> dict:
    """Return the EDA aggregates.

    On a deployment (Streamlit Cloud) the gitignored parquet is absent, so we prefer the precomputed
    'eda' block committed inside reports/dashboard_data.json. We fall back to computing live from the
    parquet when it is present (local development), and to {} only if neither exists.
    """
    bundle = load_bundle()
    if "eda" in bundle:
        return _restore_eda(bundle["eda"])
    if not PARQUET.exists():
        return {}
    df = pd.read_parquet(PARQUET, columns=[
        "birth_year", "mager", "meduc", "bmi", "cig_0", "priorlive", "medical_risk_count",
        "rf_ppterm", "rf_phype", "rf_pdiab", "rf_cesar", "wic", "first_trimester_care", "married",
        "priordead", "mother_race_ethnicity", "preterm"])
    d = df.dropna(subset=["preterm"])

    def rate(col, frame=d):
        g = frame.groupby(col)["preterm"].agg(rate="mean", n="size")
        g["rate"] *= 100
        return g

    num = ["mager", "meduc", "bmi", "cig_0", "priorlive", "priordead", "medical_risk_count",
           "rf_ppterm", "rf_phype", "rf_pdiab", "rf_cesar", "wic", "first_trimester_care", "married"]
    edu_race = (d[d.mother_race_ethnicity.isin(RACE_ORDER)]
                .dropna(subset=["meduc"]).assign(meduc=lambda x: x.meduc.astype(int))
                .groupby(["mother_race_ethnicity", "meduc"])["preterm"].mean().mul(100)
                .rename("rate").reset_index())
    return {
        "n_rows": int(len(df)), "n_cols": 108, "overall_rate": float(d.preterm.mean() * 100),
        "n_preterm": int(d.preterm.sum()), "n_missing_target": int(df.preterm.isna().sum()),
        "rate_by_race": rate("mother_race_ethnicity").reset_index().to_dict("records"),
        "by_year": rate("birth_year").reset_index().to_dict("records"),
        "prior_preterm": rate("rf_ppterm").reset_index().to_dict("records"),
        "medical_risk": rate("medical_risk_count").reset_index().to_dict("records"),
        "smoking": d.assign(s=np.where(d.cig_0 > 0, "Smoker", "Non-smoker")).groupby("s")["preterm"]
                    .mean().mul(100).to_dict(),
        "corr": d[num + ["preterm"]].corr()["preterm"].drop("preterm").sort_values().to_dict(),
        "age_hist": np.histogram(d["mager"].dropna(), bins=range(12, 51)),
        "bmi_hist": np.histogram(d["bmi"][d.bmi < 70].dropna(), bins=40),
        "edu_dist": d["meduc"].dropna().astype(int).value_counts().sort_index().to_dict(),
        "edu_race": edu_race.to_dict("records"),
    }


# --------------------------------------------------------------------------------------- chart theme
def theme(fig: go.Figure, title: str, xtitle: str, ytitle: str, height: int = 430) -> go.Figure:
    """Every text element is dark and high-contrast. Gridlines are faint; labels never are."""
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=19, color=INK, family="Georgia, serif")),
        xaxis_title=xtitle, yaxis_title=ytitle, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, system-ui, sans-serif", size=14, color=INK),
        margin=dict(l=64, r=24, t=58, b=52), height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK, size=13)),
        hoverlabel=dict(font=dict(color="white", family="Inter, system-ui, sans-serif")))
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont=dict(color=INK, size=13),
                     title_font=dict(color=INK, size=14))
    fig.update_yaxes(showgrid=True, gridcolor="#E6E9F0", zeroline=False,
                     tickfont=dict(color=INK, size=13), title_font=dict(color=INK, size=14))
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ------------------------------------------------------------------------------------- HTML snippets
def callout(kind: str, title: str, body: str) -> None:
    color = {"insight": C["blue"], "warn": C["red"], "research": C["purple"], "good": C["green"]}[kind]
    icon = {"insight": "Key insight", "warn": "Caution", "research": "From the research",
            "good": "What works"}[kind]
    st.markdown(
        f"""<div class="callout" style="border-left:5px solid {color}">
        <div class="callout-tag" style="color:{color}">{icon}</div>
        <div class="callout-title">{title}</div>
        <div class="callout-body">{body}</div></div>""", unsafe_allow_html=True)


def source(text: str) -> None:
    st.markdown(f'<p class="source">{text}</p>', unsafe_allow_html=True)


def section(anchor: str, kicker: str, heading: str, lead: str) -> None:
    st.markdown(
        f"""<div id="{anchor}" class="anchor"></div>
        <div class="kicker">{kicker}</div>
        <h2 class="section-h">{heading}</h2>
        <p class="lead">{lead}</p>""", unsafe_allow_html=True)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Georgia&display=swap');
:root { --ink:#16203A; --ink-soft:#3D4663; }
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; color: #16203A; }
.block-container { max-width: 1060px; padding-top: 0.5rem; padding-bottom: 4rem; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }

/* hero */
.hero { background: linear-gradient(135deg, #0E1A33 0%, #1C2E55 55%, #243B6B 100%);
        border-radius: 18px; padding: 44px 46px 38px; margin: 6px 0 14px; color: #fff; }
.hero h1 { font-size: 2.9rem; line-height: 1.05; font-weight: 800; margin: 0 0 10px; color: #fff;
           letter-spacing: -0.5px; }
.hero .sub { font-size: 1.22rem; color: #D7E0F4; font-weight: 500; margin: 0 0 22px; max-width: 760px; }
.hero .chips { display: flex; flex-wrap: wrap; gap: 14px; }
.chip { background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.22);
        border-radius: 12px; padding: 12px 18px; min-width: 150px; }
.chip .num { font-size: 1.7rem; font-weight: 800; color: #fff; line-height: 1; }
.chip .lab { font-size: 0.82rem; color: #C5D2EE; margin-top: 5px; }

/* sticky nav */
.navbar { position: sticky; top: 0; z-index: 99; background: rgba(251,251,253,0.94);
          backdrop-filter: blur(8px); border-bottom: 1px solid #E2E6EF; padding: 9px 2px;
          margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 6px 18px; }
.navbar a { color: #3D4663; text-decoration: none; font-size: 0.86rem; font-weight: 600; }
.navbar a:hover { color: #0B6FB8; }

/* section */
.anchor { position: relative; top: -70px; visibility: hidden; }
.kicker { text-transform: uppercase; letter-spacing: 2px; font-size: 0.76rem; font-weight: 700;
          color: #0B6FB8; margin: 30px 0 2px; }
.section-h { font-family: 'Georgia', serif; font-size: 2.0rem; font-weight: 700; color: #16203A;
             margin: 0 0 8px; line-height: 1.12; }
.lead { font-size: 1.08rem; color: #3D4663; line-height: 1.6; margin: 0 0 8px; }
.stMarkdown p { color: #2A3450; font-size: 1.02rem; line-height: 1.62; }

/* callouts */
.callout { background: #fff; border-radius: 10px; padding: 16px 20px; margin: 14px 0;
           box-shadow: 0 1px 3px rgba(20,30,60,0.06); }
.callout-tag { text-transform: uppercase; letter-spacing: 1.4px; font-size: 0.72rem; font-weight: 800; }
.callout-title { font-size: 1.12rem; font-weight: 700; color: #16203A; margin: 3px 0 5px; }
.callout-body { font-size: 1.0rem; color: #2A3450; line-height: 1.58; }
.callout-body a { color: #0B6FB8; }

.source { font-size: 0.82rem !important; color: #59617A !important; margin-top: 2px; }
.disclaimer { background: #FCEDEE; border-left: 6px solid #D7263D; border-radius: 8px;
              padding: 14px 18px; margin: 4px 0 6px; color: #5A1620; font-size: 0.98rem; }
.disclaimer b { color: #A4121F; }

/* metric cards */
.mcard { background:#fff; border:1px solid #E4E8F1; border-radius:12px; padding:18px 20px; height:100%; }
.mcard .mnum { font-size:1.95rem; font-weight:800; line-height:1; }
.mcard .mlab { font-size:0.9rem; color:#3D4663; margin-top:6px; font-weight:600; }
.mcard .msub { font-size:0.8rem; color:#59617A; margin-top:3px; }
hr { border: none; border-top: 1px solid #E2E6EF; margin: 22px 0; }
.foot { color:#3D4663; font-size:0.95rem; margin-top:8px; }
.foot a { color:#0B6FB8; }
</style>
"""


# --------------------------------------------------------------------------------------- chart specs
def fig_rate_by_race(rows) -> go.Figure:
    rows = sorted([r for r in rows if r["mother_race_ethnicity"] in RACE_ORDER], key=lambda r: r["rate"])
    labels = [r["mother_race_ethnicity"] for r in rows]
    rates = [r["rate"] * 100 for r in rows]  # bundle stores the rate as a fraction
    fig = go.Figure(go.Bar(
        x=rates, y=labels, orientation="h", marker_color=[RACE_COLOR[lab] for lab in labels],
        text=[f"{v:.1f}%" for v in rates], textposition="outside",
        textfont=dict(color=INK, size=13), customdata=[r["n"] for r in rows],
        hovertemplate="%{y}: %{x:.2f}%<br>n = %{customdata:,}<extra></extra>"))
    theme(fig, "NH Black mothers have the highest singleton preterm rate", "Preterm rate (%)", "", 400)
    fig.update_xaxes(range=[0, max(rates) * 1.16])
    return fig


def fig_edu_race(records, chosen) -> go.Figure:
    fig = go.Figure()
    for race in chosen:
        pts = sorted([r for r in records if r["mother_race_ethnicity"] == race], key=lambda r: r["meduc"])
        fig.add_trace(go.Scatter(
            x=[EDU[r["meduc"]] for r in pts], y=[r["rate"] for r in pts], mode="lines+markers",
            name=race, line=dict(color=RACE_COLOR.get(race, "#9AA3B2"), width=3),
            marker=dict(size=8), hovertemplate=race + ": %{y:.1f}% (%{x})<extra></extra>"))
    theme(fig, "The racial gap does not close as education rises", "Maternal education", "Preterm rate (%)")
    return fig


def fig_corr(corr: dict) -> go.Figure:
    items = sorted(corr.items(), key=lambda kv: kv[1])
    names = {"rf_ppterm": "Prior preterm birth", "medical_risk_count": "Medical-risk count",
             "rf_phype": "Pre-preg hypertension", "rf_pdiab": "Pre-preg diabetes", "bmi": "BMI",
             "rf_cesar": "Prior cesarean", "priorlive": "Prior live births", "cig_0": "Smoking",
             "mager": "Maternal age", "priordead": "Prior child deaths", "wic": "WIC",
             "first_trimester_care": "1st-trimester care", "meduc": "Education", "married": "Married"}
    labels = [names.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    cols = [C["blue"] if v < 0 else C["red"] for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=cols,
        text=[f"{v:+.3f}" for v in vals], textposition="outside", textfont=dict(color=INK, size=12),
        hovertemplate="%{y}: r = %{x:+.3f}<extra></extra>"))
    theme(fig, "Every feature correlates only weakly with preterm birth",
          "Pearson correlation with the target", "", 480)
    fig.update_xaxes(range=[min(vals) - 0.03, max(vals) + 0.05], zeroline=True, zerolinecolor=LINE)
    return fig


def fig_temporal(rows, highlight) -> go.Figure:
    fig = go.Figure()
    for race in RACE_ORDER:
        pts = sorted([r for r in rows if r["mother_race_ethnicity"] == race and r["rate"] is not None],
                     key=lambda r: r["birth_year"])
        if not pts:
            continue
        on = race in highlight
        fig.add_trace(go.Scatter(
            x=[p["birth_year"] for p in pts], y=[p["rate"] * 100 for p in pts], mode="lines+markers",
            name=race, line=dict(color=RACE_COLOR[race], width=3.4 if on else 1.4),
            opacity=1.0 if on else 0.45, marker=dict(size=7 if on else 4),
            hovertemplate=race + ": %{y:.2f}% (%{x})<extra></extra>"))
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor=C["yellow"], opacity=0.16, line_width=0)
    fig.add_annotation(x=2020.5, y=1.0, yref="paper", text="COVID era", showarrow=False,
                       font=dict(color=INK, size=12))
    theme(fig, "The annual national signal barely moves through COVID", "Birth year", "Preterm rate (%)")
    return fig


def fig_ceiling(ablation: dict, leaky: float) -> go.Figure:
    order = ["race_only", "demographic", "demographic+SES", "full_clinical"]
    names = {"race_only": "Race only", "demographic": "+ Demographics",
             "demographic+SES": "+ SES proxies", "full_clinical": "+ Clinical history"}
    xs = [names[k] for k in order if k in ablation]
    ys = [ablation[k]["auc_roc"] for k in order if k in ablation]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=C["blue"], text=[f"{v:.3f}" for v in ys],
                           textposition="outside", textfont=dict(color=INK, size=13),
                           hovertemplate="%{x}: AUC %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dot", line_color=LINE,
                  annotation_text="chance", annotation_font=dict(color=INK, size=12))
    fig.add_hline(y=leaky, line_dash="dash", line_color=C["red"],
                  annotation_text=f"leaky post-delivery model ({leaky:.3f})",
                  annotation_font=dict(color=C["red"], size=12), annotation_position="top left")
    theme(fig, "Each clean feature layer adds little; the honest ceiling is intrinsic",
          "Feature set (cumulative)", "Held-out AUC-ROC")
    fig.update_yaxes(range=[0.45, 0.97])
    return fig


def fig_cluster_rates(clusters) -> go.Figure:
    cl = sorted([c for c in clusters if c["n"] >= 100], key=lambda c: c["preterm_rate"])
    xs = [f"Cluster {c['cluster']}" for c in cl]
    ys = [c["preterm_rate"] * 100 for c in cl]
    ns = [c["n"] for c in cl]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=C["purple"], customdata=ns,
                           text=[f"{v:.1f}%" for v in ys], textposition="outside",
                           textfont=dict(color=INK, size=12),
                           hovertemplate="%{x}: %{y:.1f}% preterm<br>n = %{customdata:,}<extra></extra>"))
    theme(fig, "Clusters span a 4x range in preterm rate",
          "Feature-space cluster (KMeans)", "Preterm rate (%)")
    fig.update_yaxes(range=[0, max(ys) * 1.15])
    return fig


def fig_cluster_heatmap(clusters) -> go.Figure:
    cl = sorted([c for c in clusters if c["n"] >= 100], key=lambda c: c["preterm_rate"])
    feats = ["rf_ppterm", "medical_risk_count", "bmi", "mager", "priorlive", "cig_0", "wic"]
    flab = {"rf_ppterm": "Prior preterm", "medical_risk_count": "Medical risk", "bmi": "BMI",
            "mager": "Maternal age", "priorlive": "Prior births", "cig_0": "Smoking", "wic": "WIC"}
    z, txt = [], []
    for f in feats:
        col = [c["feature_means"].get(f, 0) for c in cl]
        lo, hi = min(col), max(col)
        z.append([(v - lo) / (hi - lo) if hi > lo else 0.5 for v in col])
        txt.append([f"{v:.2f}" for v in col])
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"C{c['cluster']}" for c in cl], y=[flab[f] for f in feats], text=txt,
        texttemplate="%{text}", textfont=dict(color=INK, size=11),
        colorscale=[[0, "#EAF2FB"], [1, C["blue"]]], showscale=False,
        hovertemplate="%{y} in %{x}: %{text}<extra></extra>"))
    theme(fig, "Cluster profiles are clinical, not racial",
          "Cluster (low to high preterm rate)", "", 360)
    fig.update_yaxes(showgrid=False)
    return fig


def fig_decomposition(mf: dict) -> go.Figure:
    shares = [mf["explained_share_primary"] * 100, mf["unexplained_share_primary"] * 100]
    fig = go.Figure(go.Bar(
        x=["Explained by measured SES", "Unexplained (lower bound)"], y=shares,
        marker_color=[C["green"], C["red"]], text=[f"{v:.0f}%" for v in shares], textposition="outside",
        textfont=dict(color=INK, size=14), hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
    theme(fig, "Most of the gap is unexplained by SES",
          "", "Share of the raw gap (%)", 380)
    fig.update_yaxes(range=[0, 82])
    return fig


def fig_fairness(mf: dict) -> go.Figure:
    vals = [1.0, mf["disparate_impact_black"]]
    fig = go.Figure(go.Bar(
        x=["Detection (equal opportunity ok)", "Flagging frequency (disparate impact)"], y=vals,
        marker_color=[C["green"], C["red"]], text=[f"{v:.2f}x" for v in vals], textposition="outside",
        textfont=dict(color=INK, size=14), hovertemplate="%{x}: %{y:.2f}x vs NH White<extra></extra>"))
    fig.add_hline(y=1.0, line_dash="dot", line_color=LINE)
    theme(fig, "Higher detection bought with more over-flagging", "",
          "Ratio, NH Black vs NH White", 380)
    fig.update_yaxes(range=[0, mf["disparate_impact_black"] + 0.5])
    return fig


def fig_dist_age(hist) -> go.Figure:
    counts, edges = hist
    fig = go.Figure(go.Bar(x=list(edges[:-1]), y=list(counts), marker_color=C["blue"],
                           hovertemplate="age %{x}: %{y:,} births<extra></extra>"))
    theme(fig, "Maternal age is centered near 29", "Age (years)", "Births", 360)
    return fig


def fig_dist_bmi(hist) -> go.Figure:
    counts, edges = hist
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    fig = go.Figure(go.Bar(x=centers, y=list(counts), marker_color=C["green"],
                           hovertemplate="BMI %{x:.0f}: %{y:,} births<extra></extra>"))
    theme(fig, "Pre-pregnancy BMI is right-skewed (99.9 sentinel removed)", "BMI", "Births", 360)
    return fig


def fig_dist_edu(edu_dist) -> go.Figure:
    xs = [EDU[k] for k in sorted(edu_dist)]
    ys = [edu_dist[k] for k in sorted(edu_dist)]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=C["purple"],
                           hovertemplate="%{x}: %{y:,} births<extra></extra>"))
    theme(fig, "Education concentrates at high-school and bachelor levels", "", "Births", 360)
    fig.update_xaxes(tickangle=-35)
    return fig


def fig_bivariate(eda) -> go.Figure:
    pp = {r["rf_ppterm"]: r for r in eda["prior_preterm"]}
    mr = sorted(eda["medical_risk"], key=lambda r: r["medical_risk_count"])
    sm = eda["smoking"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["No prior preterm", "Prior preterm"],
                         y=[pp[0.0]["rate"], pp[1.0]["rate"]], marker_color=[C["sky"], C["red"]],
                         text=[f"{pp[0.0]['rate']:.1f}%", f"{pp[1.0]['rate']:.1f}%"],
                         textposition="outside", textfont=dict(color=INK), name="Prior preterm",
                         hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
    theme(fig, "A prior preterm birth more than triples the rate", "", "Preterm rate (%)", 360)
    fig.update_yaxes(range=[0, max(pp[1.0]["rate"], 30) * 1.15])
    return fig, mr, sm


# ------------------------------------------------------------------ weeks 5-9 chart helpers
def fig_model_benchmark(model_comparison: list) -> go.Figure:
    """AUC-PR comparison bar chart for the 5 models from week 5."""
    rows = sorted(model_comparison, key=lambda r: r["held_out_auc_pr"])
    labels = [r["model"] for r in rows]
    vals = [r["held_out_auc_pr"] for r in rows]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=[C["blue"] if r["model"] == "xgboost" else C["sky"] for r in rows],
        text=[f"{v:.4f}" for v in vals], textposition="outside",
        textfont=dict(color=INK, size=13),
        hovertemplate="%{y}: AUC-PR %{x:.4f}<extra></extra>"))
    theme(fig, "Model benchmark: held-out AUC-PR", "AUC-PR", "", 380)
    fig.update_xaxes(range=[0, max(vals) * 1.18])
    return fig


def fig_shap_top(shap_top_15: list) -> go.Figure:
    """SHAP top features bar chart."""
    rows = list(reversed(shap_top_15))  # ascending for horizontal bars
    labels = [r["feature"] for r in rows]
    vals = [r["importance"] for r in rows]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=C["purple"],
        text=[f"{v:.3f}" for v in vals], textposition="outside",
        textfont=dict(color=INK, size=11),
        hovertemplate="%{y}: |SHAP| %{x:.4f}<extra></extra>"))
    theme(fig, "Global SHAP feature importance (top 15)", "Mean |SHAP value|", "", 520)
    fig.update_xaxes(range=[0, max(vals) * 1.2])
    return fig


def fig_drift_line(drift: list) -> go.Figure:
    """Per-year AUC line chart from week 8."""
    years = [d["year"] for d in drift]
    auc_pr = [d["auc_pr"] for d in drift]
    auc_roc = [d["auc_roc"] for d in drift]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years, y=auc_pr, mode="lines+markers", name="AUC-PR",
        line=dict(color=C["blue"], width=3), marker=dict(size=8),
        hovertemplate="Year %{x}: AUC-PR %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=years, y=auc_roc, mode="lines+markers", name="AUC-ROC",
        line=dict(color=C["green"], width=3), marker=dict(size=7),
        hovertemplate="Year %{x}: AUC-ROC %{y:.4f}<extra></extra>"))
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor=C["yellow"], opacity=0.12, line_width=0)
    fig.add_annotation(x=2020.5, y=min(auc_pr) - 0.003, text="COVID era", showarrow=False,
                       font=dict(color=INK, size=11))
    theme(fig, "Temporal stability: per-year model performance (2018-2024)",
          "Birth year", "AUC")
    return fig


# ----------------------------------------------------------------------------------------------- app
def main() -> None:
    st.set_page_config(page_title="EquiBirth AI", page_icon="EB", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    b = load_bundle()
    eda = load_eda()
    mf = b["model_findings"]
    n_rows = eda.get("n_rows", b["meta"]["n_rows"])
    rate = eda.get("overall_rate", 8.92)

    # ---- hero
    st.markdown(
        f"""<div class="hero">
        <h1>Born Too Early, Unequally</h1>
        <div class="sub">How well can we predict preterm birth from a U.S. birth certificate, and how
        equitable are the outcomes? A data-science investigation of {n_rows:,} singleton births,
        2019 to 2024.</div>
        <div class="chips">
          <div class="chip"><div class="num">{n_rows/1e6:.2f}M</div><div class="lab">singleton births analysed</div></div>
          <div class="chip"><div class="num">{rate:.1f}%</div><div class="lab">overall preterm rate</div></div>
          <div class="chip"><div class="num">{mf['raw_black_white_gap_pp']:.1f} pp</div><div class="lab">Black-White gap</div></div>
          <div class="chip"><div class="num">{mf['auc_roc']:.2f}</div><div class="lab">honest model AUC</div></div>
        </div></div>""", unsafe_allow_html=True)

    # ---- sticky nav
    st.markdown(
        """<div class="navbar">
        <a href="#problem">The problem</a><a href="#data">The data</a><a href="#who">Who is here</a>
        <a href="#predict">What predicts it</a><a href="#disparity">The disparity</a>
        <a href="#covid">COVID</a><a href="#clusters">Risk clusters</a>
        <a href="#ceiling">The ceiling</a><a href="#fairness">Fairness</a>
        <a href="#why">Why the gap</a><a href="#meaning">What it means</a>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        """<div class="disclaimer"><b>Educational and research demonstration, not a clinical tool.</b>
        Nothing here should be used to screen, ration, prioritize, or deny care to any person. It
        illustrates measured population-level disparities and the limits of prediction.</div>""",
        unsafe_allow_html=True)

    # ===================================================================== 1. the problem
    section("problem", "The question", "A leading cause of infant death, distributed unequally",
            "Preterm birth, delivery before 37 weeks, is the leading driver of neonatal mortality and "
            "lifelong morbidity. In the United States it is also one of the most durable racial "
            "inequities in public health. This project asks two honest questions of the national data: "
            "how well can preterm birth be predicted before delivery, and how equitable are the outcomes?")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="mcard"><div class="mnum" style="color:{C['red']}">1.6x</div>
        <div class="mlab">NH Black vs NH White preterm rate</div>
        <div class="msub">12.9% versus 7.8%</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="mcard"><div class="mnum" style="color:{C['blue']}">{mf['auc_roc']:.3f}</div>
        <div class="mlab">Honest pre-delivery model (AUC-ROC)</div>
        <div class="msub">published admin-data range 0.62 to 0.70</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="mcard"><div class="mnum" style="color:{C['orange']}">{mf['leaky_auc']:.3f}</div>
        <div class="mlab">Leaky post-delivery model (the trap)</div>
        <div class="msub">looks great, predicts nothing</div></div>""", unsafe_allow_html=True)
    source("Sources: CDC/NCHS NVSS; March of Dimes; KFF. Race is a marker of exposure to racism, "
           "not a biological cause.")

    # ===================================================================== 2. the data
    section("data", "The data", "Nearly two million real birth records",
            "Everything here is computed from the CDC/NCHS Natality public-use files, a near-census of "
            "U.S. births served by the NBER mirror. We use a validated representative spread sample, a "
            "clean pre-delivery feature set, and a carefully partitioned held-out test set.")
    d1, d2, d3 = st.columns(3)
    d1.markdown(f"""<div class="mcard"><div class="mnum">{n_rows:,}</div>
        <div class="mlab">singleton births (rows)</div><div class="msub">2019 to 2024</div></div>""",
                unsafe_allow_html=True)
    d2.markdown(f"""<div class="mcard"><div class="mnum">{eda.get('n_cols',108)}</div>
        <div class="mlab">engineered features (columns)</div>
        <div class="msub">all pre-delivery, leakage-audited</div></div>""", unsafe_allow_html=True)
    d3.markdown(f"""<div class="mcard"><div class="mnum">{rate:.2f}%</div>
        <div class="mlab">preterm (the target)</div>
        <div class="msub">a rare, imbalanced outcome</div></div>""", unsafe_allow_html=True)
    callout("warn", "The leakage trap we refuse",
            "Variables recorded at or after delivery (birthweight, Apgar, labor interventions) push the "
            "AUC to 0.91 and are useless as predictors, because they are consequences of the outcome. A "
            "tested gate bars them. The honest model uses only what is known before labor.")
    callout("research", "Honest provenance: a sample, not the full census",
            "The full per-year files are about 2.2 GB each and the server throttles long transfers, so "
            "the data were gathered as a validated representative spread sample (ten chunks at evenly "
            "spaced byte offsets per year), about 8 to 10% of the roughly 21.2M singleton births (of "
            "21.9M total) in this window, not the full 18M-plus the project was first sketched around. "
            "This is adequate because the model ceiling is set by signal, not sample size, and we confirm "
            "representativeness against the numbers we compute: the per-race singleton rates reproduce the "
            "published NVSS ordering and magnitudes, and our overall singleton rate (8.9%) sits correctly "
            "below the 10.4% all-births headline because multiple gestations, about 60% preterm, are "
            "excluded. One consequence we respect everywhere: the sampling fraction is not constant across "
            "years, so every cross-year comparison uses rates, never raw counts.")
    callout("warn", "Three data-quality caveats a domain expert keeps in view",
            "Birth certificates are a criterion standard, but not flawless. (1) Self-reported race is "
            "highly accurate for White, Black, and Hispanic groups but less so for AIAN and NHOPI, whose "
            "estimates also rest on small samples, so we read those two with extra caution. (2) NCHS "
            "quality studies never validated the worksheet height and weight, so BMI is the "
            "least-audited field. (3) Risk-factor checkboxes are known to undercount, which biases their "
            "measured associations toward zero. Sources: CDC/NCHS Vital Statistics documentation.")

    # ===================================================================== 3. who is here (EDA tabs)
    if eda:
        section("who", "Exploration", "Who is in the data?",
                "Before relating features to the outcome, we look at the mothers themselves. Switch tabs "
                "to explore the three most important distributions.")
        t1, t2, t3 = st.tabs(["Maternal age", "Pre-pregnancy BMI", "Education"])
        with t1:
            chart(fig_dist_age(eda["age_hist"]))
            st.markdown("A roughly bell-shaped distribution centered near 29 years, with a long tail of "
                        "older mothers whose risk profile differs.")
        with t2:
            chart(fig_dist_bmi(eda["bmi_hist"]))
            st.markdown("A right-skewed distribution with a real obesity tail. The raw data also contain "
                        "an NCHS not-stated code of 99.9, a data-quality detail we exclude rather than "
                        "let it distort the axis.")
        with t3:
            chart(fig_dist_edu(eda["edu_dist"]))
            st.markdown("Education concentrates at high-school-graduate and bachelor levels. We return to "
                        "education as the key test of the socioeconomic explanation below.")

        # ================================================================= 4. what predicts it
        section("predict", "Exploration", "What actually predicts preterm birth?",
                "We compute the preterm rate within levels of each candidate predictor. The strongest "
                "legitimate signals are clinical, and even they are individually weak.")
        bivar, mr, sm = fig_bivariate(eda)
        col_a, col_b = st.columns(2)
        with col_a:
            chart(bivar)
        with col_b:
            mr_fig = go.Figure(go.Scatter(
                x=[r["medical_risk_count"] for r in mr], y=[r["rate"] for r in mr],
                mode="lines+markers+text", line=dict(color=C["blue"], width=3), marker=dict(size=9),
                text=[f"{r['rate']:.0f}%" for r in mr], textposition="top center",
                textfont=dict(color=INK, size=12),
                hovertemplate="%{x} risk factors: %{y:.1f}%<extra></extra>"))
            theme(mr_fig, "Medical-risk count: a clean dose-response", "Number of risk factors",
                  "Preterm rate (%)", 360)
            chart(mr_fig)
        callout("insight", "The signal is real but diffuse",
                f"A prior preterm birth more than triples the rate (about 27% versus 8%). The medical-risk "
                f"count climbs from roughly 8% to 66%. Smoking before pregnancy lifts the rate from "
                f"{sm['Non-smoker']:.1f}% to {sm['Smoker']:.1f}%. These are the legitimate predictors a "
                f"model should use, and SHAP confirms it does.")
        chart(fig_corr(eda["corr"]))
        callout("insight", "This one chart explains the whole modeling story",
                "The strongest correlate, prior preterm birth, reaches only about +0.13; everything else "
                "is weaker. There is no dominant signal hiding in these fields. A model can only combine "
                "many faint, overlapping hints, which is exactly why the honest ceiling lands near AUC "
                "0.66 rather than 0.9. The causes of preterm birth largely live outside what a birth "
                "certificate records.")

    # ===================================================================== 5. the disparity
    section("disparity", "The finding", "The racial disparity, and the test that breaks the easy story",
            "The central equity fact, computed straight from the data. The ordering reproduces published "
            "NVSS figures, which both validates the sample and motivates the entire fairness analysis.")
    chart(fig_rate_by_race(b["rate_by_race"]))
    source("Each bar is the full-population singleton rate. The NH AIAN and NH NHOPI bars rest on small "
           "samples and known race misclassification, so the Black, White, Hispanic, and Asian estimates "
           "are the most reliable. Race is a marker of exposure to racism and structural barriers, not a "
           "biological cause.")

    if eda:
        st.markdown("**Does education close the gap?** A natural hypothesis is that the racial gap is "
                    "really a socioeconomic gap. Choose which groups to overlay and judge for yourself.")
        chosen = st.multiselect("Groups to compare", RACE_ORDER,
                                default=["NH White", "NH Black", "Hispanic", "NH Asian"],
                                key="edu_races")
        if chosen:
            chart(fig_edu_race(eda["edu_race"], chosen))
    sc = b["intersectional_scalars"]
    if sc.get("black_college_rate") and sc.get("white_hs_or_less_rate"):
        callout("warn", "The starkest single comparison",
                f"College-educated NH Black mothers have a preterm rate of "
                f"{sc['black_college_rate']*100:.1f}% (n={sc['black_college_n']:,}), <b>higher</b> than "
                f"NH White mothers with a high-school education or less "
                f"({sc['white_hs_or_less_rate']*100:.1f}%, n={sc['white_hs_or_less_n']:,}). A Black mother "
                f"with a doctorate has a higher preterm rate than a White mother who never finished high "
                f"school. Education, the great equalizer, does not equalize this.")

    # ===================================================================== 6. covid
    section("covid", "A curious detour", "Where did COVID go?",
            "With a 2019 to 2024 dataset we asked the obvious question. Published work found a real "
            "signal: a 5 to 6% national decline in spring 2020, then a rebound to record highs in 2021. "
            "Click a group in the legend to highlight it.")
    hi = st.radio("Highlight", ["NH Black and NH White", "NH Asian (clearest dip)", "All groups"],
                  horizontal=True, key="covid_hi")
    highlight = {"NH Black and NH White": ["NH Black", "NH White"],
                 "NH Asian (clearest dip)": ["NH Asian"],
                 "All groups": RACE_ORDER}[hi]
    chart(fig_temporal(b["temporal_by_race"], highlight))
    callout("insight", "A real effect at one time scale can vanish at another",
            "Our annual curve shows the published direction, a small 2020 dip and a 2021 rebound, but the "
            "magnitude is a few hundredths of a point where the literature reported several percent. The "
            "published decline was a monthly March-to-June 2020 phenomenon; averaging it into an annual "
            "national rate dilutes a sharp three-month dip into near-invisibility. Aggregate flatness is "
            "not evidence of no effect, a genuine lesson in temporal granularity and statistical power.")
    source("Sources: AAP Pediatrics 2021; CDC NCHS Data Brief 477.")

    # ===================================================================== 7. clusters
    section("clusters", "Unsupervised view", "Latent risk is clinical, not racial",
            "Setting the outcome aside, we let KMeans group mothers by their feature profiles alone. If "
            "risk were racially clustered, the groups would sort by race. They do not.")
    cl1, cl2 = st.columns([1, 1])
    with cl1:
        chart(fig_cluster_rates(b["clusters"]))
    with cl2:
        chart(fig_cluster_heatmap(b["clusters"]))
    callout("insight", "Risk is a continuum, organized by clinical profile",
            "Clusters are defined by prior preterm birth, BMI, medical-risk count, and parity, not by "
            "race, and every high-risk cluster is NH-White-plurality. Weak separation (silhouette about "
            "0.25) means risk is a smooth continuum rather than crisp subpopulations. The measured "
            "feature space sorts people by clinical risk, and that sorting does not reconstruct the "
            "racial gap, which stays diffuse.")

    # ===================================================================== 8. the ceiling
    section("ceiling", "Modeling", "Why prediction is modest: the ceiling",
            "A layered logistic ablation shows where the little available signal lives. Each bar adds a "
            "feature group; the dashed red line is the leaky model we refuse to use.")
    if b.get("ceiling_probe", {}).get("ablation"):
        chart(fig_ceiling(b["ceiling_probe"]["ablation"], mf["leaky_auc"]))
    callout("good", "The honest result",
            f"Race alone is barely better than a coin flip; even all clean features together reach about "
            f"{mf['auc_roc']:.3f}, squarely in the published 0.62 to 0.70 administrative-data range. After "
            f"isotonic calibration the scores are well-calibrated in aggregate (ECE about {mf['ece']:.3f}), "
            f"but roughly {mf['fraction_near_base_rate']*100:.0f}% of predictions sit within 5 points of "
            f"the base rate. This supports population-level diagnosis, not reliable individual risk scoring.")

    # ===================================================================== 9. fairness
    section("fairness", "Responsible AI", "The fairness tradeoff",
            "We audited the model with tested fairness metrics, each with a bootstrap interval and a "
            "permutation p-value. The result is the opposite of the naive worry.")
    fc1, fc2 = st.columns([1, 1])
    with fc1:
        chart(fig_fairness(mf))
    with fc2:
        st.markdown(
            f"At a single screening threshold the model <b>detects more</b> real preterm cases in "
            f"higher-risk groups (equal opportunity satisfied), but <b>flags NH Black mothers about "
            f"{mf['disparate_impact_black']:.1f}x as often</b> as White mothers. That ratio reflects "
            f"genuinely higher prevalence, not a model bug. Equalizing it would require race-specific "
            f"thresholds, which is disparate treatment, in legal and ethical tension with "
            f"disparate-impact goals. No single mitigation is presented as correct.",
            unsafe_allow_html=True)
    callout("warn", "There is no free lunch",
            "Post-processing with race-specific thresholds cuts NH Black disparate impact from about "
            "2.69x to 1.11x (the mitigation experiment uses its own per-group thresholds, so its baseline "
            "differs slightly from the 2.9x single-threshold figure above). But it uses a different "
            "decision rule by race. You can equalize error rates or equalize treatment, not both. This "
            "dashboard deliberately exposes no race-specific thresholds and no individual risk calculator.")

    # ===================================================================== 10. why the gap
    section("why", "Explanation", "Why does the gap persist?",
            "A g-computation decomposition asks how much of the gap measured socioeconomic factors can "
            "explain. The answer points beyond the birth certificate.")
    wc1, wc2 = st.columns([1, 1])
    with wc1:
        chart(fig_decomposition(mf))
    with wc2:
        st.markdown(
            "Epidemiology has a name for the pattern. Arline Geronimus's <b>weathering hypothesis</b> "
            "proposes that the chronic physiological stress of lifelong exposure to racism produces "
            "cumulative <b>allostatic load</b> that erodes reproductive health, and that, unlike material "
            "hardship, this exposure is not bought off by a degree or a paycheck. Published work finds "
            "the gap persists among college-educated, privately-insured Black women.", unsafe_allow_html=True)
    callout("research", "The unexplained share is a lower bound",
            "Education, payment, and WIC are themselves products of structural racism, so adjusting for "
            "them over-subtracts: the explained share is an overstatement and the roughly two-thirds "
            "unexplained residual is a lower bound on the structural contribution. Everything here is "
            "association, not proof of mechanism. Sources: "
            "<a href='https://www.kff.org/racial-equity-and-health-policy/racial-disparities-in-maternal-and-infant-health-current-status-and-key-issues/'>KFF</a>, "
            "<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC7654959/'>high-SES disparity study</a>, "
            "<a href='https://en.wikipedia.org/wiki/Weathering_hypothesis'>weathering hypothesis</a>.")

    # ===================================================================== 11. what it means
    section("meaning", "Conclusion", "What this means, and what it cannot",
            "The honest thesis, the policy reading, and the limits.")
    st.markdown(
        "- **The thesis.** Individual prediction from birth-certificate data is modest (AUC about 0.66); "
        "the racial disparity is large and only partly explained by measured factors.\n"
        "- **Policy.** Targeting only measured socioeconomic and access factors would close at most a "
        "third to a half of the gap; the majority points to structural drivers not on the birth "
        "certificate. The model is a population-level diagnostic, not an individual risk scorer.\n"
        "- **Equity is a value judgment** the data can frame but not resolve: the equal-opportunity "
        "versus disparate-impact tension requires a human, democratic decision about which error to bear.")
    callout("warn", "Limitations, stated plainly",
            "Calibration is aggregate-only, not individual. Public-use files carry no geography "
            "(post-2005). The data are representative spread samples validated against NVSS, not the full "
            "census. Everything is associational; no causal claim is made. This is not a clinical tool.")

    # ===================================================================== 12. weeks 5-9 deep dive
    w59 = load_weeks5to9()
    if w59:
        section("deep", "Deep dive", "Weeks 5-9: advanced modeling, fairness, and robustness",
                "The extended pipeline explored deep learning benchmarks, fairness mitigation, "
                "SHAP explainability, disparity decomposition, and temporal robustness testing.")
        callout("warn", "Not a clinical tool", DISCLAIMER_TEXT)
        st.markdown("---")

        # Model benchmark
        st.markdown("**Model benchmark (Week 5):** five models compared on held-out AUC-PR. "
                    "The tuned XGBoost was selected as the production model.")
        chart(fig_model_benchmark(w59["model_comparison"]))

        # SHAP features
        st.markdown("**Feature importance (Week 7):** SHAP values confirm clinical features dominate. "
                    "Prior preterm birth, medical-risk count, and maternal age are the top three globally.")
        chart(fig_shap_top(w59["shap_top_15"]))

        # Drift
        st.markdown("**Temporal stability (Week 8):** per-year AUC shows no meaningful degradation "
                    "across seven years, including through the COVID discontinuity.")
        chart(fig_drift_line(w59["drift"]))

        # Summary card
        st.markdown(f'<div class="mcard"><div class="mlab">{w59["model_card_oneliner"]}</div></div>',
                    unsafe_allow_html=True)
        source("All results from the EquiBirth Weeks 5-9 analysis pipeline. "
               "Guardrails enforced: no race-specific thresholds, no individual risk calculator, "
               "race never a modifiable input.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """<p class="foot">Built by <b>Bhavika Kothapalli</b>, <b>Parshv Patel</b>
        (<a href="https://www.linkedin.com/in/parshv-patel-65a90326b/">LinkedIn</a>),
        <b>Nadia Woo</b>, <b>Pramish Pandey</b>, <b>Nathalie Rodriguez</b>, <b>Andrew Pierre</b>
        and <b>Bhavya Agarwal</b><br>
        EquiBirth AI, an educational and research demonstration. Not a clinical tool. Race is a social
        construct and a marker of exposure to racism, not a biological cause of preterm birth.</p>""",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
