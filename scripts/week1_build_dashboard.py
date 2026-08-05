"""Build the standalone Week 1 HTML dashboard.

Produces reports/week1_dashboard.html: one self-contained, scrollable page summarizing this week's
work (data acquisition, exploration, EDA, visualization). It embeds the precomputed aggregate bundle
(reports/week1_dashboard_data.json) directly in the file and renders every chart client-side with
Plotly from a CDN, so the result is a single file you can open in any browser with no server, no
Python, and no install. No raw rows are embedded, only small aggregates plus eight illustrative
sample rows.

Run: ./.venv/bin/python scripts/week1_build_dashboard.py   (after scripts/week1_build_data.py)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "week1_dashboard_data.json"
OUT = ROOT / "reports" / "week1_dashboard.html"


def build_html(bundle: dict) -> str:
    data_json = json.dumps(bundle)
    # The page template. Data is injected as a JS const; all rendering happens client-side.
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EquiBirth AI &middot; Week 1: Data, Exploration &amp; EDA</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#1a1a2e; --muted:#5b5b6b; --line:#e7e7ee; --bg:#fbfbfd; --card:#ffffff;
    --blue:#0072B2; --orange:#E69F00; --green:#009E73; --red:#D55E00;
    --purple:#CC79A7; --sky:#56B4E9; --grey:#999999;
    --maxw:1080px;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0; background:var(--bg); color:var(--ink);
    font-family:Inter,system-ui,-apple-system,sans-serif; line-height:1.6; -webkit-font-smoothing:antialiased;}
  a{color:var(--blue); text-decoration:none}
  a:hover{text-decoration:underline}

  /* ---- hero ---- */
  header.hero{padding:84px 24px 60px; text-align:center;
    background:radial-gradient(1200px 500px at 50% -10%, #eef4fb 0%, var(--bg) 60%);
    border-bottom:1px solid var(--line);}
  .kicker{font-size:13px; letter-spacing:.18em; text-transform:uppercase; color:var(--blue); font-weight:700;}
  header.hero h1{font-family:Newsreader,Georgia,serif; font-weight:500; font-size:clamp(34px,5.2vw,58px);
    line-height:1.08; margin:14px auto 10px; max-width:18ch;}
  header.hero p.sub{max-width:62ch; margin:0 auto; color:var(--muted); font-size:18px;}
  .byline{margin-top:22px; font-size:14px; color:var(--muted);}
  .scope-pill{display:inline-block; margin-top:18px; padding:7px 16px; border:1px solid var(--line);
    border-radius:999px; background:var(--card); font-size:13px; color:var(--muted); font-weight:500;}

  /* ---- layout ---- */
  main{max-width:var(--maxw); margin:0 auto; padding:0 24px;}
  section{padding:54px 0; border-bottom:1px solid var(--line);}
  section:last-child{border-bottom:none}
  .eyebrow{font-size:12px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); font-weight:700;}
  h2{font-family:Newsreader,Georgia,serif; font-weight:500; font-size:clamp(26px,3.4vw,38px);
    margin:8px 0 6px; line-height:1.15;}
  h3{font-size:19px; font-weight:700; margin:30px 0 8px;}
  p.lead{font-size:18px; color:#37374a; max-width:70ch;}
  p, li{font-size:16.5px; color:#37374a; max-width:74ch;}
  .note{font-size:14.5px; color:var(--muted); max-width:74ch;}
  .analysis{background:#f4f7fb; border-left:3px solid var(--blue); padding:14px 18px; border-radius:0 8px 8px 0;
    margin:16px 0; font-size:15.5px; color:#33374a; max-width:74ch;}
  .analysis b{color:var(--ink)}

  /* ---- stat cards ---- */
  .stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:26px 0 8px;}
  .stat{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 16px;
    box-shadow:0 1px 2px rgba(20,20,50,.04);}
  .stat .num{font-size:30px; font-weight:800; letter-spacing:-.02em; line-height:1;}
  .stat .lab{font-size:13px; color:var(--muted); margin-top:8px;}
  .stat.accent .num{color:var(--blue)}
  .stat.warn .num{color:var(--red)}

  /* ---- charts ---- */
  .chart{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:10px 8px 4px;
    margin:18px 0; box-shadow:0 1px 2px rgba(20,20,50,.04);}
  .grid2{display:grid; grid-template-columns:1fr 1fr; gap:18px;}
  @media(max-width:760px){ .grid2{grid-template-columns:1fr} }

  /* ---- table ---- */
  .tablewrap{overflow-x:auto; border:1px solid var(--line); border-radius:14px; background:var(--card);
    box-shadow:0 1px 2px rgba(20,20,50,.04); margin:18px 0;}
  table{border-collapse:collapse; width:100%; font-size:14px;}
  th,td{padding:10px 14px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap;}
  th{background:#f7f8fb; font-weight:600; color:var(--ink); font-size:12.5px; letter-spacing:.02em; text-transform:uppercase;}
  tbody tr:last-child td{border-bottom:none}
  td.pt-yes{color:var(--red); font-weight:700}
  td.pt-no{color:var(--green); font-weight:600}

  /* ---- TOC nav ---- */
  nav.toc{position:sticky; top:0; z-index:10; background:rgba(251,251,253,.92); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--line); padding:11px 24px; font-size:13.5px;}
  nav.toc .inner{max-width:var(--maxw); margin:0 auto; display:flex; gap:18px; flex-wrap:wrap; align-items:center;}
  nav.toc b{color:var(--ink); font-weight:700; margin-right:6px}
  nav.toc a{color:var(--muted); font-weight:500}

  /* ---- plan / next steps ---- */
  .plan{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:16px; margin-top:22px;}
  .plan .step{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px;}
  .plan .step .n{display:inline-flex; width:26px; height:26px; align-items:center; justify-content:center;
    border-radius:50%; background:#eef4fb; color:var(--blue); font-weight:800; font-size:14px;}
  .plan .step h4{margin:12px 0 6px; font-size:16px;}
  .plan .step p{font-size:14px; color:var(--muted); margin:0;}
  .plan .step .tag{display:inline-block; margin-top:10px; font-size:11px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--orange); font-weight:700;}

  footer{padding:46px 24px 70px; text-align:center; color:var(--muted); font-size:14px;}
  .disclaimer{max-width:64ch; margin:0 auto 14px; font-size:13.5px;}
</style>
</head>
<body>

<header class="hero">
  <div class="kicker">EquiBirth AI &middot; Week 1 Deliverable</div>
  <h1>Acquiring and exploring every U.S. singleton birth, 2018&ndash;2024</h1>
  <p class="sub">Where the data comes from, whether we gathered the right thing, and what an honest first
  exploration reveals about preterm birth and racial equity. Every number on this page is computed from the
  real data.</p>
  <div class="scope-pill">Scope this week: data acquisition &middot; exploration &middot; EDA &middot; visualization</div>
  <div class="byline">Parshv Patel &middot; <a href="https://www.linkedin.com/in/parshv-patel-65a90326b/">LinkedIn</a></div>
</header>

<nav class="toc"><div class="inner">
  <b>On this page</b>
  <a href="#snapshot">Snapshot</a>
  <a href="#sample">Sample rows</a>
  <a href="#quality">Data quality</a>
  <a href="#who">Who is in the data</a>
  <a href="#signals">What predicts preterm</a>
  <a href="#equity">The equity finding</a>
  <a href="#corr">Effect size</a>
  <a href="#next">What comes next</a>
</div></nav>

<main>

  <section id="snapshot">
    <div class="eyebrow">The dataset at a glance</div>
    <h2>What we are working with</h2>
    <p class="lead">The complete CDC/NCHS Natality public-use files for 2018 to 2024, restricted to singleton
    births, with only the pre-delivery certificate fields we explore this week.</p>
    <div class="stats" id="stat-cards"></div>
    <p class="note">Source: CDC/NCHS Natality public-use files via the NBER mirror. Singletons only (multiples
    are excluded by design because they are roughly 60% preterm). The overall singleton rate sits below the
    familiar 10.4% all-births headline exactly as expected.</p>
  </section>

  <section id="sample">
    <div class="eyebrow">One record is one birth</div>
    <h2>A handful of real rows</h2>
    <p>We always begin the way a real analysis session does: load the table and look at actual rows. These
    eight are a random, reproducible sample of the raw fields (coded values mapped to readable labels).</p>
    <div class="tablewrap"><table id="sample-table"></table></div>
    <p class="note">Only these eight illustrative rows are shipped with the dashboard; everything else on this
    page is an aggregate, never raw microdata.</p>
  </section>

  <section id="quality">
    <div class="eyebrow">Before trusting anything</div>
    <h2>Data quality: missingness and sentinels</h2>
    <p>Missingness in administrative data is never random, so we map it first. Then we hunt the NCHS "not
    stated" sentinel codes that masquerade as real numbers (a BMI of 99.9, 98 cigarettes a day).</p>
    <div class="grid2">
      <div class="chart" id="chart-miss"></div>
      <div class="chart" id="chart-sentinel"></div>
    </div>
    <div class="analysis"><b>The discipline.</b> Marital status is missing for about 10% of rows, correlated
    with reporting jurisdiction (missing-not-at-random); everything else is under 2%. Sentinel values get
    excluded from quantitative summaries, but genuinely rare-but-real values (a 13-year-old mother, a mother
    with 12 prior births) are kept, because they exist and are clinically high-risk.</div>
  </section>

  <section id="who">
    <div class="eyebrow">Univariate exploration</div>
    <h2>Who is in the data?</h2>
    <p>Before relating anything to the outcome, we look at the cast of characters: the distributions of age,
    pre-pregnancy BMI, education, and payment source.</p>
    <div class="grid2">
      <div class="chart" id="chart-age"></div>
      <div class="chart" id="chart-bmi"></div>
    </div>
    <div class="grid2">
      <div class="chart" id="chart-edu"></div>
      <div class="chart" id="chart-pay"></div>
    </div>
    <div class="analysis"><b>All as expected.</b> Age is bell-shaped near 29, BMI is right-skewed with a real
    obesity tail (sentinels removed), education peaks at high-school-graduate and bachelor levels, and payment
    splits between private insurance and Medicaid. Nothing preterm-specific yet, which is exactly what a clean
    first look should show.</div>
  </section>

  <section id="signals">
    <div class="eyebrow">Bivariate exploration</div>
    <h2>What actually predicts preterm birth?</h2>
    <p>For each candidate predictor we compute the preterm rate within each level and let the gradient speak.
    The strongest legitimate signals are clinical.</p>
    <div class="grid2">
      <div class="chart" id="chart-prior"></div>
      <div class="chart" id="chart-smoke"></div>
    </div>
    <div class="chart" id="chart-dose"></div>
    <div class="analysis"><b>Clinical history dominates.</b> A prior preterm birth more than triples the rate
    (about a 3.4x risk ratio), summed pre-pregnancy risk flags show a clean dose-response climbing toward
    about 61% at four flags, and smoking adds roughly four points. Each is a group-level gradient, not a sharp
    individual separator, which foreshadows a modest modeling ceiling.</div>
  </section>

  <section id="equity">
    <div class="eyebrow">The finding we did not go looking for</div>
    <h2>The racial disparity, and whether education closes it</h2>
    <p>Preterm rate by race shows a roughly 4.7-point gap between NH Black and NH White mothers. A
    two-proportion z-test gives z about 280 (p indistinguishable from zero, and that huge z is mostly the
    millions of rows per group shrinking the standard error; the substantive number is the 4.7-point gap).
    The natural hypothesis is that it is really socioeconomic, so we try to falsify that by stratifying on
    education.</p>
    <div class="chart" id="chart-race"></div>
    <div class="chart" id="chart-edurace"></div>
    <div class="analysis"><b>The hypothesis is falsified, and that is the finding.</b> The race lines do not
    converge as education rises. A college-educated NH Black mother has a higher preterm rate than a NH White
    mother who never finished high school. Education, the supposed great equalizer, does not equalize this.
    Throughout, <b>race is treated as a marker of exposure to racism and structural barriers, not a biological
    cause.</b></div>
  </section>

  <section id="corr">
    <div class="eyebrow">The statistical habit of the week</div>
    <h2>Significance is not the same as importance</h2>
    <p>How strongly does each numeric field correlate with the target? Every one is weak: the strongest, prior
    preterm birth, reaches only about +0.13.</p>
    <div class="chart" id="chart-corr"></div>
    <div class="analysis"><b>No dominant signal hides here.</b> Across tens of millions of rows, a t-test will
    call a one-unit BMI difference "wildly significant," yet no clinician would act on it. Effect size, not the
    p-value, is what matters, and the effect sizes are all small. This strongly suggests the honest modeling
    ceiling will be modest, because the causes of preterm birth largely live outside what a birth certificate
    records.</div>
  </section>

  <section id="next">
    <div class="eyebrow">Planned next steps</div>
    <h2>What this exploration sets up</h2>
    <p>This week is acquisition, exploration, quality auditing, EDA, and visualization. The EDA is exactly what
    tells us how to do the following work well. None of it is done yet; it is the plan for the coming weeks.</p>
    <div class="plan">
      <div class="step"><span class="n">1</span><h4>Leakage-safe feature engineering</h4>
        <p>A tested gate that bars at-delivery fields (birthweight, Apgar, NICU), which are near-perfect but
        useless predictors. Given how weak the legitimate signals are, this is the most important decision ahead.</p>
        <span class="tag">Planned</span></div>
      <div class="step"><span class="n">2</span><h4>Calibrated baseline models</h4>
        <p>Logistic regression and tree models judged on rate-aware metrics and calibration rather than
        accuracy, given the class imbalance. A modest ceiling is expected and will be reported honestly.</p>
        <span class="tag">Planned</span></div>
      <div class="step"><span class="n">3</span><h4>A formal fairness audit</h4>
        <p>Turn this week's disparity finding into rigorous metrics: demographic parity, equal opportunity,
        calibration by group, each with confidence intervals and significance tests.</p>
        <span class="tag">Planned</span></div>
      <div class="step"><span class="n">4</span><h4>Explainability &amp; gap decomposition</h4>
        <p>Verify what any model leans on, then ask how much of the racial gap measured factors actually
        explain, with appropriate causal humility.</p>
        <span class="tag">Planned</span></div>
    </div>
  </section>

</main>

<footer>
  <p class="disclaimer">This is an educational and research project. Nothing here is a clinical decision tool.
  All analysis is descriptive and associational; no causal claim is made. Race is used as a social marker of
  exposure to racism, never as a biological cause.</p>
  <p>EquiBirth AI &middot; Week 1 &middot; Parshv Patel &middot;
  <a href="https://www.linkedin.com/in/parshv-patel-65a90326b/">LinkedIn</a></p>
</footer>

<script>
const DATA = __DATA_JSON__;

// ---- Okabe-Ito colorblind-safe palette, matching the notebooks ----
const C = {blue:"#0072B2", orange:"#E69F00", green:"#009E73", red:"#D55E00",
           purple:"#CC79A7", sky:"#56B4E9", yellow:"#F0E442", grey:"#999999", ink:"#1a1a2e"};
const RACE_COLOR = {"NH White":C.blue, "NH Black":C.red, "Hispanic":C.green, "NH Asian":C.orange,
                    "NH Multiracial":C.grey, "NH NHOPI":C.grey, "NH AIAN":C.grey};
const FONT = {family:"Inter, system-ui, sans-serif", size:13, color:C.ink};
const LAYOUT = (title, x, y, extra={}) => Object.assign({
  title:{text:title, x:0.01, font:{size:15, color:C.ink}},
  xaxis:Object.assign({title:x, showgrid:false, linecolor:"#ccc", automargin:true}, extra.xaxis||{}),
  yaxis:Object.assign({title:y, gridcolor:"#ECECEC", zeroline:false, automargin:true}, extra.yaxis||{}),
  font:FONT, plot_bgcolor:"white", paper_bgcolor:"white",
  margin:{l:60,r:22,t:48,b:50}, height:extra.height||360,
  showlegend:extra.showlegend||false,
  legend:{orientation:"h", y:1.06, x:0, font:FONT}
}, extra.layout||{});
const CFG = {responsive:true, displayModeBar:false};
const fmtN = n => n.toLocaleString("en-US");
const EDU = DATA.meta.education_labels, PAY = DATA.meta.payment_labels;

// ---- stat cards ----
(function(){
  const m = DATA.meta;
  const cards = [
    {n:fmtN(m.n_rows), l:"singleton birth records", cls:"accent"},
    {n:m.n_fields, l:"pre-delivery fields explored"},
    {n:m.years[0]+"&ndash;"+m.years[m.years.length-1], l:"years pooled (same certificate)"},
    {n:m.overall_rate.toFixed(1)+"%", l:"overall preterm rate", cls:"accent"},
    {n:fmtN(m.n_preterm), l:"preterm births observed"},
    {n:m.n_race_groups, l:"race / ethnicity groups"},
  ];
  document.getElementById("stat-cards").innerHTML = cards.map(c =>
    `<div class="stat ${c.cls||""}"><div class="num">${c.n}</div><div class="lab">${c.l}</div></div>`).join("");
})();

// ---- sample table ----
(function(){
  const rows = DATA.sample_rows;
  const cols = [["birth_year","Year"],["mager","Age"],["education","Education"],["bmi","BMI"],
                ["cig_0","Cig/day"],["priorlive","Prior births"],["payment","Payment"],
                ["mother_race_ethnicity","Race/ethnicity"],["preterm","Preterm?"]];
  let h = "<thead><tr>"+cols.map(c=>`<th>${c[1]}</th>`).join("")+"</tr></thead><tbody>";
  rows.forEach(r=>{
    h += "<tr>"+cols.map(c=>{
      let v = r[c[0]];
      if(c[0]==="preterm"){ const yes = v===1; return `<td class="${yes?'pt-yes':'pt-no'}">${yes?'Preterm':'Term'}</td>`; }
      if(v===null||v===undefined) v="&mdash;";
      return `<td>${v}</td>`;
    }).join("")+"</tr>";
  });
  document.getElementById("sample-table").innerHTML = h+"</tbody>";
})();

// ---- missingness ----
(function(){
  const entries = Object.entries(DATA.missingness).sort((a,b)=>a[1]-b[1]);
  Plotly.newPlot("chart-miss", [{
    type:"bar", orientation:"h", y:entries.map(e=>e[0]), x:entries.map(e=>e[1]),
    marker:{color:C.orange}, text:entries.map(e=>e[1].toFixed(1)+"%"), textposition:"outside",
    hovertemplate:"%{y}: %{x:.2f}% missing<extra></extra>"
  }], LAYOUT("Missingness concentrates in a few fields", "Missing (%)", "",
            {height:360, layout:{xaxis:{range:[0, Math.max(...entries.map(e=>e[1]))*1.18]}}}), CFG);
})();

// ---- sentinels ----
(function(){
  const s = DATA.sentinels;
  const items = [
    ["BMI &ge; 70", s.bmi_ge_70.n], ["Cig/day &ge; 90", s.cig_ge_90.n],
    ["Age &le; 15", s.age_le_15], ["Age &ge; 45", s.age_ge_45], ["Prior births &ge; 10", s.priorlive_ge_10]
  ];
  const colors = [C.red, C.red, C.green, C.green, C.green]; // red = sentinel/drop, green = rare-but-real/keep
  Plotly.newPlot("chart-sentinel", [{
    type:"bar", orientation:"h", y:items.map(i=>i[0]), x:items.map(i=>i[1]),
    marker:{color:colors}, text:items.map(i=>fmtN(i[1])), textposition:"outside",
    hovertemplate:"%{y}: %{x:,} rows<extra></extra>"
  }], LAYOUT("Sentinels (red, dropped) vs rare-but-real values (green, kept)", "Rows", "",
            {height:360, layout:{xaxis:{type:"log", title:"Rows (log scale)"}}}), CFG);
})();

// ---- age histogram ----
(function(){
  const h = DATA.age_hist, centers = h.edges.slice(0,-1);
  Plotly.newPlot("chart-age", [{
    type:"bar", x:centers, y:h.counts, marker:{color:C.blue},
    hovertemplate:"Age %{x}: %{y:,} births<extra></extra>"
  }], LAYOUT("Maternal age is centered near 29", "Age (years)", "Births", {height:330}), CFG);
})();

// ---- bmi histogram ----
(function(){
  const h = DATA.bmi_hist, centers = h.edges.slice(0,-1).map((e,i)=>(e+h.edges[i+1])/2);
  Plotly.newPlot("chart-bmi", [{
    type:"bar", x:centers, y:h.counts, marker:{color:C.green},
    hovertemplate:"BMI %{x:.1f}: %{y:,} births<extra></extra>"
  }], LAYOUT("Pre-pregnancy BMI is right-skewed (sentinels removed)", "BMI", "Births", {height:330}), CFG);
})();

// ---- education distribution ----
(function(){
  const d = DATA.edu_dist, keys = Object.keys(d).sort((a,b)=>a-b);
  Plotly.newPlot("chart-edu", [{
    type:"bar", x:keys.map(k=>EDU[k]), y:keys.map(k=>d[k]), marker:{color:C.purple},
    hovertemplate:"%{x}: %{y:,}<extra></extra>"
  }], LAYOUT("Education peaks at HS-grad and bachelor", "", "Births",
            {height:340, xaxis:{tickangle:-35}}), CFG);
})();

// ---- payment distribution ----
(function(){
  const d = DATA.pay_dist, keys = Object.keys(d).sort((a,b)=>d[b]-d[a]);
  Plotly.newPlot("chart-pay", [{
    type:"bar", x:keys.map(k=>PAY[k]||k), y:keys.map(k=>d[k]), marker:{color:C.sky},
    hovertemplate:"%{x}: %{y:,}<extra></extra>"
  }], LAYOUT("Payment splits between private and Medicaid", "", "Births",
            {height:340, xaxis:{tickangle:-35}}), CFG);
})();

// ---- prior preterm ----
(function(){
  const p = DATA.prior_preterm.sort((a,b)=>a.rf_ppterm-b.rf_ppterm);
  Plotly.newPlot("chart-prior", [{
    type:"bar", x:["No prior preterm","Prior preterm"], y:p.map(d=>d.rate),
    marker:{color:[C.grey, C.red]}, text:p.map(d=>d.rate.toFixed(1)+"%"), textposition:"outside",
    hovertemplate:"%{x}: %{y:.1f}%<extra></extra>"
  }], LAYOUT("A prior preterm birth more than triples the rate", "", "Preterm rate (%)",
            {height:330, layout:{yaxis:{range:[0, Math.max(...p.map(d=>d.rate))*1.2]}}}), CFG);
})();

// ---- smoking ----
(function(){
  const s = DATA.smoking;
  Plotly.newPlot("chart-smoke", [{
    type:"bar", x:["Non-smoker","Smoker"], y:[s["Non-smoker"], s["Smoker"]],
    marker:{color:[C.grey, C.orange]}, text:[s["Non-smoker"].toFixed(1)+"%", s["Smoker"].toFixed(1)+"%"],
    textposition:"outside", hovertemplate:"%{x}: %{y:.1f}%<extra></extra>"
  }], LAYOUT("Smoking lifts the preterm rate by about four points", "", "Preterm rate (%)",
            {height:330, layout:{yaxis:{range:[0, Math.max(s["Non-smoker"],s["Smoker"])*1.25]}}}), CFG);
})();

// ---- dose response ----
(function(){
  const d = DATA.risk_dose_response.sort((a,b)=>a.risk_n-b.risk_n);
  Plotly.newPlot("chart-dose", [{
    type:"scatter", mode:"lines+markers+text", x:d.map(r=>r.risk_n), y:d.map(r=>r.preterm),
    line:{color:C.blue, width:3}, marker:{size:9, color:C.blue},
    text:d.map(r=>r.preterm.toFixed(0)+"%"), textposition:"top center",
    hovertemplate:"%{x} risk factors: %{y:.1f}%<extra></extra>"
  }], LAYOUT("Summed risk flags show a clean dose-response (worth engineering later)",
            "Number of pre-pregnancy risk factors", "Preterm rate (%)",
            {height:360, xaxis:{dtick:1}, layout:{yaxis:{range:[0, 75]}}}), CFG);
})();

// ---- rate by race ----
(function(){
  const order = DATA.meta.race_order;
  const r = DATA.rate_by_race.slice().sort((a,b)=>a.rate-b.rate);
  Plotly.newPlot("chart-race", [{
    type:"bar", orientation:"h", y:r.map(d=>d.mother_race_ethnicity), x:r.map(d=>d.rate),
    marker:{color:r.map(d=>RACE_COLOR[d.mother_race_ethnicity]||C.grey)},
    text:r.map(d=>d.rate.toFixed(1)+"%"), textposition:"outside",
    customdata:r.map(d=>d.n),
    hovertemplate:"%{y}: %{x:.2f}%  (n=%{customdata:,})<extra></extra>"
  }], LAYOUT("NH Black mothers have the highest singleton preterm rate", "Preterm rate (%)", "",
            {height:360, layout:{xaxis:{range:[0, Math.max(...r.map(d=>d.rate))*1.15]}}}), CFG);
})();

// ---- education x race ----
(function(){
  const races = ["NH White","NH Black","Hispanic","NH Asian"];
  const traces = races.map(race=>{
    const pts = DATA.edu_race.filter(d=>d.mother_race_ethnicity===race).sort((a,b)=>a.meduc-b.meduc);
    return {type:"scatter", mode:"lines+markers", name:race,
      x:pts.map(d=>EDU[d.meduc]), y:pts.map(d=>d.rate),
      line:{color:RACE_COLOR[race], width:2.4}, marker:{size:7},
      hovertemplate:race+", %{x}: %{y:.1f}%<extra></extra>"};
  });
  Plotly.newPlot("chart-edurace", traces,
    LAYOUT("The racial gap does not close as education rises", "Maternal education", "Preterm rate (%)",
           {height:420, showlegend:true, xaxis:{tickangle:-30}}), CFG);
})();

// ---- correlations ----
(function(){
  const e = Object.entries(DATA.corr).sort((a,b)=>a[1]-b[1]);
  Plotly.newPlot("chart-corr", [{
    type:"bar", orientation:"h", y:e.map(d=>d[0]), x:e.map(d=>d[1]),
    marker:{color:e.map(d=>d[1]<0?C.blue:C.red)},
    text:e.map(d=>(d[1]>=0?"+":"")+d[1].toFixed(3)), textposition:"outside",
    hovertemplate:"%{y}: %{x:.3f}<extra></extra>"
  }], LAYOUT("Every feature correlates only weakly with preterm birth",
            "Pearson correlation with the target", "", {height:460}), CFG);
})();
</script>
</body>
</html>
""".replace("__DATA_JSON__", data_json)


def main() -> None:
    bundle = json.loads(DATA.read_text())
    OUT.write_text(build_html(bundle), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
