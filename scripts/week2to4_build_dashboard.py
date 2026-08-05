"""Build the standalone Weeks 2-4 HTML dashboard: the modeling arc.

Produces reports/week2to4_dashboard.html: one self-contained, scrollable page that carries the
Week 2-4 story (feature engineering + the leakage audit, calibration-first baselines, gradient
boosting + the first fairness audit) in the same visual house style as the Week 1 dashboard. It
embeds the small aggregate bundle assembled from the three verified JSON reports and renders every
chart client-side with Plotly from a CDN, so it is a single file that opens in any browser.

No raw rows are shipped, only the already-computed metrics from:
  reports/week2_phase2_audit.json, week3_baseline_metrics.json, week4_boosting_fairness.json

Later phases (mitigation, decomposition, transparency docs) are shown as PLANNED, matching the
weekly-deliverable framing. Run AFTER the three week*_ run scripts have produced their JSON.

Run: PYTHONPATH=src ./.venv/bin/python scripts/week2to4_build_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "week2to4_dashboard.html"


def load_bundle() -> dict:
    w2 = json.loads((REPORTS / "week2_phase2_audit.json").read_text())
    w3 = json.loads((REPORTS / "week3_baseline_metrics.json").read_text())
    w4 = json.loads((REPORTS / "week4_boosting_fairness.json").read_text())
    return {"w2": w2, "w3": w3, "w4": w4}


def build_html(bundle: dict) -> str:
    data_json = json.dumps(bundle)
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EquiBirth AI &middot; Weeks 2-4: Features, Leakage, Models &amp; Fairness</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#1a1a2e; --muted:#5b5b6b; --line:#e7e7ee; --bg:#fbfbfd; --card:#ffffff;
    --blue:#0072B2; --orange:#E69F00; --green:#009E73; --red:#D55E00;
    --purple:#CC79A7; --sky:#56B4E9; --grey:#999999; --maxw:1080px;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0; background:var(--bg); color:var(--ink);
    font-family:Inter,system-ui,-apple-system,sans-serif; line-height:1.6; -webkit-font-smoothing:antialiased;}
  a{color:var(--blue); text-decoration:none}
  a:hover{text-decoration:underline}
  header.hero{padding:84px 24px 60px; text-align:center;
    background:radial-gradient(1200px 500px at 50% -10%, #eef4fb 0%, var(--bg) 60%);
    border-bottom:1px solid var(--line);}
  .kicker{font-size:13px; letter-spacing:.18em; text-transform:uppercase; color:var(--blue); font-weight:700;}
  header.hero h1{font-family:Newsreader,Georgia,serif; font-weight:500; font-size:clamp(34px,5.2vw,58px);
    line-height:1.08; margin:14px auto 10px; max-width:20ch;}
  header.hero p.sub{max-width:64ch; margin:0 auto; color:var(--muted); font-size:18px;}
  .byline{margin-top:22px; font-size:14px; color:var(--muted);}
  .scope-pill{display:inline-block; margin-top:18px; padding:7px 16px; border:1px solid var(--line);
    border-radius:999px; background:var(--card); font-size:13px; color:var(--muted); font-weight:500;}
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
  .analysis.warn{background:#fdf3ee; border-left-color:var(--red);}
  .stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:26px 0 8px;}
  .stat{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 16px;
    box-shadow:0 1px 2px rgba(20,20,50,.04);}
  .stat .num{font-size:30px; font-weight:800; letter-spacing:-.02em; line-height:1;}
  .stat .lab{font-size:13px; color:var(--muted); margin-top:8px;}
  .stat.accent .num{color:var(--blue)}
  .stat.warn .num{color:var(--red)}
  .stat.good .num{color:var(--green)}
  .chart{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:10px 8px 4px;
    margin:18px 0; box-shadow:0 1px 2px rgba(20,20,50,.04);}
  .grid2{display:grid; grid-template-columns:1fr 1fr; gap:18px;}
  @media(max-width:760px){ .grid2{grid-template-columns:1fr} }
  .tablewrap{overflow-x:auto; border:1px solid var(--line); border-radius:14px; background:var(--card);
    box-shadow:0 1px 2px rgba(20,20,50,.04); margin:18px 0;}
  table{border-collapse:collapse; width:100%; font-size:14px;}
  th,td{padding:10px 14px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap;}
  th{background:#f7f8fb; font-weight:600; color:var(--ink); font-size:12.5px; letter-spacing:.02em; text-transform:uppercase;}
  tbody tr:last-child td{border-bottom:none}
  nav.toc{position:sticky; top:0; z-index:10; background:rgba(251,251,253,.92); backdrop-filter:blur(8px);
    border-bottom:1px solid var(--line); padding:11px 24px; font-size:13.5px;}
  nav.toc .inner{max-width:var(--maxw); margin:0 auto; display:flex; gap:18px; flex-wrap:wrap; align-items:center;}
  nav.toc b{color:var(--ink); font-weight:700; margin-right:6px}
  nav.toc a{color:var(--muted); font-weight:500}
  .plan{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:16px; margin-top:22px;}
  .plan .step{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px;}
  .plan .step h4{margin:6px 0 6px; font-size:16px;}
  .plan .step p{font-size:14px; color:var(--muted); margin:0;}
  .plan .step .tag{display:inline-block; margin-top:10px; font-size:11px; letter-spacing:.1em; text-transform:uppercase;
    color:var(--orange); font-weight:700;}
  footer{padding:46px 24px 70px; text-align:center; color:var(--muted); font-size:14px;}
  .disclaimer{max-width:66ch; margin:0 auto 14px; font-size:13.5px;}
</style>
</head>
<body>

<header class="hero">
  <div class="kicker">EquiBirth AI &middot; Weeks 2-4 Deliverable</div>
  <h1>From raw fields to an honest model, without cheating</h1>
  <p class="sub">Engineering a leakage-safe feature set, proving the leakage discipline holds, and
  reporting the first calibrated models and fairness audit. Every number here is computed on the
  complete 2018 to 2024 data.</p>
  <div class="scope-pill">Scope: feature engineering &middot; leakage audit &middot; calibrated baselines &middot; boosting &middot; first fairness read</div>
  <div class="byline">Parshv Patel &middot; <a href="https://www.linkedin.com/in/parshv-patel-65a90326b/">LinkedIn</a></div>
</header>

<nav class="toc"><div class="inner">
  <b>On this page</b>
  <a href="#leakage">The leakage trap</a>
  <a href="#ceiling">The honest ceiling</a>
  <a href="#calib">Calibration</a>
  <a href="#fairness">Fairness</a>
  <a href="#next">What comes next</a>
</div></nav>

<main>

  <section id="leakage">
    <div class="eyebrow">Week 2 &middot; the most important engineering decision</div>
    <h2>The leakage trap, measured</h2>
    <p class="lead">Birth certificates record many fields at delivery (birthweight, Apgar, labor
    interventions, total prenatal visits) that predict prematurity almost perfectly and are useless as
    predictors, because they do not exist when a prediction would be made. The whole project depends on
    keeping them out.</p>
    <div class="stats" id="cards-leak"></div>
    <div class="chart" id="chart-leak"></div>
    <div class="analysis warn"><b>The lesson in one number.</b> A model trained on the barred
    post-delivery fields alone scores an AUC near 0.92. It looks near-perfect and means nothing. An
    automated gate fails the build if a barred field ever sneaks in, so the honest model below is
    trustworthy by construction. Race is a marker of exposure to racism, not a biological cause, and
    nothing here is a clinical tool.</div>
  </section>

  <section id="ceiling">
    <div class="eyebrow">Week 3 &middot; calibrated baselines</div>
    <h2>The honest performance ceiling</h2>
    <p class="lead">Trained only on legitimate pre-delivery fields and judged on rate-aware metrics
    rather than accuracy (a rare outcome makes accuracy meaningless), the models land right in the
    published administrative-data band, around AUC-ROC 0.65.</p>
    <div class="chart" id="chart-ceiling"></div>
    <div class="analysis"><b>Modest, and that is the finding.</b> The gap between the leaky model
    (about 0.92) and the honest model (about 0.65) is the point: individual prediction from birth
    records is modest, which tells us these data support population-level equity diagnosis, not
    individual risk scoring.</div>
  </section>

  <section id="calib">
    <div class="eyebrow">Week 3 &middot; the metric that matters for equity</div>
    <h2>Calibration holds, in aggregate and across race</h2>
    <p>For an equity audit, calibration matters more than raw accuracy: when the model says 10% risk,
    the real rate should be about 10%. After isotonic recalibration the error is tiny, and it holds
    within each large race group, which is the green light the fairness audit needs.</p>
    <div class="grid2">
      <div class="chart" id="chart-ece"></div>
      <div class="chart" id="chart-byrace"></div>
    </div>
    <div class="analysis"><b>An honest qualification.</b> A low calibration error means the scores are
    trustworthy on average, not that the model stratifies individuals well: most predictions sit within
    a few points of the base rate. Both things are true, and both are stated.</div>
  </section>

  <section id="fairness">
    <div class="eyebrow">Week 4 &middot; boosting and the first fairness audit</div>
    <h2>The fairness result is not the one you would guess</h2>
    <p class="lead">Tuned gradient boosting barely moves the ceiling. At a single screening threshold,
    the model is not less sensitive for higher-risk groups (the naive worry), it is more sensitive. The
    cost appears as disparate impact: those groups are flagged more often.</p>
    <div class="chart" id="chart-eo"></div>
    <div class="tablewrap"><table id="table-fair"></table></div>
    <div class="analysis"><b>A value judgment, not a verdict.</b> Higher sensitivity for higher-risk
    groups with more frequent flagging reflects the genuinely higher preterm rate those groups face. The
    tension between disparate impact and disparate treatment is real and is examined, not resolved, here.
    This is one model at one operating point.</div>
  </section>

  <section id="next">
    <div class="eyebrow">The plan from here</div>
    <h2>What comes next</h2>
    <p>The modeling scaffold is honest and calibrated. The coming weeks turn it into a full
    responsible-AI analysis.</p>
    <div class="plan">
      <div class="step"><h4>Mitigation experiments</h4><p>Compare threshold strategies and reduction
      methods, each with its accuracy, fairness, and calibration tradeoff shown honestly.</p>
      <span class="tag">Planned</span></div>
      <div class="step"><h4>Decomposition of the gap</h4><p>Quantify how much of the 4.7-point racial
      gap measured factors can account for, and how much is left unexplained.</p>
      <span class="tag">Planned</span></div>
      <div class="step"><h4>Explainability</h4><p>Global and race-stratified SHAP to see what the model
      actually leans on.</p><span class="tag">Planned</span></div>
      <div class="step"><h4>Transparency docs</h4><p>A model card and a datasheet, so the pipeline can be
      judged by someone who was not in the room.</p><span class="tag">Planned</span></div>
    </div>
  </section>

</main>

<footer>
  <p class="disclaimer">This is an educational research project, not a clinical tool. Race is used as a
  social construct and a marker of exposure to racism and structural barriers, never as a biological
  cause. All findings are associational.</p>
  <p>Parshv Patel &middot; EquiBirth AI &middot; data: CDC/NCHS Natality public-use files (NBER mirror), complete 2018-2024, singletons</p>
</footer>

<script>
const D = __DATA__;
const C = {blue:"#0072B2", orange:"#E69F00", green:"#009E73", red:"#D55E00", purple:"#CC79A7",
           sky:"#56B4E9", grey:"#999999", ink:"#1a1a2e"};
const FONT = {family:"Inter, system-ui, sans-serif", size:13, color:C.ink};
const LAYOUT = (title, x, y, extra) => Object.assign({
  title:{text:title, x:0.02, font:{size:15, color:C.ink}},
  xaxis:Object.assign({title:x, showgrid:false, linecolor:"#ccc"}, (extra&&extra.xaxis)||{}),
  yaxis:Object.assign({title:y, gridcolor:"#ECECEC", zeroline:false}, (extra&&extra.yaxis)||{}),
  plot_bgcolor:"white", paper_bgcolor:"white", font:FONT,
  margin:{l:60,r:24,t:48,b:48}, height:(extra&&extra.height)||380,
  showlegend:(extra&&extra.showlegend)||false
}, (extra&&extra.root)||{});
const CFG = {displayModeBar:false, responsive:true};

// ---- Week 2: leakage cards + bar ----
const raceRows = D.w2.rate_by_race;
document.getElementById("cards-leak").innerHTML = [
  `<div class="stat"><div class="num">${(D.w2.n_singletons_full/1e6).toFixed(1)}M</div><div class="lab">singleton births, 2018-2024</div></div>`,
  `<div class="stat warn"><div class="num">${D.w2.leakage_detector_auc.toFixed(2)}</div><div class="lab">AUC of the leaky model (barred fields)</div></div>`,
  `<div class="stat good"><div class="num">${D.w2.n_primary_features}</div><div class="lab">leakage-safe features</div></div>`,
  `<div class="stat accent"><div class="num">${D.w2.black_white_gap_pp.toFixed(1)} pp</div><div class="lab">NH Black minus NH White gap</div></div>`
].join("");

Plotly.newPlot("chart-leak", [{
  type:"bar", orientation:"h",
  x:[D.w2.leakage_detector_auc, D.w3.models.random_forest.auc_roc],
  y:["Leaky model (barred fields)", "Honest model (pre-delivery only)"],
  marker:{color:[C.red, C.blue]},
  text:[D.w2.leakage_detector_auc.toFixed(3), D.w3.models.random_forest.auc_roc.toFixed(3)],
  textposition:"auto"
}], LAYOUT("Barred fields look near-perfect; the honest model does not", "AUC-ROC", "", {
  xaxis:{range:[0.5,1.0]}, height:260}), CFG);

// ---- Week 3: ceiling bar ----
const mn = Object.keys(D.w3.models);
Plotly.newPlot("chart-ceiling", [{
  type:"bar", x:mn, y:mn.map(m=>D.w3.models[m].auc_roc),
  marker:{color:[C.grey,C.grey,C.blue]},
  text:mn.map(m=>D.w3.models[m].auc_roc.toFixed(3)), textposition:"auto"
}], LAYOUT("All three baselines sit in the published 0.62-0.70 band", "", "AUC-ROC (held-out)", {
  yaxis:{range:[0.5,0.72]},
  root:{shapes:[{type:"rect", xref:"paper", x0:0, x1:1, y0:0.62, y1:0.70,
    fillcolor:C.green, opacity:0.10, line:{width:0}}]}}), CFG);

// ---- Week 3: ECE raw vs cal ----
Plotly.newPlot("chart-ece", [
  {type:"bar", name:"raw", x:mn, y:mn.map(m=>D.w3.models[m].ece_raw), marker:{color:C.grey}},
  {type:"bar", name:"calibrated", x:mn, y:mn.map(m=>D.w3.models[m].ece_cal), marker:{color:C.green}}
], LAYOUT("Isotonic recalibration pulls the error to near zero", "", "Expected calibration error",
  {showlegend:true, root:{barmode:"group", legend:{orientation:"h", y:1.12, x:0}}}), CFG);

// ---- Week 3: AUC by race ----
const strat = D.w3.stratified.filter(s=>s.status!=="suppressed").sort((a,b)=>b.n-a.n);
Plotly.newPlot("chart-byrace", [{
  type:"bar", orientation:"h", x:strat.map(s=>s.auc_roc), y:strat.map(s=>s.group),
  marker:{color:C.blue}, text:strat.map(s=>s.auc_roc.toFixed(3)), textposition:"auto"
}], LAYOUT("Modest and similar discrimination across race groups", "AUC-ROC", "",
  {xaxis:{range:[0.5,0.7]}, height:380}), CFG);

// ---- Week 4: equal-opportunity gaps ----
const fair = D.w4.fairness.filter(r=>r.status!=="suppressed").sort((a,b)=>a.equal_opportunity_diff-b.equal_opportunity_diff);
Plotly.newPlot("chart-eo", [{
  type:"bar", orientation:"h", x:fair.map(r=>r.equal_opportunity_diff), y:fair.map(r=>r.group),
  marker:{color:fair.map(r=>r.equal_opportunity_diff>=0?C.blue:C.red)},
  error_x:{type:"data", symmetric:false,
    array:fair.map(r=>r.eo_ci[1]-r.equal_opportunity_diff),
    arrayminus:fair.map(r=>r.equal_opportunity_diff-r.eo_ci[0]), color:"#555"},
  text:fair.map(r=>(r.equal_opportunity_diff>=0?"+":"")+r.equal_opportunity_diff.toFixed(2)), textposition:"auto"
}], LAYOUT("Positive gaps: higher sensitivity for higher-risk groups (not lower)",
  "Equal-opportunity difference vs NH White", "", {height:360}), CFG);

// ---- Week 4: fairness table ----
const ft = D.w4.fairness.filter(r=>r.status!=="suppressed").sort((a,b)=>b.equal_opportunity_diff-a.equal_opportunity_diff);
document.getElementById("table-fair").innerHTML =
  "<thead><tr><th>group</th><th>equal-opp diff</th><th>disparate impact</th><th>BH p</th></tr></thead><tbody>" +
  ft.map(r=>`<tr><td>${r.group}</td><td>${(r.equal_opportunity_diff>=0?"+":"")+r.equal_opportunity_diff.toFixed(3)}</td>`+
    `<td>${r.disparate_impact_ratio.toFixed(2)}x</td><td>${r.eo_pvalue_bh.toFixed(3)}${r.eo_pvalue_bh<0.05?" *":""}</td></tr>`).join("") +
  "</tbody>";
</script>
</body>
</html>
""".replace("__DATA__", data_json)


def main() -> None:
    bundle = load_bundle()
    OUT.write_text(build_html(bundle), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
