# EquiBirth AI

Predicting preterm birth and auditing racial equity in U.S. maternal health.

EquiBirth AI predicts preterm birth (delivery before 37 completed weeks) from U.S.
birth-certificate features and audits whether such models, and the underlying outcomes, are
equitable across race and ethnicity and SES proxies (education, payment source, WIC). The honest
thesis: individual prediction from administrative data is modest, while the racial disparity is
large and only partly explained by measured factors. Responsible AI is the organizing narrative,
not a final checkbox.

**This is an educational and research project. Nothing here is a clinical decision tool.**
Race is treated throughout as a marker of exposure to racism, never as a biological cause.

## Status

Nine weeks of work, complete, on the full CDC/NCHS Natality public-use files for 2018-2024:
24,920,407 singleton births with a valid target. The outcome and equity audit uses every row;
model development uses a stratified 3,000,000-row working sample with a 1,498,760-row untouched
held-out partition. Findings F1-F27 are logged in `FINDINGS.md`, decisions in `DECISION_LOG.md`.

Delivered: ingestion with a format-confirmation gate, feature engineering behind enforced leakage
gates, calibration-first baselines, gradient boosting with Optuna, a deep-learning benchmark
(TabNet, FT-Transformer), out-of-fold stacking, a formal fairness audit, three priced fairness
mitigations, SHAP, a g-computation gap decomposition, out-of-time and drift validation,
race-stratified models, and a transparency layer (Model Card, Datasheet).

### Headline results

| Result | Value | Source |
|---|---|---|
| Overall singleton preterm rate | 8.57% | `reports/week2_phase2_audit.md` |
| NH Black minus NH White gap | 4.74 pp (12.28% vs 7.54%) | `reports/week2_phase2_audit.md` |
| Honest pre-delivery ceiling | AUC-ROC 0.6561, AUC-PR 0.1720 (XGBoost, held-out) | `reports/week5_models.md` |
| Leaky post-delivery model | AUC 0.917 on excluded variables alone (the documented trap) | `reports/week2_phase2_audit.md` |
| Best deep net | TabNet AUC-PR 0.1299, below the boosters | `reports/week5_models.md` |
| Black-White gap decomposition | raw 0.04912, unexplained 0.02783 | `reports/week7_explain.md` |
| Out-of-time stability | AUC-PR 0.1688 (COVID 2020-2021), 0.1721 (2022-2024) | `reports/week8_robustness.md` |

Two results deserve emphasis because they are the opposite of the naive worry. At a single
screening threshold the model has *higher* sensitivity for higher-risk groups (NH Black equal
opportunity +0.389 vs NH White), so the cost appears as disparate impact, not missed cases. And
low calibration error holds only in aggregate: 82% of predictions sit within 5 pp of the base
rate, so this is not reliable individual risk stratification. The unexplained share of the gap is
a lower bound on structural contributions, because the measured covariates are themselves
products of structural racism.

## Deliverables

**Notebooks** (`notebooks/`) are the storytelling spine. Read them in order; each is
self-contained, computes every number live, and cross-links to the next.

| Notebook | What it delivers |
|---|---|
| `week1_00_project_code_reference.ipynb` | Live source of every module the notebooks import, with a module-to-test map |
| `week1_01_problem_and_acquisition.ipynb` | Problem framing, data provenance and governance, lifecycle map |
| `week1_02_exploration.ipynb` | Shape, missingness and its MNAR mechanism, field dictionary, known biases |
| `week1_03_eda_and_visualization.ipynb` | Full EDA: prevalence, the racial disparity, effect sizes with CIs, COVID deltas |
| `week2_01_features_and_leakage.ipynb` | The leakage trap measured, the 103-feature safe matrix, per-feature reasoning |
| `week2_02_feature_experiments.ipynb` | Dual-track leakage-vs-clean experiment, contribution, MI, PCA, clustering |
| `week3_01_calibrated_baselines.ipynb` | Why accuracy is the wrong metric; calibration as the real target |
| `week4_01_boosting_and_fairness.ipynb` | Does boosting move the ceiling; the first fairness audit |
| `week5_01_deep_learning_and_selection.ipynb` | Deep-learning benchmark, stacking, final model selection |
| `week6_01_fairness_mitigation.ipynb` | Three mitigations priced on one tradeoff table |
| `week7_01_explainability_and_decomposition.ipynb` | Global and race-stratified SHAP, gap decomposition |
| `week8_01_robustness_and_stratified.ipynb` | Out-of-time validation, yearly drift, stratified models |
| `week9_01_transparency_and_deployment.ipynb` | Model Card, Datasheet, external context, the guarded demo |

**Reports** (`reports/`) hold the machine-written artifacts each week's run produced: paired
`.md`/`.json` per week, plus `model_card.md` (Mitchell et al. 2019) and `datasheet.md`
(Gebru et al. 2021).

**Dashboards.** Two static, single-file HTML pages, `reports/week1_dashboard.html` and
`reports/week2to4_dashboard.html` (open directly in a browser; they fetch Plotly from a pinned
CDN). Weeks 5-9 are surfaced instead through the interactive Streamlit app rather than a static
page.

**Interactive app.** `src/equibirth/app/dashboard.py`, a continuous-scroll equity dashboard.
Guardrails are enforced in code and tested: a prominent not-a-clinical-tool disclaimer, no
individual risk calculator, no race-specific thresholds, and race is never a modifiable what-if
input.

> Note on the app's data: the exploration and EDA sections read
> `reports/dashboard_data.json`, a validated representative spread sample (1,986,314 births,
> 2019-2024), which the page discloses on screen. The weeks 5-9 panels read
> `reports/weeks5to9_dashboard_data.json`, which is full-data. The canonical numbers for citation
> are the full-data ones in `FINDINGS.md` and `reports/week*`.

**Final presentation** (`presentation/`). A 13-slide editable `.pptx` for a 10-minute talk, plus
`SPEAKER_SCRIPT.md`, a word-for-word script sized to land at 9:40 at a normal speaking pace. Every
figure and every number on the slides is generated from the tracked artifacts in `reports/`; two
build scripts assemble it and three auditors gate it, so no slide claim is hand-typed. See
[`presentation/README.md`](presentation/README.md) for the slide-by-slide narrative.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                      # 137 passed, 4 skipped (skips need the gitignored raw data)
```

Run the interactive dashboard (needs only the 4 packages in `requirements-app.txt`):

```bash
streamlit run src/equibirth/app/dashboard.py
```

## Reproducing the analysis from a clean clone

No data is committed: the raw files are ~13 GB and the derived tables are gitignored. Everything
below regenerates from public sources. Expect the download to take hours (the mirror throttles
per connection) and the modeling steps to be CPU-bound.

```bash
# 0. download the 7 full-year CSVs (~13 GB, resumable, byte-size verified)
bash scripts/week1_download_full.sh

# 1. build the cleaned per-year and full tables
PYTHONPATH=src python scripts/week1_build_full_dataset.py
PYTHONPATH=src python scripts/week1_build_data.py

# 2. features, leakage audit, and the full-population equity audit
PYTHONPATH=src python scripts/week2_feature_leakage_audit.py
PYTHONPATH=src python scripts/week2_experiments.py

# 3-4. calibrated baselines, then boosting + the first fairness audit
PYTHONPATH=src python scripts/week3_baselines.py
PYTHONPATH=src python scripts/week4_boosting_fairness.py

# 5-9. deep benchmark and selection, mitigation, SHAP, robustness, transparency
PYTHONPATH=src python scripts/week5_deep_and_ensemble.py
PYTHONPATH=src python scripts/week6_mitigation.py
PYTHONPATH=src python scripts/week7_explain_decompose.py
PYTHONPATH=src python scripts/week8_robustness.py
PYTHONPATH=src python scripts/week9_transparency.py

# dashboards and notebooks are compiled LAST, from the verified artifacts above
PYTHONPATH=src python scripts/build_weeks5to9_dashboard_data.py
PYTHONPATH=src python scripts/week1_build_dashboard.py
PYTHONPATH=src python scripts/week2to4_build_dashboard.py
PYTHONPATH=src python scripts/week1_build_notebooks.py    # and week2..week9_build_notebook.py
```

Steps 5-8 read the tuned hyperparameters out of `reports/week4_boosting_fairness.json` rather
than re-tuning, so run them in order. No model artifacts are serialized; every script retrains
from scratch, which keeps reproduction fully determined by data plus seed.

`make test`, `make lint`, and `make typecheck` are also available.

## Repository layout

```
src/equibirth/   ingest, features, models, fairness, explain, eval, explore, viz, app, utils
tests/           mirrors src/ 1:1 (137 passing)
notebooks/       week1..week9 storytelling series, compiled last from verified scripts
scripts/         weekN_* pipeline runners, notebook/dashboard builders, presentation builders
reports/         per-week md/json artifacts, model card, datasheet, HTML dashboards
presentation/    the 13-slide final deck, its speaker script, and generated figures
data/            raw, interim, processed (gitignored; populated by the scripts above)
docs/process/    week-1 working notes
```

## Data

CDC/NCHS Natality public-use microdata for 2018-2024, retrieved from the NBER mirror
(`https://data.nber.org/nvss/natality/csv/{year}/natality{year}us.csv`). No authentication and no
API key: the files are free, de-identified, and geography-suppressed post-2005, so no IRB is
required. Aggregate context (CDC WONDER, published NVSR tables) is cited where used rather than
re-downloaded. Only small aggregates are committed; row-level data never leaves your machine.

## Reproducibility

Fixed seed 42 set in every runner, a single `config.yaml`, deterministic saved splits, byte-size
integrity gates on every download, and a test suite mirroring `src/`. Every reported number
carries its source script in `FINDINGS.md`.

## License

MIT, see [LICENSE](LICENSE). The research write-ups in `FINDINGS.md`, `DECISION_LOG.md`, and
`reports/` are shared for educational use; please cite rather than reproduce wholesale, and do not
present any of it as clinical guidance.

---

Author: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)
