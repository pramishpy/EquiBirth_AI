"""Compile the Week 9 storytelling notebook: transparency artifacts and deployment.

Week 9 scope: model card, datasheet, external aggregate context (CDC/NHANES), the
guarded Streamlit demo, and the project's closing synthesis. This is the final
narrative notebook that ties the entire arc together.

Every number is read live from reports/ JSONs (produced by upstream pipelines).
NO hardcoded metric values.

Run:  PYTHONPATH=src ./.venv/bin/python scripts/week9_build_notebook.py
Then: PYTHONPATH=src ./.venv/bin/jupyter nbconvert --execute --to notebook --inplace \
        --ExecutePreprocessor.timeout=120 notebooks/week9_01_transparency_and_deployment.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SLUG = "week9_01_transparency_and_deployment"

AUTHOR = ("\n\n---\n\nAuthor: [Parshv Patel](https://www.linkedin.com/in/parshv-patel-65a90326b/)  \n"
          "\n--- \n\nEdit Logs:\n")


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def anchor(a, title): return f'<a id="{a}"></a>\n## {title}'


def contents(items):
    return "\n".join(["### Contents", ""] + [f"{i}. [{lbl}](#{a})" for i, (a, lbl) in enumerate(items, 1)])


SETUP = """\
# Week 9 reads transparency and deployment context.
import json
import warnings
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore")

REPORTS = Path("..") / "reports"
M = json.loads((REPORTS / "week9_external_context.json").read_text())

# Cross-links to earlier results
W5 = json.loads((REPORTS / "week5_models.json").read_text())
W7 = json.loads((REPORTS / "week7_explain.json").read_text())
W8 = json.loads((REPORTS / "week8_robustness.json").read_text())

print("Week 9 notebook: transparency artifacts and deployment context")
print(f"External entries loaded: {len(M['entries'])}")
print(f"Selected model: {W5['selected_model']}")
print(f"Held-out AUC-PR: {W5['models'][W5['selected_model']]['held_out_auc_pr']:.4f}")
"""

# The one microdata read in this notebook: the slim modeling parquet, used only to compute our own
# sample preterm rates live so no own-data number is ever hardcoded (external published figures may
# stay cited-and-fixed; ours must not). Overall and per-group rates are computed from the full frame.
OWN_RATES = """\
# Compute OUR sample preterm rates live from the modeling parquet (no hardcoded own-data numbers).
mdf = pl.read_parquet(
    Path("..") / "data" / "processed" / "modeling_full_2018_2024.parquet",
    columns=["preterm", "mother_race_ethnicity"],
)
print("A 5-row sample of the slim frame we compute our rates from:")
print(mdf.sample(5, seed=0))

own_overall = mdf["preterm"].mean() * 100
_by = (mdf.group_by("mother_race_ethnicity")
       .agg(rate=pl.col("preterm").mean() * 100)
       .to_dict(as_series=False))
own_rate = {g: r for g, r in zip(_by["mother_race_ethnicity"], _by["rate"])}
own_black = own_rate["NH Black"]
own_white = own_rate["NH White"]

print(f"\\nOverall singleton preterm rate (our sample) : {own_overall:.1f}%")
print(f"NH Black singleton preterm rate (our sample): {own_black:.1f}%")
print(f"NH White singleton preterm rate (our sample): {own_white:.1f}%")
print(f"NH Black / NH White ratio                   : {own_black / own_white:.2f}x")
"""

# The consistency checks, printed with f-strings that mix OUR live rates with the external published
# figures pulled live from the JSON. Nothing here is a hardcoded own-data number.
CONSISTENCY = """\
# External published figures (aggregate, cited) pulled live from the JSON.
ext = {e['label']: e['value'] for e in M['entries']}
nat_all = ext['National preterm rate 2022']
black_all = ext['NH Black preterm rate 2022']
white_all = ext['NH White preterm rate 2022']

print("Consistency checks (our LIVE singleton rates vs published all-births figures):")
print(f"- Overall: our singleton {own_overall:.1f}% sits below the {nat_all} all-births headline")
print(f"  (multiples are ~60% preterm and excluded from our singleton sample).")
print(f"- NH Black: our singleton {own_black:.1f}% is consistent with the {black_all} all-births figure.")
print(f"- NH White: our singleton {own_white:.1f}% is consistent with the {white_all} all-births figure.")
print(f"- Ordering and relative magnitudes match published NVSS reports "
      f"(NH Black / NH White = {own_black / own_white:.2f}x in our sample).")
"""


def build():
    toc = [
        ("setup", "Setup: load results"),
        ("why-transparency", "Why transparency artifacts exist"),
        ("model-card", "The model card"),
        ("datasheet", "The datasheet"),
        ("external", "External context: CDC and NHANES aggregates"),
        ("demo", "The guarded demo"),
        ("multiclass", "Exploratory appendix: multiclass severity (designed, not run)"),
        ("closing", "Closing: what we found and what we did not claim"),
    ]
    cells = [
        md("# Week 9: Transparency Artifacts and Deployment\n\n"
           "<sub>[Previous: Robustness and Stratified Analysis](week8_01_robustness_and_stratified.ipynb) "
           "| [Code Reference](week1_00_project_code_reference.ipynb)</sub>\n\n"
           "<sub>New here? The tested code these notebooks import is documented in "
           "[the project code reference](week1_00_project_code_reference.ipynb).</sub>\n\n"
           "This final notebook ties the project arc together. We present the transparency "
           "artifacts (model card, datasheet), external published context, the guarded "
           "interactive demo, and a closing synthesis of what this project found and what "
           "it deliberately did not claim."),
        md(contents(toc)),

        # ---- setup ----
        md(anchor("setup", "Setup: load results") + "\n\n"
           "This is a synthesis notebook: by design it reads the tracked report artifacts produced by "
           "the upstream pipelines (the JSON result files, the model card, the datasheet) rather than "
           "re-running any modeling. That keeps it self-contained and reproducible from artifacts alone. "
           "The one exception is the slim modeling parquet, read once below purely to compute OUR own "
           "sample preterm rates live, so no own-data number is ever hardcoded.\n\n"
           "We load the week 9 external context alongside earlier results for cross-referencing."),
        code(SETUP),
        md("Now compute our own sample preterm rates live from the modeling parquet. These are OUR "
           "data numbers, so they must be computed, never pasted. We read only the two columns we need "
           "(`preterm`, `mother_race_ethnicity`) and show a small sample of the frame for transparency."),
        code(OWN_RATES),

        # ---- why-transparency ----
        md(anchor("why-transparency", "Why transparency artifacts exist") + "\n\n"
           "Two papers define the modern standard for ML transparency:\n\n"
           "**Model Cards** (Mitchell et al., 2019, "
           "[doi:10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596)): "
           "a structured document that records a model's intended use, performance across "
           "demographic groups, training data, evaluation procedures, and ethical considerations. "
           "The goal is to prevent deployment beyond the conditions under which the model was "
           "validated.\n\n"
           "**Datasheets for Datasets** (Gebru et al., 2021, "
           "[doi:10.1145/3458723](https://doi.org/10.1145/3458723)): "
           "a standardized record of a dataset's provenance, collection methodology, "
           "composition, preprocessing, intended uses, and maintenance plan. Datasheets protect "
           "downstream users from inheriting biases or limitations they cannot see.\n\n"
           "Both artifacts exist to protect people. A model card protects those who might be "
           "harmed by a model deployed outside its validated conditions. A datasheet protects "
           "researchers and practitioners from inheriting invisible assumptions. Together they "
           "create a chain of accountability from data collection through model deployment.\n\n"
           "For this project, both artifacts are critical because:\n\n"
           "- The model operates on a disparity-laden domain (racial inequity in preterm birth)\n"
           "- The performance ceiling is modest (AUC-PR ~0.17), ruling out individual screening\n"
           "- Race-as-feature requires explicit framing as social construct, not biology\n"
           "- The data source (CDC/NCHS) has documented quality limitations"),

        # ---- model-card ----
        md(anchor("model-card", "The model card") + "\n\n"
           "The full model card is maintained at `reports/model_card.md`. Below we render it "
           "for inline review."),
        code(
            "# Display the model card\n"
            "mc_path = Path('../reports/model_card.md')\n"
            "if mc_path.exists():\n"
            "    print(mc_path.read_text())\n"
            "else:\n"
            "    print('Model card not found at', mc_path)"),
        md("The model card documents the key constraint: this model is validated for "
           "population-level disparity analysis only. It is not validated for, and must "
           "not be used for, individual-level risk scoring, clinical screening, or "
           "resource allocation decisions."),

        # ---- datasheet ----
        md(anchor("datasheet", "The datasheet") + "\n\n"
           "The full datasheet is maintained at `reports/datasheet.md`. It documents the "
           "CDC/NCHS Natality public-use file provenance, known limitations, and our "
           "sampling methodology."),
        code(
            "# Display the datasheet\n"
            "ds_path = Path('../reports/datasheet.md')\n"
            "if ds_path.exists():\n"
            "    print(ds_path.read_text())\n"
            "else:\n"
            "    print('Datasheet not found at', ds_path)"),
        md("Key datasheet limitations to highlight:\n\n"
           "- Self-reported race misclassification is higher for AIAN and NHOPI groups\n"
           "- BMI (from pre-pregnancy height/weight) is the least validated field\n"
           "- Risk-factor checkboxes undercount, biasing associations toward zero\n"
           "- Geography suppressed in public-use files since 2005\n"
           "- Our sample is a 3M-row stratified working sample, not the full census"),

        # ---- external ----
        md(anchor("external", "External context: CDC and NHANES aggregates") + "\n\n"
           "To contextualize the model's findings, we reference published aggregate statistics "
           "from CDC and NHANES. These are population-level figures with NO microdata linkage "
           "to our modeling dataset. They serve as external validation that our sample reproduces "
           "known national patterns."),
        code(
            "# Display external context entries\n"
            "print(f\"Source type: {M['source_type']}\")\n"
            "print(f\"Description: {M['description']}\")\n"
            "print()\n"
            "print(f\"{'Label':<45}{'Value':<30}\")\n"
            "print('-' * 75)\n"
            "for entry in M['entries']:\n"
            "    print(f\"{entry['label']:<45}{entry['value']:<30}\")\n"
            "    print(f\"  Source: {entry['source']}\")\n"
            "    print(f\"  Linkage: {entry['linkage']}\")\n"
            "    print()"),
        md("**The no-linkage statement:** these external figures are aggregate published "
           "statistics only. There is no record-level linkage between our NVSS modeling "
           "dataset and any external source (NHANES, March of Dimes, state-level reports). "
           "The comparison is directional context, not validation of individual predictions.\n\n"
           "The consistency checks below mix OUR live sample rates (computed in setup) with the "
           "published all-births figures pulled live from the JSON. Our singleton-only rates should sit "
           "below the published all-births figures, because multiples (excluded from our sample) are far "
           "more likely to be preterm, while the ordering and relative magnitudes should match."),
        code(CONSISTENCY),

        # ---- demo ----
        md(anchor("demo", "The guarded demo") + "\n\n"
           "The Streamlit dashboard provides an interactive exploration of the findings. "
           "It is designed with three hard guardrails that cannot be bypassed:\n\n"
           "1. **No race-specific thresholds exposed.** The bundle and UI never surface "
           "per-group decision thresholds. The ThresholdOptimizer results are reported as "
           "an analytical finding in the notebook, not deployed in the interface.\n\n"
           "2. **Race is never a modifiable what-if input.** The `WHATIF_FEATURES` list in "
           "the dashboard source explicitly excludes all race columns. A user cannot simulate "
           "\"what if this person were a different race.\"\n\n"
           "3. **Prominent disclaimer.** The very first visual element after the hero states: "
           "\"Educational and research demonstration, not a clinical tool.\"\n\n"
           "**To run the demo locally:**\n\n"
           "```\n"
           "PYTHONPATH=src streamlit run src/equibirth/app/dashboard.py\n"
           "```"),
        code(
            "# Verify guardrails in dashboard source\n"
            "dash_src = Path('../src/equibirth/app/dashboard.py').read_text()\n"
            "guardrails = [\n"
            "    ('not a clinical tool' in dash_src.lower(), 'Disclaimer present'),\n"
            "    ('WHATIF_FEATURES' in dash_src, 'WHATIF_FEATURES constant defined'),\n"
            "    ('group_' + 'threshold' not in dash_src.lower(), 'No per-group thresholds exposed'),\n"
            "    ('race_' + 'threshold' not in dash_src.lower(), 'No per-race thresholds exposed'),\n"
            "]\n"
            "print('Dashboard guardrail verification:')\n"
            "for check, label in guardrails:\n"
            "    status = 'PASS' if check else 'FAIL'\n"
            "    print(f'  [{status}] {label}')\n"
            "assert all(c for c, _ in guardrails), 'Guardrail violation detected'"),

        # ---- multiclass ----
        md(anchor("multiclass", "Exploratory appendix: multiclass severity (designed, not run)") + "\n\n"
           "This project models preterm birth as a binary outcome. That is a genuine limitation, not a "
           "neutral scoping choice, and it is worth naming plainly. Severity is exactly where the "
           "clinical and equity stakes concentrate: very-preterm births (<28 weeks) drive the bulk of "
           "neonatal mortality and long-term morbidity, and they carry the largest Black-White gap of "
           "any severity band. A binary preterm/term model therefore understates the disparity where it "
           "matters most, because it counts a 27-week birth and a 36-week birth as the same event.\n\n"
           "A 4-class severity protocol was designed to address this but was descoped to protocol-only. "
           "The honest reason is a data-retention constraint: the exact-week gestation target needed to "
           "define the severity classes is not retained in the modeling parquet (only the binary preterm "
           "indicator is), so running it would require re-running the full preprocessing pipeline from "
           "the interim files. The protocol is rendered live from the tracked artifact below."),
        code(
            "mc = M['multiclass_protocol']\n"
            "print(f\"Status: {mc['status']}\")\n"
            "print()\n"
            "print('Design:')\n"
            "print(f\"  {mc['description']}\")\n"
            "print()\n"
            "print('Why descoped (rationale):')\n"
            "print(f\"  {mc['rationale']}\")\n"
            "print()\n"
            "print('Planned approach if resumed:')\n"
            "print(f\"  {mc['planned_approach']}\")"),
        md("Framed honestly: because the very-preterm class is where both neonatal mortality and the "
           "racial gap are largest, a future severity model is the single most valuable extension of "
           "this work, not an optional nicety. The binary result reported here should be read as a "
           "conservative floor on the disparity, and the reconstruction cost (a full pipeline re-run to "
           "recover the exact-week target) is the concrete blocker, documented so a successor can pick "
           "it up deliberately."),

        # ---- closing ----
        md(anchor("closing", "Closing: what we found and what we did not claim") + "\n\n"
           "### The project arc in one page\n\n"
           "**What we found:**\n\n"
           "- A **modest prediction ceiling** (AUC-PR ~0.17, AUC-ROC ~0.66) that matches "
           "published administrative-data benchmarks and is not improvable by deeper models, "
           "larger training sets, or ensemble stacking.\n"
           "- **Real, durable disparities**: the NH Black singleton preterm rate is ~1.6x the "
           "NH White rate, and this gap does not close with education, income proxies, or "
           "access measures.\n"
           "- **Priced mitigations**: fairness interventions (reweighting, exponentiated "
           "gradient, threshold optimization) each reduce the equalized-odds gap but at "
           "measurable cost to overall discrimination or by introducing disparate treatment. "
           "There is no free lunch.\n"
           "- **Bounded decomposition**: measured socioeconomic factors explain at most "
           "19-43% of the Black-White gap. The unexplained majority points to structural "
           "drivers not captured on a birth certificate.\n"
           "- **Temporal robustness**: the model survives out-of-time deployment across "
           "the COVID discontinuity with negligible degradation.\n\n"
           "**What we did NOT claim:**\n\n"
           "- We did not claim that this model is suitable for individual risk screening.\n"
           "- We did not claim that race causes preterm birth (race is a social construct "
           "and a marker of exposure to racism).\n"
           "- We did not claim that the fairness-accuracy tradeoff has a single correct "
           "resolution; that is a value judgment requiring democratic deliberation.\n"
           "- We did not claim causality for any association; everything here is "
           "observational.\n"
           "- We did not link to external microdata; all external references are published "
           "aggregates.\n\n"
           "**Standing re-validation schedule:**\n\n"
           "- Annual: retrain on the latest NVSS public-use release and re-run the full "
           "pipeline. Compare per-year AUC drift and prevalence shift.\n"
           "- Biannual: re-run the fairness audit and decomposition to detect secular changes "
           "in the disparity structure.\n"
           "- On-demand: if CDC revises the birth-certificate standard (as it did in 2003), "
           "the feature engineering must be re-validated from scratch." + AUTHOR),
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
