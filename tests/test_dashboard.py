"""Smoke and guardrail tests for the continuous-scroll equity dashboard.

These do not assert pixel layout; they assert that the app runs, renders its narrative and figures, keeps
the SPEC FR11 guardrails (not-a-clinical-tool disclaimer, no individual risk calculator, no race-specific
threshold control), and contains no em or en dashes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "equibirth" / "app" / "dashboard.py"


@pytest.fixture(scope="module")
def app():
    at = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(str(APP), default_timeout=180)
    return at.run()


def test_runs_without_exception(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_renders_many_charts(app):
    # The story is chart-driven; a continuous publication page should render a rich set of figures.
    assert len(list(app.get("plotly_chart"))) >= 10


def test_not_a_clinical_tool_disclaimer_present(app):
    blob = " ".join(m.value for m in app.markdown).lower()
    assert "not a clinical tool" in blob
    assert "marker of exposure to racism" in blob


def test_source_has_no_em_or_en_dashes():
    src = APP.read_text(encoding="utf-8")
    assert src.count("—") == 0, "em dash present"
    assert src.count("–") == 0, "en dash present"
    assert "&mdash;" not in src and "&ndash;" not in src


def test_rate_by_race_is_in_percent_not_fraction():
    # Regression: the bundle stores rates as fractions; the chart must scale to percent (7-13%, not 0.1).
    import importlib.util

    spec = importlib.util.spec_from_file_location("equibirth_dashboard", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [{"mother_race_ethnicity": "NH White", "rate": 0.0779, "n": 989848},
            {"mother_race_ethnicity": "NH Black", "rate": 0.1286, "n": 305837}]
    fig = mod.fig_rate_by_race(rows)
    xvals = list(fig.data[0].x)
    assert max(xvals) > 5, f"rates look like fractions, not percent: {xvals}"
    assert max(xvals) == pytest.approx(12.86, abs=0.01)


def test_eda_aggregates_embedded_for_deployment():
    # Deployment guarantee: the committed bundle must carry the precomputed EDA block, so the cloud app
    # (which has no access to the gitignored parquet) still renders the exploration sections.
    import json

    bundle = json.loads((ROOT / "reports" / "dashboard_data.json").read_text())
    assert "eda" in bundle, "dashboard_data.json missing 'eda' block; run scripts/build_dashboard_data.py"
    eda = bundle["eda"]
    for key in ("rate_by_race", "by_year", "prior_preterm", "medical_risk", "corr", "age_hist",
                "edu_race"):
        assert key in eda and eda[key], f"eda block missing/empty: {key}"
    # rates are stored in percent, not fractions
    assert max(r["rate"] for r in eda["rate_by_race"]) > 5


ROOT = APP.parents[3]


def test_no_individual_risk_calculator_or_race_threshold():
    # Guardrail: the page must not offer person-level inputs or race-specific decision controls.
    src = APP.read_text(encoding="utf-8").lower()
    assert "number_input" not in src, "no individual numeric risk input allowed"
    assert "risk calculator" not in src or "no individual risk calculator" in src


def test_bundle_contains_no_race_threshold_controls():
    from pathlib import Path
    p = Path("reports/weeks5to9_dashboard_data.json")
    if not p.exists():
        import pytest
        pytest.skip("bundle not built yet")
    text = p.read_text().lower()
    assert "group_threshold" not in text and "race_threshold" not in text


def test_dashboard_source_has_disclaimer():
    from pathlib import Path
    src = Path("src/equibirth/app/dashboard.py").read_text()
    assert "not a clinical tool" in src.lower()
