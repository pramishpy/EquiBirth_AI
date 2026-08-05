"""Assemble the weeks 5-9 dashboard data bundle from individual week JSONs.

Reads week5_models.json, week6_mitigation.json, week7_explain.json,
week8_robustness.json, and week9_external_context.json, then writes a single
unified bundle to reports/weeks5to9_dashboard_data.json.

HARD GUARDRAIL: NO group_threshold or race_threshold keys anywhere in output.
Race-specific thresholds are NEVER exposed in any deployed artifact.

Run:  PYTHONPATH=src ./.venv/bin/python scripts/build_weeks5to9_dashboard_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name: str) -> dict:
    p = REPORTS / name
    if not p.exists():
        raise FileNotFoundError(f"Required report not found: {p}")
    return json.loads(p.read_text())


def build_model_comparison(w5: dict) -> list[dict]:
    """5-model benchmark table: name, dev_test AUC-PR, held-out AUC-PR, held-out AUC-ROC."""
    rows = []
    for name, metrics in w5["models"].items():
        rows.append({
            "model": name,
            "dev_test_auc_pr": metrics["dev_test_auc_pr"],
            "held_out_auc_pr": metrics["held_out_auc_pr"],
            "held_out_auc_roc": metrics["held_out_auc_roc"],
        })
    return sorted(rows, key=lambda r: r["held_out_auc_pr"], reverse=True)


def build_mitigation_summary(w6: dict) -> list[dict]:
    """Per-method mitigation summary: method, worst_eo_gap, mean_abs_eo_gap, auc_pr (if available)."""
    rows = []
    for key, data in w6["mitigations"].items():
        entry = {
            "key": key,
            "method": data["method"],
            "worst_eo_gap": data["worst_group_eo_gap"],
            "mean_abs_eo_gap": data["mean_abs_eo_gap"],
        }
        if "auc_pr" in data:
            entry["auc_pr"] = data["auc_pr"]
        rows.append(entry)
    return rows


def build_shap_top_15(w7: dict) -> list[dict]:
    """Global top 15 SHAP features from week 7's top 25."""
    return [{"feature": feat, "importance": val} for feat, val in w7["global_top_25"][:15]]


def build_decomposition(w7: dict) -> dict:
    """Decomposition summary: raw_gap, explained_range, CI."""
    dec = w7["decomposition"]
    return {
        "raw_gap": dec["raw_gap"],
        "explained_range": [dec["range_low"], dec["range_high"]],
        "ci": dec["explained_pct_full_ci"],
    }


def build_drift(w8: dict) -> list[dict]:
    """7-year rows: year, n, prevalence, auc_roc, auc_pr."""
    return [
        {
            "year": d["year"],
            "n": d["n"],
            "prevalence": d["prevalence"],
            "auc_roc": d["auc_roc"],
            "auc_pr": d["auc_pr"],
        }
        for d in w8["drift"]
    ]


def enforce_guardrail(bundle: dict) -> None:
    """HARD GUARDRAIL: assert no group_threshold or race_threshold anywhere."""
    text = json.dumps(bundle).lower()
    forbidden = ["group_threshold", "race_threshold"]
    for term in forbidden:
        if term in text:
            raise ValueError(
                f"GUARDRAIL VIOLATION: '{term}' found in bundle. "
                "Race-specific thresholds must NEVER be exposed in deployed artifacts."
            )


def main():
    w5 = load("week5_models.json")
    w6 = load("week6_mitigation.json")
    w7 = load("week7_explain.json")
    w8 = load("week8_robustness.json")

    bundle = {
        "model_comparison": build_model_comparison(w5),
        "mitigation_summary": build_mitigation_summary(w6),
        "shap_top_15": build_shap_top_15(w7),
        "decomposition": build_decomposition(w7),
        "drift": build_drift(w8),
        "model_card_oneliner": (
            f"Selected model: {w5['selected_model']}; "
            f"held-out AUC-PR: {w5['models'][w5['selected_model']]['held_out_auc_pr']:.4f}"
        ),
    }

    enforce_guardrail(bundle)

    out = REPORTS / "weeks5to9_dashboard_data.json"
    out.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote {out.relative_to(ROOT)} ({len(json.dumps(bundle))} bytes)")
    print("Guardrail check: PASS (no group_threshold or race_threshold keys)")


if __name__ == "__main__":
    main()
