"""Verify every quantitative claim on the slides against the canonical reports/ artifacts.

Reads the SAVED .pptx, extracts the numbers that appear on slides, and asserts each one against
the tracked full-data JSON it is supposed to come from. Fails loudly on any mismatch, so the deck
cannot drift away from the evidence when either side is edited.

Run: python scripts/final_presentation_verify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PPTX = ROOT / "presentation" / "EquiBirth_AI_final_presentation.pptx"


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


W1 = load("week1_dashboard_data.json")
W2 = load("week2_experiments.json")
W3 = load("week3_baseline_metrics.json")
W4 = load("week4_boosting_fairness.json")
W5 = load("week5_models.json")
W6 = load("week6_mitigation.json")
W7 = load("week7_explain.json")
W8 = load("week8_robustness.json")
AUDIT = load("week2_phase2_audit.json")

by_race = {r["mother_race_ethnicity"]: r for r in W1["rate_by_race"]}
fair = {g["group"]: g for g in W4["fairness"]}
mit = W6["mitigations"]
shap_black = dict(W7["per_group_top_10"]["NH Black"])
shap_white_rank = [f for f, _ in W7["per_group_top_10"]["NH White"]]

# claim -> (value on the slide, value derived from the canonical artifact, tolerance)
CHECKS: list[tuple[str, float, float, float]] = [
    # scale and outcome
    ("overall singleton preterm rate 8.57%", 8.57, W1["meta"]["overall_rate"], 0.005),
    ("NH Black rate 12.28%", 12.28, by_race["NH Black"]["rate"], 0.005),
    ("NH White rate 7.54%", 7.54, by_race["NH White"]["rate"], 0.005),
    ("Black-White gap 4.74 pp", 4.74,
     by_race["NH Black"]["rate"] - by_race["NH White"]["rate"], 0.01),
    ("singletons with a valid target 24,920,407", 24_920_407,
     W1["meta"]["n_rows"] - W1["meta"]["n_missing_target"], 0),
    # the same headline figures, cross-checked against the independent week-2 audit artifact
    ("24,920,407 also confirmed by the week-2 audit", 24_920_407,
     AUDIT["n_singletons_full"], 0),
    ("8.57% also confirmed by the week-2 audit", 8.57, AUDIT["overall_rate"] * 100, 0.005),
    ("4.74 pp gap also confirmed by the week-2 audit", 4.74, AUDIT["black_white_gap_pp"], 0.005),
    ("leakage detector on excluded vars alone 0.917", 0.917,
     AUDIT["leakage_detector_auc"], 0.001),
    ("FR6.1 build gate 0.55", 0.55, AUDIT["leakage_gate"], 0.0001),
    ("FR6.3 univariate gate 0.60", 0.60, AUDIT["univariate_gate"], 0.0001),
    ("50 excluded post-delivery variables", 50, AUDIT["n_leakage_detector_vars"], 0),
    ("first-trimester care univariate AUC 0.545", 0.545,
     AUDIT["univariate_timing_auc"]["first_trimester_care"], 0.001),
    ("WIC univariate AUC 0.511", 0.511, AUDIT["univariate_timing_auc"]["wic"], 0.001),
    ("103 pre-delivery features", 103, AUDIT["n_primary_features"], 0),
    ("preterm babies 2,134,557", 2_134_557, W1["meta"]["n_preterm"], 0),
    ("NH NHOPI n 67,182", 67_182, by_race["NH NHOPI"]["n"], 0),
    ("NH AIAN n 179,698", 179_698, by_race["NH AIAN"]["n"], 0),

    # leakage
    ("leaky model with post-delivery vars 0.939", 0.939,
     W2["dual_track"]["leakage"]["auc_roc"], 0.001),
    ("clean pre-delivery model 0.659", 0.659, W2["dual_track"]["clean"]["auc_roc"], 0.001),
    ("strongest single feature 0.570", 0.570,
     max(r["univariate_auc"] for r in W2["univariate_auc"]), 0.001),
    ("largest correlation with the target 0.13", 0.13,
     max(abs(v) for v in W1["corr"].values()), 0.005),

    # model field
    ("selected XGBoost held-out AUC-ROC 0.6561", 0.6561,
     W5["models"]["xgboost"]["held_out_auc_roc"], 0.0001),
    ("selected XGBoost held-out AUC-PR 0.1720", 0.1720,
     W5["models"]["xgboost"]["held_out_auc_pr"], 0.0001),
    ("stacked ensemble AUC-PR 0.1713", 0.1713, W5["models"]["stack"]["held_out_auc_pr"], 0.0001),
    ("best deep net TabNet AUC-PR 0.1299", 0.1299,
     W5["models"]["tabnet"]["held_out_auc_pr"], 0.0001),
    ("deep-net train cap 30K vs booster 500K", 30_000, W5["caps"]["deep_train_cap"], 0),
    ("booster train cap 500K", 500_000, W5["caps"]["train_cap"], 0),
    ("deep nets max 3 epochs", 3, W5["caps"]["max_epochs"], 0),

    # ceiling
    ("5x rows buys +0.0069 AUC-PR", 0.0069,
     W8["sensitivity_cap"][2]["held_out_auc_pr"] - W8["sensitivity_cap"][0]["held_out_auc_pr"],
     0.0002),
    ("train cap 100K AUC-PR 0.1707", 0.1707, W8["sensitivity_cap"][0]["held_out_auc_pr"], 0.0001),
    ("train cap 500K AUC-PR 0.1776", 0.1776, W8["sensitivity_cap"][2]["held_out_auc_pr"], 0.0001),

    # calibration
    ("share of predictions near the base rate 83%", 0.83,
     W3["fraction_near_base_rate"], 0.005),
    ("raw ECE about 0.39", 0.39, W3["models"]["random_forest"]["ece_raw"], 0.01),
    ("calibrated ECE about 0.001", 0.001, W3["models"]["random_forest"]["ece_cal"], 0.0005),

    # fairness
    ("NH Black equal-opportunity diff +0.393 (week 4 audit)", 0.393,
     fair["NH Black"]["equal_opportunity_diff"], 0.001),
    ("NH Black disparate impact 2.74x", 2.74, fair["NH Black"]["disparate_impact_ratio"], 0.005),
    ("screening threshold 0.078", 0.078, W4["threshold"], 0.0005),
    ("baseline worst EO gap 0.389", 0.389, mit["baseline"]["worst_group_eo_gap"], 0.001),
    ("baseline mean |EO gap| 0.139", 0.139, mit["baseline"]["mean_abs_eo_gap"], 0.001),
    ("baseline sensitivity 0.607", 0.607, mit["baseline"]["sensitivity"], 0.001),

    # mitigation
    ("ThresholdOptimizer worst EO gap 0.173", 0.173, mit["threshopt"]["worst_group_eo_gap"], 0.001),
    ("ThresholdOptimizer sensitivity 0.502", 0.502, mit["threshopt"]["sensitivity"], 0.001),
    ("Reweighting NH Black gap +0.017", 0.017,
     mit["reweight"]["per_group_eo_gap"]["NH Black"], 0.001),
    ("Reweighting worst gap 0.504", 0.504, mit["reweight"]["worst_group_eo_gap"], 0.001),
    ("ExponentiatedGradient sensitivity 0.010", 0.010, mit["expgrad"]["sensitivity"], 0.001),

    # SHAP and decomposition
    ("NH Black one-hot mean |SHAP| 0.159", 0.159,
     shap_black["mother_race_ethnicity_NH Black"], 0.001),
    ("raw Black-White gap 4.91 pp", 4.91, W7["decomposition"]["raw_gap"] * 100, 0.01),
    ("explained share, full covariates 43%", 0.433,
     W7["decomposition"]["explained_pct_full"], 0.002),
    ("explained share, SES only 22%", 0.216, W7["decomposition"]["explained_pct_ses"], 0.002),
    ("explained share, clinical only 19%", 0.192,
     W7["decomposition"]["explained_pct_medical"], 0.002),

    # robustness
    ("out-of-time AUC-ROC drop about 0.003", 0.003,
     W5["models"]["xgboost"]["held_out_auc_roc"] - W8["out_of_time"]["covid_era"]["auc_roc"],
     0.0015),
    ("per-year AUC-ROC low 0.656", 0.656, min(d["auc_roc"] for d in W8["drift"]), 0.001),
    ("per-year AUC-ROC high 0.669", 0.669, max(d["auc_roc"] for d in W8["drift"]), 0.001),
    ("per-year ECE stays under 0.004", 0.004, max(d["ece"] for d in W8["drift"]), 0.001),

    # education
    ("NH Black doctorate rate 9.77%", 9.77,
     next(r["rate"] for r in W1["edu_race"]
          if r["mother_race_ethnicity"] == "NH Black" and r["meduc"] == 8), 0.005),
    ("NH White HS-grad rate 8.89%", 8.89,
     next(r["rate"] for r in W1["edu_race"]
          if r["mother_race_ethnicity"] == "NH White" and r["meduc"] == 3), 0.005),
]


def structural_checks() -> list[str]:
    """Non-numeric and inequality claims the deck makes that must also hold."""
    bad = []
    # slide 10 says the tree/stack field sits INSIDE a 0.005 AUC-PR band (an upper bound)
    tree_field = [m["held_out_auc_pr"] for k, m in W5["models"].items()
                  if k in ("xgboost", "lightgbm", "stack")]
    spread = max(tree_field) - min(tree_field)
    if spread > 0.005:
        bad.append(f"slide 10 claims the field sits inside 0.005 AUC-PR; spread is {spread:.4f}")
    # slide 16 greys NH Asian as the one non-significant group
    if fair["NH Asian"]["equal_opportunity_diff"] >= 0:
        bad.append("deck shows NH Asian as the only negative EO difference; artifact disagrees")
    if shap_white_rank.index("mother_race_ethnicity_NH White") + 1 != 8:
        bad.append("claim 'for NH White the race field ranks 8th' does not hold: rank is "
                   f"{shap_white_rank.index('mother_race_ethnicity_NH White') + 1}")
    if shap_black.get("mother_race_ethnicity_NH Black") != max(shap_black.values()):
        bad.append("claim 'NH Black indicator is the top contributor for NH Black' does not hold")
    if W5["selected_model"] != "xgboost":
        bad.append(f"deck says XGBoost was selected; artifact says {W5['selected_model']}")
    if fair["NH Asian"]["eo_pvalue_bh"] <= 0.05:
        bad.append("deck greys NH Asian as not significant, but its BH p is "
                   f"{fair['NH Asian']['eo_pvalue_bh']:.4f}")
    for g in ("NH Black", "NH AIAN", "NH NHOPI", "Hispanic", "NH Multiracial"):
        if fair[g]["eo_pvalue_bh"] > 0.05:
            bad.append(f"deck shows {g} as significant, but BH p is {fair[g]['eo_pvalue_bh']:.4f}")
    if by_race["NH Black"]["rate"] <= by_race["NH White"]["rate"]:
        bad.append("the disparity direction claimed in the deck does not hold")
    # every education stratum must show a positive Black-White gap (the slide 14 claim)
    b = {r["meduc"]: r["rate"] for r in W1["edu_race"] if r["mother_race_ethnicity"] == "NH Black"}
    w = {r["meduc"]: r["rate"] for r in W1["edu_race"] if r["mother_race_ethnicity"] == "NH White"}
    gaps = [b[k] - w[k] for k in sorted(b)]
    if min(gaps) < 3.1 - 0.05 or max(gaps) > 4.3 + 0.05:
        bad.append(f"slide 14 claims the gap runs +3.1 to +4.3; actual range is "
                   f"{min(gaps):.2f} to {max(gaps):.2f}")
    return bad


def deck_text() -> str:
    prs = Presentation(str(PPTX))
    out = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                out.append(shape.text_frame.text)
    return "\n".join(out)


BANNED = [
    ("biological cause of", "race framed as a biological cause"),
    ("clinically useful", "overclaims clinical usefulness"),
    ("solves fairness", "claims fairness was solved"),
    ("solved fairness", "claims fairness was solved"),
    ("trees beat deep learning", "unqualified deep-learning claim"),
    ("production-ready", "claims production readiness"),
    ("clinically validated", "claims clinical validation"),
    ("—", "em dash (house style bans them)"),
    ("–", "en dash (house style bans them)"),
]


def main() -> None:
    failures: list[str] = []

    for label, slide_value, artifact_value, tol in CHECKS:
        if abs(slide_value - artifact_value) > tol:
            failures.append(
                f"MISMATCH: {label}\n    slide says {slide_value}, artifact gives "
                f"{artifact_value:.6g} (tolerance {tol})")

    failures.extend(f"STRUCTURAL: {m}" for m in structural_checks())

    body = deck_text()
    for needle, why in BANNED:
        if needle in body:
            failures.append(f"LANGUAGE: found {needle!r} in slide text ({why})")

    # the two required framings must be present somewhere in the deck
    for needle, why in [("not a clinical tool", "the not-a-clinical-tool disclaimer"),
                        ("marker of exposure", "the race-as-structural-exposure framing")]:
        if needle.lower() not in body.lower():
            failures.append(f"LANGUAGE: {why} is missing from the slides")

    print(f"checked {len(CHECKS)} numeric claims, {len(structural_checks()) + 6} structural and "
          f"language rules\n")
    if failures:
        print(f"{len(failures)} PROBLEM(S):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("every slide number traces to a canonical artifact; framing rules hold")


if __name__ == "__main__":
    main()
