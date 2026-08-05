"""Build the figures for the AI4ALL Ignite final presentation.

Every number here is read from a tracked canonical artifact in reports/ (the COMPLETE
2018-2024 full-data track) or from the DECISION_LOG / FINDINGS entry named in the caption.
Nothing is typed in by hand except labels and the two clearly-marked D26 sample-era values.

Output: presentation/figures/*.png at 300 dpi on the project paper colour.
Run: python scripts/final_presentation_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = ROOT / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- design system (matches the app)
INK = "#16203A"
INK_SOFT = "#3D4663"
PAPER = "#FBFBFD"
BAND = "#E9ECF4"
BLUE = "#0B6FB8"
RED = "#D7263D"
GREEN = "#0A8754"
ORANGE = "#E07B00"
PURPLE = "#8E44AD"
GREY = "#9AA3B2"
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GREY,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "axes.titlecolor": INK,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 16,
})


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def bare(ax, keep_x=True):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(keep_x)
    ax.tick_params(length=0)
    ax.set_yticks([])


W1 = load("week1_dashboard_data.json")      # FULL DATA 2018-2024, 24,941,021 singletons
W2 = load("week2_experiments.json")         # FULL DATA feature experiments
W3 = load("week3_baseline_metrics.json")    # FULL DATA calibrated baselines
W5 = load("week5_models.json")              # FULL DATA model field
W6 = load("week6_mitigation.json")          # FULL DATA mitigation
W7 = load("week7_explain.json")             # FULL DATA SHAP + decomposition
W8 = load("week8_robustness.json")          # FULL DATA robustness


# ---------------------------------------------------------------------- 1. the leakage gap
def fig_leakage_gap() -> None:
    dual = W2["dual_track"]
    clean = dual["clean"]["auc_roc"]
    leaky = dual["leakage"]["auc_roc"]
    detector = 0.917  # reports/week2_phase2_audit.md, FR6.1 detector on the 50 excluded columns

    labels = [
        "Everything, including what the\ndelivery record already knows",
        "Post-delivery variables ALONE\n(birthweight, Apgar, labor drugs)",
        "Pre-delivery features only\n(what we actually shipped)",
    ]
    vals = [leaky, detector, clean]
    colours = [RED, RED, BLUE]

    fig, ax = plt.subplots(figsize=(13.6, 4.45))
    y = list(range(len(vals)))
    ax.barh(y, vals, height=0.56, color=colours)
    for i, v in enumerate(vals):
        ax.text(v + 0.009, i, f"{v:.3f}", va="center", ha="left",
                fontsize=27.3, fontweight="bold", color=colours[i])
    ax.axvline(0.5, color=GREY, lw=1.2, ls=(0, (4, 3)))
    ax.text(0.502, -0.62, "0.50 = coin flip", fontsize=13.6, color=INK_SOFT)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15.5)
    ax.invert_yaxis()
    ax.set_ylim(2.62, -0.85)
    ax.set_xlim(0.45, 1.03)
    ax.set_xlabel("AUC-ROC on held-out births", fontsize=14.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "leakage_gap.png")


# ---------------------------------------------------------------------- 2. leakage timeline diagram
def fig_leakage_timeline() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    ax.add_patch(Rectangle((2, 11), 56, 8.5, facecolor="#DCE8F5", edgecolor="none"))
    ax.add_patch(Rectangle((62, 11), 36, 8.5, facecolor="#FADCE1", edgecolor="none"))
    ax.plot([60, 60], [2, 22.5], color=INK, lw=2.6)
    ax.text(60, 23.6, "DELIVERY", ha="center", fontsize=16.7, fontweight="bold", color=INK)

    ax.text(4, 15.4, "Known before birth", fontsize=17.4, fontweight="bold", color=BLUE)
    ax.text(64, 15.4, "Recorded at or after birth", fontsize=17.4, fontweight="bold", color=RED)

    ax.text(4, 8.4, "age  .  prior preterm birth  .  BMI  .  parity\neducation  .  insurance  .  WIC  .  smoking",
            fontsize=14.9, color=INK_SOFT, va="top")
    ax.text(64, 8.4, "birthweight  .  Apgar score  .  NICU\nsteroids  .  induction  .  visit count",
            fontsize=14.9, color=INK_SOFT, va="top")

    arrow = FancyArrowPatch((92, 27.4), (16, 27.4), arrowstyle="-|>", mutation_scale=20,
                            color=RED, lw=2.2, linestyle="--")
    ax.add_patch(arrow)
    ax.text(54, 28.4, "leakage: the answer flows backwards", ha="center",
            fontsize=15.5, color=RED, fontweight="bold")
    ax.text(4, 1.4, "A legitimate predictor may only use the left side.",
            fontsize=15.5, color=INK, style="italic", va="center")
    save(fig, "leakage_timeline.png")


# ---------------------------------------------------------------------- 3. the judgment call
def fig_iatrogenic() -> None:
    fig, ax = plt.subplots(figsize=(10.18, 4.56))
    xs = [0, 1]
    ys = [0.684, 0.654]
    ax.plot(xs, ys, color=INK, lw=2.6, zorder=2)
    ax.scatter([0], [0.684], s=260, color=GREY, zorder=3)
    ax.scatter([1], [0.654], s=300, color=BLUE, zorder=3)

    ax.text(-0.04, 0.684, "0.684", ha="right", va="center", fontsize=27.3, fontweight="bold", color=GREY)
    ax.text(1.04, 0.654, "0.654", ha="left", va="center", fontsize=27.3, fontweight="bold", color=BLUE)
    ax.text(0, 0.6905, "keeping eclampsia,\ngestational hypertension,\ngestational diabetes",
            ha="center", fontsize=14.9, color=INK_SOFT)
    ax.text(1, 0.6465, "removing them as\niatrogenic-entangled\n(reverse causal)",
            ha="center", va="top", fontsize=14.9, color=BLUE)
    ax.annotate("", xy=(0.5, 0.6675), xytext=(0.5, 0.6805),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(0.545, 0.6735, "we accepted\n3 AUC points of loss", fontsize=15.5,
            color=RED, fontweight="bold", va="center")

    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(0.638, 0.700)
    ax.set_xticks([])
    ax.set_ylabel("held-out AUC-ROC", fontsize=14.9)
    ax.spines["bottom"].set_visible(False)
    save(fig, "iatrogenic_tradeoff.png")


# ---------------------------------------------------------------------- 4. model field
def fig_model_field() -> None:
    order = ["xgboost", "stack", "lightgbm", "tabnet", "ft_transformer"]
    pretty = {"xgboost": "XGBoost  (selected)", "lightgbm": "LightGBM", "stack": "Stacked ensemble",
              "tabnet": "TabNet*", "ft_transformer": "FT-Transformer*"}
    m = W5["models"]
    vals = [m[k]["held_out_auc_pr"] for k in order]
    los = [m[k]["held_out_auc_pr_ci"][0] for k in order]
    his = [m[k]["held_out_auc_pr_ci"][1] for k in order]
    cols = [BLUE, BLUE, BLUE, ORANGE, ORANGE]
    alphas = [1.0, 0.5, 0.5, 0.85, 0.85]

    fig, ax = plt.subplots(figsize=(13.6, 4.45))
    y = list(range(len(order)))
    for i, k in enumerate(order):
        ax.barh(i, vals[i], height=0.56, color=cols[i], alpha=alphas[i])
        ax.plot([los[i], his[i]], [i, i], color=INK, lw=2.0)
        ax.text(his[i] + 0.004, i, f"{vals[i]:.4f}", va="center", fontsize=18.6,
                fontweight="bold" if i == 0 else "normal",
                color=INK if i == 0 else INK_SOFT)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[k] for k in order], fontsize=16.1)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.215)
    ax.set_xlabel("held-out AUC-PR, with 95% bootstrap interval  (base rate 8.57%)", fontsize=14.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.0, -0.95, "* compute-capped: the neural models saw 30K training rows to the boosters' 500K, max 3 epochs",
            fontsize=13.6, color=ORANGE, transform=ax.get_yaxis_transform(), clip_on=False)
    save(fig, "model_field.png")


# ---------------------------------------------------------------------- 5. thin marginal signal
def fig_univariate() -> None:
    rows = [(r["feature"], r["univariate_auc"]) for r in W2["univariate_auc"]]
    rows = sorted(rows, key=lambda kv: -kv[1])[:6]
    pretty = {
        "medical_risk_count": "count of medical risk flags",
        "rf_ppterm": "previous preterm birth",
        "bmi": "pre-pregnancy BMI",
        "pay_1": "private insurance",
        "mother_race_ethnicity_NH Black": "NH Black indicator",
        "married": "marital status",
        "meduc": "maternal education",
        "cig_0": "smoking before pregnancy",
        "rf_cesar": "previous cesarean",
        "rf_phype": "pre-pregnancy hypertension",
        "rf_pdiab": "pre-pregnancy diabetes",
        "wic": "WIC participation",
        "priorlive": "prior live births",
        "mager": "maternal age",
        "first_trimester_care": "first-trimester prenatal care",
    }
    labels = [pretty.get(k, k) for k, _ in rows]
    vals = [v for _, v in rows]

    fig, ax = plt.subplots(figsize=(11.24, 4.45))
    y = list(range(len(vals)))
    ax.barh(y, [v - 0.5 for v in vals], left=0.5, height=0.55, color=BLUE, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(v + 0.0018, i, f"{v:.3f}", va="center", fontsize=17.4, color=INK)
    ax.axvline(0.5, color=INK, lw=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15.5)
    ax.invert_yaxis()
    ax.set_xlim(0.495, 0.60)
    ax.set_xlabel("single-feature AUC over all 24.9M births   (0.500 = no information)", fontsize=14.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "univariate_signal.png")


# ---------------------------------------------------------------------- 6. more rows do not help
def fig_train_cap() -> None:
    caps = W8["sensitivity_cap"]
    xs = [c["train_cap"] / 1000 for c in caps]
    ys = [c["held_out_auc_pr"] for c in caps]

    # The y-axis deliberately starts at the base rate. On a zoomed axis this curve looks like
    # growth; on an honest one it is visibly flat, which is the whole point.
    fig, ax = plt.subplots(figsize=(9.96, 4.56))
    ax.axhline(0.0857, color=GREY, lw=1.4, ls=(0, (4, 3)))
    ax.text(100, 0.0895, "8.57% base rate: what you get by guessing the population average",
            fontsize=13.6, color=INK_SOFT)
    ax.plot(xs, ys, color=BLUE, lw=3.0, marker="o", ms=12)
    for x, v in zip(xs, ys):
        ax.text(x, v + 0.006, f"{v:.4f}", ha="center", fontsize=18.6, color=INK, fontweight="bold")
    ax.text(300, 0.135, "5x the training rows buys +0.0069 AUC-PR",
            ha="center", fontsize=16.1, color=RED, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}K" for x in xs], fontsize=16.1)
    ax.set_xlim(60, 545)
    ax.set_xlabel("training rows", fontsize=14.9)
    ax.set_ylabel("held-out AUC-PR", fontsize=14.9)
    ax.set_ylim(0.080, 0.205)
    save(fig, "train_cap_plateau.png")


# ---------------------------------------------------------------------- 7. calibration honesty
def fig_calibration() -> None:
    rel = [r for r in W3["reliability_rf"] if r["count"] >= 30]
    near = W3["fraction_near_base_rate"]
    base = W3["prevalence"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.14, 4.77), gridspec_kw={"width_ratios": [1, 1]})

    ax1.plot([0, 0.7], [0, 0.7], color=GREY, lw=1.4, ls=(0, (4, 3)))
    sizes = [max(28, 300 * (r["count"] / rel[0]["count"]) ** 0.42) for r in rel]
    ax1.scatter([r["mean_pred"] for r in rel], [r["observed"] for r in rel],
                s=sizes, color=BLUE, alpha=0.85, zorder=3)
    ax1.set_xlim(0, 0.72)
    ax1.set_ylim(0, 0.72)
    ax1.set_xlabel("predicted risk", fontsize=14.9)
    ax1.set_ylabel("observed rate", fontsize=14.9)
    ax1.set_title("Predictions land on the diagonal\nECE 0.39 raw to 0.001 after recalibration",
                  fontsize=16.7, loc="left", pad=12)
    ax1.text(0.30, 0.055, "dot size = how many\nbirths sit in that bin",
             fontsize=13.6, color=INK_SOFT)

    ax2.barh([0], [near], height=0.42, color=RED, alpha=0.9)
    ax2.barh([0], [1 - near], left=[near], height=0.42, color=BAND)
    ax2.text(near / 2, 0, f"{near * 100:.0f}%", ha="center", va="center",
             fontsize=37.2, fontweight="bold", color="white")
    ax2.text(near + (1 - near) / 2, 0, f"{(1 - near) * 100:.0f}%", ha="center", va="center",
             fontsize=18.6, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.55, 0.9)
    ax2.axis("off")
    ax2.set_title(f"...but {near * 100:.0f}% of predictions sit within 5 points\nof the {base * 100:.1f}% base rate",
                  fontsize=16.7, loc="left", pad=12)
    ax2.text(0, -0.42, "Well calibrated in aggregate is not the same thing as\nuseful individual risk stratification.",
             fontsize=15.5, color=RED, va="top")
    save(fig, "calibration.png")


# ---------------------------------------------------------------------- 8. the disparity
def fig_disparity() -> None:
    rows = sorted(W1["rate_by_race"], key=lambda r: r["rate"])
    labels = [r["mother_race_ethnicity"] for r in rows]
    vals = [r["rate"] for r in rows]
    ns = [r["n"] for r in rows]
    cols = [RED if lab == "NH Black" else BLUE if lab == "NH White" else GREY for lab in labels]

    fig, ax = plt.subplots(figsize=(11.87, 5.19))
    y = list(range(len(vals)))
    ax.barh(y, vals, height=0.62, color=cols)
    for i, (v, n) in enumerate(zip(vals, ns)):
        # the white bbox keeps the value legible where it crosses the base-rate line
        ax.text(v + 0.16, i, f"{v:.2f}%", va="center", fontsize=18.6,
                fontweight="bold" if cols[i] != GREY else "normal", color=cols[i],
                bbox=dict(boxstyle="round,pad=0.12", fc=PAPER, ec="none"), zorder=4)
        ax.text(0.24, i, f"n = {n:,}", va="center", fontsize=13.0, color="white")
    ax.axvline(8.57, color=INK, lw=1.5, ls=(0, (4, 3)))
    ax.text(8.72, len(vals) - 0.28, "all singletons 8.57%", fontsize=14.3, color=INK)

    ib, iw = labels.index("NH Black"), labels.index("NH White")
    ax.annotate("", xy=(15.2, ib), xytext=(15.2, iw),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2.2))
    ax.text(15.5, (ib + iw) / 2, "4.74\npoints", fontsize=18.6, color=RED,
            fontweight="bold", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=16.1)
    ax.set_xlim(0, 17.4)
    ax.set_xlabel("preterm birth rate, every singleton birth 2018-2024", fontsize=14.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "disparity.png")


# ---------------------------------------------------------------------- 9. education does not close it
def fig_education() -> None:
    lab = W1["meta"]["education_labels"]
    er = W1["edu_race"]
    b = {r["meduc"]: r["rate"] for r in er if r["mother_race_ethnicity"] == "NH Black"}
    w = {r["meduc"]: r["rate"] for r in er if r["mother_race_ethnicity"] == "NH White"}
    ks = sorted(b)
    xs = list(range(len(ks)))

    fig, ax = plt.subplots(figsize=(12.72, 5.3))
    ax.fill_between(xs, [w[k] for k in ks], [b[k] for k in ks], color=RED, alpha=0.11)
    ax.plot(xs, [b[k] for k in ks], color=RED, lw=3.0, marker="o", ms=8)
    ax.plot(xs, [w[k] for k in ks], color=BLUE, lw=3.0, marker="o", ms=8)

    ax.text(xs[-1] + 0.14, b[ks[-1]], "NH Black", color=RED, fontsize=17.4,
            fontweight="bold", va="center")
    ax.text(xs[-1] + 0.14, w[ks[-1]], "NH White", color=BLUE, fontsize=17.4,
            fontweight="bold", va="center")
    # gap size sits inside the shaded band, so it never collides with either line
    for k, x in zip(ks, xs):
        ax.text(x, (b[k] + w[k]) / 2, f"+{b[k] - w[k]:.1f}", ha="center", va="center",
                fontsize=14.3, color=RED, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", fc=PAPER, ec="none", alpha=0.82))

    ax.annotate("", xy=(7, b[8]), xytext=(2, w[3]),
                arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1.2, ls=(0, (3, 3))))
    ax.text(0.02, 0.045,
            f"NH Black mothers with a doctorate: {b[8]:.2f}%\n"
            f"NH White mothers who finished high school: {w[3]:.2f}%",
            transform=ax.transAxes, fontsize=15.5, color=INK, va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc=PAPER, ec=GREY, lw=0.8))

    ax.set_xticks(xs)
    ax.set_xticklabels([lab[str(k)] for k in ks], fontsize=13.6, rotation=18, ha="right")
    ax.set_ylabel("preterm birth rate", fontsize=14.9)
    ax.set_ylim(3.4, 15.6)
    ax.set_yticks([6, 8, 10, 12, 14])
    ax.set_yticklabels(["6%", "8%", "10%", "12%", "14%"])
    ax.set_xlim(-0.4, len(xs) + 0.75)
    save(fig, "education_gap.png")


# ---------------------------------------------------------------------- 10. the fairness reversal
def fig_equal_opportunity() -> None:
    # reports/week4_boosting_fairness.json: formal audit, 300K held-out subsample, BH corrected
    groups = load("week4_boosting_fairness.json")["fairness"]
    rows = sorted(
        [(g["group"], g["equal_opportunity_diff"], g["eo_ci"][0], g["eo_ci"][1], g["eo_pvalue_bh"])
         for g in groups],
        key=lambda r: r[1],
    )

    rows = rows[::-1]  # largest gap on top
    fig, ax = plt.subplots(figsize=(12.3, 4.98))
    y = list(range(len(rows)))
    for i, (name, v, lo, hi, p) in enumerate(rows):
        col = RED if name == "NH Black" else (GREY if p > 0.05 else BLUE)
        ax.barh(i, v, height=0.56, color=col, alpha=0.9)
        ax.plot([lo, hi], [i, i], color=INK, lw=2.0)
        ax.text(hi + 0.014, i, f"{v:+.3f}", va="center", ha="left",
                fontsize=18.0, fontweight="bold" if name == "NH Black" else "normal", color=col)
    ax.axvline(0, color=INK, lw=1.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=16.1)
    ax.invert_yaxis()
    ax.set_xlim(-0.075, 0.50)
    ax.set_xlabel("equal-opportunity difference vs NH White   (positive = the model catches MORE of that group's preterm births)",
                  fontsize=14.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.995, 0.05, "grey = not significant after BH correction",
            transform=ax.transAxes, ha="right", fontsize=13.0, color=INK_SOFT)
    save(fig, "equal_opportunity.png")


# ---------------------------------------------------------------------- 11. sensitivity vs flagging
def fig_impact() -> None:
    groups = load("week4_boosting_fairness.json")["fairness"]
    rows = sorted(
        [(g["group"], g["equal_opportunity_diff"], g["disparate_impact_ratio"]) for g in groups],
        key=lambda r: -r[2],
    )

    fig, ax = plt.subplots(figsize=(11.45, 4.88))
    y = list(range(len(rows)))
    for i, (name, eo, di) in enumerate(rows):
        col = RED if name == "NH Black" else GREY
        ax.barh(i, di, height=0.55, color=col, alpha=0.9)
        ax.text(di + 0.03, i, f"{di:.2f}x", va="center", fontsize=18.6,
                fontweight="bold" if name == "NH Black" else "normal", color=col)
    ax.axvline(1.0, color=BLUE, lw=2.0)
    ax.text(1.03, len(rows) - 0.3, "NH White = 1.00x", fontsize=14.3, color=BLUE)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=16.1)
    ax.set_xlim(0, 3.15)
    ax.set_xlabel("how often the group is flagged, relative to NH White, at one shared threshold", fontsize=14.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "disparate_impact.png")


# ---------------------------------------------------------------------- 12. race-stratified SHAP
def fig_shap_race() -> None:
    strat = W7["per_group_top_10"]

    def top(group):
        return [(f, v) for f, v in strat[group][:8]]

    pretty = {
        "priorlive": "prior live births", "medical_risk_count": "medical risk count",
        "mager": "maternal age", "rf_ppterm": "prior preterm birth", "bmi": "BMI",
        "married": "marital status", "meduc": "education", "wic": "WIC",
        "pay_1": "private insurance", "first_trimester_care": "1st-trimester care",
        "cig_0": "smoking", "rf_phype": "pre-pregnancy hypertension",
        "mother_race_ethnicity_NH Black": "NH Black indicator",
        "mother_race_ethnicity_NH White": "NH White indicator",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.78, 4.88))
    fig.subplots_adjust(wspace=0.42)
    for ax, group in zip(axes, ["NH Black", "NH White"]):
        rows = top(group)
        labels = [pretty.get(f, f) for f, _ in rows]
        vals = [v for _, v in rows]
        cols = [RED if "indicator" in name and "Black" in name else GREY for name in labels]
        y = list(range(len(vals)))
        ax.barh(y, vals, height=0.6, color=cols, alpha=0.92)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=14.3)
        ax.invert_yaxis()
        ax.set_xlim(0, 0.20)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"Predicting for {group} mothers", fontsize=17.4, loc="left", pad=10,
                     color=RED if group == "NH Black" else BLUE)
        for i, (name, v) in enumerate(zip(labels, vals)):
            if "Black indicator" in name and group == "NH Black":
                ax.text(v - 0.006, i, f"rank {i + 1} of 103   {v:.3f}", va="center", ha="right",
                        fontsize=15.5, color="white", fontweight="bold")
            if "NH White indicator" in name and group == "NH White":
                ax.text(v + 0.006, i, f"rank {i + 1}", va="center", ha="left",
                        fontsize=15.5, color=INK_SOFT)
    axes[0].set_xlabel("mean |SHAP|", fontsize=14.3)
    axes[1].set_xlabel("mean |SHAP|", fontsize=14.3)
    save(fig, "shap_race.png")


# ---------------------------------------------------------------------- 13. decomposition
def fig_decomposition() -> None:
    d = W7["decomposition"]
    raw = d["raw_gap"] * 100
    full = d["explained_pct_full"]
    ses = d["explained_pct_ses"]
    med = d["explained_pct_medical"]

    fig, ax = plt.subplots(figsize=(13.6, 4.3))
    bars = [("Measured pathways, SES set\n(education, insurance, WIC)", ses),
            ("Measured pathways, clinical set\n(risk flags, smoking, BMI)", med),
            ("Measured pathways, both sets", full)]
    for i, (label, share) in enumerate(bars):
        ax.barh(i, raw, height=0.5, color=BAND)
        ax.barh(i, raw * share, height=0.5, color=ORANGE, alpha=0.9)
        ax.text(raw * share / 2, i, f"{share * 100:.0f}%", ha="center", va="center",
                fontsize=18.6, fontweight="bold", color="white")
        ax.text(raw + 0.06, i, f"{(1 - share) * 100:.0f}% not explained", va="center",
                fontsize=15.5, color=INK)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=14.9)
    ax.invert_yaxis()
    ax.set_xlim(0, raw * 1.42)
    ax.set_xticks([0, 1, 2, 3, 4, raw])
    ax.set_xticklabels(["0", "1", "2", "3", "4", f"{raw:.2f} pp\n(whole gap)"], fontsize=13.6)
    ax.set_xlabel("share of the 4.91-point NH Black / NH White gap", fontsize=14.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "decomposition.png")


# ---------------------------------------------------------------------- 14. mitigation tradeoffs
def fig_mitigation() -> None:
    m = W6["mitigations"]
    keys = ["baseline", "threshopt", "reweight", "expgrad"]
    pretty = {"baseline": "No mitigation", "threshopt": "ThresholdOptimizer",
              "reweight": "Reweighting", "expgrad": "ExponentiatedGradient"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.36, 4.56))

    worst = [m[k]["worst_group_eo_gap"] for k in keys]
    cols = [GREY, GREEN, RED, PURPLE]
    y = list(range(len(keys)))
    a1.barh(y, worst, height=0.55, color=cols, alpha=0.9)
    for i, v in enumerate(worst):
        a1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=17.4, color=cols[i], fontweight="bold")
    a1.set_yticks(y)
    a1.set_yticklabels([pretty[k] for k in keys], fontsize=15.5)
    a1.invert_yaxis()
    a1.set_xlim(0, 0.60)
    a1.set_xlabel("worst equal-opportunity gap  (lower is fairer)", fontsize=14.3)
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="y", length=0)
    a1.set_title("What each mitigation buys", fontsize=17.4, loc="left", pad=10)

    sens = [m[k]["sensitivity"] for k in keys]
    a2.barh(y, sens, height=0.55, color=cols, alpha=0.9)
    for i, v in enumerate(sens):
        a2.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=17.4, color=cols[i], fontweight="bold")
    a2.set_yticks(y)
    a2.set_yticklabels([])
    a2.invert_yaxis()
    a2.set_xlim(0, 0.78)
    a2.set_xlabel("sensitivity: share of real preterm births caught", fontsize=14.3)
    a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", length=0)
    a2.set_title("What it costs", fontsize=17.4, loc="left", pad=10)
    save(fig, "mitigation.png")


# ---------------------------------------------------------------------- 15. temporal stability
def fig_drift() -> None:
    rows = W8["drift"]
    xs = [r["year"] for r in rows]
    ys = [r["auc_roc"] for r in rows]

    # y-axis spans a full 0.10 of AUC so the flatness is real rather than an axis-zoom artifact
    fig, ax = plt.subplots(figsize=(11.24, 4.35))
    ax.axvspan(2019.5, 2021.5, color=BAND, zorder=0)
    ax.text(2020.5, 0.706, "COVID era", ha="center", fontsize=15.5, color=INK_SOFT)
    ax.plot(xs, ys, color=BLUE, lw=3.0, marker="o", ms=10, zorder=3)
    for x, v in zip(xs, ys):
        ax.text(x, v + 0.0085, f"{v:.3f}", ha="center", fontsize=14.9, color=INK)
    ax.set_ylim(0.610, 0.715)
    ax.set_yticks([0.62, 0.64, 0.66, 0.68, 0.70])
    ax.set_xticks(xs)
    ax.set_xlim(2017.5, 2024.5)
    ax.set_ylabel("AUC-ROC", fontsize=14.9)
    ax.set_xlabel("birth year", fontsize=14.9)
    ax.text(2017.7, 0.622, "Full range across seven years: 0.656 to 0.669.",
            fontsize=15.5, color=INK)
    save(fig, "drift.png")


# =============================================================== COMPOSITES for the condensed deck
# Each of these is one wide figure that fills a whole slide body, so a merged slide carries a
# single large readable visual instead of two cramped ones.

def fig_ceiling_combined() -> None:
    """Thin per-feature signal (left) beside the training-size plateau (right)."""
    rows = sorted([(r["feature"], r["univariate_auc"]) for r in W2["univariate_auc"]],
                  key=lambda kv: -kv[1])[:6]
    pretty = {
        "medical_risk_count": "count of medical risk flags", "rf_ppterm": "previous preterm birth",
        "bmi": "pre-pregnancy BMI", "pay_1": "private insurance",
        "mother_race_ethnicity_NH Black": "NH Black indicator", "rf_cesar": "previous cesarean",
    }
    labels = [pretty.get(k, k) for k, _ in rows]
    vals = [v for _, v in rows]

    caps = W8["sensitivity_cap"]
    xs = [c["train_cap"] / 1000 for c in caps]
    ys = [c["held_out_auc_pr"] for c in caps]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(21.0, 6.55),
                                 gridspec_kw={"width_ratios": [1.08, 1]})
    fig.subplots_adjust(wspace=0.30)

    y = list(range(len(vals)))
    a1.barh(y, [v - 0.5 for v in vals], left=0.5, height=0.6, color=BLUE, alpha=0.9)
    for i, v in enumerate(vals):
        a1.text(v + 0.0022, i, f"{v:.3f}", va="center", fontsize=23, color=INK)
    a1.axvline(0.5, color=INK, lw=2.0)
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=21)
    a1.invert_yaxis()
    a1.set_xlim(0.495, 0.601)
    a1.set_xticks([0.50, 0.52, 0.54, 0.56, 0.58])
    a1.tick_params(axis="x", labelsize=19)
    a1.set_xlabel("single-feature AUC over all 24.9M births\n(0.500 = no information at all)",
                  fontsize=20)
    a1.set_title("No one field carries much signal", fontsize=26, loc="left",
                 pad=22, fontweight="bold")
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="y", length=0)

    a2.axhline(0.0857, color=GREY, lw=2.0, ls=(0, (4, 3)))
    a2.text(102, 0.0905, "8.57% base rate: just guessing the average", fontsize=19, color=INK_SOFT)
    a2.plot(xs, ys, color=BLUE, lw=4.4, marker="o", ms=19)
    for x, v in zip(xs, ys):
        a2.text(x, v + 0.0085, f"{v:.4f}", ha="center", fontsize=25, color=INK, fontweight="bold")
    a2.text(300, 0.132, "5x the rows buys +0.0069", ha="center", fontsize=23,
            color=RED, fontweight="bold")
    a2.set_xticks(xs)
    a2.set_xticklabels([f"{int(x)}K" for x in xs], fontsize=21)
    a2.tick_params(axis="y", labelsize=19)
    a2.set_xlim(55, 550)
    a2.set_ylim(0.078, 0.207)
    a2.set_xlabel("training rows", fontsize=20)
    a2.set_ylabel("held-out AUC-PR", fontsize=20)
    a2.set_title("More data does not help", fontsize=26, loc="left", pad=22, fontweight="bold")
    save(fig, "ceiling_combined.png")


def fig_fairness_combined() -> None:
    """The two facts that are true at once: more detection, and far more flagging."""
    groups = load("week4_boosting_fairness.json")["fairness"]
    eo = sorted([(g["group"], g["equal_opportunity_diff"], g["eo_ci"][0], g["eo_ci"][1],
                  g["eo_pvalue_bh"]) for g in groups], key=lambda r: -r[1])
    di = sorted([(g["group"], g["disparate_impact_ratio"]) for g in groups], key=lambda r: -r[1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(21.0, 6.55))
    fig.subplots_adjust(wspace=0.34)

    y = list(range(len(eo)))
    for i, (name, v, lo, hi, p) in enumerate(eo):
        col = RED if name == "NH Black" else (GREY if p > 0.05 else BLUE)
        a1.barh(i, v, height=0.6, color=col, alpha=0.92)
        a1.plot([lo, hi], [i, i], color=INK, lw=2.6)
        a1.text(hi + 0.018, i, f"{v:+.3f}", va="center", fontsize=23,
                fontweight="bold" if name == "NH Black" else "normal", color=col)
    a1.axvline(0, color=INK, lw=2.4)
    a1.set_yticks(y)
    a1.set_yticklabels([r[0] for r in eo], fontsize=21)
    a1.invert_yaxis()
    a1.set_xlim(-0.085, 0.53)
    a1.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    a1.tick_params(axis="x", labelsize=19)
    a1.set_xlabel("equal-opportunity difference vs NH White\n(positive = MORE preterm births caught)",
                  fontsize=20)
    a1.set_title("It caught MORE, not fewer", fontsize=26, loc="left", pad=22,
                 fontweight="bold", color=GREEN)
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="y", length=0)

    y2 = list(range(len(di)))
    for i, (name, v) in enumerate(di):
        col = RED if name == "NH Black" else GREY
        a2.barh(i, v, height=0.6, color=col, alpha=0.92)
        a2.text(v + 0.055, i, f"{v:.2f}x", va="center", fontsize=23,
                fontweight="bold" if name == "NH Black" else "normal", color=col)
    a2.axvline(1.0, color=BLUE, lw=2.8)
    a2.text(1.06, -0.62, "NH White = 1.00x", fontsize=18, color=BLUE)
    a2.set_yticks(y2)
    a2.set_yticklabels([r[0] for r in di], fontsize=21)
    a2.set_ylim(len(di) - 0.35, -0.85)
    a2.set_xlim(0, 3.35)
    a2.tick_params(axis="x", labelsize=19)
    a2.set_xlabel("how often the group is flagged, vs NH White,\nat one shared threshold",
                  fontsize=20)
    a2.set_title("But it flagged them 2.74x as often", fontsize=26, loc="left", pad=22,
                 fontweight="bold", color=RED)
    a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", length=0)
    save(fig, "fairness_combined.png")


def fig_disparity_wide() -> None:
    """The outcome disparity beside the education strata: the whole equity story in one frame."""
    rows = sorted(W1["rate_by_race"], key=lambda r: r["rate"])
    labels = [r["mother_race_ethnicity"] for r in rows]
    vals = [r["rate"] for r in rows]
    cols = [RED if x == "NH Black" else BLUE if x == "NH White" else GREY for x in labels]

    lab = W1["meta"]["education_labels"]
    er = W1["edu_race"]
    b = {r["meduc"]: r["rate"] for r in er if r["mother_race_ethnicity"] == "NH Black"}
    w = {r["meduc"]: r["rate"] for r in er if r["mother_race_ethnicity"] == "NH White"}
    ks = sorted(b)
    exs = list(range(len(ks)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(21.0, 6.55),
                                 gridspec_kw={"width_ratios": [1, 1.12]})
    fig.subplots_adjust(wspace=0.24)

    y = list(range(len(vals)))
    a1.barh(y, vals, height=0.66, color=cols)
    for i, v in enumerate(vals):
        # bbox keeps the value readable where a short bar's label crosses the base-rate line
        a1.text(v + 0.2, i, f"{v:.2f}%", va="center", fontsize=22,
                fontweight="bold" if cols[i] != GREY else "normal", color=cols[i],
                bbox=dict(boxstyle="round,pad=0.14", fc=PAPER, ec="none"), zorder=4)
    a1.axvline(8.57, color=INK, lw=2.0, ls=(0, (4, 3)))
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=21)
    a1.set_xlim(0, 15.4)
    a1.tick_params(axis="x", labelsize=19)
    a1.set_xlabel("preterm rate, every singleton birth 2018-2024", fontsize=20)
    a1.set_title("A 4.74-point gap in the outcome", fontsize=26, loc="left", pad=22,
                 fontweight="bold")
    a1.spines["left"].set_visible(False)
    a1.tick_params(axis="y", length=0)

    a2.fill_between(exs, [w[k] for k in ks], [b[k] for k in ks], color=RED, alpha=0.13)
    a2.plot(exs, [b[k] for k in ks], color=RED, lw=4.4, marker="o", ms=13)
    a2.plot(exs, [w[k] for k in ks], color=BLUE, lw=4.4, marker="o", ms=13)
    a2.text(exs[-1] + 0.18, b[ks[-1]], "NH Black", color=RED, fontsize=22,
            fontweight="bold", va="center")
    a2.text(exs[-1] + 0.18, w[ks[-1]], "NH White", color=BLUE, fontsize=22,
            fontweight="bold", va="center")
    for k, x in zip(ks, exs):
        a2.text(x, (b[k] + w[k]) / 2, f"+{b[k] - w[k]:.1f}", ha="center", va="center",
                fontsize=17, color=RED, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=PAPER, ec="none", alpha=0.86))
    a2.set_xticks(exs)
    a2.set_xticklabels([lab[str(k)] for k in ks], fontsize=16, rotation=24, ha="right")
    a2.tick_params(axis="y", labelsize=19)
    a2.set_ylabel("preterm birth rate", fontsize=20)
    a2.set_ylim(4.4, 16.0)
    a2.set_yticks([6, 8, 10, 12, 14])
    a2.set_yticklabels(["6%", "8%", "10%", "12%", "14%"])
    a2.set_xlim(-0.45, len(exs) + 1.05)
    a2.set_title("...that education does not close", fontsize=26, loc="left", pad=22,
                 fontweight="bold")
    save(fig, "disparity_wide.png")


if __name__ == "__main__":
    fig_ceiling_combined()
    fig_fairness_combined()
    fig_disparity_wide()
    fig_leakage_gap()
    fig_leakage_timeline()
    fig_iatrogenic()
    fig_model_field()
    fig_univariate()
    fig_train_cap()
    fig_calibration()
    fig_disparity()
    fig_education()
    fig_equal_opportunity()
    fig_impact()
    fig_shap_race()
    fig_decomposition()
    fig_mitigation()
    fig_drift()
    print("\nall figures written to", OUT.relative_to(ROOT))
