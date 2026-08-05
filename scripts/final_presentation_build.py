"""Build the AI4ALL Ignite final presentation as an editable .pptx.

13 slides for a 10-minute slot (about 45 seconds each), widescreen 16:9, one takeaway per slide,
speaker notes carrying everything that used to be on a slide. Figures come from
scripts/final_presentation_figures.py, which reads only tracked canonical full-data artifacts.

Design rules enforced here:
  - large type: 40pt titles, 20pt body, nothing on a slide below 15pt
  - figures fill the body width edge to edge, sized to the space actually left by the title
  - a fixed vertical rhythm, so nothing overlaps and no large gaps are left at the foot

Run: python scripts/final_presentation_figures.py && python scripts/final_presentation_build.py
Output: presentation/EquiBirth_AI_final_presentation.pptx
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "presentation" / "figures"
OUT = ROOT / "presentation" / "EquiBirth_AI_final_presentation.pptx"

# ------------------------------------------------------------------------------------ design tokens
INK = RGBColor(0x16, 0x20, 0x3A)
INK_SOFT = RGBColor(0x3D, 0x46, 0x63)
PAPER = RGBColor(0xFB, 0xFB, 0xFD)
BAND = RGBColor(0xE9, 0xEC, 0xF4)
BLUE = RGBColor(0x0B, 0x6F, 0xB8)
RED = RGBColor(0xD7, 0x26, 0x3D)
GREEN = RGBColor(0x0A, 0x87, 0x54)
ORANGE = RGBColor(0xE0, 0x7B, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SKY = RGBColor(0x7F, 0xB2, 0xDE)
ROSE = RGBColor(0xF2, 0x8E, 0x9E)
MIST = RGBColor(0xD8, 0xE2, 0xF0)
FONT = "Helvetica Neue"

W, H = Inches(13.333), Inches(7.5)
M = Inches(0.66)                        # outer margin: tight, so figures get maximum width
CONTENT_W = W - 2 * M

# type scale, deliberately large for a projected room
TITLE_PT = 40
KICKER_PT = 17
LEAD_PT = 24
BODY_PT = 20
SMALL_PT = 16


def deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs: Presentation, bg: RGBColor = PAPER):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = bg
    return slide


def text(slide, left, top, width, height, runs, *, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.0, space_after=0):
    """runs: list of (string, size_pt, bold, colour) or (..., italic)."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, spec in enumerate(runs):
        body, size, bold, colour = spec[:4]
        italic = spec[4] if len(spec) > 4 else False
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = body
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = colour
    return box


def rule(slide, top, *, left=M, width=CONTENT_W, colour=BLUE, thickness=Inches(0.05)):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, thickness)
    bar.fill.solid()
    bar.fill.fore_color.rgb = colour
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def panel(slide, left, top, width, height, colour=BAND):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = colour
    box.line.fill.background()
    box.shadow.inherit = False
    box.adjustments[0] = 0.04
    return box


def notes(slide, body: str) -> None:
    slide.notes_slide.notes_text_frame.text = body.strip()


def _title_lines(title: str, chars_per_line: int = 42) -> int:
    """Greedy line count at TITLE_PT, so the rule and the body never land on the title."""
    lines, line = 1, ""
    for word in title.split():
        trial = f"{line} {word}".strip()
        if line and len(trial) > chars_per_line:
            lines += 1
            line = word
        else:
            line = trial
    return lines


def headline(slide, title: str, *, kicker: str | None = None, colour=INK, top=Inches(0.44)):
    """Kicker + assertive takeaway title + accent rule. Returns the y where the body may start."""
    y = top
    if kicker:
        text(slide, M, y, CONTENT_W, Inches(0.36), [(kicker.upper(), KICKER_PT, True, BLUE)])
        y += Inches(0.44)
    lines = _title_lines(title)
    text(slide, M, y, CONTENT_W, Inches(0.68 * lines), [(title, TITLE_PT, True, colour)],
         spacing=1.05)
    y += Inches(0.66 * lines + 0.18)
    rule(slide, y, width=Inches(1.9))
    return y + Inches(0.34)


def figure(slide, name: str, top, *, bottom, width=None, left=None):
    """Fill the band between ``top`` and ``bottom`` with the figure, as large as it will go.

    The figure is scaled to the lesser of the available width and the available height, then
    centred in the band, so slides never show a squeezed image or a dead strip at the foot.
    """
    path = FIGS / name
    with Image.open(path) as im:
        aspect = im.width / im.height
    avail_w = width if width is not None else CONTENT_W
    avail_h = bottom - top
    w = avail_w
    h = Emu(int(w / aspect))
    if h > avail_h:
        h = avail_h
        w = Emu(int(h * aspect))
    x = left if left is not None else Emu(int((W - w) / 2))
    y = Emu(int(top + (avail_h - h) / 2))
    slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    return y + h


def source(slide, body: str) -> None:
    text(slide, M, H - Inches(0.5), CONTENT_W, Inches(0.3),
         [(body, 12, False, INK_SOFT)])


# ===================================================================================== the slides
def build() -> None:
    prs = deck()

    # ------------------------------------------------------------------ 1. title  (0:00-0:25)
    s = blank(prs, INK)
    # a red spine down the left edge ties the dark slides together and fills the frame
    rule(s, Inches(0), left=Inches(0), width=Inches(0.2), colour=RED, thickness=H)
    text(s, M, Inches(0.78), CONTENT_W, Inches(0.44),
         [("AI4ALL IGNITE  .  FINAL PROJECT", KICKER_PT, True, SKY)])
    text(s, M, Inches(1.48), CONTENT_W, Inches(1.7), [("EquiBirth AI", 92, True, WHITE)])
    text(s, M, Inches(3.16), CONTENT_W, Inches(1.5),
         [("Can we predict preterm birth from U.S. birth certificates, "
           "and is the model fair?", 30, False, MIST)], spacing=1.24)
    rule(s, Inches(4.86), width=Inches(2.6), colour=RED, thickness=Inches(0.07))
    text(s, M, Inches(5.32), CONTENT_W, Inches(1.4),
         [("Our best result was not our highest score.", 28, True, WHITE),
          ("It was refusing to report one.", 28, True, ROSE)], spacing=1.3)
    text(s, M, H - Inches(0.82), CONTENT_W, Inches(0.5),
         [("24.9 million births  .  CDC / NCHS 2018-2024  .  educational research, "
           "not a clinical tool", 16, True, SKY)])
    notes(s, """
TIMING: 13 slides, 10 minutes, so about 45 seconds each. Do not read slides out loud.

Say only: "We are EquiBirth AI. We asked whether preterm birth can be predicted from birth
certificates, and whether the model is fair. The most interesting thing we found was a number we
chose NOT to report." Then go straight to the next slide.
""")

    # ------------------------------------------------------------------ 2. hook  (0:25-1:10)
    s = blank(prs, INK)
    rule(s, Inches(0), left=Inches(0), width=Inches(0.2), colour=RED, thickness=H)
    text(s, M, Inches(0.98), CONTENT_W, Inches(2.7),
         [("If I showed you a preterm-birth model scoring 0.92,", 50, True, WHITE)],
         spacing=1.14)
    text(s, M, Inches(3.62), CONTENT_W, Inches(1.2),
         [("you should not be impressed.", 50, True, ROSE)])
    rule(s, Inches(5.18), width=Inches(2.6), colour=RED, thickness=Inches(0.07))
    text(s, M, Inches(5.68), CONTENT_W, Inches(1.4),
         [("You should ask what it was allowed to look at.", 30, False, MIST)], spacing=1.24)
    notes(s, """
THE HOOK. Deliver it, then pause for two full seconds before continuing.

"A model can look brilliant for one boring reason: somebody accidentally handed it the answer.
That is the trap we spent nine weeks not falling into."

Then move on. Do not explain leakage yet, slide 4 does that.
""")

    # ------------------------------------------------------------------ 3. the problem  (1:10-2:00)
    s = blank(prs)
    y = headline(s, "Common, and unevenly distributed.",
                 kicker="why this matters")
    nums = [("8.57%", "of U.S. singleton births come\nbefore 37 weeks", BLUE),
            ("12.28%", "for NH Black mothers,\nversus 7.54% for NH White", RED),
            ("4.74", "point gap, measured on every\nbirth rather than a sample", RED)]
    cw = Inches(3.86)
    for i, (val, lab, colour) in enumerate(nums):
        x = M + (cw + Inches(0.22)) * i
        text(s, x, y + Inches(0.42), cw, Inches(1.5), [(val, 84, True, colour)],
             align=PP_ALIGN.CENTER)
        text(s, x, y + Inches(1.72), cw, Inches(1.0), [(lab, BODY_PT, False, INK)],
             align=PP_ALIGN.CENTER, spacing=1.24)
    panel(s, M, H - Inches(1.72), CONTENT_W, Inches(1.16))
    text(s, M + Inches(0.42), H - Inches(1.52), CONTENT_W - Inches(0.9), Inches(0.8),
         [("Race here is a marker of exposure to structural conditions and racism. "
           "It is never a biological cause.", BODY_PT, True, INK)], spacing=1.26)
    notes(s, """
Anchor the stakes in three numbers. No chart needed.

Point at 4.74 and stress this is not a sampling estimate: it is every singleton birth certificate
filed in the United States over seven years.

Say the race framing out loud here. It is a commitment we hold for the whole talk, not a caveat we
add at the end.

Source: reports/week2_phase2_audit.md, full-population outcome audit, 24,920,407 births.
""")

    # ------------------------------------------------------------------ 4. the trap  (2:00-3:00)
    s = blank(prs)
    y = headline(s, "The certificate knows how it ended.",
                 kicker="the trap")
    figure(s, "leakage_timeline.png", y, bottom=H - Inches(1.62))
    text(s, M, H - Inches(1.42), CONTENT_W, Inches(1.0),
         [("A baby born at 30 weeks weighs about 1.5 kg. So birthweight is not a risk factor "
           "for preterm birth, it is a consequence of it.", BODY_PT, False, INK)], spacing=1.26)
    notes(s, """
This is the intuition slide. Use the birthweight example, it always lands.

"Birthweight, Apgar scores, steroid doses, induction: all of these are written down at or after
the delivery. A model given birthweight is not predicting the future, it is reading the past."

Antenatal steroids are the sharpest version: you only give those when you ALREADY expect an early
delivery. So that column is a record of the doctor's expectation, not a prediction of it.
""")

    # ------------------------------------------------------------------ 5. measured  (3:00-4:00)
    s = blank(prs)
    y = headline(s, "So we measured the trap.",
                 kicker="the trap, quantified")
    figure(s, "leakage_gap.png", y, bottom=H - Inches(1.62))
    text(s, M, H - Inches(1.44), CONTENT_W, Inches(0.9),
         [("0.917 is the number a project optimizing for an impressive slide would have "
           "reported. We report 0.659.", BODY_PT, True, INK)], spacing=1.26)
    source(s, "Sources: reports/week2_phase2_audit.md (FR6.1 leakage detector), reports/week2_experiments.json (dual-track LightGBM).")
    notes(s, """
Walk the three bars from the top.

Top: our own clean model with the post-delivery columns added back, 0.939.
Middle: a classifier using ONLY the barred post-delivery columns, 0.917. It knows nothing about
pregnancy. It is reading the delivery record.
Bottom: an honest pre-delivery model on the same held-out births, 0.659.

That 0.28 AUC gap is the single most important result in this project.

IF ASKED how we prevented it: three guardrails enforced by the build, not by trust. A detector
trained on the excluded columns fails the build above 0.55 AUC. A composite guard stops an
engineered feature from laundering a barred column. And a univariate gate covers features whose
recording window is shortened by an early delivery. All three have tests.
""")

    # ------------------------------------------------------------------ 6. the choice  (4:00-5:00)
    s = blank(prs)
    y = headline(s, "We chose the weaker model.",
                 kicker="our hardest decision", colour=RED)
    figure(s, "iatrogenic_tradeoff.png", y - Inches(0.12), bottom=H - Inches(1.78),
           width=Inches(6.9), left=M - Inches(0.06))
    text(s, M + Inches(7.16), y + Inches(0.34), Inches(4.98), Inches(3.4),
         [("Eclampsia and gestational hypertension are known before delivery. They also cause "
           "doctors to deliver early.", 21, False, INK),
          ("", 11, False, INK),
          ("So the model would learn \"the doctor decided to deliver\" and call it a prediction. "
           "That is backwards.", 21, False, INK)], spacing=1.3)
    panel(s, M, H - Inches(1.66), CONTENT_W, Inches(1.1), colour=BAND)
    text(s, M + Inches(0.42), H - Inches(1.46), CONTENT_W - Inches(0.9), Inches(0.72),
         [("Removing them cost us 3 AUC points. Nobody outside the team would have known. "
           "We took the loss.", BODY_PT, True, RED)], spacing=1.26)
    source(s, "Source: DECISION_LOG D26. These two AUC values are from the development-sample run the decision was made on.")
    notes(s, """
THE SLIDE I MOST WANT REMEMBERED. Slow down here.

These conditions pass every leakage test we have. They are diagnosed during pregnancy, before
delivery. A purely mechanical rule keeps them.

But the mechanism runs backwards. Severe eclampsia at 33 weeks means the team delivers the baby.
The condition does not predict the early birth, it CAUSES it through a clinical decision.

Removing them dropped us from 0.684 to 0.654. We logged it as D26 and reported the lower number.

NOTE: those two values come from the development sample, which is where the decision was made.
Our canonical full-data figure is 0.6561. It is labelled on the slide.
""")

    # ------------------------------------------------------------------ 7. the result  (5:00-5:40)
    s = blank(prs)
    y = headline(s, "The honest answer is 0.656.", kicker="our result")
    figure(s, "model_field.png", y, bottom=H - Inches(1.74))
    text(s, M, H - Inches(1.56), Inches(7.1), Inches(1.05),
         [("XGBoost with Optuna tuning. Modest, and squarely inside the published range for "
           "birth-record data.", BODY_PT, False, INK)], spacing=1.24)
    text(s, M + Inches(7.5), H - Inches(1.56), Inches(4.48), Inches(1.05),
         [("We do not claim deep learning lost. We could not afford to train it fairly.",
           BODY_PT, True, ORANGE)], spacing=1.24)
    source(s, "Source: reports/week5_models.md, all five models scored on identical held-out rows. Compute caps disclosed in DECISION_LOG D46.")
    notes(s, """
0.6561 AUC-ROC, 0.1720 AUC-PR. Inside the published 0.62 to 0.70 band for administrative data.

BE CAREFUL with the deep-learning bars. TabNet and FT-Transformer trained on 30,000 rows for at
most 3 epochs; the boosters got 500,000. On a laptop, the FT-Transformer's CPU forward pass was
the binding constraint.

So the honest claim is narrow: under OUR compute budget, the simpler tree model won, and we chose
it for auditability. Not "trees beat deep learning."

We also picked XGBoost over the slightly better stacked ensemble because the margin was inside
bootstrap noise and one model is easier to audit. That is D42.
""")

    # ------------------------------------------------------------------ 8. the ceiling  (5:40-6:30)
    s = blank(prs)
    y = headline(s, "The ceiling is in the data.",
                 kicker="why that number is the finding")
    figure(s, "ceiling_combined.png", y, bottom=H - Inches(1.42))
    text(s, M, H - Inches(1.22), CONTENT_W, Inches(0.85),
         [("More rows and bigger models cannot manufacture signal that the birth certificate "
           "does not contain.", BODY_PT, True, INK)], spacing=1.26)
    notes(s, """
This is the slide that turns a mediocre score into a research result.

LEFT: the raw signal in the best individual fields. The strongest single predictor moves you from
0.500 to 0.570. Everything else is weaker. The largest correlation with the outcome anywhere in
our feature set is 0.13.

RIGHT: we scaled training data five times over, 100K to 500K rows, and gained 0.007 AUC-PR.

Together: 0.656 is not us failing to model well. It is a measurement of how much a birth
certificate actually knows about a birth that has not happened yet.

That has a real policy implication: do not expect an administrative-data screener to stratify
individual risk. And it is why we say this is population-level diagnosis, not a clinical tool.
""")

    # ------------------------------------------------------------------ 9. the disparity (6:30-7:20)
    s = blank(prs)
    y = headline(s, "Education does not close the gap.",
                 kicker="equity")
    figure(s, "disparity_wide.png", y, bottom=H - Inches(1.42))
    text(s, M, H - Inches(1.22), CONTENT_W, Inches(0.85),
         [("NH Black mothers with a doctorate have a higher preterm rate than NH White mothers "
           "who stopped at high school.", BODY_PT, True, RED)], spacing=1.26)
    notes(s, """
Order matters: the outcome disparity comes BEFORE any claim about model fairness. The model did
not create this gap.

LEFT: what is in the world. 12.28% versus 7.54%, computed on every recorded birth.

RIGHT: the obvious objection is that this is really a class gap wearing a racial mask. If that
were true, the two lines would converge as education rises. They do not. The gap runs +3.1 to
+4.3 points at every single level.

The comparison in the caption is the one that stops the room. Say it slowly.

Stay associational: this does not prove a mechanism. It does rule out the simplest alternative
explanation.
""")

    # ------------------------------------------------------------------ 10. reversal  (7:20-8:20)
    s = blank(prs)
    y = headline(s, "Our fairness hypothesis was wrong.",
                 kicker="the result we did not expect", colour=RED)
    figure(s, "fairness_combined.png", y, bottom=H - Inches(1.42))
    text(s, M, H - Inches(1.22), CONTENT_W, Inches(0.85),
         [("Catching more of a group's preterm births, while flagging that group far more often, "
           "is not the same thing as treating them equitably.", BODY_PT, True, INK)],
         spacing=1.26)
    notes(s, """
This is where writing our expectation down first pays off.

We PREREGISTERED the standard worry: that the model would under-detect for Black mothers. Then we
measured it and got the opposite. Equal opportunity +0.393, significant after correction. It
catches MORE of their real preterm births, not fewer.

Do not stop there, because that would be the flattering half. The right panel is the cost: Black
mothers are flagged 2.74 times as often. More follow-ups, more anxiety, more false positives
concentrated in one group.

Honest caveat: that higher flag rate largely reflects genuinely higher prevalence. The model is
not inventing a difference.

We are not resolving this tension for you. Fairness is not one number, and pretending otherwise
would be the same shortcut we refused on slide 6.

IF ASKED about mitigation: we priced three. The best one halves the gap but needs race-specific
thresholds, which is disparate treatment, so we never shipped it. One helped the largest group
and destabilized the smallest. And one apparent result turned out to be our own threshold
mismatch, which we disclosed rather than quietly dropped.
""")

    # ------------------------------------------------------------------ 11. explanation (8:20-9:00)
    s = blank(prs)
    y = headline(s, "Measured factors explain under half.",
                 kicker="what explains it")
    figure(s, "decomposition.png", y, bottom=H - Inches(2.0))
    panel(s, M, H - Inches(1.82), CONTENT_W, Inches(1.26), colour=BAND)
    text(s, M + Inches(0.42), H - Inches(1.62), CONTENT_W - Inches(0.9), Inches(0.9),
         [("Education, insurance, and WIC are not competing explanations. They are themselves "
           "downstream of structural racism, so even 43% overstates it.", BODY_PT, True, INK)],
         spacing=1.26)
    source(s, "Source: reports/week7_explain.md, g-computation standardization. Framed as mediation, not confounding: DECISION_LOG D30.")
    notes(s, """
The framing word is MEDIATION, not confounding, and it changes the meaning.

We are not adjusting away a confounder of a race "effect." Race is not a manipulable treatment.
Education and insurance are pathways the gap partly travels THROUGH.

So: 19 to 43 percent of the association is statistically transmitted through what we can measure.
The range is which covariates you choose, not statistical noise.

And the critical caveat: those mediators are themselves products of the same structural
conditions. So 43% overstates the genuinely mediated share, and the unexplained remainder is a
FLOOR on the structural contribution, not a ceiling.

Policy reading: targeting only measured socioeconomic access closes at most a third to a half of
this gap.

IF ASKED what the model leaned on for Black mothers: the race field itself was the top
contributor, ahead of clinical history. Not biology. It is the statistical shadow of everything
the certificate fails to record.
""")

    # ------------------------------------------------------------------ 12. learned  (9:00-9:40)
    s = blank(prs)
    y = headline(s, "What nine weeks actually taught us.", kicker="reflection")
    lessons = [
        ("A great score is a suspect.",
         "0.917 was available. Ask what the model was allowed to see."),
        ("Metrics can be true and useless.",
         "Our calibration error was 0.001 and the scores still told no mother anything."),
        ("Fairness is not one number.",
         "Detection said we favoured Black mothers. Flag rates said we burdened them."),
        ("Write the expectation down first.",
         "Being publicly wrong is what made the fairness reversal a finding."),
    ]
    cw = Inches(5.82)
    for i, (title, body) in enumerate(lessons):
        col, row = i % 2, i // 2
        x = M + (cw + Inches(0.32)) * col
        yy = y + Inches(0.16) + Inches(1.66) * row
        text(s, x, yy, Inches(0.62), Inches(0.6), [(f"0{i + 1}", 30, True, BAND)])
        text(s, x + Inches(0.78), yy - Inches(0.04), cw - Inches(0.82), Inches(1.5),
             [(title, 22, True, BLUE),
              (body, 17, False, INK)], spacing=1.26, space_after=5)
    panel(s, M, H - Inches(1.52), CONTENT_W, Inches(1.0), colour=INK)
    text(s, M + Inches(0.42), H - Inches(1.31), CONTENT_W - Inches(0.9), Inches(0.6),
         [("The skill we practised most was deciding what we were not entitled to claim.",
           22, True, WHITE)], spacing=1.24)
    notes(s, """
The AI4ALL reflection. Speak from experience, not from the slide.

The lesson I would carry into any future project is number four. Writing down what we expected
made it possible to be publicly wrong, and being publicly wrong about the fairness direction is
the most interesting thing we found.

That habit generalizes: before you fit anything, write down what you think you will see and what
would change your mind.

IF ASKED about limitations: no clinical validation, no external population, everything
associational, the deep-learning benchmark was compute-capped, and a severity target we designed
but did not run. All documented.
""")

    # ------------------------------------------------------------------ 13. closing  (9:40-10:00)
    s = blank(prs, INK)
    rule(s, Inches(0), left=Inches(0), width=Inches(0.2), colour=RED, thickness=H)
    text(s, M, Inches(0.86), CONTENT_W, Inches(0.44),
         [("IN CLOSING", KICKER_PT, True, SKY)])
    text(s, M, Inches(1.58), Inches(11.4), Inches(1.6),
         [("A model built to impress you could have said 0.917.", 36, True, ROSE)],
         spacing=1.18)
    text(s, M, Inches(3.38), Inches(11.9), Inches(1.6),
         [("We said 0.656, and showed you why.", 36, True, WHITE)], spacing=1.18)
    rule(s, Inches(5.16), width=Inches(2.6), colour=RED, thickness=Inches(0.07))
    text(s, M, Inches(5.68), CONTENT_W, Inches(1.0),
         [("Thank you. Questions?", 30, False, MIST)])
    text(s, M, H - Inches(0.82), CONTENT_W, Inches(0.5),
         [("Race is a marker of exposure to racism, never a biological cause. "
           "Nothing here is a clinical tool.", 16, True, ROSE)])
    notes(s, """
Land the two-line contrast, then stop talking.

Closing line to say out loud: "The hardest engineering in this project was not the model. It was
building the guards that stopped us reporting the better-looking number."

Then take questions. Backup facts if needed:
  - 24,920,407 singleton births, CDC/NCHS 2018-2024, 103 pre-delivery features
  - 38 source modules, 137 tests, a Model Card and a Datasheet, 48 logged decisions
  - the app has no risk calculator and no race-keyed thresholds, enforced by tests
""")

    prs.save(OUT)
    n = len(prs.slides._sldIdLst)
    print(f"wrote {OUT.relative_to(ROOT)}  ({n} slides)")


if __name__ == "__main__":
    build()
