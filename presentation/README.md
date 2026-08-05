# Final presentation

`EquiBirth_AI_final_presentation.pptx` is the AI4ALL Ignite final project deck: 13 widescreen
slides sized for a 10-minute slot, with speaker notes on every slide. It is fully editable in
PowerPoint, Google Slides, Keynote, or Canva (import the `.pptx`).

`SPEAKER_SCRIPT.md` is the word-for-word script to read and rehearse from: what to say on each
slide, in spoken English, sized to land at about 9:40 at a normal speaking pace, with delivery
notes, a cut list for a slow day, and prepared answers to the likely questions.

**Nothing in the deck is hand-typed.** Every figure and every number is generated from the tracked
canonical artifacts in `reports/`, and a verifier asserts each slide claim against them.

## Narrative

13 slides for a 10-minute slot. Time is weighted by narrative load rather than spread evenly: the
leakage arc (slides 4 to 6) and the fairness reversal (slide 10) get about a minute each, the
bookends about twenty seconds. The arc is a single story: a suspiciously good number, why it was
fake, what the honest number is, what it revealed about inequity, and what we got wrong.

| # | Slide | Beat |
|---|---|---|
| 1 | EquiBirth AI | Our best result was not our highest score |
| 2 | You should not be impressed | The hook: ask what the model could see |
| 3 | Common, and unevenly distributed | 8.57%, 12.28% vs 7.54%, a 4.74 pp gap |
| 4 | The certificate knows how it ended | Leakage, explained through birthweight |
| 5 | So we measured the trap | 0.939 / 0.917 / 0.659 |
| 6 | We chose the weaker model | Removing reverse-causal conditions cost 3 AUC points |
| 7 | The honest answer is 0.656 | The model field, with the compute caveat |
| 8 | The ceiling is in the data | Thin per-field signal, and a training-size plateau |
| 9 | Education does not close the gap | The outcome disparity, and the education strata |
| 10 | Our fairness hypothesis was wrong | More detection AND 2.74x more flagging |
| 11 | Measured factors explain under half | 19-43%, framed as mediation |
| 12 | What nine weeks taught us | Four lessons |
| 13 | We said 0.656, and showed you why | Close, then questions |

Everything cut from the earlier 26-slide version lives in the speaker notes: the three leakage
guardrails, the mitigation tradeoffs, robustness, the decision log, engineering stats, and the
limitations list. They are answers to questions, not slides.

Design rules, enforced by `final_presentation_preview.py`:
40pt titles, 20pt body, nothing below 15pt, figures filling the full content width, and no more
than 1.15in of dead space at the foot of any slide.

## Rebuilding

The deck needs three packages the analysis pipeline does not, so it gets its own throwaway venv:

```bash
python3 -m venv .venv-deck
.venv-deck/bin/pip install python-pptx matplotlib pillow

# PY points make at that venv (the default is ./.venv, which is the analysis one)
make presentation       PY=.venv-deck/bin/python   # figures, then the .pptx
make presentation-check PY=.venv-deck/bin/python   # layout, claims, then script timing
```

Five scripts, each with one job:

| Script | Role |
|---|---|
| `scripts/final_presentation_figures.py` | 18 charts from `reports/*.json`, including 3 wide composites, project palette, 300 dpi |
| `scripts/final_presentation_build.py` | assembles the `.pptx` and writes the speaker notes |
| `scripts/final_presentation_preview.py` | renders the saved deck to `preview/` and fails on text overflow, figure/text collisions, undersized type, right-margin overruns, and dead space at the foot |
| `scripts/final_presentation_verify.py` | 57 numeric claims checked against `reports/`, plus framing rules |
| `scripts/final_presentation_script_check.py` | word-counts `SPEAKER_SCRIPT.md`, fails if it runs past 9:40 at 170 wpm, and checks every spoken figure against an artifact plus the on-slide facts (bar positions, education-gap range) |

`final_presentation_verify.py` is the guard that matters. It re-derives every figure on the slides
from the artifacts, cross-checks the headline numbers against a second independent artifact, and
fails the build if the deck ever asserts race is a biological cause, claims clinical usefulness,
claims fairness was solved, states an unqualified "trees beat deep learning", or drops the
not-a-clinical-tool disclaimer.

## Data provenance

All figures use the canonical **full-data 2018-2024 track** (24,920,407 singleton births).
One labelled exception: the `0.684 -> 0.654` pair on slide 6 is the development-sample measurement
that decision was actually made on. It is disclosed in the slide's source line, in its speaker
notes, and out loud in the speaker script.

The superseded ~2M-row sample-era results are not cited anywhere in the deck.
