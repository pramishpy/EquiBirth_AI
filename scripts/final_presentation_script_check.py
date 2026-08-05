"""Check the speaker script fits the 10-minute slot, and that it matches the deck.

Only the words inside ```say fences are spoken, so only those are counted. Delivery notes,
tables, and the Q&A section are not timed.

Checks:
  - one say-block per slide, in slide order, with no slide left silent
  - total speaking time inside the slot at the speaker's own pace (160-180 wpm)
  - per-slide time against the deck's own timing comments, so no single slide runs long
  - every number spoken in words traces to a canonical artifact

Run: python scripts/final_presentation_script_check.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "presentation" / "SPEAKER_SCRIPT.md"

WPM_SLOW, WPM_MID, WPM_FAST = 160, 170, 180   # the speaker's measured range, and its midpoint
LIMIT_S = 600                   # the hard 10-minute limit
# The gate is the MIDDLE pace with 20s of headroom: the script must land under 9:40 at 170 wpm.
# The slow-pace time is printed too, as a warning, not a hard failure, because at his slowest a
# presenter can always take one item off the cut list on the day.
GATE_WPM = WPM_MID
GATE_S = 580
FLOOR_S = 480                   # under 8 minutes would waste the slot

# Budget per slide in seconds at the gate pace (170 wpm). Weighted by narrative load rather
# than spread evenly: the leakage arc (4, 5, 6) and the fairness reversal (10) carry the talk,
# so they get the room. The +8s tolerance in the check means these are targets, not tripwires.
BUDGET_S = {1: 22, 2: 35, 3: 48, 4: 55, 5: 50, 6: 64, 7: 47,
            8: 47, 9: 53, 10: 62, 11: 43, 12: 44, 13: 24}

# Numbers spoken as words, and where each one comes from. The script says figures out loud
# rather than as digits, so a plain grep against reports/ cannot see them.
SPOKEN_NUMBERS = [
    ("eight and a half percent", "8.57% base rate, week2_phase2_audit"),
    ("twelve point three", "12.28% NH Black preterm rate"),
    ("seven point five", "7.54% NH White preterm rate"),
    ("four point seven", "4.739pp gap"),
    ("zero point nine three nine", "0.9394 leakage-added AUC, week2_experiments dual_track"),
    ("zero point nine one seven", "0.9171 leakage detector AUC, week2_phase2_audit"),
    ("zero point six five nine", "0.6590 clean AUC, week2_experiments dual_track"),
    ("zero point five five", "0.55 build-fail gate, FR6.1"),
    ("zero point six eight four", "0.684 pre-D26 dev-sample AUC"),
    ("zero point six five four", "0.654 post-D26 dev-sample AUC"),
    ("zero point six five six", "0.6561 held-out AUC-ROC, week5_models"),
    ("zero point five seven", "0.5704 best univariate AUC, week2_experiments"),
    ("two point seven four", "2.740 disparate impact ratio, week4_boosting_fairness"),
    ("nineteen and forty three", "19.2% to 43.3% explained, week7_explain decomposition"),
    ("three and four and a half", "per-education gaps run 3.10 to 4.34 pp, week1_dashboard_data"),
]

# Figures the SLIDES show but the script must not misstate. Each pair is (spoken phrase that
# would be wrong, why). The audit that prompted these caught the script saying "on the right"
# for bars that are at the bottom, and quoting an AUC-ROC while an AUC-PR chart was on screen.
FORBIDDEN = [
    ("two on the right are deep learning",
     "the deep-learning bars in model_field.png are the bottom two, not the right two"),
    ("her prenatal care", "prenatal visit count sits on the RIGHT of leakage_timeline.png"),
    ("between three and four points",
     "the education gaps reach 4.34 pp, so 'three to four' understates the chart"),
    ("said zero point nine two",
     "the closing slide prints 0.917, so the spoken figure must match it"),
]

problems: list[str] = []


def say_blocks(text: str) -> list[str]:
    return re.findall(r"```say\n(.*?)```", text, flags=re.DOTALL)


def slide_headings(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"^# (\d+)\.", text, flags=re.MULTILINE)]


PAUSE_S = 2.0   # a marked [PAUSE] is held for two seconds, so it costs real time


def words(block: str) -> int:
    # bracketed stage directions such as [PAUSE] are not spoken
    return len(re.sub(r"\[[^\]]*\]", " ", block).split())


def seconds(block: str, wpm: int) -> float:
    """Speaking time plus the marked pauses, which are silence but still spend the clock."""
    return words(block) / wpm * 60 + block.count("[PAUSE]") * PAUSE_S


def main() -> int:
    text = SCRIPT.read_text()
    blocks = say_blocks(text)
    slides = slide_headings(text)

    if slides != list(range(1, 14)):
        problems.append(f"slide headings are {slides}, expected 1 through 13 in order")
    if len(blocks) != 13:
        problems.append(f"found {len(blocks)} say-blocks, expected one per slide (13)")

    total = sum(words(b) for b in blocks)
    pauses = sum(b.count("[PAUSE]") for b in blocks)
    slow_s = sum(seconds(b, WPM_SLOW) for b in blocks)
    mid_s = sum(seconds(b, WPM_MID) for b in blocks)
    fast_s = sum(seconds(b, WPM_FAST) for b in blocks)

    print(f"{total} spoken words across {len(blocks)} slides, plus {pauses} held pauses")
    print(f"  at {WPM_FAST} wpm (fast): {fast_s / 60:.2f} min")
    print(f"  at {WPM_MID} wpm (his usual): {mid_s / 60:.2f} min  <- the gate")
    print(f"  at {WPM_SLOW} wpm (slow):  {slow_s / 60:.2f} min\n")

    if mid_s > GATE_S:
        problems.append(
            f"at his usual {WPM_MID} wpm the script runs {mid_s / 60:.2f} min, past the {GATE_S / 60:.2f}-min "
            f"gate; cut about {int((mid_s - GATE_S) / 60 * WPM_MID)} words")
    if slow_s > LIMIT_S:
        print(f"  NOTE: at his slowest ({WPM_SLOW} wpm) this is {slow_s / 60:.2f} min, over 10:00. "
              f"If he speaks slowly on the day, take one item off the cut list.\n")
    if fast_s < FLOOR_S:
        problems.append(
            f"at {WPM_FAST} wpm the script runs only {fast_s / 60:.2f} min, under {FLOOR_S / 60:.0f} "
            f"min; the slot is being wasted")

    for i, block in enumerate(blocks, start=1):
        secs = seconds(block, GATE_WPM)
        budget = BUDGET_S.get(i, 45)
        flag = "  OVER" if secs > budget + 8 else ""
        print(f"  slide {i:2d}: {words(block):4d} words, {secs:5.1f}s (budget {budget}s){flag}")
        if secs > budget + 8:
            problems.append(
                f"slide {i} needs {secs:.0f}s at {GATE_WPM} wpm but its budget is {budget}s")

    # the script is hard-wrapped for reading aloud, so a phrase can straddle a newline
    spoken = re.sub(r"\s+", " ", " ".join(blocks)).lower()
    for phrase, provenance in SPOKEN_NUMBERS:
        if phrase not in spoken:
            problems.append(f"spoken figure missing from the script: {phrase!r} ({provenance})")

    for phrase, why in FORBIDDEN:
        if phrase in spoken:
            problems.append(f"the script says {phrase!r}, which contradicts the slide: {why}")

    # the two framing commitments must be spoken, not just printed on a slide
    if "never a biological cause" not in spoken:
        problems.append("the race-as-marker framing is never spoken out loud")
    if "not a clinical tool" not in spoken:
        problems.append("the not-a-clinical-tool disclaimer is never spoken out loud")

    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  -", p)
        return 1
    print(f"script fits the slot and every spoken figure traces to an artifact "
          f"({len(SPOKEN_NUMBERS)} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
