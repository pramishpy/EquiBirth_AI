"""Render the built .pptx back to PNG previews, for visual audit and overflow detection.

This reads the SAVED presentation, so what you inspect is the actual artifact, not a parallel
model of it. Text is laid out with real matplotlib font metrics and wrapped at the shape width,
so a run that would overflow its textbox or the slide is reported.

Run: python scripts/final_presentation_preview.py
Output: presentation/preview/slide_NN.png  plus an overflow report on stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "presentation" / "EquiBirth_AI_final_presentation.pptx"
OUT = ROOT / "presentation" / "preview"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 110
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})

problems: list[str] = []


def emu_in(v) -> float:
    return Emu(int(v)).inches


def rgb(colour) -> str:
    try:
        return f"#{colour.rgb}"
    except Exception:
        return "#FBFBFD"


def wrap(fig, body: str, size_pt: float, bold: bool, italic: bool, max_w_in: float) -> list[str]:
    """Greedy wrap using measured text width at the target point size."""
    renderer = fig.canvas.get_renderer()

    def width_in(text: str) -> float:
        t = fig.text(0, 0, text, fontsize=size_pt,
                     fontweight="bold" if bold else "normal",
                     fontstyle="italic" if italic else "normal")
        bb = t.get_window_extent(renderer=renderer)
        t.remove()
        return bb.width / fig.dpi

    out: list[str] = []
    for para in body.split("\n"):
        if not para:
            out.append("")
            continue
        line = ""
        for word in para.split(" "):
            trial = f"{line} {word}".strip()
            if line and width_in(trial) > max_w_in:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    return out


MIN_PT = 12          # nothing on a slide may be smaller than the source line
MIN_BODY_PT = 15     # nothing that carries meaning may be smaller than this
MAX_FOOT_GAP = 1.15  # inches of empty space allowed below the lowest element


def check_typography_and_coverage(idx: int, slide, prs: Presentation) -> None:
    """Enforce the large-type rule and flag slides that leave the lower band empty."""
    sh, sw = emu_in(prs.slide_height), emu_in(prs.slide_width)
    lowest = 0.0
    rightmost = 0.0
    for shape in slide.shapes:
        bottom = emu_in(shape.top) + emu_in(shape.height)
        right = emu_in(shape.left) + emu_in(shape.width)
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE or (
                shape.has_text_frame and shape.text_frame.text.strip()):
            lowest = max(lowest, bottom)
            rightmost = max(rightmost, right)
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            for run in para.runs:
                if not run.text.strip():
                    continue
                pt = run.font.size.pt if run.font.size else 18
                floor = MIN_PT if len(run.text) > 90 else MIN_BODY_PT
                if pt < floor:
                    problems.append(
                        f"slide {idx}: {pt}pt text is below the {floor}pt floor :: "
                        f"{run.text[:44]!r}")
        # a text box must not extend past the right margin (word wrap cannot save it there)
        if emu_in(shape.left) + emu_in(shape.width) > sw - 0.5:
            problems.append(
                f"slide {idx}: text box reaches {emu_in(shape.left) + emu_in(shape.width):.2f}in, "
                f"past the {sw - 0.5:.2f}in right margin :: "
                f"{shape.text_frame.text[:44]!r}")

    if sh - lowest > MAX_FOOT_GAP:
        problems.append(
            f"slide {idx}: {sh - lowest:.2f}in of dead space at the foot "
            f"(content ends at {lowest:.2f}in of {sh:.2f}in); enlarge the figure or rebalance")
    if sw - rightmost > 1.6:
        problems.append(
            f"slide {idx}: {sw - rightmost:.2f}in of dead space on the right "
            f"(content ends at {rightmost:.2f}in of {sw:.2f}in)")


def check_overlaps(idx: int, slide, prs: Presentation) -> None:
    """Report text boxes that overlap a picture. Panels and rules may sit behind text by design."""
    pics, texts = [], []
    for shape in slide.shapes:
        box = (emu_in(shape.left), emu_in(shape.top),
               emu_in(shape.left) + emu_in(shape.width),
               emu_in(shape.top) + emu_in(shape.height))
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            pics.append(box)
        elif shape.has_text_frame and shape.text_frame.text.strip():
            texts.append((box, shape.text_frame.text[:46].replace("\n", " ")))

    # a body text block must not run into the small source line pinned near the foot
    sources = [(b, t) for (b, t) in texts if b[1] > emu_in(prs.slide_height) - 0.75]
    for (sx0, sy0, sx1, sy1), slabel in sources:
        for (tx0, ty0, tx1, ty1), label in texts:
            if label == slabel:
                continue
            if ty1 > sy0 + 0.06 and min(tx1, sx1) - max(tx0, sx0) > 0.5:
                problems.append(
                    f"slide {idx}: body text reaches {ty1:.2f}in and collides with the source "
                    f"line at {sy0:.2f}in :: {label!r}")

    for (px0, py0, px1, py1) in pics:
        if px1 - px0 < 4.0:
            problems.append(
                f"slide {idx}: figure shrank to {px1 - px0:.2f}in wide, too small to read "
                f"(max_bottom is probably too tight)")
        for (tx0, ty0, tx1, ty1), label in texts:
            ox = min(px1, tx1) - max(px0, tx0)
            oy = min(py1, ty1) - max(py0, ty0)
            if ox > 0.12 and oy > 0.12:
                problems.append(
                    f"slide {idx}: text overlaps a figure by {ox:.2f}x{oy:.2f}in :: {label!r}")


def render(prs: Presentation, idx: int, slide) -> None:
    sw, sh = emu_in(prs.slide_width), emu_in(prs.slide_height)
    bg = "#FBFBFD"
    try:
        bg = rgb(slide.background.fill.fore_color)
    except Exception:
        pass

    fig = plt.figure(figsize=(sw, sh), dpi=DPI)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, sw)
    ax.set_ylim(sh, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), sw, sh, facecolor=bg, edgecolor="none", zorder=0))

    for shape in slide.shapes:
        x, y = emu_in(shape.left), emu_in(shape.top)
        w, h = emu_in(shape.width), emu_in(shape.height)

        if x < -0.01 or y < -0.01 or x + w > sw + 0.01 or y + h > sh + 0.01:
            problems.append(
                f"slide {idx}: shape out of bounds: x={x:.2f} y={y:.2f} w={w:.2f} h={h:.2f}")

        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            img = plt.imread(str(shape.image.blob and _blob_path(shape)))
            ax.imshow(img, extent=(x, x + w, y + h, y), aspect="auto", zorder=2)
            continue

        if shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
            fc = rgb(shape.fill.fore_color) if shape.fill.type is not None else "none"
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=fc, edgecolor="none", zorder=1))

        if not shape.has_text_frame:
            continue

        cursor = y
        total = 0.0
        for para in shape.text_frame.paragraphs:
            runs = [r for r in para.runs if r.text]
            spacing = para.line_spacing or 1.0
            after = (para.space_after.pt / 72) if para.space_after else 0.0
            if not runs:
                total += 0.14 + after
                cursor += 0.14 + after
                continue
            r = runs[0]
            size = r.font.size.pt if r.font.size else 18
            bold = bool(r.font.bold)
            italic = bool(r.font.italic)
            colour = rgb(r.font.color) if r.font.color and r.font.color.type else "#16203A"
            body = "".join(run.text for run in runs)
            lines = wrap(fig, body, size, bold, italic, max(w, 0.4))
            lh = size / 72 * 1.22 * spacing
            align = str(para.alignment or "")
            for line in lines:
                if "CENTER" in align:
                    tx, ha = x + w / 2, "center"
                elif "RIGHT" in align:
                    tx, ha = x + w, "right"
                else:
                    tx, ha = x, "left"
                ax.text(tx, cursor + lh * 0.78, line, fontsize=size, ha=ha, va="baseline",
                        color=colour, fontweight="bold" if bold else "normal",
                        fontstyle="italic" if italic else "normal", zorder=3)
                cursor += lh
                total += lh
            total += after
            cursor += after

        if total > h + 0.30:
            snippet = shape.text_frame.text[:52].replace("\n", " ")
            problems.append(
                f"slide {idx}: text overflows box by {total - h:.2f}in "
                f"(needs {total:.2f}, box {h:.2f}) :: {snippet!r}")
        if cursor > sh + 0.05:
            snippet = shape.text_frame.text[:52].replace("\n", " ")
            problems.append(
                f"slide {idx}: text runs off the slide bottom to {cursor:.2f}in :: {snippet!r}")

    path = OUT / f"slide_{idx:02d}.png"
    fig.savefig(path, dpi=DPI, facecolor=bg)
    plt.close(fig)


_blobs: dict[int, Path] = {}


def _blob_path(shape) -> Path:
    """Cache picture blobs to temp files so imread can load them."""
    key = id(shape.image.blob)
    if key not in _blobs:
        p = OUT / f"_img_{len(_blobs)}.png"
        p.write_bytes(shape.image.blob)
        _blobs[key] = p
    return _blobs[key]


def main() -> None:
    prs = Presentation(str(PPTX))
    for i, slide in enumerate(prs.slides, start=1):
        check_typography_and_coverage(i, slide, prs)
        check_overlaps(i, slide, prs)
        render(prs, i, slide)
    for p in OUT.glob("_img_*.png"):
        p.unlink()

    print(f"rendered {len(prs.slides._sldIdLst)} slides to {OUT.relative_to(ROOT)}\n")
    if problems:
        print(f"{len(problems)} LAYOUT PROBLEM(S):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("no layout problems detected")


if __name__ == "__main__":
    main()
