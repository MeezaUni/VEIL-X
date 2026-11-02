# ruff: noqa: E501, RUF001  (slide and document text uses typographic characters)
"""Build the university course-project deliverables into ``deliverables/`` (local; binaries are
git-ignored and never pushed).

Inputs (committed):
- Markdown sources in ``docs/university/`` (report, demo guide, viva questions, project
  summary);
- committed reports for the figures.

Inputs (local only):
- real application screenshots in ``deliverables/assets/shot_*.png``, captured from the running
  app with synthetic DigiFace examples (see ``docs/university/demo_guide.md``).

Outputs:
- the report as DOCX + PDF, the presentation as PPTX + PDF;
- PDFs of the demo guide, viva questions and project summary;
- page previews (``deliverables/preview/``) for visual checking;
- ``MANIFEST.md`` with SHA-256 checksums.

Needs the documentation environment and Microsoft Office (Word / PowerPoint, used for PDF
export through COM):

    uv venv --python 3.12 .venv-docs
    uv pip install --python .venv-docs\\Scripts\\python.exe "python-pptx>=1.0" "python-docx>=1.1" "pymupdf>=1.24" "matplotlib>=3.8"
    .venv-docs\\Scripts\\python.exe scripts\\build_university_deliverables.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
ASSETS = OUT / "assets"
SRC = ROOT / "docs" / "university"
TITLE = "VEIL-X: An AI-Based Face Verification System with Quality Assessment and Intelligent Abstention"
NAVY, INK, GREY, ACCENT, AMBER = "1D4E89", "1F1F1F", "5A5A5A", "2A78D6", "9A5B00"
PLACEHOLDERS = [
    ("Student", "[Student Name]"),
    ("Register Number", "[Register Number]"),
    ("Course", "[Course Code and Name]"),
    ("Department", "[Department]"),
    ("College / University", "[College / University Name]"),
    ("Guide", "[Guide / Supervisor Name]"),
    ("Academic Year", "[Academic Year]"),
]


# --------------------------------------------------------------------------- figures
def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    def box(ax, x, y, w, h, text, fc, ec, dashed=False, bold=False, fs=9.5):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=fc,
                ec=ec,
                lw=1.6,
                ls="--" if dashed else "-",
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            color="#1f1f1f",
            fontweight="bold" if bold else "normal",
            wrap=True,
        )

    def arrow(ax, x0, y0, x1, y1, color="#52514e", dashed=False):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                color=color,
                lw=1.5,
                ls="--" if dashed else "-",
            )
        )

    # architecture (vertical decision path; sized so the text prints at about 9 pt at page width)
    fig, ax = plt.subplots(figsize=(8, 8.9))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8.9)
    ax.axis("off")
    blue, blue_e = "#e6f0fb", "#2a78d6"
    grey, grey_e = "#f1f1ef", "#8a8984"
    amber, amber_e = "#fdf1dc", "#9a5b00"
    ax.text(
        0.3,
        8.6,
        "Decision path (frozen models, calibrated rule)",
        fontsize=12,
        color="#1d4e89",
        fontweight="bold",
    )
    path = [
        "Two images (upload or example)",
        "Input checks: format, size, corruption",
        "SCRFD detection + 5-point alignment\nto 112×112",
        "AdaFace IR-101: embedding\n+ feature norm",
        "Similarity s, quality q = min(norms)",
        "M1 fused score p(s, q, s·q)",
        "A3 quality-stratified rule\n+ support guard",
        "MATCH / NON-MATCH / ABSTAIN",
    ]
    ys = [7.75 - k * 1.0 for k in range(len(path))]
    for k, (y, t) in enumerate(zip(ys, path, strict=True)):
        last = k == len(path) - 1
        box(
            ax,
            0.3,
            y,
            3.9,
            0.7,
            t,
            "#ffffff" if last else blue,
            "#1d4e89" if last else blue_e,
            bold=k >= 6,
            fs=11,
        )
        if k:
            arrow(ax, 2.25, ys[k - 1], 2.25, y + 0.7)
    rx, rw = 4.8, 3.0
    box(
        ax,
        rx,
        ys[0],
        rw,
        0.7,
        "Frozen files verified by\nSHA-256 at load",
        "#ffffff",
        "#52514e",
        fs=10.5,
    )
    box(
        ax,
        rx,
        ys[2],
        rw,
        0.7,
        "Refusals: several faces,\nno reliable face",
        amber,
        amber_e,
        fs=10.5,
    )
    arrow(ax, 4.2, ys[2] + 0.35, rx, ys[2] + 0.35, color=amber_e)
    ax.text(
        rx,
        ys[3] + 0.82,
        "Explanations only (after the decision)",
        fontsize=10,
        color="#5a5a5a",
        fontweight="bold",
    )
    box(
        ax,
        rx,
        ys[3],
        rw,
        0.7,
        "ArcFace IR-101 similarity\n(second opinion)",
        grey,
        grey_e,
        dashed=True,
        fs=10.5,
    )
    box(
        ax,
        rx,
        ys[4],
        rw,
        0.7,
        "Visibility network:\nheat maps + novelty",
        grey,
        grey_e,
        dashed=True,
        fs=10.5,
    )
    box(
        ax,
        rx,
        ys[6],
        rw,
        0.7,
        'Explanation engine:\n"Why this result?"',
        blue,
        blue_e,
        dashed=True,
        fs=10.5,
    )
    box(
        ax,
        rx,
        ys[7],
        rw,
        0.7,
        "Gradio interface on 127.0.0.1\n(no cloud, no storage)",
        "#ffffff",
        "#52514e",
        fs=10.5,
    )
    arrow(ax, 4.2, ys[2] + 0.2, rx, ys[3] + 0.45, dashed=True, color=grey_e)
    arrow(ax, 4.2, ys[2] + 0.1, rx, ys[4] + 0.45, dashed=True, color=grey_e)
    arrow(ax, 4.2, ys[6] + 0.35, rx, ys[6] + 0.35, dashed=True, color=blue_e)
    arrow(ax, 4.2, ys[7] + 0.35, rx, ys[7] + 0.35)
    arrow(ax, rx + rw / 2, ys[6], rx + rw / 2, ys[7] + 0.7)
    fig.tight_layout()
    fig.savefig(ASSETS / "architecture.png", dpi=200, facecolor="white")
    plt.close(fig)

    # workflow (two rows of three steps)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    flow = [
        "1. Upload two images\nor pick an example",
        "2. Confirm authorisation\nand limitations",
        "3. Detect, align and\ncheck the faces",
        "4. Embedding, similarity\nand quality",
        "5. Calibrated three-way\ndecision",
        '6. Read the result and\n"Why this result?"',
    ]
    pos = [(0.2, 2.0), (2.85, 2.0), (5.5, 2.0), (5.5, 0.3), (2.85, 0.3), (0.2, 0.3)]
    for k, (t, (x, y)) in enumerate(zip(flow, pos, strict=True)):
        box(ax, x, y, 2.3, 1.15, t, amber if k == 1 else blue, amber_e if k == 1 else blue_e, fs=11)
    for (x0, y0), (x1, y1) in itertools.pairwise(pos):
        if y0 == y1:
            if x1 > x0:
                arrow(ax, x0 + 2.3, y0 + 0.57, x1, y1 + 0.57)
            else:
                arrow(ax, x0, y0 + 0.57, x1 + 2.3, y1 + 0.57)
        else:
            arrow(ax, x0 + 1.15, y0, x1 + 1.15, y1 + 1.15)
    fig.tight_layout()
    fig.savefig(ASSETS / "workflow.png", dpi=200, facecolor="white")
    plt.close(fig)

    # decision rule of the default system (F1_mask), with the three built-in example results
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.set_xlim(-0.46, 1.05)
    ax.set_ylim(-0.9, 4.1)
    ax.axis("off")
    zone = {
        "NON-MATCH": ("#d9d9d6", "#4a4a4a"),
        "ABSTAIN": ("#f3d9a8", "#9a5b00"),
        "MATCH": ("#c9dcf2", "#1d4e89"),
    }
    groups = [
        (2.9, "High quality\nq ≥ 21.97", [(0.0, 0.673, "NON-MATCH"), (0.673, 1.0, "MATCH")]),
        (
            1.65,
            "Medium quality\n19.90 ≤ q < 21.97",
            [(0.0, 0.452, "NON-MATCH"), (0.452, 1.0, "MATCH")],
        ),
        (
            0.4,
            "Low quality\n10.82 ≤ q < 19.90",
            [(0.0, 0.154, "NON-MATCH"), (0.154, 0.833, "ABSTAIN"), (0.833, 1.0, "MATCH")],
        ),
    ]
    for y, label, zones in groups:
        ax.text(-0.03, y + 0.3, label, ha="right", va="center", fontsize=11, color="#1f1f1f")
        for a, b2, name in zones:
            fc, ec = zone[name]
            ax.add_patch(
                FancyBboxPatch(
                    (a, y), b2 - a, 0.6, boxstyle="square,pad=0", fc=fc, ec="white", lw=2
                )
            )
            ax.text(
                (a + b2) / 2,
                y + 0.3,
                name if b2 - a > 0.2 else name.replace("-", "-\n"),
                ha="center",
                va="center",
                fontsize=10.5 if b2 - a > 0.2 else 9,
                color=ec,
                fontweight="bold",
            )
        for a, _, _ in zones[1:]:
            ax.text(a, y - 0.1, f"{a:.3f}", ha="center", va="top", fontsize=10, color="#333333")
    for x, y, lab in (
        (1.0, 2.9, "Same person: p = 1.000"),
        (0.001, 1.65, "Different people: p = 0.001"),
        (0.542, 0.4, "Masked: p = 0.542"),
    ):
        ax.plot([x, x], [y - 0.05, y + 0.65], color="#111111", lw=2.4)
        ax.text(
            min(x, 0.99),
            y + 0.72,
            lab,
            ha="right" if x > 0.9 else "left" if x < 0.1 else "center",
            va="bottom",
            fontsize=10,
            color="#111111",
        )
    ax.text(
        0.3,
        -0.62,
        "Decision score p (0 to 1). Pairs with q < 10.82 always ABSTAIN (support guard).",
        ha="center",
        fontsize=10,
        color="#5a5a5a",
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "decision_rule.png", dpi=200, facecolor="white")
    plt.close(fig)

    # use case diagram (actor outside the system boundary; one column of use cases, no crossings)
    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (3.2, 0.15),
            6.6,
            7.05,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            fc="#fafbfd",
            ec="#1d4e89",
            lw=1.6,
        )
    )
    ax.text(
        6.5,
        6.88,
        "VEIL-X local application",
        ha="center",
        fontsize=12.5,
        color="#1d4e89",
        fontweight="bold",
    )
    ay = 3.55
    ax.add_patch(
        matplotlib.patches.Circle((1.1, ay + 0.75), 0.24, fc="white", ec="#1f1f1f", lw=1.8)
    )
    ax.plot([1.1, 1.1], [ay + 0.51, ay - 0.25], color="#1f1f1f", lw=1.8)
    ax.plot([0.65, 1.55], [ay + 0.3, ay + 0.3], color="#1f1f1f", lw=1.8)
    ax.plot([1.1, 0.72], [ay - 0.25, ay - 0.8], color="#1f1f1f", lw=1.8)
    ax.plot([1.1, 1.48], [ay - 0.25, ay - 0.8], color="#1f1f1f", lw=1.8)
    ax.text(1.1, ay - 1.15, "User", ha="center", fontsize=12.5, fontweight="bold")
    cases = [
        "Upload two face images",
        "Load a built-in example",
        "Set advanced options",
        "Confirm authorisation and limitations",
        "Compare faces",
        'Read the result and "Why this result?"',
        "View how it works, limitations, privacy",
        "Clear the inputs",
    ]
    for k, c in enumerate(cases):
        x, y = 6.5, 6.25 - k * 0.8
        ax.add_patch(
            matplotlib.patches.Ellipse((x, y), 5.9, 0.64, fc="#e6f0fb", ec="#2a78d6", lw=1.4)
        )
        ax.text(x, y, c, ha="center", va="center", fontsize=11.5)
        ax.plot([1.6, x - 2.95], [ay + 0.3, y], color="#8a8984", lw=1.0)
    fig.tight_layout()
    fig.savefig(ASSETS / "use_case.png", dpi=200, facecolor="white")
    plt.close(fig)

    # sequence diagram of one comparison
    fig, ax = plt.subplots(figsize=(8.6, 9.0))
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 9.0)
    ax.axis("off")
    actors = [
        "User",
        "Browser\n(Gradio UI)",
        "App server",
        "Pipeline\n(face, models)",
        "Decision\n(A3 + guard)",
        "Explanation\nengine",
    ]
    xs = [0.7 + k * 1.44 for k in range(len(actors))]
    for x, a in zip(xs, actors, strict=True):
        box(ax, x - 0.66, 8.25, 1.32, 0.66, a, "#e6f0fb", "#2a78d6", fs=10)
        ax.plot([x, x], [0.15, 8.25], color="#8a8984", lw=1, ls="--")
    msgs = [
        (0, 1, "upload image\nA / B"),
        (1, 2, "validate, keep\nbytes, delete file"),
        (2, 1, '"Image ready"'),
        (0, 1, "confirm, click\nCompare faces"),
        (1, 2, "compare\n(options)"),
        (2, 3, "detect, align,\nembed"),
        (3, 4, "s, q"),
        (4, 3, "p and\ndecision"),
        (3, 2, "result +\ndiagnostics"),
        (2, 5, "build the explanation"),
        (5, 2, "steps, exact comparison,\nwhat would change"),
        (2, 1, "result, bar,\nsteps, images"),
    ]
    for k, (a, b2, label) in enumerate(msgs):
        y = 7.75 - k * 0.64
        color = "#9a5b00" if a == 0 else "#1d4e89"
        arrow(ax, xs[a], y, xs[b2], y, color=color, dashed=b2 < a)
        ax.text(
            (xs[a] + xs[b2]) / 2,
            y + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#1f1f1f",
            linespacing=1.05,
        )
    fig.tight_layout()
    fig.savefig(ASSETS / "sequence.png", dpi=200, facecolor="white")
    plt.close(fig)

    # research figures as PNG (from committed reports, same code as the SVGs)
    spec = importlib.util.spec_from_file_location(
        "figs", ROOT / "scripts" / "make_phase6_figures.py"
    )
    assert spec is not None and spec.loader is not None
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)
    figs.OUT = ASSETS
    orig = plt.Figure.savefig

    def as_png(self, path, **kw):  # type: ignore[no-untyped-def]
        name = {
            "phase6_a3_population_shift.svg": "fig_population_shift.png",
            "phase6_lowq_tail.svg": "fig_lowq.png",
            "phase6_m1_extrapolation.svg": "fig_m1.png",
        }[Path(path).name]
        orig(self, ASSETS / name, dpi=170, **kw)

    plt.Figure.savefig = as_png  # type: ignore[method-assign]
    try:
        figs.plt.rcParams.update({"font.family": "DejaVu Sans"})
        figs.fig_population()
        figs.fig_lowq()
        figs.fig_m1()
    finally:
        plt.Figure.savefig = orig  # type: ignore[method-assign]


# --------------------------------------------------------------------------- Markdown -> DOCX
LABEL_RE = re.compile(r"\s*\{#((?:fig|tab):[\w-]+)(?:\s+h=([\d.]+))?\}")
REF_RE = re.compile(r"@((?:fig|tab):[\w-]+)")
REPORT_HEADER = "VEIL-X  |  University Course Project Report"
HEAD_FONT = "Calibri"
SUB_RE = re.compile(r"(τ_high|τ_low|[ef]_[AB])")


def number_labels(text: str) -> str:
    """Number figures and tables in order of appearance and resolve ``@fig:x`` / ``@tab:x``.

    A figure is a line ``![caption](path){#fig:x}``; a table caption is a line
    ``Table: caption {#tab:x}`` placed directly above the table. The builder numbers captions in
    the same order, so the references in the text always match the captions."""
    nums: dict[str, str] = {}
    counts = {"fig": 0, "tab": 0}
    for line in text.splitlines():
        s = line.strip()
        kind = "fig" if s.startswith("![") else "tab" if s.startswith("Table:") else None
        if kind is None:
            continue
        counts[kind] += 1
        m = LABEL_RE.search(s)
        if m:
            nums[m.group(1)] = f"{'Figure' if kind == 'fig' else 'Table'} {counts[kind]}"
    missing = sorted(set(REF_RE.findall(text)) - set(nums))
    if missing:
        raise ValueError(f"unresolved figure/table references: {missing}")
    text = REF_RE.sub(lambda m: nums[m.group(1)], text)
    return LABEL_RE.sub(lambda m: f"{{h={m.group(2)}}}" if m.group(2) else "", text)


def column_widths(rows: list[list[str]], total_cm: float) -> list[float]:
    """Column widths from the content: long-text columns get more room, and no column is
    narrower than its longest word."""
    ncol = len(rows[0])
    weights, minimum = [], []
    for ci in range(ncol):
        texts = [re.sub(r"[*`]", "", r[ci]) for r in rows]
        longest = max((len(w) for t in texts for w in t.split()), default=1)
        avg = sum(len(t) for t in texts) / len(texts)
        weights.append(max(avg, longest * 1.3, 5.0))
        minimum.append(min(0.21 * longest + 0.4, total_cm / ncol * 1.6))
    widths = [total_cm * w / sum(weights) for w in weights]
    for _ in range(4):  # raise narrow columns to their minimum, take the room from wide ones
        short = sum(max(0.0, m - w) for w, m in zip(widths, minimum, strict=True))
        if short <= 1e-6:
            break
        spare = [max(0.0, w - m) for w, m in zip(widths, minimum, strict=True)]
        widths = [
            max(w, m) - (short * s / sum(spare) if sum(spare) else 0.0)
            for w, m, s in zip(widths, minimum, spare, strict=True)
        ]
    return widths


def md_to_docx(
    md_path: Path,
    docx_path: Path,
    title_page: bool,
    toc: bool,
    compact: bool = False,
    report: bool = False,
) -> None:
    """Convert one Markdown source to DOCX.

    ``report=True`` produces the formal report layout: title page (unchanged), front matter
    with roman page numbers, chapters with arabic numbers starting on new pages, numbered
    figure and table captions (Word SEQ fields), lists of contents / figures / tables, repeated
    table header rows, rows that never split across pages, content-based column widths,
    justified 1.5-spaced text. Markers in the source: ``<!-- FRONTMATTER -->``,
    ``<!-- MAINMATTER -->``, ``<!-- TOC -->``, ``<!-- LOF -->``, ``<!-- LOT -->``,
    ``<!-- PAGEBREAK -->``."""
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    text = md_path.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        head, text = text[3:].split("---", 1)
        for line in head.strip().splitlines():
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    if report:
        text = number_labels(text)
        text = re.sub(r"(\d) %", "\\1\u00a0%", text)
        text = re.sub(
            r"\b(Figure|Table|Section|Chapter|Appendix) (\d|[AB]\b)", "\\1\u00a0\\2", text
        )
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.9 if compact else 2.3)
    sec.top_margin = sec.bottom_margin = Cm(1.6 if compact else 2.2)
    body_font = "Times New Roman" if report else "Calibri"
    body_size = 12.0 if report else (10.0 if compact else 11.0)
    text_w = 16.0 if report else 21.0 - 2 * (1.9 if compact else 2.3)
    st = doc.styles["Normal"]
    st.font.name = body_font
    st.font.size = Pt(body_size)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.5 if report else 1.15
    sizes = ((1, 17), (2, 13.5), (3, 12)) if report else ((1, 16), (2, 13), (3, 11.5))
    for lvl, size in sizes:
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = body_font
        h.font.size = Pt(size)
        h.font.color.rgb = RGBColor.from_string(NAVY)
        h.font.bold = True
        h.font.italic = lvl == 3 and report
        h.paragraph_format.space_before = Pt(14 if lvl == 1 else 12 if report else 10)
        h.paragraph_format.space_after = Pt(10 if report and lvl == 1 else 6)
        h.paragraph_format.keep_with_next = True
        h.paragraph_format.line_spacing = 1.15
        if report and lvl == 1:
            h.paragraph_format.page_break_before = True
    if report:
        from docx.enum.style import WD_STYLE_TYPE

        for lvl in (1, 2, 3):
            _plain_font(doc.styles[f"Heading {lvl}"], HEAD_FONT, OxmlElement, qn)
        for name in ("TOC 1", "TOC 2", "Table of Figures"):
            try:
                ts = doc.styles[name]
            except KeyError:
                ts = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
                ts.base_style = doc.styles["Normal"]
            ts.paragraph_format.line_spacing = 1.2
            ts.paragraph_format.space_after = Pt(3)
            ts.paragraph_format.space_before = Pt(0)
    cap = doc.styles["Caption"]
    cap.font.name = body_font
    cap.font.size = Pt(10.5 if report else 9.5)
    cap.font.italic = False
    cap.font.bold = False
    cap.font.color.rgb = RGBColor.from_string("333333")
    cap.paragraph_format.line_spacing = 1.0
    cap.paragraph_format.space_after = Pt(10)

    def footer_page(section, prefix: str = "") -> None:
        fp = section.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if prefix:
            r = fp.add_run(prefix)
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor.from_string(GREY)
        _field(fp, "PAGE", OxmlElement, qn, cached="1")
        for r in fp.runs:
            r.font.size = Pt(9 if report else 8.5)
            r.font.color.rgb = RGBColor.from_string(GREY)

    if not report:
        footer_page(sec, meta.get("title", "").split(":")[0] + "  ·  page ")

    def new_section(kind: str) -> None:
        s = doc.add_section(WD_SECTION.NEW_PAGE)
        s.page_height, s.page_width = Cm(29.7), Cm(21.0)
        s.left_margin, s.right_margin = Cm(3.0), Cm(2.0)
        s.top_margin, s.bottom_margin = Cm(2.5), Cm(2.3)
        s.header_distance = s.footer_distance = Cm(1.2)
        s.footer.is_linked_to_previous = False
        s.header.is_linked_to_previous = False
        footer_page(s)
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:fmt"), "lowerRoman" if kind == "front" else "decimal")
        pg.set(qn("w:start"), "1")
        s._sectPr.insert_element_before(
            pg,
            "w:cols",
            "w:formProt",
            "w:vAlign",
            "w:noEndnote",
            "w:titlePg",
            "w:textDirection",
            "w:bidi",
            "w:rtlGutter",
            "w:docGrid",
            "w:printerSettings",
            "w:sectPrChange",
        )
        hp = s.header.paragraphs[0]
        if kind == "main":
            hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            r = hp.add_run(REPORT_HEADER)
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor.from_string(GREY)
            _border(hp, "bottom", "A6A6A6", OxmlElement, qn)

    def add_inline(par, s: str, size: float | None = None, color: str | None = None) -> None:
        tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)", s)
        for tok in tokens:
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                r = par.add_run(tok[2:-2])
                r.bold = True
            elif tok.startswith("`") and tok.endswith("`"):
                r = par.add_run(tok[1:-1])
                r.font.name = "Consolas"
                r.font.size = Pt((size or body_size) - (1.5 if report else 1))
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                r = par.add_run(tok[1:-1])
                r.italic = True
            elif report and SUB_RE.search(tok):
                for k, part in enumerate(SUB_RE.split(tok)):
                    if not part:
                        continue
                    if k % 2:
                        base, sub = part.split("_", 1)
                        rr = par.add_run(base)
                        rs = par.add_run(sub)
                        rs.font.subscript = True
                        for x in (rr, rs):
                            if size:
                                x.font.size = Pt(size)
                            if color:
                                x.font.color.rgb = RGBColor.from_string(color)
                    else:
                        r = par.add_run(part)
                        if size:
                            r.font.size = Pt(size)
                        if color:
                            r.font.color.rgb = RGBColor.from_string(color)
                continue
            else:
                r = par.add_run(tok)
            if size and not (tok.startswith("`") and tok.endswith("`")):
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = RGBColor.from_string(color)

    def list_title(title: str, instr: str) -> None:
        p = doc.add_paragraph()
        p.paragraph_format.page_break_before = True
        p.paragraph_format.space_after = Pt(12)
        r = p.add_run(title)  # not a Heading style, so the table of contents does not list it
        r.font.name = HEAD_FONT
        r.bold = True
        r.font.size = Pt(17)
        r.font.color.rgb = RGBColor.from_string(NAVY)
        _border(p, "bottom", NAVY, OxmlElement, qn)
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.3
        _field(p, instr, OxmlElement, qn)

    if title_page:
        start = len(doc.paragraphs)
        for _ in range(5):
            doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(meta.get("title", TITLE))
        r.bold = True
        r.font.size = Pt(22)
        r.font.color.rgb = RGBColor.from_string(NAVY)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(meta.get("subtitle", ""))
        r.font.size = Pt(14)
        r.font.color.rgb = RGBColor.from_string(GREY)
        doc.add_paragraph()
        doc.add_paragraph()
        for label, val in PLACEHOLDERS:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, f"**{label}:** {val}", size=12)
        doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline(
            p,
            "Academic demonstration project. Not a certified or production identity-verification system.",
            size=10,
            color=GREY,
        )
        if report:  # keep the title page exactly as in the original layout
            for tp in doc.paragraphs[start:]:
                tp.paragraph_format.line_spacing = 1.15
                tp.paragraph_format.space_after = Pt(6)
                for r in tp.runs:
                    r.font.name = "Calibri"
                    if r.font.size is None:
                        r.font.size = Pt(11)
        else:
            p.add_run().add_break(WD_BREAK.PAGE)
    elif not report:
        p = doc.add_paragraph()
        r = p.add_run(meta.get("title", ""))
        r.bold = True
        r.font.size = Pt(18)
        r.font.color.rgb = RGBColor.from_string(NAVY)
        if meta.get("subtitle"):
            p = doc.add_paragraph()
            r = p.add_run(meta["subtitle"])
            r.font.size = Pt(12)
            r.font.color.rgb = RGBColor.from_string(GREY)
    if toc and not report:
        h = doc.add_paragraph()  # not a Heading style, so the TOC does not list itself
        r = h.add_run("Table of Contents")
        r.bold = True
        r.font.size = Pt(16)
        r.font.color.rgb = RGBColor.from_string(NAVY)
        p = doc.add_paragraph()
        _field(p, 'TOC \\o "1-2" \\h \\z \\u', OxmlElement, qn)
        p.add_run().add_break(WD_BREAK.PAGE)
        del h

    def add_table(rows: list[list[str]], caption: str | None) -> None:
        nonlocal tab_n
        ncol = max(len(r) for r in rows)
        rows = [r + [""] * (ncol - len(r)) for r in rows]
        if caption is not None:
            tab_n += 1
            cp = doc.add_paragraph(style="Caption")
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.paragraph_format.keep_with_next = True
            cp.paragraph_format.space_before = Pt(8)
            cp.paragraph_format.space_after = Pt(4)
            rb = cp.add_run("Table ")
            rb.bold = True
            _field(cp, "SEQ Table \\* ARABIC", OxmlElement, qn, cached=str(tab_n), bold=True)
            rb = cp.add_run(": ")
            rb.bold = True
            add_inline(cp, caption)
        t = doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        widths = column_widths(rows, text_w) if report else None
        if widths:
            t.autofit = False
            for ci, w in enumerate(widths):
                t.columns[ci].width = Cm(w)
        keep = report and len(rows) <= 12
        for ri, row in enumerate(rows):
            tr_pr = t.rows[ri]._tr.get_or_add_trPr()
            el = OxmlElement("w:cantSplit")
            tr_pr.append(el)
            if ri == 0:
                el = OxmlElement("w:tblHeader")
                tr_pr.append(el)
            for ci in range(ncol):
                cell = t.cell(ri, ci)
                if widths:
                    cell.width = Cm(widths[ci])
                par = cell.paragraphs[0]
                pf = par.paragraph_format
                pf.space_after = Pt(1.5 if report else 1)
                pf.space_before = Pt(1.5 if report else 0)
                if report:
                    pf.line_spacing = 1.05
                    pf.keep_with_next = keep and ri < len(rows) - 1
                add_inline(par, row[ci], size=10.0 if report else 9.5)
                if ri == 0:
                    for r in par.runs:
                        r.bold = True
                        r.font.color.rgb = RGBColor.from_string("FFFFFF")
                    _shade(cell, NAVY, OxmlElement, qn)
                elif report and ri % 2 == 0:
                    _shade(cell, "EEF3F9", OxmlElement, qn)
        sp = doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(4 if report else 2)
        sp.paragraph_format.line_spacing = 1.0

    lines = text.replace("<!-- TITLEPAGE -->", "").splitlines()
    i, fig_n, tab_n = 0, 0, 0
    pending_caption: str | None = None
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if report and s.startswith("<!--") and s.endswith("-->"):
            marker = s[4:-3].strip()
            if marker in ("FRONTMATTER", "MAINMATTER"):
                new_section("front" if marker == "FRONTMATTER" else "main")
            elif marker == "TOC":
                list_title("Table of Contents", 'TOC \\o "1-2" \\h \\z \\u')
            elif marker == "LOF":
                list_title("List of Figures", 'TOC \\h \\z \\c "Figure"')
            elif marker == "LOT":
                list_title("List of Tables", 'TOC \\h \\z \\c "Table"')
            elif marker == "PAGEBREAK":
                doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            for k, b in enumerate(block):
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.left_indent = Cm(0.6)
                if report:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.keep_with_next = k < len(block) - 1
                    _shade_par(p, "F2F2F2", OxmlElement, qn)
                r = p.add_run(b if b else " ")
                r.font.name = "Consolas"
                r.font.size = Pt(9)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", s)
        if m:
            hp = doc.add_paragraph(m.group(2), style=f"Heading {len(m.group(1))}")
            if report and len(m.group(1)) == 1:
                _border(hp, "bottom", NAVY, OxmlElement, qn)
            i += 1
            continue
        if report and s.startswith("Table:"):
            pending_caption = s[len("Table:") :].strip()
            i += 1
            continue
        if report and s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.keep_together = True
            add_inline(p, s[2:-2].strip())
            for r in p.runs:
                r.font.name = "Cambria Math"
            i += 1
            continue
        m = re.match(r"^!\[(.*)\]\(([^)]*)\)(?:\{h=([\d.]+)\})?$", s)
        if m:
            caption, path = m.group(1), OUT / m.group(2)
            if path.exists():
                from PIL import Image

                w, h = Image.open(path).size
                if report:
                    max_h = float(m.group(3)) if m.group(3) else 15.5  # optional {#fig:x h=12}
                    width = Cm(min(text_w, max_h * w / h))
                else:
                    width = Cm(16.4) if w / h > 1.25 else Cm(min(16.4, 11.0 * w / h))
                doc.add_picture(str(path), width=width)
                pic = doc.paragraphs[-1]
                pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pic.paragraph_format.keep_with_next = True  # caption stays with figure
                if report:
                    pic.paragraph_format.line_spacing = 1.0
                    pic.paragraph_format.space_before = Pt(6)
            else:
                p = doc.add_paragraph()
                add_inline(p, f"[MISSING IMAGE: {m.group(2)}]", color="C00000")
            fig_n += 1
            if report:
                p = doc.add_paragraph(style="Caption")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                rb = p.add_run("Figure ")
                rb.bold = True
                _field(p, "SEQ Figure \\* ARABIC", OxmlElement, qn, cached=str(fig_n), bold=True)
                rb = p.add_run(": ")
                rb.bold = True
                add_inline(p, caption)
            else:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, caption, size=9.5, color=GREY)
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            add_table(rows, pending_caption)
            pending_caption = None
            continue
        if s.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8)
            p.paragraph_format.right_indent = Cm(0.8)
            if report:
                p.paragraph_format.line_spacing = 1.3
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(10)
                _shade_par(p, "EEF3F9", OxmlElement, qn)
                _border(p, "left", NAVY, OxmlElement, qn, size=18)
                p.paragraph_format.keep_together = True
            add_inline(p, s.lstrip("> ").strip(), color=None if report else NAVY)
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                p.add_run(" ")
                add_inline(p, lines[i].strip().lstrip("> ").strip(), color=None if report else NAVY)
                i += 1
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
        if m:
            indent = len(m.group(1)) // 2
            numbered = m.group(2)[0].isdigit()
            if numbered:  # explicit numbers: Word's List Number would continue across the document
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.9 + 0.6 * indent)
                p.paragraph_format.first_line_indent = Cm(-0.6)
                p.add_run(m.group(2) + " ")
            else:
                p = doc.add_paragraph(style="List Bullet 2" if indent else "List Bullet")
            p.paragraph_format.space_after = Pt(3 if report else 2)
            if report:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.line_spacing = 1.35
            content = m.group(3)
            i += 1
            while (
                i < len(lines)
                and lines[i][:1] == " "
                and lines[i].strip()
                and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i])
            ):
                content += " " + lines[i].strip()
                i += 1
            add_inline(p, content)
            continue
        para = s
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(
                r"^(#|\||!\[|```|>|<!--|\$\$|Table:|\s*([-*]|\d+\.)\s)", lines[i].strip()
            )
        ):
            para += " " + lines[i].strip()
            i += 1
        p = doc.add_paragraph()
        if report:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_after = Pt(8)
        add_inline(p, para)
    if pending_caption is not None:
        raise ValueError(f"table caption without a table: {pending_caption}")
    doc.core_properties.title = meta.get("title", "")
    doc.core_properties.author = "[Student Name]"
    doc.save(str(docx_path))


def _field(
    par,
    instr,
    OxmlElement,  # noqa: N803
    qn,
    cached: str = "(right-click → Update Field)",
    bold: bool = False,
) -> None:
    r = par.add_run()
    if bold:
        r.bold = True
    for kind, txt in (
        ("begin", None),
        (None, instr),
        ("separate", None),
        (None, cached),
        ("end", None),
    ):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
            r._r.append(el)
        elif txt == instr:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = f" {instr} "
            r._r.append(el)
        else:
            el = OxmlElement("w:t")
            el.text = txt
            r._r.append(el)


def _plain_font(style, font: str, OxmlElement, qn) -> None:  # noqa: N803
    """Set an explicit font on a style and drop theme fonts, which would otherwise override it."""
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        rfonts.attrib.pop(qn(f"w:{attr}"), None)
    for attr in ("ascii", "hAnsi", "cs"):
        rfonts.set(qn(f"w:{attr}"), font)


def _shade(cell, hex_color, OxmlElement, qn) -> None:  # noqa: N803
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _shade_par(par, hex_color, OxmlElement, qn) -> None:  # noqa: N803
    p_pr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    p_pr.insert_element_before(
        shd,
        "w:tabs",
        "w:suppressAutoHyphens",
        "w:kinsoku",
        "w:wordWrap",
        "w:overflowPunct",
        "w:topLinePunct",
        "w:autoSpaceDE",
        "w:autoSpaceDN",
        "w:bidi",
        "w:adjustRightInd",
        "w:snapToGrid",
        "w:spacing",
        "w:ind",
        "w:contextualSpacing",
        "w:mirrorIndents",
        "w:suppressOverlap",
        "w:jc",
        "w:textDirection",
        "w:textAlignment",
        "w:textboxTightWrap",
        "w:outlineLvl",
        "w:divId",
        "w:cnfStyle",
        "w:rPr",
        "w:sectPr",
        "w:pPrChange",
    )


def _border(par, side, hex_color, OxmlElement, qn, size: int = 8) -> None:  # noqa: N803
    p_pr = par._p.get_or_add_pPr()
    bdr = p_pr.find(qn("w:pBdr"))
    if bdr is None:
        bdr = OxmlElement("w:pBdr")
        p_pr.insert_element_before(
            bdr,
            "w:shd",
            "w:tabs",
            "w:suppressAutoHyphens",
            "w:kinsoku",
            "w:wordWrap",
            "w:overflowPunct",
            "w:topLinePunct",
            "w:autoSpaceDE",
            "w:autoSpaceDN",
            "w:bidi",
            "w:adjustRightInd",
            "w:snapToGrid",
            "w:spacing",
            "w:ind",
            "w:contextualSpacing",
            "w:mirrorIndents",
            "w:suppressOverlap",
            "w:jc",
            "w:textDirection",
            "w:textAlignment",
            "w:textboxTightWrap",
            "w:outlineLvl",
            "w:divId",
            "w:cnfStyle",
            "w:rPr",
            "w:sectPr",
            "w:pPrChange",
        )
    el = OxmlElement(f"w:{side}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(size))
    el.set(qn("w:space"), "4")
    el.set(qn("w:color"), hex_color)
    bdr.append(el)


# --------------------------------------------------------------------------- Office PDF export
def office_pdf(docx_or_pptx: Path, pdf: Path) -> None:
    src, dst = str(docx_or_pptx.resolve()), str(pdf.resolve())
    if docx_or_pptx.suffix == ".docx":
        ps = f"""
$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
try {{ $d = $w.Documents.Open('{src}', $false, $false)
  $d.Fields.Update() | Out-Null
  foreach ($t in $d.TablesOfContents) {{ $t.Update() }}
  foreach ($t in $d.TablesOfFigures) {{ $t.Update() }}
  $d.Repaginate()
  foreach ($t in $d.TablesOfContents) {{ $t.UpdatePageNumbers() }}
  foreach ($t in $d.TablesOfFigures) {{ $t.UpdatePageNumbers() }}
  $d.Save(); $d.ExportAsFixedFormat('{dst}', 17); $d.Close(0) }} finally {{ $w.Quit() }}"""
    else:
        ps = f"""
$p = New-Object -ComObject PowerPoint.Application
try {{ $pr = $p.Presentations.Open('{src}', $true, $false, $false); $pr.SaveAs('{dst}', 32); $pr.Close() }} finally {{ $p.Quit() }}"""
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True, timeout=600
    )


# --------------------------------------------------------------------------- PowerPoint
def build_pptx(path: Path, tests_text: str) -> None:
    """The deck is defined in scripts/university_slides.py (native charts, diagrams, notes)."""
    spec = importlib.util.spec_from_file_location(
        "university_slides", ROOT / "scripts" / "university_slides.py"
    )
    assert spec is not None and spec.loader is not None
    slides = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slides)
    slides.build(path, ASSETS, tests_text)


# --------------------------------------------------------------------------- previews, manifest
def previews(pdf: Path, max_pages: int = 60) -> list[Path]:
    import pymupdf

    out = OUT / "preview" / pdf.stem
    out.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf))
    paths = []
    for k, page in enumerate(doc):
        if k >= max_pages:
            break
        pix = page.get_pixmap(dpi=60)
        p = out / f"page{k + 1:02d}.png"
        pix.save(str(p))
        paths.append(p)
    return paths


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    import pymupdf
    from docx import Document
    from pptx import Presentation

    tests_text = sys.argv[1] if len(sys.argv) > 1 else "400+ passing (CPU and GPU)"
    OUT.mkdir(exist_ok=True)
    figures()
    jobs = [
        ("report.md", "VEIL-X_University_Project_Report", True, True),
        ("demo_guide.md", "VEIL-X_Demo_Guide", False, False),
        ("viva_questions.md", "VEIL-X_Viva_Questions", False, False),
        ("project_summary.md", "VEIL-X_Project_Summary", False, False),
    ]
    produced: list[tuple[Path, str]] = []
    for src, stem, title_page, toc in jobs:
        docx = OUT / f"{stem}.docx"
        md_to_docx(
            SRC / src,
            docx,
            title_page,
            toc,
            compact=stem == "VEIL-X_Project_Summary",
            report=stem == "VEIL-X_University_Project_Report",
        )
        office_pdf(docx, OUT / f"{stem}.pdf")
        if stem == "VEIL-X_University_Project_Report":
            produced.append((docx, "University project report (editable)"))
        else:
            docx.unlink()  # only the PDF is a deliverable for these
        produced.append(
            (
                OUT / f"{stem}.pdf",
                {
                    "VEIL-X_University_Project_Report": "University project report (PDF)",
                    "VEIL-X_Demo_Guide": "3-5 minute demo guide + backup plan",
                    "VEIL-X_Viva_Questions": "Viva Q&A (35) + one-page revision sheet",
                    "VEIL-X_Project_Summary": "One-page project summary",
                }[stem],
            )
        )
    pptx = OUT / "VEIL-X_University_Presentation.pptx"
    build_pptx(pptx, tests_text)
    office_pdf(pptx, OUT / "VEIL-X_University_Presentation.pdf")
    produced += [
        (pptx, "Presentation: 8 slides, native charts, speaker notes"),
        (OUT / "VEIL-X_University_Presentation.pdf", "Presentation (PDF export)"),
    ]
    # verification: every file opens; page counts
    rows = []
    for p, purpose in produced:
        if p.suffix == ".pdf":
            n = len(pymupdf.open(str(p)))
            previews(p)
            info = f"{n} pages"
        elif p.suffix == ".docx":
            info = f"{len(Document(str(p)).paragraphs)} paragraphs"
        else:
            info = f"{len(Presentation(str(p)).slides)} slides"
        rows.append(
            f"| `{p.name}` | {purpose} | {info} | {p.stat().st_size / 1024:.0f} KB | `{sha256(p)}` |"
        )
    head = f"""# VEIL-X University Deliverables: Manifest

Built {date.today().isoformat()} by `scripts/build_university_deliverables.py` from the sources in
`docs/university/`.

**Storage.** The files are kept locally in `deliverables/` and are not pushed: the repository
refuses binary documents and images by design (privacy policy).

**Images.** Screenshots show the real running application. The face images in them are synthetic
DigiFace faces generated locally; no real person appears.

| File | Purpose | Check | Size | SHA-256 |
|---|---|---|---|---|
"""
    (OUT / "MANIFEST.md").write_text(head + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
