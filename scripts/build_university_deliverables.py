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

    # architecture
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    blue, blue_e = "#e6f0fb", "#2a78d6"
    grey, grey_e = "#f1f1ef", "#8a8984"
    amber, amber_e = "#fdf1dc", "#9a5b00"
    ax.text(
        0.1,
        5.95,
        "Decision path (frozen models and calibrated rules)",
        fontsize=10.5,
        color="#1d4e89",
        fontweight="bold",
    )
    steps = [
        (0.1, "Two images\n(upload)"),
        (1.75, "Input checks\nformat, size,\ncorruption"),
        (3.4, "SCRFD detection\n+ 5-point\nalignment 112×112"),
        (5.05, "AdaFace IR-101\nembedding +\nfeature norm"),
        (6.7, "Similarity s\nQuality q =\nmin(norms)"),
        (8.35, "M1 logistic\nfusion\np(s, q, s·q)"),
    ]
    for x, t in steps:
        box(ax, x, 3.95, 1.45, 1.6, t, blue, blue_e)
    for (x, _), (x2, _) in itertools.pairwise(steps):
        arrow(ax, x + 1.45, 4.75, x2, 4.75)
    box(
        ax,
        10.0,
        3.95,
        1.9,
        1.6,
        "A3 quality-\nstratified rule\n+ support guard",
        blue,
        blue_e,
        bold=True,
    )
    arrow(ax, 9.8, 4.75, 10.0, 4.75)
    box(
        ax,
        10.0,
        2.2,
        1.9,
        1.2,
        "MATCH\nNON-MATCH\nABSTAIN",
        "#ffffff",
        "#1d4e89",
        bold=True,
        fs=10.5,
    )
    arrow(ax, 10.95, 3.95, 10.95, 3.4)
    box(
        ax,
        3.4,
        2.25,
        1.45,
        1.05,
        "Refusals:\nseveral faces,\nno reliable face",
        amber,
        amber_e,
        fs=8.5,
    )
    arrow(ax, 4.12, 3.95, 4.12, 3.3, color=amber_e)
    ax.text(
        0.1,
        1.75,
        "Explanations only (computed after the decision; never change it)",
        fontsize=10.5,
        color="#5a5a5a",
        fontweight="bold",
    )
    diags = [
        (3.4, "ArcFace IR-101\nsimilarity\n(comparison)"),
        (5.4, "Visibility network\n(U-Net style)\nheat maps"),
        (7.4, "Novelty score\n'unusual input'\nflag"),
    ]
    for x, t in diags:
        box(ax, x, 0.25, 1.75, 1.25, t, grey, grey_e, dashed=True, fs=8.8)
    arrow(ax, 5.77, 3.95, 4.3, 1.5, dashed=True, color=grey_e)
    arrow(ax, 5.77, 3.95, 6.27, 1.5, dashed=True, color=grey_e)
    arrow(ax, 6.27, 1.5, 8.0, 1.5, dashed=True, color=grey_e)
    box(
        ax,
        9.6,
        0.25,
        2.3,
        1.25,
        "Gradio interface on\n127.0.0.1 only\n(no cloud, no storage)",
        "#ffffff",
        "#52514e",
        fs=8.8,
    )
    box(
        ax,
        0.1,
        2.25,
        2.9,
        1.05,
        "Frozen files checked by SHA-256\nat load (models, detector,\nthresholds)",
        "#ffffff",
        "#52514e",
        fs=8.5,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "architecture.png", dpi=170, facecolor="white")
    plt.close(fig)

    # workflow
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    flow = [
        "1. Upload\ntwo images",
        "2. Confirm\nauthorisation +\nlimitations",
        "3. Detect,\nalign, check\nfaces",
        "4. Embed,\nsimilarity,\nquality",
        "5. Calibrated\nthree-way\ndecision",
        "6. Read result,\nreason, diagnostics,\nheat maps",
    ]
    for k, t in enumerate(flow):
        x = 0.1 + k * 2.0
        box(
            ax,
            x,
            0.5,
            1.7,
            1.6,
            t,
            blue if k not in (1,) else amber,
            blue_e if k not in (1,) else amber_e,
            fs=9.2,
        )
        if k < len(flow) - 1:
            arrow(ax, x + 1.7, 1.3, x + 2.0, 1.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "workflow.png", dpi=170, facecolor="white")
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
def md_to_docx(
    md_path: Path, docx_path: Path, title_page: bool, toc: bool, compact: bool = False
) -> None:
    from docx import Document
    from docx.enum.section import WD_ORIENT  # noqa: F401
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
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(1.9 if compact else 2.3)
    sec.top_margin = sec.bottom_margin = Cm(1.6 if compact else 2.2)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10 if compact else 11)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15
    for lvl, size in ((1, 16), (2, 13), (3, 11.5)):
        h = doc.styles[f"Heading {lvl}"]
        h.font.name = "Calibri"
        h.font.size = Pt(size)
        h.font.color.rgb = RGBColor.from_string(NAVY)
        h.font.bold = True
        h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True

    # footer page numbers
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run(meta.get("title", "").split(":")[0] + "  ·  page ")
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(GREY)
    _field(fp, "PAGE", OxmlElement, qn)

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
                r.font.size = Pt((size or 11) - 1)
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                r = par.add_run(tok[1:-1])
                r.italic = True
            else:
                r = par.add_run(tok)
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = RGBColor.from_string(color)

    if title_page:
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
        p.add_run().add_break(WD_BREAK.PAGE)
    else:
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
    if toc:
        h = doc.add_paragraph()  # not a Heading style, so the TOC does not list itself
        r = h.add_run("Table of Contents")
        r.bold = True
        r.font.size = Pt(16)
        r.font.color.rgb = RGBColor.from_string(NAVY)
        p = doc.add_paragraph()
        _field(p, 'TOC \\o "1-2" \\h \\z \\u', OxmlElement, qn)
        p.add_run().add_break(WD_BREAK.PAGE)
        del h

    lines = text.replace("<!-- TITLEPAGE -->", "").splitlines()
    i, fig_n = 0, 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            for b in block:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.left_indent = Cm(0.6)
                r = p.add_run(b if b else " ")
                r.font.name = "Consolas"
                r.font.size = Pt(9)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", s)
        if m:
            doc.add_paragraph(m.group(2), style=f"Heading {len(m.group(1))}")
            i += 1
            continue
        m = re.match(r"^!\[(.*)\]\((.*)\)$", s)
        if m:
            cap, path = m.group(1), OUT / m.group(2)
            if path.exists():
                from PIL import Image

                w, h = Image.open(path).size
                width = Cm(16.4) if w / h > 1.25 else Cm(min(16.4, 11.0 * w / h))
                doc.add_picture(str(path), width=width)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.paragraphs[
                    -1
                ].paragraph_format.keep_with_next = True  # caption stays with figure
            else:
                p = doc.add_paragraph()
                add_inline(p, f"[MISSING IMAGE: {m.group(2)}]", color="C00000")
            fig_n += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, cap, size=9.5, color=GREY)
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            ncol = max(len(r) for r in rows)
            t = doc.add_table(rows=len(rows), cols=ncol)
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for ri, row in enumerate(rows):
                for ci in range(ncol):
                    cell = t.cell(ri, ci)
                    cell.paragraphs[0].paragraph_format.space_after = Pt(1)
                    add_inline(cell.paragraphs[0], row[ci] if ci < len(row) else "", size=9.5)
                    if ri == 0:
                        for r in cell.paragraphs[0].runs:
                            r.bold = True
                            r.font.color.rgb = RGBColor.from_string("FFFFFF")
                        _shade(cell, NAVY, OxmlElement, qn)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue
        if s.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8)
            p.paragraph_format.right_indent = Cm(0.8)
            add_inline(p, s.lstrip("> ").strip(), color=NAVY)
            i += 1
            while i < len(lines) and lines[i].strip().startswith(">"):
                p.add_run(" ")
                add_inline(p, lines[i].strip().lstrip("> ").strip(), color=NAVY)
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
            p.paragraph_format.space_after = Pt(2)
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
            and not re.match(r"^(#|\||!\[|```|>|\s*([-*]|\d+\.)\s)", lines[i].strip())
        ):
            para += " " + lines[i].strip()
            i += 1
        p = doc.add_paragraph()
        add_inline(p, para)
    doc.core_properties.title = meta.get("title", "")
    doc.core_properties.author = "[Student Name]"
    doc.save(str(docx_path))


def _field(par, instr, OxmlElement, qn) -> None:  # noqa: N803
    r = par.add_run()
    for kind, txt in (
        ("begin", None),
        (None, instr),
        ("separate", None),
        (None, "(right-click → Update Field)"),
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


def _shade(cell, hex_color, OxmlElement, qn) -> None:  # noqa: N803
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


# --------------------------------------------------------------------------- Office PDF export
def office_pdf(docx_or_pptx: Path, pdf: Path) -> None:
    src, dst = str(docx_or_pptx.resolve()), str(pdf.resolve())
    if docx_or_pptx.suffix == ".docx":
        ps = f"""
$w = New-Object -ComObject Word.Application; $w.Visible = $false; $w.DisplayAlerts = 0
try {{ $d = $w.Documents.Open('{src}', $false, $false)
  foreach ($t in $d.TablesOfContents) {{ $t.Update() }}
  $d.Fields.Update() | Out-Null; $d.Save(); $d.ExportAsFixedFormat('{dst}', 17); $d.Close(0) }} finally {{ $w.Quit() }}"""
    else:
        ps = f"""
$p = New-Object -ComObject PowerPoint.Application
try {{ $pr = $p.Presentations.Open('{src}', $true, $false, $false); $pr.SaveAs('{dst}', 32); $pr.Close() }} finally {{ $p.Quit() }}"""
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], check=True, timeout=600
    )


# --------------------------------------------------------------------------- PowerPoint
SLIDES = [
    {
        "title": "VEIL-X",
        "kind": "title",
        "notes": "Introduce yourself and the project. VEIL-X is a local face verification application that can say MATCH, NON-MATCH or ABSTAIN, and explains why. It is a university course project and an academic demonstration, not a certified identity system.",
    },
    {
        "title": "Introduction",
        "bullets": [
            "Face verification: do two photos show the **same person**? (1:1)",
            "Deep CNNs turn a face into a 512-number **embedding**; same person → similar direction",
            "Used in phone unlock, e-gates, photo organisation",
            "VEIL-X: a complete local application around strong **pretrained** models",
        ],
        "image": "workflow.png",
        "notes": "Explain verification versus identification: VEIL-X only compares two images and has no database. Explain embeddings simply. Mention that we use pretrained recognisers; the project's work is the decision layer, abstention, explanations and the tested application.",
    },
    {
        "title": "Problem Statement",
        "bullets": [
            "Occlusion (masks, sunglasses, hands, patches) and low quality break face matching",
            "Our baseline: false non-match rate rose from **1.4 %** (clean) to **6.8–21.6 %** with synthetic occluders",
            "Basic systems must always answer yes / no → **silent errors**",
            "Goal: decide only when evidence is sufficient, explain the result, keep data private",
        ],
        "notes": "Numbers from the Phase 2 baseline: at a fixed clean threshold, the false non-match rate of the frozen AdaFace model rose from 1.4 % to 6.8–21.6 % depending on the unseen occluder (44.9 % for high-severity random patches).",
    },
    {
        "title": "Objectives",
        "bullets": [
            "Two-image verification with pretrained deep models (AdaFace, SCRFD)",
            "Image-quality assessment built into the decision",
            "Three outcomes: **MATCH / NON-MATCH / ABSTAIN** with calibrated thresholds",
            "Explanations: diagnostics, abstention reasons, visibility heat maps",
            "Local, private, tested and reproducible",
        ],
        "notes": "Read the objectives briefly. Emphasise abstention and explanation as the key ideas.",
    },
    {
        "title": "Existing vs Proposed System",
        "table": [
            ["Existing", "VEIL-X (proposed)"],
            ["Single threshold, always yes/no", "Quality-dependent thresholds + ABSTAIN"],
            ["No confidence information", "Similarity, quality, threshold region, reason"],
            ["Often cloud-based", "Runs only on 127.0.0.1; uploads deleted"],
            ["Errors on poor images are silent", "Refuses ambiguous / low-quality inputs"],
            ["Opaque behaviour", "Visibility heat maps + documented limits"],
        ],
        "notes": "Be clear that VEIL-X does not invent a new recognition network; the improvement is the decision layer and the application.",
    },
    {
        "title": "System Architecture",
        "image_full": "architecture.png",
        "notes": "Walk left to right: input checks, SCRFD detection and alignment, AdaFace embedding and feature norm, similarity and quality, logistic fusion M1, quality-stratified A3 rule with the support guard, three outcomes. The dashed boxes (ArcFace similarity, visibility maps, novelty) are explanations computed after the decision and never change it.",
    },
    {
        "title": "Technologies Used",
        "table": [
            ["Area", "Technology"],
            ["Language", "Python 3.12"],
            ["Deep learning", "PyTorch 2.11 + CUDA 12.8 (fp16 on GPU)"],
            ["Recognition", "AdaFace IR-101, ArcFace IR-101 (pretrained, WebFace4M)"],
            ["Detection", "SCRFD-10G via ONNX Runtime"],
            ["Visibility", "U-Net-style network (3.6 M params, trained here)"],
            ["Interface", "Gradio 5 + FastAPI, localhost only"],
            ["Quality", "pytest, Ruff, strict mypy, GitHub Actions"],
            ["Hardware", "RTX 4060 Laptop GPU, Ryzen 9 7940HS"],
        ],
        "notes": "Point out what is pretrained (recognisers, detector) versus what was trained or calibrated in this project (visibility network, logistic formula, thresholds).",
    },
    {
        "title": "How Face Verification Works",
        "bullets": [
            "Detect face + 5 landmarks → align to 112×112",
            "AdaFace → embedding e and feature norm n",
            "Similarity **s = cos(e_A, e_B)**; quality **q = min(n_A, n_B)**",
            "Score **p = logistic(s, q, s·q)**, between 0 and 1",
            "p is a fitted score, **not** a probability of identity",
        ],
        "image": "shot_match_faces.png",
        "notes": "Use the screenshot (synthetic faces) to show the aligned crops. Explain cosine similarity as the angle between embeddings. The feature norm is a quality signal: blurred or covered faces give smaller norms.",
    },
    {
        "title": "Quality Assessment and Abstention",
        "bullets": [
            "Pairs split into low / medium / high quality",
            "Each group has calibrated thresholds; low quality has an **abstention band**",
            "Example: same person, mask on B → score 0.542 inside [0.154, 0.833) → **ABSTAIN**",
            "**Support guard:** below the calibrated quality range → ABSTAIN",
        ],
        "image": "shot_abstain_result.png",
        "notes": "Explain the threshold bar in the screenshot. Thresholds were calibrated with the Learn-then-Test procedure on separate identities, with a 5 % research target for errors among accepted decisions. The support guard exists because random-noise images scored 1.000 under the unguarded research rule.",
    },
    {
        "title": "Visibility Heat Maps",
        "bullets": [
            "U-Net-style network predicts visible (blue) vs covered (orange) face areas",
            "Shown to help users understand occlusions",
            "**Not used for decisions:** on unseen random patches it called covered areas visible (IoU 0.13)",
            "Quality-based rule stayed within target (2.12 %) where visibility-based failed (8.52 %)",
        ],
        "image": "vis_example4.png",
        "notes": "This is an important honest finding: learned visibility did not generalise to new occluder types, so it is used only as an explanation.",
    },
    {
        "title": "Application Interface",
        "image_full": "shot_home.png",
        "notes": "Show the main screen: two uploads, the active configuration card that explains what the calibration is, the options, and the two required confirmations (authorisation and calibration limits).",
    },
    {
        "title": "Implementation and Test Results",
        "table": [
            ["Check", "Result"],
            ["Automated tests", "__TESTS__"],
            ["Parity with research code", "1,000 / 1,000 identical decisions"],
            ["CPU vs GPU", "100 / 100 identical decisions"],
            ["Speed (RTX 4060, warm, median)", "226 ms per photo pair; 43 ms pre-aligned"],
            ["CPU only", "0.3–0.5 s per pair"],
            ["Privacy", "127.0.0.1 only; external requests blocked; uploads deleted"],
        ],
        "notes": "Stress that parity is not accuracy: 1000/1000 means the application reproduces the research system exactly, not that it recognises faces perfectly. Speed measured with one pair per request after warm-up; face detection on CPU dominates.",
    },
    {
        "title": "Advantages and Limitations",
        "two_col": (
            [
                "Abstains instead of guessing",
                "Explains every result",
                "Private, local, fast",
                "Strong pretrained models",
                "Extensively tested",
            ],
            [
                "Synthetic faces and occlusions only",
                "5 % target failed on a harder population (5.2–6.5 %), confirmed twice",
                "Low-quality false matches (22.97 % below range) → guard added",
                "Not certified; not for real identity decisions",
            ],
        ),
        "image": "fig_population_shift.png",
        "notes": "Explain population shift with the chart: the rule met its target on data like its calibration data (blue), but not on a harder group of identities (orange), and an independent repeat confirmed it (green). Calibration must be redone for any new population. Mention the low-quality guard honestly: an application safeguard, not a proven universal fix.",
    },
    {
        "title": "Future Enhancements",
        "bullets": [
            "Re-calibrate on the intended users' photos and camera",
            "Test on real occluded photographs (e.g. IJB-C, after access approval)",
            "Fused score that cannot extrapolate below its quality range",
            "GPU face detection for lower latency",
            "Clickable face selection for group photos",
        ],
        "notes": "Keep this short.",
    },
    {
        "title": "Conclusion and Demo",
        "bullets": [
            "Working local face verification with quality-aware **MATCH / NON-MATCH / ABSTAIN**",
            "Explanations and visibility heat maps; private and tested",
            "Honest about limits: calibration depends on the population",
            "Live demo: MATCH → NON-MATCH → ABSTAIN (mask) → support guard (noise)",
        ],
        "image": "shot_guard_result.png",
        "notes": "Switch to the live demo (see the demo guide). If the demo cannot run, use the screenshots on the previous slides. Thank the audience and invite questions.",
    },
]


def build_pptx(path: Path, tests_text: str) -> None:
    from PIL import Image
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Inches, Pt

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    navy, ink, grey = (
        RGBColor.from_string(NAVY),
        RGBColor.from_string(INK),
        RGBColor.from_string(GREY),
    )

    def text_box(slide, x, y, w, h, paras, size=20, color=ink, bold_first=False):
        tb = slide.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        for k, ptxt in enumerate(paras):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.space_after = Pt(10)
            for j, tok in enumerate(re.split(r"(\*\*[^*]+\*\*)", ptxt)):
                if not tok:
                    continue
                r = p.add_run()
                r.text = tok.strip("*") if tok.startswith("**") else tok
                r.font.size = Pt(size)
                r.font.color.rgb = color
                r.font.bold = tok.startswith("**") or (bold_first and k == 0 and j == 0)
        return tb

    def picture(slide, name, x, y, w, h):
        p = ASSETS / name
        if not p.exists():
            text_box(slide, x, y, w, h, [f"[missing image {name}]"], 14, RGBColor(0xC0, 0, 0))
            return
        iw, ih = Image.open(p).size
        scale = min(w / iw, h / ih)
        pw, ph = int(iw * scale), int(ih * scale)
        pic = slide.shapes.add_picture(str(p), x + (w - pw) // 2, y + (h - ph) // 2, pw, ph)
        pic.line.color.rgb = RGBColor(0xD9, 0xD9, 0xD9)

    for n, spec in enumerate(SLIDES, start=1):
        s = prs.slides.add_slide(blank)
        if spec.get("kind") == "title":
            bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
            bg.fill.solid()
            bg.fill.fore_color.rgb = navy
            bg.line.fill.background()
            text_box(
                s,
                Inches(0.9),
                Inches(1.5),
                Inches(11.5),
                Inches(1.2),
                ["VEIL-X"],
                54,
                RGBColor(0xFF, 0xFF, 0xFF),
                True,
            )
            text_box(
                s,
                Inches(0.9),
                Inches(2.6),
                Inches(11.5),
                Inches(1.4),
                [
                    "An AI-Based Face Verification System with Quality Assessment and Intelligent Abstention"
                ],
                28,
                RGBColor(0xFF, 0xFF, 0xFF),
            )
            text_box(
                s,
                Inches(0.9),
                Inches(4.3),
                Inches(11.5),
                Inches(2.4),
                [
                    "[Student Name]  ·  [Register Number]",
                    "[Course Code and Name]  ·  [Department], [College / University]",
                    "Guide: [Guide / Supervisor Name]  ·  [Academic Year]",
                ],
                18,
                RGBColor(0xDD, 0xE6, 0xF2),
            )
            text_box(
                s,
                Inches(0.9),
                Inches(6.7),
                Inches(11.5),
                Inches(0.5),
                [
                    "University course project · academic demonstration, not a certified identity system"
                ],
                12,
                RGBColor(0xBF, 0xCD, 0xE0),
            )
        else:
            bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), prs.slide_height)
            bar.fill.solid()
            bar.fill.fore_color.rgb = navy
            bar.line.fill.background()
            text_box(
                s,
                Inches(0.55),
                Inches(0.35),
                Inches(12.2),
                Inches(0.9),
                [spec["title"]],
                34,
                navy,
                True,
            )
            line = s.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.22), Inches(1.6), Inches(0.06)
            )
            line.fill.solid()
            line.fill.fore_color.rgb = RGBColor.from_string(ACCENT)
            line.line.fill.background()
            top, height = Inches(1.55), Inches(5.3)
            if "image_full" in spec:
                picture(s, spec["image_full"], Inches(0.6), top, Inches(12.2), height)
            elif "table" in spec:
                rows = [[c.replace("__TESTS__", tests_text) for c in r] for r in spec["table"]]
                tw = Inches(12.0)
                shape = s.shapes.add_table(
                    len(rows), len(rows[0]), Inches(0.65), top, tw, Inches(0.5) * len(rows)
                )
                tbl = shape.table
                for ri, row in enumerate(rows):
                    for ci, val in enumerate(row):
                        cell = tbl.cell(ri, ci)
                        cell.text = val
                        para = cell.text_frame.paragraphs[0]
                        para.runs[0].font.size = Pt(18 if ri else 19)
                        para.runs[0].font.bold = ri == 0
                        para.runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if ri == 0 else ink
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = (
                            navy
                            if ri == 0
                            else (
                                RGBColor(0xF2, 0xF5, 0xFA) if ri % 2 else RGBColor(0xFF, 0xFF, 0xFF)
                            )
                        )
            elif "two_col" in spec:
                left, right = spec["two_col"]
                text_box(
                    s,
                    Inches(0.6),
                    top,
                    Inches(3.3),
                    Inches(0.5),
                    ["Advantages"],
                    22,
                    RGBColor.from_string(ACCENT),
                    True,
                )
                text_box(
                    s,
                    Inches(0.6),
                    top + Inches(0.55),
                    Inches(3.3),
                    Inches(4.6),
                    ["• " + x for x in left],
                    16,
                )
                text_box(
                    s,
                    Inches(4.0),
                    top,
                    Inches(3.5),
                    Inches(0.5),
                    ["Limitations"],
                    22,
                    RGBColor.from_string(AMBER),
                    True,
                )
                text_box(
                    s,
                    Inches(4.0),
                    top + Inches(0.55),
                    Inches(3.5),
                    Inches(4.6),
                    ["• " + x for x in right],
                    16,
                )
                picture(s, spec["image"], Inches(7.6), top + Inches(0.2), Inches(5.5), Inches(4.8))
            else:
                img = spec.get("image")
                bullets = ["• " + b for b in spec.get("bullets", [])]
                wide = False
                if img and (ASSETS / img).exists():
                    iw, ih = Image.open(ASSETS / img).size
                    wide = iw / ih > 1.6
                if (
                    img and wide
                ):  # stacked: bullets on top, wide screenshot / diagram full width below
                    text_box(s, Inches(0.6), top, Inches(12.2), Inches(2.4), bullets, 18)
                    picture(s, img, Inches(0.6), top + Inches(2.45), Inches(12.2), Inches(2.85))
                else:
                    width = Inches(6.6) if img else Inches(12.0)
                    text_box(s, Inches(0.6), top, width, height, bullets, 20)
                    if img:
                        picture(s, img, Inches(7.4), top, Inches(5.6), height)
            text_box(
                s,
                Inches(0.55),
                Inches(7.0),
                Inches(9),
                Inches(0.4),
                ["VEIL-X · University Course Project"],
                11,
                grey,
            )
            text_box(s, Inches(12.2), Inches(7.0), Inches(1.0), Inches(0.4), [str(n)], 11, grey)
        s.notes_slide.notes_text_frame.text = spec["notes"]
    del Emu, PP_ALIGN
    prs.core_properties.title = TITLE
    prs.save(str(path))


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
        md_to_docx(SRC / src, docx, title_page, toc, compact=stem == "VEIL-X_Project_Summary")
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
        (pptx, "15-slide presentation with speaker notes"),
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
