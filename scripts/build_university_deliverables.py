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
        (pptx, "Presentation: 24 slides, native charts, speaker notes"),
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
