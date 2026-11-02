# ruff: noqa: E501, RUF001  (slide text uses typographic characters)
"""Professional university presentation for VEIL-X (python-pptx), used by
``scripts/build_university_deliverables.py``.

Every number on a slide comes from a committed report (paths in the speaker notes and in
``docs/university/report.md``). Charts are native PowerPoint charts, so they remain editable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

FONT = "Segoe UI"
NAVY = RGBColor(0x0F, 0x2A, 0x4A)
BLUE = RGBColor(0x1D, 0x4E, 0x89)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
TEAL = RGBColor(0x1B, 0xAF, 0x7A)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
AMBER = RGBColor(0x9A, 0x5B, 0x00)
SLATE = RGBColor(0x4A, 0x4A, 0x4A)
INK = RGBColor(0x1F, 0x1F, 0x1F)
GREY = RGBColor(0x6B, 0x6B, 0x6B)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
LINE = RGBColor(0xD5, 0xDC, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SW, SH = Inches(13.333), Inches(7.5)
X0, CW = Inches(0.6), Inches(12.13)  # content left, content width
Y0, Y1 = Inches(1.6), Inches(6.85)  # content top, bottom


class Deck:
    def __init__(self, assets: Path, tests_text: str) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = SW, SH
        self.assets = assets
        self.tests_text = tests_text
        self.n = 0

    # ------------------------------------------------------------------ primitives
    def text(
        self,
        s: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        paras: list[Any],
        size: float = 16,
        color: RGBColor = INK,
        bold: bool = False,
        align: Any = PP_ALIGN.LEFT,
        anchor: Any = MSO_ANCHOR.TOP,
        spacing: float = 6,
    ) -> Any:
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.05)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        self._fill(tf, paras, size, color, bold, align, spacing)
        return tb

    def _fill(
        self,
        tf: Any,
        paras: list[Any],
        size: float,
        color: RGBColor,
        bold: bool,
        align: Any,
        spacing: float,
    ) -> None:
        for k, spec in enumerate(_expand(paras)):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = spec.get("align", align)
            p.space_after = Pt(spec.get("after", spacing))
            p.level = spec.get("level", 0)
            txt = spec["t"]
            if spec.get("bullet"):
                txt = ("•  " if spec.get("level", 0) == 0 else "–  ") + txt
            for j, tok in enumerate(_tokens(txt)):
                r = p.add_run()
                r.text = tok[0]
                f = r.font
                f.name = FONT
                f.size = Pt(spec.get("size", size))
                f.bold = tok[1] or spec.get("bold", bold)
                f.color.rgb = spec.get("color", color)
                del j

    def box(
        self,
        s: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        fill: RGBColor | None = LIGHT,
        line: RGBColor | None = None,
        shape: Any = MSO_SHAPE.ROUNDED_RECTANGLE,
        radius: float = 0.08,
        dash: bool = False,
    ) -> Any:
        sp = s.shapes.add_shape(shape, x, y, w, h)
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            sp.adjustments[0] = radius
        if fill is None:
            sp.fill.background()
        else:
            sp.fill.solid()
            sp.fill.fore_color.rgb = fill
        if line is None:
            sp.line.fill.background()
        else:
            sp.line.color.rgb = line
            sp.line.width = Pt(1.25)
            if dash:
                sp.line.dash_style = 4  # MSO_LINE_DASH_STYLE.DASH
        sp.shadow.inherit = False
        return sp

    def card(
        self,
        s: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        body: list[Any],
        color: RGBColor = ACCENT,
        size: float = 13.5,
        title_size: float = 16,
        fill: RGBColor = LIGHT,
    ) -> None:
        self.box(s, x, y, w, h, fill=fill)
        self.box(s, x, y, Inches(0.09), h, fill=color, shape=MSO_SHAPE.RECTANGLE)
        self.text(
            s,
            x + Inches(0.22),
            y + Inches(0.12),
            w - Inches(0.35),
            Inches(0.5),
            [title],
            title_size,
            color,
            True,
        )
        self.text(
            s,
            x + Inches(0.22),
            y + Inches(0.6),
            w - Inches(0.35),
            h - Inches(0.7),
            body,
            size,
            INK,
            spacing=4,
        )

    def kpi(
        self,
        s: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        value: str,
        label: str,
        color: RGBColor = BLUE,
        sub: str = "",
    ) -> None:
        self.box(s, x, y, w, h, fill=LIGHT)
        self.box(s, x, y, w, Inches(0.07), fill=color, shape=MSO_SHAPE.RECTANGLE)
        self.text(s, x, y + Inches(0.18), w, Inches(0.8), [value], 30, color, True, PP_ALIGN.CENTER)
        self.text(
            s,
            x + Inches(0.15),
            y + Inches(0.95),
            w - Inches(0.3),
            h - Inches(1.0),
            [{"t": label, "bold": True, "size": 13}]
            + ([{"t": sub, "size": 11, "color": GREY}] if sub else []),
            13,
            INK,
            align=PP_ALIGN.CENTER,
            spacing=2,
        )

    def chip(
        self, s: Any, x: int, y: int, w: int, h: int, label: str, color: RGBColor, size: float = 14
    ) -> None:
        sp = self.box(s, x, y, w, h, fill=color, radius=0.5)
        tf = sp.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.word_wrap = False
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        self._fill(
            tf, [{"t": label, "align": PP_ALIGN.CENTER}], size, WHITE, True, PP_ALIGN.CENTER, 0
        )

    def arrow(
        self, s: Any, x1: int, y1: int, x2: int, y2: int, color: RGBColor = GREY, dash: bool = False
    ) -> None:
        c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
        c.line.color.rgb = color
        c.line.width = Pt(1.75)
        if dash:
            c.line.dash_style = 4
        ln = c.line._get_or_add_ln()
        tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(tail)

    def picture(
        self, s: Any, name: str, x: int, y: int, w: int, h: int, border: bool = True
    ) -> tuple[int, int, int, int]:
        p = self.assets / name
        iw, ih = Image.open(p).size
        sc = min(w / iw, h / ih)
        pw, ph = int(iw * sc), int(ih * sc)
        px, py = x + (w - pw) // 2, y + (h - ph) // 2
        pic = s.shapes.add_picture(str(p), px, py, pw, ph)
        if border:
            pic.line.color.rgb = LINE
            pic.line.width = Pt(1)
        return px, py, pw, ph

    def table(
        self,
        s: Any,
        x: int,
        y: int,
        w: int,
        rows: list[list[str]],
        col_w: list[float] | None = None,
        size: float = 13,
        row_h: float = 0.42,
        head: RGBColor = BLUE,
    ) -> None:
        shape = s.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h) * len(rows))
        tbl = shape.table
        tbl.first_row = True
        if col_w:
            for k, cw in enumerate(col_w):
                tbl.columns[k].width = Inches(cw)
        for ri, row in enumerate(rows):
            tbl.rows[ri].height = Inches(row_h)
            for ci, val in enumerate(row):
                cell = tbl.cell(ri, ci)
                cell.margin_left = cell.margin_right = Inches(0.1)
                cell.margin_top = cell.margin_bottom = Inches(0.04)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                tf.word_wrap = True
                tf.paragraphs[0].text = ""
                self._fill(tf, [val], size, WHITE if ri == 0 else INK, ri == 0, PP_ALIGN.LEFT, 0)
                cell.fill.solid()
                cell.fill.fore_color.rgb = head if ri == 0 else (LIGHT if ri % 2 else WHITE)

    def chart(
        self,
        s: Any,
        kind: Any,
        x: int,
        y: int,
        w: int,
        h: int,
        cats: list[str],
        series: list[tuple[str, list[float], RGBColor]],
        fmt: str = '0.0"%"',
        vmax: float | None = None,
        legend: bool = True,
        title: str | None = None,
        label_size: float = 11,
        tick_fmt: str | None = None,
    ) -> Any:
        cd = CategoryChartData()
        cd.categories = cats
        for name, vals, _ in series:
            cd.add_series(name, vals)
        ch = s.shapes.add_chart(kind, x, y, w, h, cd).chart
        ch.font.name = FONT
        ch.font.size = Pt(12)
        ch.font.color.rgb = SLATE
        ch.has_legend = legend and len(series) > 1
        if ch.has_legend:
            ch.legend.position = XL_LEGEND_POSITION.TOP
            ch.legend.include_in_layout = False
            ch.legend.font.size = Pt(12)
        ch.has_title = bool(title)
        if title:
            ch.chart_title.text_frame.text = title
            ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
            ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
        plot = ch.plots[0]
        plot.gap_width = 60
        if hasattr(plot, "overlap") and len(series) > 1:
            plot.overlap = -10
        plot.has_data_labels = True
        dl = plot.data_labels
        dl.number_format = fmt
        dl.number_format_is_linked = False
        dl.font.size = Pt(label_size)
        dl.font.color.rgb = INK
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
        for (_, _, col), ser in zip(series, plot.series, strict=True):
            ser.format.fill.solid()
            ser.format.fill.fore_color.rgb = col
        va = ch.value_axis
        va.has_major_gridlines = True
        va.major_gridlines.format.line.color.rgb = RGBColor(0xE6, 0xE9, 0xEE)
        va.format.line.fill.background()
        va.tick_labels.font.size = Pt(11)
        va.tick_labels.number_format = tick_fmt or fmt.replace("0.00", "0").replace("0.0", "0")
        va.tick_labels.number_format_is_linked = False
        if vmax is not None:
            va.maximum_scale = vmax
        va.minimum_scale = 0
        ca = ch.category_axis
        ca.tick_labels.font.size = Pt(12)
        ca.format.line.color.rgb = LINE
        return ch

    # ------------------------------------------------------------------ slide frames
    def slide(self, kicker: str, title: str, notes: str) -> Any:
        self.n += 1
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.box(s, 0, 0, SW, Inches(0.12), fill=NAVY, shape=MSO_SHAPE.RECTANGLE)
        self.text(s, X0, Inches(0.3), CW, Inches(0.35), [kicker.upper()], 12, ACCENT, True)
        self.text(s, X0, Inches(0.58), CW, Inches(0.75), [title], 30, NAVY, True)
        self.box(
            s,
            X0 + Inches(0.03),
            Inches(1.32),
            Inches(1.1),
            Inches(0.05),
            fill=ACCENT,
            shape=MSO_SHAPE.RECTANGLE,
        )
        self.box(s, X0, Inches(7.0), CW, Emu(9525), fill=LINE, shape=MSO_SHAPE.RECTANGLE)
        self.text(
            s,
            X0,
            Inches(7.05),
            Inches(8),
            Inches(0.35),
            [
                "VEIL-X  |  AI-Based Face Verification with Quality Assessment and Intelligent Abstention"
            ],
            10,
            GREY,
        )
        self.text(
            s,
            X0 + CW - Inches(1.0),
            Inches(7.05),
            Inches(1.0),
            Inches(0.35),
            [str(self.n)],
            10,
            GREY,
            align=PP_ALIGN.RIGHT,
        )
        s.notes_slide.notes_text_frame.text = notes
        return s

    def save(self, path: Path) -> None:
        self.prs.core_properties.title = "VEIL-X: An AI-Based Face Verification System with Quality Assessment and Intelligent Abstention"
        self.prs.save(str(path))


def _tokens(t: str) -> list[tuple[str, bool]]:
    """Split on ** markers: odd segments are bold."""
    return [(part, k % 2 == 1) for k, part in enumerate(t.split("**")) if part]


def _expand(paras: list[Any]) -> list[Any]:
    """Turn embedded line breaks into separate paragraphs (runs cannot hold line breaks)."""
    out: list[Any] = []
    for p_ in paras:
        spec = dict(p_) if isinstance(p_, dict) else {"t": p_}
        lines = spec["t"].splitlines() or [""]
        for k, line in enumerate(lines):
            item = dict(spec, t=line)
            if k < len(lines) - 1:
                item["after"] = 0
            if k > 0:
                item.pop("bullet", None)
            out.append(item)
    return out


# ====================================================================== the deck
def build(path: Path, assets: Path, tests_text: str) -> None:
    d = Deck(assets, tests_text)

    # 1 ── title
    d.n += 1
    s = d.prs.slides.add_slide(d.prs.slide_layouts[6])
    d.box(s, 0, 0, SW, SH, fill=NAVY, shape=MSO_SHAPE.RECTANGLE)
    d.box(s, 0, 0, Inches(0.25), SH, fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
    d.text(
        s,
        Inches(0.9),
        Inches(0.8),
        Inches(8),
        Inches(0.4),
        ["UNIVERSITY COURSE PROJECT"],
        13,
        RGBColor(0x8F, 0xB8, 0xEC),
        True,
    )
    d.text(s, Inches(0.9), Inches(1.25), Inches(11.5), Inches(1.2), ["VEIL-X"], 66, WHITE, True)
    d.text(
        s,
        Inches(0.9),
        Inches(2.5),
        Inches(11.2),
        Inches(1.3),
        ["An AI-Based Face Verification System with Quality Assessment and Intelligent Abstention"],
        28,
        WHITE,
    )
    for k, (lab, col) in enumerate((("MATCH", BLUE), ("NON-MATCH", SLATE), ("ABSTAIN", AMBER))):
        d.chip(s, Inches(0.9 + 2.3 * k), Inches(4.05), Inches(2.1), Inches(0.55), lab, col, 15)
        s.shapes[-1].line.color.rgb = WHITE
        s.shapes[-1].line.width = Pt(1.5)
    d.text(
        s,
        Inches(0.9),
        Inches(4.95),
        Inches(11),
        Inches(1.4),
        [
            {"t": "[Student Name]  ·  [Register Number]", "size": 17},
            {
                "t": "[Course Code and Name]  ·  [Department], [College / University Name]",
                "size": 15,
            },
            {"t": "Guide: [Guide / Supervisor Name]  ·  [Academic Year]", "size": 15},
        ],
        15,
        RGBColor(0xDD, 0xE6, 0xF2),
        spacing=4,
    )
    d.text(
        s,
        Inches(0.9),
        Inches(6.85),
        Inches(11.5),
        Inches(0.4),
        ["Academic demonstration · runs locally · not a certified identity-verification system"],
        11,
        RGBColor(0x8F, 0xA6, 0xC4),
    )
    s.notes_slide.notes_text_frame.text = "Introduce yourself and the project. VEIL-X compares two face photos on a laptop and answers MATCH, NON-MATCH or ABSTAIN, and explains its answer. Emphasise up front: academic demonstration, not a certified identity system."

    # 2 ── agenda
    s = d.slide(
        "Overview",
        "Agenda",
        "Walk through the structure: the problem and objectives, how the system is built, how it decides, how it was tested, what we learned (including honest negative findings), the live demo, and future work.",
    )
    items = [
        "Problem and motivation",
        "Objectives",
        "Existing vs proposed system",
        "System architecture",
        "Technologies used",
        "How verification works",
        "Quality assessment and abstention",
        "Visibility heat maps",
        "Data and methodology",
        "Application and demo results",
        "Testing and performance",
        "Key findings",
        "Limitations and future work",
        "Conclusion",
    ]
    for k, it in enumerate(items):
        col, row = k // 7, k % 7
        x, y = X0 + Inches(6.1) * col, Y0 + Inches(0.05) + Inches(0.72) * row
        d.chip(s, x, y, Inches(0.55), Inches(0.5), f"{k + 1:02d}", BLUE if col == 0 else ACCENT, 13)
        d.text(s, x + Inches(0.7), y + Inches(0.06), Inches(5.2), Inches(0.45), [it], 18, INK)

    # 3 ── introduction
    s = d.slide(
        "01 · Introduction",
        "What is face verification?",
        "Verification is 1:1: are these two photos the same person? Identification is 1:N: who is this, among a database? VEIL-X only does verification and has no database (privacy by design). Explain embeddings: a CNN maps a face to 512 numbers; same person gives a similar direction; cosine similarity measures that.",
    )
    d.card(
        s,
        X0,
        Y0,
        Inches(5.9),
        Inches(2.35),
        "Verification  (1 : 1)  ← VEIL-X",
        [
            "Do these **two photos** show the same person?",
            "Examples: phone unlock, passport e-gates, KYC photo checks.",
        ],
        BLUE,
    )
    d.card(
        s,
        X0 + Inches(6.23),
        Y0,
        Inches(5.9),
        Inches(2.35),
        "Identification  (1 : N)  ✗ not in VEIL-X",
        [
            "Who is this person, searched among **many** people?",
            "Needs a database of faces: deliberately excluded for privacy.",
        ],
        SLATE,
    )
    d.text(
        s,
        X0,
        Y0 + Inches(2.65),
        CW,
        Inches(0.45),
        ["How modern systems compare faces"],
        18,
        NAVY,
        True,
    )
    steps = [
        ("Face image", "photo of a person"),
        ("Deep CNN", "pretrained network\n(AdaFace IR-101)"),
        ("Embedding", "512 numbers that\ndescribe the face"),
        ("Cosine similarity", "same person →\nsimilar direction"),
    ]
    for k, (t, b) in enumerate(steps):
        x = X0 + Inches(3.05) * k
        sp = d.box(s, x, Y0 + Inches(3.2), Inches(2.6), Inches(1.75), fill=WHITE, line=LINE)
        d.text(
            s, x, Y0 + Inches(3.35), Inches(2.6), Inches(0.5), [t], 17, BLUE, True, PP_ALIGN.CENTER
        )
        d.text(
            s,
            x,
            Y0 + Inches(3.9),
            Inches(2.6),
            Inches(1.0),
            [b],
            13.5,
            SLATE,
            align=PP_ALIGN.CENTER,
        )
        del sp
        if k < 3:
            d.arrow(
                s, x + Inches(2.62), Y0 + Inches(4.07), x + Inches(3.03), Y0 + Inches(4.07), ACCENT
            )

    # 4 ── problem
    s = d.slide(
        "02 · Problem",
        "Occlusion and low quality break face matching",
        "Numbers from our own Phase 2 baseline with the frozen AdaFace model on synthetic DigiFace faces (reports/experiments/phase2_final.json): at the fixed clean threshold, false non-match rate is 1.4 % on clean pairs, rises to 6.8 % to 21.6 % when an unseen occluder is applied, and reaches 44.9 % for high-severity random patches. Ordinary systems would still answer yes or no, so these errors are silent.",
    )
    d.kpi(
        s,
        X0,
        Y0,
        Inches(3.8),
        Inches(1.9),
        "1.4 %",
        "False non-match, clean faces",
        BLUE,
        "fixed clean threshold",
    )
    d.kpi(
        s,
        X0 + Inches(4.17),
        Y0,
        Inches(3.8),
        Inches(1.9),
        "6.8 – 21.6 %",
        "With an unseen occluder",
        ORANGE,
        "depending on occluder type",
    )
    d.kpi(
        s,
        X0 + Inches(8.33),
        Y0,
        Inches(3.8),
        Inches(1.9),
        "44.9 %",
        "High-severity random patch",
        AMBER,
        "unseen occluder, worst case",
    )
    d.text(s, X0, Y0 + Inches(2.25), CW, Inches(0.45), ["Why this matters"], 18, NAVY, True)
    d.text(
        s,
        X0,
        Y0 + Inches(2.75),
        Inches(7.2),
        Inches(2.5),
        [
            {
                "t": "Masks, sunglasses, hands, objects, poor lighting and blur remove identity evidence",
                "bullet": True,
            },
            {
                "t": "A basic matcher **must** answer yes / no, so errors on weak evidence are **silent**",
                "bullet": True,
            },
            {"t": "Users get no indication of **how reliable** an answer is", "bullet": True},
            {"t": "Cloud services also raise **privacy** concerns for face images", "bullet": True},
        ],
        16,
    )
    d.card(
        s,
        X0 + Inches(7.6),
        Y0 + Inches(2.3),
        Inches(4.53),
        Inches(2.85),
        "Problem statement",
        [
            "Build a **local** face verification application that measures image quality, **abstains** when evidence is insufficient, **explains** its outputs, and honestly reports how reliable it is."
        ],
        BLUE,
        15,
    )

    # 5 ── objectives
    s = d.slide(
        "03 · Objectives",
        "Project objectives",
        "Six objectives. Stress that the novelty for a course project is the complete, tested decision layer and application around strong pretrained models, not a new recognition network.",
    )
    objs = [
        ("Verify", "Two-image face verification with pretrained deep models (SCRFD + AdaFace)"),
        ("Assess quality", "Measure image quality and build it into the decision"),
        ("Decide in three ways", "MATCH / NON-MATCH / ABSTAIN with calibrated thresholds"),
        ("Explain", "Similarity, quality, threshold region, reasons and visibility heat maps"),
        ("Protect privacy", "Run 100 % locally: no cloud, no stored images or embeddings"),
        (
            "Test and document",
            "Automated tests, parity checks, speed measurement, honest limitations",
        ),
    ]
    for k, (t, b) in enumerate(objs):
        col, row = k % 3, k // 3
        x, y = X0 + Inches(4.1) * col, Y0 + Inches(2.65) * row
        d.box(s, x, y, Inches(3.85), Inches(2.4), fill=LIGHT)
        d.chip(
            s,
            x + Inches(0.25),
            y + Inches(0.25),
            Inches(0.6),
            Inches(0.6),
            str(k + 1),
            [BLUE, ACCENT, AMBER, TEAL, SLATE, NAVY][k],
            16,
        )
        d.text(s, x + Inches(1.0), y + Inches(0.3), Inches(2.75), Inches(0.5), [t], 18, NAVY, True)
        d.text(s, x + Inches(0.25), y + Inches(1.05), Inches(3.4), Inches(1.3), [b], 14.5, INK)

    # 6 ── existing vs proposed
    s = d.slide(
        "04 · Existing vs proposed",
        "How VEIL-X differs from a basic face matcher",
        "Be clear: VEIL-X does not invent a new face-recognition network. It adds calibrated, quality-aware three-way decisions, abstention, explanations, privacy-preserving local serving and a tested application around state-of-the-art pretrained models.",
    )
    d.table(
        s,
        X0,
        Y0,
        CW,
        [
            ["Aspect", "Basic face matcher", "VEIL-X"],
            [
                "Decision",
                "One threshold, always yes / no",
                "Quality-dependent thresholds; MATCH / NON-MATCH / **ABSTAIN**",
            ],
            [
                "Image quality",
                "Ignored or checked separately",
                "Built into the score and the threshold choice",
            ],
            [
                "Explanation",
                "A single number",
                "Similarity, quality, threshold bar, abstention reason, heat maps",
            ],
            [
                "Ambiguous input",
                "Picks a face silently",
                "Refuses several faces unless one is selected",
            ],
            [
                "Very poor input",
                "Still answers",
                "Support guard abstains below the calibrated quality range",
            ],
            [
                "Privacy",
                "Often a cloud API",
                "Runs only on 127.0.0.1; uploads deleted; nothing stored",
            ],
            [
                "Evidence",
                "Rarely documented",
                "400+ tests, exact parity, measured speed, published limitations",
            ],
        ],
        [2.2, 4.3, 5.63],
        14,
        0.6,
    )

    # 7 ── architecture (native)
    s = d.slide(
        "05 · Architecture",
        "System architecture",
        "Walk left to right along the decision path: input checks, SCRFD detection and alignment, AdaFace embedding and feature norm, similarity s and quality q, logistic fusion M1, the quality-stratified rule A3 with the support guard, and the three outcomes. The grey dashed row is computed after the decision and never changes it: ArcFace similarity, the visibility heat maps and the novelty flag. Model files are checked by SHA-256 at load.",
    )
    d.text(
        s,
        X0,
        Y0 - Inches(0.05),
        CW,
        Inches(0.35),
        ["DECISION PATH  —  frozen models and calibrated rules"],
        12,
        BLUE,
        True,
    )
    pipe_steps = [
        ("Upload", "2 images"),
        ("Validate", "format, size,\ncorruption"),
        ("Detect + align", "SCRFD, 5 points\n→ 112×112"),
        ("Embed", "AdaFace IR-101\n+ feature norm"),
        ("Compare", "similarity s\nquality q"),
        ("Score", "M1 logistic\np(s, q, s·q)"),
        ("Decide", "A3 thresholds\n+ support guard"),
    ]
    bw, gap = Inches(1.5), Inches(0.27)
    for k, (t, b) in enumerate(pipe_steps):
        x = X0 + (bw + gap) * k
        last = k == len(pipe_steps) - 1
        d.box(
            s,
            x,
            Y0 + Inches(0.4),
            bw,
            Inches(1.55),
            fill=BLUE if last else LIGHT,
            line=None if last else LINE,
        )
        d.text(
            s,
            x,
            Y0 + Inches(0.52),
            bw,
            Inches(0.45),
            [t],
            15,
            WHITE if last else BLUE,
            True,
            PP_ALIGN.CENTER,
        )
        d.text(
            s,
            x,
            Y0 + Inches(0.98),
            bw,
            Inches(0.95),
            [b],
            12,
            WHITE if last else SLATE,
            align=PP_ALIGN.CENTER,
        )
        if not last:
            d.arrow(
                s,
                x + bw + Inches(0.02),
                Y0 + Inches(1.17),
                x + bw + gap - Inches(0.02),
                Y0 + Inches(1.17),
                ACCENT,
            )
    xd = X0 + (bw + gap) * 6
    d.arrow(s, xd + bw // 2, Y0 + Inches(1.97), xd + bw // 2, Y0 + Inches(2.35), BLUE)
    for k, (lab, col) in enumerate((("MATCH", BLUE), ("NON-MATCH", SLATE), ("ABSTAIN", AMBER))):
        d.chip(
            s,
            xd - Inches(0.1),
            Y0 + Inches(2.4) + Inches(0.5) * k,
            bw + Inches(0.2),
            Inches(0.42),
            lab,
            col,
            12,
        )
    d.box(
        s,
        X0 + (bw + gap) * 2,
        Y0 + Inches(2.35),
        bw,
        Inches(0.95),
        fill=RGBColor(0xFD, 0xF1, 0xDC),
        line=AMBER,
    )
    d.text(
        s,
        X0 + (bw + gap) * 2,
        Y0 + Inches(2.4),
        bw,
        Inches(0.9),
        ["Refuses:", "several faces · no reliable face"],
        11.5,
        AMBER,
        align=PP_ALIGN.CENTER,
        spacing=1,
    )
    d.arrow(
        s,
        X0 + (bw + gap) * 2 + bw // 2,
        Y0 + Inches(1.97),
        X0 + (bw + gap) * 2 + bw // 2,
        Y0 + Inches(2.33),
        AMBER,
    )
    d.text(
        s,
        X0,
        Y0 + Inches(3.55),
        CW,
        Inches(0.35),
        ["EXPLANATIONS ONLY  —  computed after the decision, never change it"],
        12,
        GREY,
        True,
    )
    diags = [
        ("ArcFace IR-101", "second similarity\n(for comparison)"),
        ("Visibility network", "U-Net-style heat map\nvisible vs covered"),
        ("Novelty score", "flags inputs unlike\nthe training data"),
    ]
    for k, (t, b) in enumerate(diags):
        x = X0 + Inches(2.6) + Inches(2.35) * k
        d.box(s, x, Y0 + Inches(3.95), Inches(2.15), Inches(1.15), fill=WHITE, line=GREY, dash=True)
        d.text(
            s, x, Y0 + Inches(4.0), Inches(2.15), Inches(0.4), [t], 14, SLATE, True, PP_ALIGN.CENTER
        )
        d.text(
            s,
            x,
            Y0 + Inches(4.4),
            Inches(2.15),
            Inches(0.7),
            [b],
            11.5,
            GREY,
            align=PP_ALIGN.CENTER,
        )
    d.box(s, X0, Y0 + Inches(3.95), Inches(2.35), Inches(1.15), fill=WHITE, line=LINE)
    d.text(
        s,
        X0,
        Y0 + Inches(4.03),
        Inches(2.35),
        Inches(1.05),
        [
            {"t": "Integrity", "bold": True, "color": NAVY, "size": 13},
            "every model file checked by SHA-256 at load",
        ],
        11.5,
        SLATE,
        align=PP_ALIGN.CENTER,
        spacing=2,
    )
    d.box(
        s,
        X0 + CW - Inches(2.6),
        Y0 + Inches(3.95),
        Inches(2.6),
        Inches(1.15),
        fill=WHITE,
        line=LINE,
    )
    d.text(
        s,
        X0 + CW - Inches(2.6),
        Y0 + Inches(4.03),
        Inches(2.6),
        Inches(1.05),
        [
            {"t": "Local web interface", "bold": True, "color": NAVY, "size": 13},
            "Gradio on 127.0.0.1 only; no cloud, no storage",
        ],
        11.5,
        SLATE,
        align=PP_ALIGN.CENTER,
        spacing=2,
    )

    # 8 ── technologies (stack)
    s = d.slide(
        "06 · Technologies",
        "Technology stack",
        "Point out what is pretrained (AdaFace, ArcFace, SCRFD) versus what this project built or trained: the visibility network, the logistic fusion, the calibrated thresholds and the whole application, server and test suite.",
    )
    layers = [
        (
            "Interface",
            "Gradio 5 web UI · FastAPI · Uvicorn on 127.0.0.1 · Content-Security-Policy",
            ACCENT,
        ),
        (
            "Application",
            "Python 3.12 · validation · decision logic · diagnostics · temp-file cleanup",
            BLUE,
        ),
        (
            "Models (pretrained)",
            "AdaFace IR-101 · ArcFace IR-101 (WebFace4M) · SCRFD-10G detector",
            NAVY,
        ),
        (
            "Trained in this project",
            "Visibility network (U-Net style, 3.6 M parameters) · logistic fusion · calibrated thresholds",
            TEAL,
        ),
        ("Compute", "PyTorch 2.11 + CUDA 12.8 (fp16) · ONNX Runtime · OpenCV · NumPy", SLATE),
        (
            "Engineering",
            "pytest (400+ tests) · Ruff · strict mypy · privacy scanner · GitHub Actions CI",
            AMBER,
        ),
    ]
    for k, (t, b, col) in enumerate(layers):
        y = Y0 + Inches(0.86) * k
        d.box(s, X0, y, Inches(3.2), Inches(0.72), fill=col, radius=0.15)
        d.text(
            s,
            X0,
            y,
            Inches(3.2),
            Inches(0.72),
            [t],
            16,
            WHITE,
            True,
            PP_ALIGN.CENTER,
            MSO_ANCHOR.MIDDLE,
        )
        d.box(s, X0 + Inches(3.35), y, CW - Inches(3.35), Inches(0.72), fill=LIGHT, radius=0.15)
        d.text(
            s,
            X0 + Inches(3.55),
            y,
            CW - Inches(3.6),
            Inches(0.72),
            [b],
            15,
            INK,
            anchor=MSO_ANCHOR.MIDDLE,
        )

    # 9 ── how it works
    s = d.slide(
        "07 · How it works",
        "From two photos to one decision",
        "Explain the four formulas. The fused score p is the output of a fitted logistic formula; it is NOT a probability that the two people are the same. Show the aligned crops in the screenshot (synthetic faces, generated locally).",
    )
    stepsx = [
        (
            "1",
            "Detect & align",
            "SCRFD finds the face and 5 landmarks; the face is warped to a standard 112×112 crop.",
        ),
        (
            "2",
            "Embed",
            "AdaFace turns each crop into a 512-number embedding e and a feature norm n (quality).",
        ),
        ("3", "Compare", "**s** = cos(e_A, e_B)\n**q** = min(n_A, n_B)"),
        (
            "4",
            "Score & decide",
            "**p** = logistic(s, q, s·q)\nthresholds of the pair's quality group",
        ),
    ]
    for k, (num, t, b) in enumerate(stepsx):
        y = Y0 + Inches(1.3) * k
        d.chip(s, X0, y + Inches(0.1), Inches(0.62), Inches(0.62), num, BLUE, 17)
        d.text(s, X0 + Inches(0.8), y, Inches(5.4), Inches(0.45), [t], 18, NAVY, True)
        d.text(
            s,
            X0 + Inches(0.8),
            y + Inches(0.43),
            Inches(5.3),
            Inches(0.85),
            [b],
            14,
            INK,
            spacing=2,
        )
    d.picture(s, "shot_match_faces.png", X0 + Inches(6.4), Y0, Inches(5.7), Inches(2.7))
    d.box(
        s,
        X0 + Inches(6.4),
        Y0 + Inches(3.0),
        Inches(5.73),
        Inches(2.1),
        fill=RGBColor(0xFD, 0xF1, 0xDC),
    )
    d.text(
        s,
        X0 + Inches(6.6),
        Y0 + Inches(3.12),
        Inches(5.4),
        Inches(1.95),
        [
            {"t": "Important", "bold": True, "color": AMBER, "size": 16},
            "The fused score p is a calibrated model output between 0 and 1. A score of 1.000 is **not** a 100 % probability that two images show the same person.",
        ],
        14.5,
        INK,
        spacing=4,
    )

    # 10 ── quality assessment
    s = d.slide(
        "08 · Quality assessment",
        "Quality decides which thresholds apply",
        "The feature norm of the AdaFace embedding is a built-in quality signal: clear faces give larger norms. The pair is assigned to a low, medium or high quality group (tertiles fixed on validation data). Each group has its own thresholds calibrated with the Learn-then-Test procedure. Only the low-quality group has an abstention band; medium and high use a single threshold. Values shown are the default frozen system F1_mask (configs and reports/experiments/phase3 freeze).",
    )
    groups = [
        ("Low quality", "q < 19.9", 0.154, 0.833, AMBER),
        ("Medium quality", "19.9 ≤ q < 22.0", 0.452, 0.452, ACCENT),
        ("High quality", "q ≥ 22.0", 0.673, 0.673, BLUE),
    ]
    bx, bwid = X0 + Inches(3.4), Inches(7.8)
    for k, (t, rng, lo, hi, col) in enumerate(groups):
        y = Y0 + Inches(0.15) + Inches(1.15) * k
        d.text(s, X0, y, Inches(3.2), Inches(0.45), [t], 17, col, True)
        d.text(s, X0, y + Inches(0.42), Inches(3.2), Inches(0.4), [rng], 13, GREY)
        d.box(
            s,
            bx,
            y + Inches(0.12),
            int(bwid * lo),
            Inches(0.5),
            fill=RGBColor(0xC9, 0xC9, 0xC9),
            shape=MSO_SHAPE.RECTANGLE,
        )
        if hi > lo:
            d.box(
                s,
                bx + int(bwid * lo),
                y + Inches(0.12),
                int(bwid * (hi - lo)),
                Inches(0.5),
                fill=RGBColor(0xF2, 0xC9, 0x8C),
                shape=MSO_SHAPE.RECTANGLE,
            )
        d.box(
            s,
            bx + int(bwid * hi),
            y + Inches(0.12),
            int(bwid * (1 - hi)),
            Inches(0.5),
            fill=RGBColor(0x9C, 0xBF, 0xEA),
            shape=MSO_SHAPE.RECTANGLE,
        )
        d.text(
            s,
            bx,
            y + Inches(0.15),
            int(bwid * lo),
            Inches(0.45),
            ["NON-MATCH"],
            11,
            SLATE,
            True,
            PP_ALIGN.CENTER,
            MSO_ANCHOR.MIDDLE,
        )
        if hi > lo:
            d.text(
                s,
                bx + int(bwid * lo),
                y + Inches(0.15),
                int(bwid * (hi - lo)),
                Inches(0.45),
                [f"ABSTAIN  [{lo:.3f}, {hi:.3f})"],
                11,
                AMBER,
                True,
                PP_ALIGN.CENTER,
                MSO_ANCHOR.MIDDLE,
            )
        d.text(
            s,
            bx + int(bwid * hi),
            y + Inches(0.15),
            int(bwid * (1 - hi)),
            Inches(0.45),
            ["MATCH"],
            11,
            BLUE,
            True,
            PP_ALIGN.CENTER,
            MSO_ANCHOR.MIDDLE,
        )
        d.text(
            s,
            bx + int(bwid * hi) - Inches(0.6),
            y + Inches(0.62),
            Inches(1.2),
            Inches(0.3),
            [f"{hi:.3f}"],
            10,
            GREY,
            align=PP_ALIGN.CENTER,
        )
    d.text(
        s,
        bx - Inches(0.2),
        Y0 + Inches(3.5),
        Inches(0.4),
        Inches(0.3),
        ["0"],
        11,
        GREY,
        align=PP_ALIGN.CENTER,
    )
    d.text(
        s,
        bx,
        Y0 + Inches(3.5),
        bwid,
        Inches(0.3),
        ["fused score p  →"],
        11,
        GREY,
        align=PP_ALIGN.CENTER,
    )
    d.text(
        s,
        bx + bwid - Inches(0.2),
        Y0 + Inches(3.5),
        Inches(0.4),
        Inches(0.3),
        ["1"],
        11,
        GREY,
        align=PP_ALIGN.CENTER,
    )
    d.card(
        s,
        X0,
        Y0 + Inches(4.0),
        Inches(5.9),
        Inches(1.25),
        "Quality signal",
        ["Feature norm of the AdaFace embedding; pair quality **q** = smaller of the two norms."],
        ACCENT,
        14,
    )
    d.card(
        s,
        X0 + Inches(6.23),
        Y0 + Inches(4.0),
        Inches(5.9),
        Inches(1.25),
        "Support guard",
        [
            "If q is below the calibrated range (q < 10.82), the app **abstains**: the score is unreliable there."
        ],
        AMBER,
        14,
    )

    # 11 ── abstention
    s = d.slide(
        "09 · Intelligent abstention",
        "ABSTAIN instead of a forced guess",
        "On 1,000 synthetic development pairs (reports/experiments/phase5_parity.json) the frozen rule decided 473 MATCH and 490 NON-MATCH and abstained on 37 (3.7 %). Explain coverage (share of pairs decided) versus selective error (errors among decided pairs): abstaining trades a little coverage for fewer wrong answers. Then show the example: same synthetic person, image A dark and rotated, image B masked; the score 0.542 falls inside the low-quality band, so the system declines and states the reason.",
    )
    d.chart(
        s,
        XL_CHART_TYPE.BAR_CLUSTERED,
        X0,
        Y0,
        Inches(5.3),
        Inches(2.7),
        ["ABSTAIN", "NON-MATCH", "MATCH"],
        [("Decisions", [37, 490, 473], BLUE)],
        fmt="0",
        legend=False,
        title="1,000 development pairs: outcomes",
        vmax=600,
    )
    ch = s.shapes[-1].chart
    for idx, col in enumerate((AMBER, SLATE, BLUE)):
        pt = ch.plots[0].series[0].points[idx]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col
    d.card(
        s,
        X0,
        Y0 + Inches(2.95),
        Inches(5.3),
        Inches(2.3),
        "Two key terms",
        [
            "**Coverage**: share of pairs that get a decision (96.3 % here).",
            "**Selective error**: errors counted only among decided pairs. Abstaining lowers it.",
        ],
        BLUE,
        14,
    )
    d.picture(s, "shot_abstain_result.png", X0 + Inches(5.6), Y0, Inches(6.53), Inches(3.4))
    d.text(
        s,
        X0 + Inches(5.6),
        Y0 + Inches(3.5),
        Inches(6.53),
        Inches(1.7),
        [
            {
                "t": "Example (synthetic faces): same person, image A dark and rotated, image B masked",
                "bold": True,
                "color": NAVY,
            },
            {
                "t": "Score 0.542 falls inside the low-quality band [0.154, 0.833) → **ABSTAIN**, with the reason shown to the user.",
                "bullet": True,
            },
        ],
        14,
        INK,
    )

    # 12 ── visibility heat maps
    s = d.slide(
        "10 · Explainability",
        "Visibility heat maps",
        "The visibility network predicts which face regions are visible (blue) or covered (orange). It helps users understand the occlusion. It is NOT used for the decision: when random patches were left out of its training, it called covered areas visible (IoU 0.13), and a visibility-based decision rule then failed its target (8.52 % false non-matches vs 2.12 % for the quality rule). Numbers: docs/phase3_results.md.",
    )
    d.picture(s, "vis_example4.png", X0, Y0, CW, Inches(3.25), border=False)
    d.card(
        s,
        X0,
        Y0 + Inches(3.45),
        Inches(3.9),
        Inches(1.8),
        "What it shows",
        [
            "28×28 map: **blue** = visible, **orange** = covered. The mask on image B is detected (39 % visible)."
        ],
        ACCENT,
        13.5,
    )
    d.card(
        s,
        X0 + Inches(4.12),
        Y0 + Inches(3.45),
        Inches(3.9),
        Inches(1.8),
        "How it is built",
        [
            "Small U-Net-style network (3.6 M parameters) trained in this project on synthetic occlusions."
        ],
        TEAL,
        13.5,
    )
    d.card(
        s,
        X0 + Inches(8.24),
        Y0 + Inches(3.45),
        Inches(3.89),
        Inches(1.8),
        "Why not for decisions",
        ["It failed on unseen occluder types, so it is shown as an **explanation only**."],
        AMBER,
        13.5,
    )

    # 13 ── data & methodology
    s = d.slide(
        "11 · Data and methodology",
        "Controlled, privacy-preserving evaluation",
        "DigiFace-1M faces are computer-generated, so no real people are used for the main experiments. Occlusions are synthetic and reproducible. Each of five calibration systems leaves one occluder family out, then is tested on it. Thresholds were calibrated with Learn-then-Test on 6,146 identity-disjoint pairs over 9,219 synthetic identities, with a research target of at most 5 % errors among accepted decisions. Each confirmatory run has 15 evaluation cells (5 calibration systems x seen / unseen / never-seen hair occluder). The final independent repeat used 4,000 never-before-used identities. Test sets were used once and logged.",
    )
    for k, (t_, body, col) in enumerate(
        (
            (
                "Synthetic faces",
                [
                    "**DigiFace-1M**: computer-generated people, no real identities",
                    "Train / calibration / test identities kept **separate**",
                ],
                BLUE,
            ),
            (
                "Synthetic occlusions",
                [
                    "5 families × 3 severities + 1 test-only hair occluder",
                    "Fully reproducible from stored seeds",
                ],
                ORANGE,
            ),
            (
                "Calibration",
                [
                    "**Learn-then-Test** on identity-disjoint pairs",
                    "Target: ≤ 5 % errors among accepted decisions",
                ],
                TEAL,
            ),
        )
    ):
        d.card(
            s,
            X0 + Inches(4.12) * k,
            Y0,
            Inches(3.89) if k == 2 else Inches(3.9),
            Inches(2.0),
            t_,
            body,
            col,
            13.5,
        )
    d.text(
        s,
        X0,
        Y0 + Inches(2.2),
        CW,
        Inches(0.4),
        ["Occluder families (leave-one-family-out testing)"],
        16,
        NAVY,
        True,
    )
    fams = [
        ("Mask", BLUE),
        ("Sunglasses", ACCENT),
        ("Hand / object", TEAL),
        ("Random patch", ORANGE),
        ("Extreme lighting", AMBER),
        ("Hair (test-only)", SLATE),
    ]
    for k, (t_, col) in enumerate(fams):
        d.chip(
            s, X0 + Inches(2.02) * k, Y0 + Inches(2.65), Inches(1.9), Inches(0.48), t_, col, 12.5
        )
    d.text(
        s,
        X0,
        Y0 + Inches(3.3),
        CW,
        Inches(0.5),
        [
            {
                "t": "Each calibration system **never sees one family**, then is tested on it. Real photographs (LFW) were used only for follow-up checks with synthetic overlays.",
                "bullet": True,
            }
        ],
        14,
    )
    figs_ = [
        ("6,146", "calibration pairs", BLUE),
        ("9,219", "synthetic identities in calibration", ACCENT),
        ("15", "test cells per confirmatory run", TEAL),
        ("4,000", "new identities in the final repeat", ORANGE),
    ]
    for k, (v, lab, col) in enumerate(figs_):
        x = X0 + Inches(3.07) * k
        d.box(s, x, Y0 + Inches(4.05), Inches(2.86), Inches(1.2), fill=LIGHT)
        d.box(
            s, x, Y0 + Inches(4.05), Inches(0.08), Inches(1.2), fill=col, shape=MSO_SHAPE.RECTANGLE
        )
        d.text(s, x + Inches(0.2), Y0 + Inches(4.12), Inches(2.6), Inches(0.6), [v], 26, col, True)
        d.text(s, x + Inches(0.2), Y0 + Inches(4.7), Inches(2.6), Inches(0.5), [lab], 12.5, SLATE)

    # 14 ── application interface
    s = d.slide(
        "12 · Application",
        "The VEIL-X application",
        "Show the main screen. Callouts on the right summarise what the user sees. The two confirmations make the user acknowledge authorisation and the calibration limits before each comparison.",
    )
    d.picture(s, "shot_home.png", X0, Y0, Inches(7.6), Inches(5.25))
    feats = [
        ("Two-image upload", "JPEG / PNG / BMP / WEBP, validated"),
        ("Configuration card", "states what the calibration is"),
        ("Options", "face selection, pre-aligned mode, diagnostics"),
        ("Two confirmations", "authorisation + limitations"),
        ("Result panel", "decision, score, threshold bar, reason"),
        ("Diagnostics", "table, heat maps, novelty, timings"),
        ("Information tabs", "limitations, model info, examples"),
    ]
    for k, (t, b) in enumerate(feats):
        y = Y0 + Inches(0.74) * k
        d.chip(
            s, X0 + Inches(7.85), y + Inches(0.08), Inches(0.42), Inches(0.42), str(k + 1), BLUE, 11
        )
        d.text(s, X0 + Inches(8.4), y, Inches(3.75), Inches(0.35), [t], 14.5, NAVY, True)
        d.text(s, X0 + Inches(8.4), y + Inches(0.32), Inches(3.75), Inches(0.4), [b], 12, SLATE)

    # 15 ── demo results
    s = d.slide(
        "13 · Results in the app",
        "MATCH, NON-MATCH and ABSTAIN on real runs",
        "Real outputs of the running application on the built-in synthetic examples (computer-generated faces), with the frozen thresholds unchanged. Example 2: MATCH, score 1.000 in the high-quality group (threshold 0.673). Example 1: NON-MATCH, score 0.001 in the medium-quality group (threshold 0.452). Example 4: ABSTAIN, score 0.542 inside the low-quality band [0.154, 0.833); image A is dark and rotated, image B is masked.",
    )
    rows = [
        (
            "pair_match.png",
            "box_match.png",
            "MATCH",
            BLUE,
            "Same synthetic person",
            "1.000 ≥ 0.673 (high quality)",
        ),
        (
            "pair_nonmatch.png",
            "box_nonmatch.png",
            "NON-MATCH",
            SLATE,
            "Two different synthetic people",
            "0.001 < 0.452 (medium quality)",
        ),
        (
            "pair_abstain.png",
            "box_abstain.png",
            "ABSTAIN",
            AMBER,
            "Same person; A dark, B masked",
            "0.542 in band [0.154, 0.833) (low quality)",
        ),
    ]
    for k, (pair, box_img, lab, col, ex, why) in enumerate(rows):
        y = Y0 + Inches(1.76) * k
        d.box(s, X0, y, CW, Inches(1.6), fill=LIGHT)
        d.picture(
            s, pair, X0 + Inches(0.15), y + Inches(0.12), Inches(2.75), Inches(1.36), border=False
        )
        d.picture(s, box_img, X0 + Inches(3.05), y + Inches(0.15), Inches(6.0), Inches(0.9))
        d.text(s, X0 + Inches(3.05), y + Inches(1.1), Inches(6.0), Inches(0.4), [ex], 12, SLATE)
        d.chip(s, X0 + Inches(9.35), y + Inches(0.22), Inches(2.6), Inches(0.5), lab, col, 15)
        d.text(
            s,
            X0 + Inches(9.25),
            y + Inches(0.82),
            Inches(2.8),
            Inches(0.7),
            [why],
            12.5,
            INK,
            align=PP_ALIGN.CENTER,
        )

    # 16 ── testing
    s = d.slide(
        "14 · Testing and validation",
        "Engineering results",
        "Parity means the application reproduces the offline research system exactly; it is NOT recognition accuracy. Sources: reports/experiments/phase5_parity.json; test counts from the final quality run (CPU and GPU environments).",
    )
    vals = [
        ("1000 / 1000", "Identical decisions", "app vs research code", BLUE),
        ("100 / 100", "CPU = GPU decisions", "min. cosine 0.99991", ACCENT),
        ("244 / 244", "Pixel-identical crops", "real photographs (LFW dev)", TEAL),
        (d.tests_text, "Automated tests passing", "CPU + GPU suites; lint, typing, privacy", NAVY),
    ]
    for k, (v, lab, sub, col) in enumerate(vals):
        d.kpi(s, X0 + Inches(3.07) * k, Y0, Inches(2.86), Inches(2.0), v, lab, col, sub)
    d.table(
        s,
        X0,
        Y0 + Inches(2.3),
        CW,
        [
            ["Check", "How it was verified", "Result"],
            [
                "Local-only operation",
                "Browser network log; server binding",
                "127.0.0.1 only; only external request blocked",
            ],
            [
                "Upload cleanup",
                "Live check of the temp folder",
                "Deleted after reading; leftovers gone in 149 s",
            ],
            [
                "Safety refusals",
                "Unit tests + live checks",
                "Multiple faces, non-face photos, noise → refused / ABSTAIN",
            ],
            ["Model integrity", "SHA-256 at load (10 files)", "All match"],
        ],
        [2.8, 4.4, 4.93],
        13.5,
        0.55,
    )
    d.text(
        s,
        X0,
        Y0 + Inches(5.15),
        CW,
        Inches(0.3),
        ["Parity is not accuracy: it shows the app reproduces the research system exactly."],
        12,
        AMBER,
        True,
    )

    # 17 ── performance
    s = d.slide(
        "15 · Performance",
        "Fast enough for interactive use",
        "Measured on an NVIDIA RTX 4060 Laptop GPU and AMD Ryzen 9 7940HS, one pair per request after warm-up, medians over 27 completed photo comparisons (reports/experiments/phase5_benchmark.json). Face detection runs on the CPU and dominates. Peak GPU memory 550 MiB; model loading about 2.7 s.",
    )
    d.chart(
        s,
        XL_CHART_TYPE.BAR_CLUSTERED,
        X0,
        Y0,
        Inches(7.0),
        Inches(4.0),
        ["Diagnostics", "ArcFace embedding", "AdaFace embedding", "Detection + alignment"],
        [
            ("GPU (ms)", [5.7, 21.3, 24.2, 173.7], BLUE),
            ("CPU only (ms)", [18.1, 139.3, 165.3, 185.9], SLATE),
        ],
        fmt='0" ms"',
        vmax=220,
        title="Time per stage, one photo pair (median)",
    )
    d.kpi(
        s,
        X0 + Inches(7.35),
        Y0,
        Inches(2.3),
        Inches(1.95),
        "226 ms",
        "GPU, two photos",
        BLUE,
        "p95 285 ms",
    )
    d.kpi(
        s,
        X0 + Inches(9.83),
        Y0,
        Inches(2.3),
        Inches(1.95),
        "43 ms",
        "GPU, pre-aligned",
        ACCENT,
        "p95 49 ms",
    )
    d.kpi(
        s,
        X0 + Inches(7.35),
        Y0 + Inches(2.1),
        Inches(2.3),
        Inches(1.95),
        "509 ms",
        "CPU only, photos",
        SLATE,
        "no GPU needed",
    )
    d.kpi(
        s,
        X0 + Inches(9.83),
        Y0 + Inches(2.1),
        Inches(2.3),
        Inches(1.95),
        "550 MiB",
        "Peak GPU memory",
        TEAL,
        "8 GB laptop GPU",
    )
    d.text(
        s,
        X0,
        Y0 + Inches(4.3),
        CW,
        Inches(0.9),
        [
            {
                "t": "Hardware: RTX 4060 Laptop GPU (8 GB) · Ryzen 9 7940HS · 31 GB RAM · Windows 11. Model loading ≈ 2.7 s; first comparison ≈ 0.7 s.",
                "color": GREY,
            }
        ],
        13,
    )

    # 18 ── finding 1
    s = d.slide(
        "16 · Key finding 1",
        "Quality beat learned visibility",
        "Phase 3 confirmatory test (docs/phase3_results.md): selective false non-match rate on the held-out (unseen) occluder for each calibration system. The quality-stratified rule stayed within the 5 % target everywhere; the learned-visibility rule failed on random patches (8.52 %), because its visibility estimate failed on that unseen occluder (IoU 0.13). That is why VEIL-X decides with quality and shows visibility only as an explanation.",
    )
    d.chart(
        s,
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        X0,
        Y0,
        Inches(7.6),
        Inches(4.4),
        ["Mask", "Sunglasses", "Hand/object", "Random patch", "Lighting"],
        [
            ("Quality rule (VEIL-X)", [1.94, 1.87, 1.59, 2.12, 1.08], BLUE),
            ("Learned-visibility rule", [1.75, 1.28, 2.36, 8.52, 2.25], ORANGE),
        ],
        vmax=10,
        title="False non-match on the unseen occluder (target ≤ 5 %)",
        label_size=10,
    )
    d.chart(
        s,
        XL_CHART_TYPE.BAR_CLUSTERED,
        X0 + Inches(7.9),
        Y0,
        Inches(4.23),
        Inches(3.1),
        ["Lighting", "Random patch", "Sunglasses", "Hand/object", "Mask"],
        [("IoU", [0.005, 0.13, 0.865, 0.886, 0.898], TEAL)],
        fmt="0.00",
        vmax=1.0,
        legend=False,
        title="Visibility IoU on unseen occluder",
        tick_fmt="0.0",
        label_size=10,
    )
    ch = s.shapes[-1].chart
    for idx in (0, 1):
        pt = ch.plots[0].series[0].points[idx]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = ORANGE
    d.text(
        s,
        X0 + Inches(7.9),
        Y0 + Inches(3.2),
        Inches(4.23),
        Inches(1.4),
        [
            {
                "t": "Visibility generalised to some unseen occluders but **failed** on random patches and lighting.",
                "size": 13.5,
            }
        ],
        13.5,
    )
    d.text(
        s,
        X0,
        Y0 + Inches(4.6),
        CW,
        Inches(0.6),
        [
            {
                "t": "→ Decisions use **quality**; visibility is shown as an explanation only.",
                "color": NAVY,
                "size": 16,
            }
        ],
        16,
    )

    # 19 ── finding 2
    s = d.slide(
        "17 · Key finding 2",
        "Calibration depends on the population",
        "Blue: on data like the calibration data, the rule met the 5 % target. Orange: on a harder group of synthetic identities (72 images each, more variation) the miss rate rose above 5 % even for familiar occluders. Green: an independent, pre-registered repeat on 4,000 new identities confirmed it (4 of 5 configurations above target). The false-match side stayed within target. Lesson: thresholds must be re-calibrated for each new population. Sources: phase3/4/6 final reports.",
    )
    d.picture(s, "fig_population_shift.png", X0, Y0, Inches(8.0), Inches(5.2), border=False)
    d.kpi(
        s,
        X0 + Inches(8.3),
        Y0,
        Inches(3.83),
        Inches(1.55),
        "1.1 – 2.1 %",
        "Matching population",
        BLUE,
        "within the 5 % target",
    )
    d.kpi(
        s,
        X0 + Inches(8.3),
        Y0 + Inches(1.7),
        Inches(3.83),
        Inches(1.55),
        "5.4 – 6.3 %",
        "Harder population",
        ORANGE,
        "above target, 5 of 5",
    )
    d.kpi(
        s,
        X0 + Inches(8.3),
        Y0 + Inches(3.4),
        Inches(3.83),
        Inches(1.55),
        "5.2 – 6.5 %",
        "Independent repeat",
        TEAL,
        "4,000 new identities, 4 of 5",
    )

    # 20 ── finding 3
    s = d.slide(
        "18 · Key finding 3",
        "Low-quality inputs and the support guard",
        "In final testing, two random-noise images received a score of 1.000 (MATCH) from the research rule: the formula extrapolates far below the quality range it was calibrated on. On untouched test data, the roughly 1 % of pairs below that range had a 22.97 % false-match rate versus 1.50 % above it (reports/experiments/phase6_final.json). The application therefore abstains below the calibrated range. This is an application safeguard, not a proven universal fix; the original research results are unchanged.",
    )
    d.chart(
        s,
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        X0,
        Y0,
        Inches(5.2),
        Inches(4.0),
        ["Below calibrated quality", "Within calibrated quality"],
        [("False-match rate", [22.97, 1.50], ORANGE)],
        fmt='0.00"%"',
        vmax=28,
        legend=False,
        title="False-match rate on new test data",
    )
    ch = s.shapes[-1].chart
    pt = ch.plots[0].series[0].points[1]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = BLUE
    d.picture(s, "shot_guard_result.png", X0 + Inches(5.5), Y0, Inches(6.63), Inches(2.9))
    d.card(
        s,
        X0 + Inches(5.5),
        Y0 + Inches(3.05),
        Inches(6.63),
        Inches(2.2),
        "Support guard (application safeguard)",
        [
            "Abstains when quality is below the calibrated range: two noise images now give **ABSTAIN** with the reason shown.",
            "Not a proven universal fix; research results unchanged.",
        ],
        AMBER,
        13.5,
    )
    d.text(
        s,
        X0,
        Y0 + Inches(4.2),
        Inches(5.2),
        Inches(1.0),
        [
            {
                "t": "Below the range: about 1 % of pairs, but 22.97 % of them were false matches.",
                "color": SLATE,
            }
        ],
        13,
    )

    # 21 ── advantages & limitations
    s = d.slide(
        "19 · Evaluation",
        "Advantages and limitations",
        "Be balanced and honest. The limitations are measured, documented and visible in the app itself.",
    )
    adv = [
        "**Honest outputs**: abstains instead of guessing",
        "**Explainable**: score, quality, threshold, reason, heat maps",
        "**Private**: runs only on the local machine",
        "**Fast**: ~0.2 s per photo pair on a laptop GPU",
        "**Reliable engineering**: 400+ tests, exact parity",
        "**Safe defaults**: refuses ambiguous / unsuitable input",
    ]
    lim = [
        "Synthetic faces and occlusions; no real physical occlusions tested",
        "5 % target failed on a harder population (confirmed twice)",
        "Low-quality false matches; guarded only in the app",
        "Five calibration variants; none universal",
        "Not certified; not for real identity decisions",
    ]
    d.box(s, X0, Y0, Inches(5.95), Inches(5.25), fill=LIGHT)
    d.box(s, X0, Y0, Inches(5.95), Inches(0.6), fill=BLUE, shape=MSO_SHAPE.RECTANGLE)
    d.text(
        s,
        X0,
        Y0,
        Inches(5.95),
        Inches(0.6),
        ["Advantages"],
        18,
        WHITE,
        True,
        PP_ALIGN.CENTER,
        MSO_ANCHOR.MIDDLE,
    )
    d.text(
        s,
        X0 + Inches(0.25),
        Y0 + Inches(0.8),
        Inches(5.5),
        Inches(4.4),
        [{"t": a, "bullet": True} for a in adv],
        18,
        spacing=16,
    )
    d.box(s, X0 + Inches(6.18), Y0, Inches(5.95), Inches(5.25), fill=RGBColor(0xFD, 0xF6, 0xEC))
    d.box(
        s, X0 + Inches(6.18), Y0, Inches(5.95), Inches(0.6), fill=AMBER, shape=MSO_SHAPE.RECTANGLE
    )
    d.text(
        s,
        X0 + Inches(6.18),
        Y0,
        Inches(5.95),
        Inches(0.6),
        ["Limitations"],
        18,
        WHITE,
        True,
        PP_ALIGN.CENTER,
        MSO_ANCHOR.MIDDLE,
    )
    d.text(
        s,
        X0 + Inches(6.43),
        Y0 + Inches(0.8),
        Inches(5.5),
        Inches(4.4),
        [{"t": x, "bullet": True} for x in lim],
        18,
        spacing=16,
    )

    # 22 ── future work
    s = d.slide(
        "20 · Future enhancements",
        "What would make it deployment-grade",
        "Keep this brief. Each item addresses a measured limitation.",
    )
    fut = [
        ("Re-calibrate", "on photos from the intended users and camera", BLUE),
        (
            "Real occlusions",
            "evaluate on naturally occluded photos (e.g. IJB-C, after access approval)",
            ORANGE,
        ),
        ("Safer score", "a fused score that cannot extrapolate below its quality range", AMBER),
        ("Faster", "GPU face detection (needs a compatible CUDA runtime)", TEAL),
        ("Better UX", "clickable face selection for group photos", ACCENT),
    ]
    for k, (t, b, col) in enumerate(fut):
        x = X0 + Inches(2.45) * k
        d.box(s, x, Y0 + Inches(0.4), Inches(2.25), Inches(3.6), fill=LIGHT)
        d.box(
            s, x, Y0 + Inches(0.4), Inches(2.25), Inches(0.12), fill=col, shape=MSO_SHAPE.RECTANGLE
        )
        d.chip(
            s, x + Inches(0.8), Y0 + Inches(0.75), Inches(0.65), Inches(0.65), str(k + 1), col, 16
        )
        d.text(
            s,
            x + Inches(0.1),
            Y0 + Inches(1.55),
            Inches(2.05),
            Inches(0.5),
            [t],
            17,
            NAVY,
            True,
            PP_ALIGN.CENTER,
        )
        d.text(
            s,
            x + Inches(0.15),
            Y0 + Inches(2.1),
            Inches(1.95),
            Inches(1.8),
            [b],
            13.5,
            SLATE,
            align=PP_ALIGN.CENTER,
        )
        if k < 4:
            d.arrow(s, x + Inches(2.27), Y0 + Inches(2.2), x + Inches(2.43), Y0 + Inches(2.2), GREY)

    # 23 ── conclusion
    s = d.slide(
        "21 · Conclusion",
        "Conclusion",
        "Summarise in three sentences, then switch to the live demo following the demo guide: MATCH (example 2), NON-MATCH (example 1), ABSTAIN (example 4), then two noise images showing the support guard.",
    )
    d.text(
        s,
        X0,
        Y0,
        Inches(7.2),
        Inches(3.6),
        [
            {
                "t": "A **working, local** face verification application with quality-aware **MATCH / NON-MATCH / ABSTAIN** decisions",
                "bullet": True,
            },
            {
                "t": "**Explains** every result: diagnostics, abstention reasons and visibility heat maps",
                "bullet": True,
            },
            {
                "t": "**Private, fast and thoroughly tested**: exact parity with the research pipeline",
                "bullet": True,
            },
            {
                "t": "**Honest about its limits**: calibration depends on the population; low-quality inputs are guarded",
                "bullet": True,
            },
        ],
        17,
        spacing=12,
    )
    d.box(s, X0 + Inches(7.5), Y0, Inches(4.63), Inches(5.25), fill=NAVY)
    d.text(
        s,
        X0 + Inches(7.75),
        Y0 + Inches(0.25),
        Inches(4.2),
        Inches(0.5),
        ["LIVE DEMO"],
        14,
        RGBColor(0x8F, 0xB8, 0xEC),
        True,
    )
    for k, (lab, col, sub) in enumerate(
        (
            ("MATCH", BLUE, "example 2: same person"),
            ("NON-MATCH", SLATE, "example 1: different people"),
            ("ABSTAIN", AMBER, "example 4: masked, low quality"),
            ("ABSTAIN", AMBER, "noise images: support guard"),
        )
    ):
        y = Y0 + Inches(0.85) + Inches(1.05) * k
        d.chip(s, X0 + Inches(7.75), y, Inches(1.75), Inches(0.5), lab, col, 13)
        s.shapes[-1].line.color.rgb = WHITE
        d.text(s, X0 + Inches(9.65), y + Inches(0.05), Inches(2.4), Inches(0.5), [sub], 13, WHITE)
    d.text(
        s,
        X0,
        Y0 + Inches(4.0),
        Inches(7.2),
        Inches(1.2),
        [{"t": "Thank you. Questions?", "size": 26, "bold": True, "color": NAVY}],
        26,
    )

    # 24 ── references
    s = d.slide(
        "References",
        "References",
        "Full research record: docs/paper/veilx_paper.md, docs/phase6_audit.md, docs/phase6_results.md in the project repository.",
    )
    refs = [
        "Kim, M., Jain, A. K., Liu, X. (2022). AdaFace: Quality adaptive margin for face recognition. CVPR.",
        "Deng, J., Guo, J., Xue, N., Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. CVPR.",
        "Guo, J., Deng, J., Lattas, A., Zafeiriou, S. (2022). Sample and computation redistribution for efficient face detection (SCRFD). ICLR.",
        "Bae, G. et al. (2023). DigiFace-1M: 1 million digital face images for face recognition. WACV.",
        "Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., Lei, L. (2021). Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv:2110.01052.",
        "Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. MICCAI.",
        "Zhu, Z. et al. (2021). WebFace260M: A benchmark unveiling the power of million-scale deep face recognition. CVPR.",
        "Huang, G. B., Ramesh, M., Berg, T., Learned-Miller, E. (2007). Labeled Faces in the Wild. Tech. Report 07-49, UMass Amherst.",
        "Ngan, M., Grother, P., Hanaoka, K. (2020). Face recognition accuracy with masks using pre-COVID-19 algorithms. NISTIR 8311.",
        "Abid, A. et al. (2019). Gradio: Hassle-free sharing and testing of ML models in the wild. arXiv:1906.02569.",
    ]
    d.text(
        s,
        X0,
        Y0,
        CW,
        Inches(5.3),
        [{"t": f"[{k + 1}]  {r}", "after": 7} for k, r in enumerate(refs)],
        13.5,
    )

    d.save(path)
