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


# ====================================================================== the deck (8 slides)
def build(path: Path, assets: Path, tests_text: str) -> None:
    d = Deck(assets, tests_text)

    # 1 ── title ---------------------------------------------------------------------------
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
    s.notes_slide.notes_text_frame.text = (
        "Introduce yourself. VEIL-X compares two face photos on a laptop and answers MATCH, NON-MATCH or ABSTAIN, "
        "and explains its answer. It uses strong pretrained models; the project's work is the quality-aware decision layer, "
        "abstention, explanations and a tested, private application. Academic demonstration, not a certified identity system."
    )

    # 2 ── problem and objectives -----------------------------------------------------------
    s = d.slide(
        "01 · Problem and objectives",
        "Occlusion and low quality break face matching",
        "Face verification is 1:1: are these two photos the same person (phone unlock, e-gates). Our own baseline with the frozen AdaFace model on synthetic faces (reports/experiments/phase2_final.json): at a fixed threshold the false non-match rate is 1.4 % on clean pairs, 6.8 to 21.6 % with an unseen occluder and 44.9 % for high-severity random patches. A basic matcher still answers yes or no, so these errors are silent. Then read the six objectives.",
    )
    for k, (v, lab, col, sub) in enumerate(
        (
            ("1.4 %", "Missed matches, clean faces", BLUE, "fixed threshold"),
            ("6.8 – 21.6 %", "With an unseen occluder", ORANGE, "masks, glasses, hands, patches"),
            ("44.9 %", "Worst case", AMBER, "high-severity random patch"),
        )
    ):
        d.kpi(s, X0 + Inches(4.17) * k, Y0, Inches(3.8), Inches(1.75), v, lab, col, sub)
    d.card(
        s,
        X0,
        Y0 + Inches(2.0),
        Inches(4.9),
        Inches(3.25),
        "The problem",
        [
            {
                "t": "A basic matcher **must** answer yes / no: errors on weak evidence are **silent**",
                "bullet": True,
            },
            {"t": "No indication of **how reliable** an answer is", "bullet": True},
            {"t": "Cloud face services raise **privacy** concerns", "bullet": True},
        ],
        BLUE,
        14.5,
    )
    objs = [
        ("Verify", "two images with pretrained deep models", BLUE),
        ("Assess quality", "and build it into the decision", ACCENT),
        ("Decide 3 ways", "MATCH / NON-MATCH / ABSTAIN", AMBER),
        ("Explain", "scores, reasons, heat maps", TEAL),
        ("Stay private", "100 % local, nothing stored", SLATE),
        ("Test & document", "parity, speed, limitations", NAVY),
    ]
    d.text(
        s,
        X0 + Inches(5.2),
        Y0 + Inches(2.0),
        Inches(6.9),
        Inches(0.4),
        ["Objectives"],
        17,
        NAVY,
        True,
    )
    for k, (t, b, col) in enumerate(objs):
        col_, row = k % 2, k // 2
        x, y = X0 + Inches(5.2) + Inches(3.5) * col_, Y0 + Inches(2.5) + Inches(0.92) * row
        d.box(s, x, y, Inches(3.38), Inches(0.8), fill=LIGHT)
        d.chip(
            s, x + Inches(0.12), y + Inches(0.17), Inches(0.46), Inches(0.46), str(k + 1), col, 13
        )
        d.text(
            s,
            x + Inches(0.7),
            y + Inches(0.05),
            Inches(2.65),
            Inches(0.75),
            [
                {"t": t, "bold": True, "color": NAVY, "size": 14, "after": 0},
                {"t": b, "size": 11.5, "color": SLATE},
            ],
            12,
            spacing=0,
        )

    # 3 ── architecture ---------------------------------------------------------------------
    s = d.slide(
        "02 · System design",
        "Architecture and technology",
        "Walk left to right along the decision path: input checks, SCRFD detection and alignment, AdaFace embedding and feature norm, similarity s and quality q, logistic fusion M1, the quality-stratified rule A3 with the support guard, and the three outcomes. Several faces or no reliable face are refused. The dashed row is computed after the decision and never changes it: ArcFace similarity, the visibility heat map, the novelty flag. Pretrained: AdaFace, ArcFace, SCRFD. Built or trained in this project: the visibility network, the logistic fusion, the calibrated thresholds and the application. Model files are checked by SHA-256 at load; the web interface listens only on 127.0.0.1.",
    )
    d.text(
        s,
        X0,
        Y0 - Inches(0.05),
        CW,
        Inches(0.35),
        ["DECISION PATH  —  pretrained models + calibrated rules"],
        12,
        BLUE,
        True,
    )
    steps = [
        ("Upload", "2 images"),
        ("Validate", "format, size,\ncorruption"),
        ("Detect + align", "SCRFD, 5 points\n→ 112×112"),
        ("Embed", "AdaFace IR-101\n+ feature norm"),
        ("Compare", "similarity s\nquality q"),
        ("Score", "M1 logistic\np(s, q, s·q)"),
        ("Decide", "A3 thresholds\n+ support guard"),
    ]
    bw, gap = Inches(1.5), Inches(0.27)
    for k, (t, b) in enumerate(steps):
        x = X0 + (bw + gap) * k
        last = k == len(steps) - 1
        d.box(
            s,
            x,
            Y0 + Inches(0.35),
            bw,
            Inches(1.45),
            fill=BLUE if last else LIGHT,
            line=None if last else LINE,
        )
        d.text(
            s,
            x,
            Y0 + Inches(0.45),
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
            Y0 + Inches(0.9),
            bw,
            Inches(0.9),
            [b],
            12,
            WHITE if last else SLATE,
            align=PP_ALIGN.CENTER,
        )
        if not last:
            d.arrow(
                s,
                x + bw + Inches(0.02),
                Y0 + Inches(1.07),
                x + bw + gap - Inches(0.02),
                Y0 + Inches(1.07),
                ACCENT,
            )
    xd = X0 + (bw + gap) * 6
    d.arrow(s, xd + bw // 2, Y0 + Inches(1.82), xd + bw // 2, Y0 + Inches(2.1), BLUE)
    for k, (lab, col) in enumerate((("MATCH", BLUE), ("NON-MATCH", SLATE), ("ABSTAIN", AMBER))):
        d.chip(
            s,
            xd - Inches(0.1),
            Y0 + Inches(2.15) + Inches(0.47) * k,
            bw + Inches(0.2),
            Inches(0.4),
            lab,
            col,
            12,
        )
    xr = X0 + (bw + gap) * 2
    d.arrow(s, xr + bw // 2, Y0 + Inches(1.82), xr + bw // 2, Y0 + Inches(2.1), AMBER)
    d.box(s, xr, Y0 + Inches(2.12), bw, Inches(0.85), fill=RGBColor(0xFD, 0xF1, 0xDC), line=AMBER)
    d.text(
        s,
        xr,
        Y0 + Inches(2.17),
        bw,
        Inches(0.8),
        ["Refuses: several faces · no reliable face"],
        11.5,
        AMBER,
        align=PP_ALIGN.CENTER,
    )
    d.text(
        s,
        X0 + (bw + gap) * 3,
        Y0 + Inches(2.15),
        Inches(5.0),
        Inches(0.35),
        ["EXPLANATIONS ONLY  —  never change the decision"],
        12,
        GREY,
        True,
    )
    for k, (t, b) in enumerate(
        (
            ("ArcFace", "2nd similarity"),
            ("Visibility map", "visible vs covered"),
            ("Novelty", "unusual input flag"),
        )
    ):
        x = X0 + (bw + gap) * 3 + Inches(1.72) * k
        d.box(s, x, Y0 + Inches(2.55), Inches(1.6), Inches(0.95), fill=WHITE, line=GREY, dash=True)
        d.text(
            s,
            x,
            Y0 + Inches(2.6),
            Inches(1.6),
            Inches(0.9),
            [
                {"t": t, "bold": True, "color": SLATE, "size": 13, "after": 0},
                {"t": b, "size": 11, "color": GREY},
            ],
            12,
            align=PP_ALIGN.CENTER,
            spacing=0,
        )
    d.text(s, X0, Y0 + Inches(3.75), CW, Inches(0.35), ["TECHNOLOGY STACK"], 12, BLUE, True)
    stack = [
        ("Interface", "Gradio 5 · FastAPI · 127.0.0.1 only", ACCENT),
        ("Pretrained models", "AdaFace & ArcFace IR-101 · SCRFD-10G", NAVY),
        ("Trained here", "visibility U-Net (3.6 M) · logistic fusion · thresholds", TEAL),
        ("Engineering", "Python 3.12 · PyTorch + CUDA · pytest · strict mypy · CI", SLATE),
    ]
    for k, (t, b, col) in enumerate(stack):
        x = X0 + Inches(3.07) * k
        d.box(s, x, Y0 + Inches(4.12), Inches(2.9), Inches(1.13), fill=LIGHT)
        d.box(
            s, x, Y0 + Inches(4.12), Inches(2.9), Inches(0.08), fill=col, shape=MSO_SHAPE.RECTANGLE
        )
        d.text(
            s,
            x + Inches(0.15),
            Y0 + Inches(4.25),
            Inches(2.65),
            Inches(1.0),
            [
                {"t": t, "bold": True, "color": col, "size": 14, "after": 2},
                {"t": b, "size": 12, "color": INK},
            ],
            12,
            spacing=0,
        )

    # 4 ── how it decides ----------------------------------------------------------------
    s = d.slide(
        "03 · Decision method",
        "Quality-aware decisions with intelligent abstention",
        "Similarity s is the cosine between the two AdaFace embeddings; quality q is the smaller of the two feature norms (blurred or covered faces give smaller norms). A logistic formula gives a score p between 0 and 1. It is NOT a probability of identity. The pair's quality picks a group, and each group has thresholds calibrated with the Learn-then-Test procedure on 6,146 separate pairs, research target at most 5 % errors among accepted decisions. Only the low-quality group has an abstention band. Below the calibrated quality range (q < 10.82) the app always abstains (support guard). On 1,000 development pairs: 473 MATCH, 490 NON-MATCH, 37 ABSTAIN, so coverage 96.3 % (reports/experiments/phase5_parity.json).",
    )
    d.box(s, X0, Y0, Inches(7.4), Inches(1.15), fill=LIGHT)
    d.text(
        s,
        X0 + Inches(0.2),
        Y0 + Inches(0.08),
        Inches(7.1),
        Inches(1.05),
        [
            {
                "t": "**s** = cosine(e_A, e_B)      **q** = min(norm_A, norm_B)      **p** = logistic(s, q, s·q)",
                "size": 15.5,
                "after": 4,
            },
            {
                "t": "p is a calibrated score between 0 and 1, **not** a probability of identity",
                "size": 12.5,
                "color": AMBER,
            },
        ],
        15,
    )
    groups = [
        ("Low quality", "q < 19.9", 0.154, 0.833, AMBER),
        ("Medium quality", "19.9 – 22.0", 0.452, 0.452, ACCENT),
        ("High quality", "q ≥ 22.0", 0.673, 0.673, BLUE),
    ]
    bx, bwid = X0 + Inches(1.9), Inches(5.5)
    for k, (t, rng, lo, hi, col) in enumerate(groups):
        y = Y0 + Inches(1.45) + Inches(0.98) * k
        d.text(s, X0, y, Inches(1.85), Inches(0.4), [t], 14.5, col, True)
        d.text(s, X0, y + Inches(0.36), Inches(1.85), Inches(0.35), [rng], 11.5, GREY)
        d.box(
            s,
            bx,
            y + Inches(0.08),
            int(bwid * lo),
            Inches(0.48),
            fill=RGBColor(0xC9, 0xC9, 0xC9),
            shape=MSO_SHAPE.RECTANGLE,
        )
        if hi > lo:
            d.box(
                s,
                bx + int(bwid * lo),
                y + Inches(0.08),
                int(bwid * (hi - lo)),
                Inches(0.48),
                fill=RGBColor(0xF2, 0xC9, 0x8C),
                shape=MSO_SHAPE.RECTANGLE,
            )
        d.box(
            s,
            bx + int(bwid * hi),
            y + Inches(0.08),
            int(bwid * (1 - hi)),
            Inches(0.48),
            fill=RGBColor(0x9C, 0xBF, 0xEA),
            shape=MSO_SHAPE.RECTANGLE,
        )
        d.text(
            s,
            bx,
            y + Inches(0.1),
            int(bwid * lo),
            Inches(0.45),
            ["NON-MATCH"],
            10.5,
            SLATE,
            True,
            PP_ALIGN.CENTER,
            MSO_ANCHOR.MIDDLE,
        )
        if hi > lo:
            d.text(
                s,
                bx + int(bwid * lo),
                y + Inches(0.1),
                int(bwid * (hi - lo)),
                Inches(0.45),
                [f"ABSTAIN  [{lo:.3f}, {hi:.3f})"],
                10.5,
                AMBER,
                True,
                PP_ALIGN.CENTER,
                MSO_ANCHOR.MIDDLE,
            )
        d.text(
            s,
            bx + int(bwid * hi),
            y + Inches(0.1),
            int(bwid * (1 - hi)),
            Inches(0.45),
            ["MATCH"],
            10.5,
            BLUE,
            True,
            PP_ALIGN.CENTER,
            MSO_ANCHOR.MIDDLE,
        )
        d.text(
            s,
            bx + int(bwid * hi) - Inches(0.5),
            y + Inches(0.56),
            Inches(1.0),
            Inches(0.28),
            [f"{hi:.3f}"],
            9.5,
            GREY,
            align=PP_ALIGN.CENTER,
        )
    d.card(
        s,
        X0,
        Y0 + Inches(4.45),
        Inches(7.4),
        Inches(0.8),
        "Support guard: q below the calibrated range (q < 10.82) → always ABSTAIN",
        [],
        AMBER,
        12,
        14,
    )
    d.chart(
        s,
        XL_CHART_TYPE.BAR_CLUSTERED,
        X0 + Inches(7.7),
        Y0,
        Inches(4.43),
        Inches(3.3),
        ["ABSTAIN", "NON-MATCH", "MATCH"],
        [("Decisions", [37, 490, 473], BLUE)],
        fmt="0",
        legend=False,
        title="1,000 test pairs: outcomes",
        vmax=600,
    )
    ch = s.shapes[-1].chart
    for idx, col in enumerate((AMBER, SLATE, BLUE)):
        pt = ch.plots[0].series[0].points[idx]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = col
    d.kpi(
        s,
        X0 + Inches(7.7),
        Y0 + Inches(3.45),
        Inches(2.12),
        Inches(1.8),
        "96.3 %",
        "Coverage",
        BLUE,
        "pairs decided",
    )
    d.kpi(
        s,
        X0 + Inches(10.01),
        Y0 + Inches(3.45),
        Inches(2.12),
        Inches(1.8),
        "3.7 %",
        "Abstained",
        AMBER,
        "too uncertain",
    )

    # 5 ── application and results ---------------------------------------------------------
    s = d.slide(
        "04 · Working application",
        "MATCH, NON-MATCH and ABSTAIN in the live app",
        "Real outputs of the running application on the built-in synthetic examples (computer-generated faces), frozen thresholds unchanged. 'Same person': MATCH, score 1.000, high-quality group (threshold 0.673). 'Different people': NON-MATCH, 0.001, medium-quality group (threshold 0.452). 'Same person, masked': ABSTAIN, 0.542 inside the low-quality uncertain zone; image A is dark and rotated, image B is masked. Under every result, a 'Why this result?' panel lists each step with its exact numbers, the threshold comparison that decided, and what would change the result. Bottom: the visibility heat map of the masked example (blue visible, orange covered). It explains the occlusion but is never used for the decision, because learned visibility failed on unseen occluder types.",
    )
    rows = [
        ("pair_match.png", "box_match.png", "MATCH", BLUE, "1.000 ≥ 0.673  (high quality)"),
        (
            "pair_nonmatch.png",
            "box_nonmatch.png",
            "NON-MATCH",
            SLATE,
            "0.001 < 0.452  (medium quality)",
        ),
        (
            "pair_abstain.png",
            "box_abstain.png",
            "ABSTAIN",
            AMBER,
            "0.542 in [0.154, 0.833)  (low quality)",
        ),
    ]
    for k, (pair, box_img, lab, col, why) in enumerate(rows):
        y = Y0 + Inches(1.16) * k
        d.box(s, X0, y, CW, Inches(1.06), fill=LIGHT)
        d.picture(
            s, pair, X0 + Inches(0.1), y + Inches(0.06), Inches(1.95), Inches(0.94), border=False
        )
        d.picture(s, box_img, X0 + Inches(2.2), y + Inches(0.1), Inches(6.6), Inches(0.86))
        d.chip(s, X0 + Inches(9.0), y + Inches(0.12), Inches(2.95), Inches(0.42), lab, col, 14)
        d.text(
            s,
            X0 + Inches(9.0),
            y + Inches(0.58),
            Inches(2.95),
            Inches(0.42),
            [why],
            11.5,
            INK,
            align=PP_ALIGN.CENTER,
        )
    d.picture(s, "vis_example4.png", X0, Y0 + Inches(3.6), Inches(6.2), Inches(1.65), border=False)
    d.card(
        s,
        X0 + Inches(6.45),
        Y0 + Inches(3.6),
        Inches(5.68),
        Inches(1.65),
        "Visibility heat map: explanation only",
        [
            "Blue = visible, orange = covered (the mask on image B). Shown to help the user, **never** used for the decision."
        ],
        TEAL,
        12.5,
        15,
    )

    # 6 ── testing and performance --------------------------------------------------------
    s = d.slide(
        "05 · Testing and performance",
        "Tested, reproducible and fast",
        "Parity means the application reproduces the offline research system exactly; it is NOT recognition accuracy (reports/experiments/phase5_parity.json). Test counts from the final quality run in the CPU and GPU environments. Speed: NVIDIA RTX 4060 Laptop GPU, Ryzen 9 7940HS, one pair per request after warm-up, medians (reports/experiments/phase5_benchmark.json). Face detection runs on the CPU and dominates. Privacy: server bound to 127.0.0.1, external requests blocked, uploads deleted after reading.",
    )
    vals = [
        ("1000 / 1000", "Identical decisions", "app = research code", BLUE),
        ("100 / 100", "CPU = GPU decisions", "min. cosine 0.99991", ACCENT),
        ("244 / 244", "Pixel-identical crops", "real photos (LFW dev)", TEAL),
        (d.tests_text, "Automated tests pass", "CPU + GPU suites", NAVY),
    ]
    for k, (v, lab, sub, col) in enumerate(vals):
        d.kpi(s, X0 + Inches(3.07) * k, Y0, Inches(2.86), Inches(1.75), v, lab, col, sub)
    d.chart(
        s,
        XL_CHART_TYPE.BAR_CLUSTERED,
        X0,
        Y0 + Inches(1.95),
        Inches(7.3),
        Inches(3.3),
        ["Diagnostics", "ArcFace embedding", "AdaFace embedding", "Detection + alignment"],
        [
            ("GPU (ms)", [5.7, 21.3, 24.2, 173.7], BLUE),
            ("CPU only (ms)", [18.1, 139.3, 165.3, 185.9], SLATE),
        ],
        fmt='0" ms"',
        vmax=220,
        title="Time per stage, one photo pair (median)",
        label_size=10,
    )
    for k, (v, lab, col) in enumerate(
        (
            ("226 ms", "GPU, two photos", BLUE),
            ("43 ms", "GPU, pre-aligned", ACCENT),
            ("509 ms", "CPU only", SLATE),
            ("550 MiB", "peak GPU memory", TEAL),
        )
    ):
        col_, row = k % 2, k // 2
        x, y = X0 + Inches(7.55) + Inches(2.32) * col_, Y0 + Inches(1.95) + Inches(1.7) * row
        d.box(s, x, y, Inches(2.2), Inches(1.55), fill=LIGHT)
        d.text(
            s, x, y + Inches(0.15), Inches(2.2), Inches(0.7), [v], 26, col, True, PP_ALIGN.CENTER
        )
        d.text(
            s,
            x,
            y + Inches(0.9),
            Inches(2.2),
            Inches(0.5),
            [lab],
            12.5,
            SLATE,
            align=PP_ALIGN.CENTER,
        )

    # 7 ── key findings ------------------------------------------------------------------
    s = d.slide(
        "06 · Key findings",
        "What the experiments showed",
        "Finding 1 (docs/phase3_results.md): when random patches were left out of training, the learned visibility rule missed 8.5 % of genuine pairs vs 2.1 % for the quality rule; its visibility estimate failed on that unseen occluder (IoU 0.13). So decisions use quality. Finding 2: the 5 % target held on data like the calibration data (1.1 to 2.1 %), but not on a harder group of synthetic identities (5.4 to 6.3 %); an independent repeat on 4,000 new identities confirmed it (5.2 to 6.5 %). Thresholds must be re-calibrated for each population. Finding 3 (reports/experiments/phase6_final.json): about 1 % of pairs fall below the calibrated quality range, and 22.97 % of those were false matches vs 1.50 % above; random-noise images even scored 1.000. The app therefore abstains below that range. This is an application safeguard, not a proven universal fix.",
    )
    cw3 = Inches(3.9)
    heads = [
        ("1 · Quality beats learned visibility", BLUE),
        ("2 · Calibration depends on the population", ORANGE),
        ("3 · Low-quality inputs need a guard", AMBER),
    ]
    for k, (t, col) in enumerate(heads):
        x = X0 + Inches(4.115) * k
        d.box(s, x, Y0, cw3, Inches(5.25), fill=LIGHT)
        d.box(s, x, Y0, cw3, Inches(0.08), fill=col, shape=MSO_SHAPE.RECTANGLE)
        d.text(
            s,
            x + Inches(0.15),
            Y0 + Inches(0.15),
            cw3 - Inches(0.3),
            Inches(0.7),
            [t],
            15,
            col,
            True,
        )
    d.chart(
        s,
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        X0 + Inches(0.1),
        Y0 + Inches(0.8),
        cw3 - Inches(0.2),
        Inches(3.0),
        ["Quality rule", "Visibility rule"],
        [("Missed matches", [2.12, 8.52], BLUE)],
        fmt='0.0"%"',
        vmax=10,
        legend=False,
        title="Unseen random patch",
    )
    ch = s.shapes[-1].chart
    pt = ch.plots[0].series[0].points[1]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = ORANGE
    d.text(
        s,
        X0 + Inches(0.15),
        Y0 + Inches(3.9),
        cw3 - Inches(0.3),
        Inches(1.3),
        [
            {
                "t": "Visibility failed on an unseen occluder (IoU 0.13) → used as an **explanation only**.",
                "size": 13,
            }
        ],
        13,
    )
    x2 = X0 + Inches(4.115)
    for k, (v, lab, col) in enumerate(
        (
            ("1.1 – 2.1 %", "matching population: within 5 % target", BLUE),
            ("5.4 – 6.3 %", "harder population: above target", ORANGE),
            ("5.2 – 6.5 %", "independent repeat, 4,000 new identities", TEAL),
        )
    ):
        y = Y0 + Inches(0.85) + Inches(1.0) * k
        d.box(s, x2 + Inches(0.15), y, cw3 - Inches(0.3), Inches(0.88), fill=WHITE, line=LINE)
        d.text(
            s,
            x2 + Inches(0.25),
            y + Inches(0.02),
            cw3 - Inches(0.5),
            Inches(0.5),
            [v],
            21,
            col,
            True,
        )
        d.text(
            s,
            x2 + Inches(0.25),
            y + Inches(0.5),
            cw3 - Inches(0.5),
            Inches(0.35),
            [lab],
            11.5,
            SLATE,
        )
    d.text(
        s,
        x2 + Inches(0.15),
        Y0 + Inches(3.9),
        cw3 - Inches(0.3),
        Inches(1.3),
        [
            {
                "t": "Missed-match rate of the decision rule. Thresholds must be **re-calibrated** for each new population.",
                "size": 13,
            }
        ],
        13,
    )
    x3 = X0 + Inches(8.23)
    d.chart(
        s,
        XL_CHART_TYPE.COLUMN_CLUSTERED,
        x3 + Inches(0.1),
        Y0 + Inches(0.8),
        cw3 - Inches(0.2),
        Inches(3.0),
        ["Below range", "Within range"],
        [("False matches", [22.97, 1.50], ORANGE)],
        fmt='0.00"%"',
        vmax=28,
        legend=False,
        title="False matches by quality",
    )
    ch = s.shapes[-1].chart
    pt = ch.plots[0].series[0].points[1]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = BLUE
    d.text(
        s,
        x3 + Inches(0.15),
        Y0 + Inches(3.9),
        cw3 - Inches(0.3),
        Inches(1.3),
        [
            {
                "t": "Even noise scored 1.000 → the app now **abstains** below the calibrated quality range (safeguard, not a proven fix).",
                "size": 13,
            }
        ],
        13,
    )

    # 8 ── limitations, future work, conclusion --------------------------------------------
    s = d.slide(
        "07 · Conclusion",
        "Limitations, future work and conclusion",
        "Be balanced: the limitations are measured and also shown inside the app. Future work addresses each one. Conclude, then start the live demo in this order: example 2 (MATCH), example 1 (NON-MATCH), example 4 (ABSTAIN), then two noise images (support guard). References: Kim et al. 2022 (AdaFace, CVPR); Deng et al. 2019 (ArcFace, CVPR); Guo et al. 2022 (SCRFD, ICLR); Bae et al. 2023 (DigiFace-1M, WACV); Angelopoulos et al. 2021 (Learn then Test, arXiv:2110.01052); Ronneberger et al. 2015 (U-Net, MICCAI).",
    )
    d.card(
        s,
        X0,
        Y0,
        Inches(3.85),
        Inches(5.25),
        "Limitations",
        [
            {
                "t": "Synthetic faces and occlusions; no real physical occlusions tested",
                "bullet": True,
                "after": 7,
            },
            {
                "t": "5 % target failed on a harder population (confirmed twice)",
                "bullet": True,
                "after": 7,
            },
            {
                "t": "Low-quality false matches, guarded only in the app",
                "bullet": True,
                "after": 14,
            },
            {"t": "Not certified; not for real identity decisions", "bullet": True},
        ],
        AMBER,
        15.5,
    )
    d.card(
        s,
        X0 + Inches(4.05),
        Y0,
        Inches(3.85),
        Inches(5.25),
        "Future work",
        [
            {
                "t": "Re-calibrate on the target users' photos and camera",
                "bullet": True,
                "after": 7,
            },
            {"t": "Test on real occluded photos (e.g. IJB-C)", "bullet": True, "after": 14},
            {
                "t": "A score that cannot extrapolate below its quality range",
                "bullet": True,
                "after": 7,
            },
            {"t": "GPU face detection; clickable face selection", "bullet": True},
        ],
        BLUE,
        15.5,
    )
    xc = X0 + Inches(8.1)
    d.box(s, xc, Y0, Inches(4.03), Inches(5.25), fill=NAVY)
    d.text(
        s,
        xc + Inches(0.25),
        Y0 + Inches(0.2),
        Inches(3.6),
        Inches(2.4),
        [
            {"t": "CONCLUSION", "size": 12, "bold": True, "color": RGBColor(0x8F, 0xB8, 0xEC)},
            {
                "t": "A working, private face verification app that **abstains** when unsure and **explains** every result, with measured, honestly reported limits.",
                "size": 14.5,
                "color": WHITE,
            },
        ],
        14,
        WHITE,
    )
    d.text(
        s,
        xc + Inches(0.25),
        Y0 + Inches(2.55),
        Inches(3.6),
        Inches(0.35),
        ["LIVE DEMO"],
        12,
        RGBColor(0x8F, 0xB8, 0xEC),
        True,
    )
    for k, (lab, col) in enumerate((("MATCH", BLUE), ("NON-MATCH", SLATE), ("ABSTAIN", AMBER))):
        d.chip(
            s,
            xc + Inches(0.25) + Inches(1.2) * k,
            Y0 + Inches(2.95),
            Inches(1.12),
            Inches(0.42),
            lab,
            col,
            10.5,
        )
        s.shapes[-1].line.color.rgb = WHITE
    d.text(
        s,
        xc + Inches(0.25),
        Y0 + Inches(3.5),
        Inches(3.6),
        Inches(0.5),
        ["+ support guard on noise images"],
        12,
        RGBColor(0xDD, 0xE6, 0xF2),
    )
    d.text(
        s,
        xc + Inches(0.25),
        Y0 + Inches(4.2),
        Inches(3.6),
        Inches(0.7),
        ["Thank you. Questions?"],
        22,
        WHITE,
        True,
    )

    d.save(path)
