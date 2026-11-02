# ruff: noqa: E501, RUF001, RUF002  (user-facing sentences; typographic characters)
"""Plain-language explanation of one VEIL-X comparison: what was checked, the exact numbers, which
rule applied and why the result is MATCH, NON-MATCH or ABSTAIN.

The explanation **describes the frozen decision rule**. It does not make causal claims about the
images. Each step reports the same quantities that the decision used: AdaFace cosine similarity
``s``, pair quality ``q`` (smaller AdaFace feature norm), the fused score ``p``, the quality
group's frozen thresholds and the support guard.

Reference ranges ("typical same-person similarity …") come from development data
(``configs/app/explain_reference.json``). They give context only and never change a decision.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GROUP_NAMES = {"q_low": "low quality", "q_mid": "medium quality", "q_high": "high quality"}
REGION_NAMES = {
    "left_eye": "left eye",
    "right_eye": "right eye",
    "nose": "nose",
    "mouth": "mouth",
    "lower_face": "lower face",
}


@dataclass
class Step:
    title: str
    status: str  # "ok" | "warn" | "stop" | "info" | "skip"
    lines: list[str] = field(default_factory=list)


@dataclass
class Explanation:
    label: str
    headline: str
    summary: str
    steps: list[Step]
    what_would_change: list[str]
    confidence: str  # short qualitative phrase about how clear-cut the decision is


def load_reference(path: Path) -> dict[str, Any] | None:
    try:
        return dict(json.loads(path.read_text("utf-8")))
    except (OSError, ValueError):
        return None


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


def similarity_level(s: float, ref: dict[str, Any] | None) -> tuple[str, str]:
    """(short level, sentence) placing s against development-data reference ranges."""
    if ref is None:
        return ("", "")
    same, diff = ref["similarity_same_person"], ref["similarity_different_people"]
    if s >= same["p25"]:
        return (
            "high",
            f"High: typical of the **same person** (same-person pairs: median {same['p50']:.2f}; different-person pairs rarely exceed {diff['p90']:.2f}).",
        )
    if s <= diff["p75"]:
        return (
            "low",
            f"Low: typical of **different people** (different-person pairs: median {diff['p50']:.2f}; same-person pairs usually above {same['p10']:.2f}).",
        )
    return (
        "medium",
        f"In between: overlaps both ranges (different people usually below {diff['p90']:.2f}, same person usually above {same['p25']:.2f}), so similarity alone is ambiguous.",
    )


def _face_line(name: str, f: Any) -> tuple[str, str]:
    """(status, sentence) for one image's face preparation."""
    st = f.status
    if st == "aligned":
        d = f.faces[f.chosen or 0] if f.faces else None
        extra = f" (detector confidence {d.score:.2f})" if d is not None else ""
        return (
            "ok",
            f"Image {name}: one face found{extra} and aligned to the standard 112×112 position.",
        )
    if st == "selected_face":
        return (
            "ok",
            f"Image {name}: {len(f.faces)} faces found; you selected face {f.chosen}, which was aligned.",
        )
    if st == "pre_aligned":
        return (
            "ok",
            f"Image {name}: used as an already-aligned 112×112 face crop (detection skipped).",
        )
    if st == "fallback_fixed_crop":
        return (
            "warn",
            f"Image {name}: no reliable face detected, so a fixed centre crop was used (you allowed this for portrait-style photos). Treat the result with extra caution.",
        )
    if st == "multiple_faces":
        return (
            "stop",
            f"Image {name}: {len(f.faces)} faces found. VEIL-X never chooses a person for you: enter the face number shown on the detection image in **Advanced options → Face to use in image {name}**.",
        )
    if st == "no_reliable_face":
        why = f.notes[0].split(";")[0] if f.notes else "no usable face"
        return (
            "stop",
            f"Image {name}: no usable face ({why}). Use a clear, front-facing photo, or, for a loosely cropped single-face portrait, tick the fixed-crop FALLBACK (LFW-style framing) in **Advanced options**.",
        )
    why = "; ".join(f.notes) if f.notes else "invalid input"
    valid = (
        f" It has {len(f.faces)} face(s): use a number from 0 to {len(f.faces) - 1}, or -1 if there is only one."
        if f.faces
        else ""
    )
    return "stop", f"Image {name}: could not be used ({why}).{valid}"


def build_explanation(
    res: Any, rule: Any, floor: float | None, ref: dict[str, Any] | None
) -> Explanation:
    dec = res.decision
    steps: list[Step] = []
    change: list[str] = []
    steps.append(
        Step(
            "Images checked",
            "ok",
            ["Both files are valid images (supported format, size and content)."],
        )
    )

    face_lines = [_face_line("A", res.face_a), _face_line("B", res.face_b)]
    stopped = any(s == "stop" for s, _ in face_lines)
    face_status = (
        "stop" if stopped else ("warn" if any(s == "warn" for s, _ in face_lines) else "ok")
    )
    steps.append(Step("Faces found and aligned", face_status, [line for _, line in face_lines]))

    if stopped or res.s_adaface is None or res.quality is None:
        for title in ("Similarity", "Image quality", "Combined score", "Decision rule"):
            steps.append(
                Step(title, "skip", ["Not reached: a usable face is needed in both images first."])
            )
        problems = [line for s, line in face_lines if s == "stop"]
        change.append("Fix the image problem above, then compare again.")
        return Explanation(
            dec.label,
            "VEIL-X could not compare these images",
            "One of the images could not be used, so no identity decision was made. "
            + (problems[0] if problems else ""),
            steps,
            change,
            "no decision",
        )

    s, q, p = res.s_adaface, res.quality, dec.fused_score
    lvl, sim_sentence = similarity_level(s, ref)
    steps.append(
        Step(
            "How similar are the two faces?",
            "info",
            [
                f"AdaFace similarity **s = {s:.3f}** (cosine between the two 512-number face embeddings; 1 = identical direction, 0 = unrelated).",
                *([sim_sentence] if sim_sentence else []),
            ],
        )
    )

    lo_e, hi_e = rule.q_edges
    group = dec.stratum
    weaker = "A" if (res.norm_a or 0) <= (res.norm_b or 0) else "B"
    q_lines = [
        f"Quality of each face (AdaFace feature norm): A = {_f(res.norm_a, 2)}, B = {_f(res.norm_b, 2)}. The pair is only as good as its weaker image, so **q = {q:.2f}** (image {weaker}).",
        f"Quality groups of this calibration: low below {lo_e:.1f}, medium {lo_e:.1f}–{hi_e:.1f}, high {hi_e:.1f} and above. This pair is in the **{GROUP_NAMES.get(group, group)}** group.",
    ]
    if group == "q_low":
        q_lines.append(
            "Low quality usually means blur, darkness, a small or turned face, or covered face parts."
        )
    guard = floor is not None and q < floor
    if guard:
        q_lines.append(
            f"q is **below {floor:.2f}**, the lowest quality seen during calibration (1st percentile). The score formula is unreliable there."
        )
    steps.append(
        Step("How good are the images?", "warn" if (group == "q_low" or guard) else "ok", q_lines)
    )

    steps.append(
        Step(
            "Combined score",
            "info",
            [
                f"The calibrated formula combines similarity and quality into **score p = {_f(p)}** (0 to 1). It is a decision score, **not** the probability that it is the same person."
            ],
        )
    )

    thr = rule.rules.get(group)
    rule_lines: list[str] = []
    confidence = ""
    if guard:
        rule_lines.append(
            f"**Support guard:** q = {q:.2f} < {floor:.2f}, so VEIL-X abstains whatever the score."
        )
        unguarded = next((n for n in res.notes if "frozen rule alone would have output" in n), "")
        if unguarded:
            would = unguarded.split("would have output")[-1].strip()
            rule_lines.append(
                f"Without the guard, the research rule would have answered {html.escape(would)}."
                + (
                    " At such low quality the rule has produced confident false matches, even for non-face images."
                    if would.startswith("MATCH")
                    else ""
                )
            )
        change.append(
            "Use clearer, well-lit photos of a real, unobstructed face. Very low quality inputs are always refused."
        )
        confidence = "refused: quality too low"
    elif thr is None:
        rule_lines.append(
            f"No calibrated rule exists for the {GROUP_NAMES.get(group, group)} group, so VEIL-X abstains."
        )
        confidence = "no rule"
    else:
        lo, hi = thr
        if math.isclose(lo, hi):
            rule_lines.append(
                f"In the {GROUP_NAMES.get(group, group)} group the rule has one cut-off: **MATCH if p ≥ {hi:.3f}, otherwise NON-MATCH**."
            )
            if dec.label == "MATCH":
                rule_lines.append(f"p = {_f(p)} ≥ {hi:.3f} → **MATCH**.")
            else:
                rule_lines.append(f"p = {_f(p)} < {hi:.3f} → **NON-MATCH**.")
            margin = abs(p - hi)
            confidence = (
                "clear-cut"
                if margin >= 0.25
                else ("fairly clear" if margin >= 0.08 else "close to the cut-off")
            )
            rule_lines.append(f"Distance from the cut-off: {margin:.3f} ({confidence}).")
            if confidence == "close to the cut-off":
                change.append(
                    "The score is close to the cut-off: a slightly different photo could change the answer. Treat this result with caution."
                )
        else:
            rule_lines.append(
                f"In the low-quality group the rule decides only when the evidence is clear: **NON-MATCH if p ≤ {lo:.3f}**, **MATCH if p ≥ {hi:.3f}**, and **ABSTAIN in between**."
            )
            if dec.label == "ABSTAIN":
                rule_lines.append(
                    f"p = {_f(p)} lies between {lo:.3f} and {hi:.3f} → **ABSTAIN** (uncertain zone)."
                )
                change.append(
                    f"A MATCH would need p ≥ {hi:.3f}; a NON-MATCH would need p ≤ {lo:.3f}."
                )
                change.append(
                    f"Better-quality photos (q of at least {lo_e:.1f}) move the pair into a group that uses a single cut-off."
                )
                confidence = "uncertain zone"
            else:
                edge = hi if dec.label == "MATCH" else lo
                rule_lines.append(
                    f"p = {_f(p)} is {'≥' if dec.label == 'MATCH' else '≤'} {edge:.3f} → **{dec.label}**, even at low quality."
                )
                confidence = "decided despite low quality"
    steps.append(
        Step("Which rule decided?", "stop" if dec.label == "ABSTAIN" else "ok", rule_lines)
    )

    if res.s_arcface is not None:
        gap = abs(res.s_adaface - res.s_arcface)
        agree = (
            "the two models broadly agree"
            if gap < 0.15
            else "the two models disagree noticeably. Treat the result with extra caution"
        )
        steps.append(
            Step(
                "Second opinion (not used for the decision)",
                "info",
                [
                    f"ArcFace similarity {res.s_arcface:.3f} vs AdaFace {res.s_adaface:.3f}: {agree}."
                ],
            )
        )

    d = res.diagnostics
    if d:
        vis_lines = []
        for k, name in enumerate(("A", "B")):
            covered = [REGION_NAMES.get(r, r) for r, v in d["regions"].items() if v[k] < 0.5]
            frac = d["visible_fraction"][k]
            vis_lines.append(
                f"Image {name}: about {round(100 * frac)} % of the face predicted visible"
                + (f"; possibly covered: {', '.join(covered)}." if covered else ".")
            )
            if d["novel"][k]:
                vis_lines.append(
                    f"Image {name} looks **unusual** compared with the visibility network's training images (novelty {d['novelty_scores'][k]:.1f} > {d['novelty_threshold']:.1f})."
                )
        vis_lines.append(
            "These maps are explanations only and never change the decision. They can miss occluders that look like skin or faces."
        )
        steps.append(Step("Where might the faces be covered?", "info", vis_lines))

    label = dec.label
    if guard:
        headline = "Not enough reliable evidence: image quality too low"
        summary = f"The image quality (q = {q:.2f}) is below anything VEIL-X was calibrated on. The input may not be a usable face, so VEIL-X declines to decide."
    elif label == "MATCH":
        headline = "Likely the same person"
        summary = f"The faces are {'very ' if lvl == 'high' else ''}similar (similarity {s:.2f}) and the score {_f(p)} passes the {GROUP_NAMES.get(group, group)} cut-off, so VEIL-X decides they likely show the same person."
    elif label == "NON-MATCH":
        headline = "Likely different people"
        how = "not similar" if lvl == "low" else "not similar enough"
        summary = f"The faces are {how} (similarity {s:.2f}): the score {_f(p)} is below the {GROUP_NAMES.get(group, group)} cut-off, so VEIL-X decides they likely show different people."
    else:
        headline = "Not enough evidence to decide"
        summary = f"Image quality is low (q = {q:.2f}) and the score {_f(p)} falls in the uncertain zone, so VEIL-X declines to decide instead of guessing."
    return Explanation(label, headline, summary, steps, change, confidence)


# ---------------------------------------------------------------------------------- rendering
_ICON = {
    "ok": ("✓", "#1b7f5a"),
    "warn": ("!", "#9a5b00"),
    "stop": ("✕", "#b3261e"),
    "info": ("i", "#1d4e89"),
    "skip": ("–", "#8a8a8a"),
}


def _md_bold(s: str) -> str:
    out, bold = [], False
    for k, part in enumerate(html.escape(s).split("**")):
        if k:
            bold = not bold
        out.append(f"<b>{part}</b>" if bold else part)
    return "".join(out)


def steps_html(ex: Explanation) -> str:
    items = []
    for k, st in enumerate(ex.steps, start=1):
        ic, col = _ICON.get(st.status, _ICON["info"])
        lines = "".join(f"<li>{_md_bold(line)}</li>" for line in st.lines)
        items.append(
            f"<div class='vx-step vx-{st.status}'><div class='vx-step-icon' style='background:{col}'>{ic}</div>"
            f"<div class='vx-step-body'><div class='vx-step-title'>{k}. {html.escape(st.title)}</div><ul>{lines}</ul></div></div>"
        )
    change = ""
    if ex.what_would_change:
        change = (
            "<div class='vx-change'><div class='vx-step-title'>What would change the result?</div><ul>"
            + "".join(f"<li>{_md_bold(c)}</li>" for c in ex.what_would_change)
            + "</ul></div>"
        )
    return (
        "<div class='vx-steps'>"
        + "".join(items)
        + change
        + "<div class='vx-note'>These reasons describe how the frozen decision rule reached its answer. They are not a statement about who the people are.</div></div>"
    )


# ------------------------------------------------------------------- result panels (HTML)
VERDICT_STYLE = {"MATCH": "match", "NON-MATCH": "nonmatch", "ABSTAIN": "abstain"}


def verdict_html(ex: Explanation, meta: str) -> str:
    kind = VERDICT_STYLE.get(ex.label, "abstain")
    if ex.confidence == "no decision":
        kind = "error"
    label = "NO DECISION" if kind == "error" else ex.label
    return (
        f"<div class='vx-verdict vx-v-{kind}' role='status'>"
        f"<div class='vx-v-label'>{html.escape(label)}</div>"
        f"<div class='vx-v-head'>{html.escape(ex.headline)}</div>"
        f"<div class='vx-v-sum'>{_md_bold(ex.summary)}</div>"
        f"<div class='vx-v-meta'>{html.escape(meta)}</div></div>"
    )


def _card(title: str, value: str, sub: str) -> str:
    return (
        f"<div class='vx-card'><div class='vx-card-t'>{html.escape(title)}</div>"
        f"<div class='vx-card-v'>{html.escape(value)}</div><div class='vx-card-s'>{html.escape(sub)}</div></div>"
    )


def metrics_html(res: Any, rule: Any, floor: float | None, ref: dict[str, Any] | None) -> str:
    if res.s_adaface is None or res.quality is None:
        return ""
    dec = res.decision
    lvl, _ = similarity_level(res.s_adaface, ref)
    lvl_txt = {
        "high": "same-person range",
        "low": "different-person range",
        "medium": "ambiguous range",
    }.get(lvl, "AdaFace cosine")
    weaker = "A" if (res.norm_a or 0) <= (res.norm_b or 0) else "B"
    guard = floor is not None and res.quality < floor
    thr = rule.rules.get(dec.stratum)
    if guard:
        rule_txt, rule_sub = "Guard", f"q < {floor:.2f}: always abstain"
    elif thr is None:
        rule_txt, rule_sub = "None", "no rule: abstain"
    elif math.isclose(thr[0], thr[1]):
        rule_txt, rule_sub = f"≥ {thr[1]:.3f}", "score needed for MATCH"
    else:
        rule_txt, rule_sub = f"{thr[0]:.3f} – {thr[1]:.3f}", "uncertain zone (abstain)"
    return (
        "<div class='vx-cards'>"
        + "".join(
            [
                _card("Similarity", f"{res.s_adaface:.3f}", lvl_txt),
                _card(
                    "Image quality",
                    f"{res.quality:.2f}",
                    f"{GROUP_NAMES.get(dec.stratum, dec.stratum)} · weaker: {weaker}",
                ),
                _card("Decision score", _f(dec.fused_score), "0 to 1 · not a probability"),
                _card("Rule applied", rule_txt, rule_sub),
            ]
        )
        + "</div>"
    )


def score_bar_html(p: float, tau_low: float | None, tau_high: float | None, guard: bool) -> str:
    """The quality group's rule drawn on the 0–1 score axis, with this pair's score marked."""
    if p is None or not math.isfinite(p):
        return ""

    def pct(v: float) -> float:
        return 100.0 * min(max(v, 0.0), 1.0)

    zones: list[tuple[float, float, str, str]] = []
    if guard or tau_low is None or tau_high is None:
        zones.append((0.0, 1.0, "abstain", "ABSTAIN (quality guard)" if guard else "ABSTAIN"))
    else:
        zones.append((0.0, tau_low if tau_high > tau_low else tau_high, "nonmatch", "NON-MATCH"))
        if tau_high > tau_low:
            zones.append((tau_low, tau_high, "abstain", "ABSTAIN"))
        zones.append((tau_high, 1.0, "match", "MATCH"))
    segs = "".join(
        f"<div class='vx-zone vx-z-{k}' style='left:{pct(a):.2f}%;width:{pct(b) - pct(a):.2f}%'><span>{t}</span></div>"
        for a, b, k, t in zones
        if b > a
    )
    ticks = sorted({0.0, 1.0, *(v for v in (tau_low, tau_high) if v is not None and not guard)})
    tick_html = "".join(
        f"<div class='vx-tick' style='left:{pct(v):.2f}%'>{v:.3g}</div>" for v in ticks
    )
    note = (
        "Score shown for transparency only: the quality guard overrides it."
        if guard
        else "Where this pair's score falls on the rule for its quality group."
    )
    # keep the score label inside the bar near either end
    label_pos = (
        "left:auto;right:-2px;transform:none"
        if p > 0.85
        else ("left:-2px;transform:none" if p < 0.15 else "")
    )
    return (
        "<div class='vx-bar-wrap'>"
        f"<div class='vx-bar'>{segs}<div class='vx-marker' style='left:{pct(p):.2f}%'><div class='vx-marker-l' style='{label_pos}'>score {p:.3f}</div></div></div>"
        f"<div class='vx-ticks'>{tick_html}</div><div class='vx-bar-note'>{note}</div></div>"
    )
