"""Plain-language explanations: they must state exactly the rule the frozen decision applied."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from veilx.app.config import FrozenRule
from veilx.app.decision import decide
from veilx.app.explain import (
    build_explanation,
    load_reference,
    metrics_html,
    score_bar_html,
    similarity_level,
    steps_html,
    verdict_html,
)
from veilx.app.face import FaceBox, PreparedFace
from veilx.app.pipeline import VerificationResult
from veilx.fusion.logreg import LogisticModel

from .conftest import ROOT

REF_PATH = ROOT / "configs" / "app" / "explain_reference.json"


def _rule() -> FrozenRule:
    m = LogisticModel(("s", "q", "s*q"), 10.0)
    m.mean, m.scale = np.zeros(3), np.ones(3)
    m.coef, m.intercept = np.array([10.0, 0.0, 0.0]), -3.0  # p = sigmoid(10 s - 3)
    rules = {"q_low": (0.2, 0.8), "q_mid": (0.5, 0.5), "q_high": (0.5, 0.5)}
    return FrozenRule("F1_mask", m, (19.0, 22.0), rules, 0.5)  # type: ignore[arg-type]


def _face(status: str = "pre_aligned", n: int = 0) -> PreparedFace:
    crop = np.zeros((112, 112, 3), np.uint8) if status in ("pre_aligned", "aligned") else None
    faces = [FaceBox((0, 0, 10, 10), 0.9, 0.5) for _ in range(n)]
    return PreparedFace(status, crop, faces, chosen=0 if status == "aligned" else None)


def _result(s: float, q: float, floor: float | None = None) -> VerificationResult:
    rule = _rule()
    dec = decide(rule, s, q)
    notes = []
    if floor is not None and q < floor:
        notes.append(
            "support guard: ... the frozen rule alone would have output "
            f"{dec.label} (fused score {dec.fused_score:.3f})"
        )
        dec = decide(rule, s, q, forced_reasons=("below support",))
    r = VerificationResult(
        dec, _face(), _face(), "F1_mask", "cpu", s, s - 0.02, q + 1.0, q, q, {"total": 5.0}
    )
    r.notes.extend(notes)
    return r


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    out = load_reference(REF_PATH)
    assert out is not None
    return out


def test_reference_file_is_aggregate_and_ordered(ref: dict[str, Any]) -> None:
    for key in ("similarity_same_person", "similarity_different_people", "quality"):
        vals = [ref[key][f"p{k}"] for k in (10, 25, 50, 75, 90)]
        assert vals == sorted(vals)
    assert ref["similarity_same_person"]["p50"] > ref["similarity_different_people"]["p90"]
    assert set(ref) <= {
        "scope",
        "n_pairs",
        "similarity_same_person",
        "similarity_different_people",
        "quality",
        "provenance",
    }
    assert load_reference(ROOT / "does_not_exist.json") is None


@pytest.mark.parametrize("q", [15.0, 20.0, 25.0])
def test_explanation_states_the_applied_rule_exactly(q: float, ref: dict[str, Any]) -> None:
    """For every outcome the explanation names the same label and the exact comparison used."""
    rule = _rule()
    for s in np.linspace(-0.2, 1.0, 61):
        res = _result(float(s), q)
        dec = res.decision
        ex = build_explanation(res, rule, None, ref)
        assert ex.label == dec.label
        decided = next(st for st in ex.steps if st.title == "Which rule decided?")
        text = " ".join(decided.lines)
        assert f"→ **{dec.label}**" in text
        thr = rule.rules[dec.stratum]
        assert thr is not None
        p, (lo, hi) = dec.fused_score, thr
        if dec.label == "MATCH":
            assert p >= hi and f"≥ {hi:.3f}" in text
            assert ex.headline == "Likely the same person"
        elif dec.label == "NON-MATCH":
            assert p <= lo or (math.isclose(lo, hi) and p < hi)
            assert ex.headline == "Likely different people"
        else:
            assert dec.stratum == "q_low" and lo < p < hi
            assert any(f"p ≥ {hi:.3f}" in c for c in ex.what_would_change)
            assert any(f"p ≤ {lo:.3f}" in c for c in ex.what_would_change)


def test_support_guard_is_explained_with_the_unguarded_output(ref: dict[str, Any]) -> None:
    res = _result(0.9, 8.0, floor=10.0)
    assert res.decision.label == "ABSTAIN"
    ex = build_explanation(res, _rule(), 10.0, ref)
    text = " ".join(" ".join(st.lines) for st in ex.steps)
    assert (
        "Support guard" in text and "8.00 < 10.00" in text and "would have answered MATCH" in text
    )
    assert "quality too low" in ex.headline
    bar = score_bar_html(
        res.decision.fused_score, res.decision.tau_low, res.decision.tau_high, guard=True
    )
    assert "quality guard" in bar and bar.count("vx-zone ") == 1


def test_unusable_face_explains_what_to_do() -> None:
    from veilx.app.decision import Decision

    dec = Decision("ABSTAIN", 0, float("nan"), "n/a", None, None, ("image A: multiple faces",))
    res = VerificationResult(dec, _face("multiple_faces", 2), _face(), "F1_mask", "cpu")
    ex = build_explanation(res, _rule(), 10.0, None)
    assert ex.confidence == "no decision" and "could not compare" in ex.headline
    text = " ".join(" ".join(st.lines) for st in ex.steps)
    assert "2 faces found" in text and "Advanced options" in text
    assert [st.status for st in ex.steps][-1] == "skip"
    assert "NO DECISION" in verdict_html(ex, "meta")
    assert metrics_html(res, _rule(), 10.0, None) == ""
    assert score_bar_html(float("nan"), None, None, guard=False) == ""


def test_similarity_levels_follow_reference_ranges(ref: dict[str, Any]) -> None:
    assert similarity_level(0.9, ref)[0] == "high"
    assert similarity_level(0.0, ref)[0] == "low"
    assert similarity_level(0.3, ref)[0] == "medium"
    assert similarity_level(0.5, None) == ("", "")


def test_score_bar_zones_match_the_rule() -> None:
    band = score_bar_html(0.5, 0.2, 0.8, guard=False)
    single = score_bar_html(0.9, 0.5, 0.5, guard=False)
    assert band.count("vx-zone ") == 3 and "ABSTAIN" in band
    assert single.count("vx-zone ") == 2 and "ABSTAIN" not in single
    assert "left:90.00%" in single  # marker at the score


def test_rendered_html_escapes_text() -> None:
    res = _result(0.9, 25.0)
    ex = build_explanation(res, _rule(), None, None)
    ex.steps[0].lines.append("<script>alert(1)</script> **bold**")
    out = steps_html(ex)
    assert "<script>" not in out and "&lt;script&gt;" in out and "<b>bold</b>" in out
    assert "<img" not in verdict_html(ex, "<img src=x>")


def test_reference_json_is_valid_json() -> None:
    json.loads(REF_PATH.read_text("utf-8"))
