"""Frozen A3 decision (pure numpy; same as scripts/evaluate_phase3.py ``decide`` for one pair).

Inputs are the AdaFace cosine ``s`` and the pair quality ``q`` (the smaller of the two
pre-normalisation feature norms). The fused score is M1 = logistic(s, q, s*q). The quality
stratum is set by the frozen tertile edges, and the stratum's frozen rule gives MATCH
(score >= tau_high), NON-MATCH (score <= tau_low and < tau_high) or ABSTAIN. Nothing else (no
visibility, novelty or ArcFace value) enters the decision.

The fused score is a logistic output fitted on synthetic DigiFace pairs. It is **not** a
probability that two images show the same person, and it is not a confidence guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from veilx.app.config import FrozenRule
from veilx.metrics.selective import ABSTAIN, MATCH, NON_MATCH, decisions_from_thresholds

LABELS = {MATCH: "MATCH", NON_MATCH: "NON-MATCH", ABSTAIN: "ABSTAIN"}


@dataclass(frozen=True)
class Decision:
    label: str  # MATCH / NON-MATCH / ABSTAIN
    code: int
    fused_score: float
    stratum: str
    tau_low: float | None
    tau_high: float | None
    reasons: tuple[str, ...] = field(default_factory=tuple)


def quality_stratum(q: float, edges: tuple[float, float]) -> str:
    lo, hi = edges
    return "q_low" if q < lo else ("q_mid" if q < hi else "q_high")


def fused_score(rule: FrozenRule, s: float, q: float) -> float:
    x = np.array([[s, q, s * q]], dtype=np.float64)
    return float(rule.model.predict_proba(x)[0])


def decide(
    rule: FrozenRule, s: float, q: float, *, forced_reasons: tuple[str, ...] = ()
) -> Decision:
    """A3 decision for one pair. ``forced_reasons`` (e.g. a face could not be prepared) force
    ABSTAIN and are recorded. The fused score is still reported for transparency."""
    p = fused_score(rule, s, q)
    st = quality_stratum(q, rule.q_edges)
    thr = rule.rules.get(st)
    if forced_reasons:
        lo_f, hi_f = thr if thr is not None else (None, None)
        return Decision("ABSTAIN", ABSTAIN, p, st, lo_f, hi_f, reasons=forced_reasons)
    if thr is None:
        return Decision(
            "ABSTAIN",
            ABSTAIN,
            p,
            st,
            None,
            None,
            reasons=(f"no certified rule for quality stratum {st}",),
        )
    lo, hi = thr
    code = int(decisions_from_thresholds(np.array([p]), lo, hi)[0])
    reasons: tuple[str, ...] = ()
    if code == ABSTAIN:
        reasons = (
            f"fused score {p:.3f} lies inside the abstention band [{lo:.3f}, {hi:.3f}) of the "
            f"{st.replace('_', '-')} quality stratum",
        )
    return Decision(LABELS[code], code, p, st, lo, hi, reasons)


def rule_description(rule: FrozenRule) -> list[dict[str, str]]:
    """Human-readable table of the frozen rule (for the UI)."""
    lo, hi = rule.q_edges
    span = {
        "q_low": f"q < {lo:.2f}",
        "q_mid": f"{lo:.2f} <= q < {hi:.2f}",
        "q_high": f"q >= {hi:.2f}",
    }
    out = []
    for st in ("q_low", "q_mid", "q_high"):
        thr = rule.rules.get(st)
        if thr is None:
            out.append(
                {"stratum": st, "quality": span[st], "rule": "always ABSTAIN (no certified rule)"}
            )
        elif thr[0] == thr[1]:
            out.append(
                {
                    "stratum": st,
                    "quality": span[st],
                    "rule": f"MATCH if score >= {thr[1]:.3f}, else NON-MATCH (no abstention band)",
                }
            )
        else:
            out.append(
                {
                    "stratum": st,
                    "quality": span[st],
                    "rule": (
                        f"MATCH if score >= {thr[1]:.3f}; NON-MATCH if score <= {thr[0]:.3f}; "
                        "else ABSTAIN"
                    ),
                }
            )
    return out
