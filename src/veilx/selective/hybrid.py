"""Quality-first hybrid strata (docs/phase4_protocol.md 4.4).

A pair is *reliable* when no side is flagged novel. Reliable pairs are stratified by
(conservative) predicted co-visibility and scored by the visibility-aware fused model; unreliable
pairs fall back to quality strata and the quality-only fused model. The six strata are a fixed,
pre-specified function of each pair, so per-stratum Learn-then-Test calibration of the WHOLE rule
(delta / 6 per stratum) is valid on the calibration distribution. Nothing is inherited from the
stand-alone quality arm: the fallback changes which pairs each stratum contains, so its rules are
recalibrated here. No out-of-distribution guarantee follows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

STRATA = ("v_low", "v_mid", "v_high", "q_low", "q_mid", "q_high")


def tertile(x: NDArray[np.floating], edges: tuple[float, float], prefix: str) -> NDArray[np.str_]:
    lo, hi = edges
    v = np.asarray(x, np.float64)
    return np.where(v < lo, f"{prefix}_low", np.where(v < hi, f"{prefix}_mid", f"{prefix}_high"))


def hybrid_strata(
    reliable: NDArray[np.bool_],
    covis: NDArray[np.floating],
    quality: NDArray[np.floating],
    cv_edges: tuple[float, float],
    q_edges: tuple[float, float],
) -> NDArray[np.str_]:
    rel = np.asarray(reliable, bool)
    return np.where(rel, tertile(covis, cv_edges, "v"), tertile(quality, q_edges, "q"))


def hybrid_score(
    reliable: NDArray[np.bool_],
    score_visibility: NDArray[np.floating],
    score_quality: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Per-pair score used inside each stratum: visibility-aware model if reliable, else the
    quality-only model (strata never mix the two scores)."""
    return np.asarray(
        np.where(np.asarray(reliable, bool), score_visibility, score_quality), dtype=np.float64
    )


def novelty_forced_strata(
    novel: NDArray[np.bool_], covis: NDArray[np.floating], cv_edges: tuple[float, float]
) -> NDArray[np.str_]:
    """N4: visibility strata, but pairs flagged novel are forced into the lowest stratum."""
    s = tertile(covis, cv_edges, "v")
    return np.where(np.asarray(novel, bool), "v_low", s)
