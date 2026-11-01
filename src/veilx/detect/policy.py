"""Detector-failure taxonomy and handling policies (Phase 4F, docs/phase4_protocol.md section 6).

Every image gets exactly one category; failed detections are never dropped from reported rates.

* ``no_face``         - the detector returned no face;
* ``low_quality``     - a face was found but detector score < ``min_score`` or the 5-point
                        alignment residual exceeds ``max_residual`` (fixed on clean development
                        images);
* ``multiple_faces``  - more than one face; the central one is used (flagged, still processed);
* ``success``         - one face, adequate score and residual.

Policies for ``no_face`` / ``low_quality`` sides (the pair decision is forced or re-routed):
``abstain``, ``non_match`` (forced rejection), ``fallback`` (detector-free fixed-crop embedding
scored by the quality-stratified rule).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from veilx.metrics.selective import ABSTAIN, NON_MATCH

Category = Literal["no_face", "low_quality", "multiple_faces", "success"]
Policy = Literal["abstain", "non_match", "fallback"]
POLICIES: tuple[Policy, ...] = ("abstain", "non_match", "fallback")


def categorize(
    n_faces: int,
    score: float | None,
    residual: float | None,
    *,
    min_score: float,
    max_residual: float,
) -> Category:
    if n_faces == 0 or score is None or residual is None:
        return "no_face"
    if score < min_score or residual > max_residual:
        return "low_quality"
    if n_faces > 1:
        return "multiple_faces"
    return "success"


def needs_policy(category: str) -> bool:
    return category in ("no_face", "low_quality")


def apply_policy(
    policy: Policy,
    categories: NDArray[np.str_],
    decisions: NDArray[np.int64],
    fallback_decisions: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Final decisions given per-pair categories of the (occluded) side. ``decisions`` are the
    normal decisions (only meaningful where the category does not need a policy)."""
    d = np.asarray(decisions, np.int64).copy()
    m = np.array([needs_policy(str(c)) for c in categories], dtype=bool)
    if policy == "abstain":
        d[m] = ABSTAIN
    elif policy == "non_match":
        d[m] = NON_MATCH
    else:
        if fallback_decisions is None:
            raise ValueError("fallback policy needs fallback decisions")
        d[m] = np.asarray(fallback_decisions, np.int64)[m]
    return d
