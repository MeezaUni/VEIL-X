"""Procedural occlusion engine (F1-F5) and held-out-aware condition planning."""

from __future__ import annotations

from veilx.occlusion.engine import (
    GENERATOR_VERSION,
    OcclusionResult,
    StressResult,
    apply_occlusion,
    apply_stress,
    mask_sha256,
    visibility_grid,
)
from veilx.occlusion.families import LEVELS, PHYSICAL
from veilx.occlusion.geometry import ARCFACE_112, Landmarks, face_region

__all__ = [
    "ARCFACE_112",
    "GENERATOR_VERSION",
    "LEVELS",
    "PHYSICAL",
    "Landmarks",
    "OcclusionResult",
    "StressResult",
    "apply_occlusion",
    "apply_stress",
    "face_region",
    "mask_sha256",
    "visibility_grid",
]
