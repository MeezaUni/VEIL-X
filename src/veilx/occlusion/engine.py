"""Occlusion engine: deterministic generation + full provenance for every sample.

``apply_occlusion`` returns the challenged image, its ground-truth mask and an
:class:`OcclusionSpec` recording: family, severity level, severity (fraction of the face region
covered/destroyed), visibility ratio (1 - severity over the face region), seed, sampled
parameters, landmark source, physical flag and a recipe hash over (source pixels, family, level,
seed, parameters, generator version). The same inputs always produce byte-identical outputs.

``apply_stress`` composes several *physical* families for stress tests; stress samples are
recorded separately and are never part of the leave-one-family-out protocol.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from veilx.data.images import pixel_sha256
from veilx.data.manifest import OcclusionSpec
from veilx.occlusion.families import GENERATORS, PHYSICAL, U8, Level
from veilx.occlusion.geometry import BoolMask, Landmarks, face_region

GENERATOR_VERSION = "occl-1.0.0"


@dataclass(frozen=True)
class OcclusionResult:
    image: U8
    mask: BoolMask
    spec: OcclusionSpec


def mask_sha256(mask: BoolMask) -> str:
    return hashlib.sha256(
        np.packbits(mask.astype(np.uint8)).tobytes() + str(mask.shape).encode()
    ).hexdigest()


def _recipe(src_hash: str, family: str, level: str, seed: int, params: dict[str, float]) -> str:
    payload = json.dumps(
        {
            "src": src_hash,
            "family": family,
            "level": level,
            "seed": seed,
            "params": {k: round(v, 6) for k, v in sorted(params.items())},
            "version": GENERATOR_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def apply_occlusion(
    image: U8,
    family: str,
    level: Level,
    seed: int,
    *,
    landmarks: Landmarks | None = None,
    source_pixel_sha256: str | None = None,
) -> OcclusionResult:
    """``source_pixel_sha256`` lets callers supply the already-known source hash for the recipe;
    it must be the hash of ``image`` (callers pass a stand-in image only for families whose
    mask is pixel-independent, see tests)."""
    if family not in GENERATORS:
        raise ValueError(f"unknown family {family!r}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB uint8 image")
    h, w = image.shape[:2]
    lm = landmarks or Landmarks.template(h, w)
    rng = np.random.default_rng(seed)
    out, mask, params = GENERATORS[family](image, lm, level, rng)
    face = face_region(lm, h, w)
    severity = float(mask[face].mean()) if face.any() else 0.0
    geometry = {**params, "landmarks_template": float(lm.source == "arcface_template")}
    spec = OcclusionSpec(
        family=family,
        geometry=geometry,
        seed=seed,
        generator_version=GENERATOR_VERSION,
        severity_level=level,
        severity=round(severity, 6),
        visibility_ratio=round(1.0 - severity, 6),
        physical=PHYSICAL[family],
        recipe_sha256=_recipe(
            source_pixel_sha256 or pixel_sha256(image), family, level, seed, params
        ),
    )
    return OcclusionResult(out, mask, spec)


@dataclass(frozen=True)
class StressResult:
    image: U8
    mask: BoolMask
    steps: tuple[OcclusionSpec, ...]
    combined_visibility_ratio: float


def apply_stress(
    image: U8, steps: Sequence[tuple[str, Level, int]], *, landmarks: Landmarks | None = None
) -> StressResult:
    """Sequentially compose physical occlusions (e.g. mask + sunglasses)."""
    if len(steps) < 2:
        raise ValueError("a stress test combines at least two families")
    if any(not PHYSICAL[f] for f, _, _ in steps):
        raise ValueError("stress compositions use physical occlusion families only")
    h, w = image.shape[:2]
    lm = landmarks or Landmarks.template(h, w)
    cur, union = image, np.zeros((h, w), dtype=bool)
    specs = []
    for family, level, seed in steps:
        res = apply_occlusion(cur, family, level, seed, landmarks=lm)
        cur, union = res.image, union | res.mask
        specs.append(res.spec)
    face = face_region(lm, h, w)
    return StressResult(cur, union, tuple(specs), round(1.0 - float(union[face].mean()), 6))


def visibility_grid(mask: BoolMask, cells: int = 7) -> NDArray[np.float64]:
    """Per-cell visible fraction on a cells x cells grid (matches a 7x7 backbone feature map)."""
    h, w = mask.shape
    ys = np.linspace(0, h, cells + 1).astype(int)
    xs = np.linspace(0, w, cells + 1).astype(int)
    out = np.empty((cells, cells))
    for i in range(cells):
        for j in range(cells):
            block = mask[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            out[i, j] = 1.0 - float(block.mean()) if block.size else 1.0
    return out
