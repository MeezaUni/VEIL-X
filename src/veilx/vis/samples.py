"""Deterministic training/evaluation samples for the visibility estimator (numpy only).

A sample is (occluded-or-clean image, soft 28x28 visibility target, metadata). Families are
restricted to ``allowed`` (the fold's SEEN families for training/validation), so the held-out
family can never enter training (checked).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from veilx.occlusion.engine import apply_occlusion
from veilx.occlusion.families import LEVELS, Level
from veilx.seeding import derive_seed
from veilx.vis.regions import GRID, visibility_target

CLEAN_FRACTION = 0.25


def draw_condition(
    rng: np.random.Generator, allowed: Sequence[str]
) -> tuple[str | None, str | None]:
    if rng.random() < CLEAN_FRACTION:
        return None, None
    return str(allowed[int(rng.integers(len(allowed)))]), str(LEVELS[int(rng.integers(3))])


def make_sample(
    image: NDArray[np.uint8],
    family: str | None,
    level: str | None,
    seed: int,
    source_pixel_sha256: str = "",
    *,
    lighting_as_corruption: bool = False,
) -> tuple[NDArray[np.uint8], NDArray[np.float64], dict[str, Any]]:
    if family is None:
        return (
            image,
            np.ones((GRID, GRID)),
            {"family": None, "level": None, "visibility_ratio": 1.0},
        )
    res = apply_occlusion(
        image, family, cast(Level, level), seed, source_pixel_sha256=source_pixel_sha256 or None
    )
    mask = res.mask
    if lighting_as_corruption and family == "F5_extreme_lighting":
        mask = lighting_information_free(image, res.image)
    return (
        res.image,
        visibility_target(mask),
        {"family": family, "level": level, "visibility_ratio": res.spec.visibility_ratio},
    )


HARD_NEGATIVE = "HN_natural_texture_blobs"  # Phase 4 generic hard negatives (not a LOFO family)


def _blob_mask(rng: np.random.Generator, size: int = 112) -> NDArray[np.bool_]:
    """1-4 irregular blobs: a seeded random walk of overlapping discs (no polygons/ellipses with
    uniform fills, so it is a different generator from every LOFO family)."""
    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=bool)
    for _ in range(int(rng.integers(1, 5))):
        cy, cx = rng.uniform(25, 95), rng.uniform(20, 92)
        r = rng.uniform(5, 11)
        for _step in range(int(rng.integers(4, 14))):
            mask |= (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 <= r * r
            cy = float(np.clip(cy + rng.normal(0, r * 0.8), 10, 102))
            cx = float(np.clip(cx + rng.normal(0, r * 0.8), 10, 102))
            r = float(np.clip(r + rng.normal(0, 1.5), 4, 14))
    return mask


def make_hard_negative(
    image: NDArray[np.uint8], donor: NDArray[np.uint8], seed: int
) -> tuple[NDArray[np.uint8], NDArray[np.float64], dict[str, Any]]:
    """Generic hard negative: blobs filled with NATURAL texture taken from another training image
    at a random offset (background, clothing, hair or face of another identity). Target from the
    blob mask (blob cells are not visible face)."""
    rng = np.random.default_rng(seed)
    mask = _blob_mask(rng, image.shape[0])
    dy, dx = (int(v) for v in rng.integers(-40, 41, 2))
    tex = np.roll(np.roll(donor, dy, axis=0), dx, axis=1)
    out = np.where(mask[..., None], tex, image).astype(np.uint8)
    vis = visibility_target(mask)
    return out, vis, {"family": HARD_NEGATIVE, "level": None, "visibility_ratio": float(vis.mean())}


def lighting_information_free(
    source: NDArray[np.uint8], lit: NDArray[np.uint8]
) -> NDArray[np.bool_]:
    """Lighting-as-corruption target (Phase 4): only pixels whose information is destroyed -
    crushed (max channel <= 12) or newly clipped (min channel >= 250 where the source was not).
    Darkened-but-informative pixels count as visible."""
    crushed = lit.max(axis=2) <= 12
    clipped = (lit.min(axis=2) >= 250) & (source.min(axis=2) < 250)
    return np.asarray(crushed | clipped)


def sample_seed(base: int, tag: str, index: int) -> int:
    return derive_seed(base, "ve", tag, index)


def check_isolation(allowed: Sequence[str], heldout: str | None) -> None:
    if heldout is not None and heldout in allowed:
        raise ValueError(f"held-out family {heldout} must not be used for training/validation")
