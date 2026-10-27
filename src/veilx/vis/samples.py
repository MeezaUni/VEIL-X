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
    return (
        res.image,
        visibility_target(res.mask),
        {"family": family, "level": level, "visibility_ratio": res.spec.visibility_ratio},
    )


def sample_seed(base: int, tag: str, index: int) -> int:
    return derive_seed(base, "ve", tag, index)


def check_isolation(allowed: Sequence[str], heldout: str | None) -> None:
    if heldout is not None and heldout in allowed:
        raise ValueError(f"held-out family {heldout} must not be used for training/validation")
