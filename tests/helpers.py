"""Test helpers: synthetic, non-biometric fixtures (random-noise PNGs, toy records)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from veilx.config.schema import SplitName
from veilx.data.manifest import ImageRecord, OcclusionSpec
from veilx.data.png import encode_png


def noise_png(seed: int, size: int = 8) -> bytes:
    rng = np.random.default_rng(seed)
    return encode_png(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def write_identity(
    root: Path, identity: str, n_images: int, *, seed: int = 0, start: int = 0
) -> Path:
    d = root / identity
    d.mkdir(parents=True, exist_ok=True)
    for k in range(start, start + n_images):
        (d / f"{k}.png").write_bytes(noise_png(seed * 1000 + int(identity) * 100 + k))
    return d


def rec(
    image_id: str,
    identity: str,
    split: SplitName = SplitName.CAL,
    *,
    source: str | None = None,
    family: str | None = None,
    content: str | None = None,
) -> ImageRecord:
    occ = OcclusionSpec(family=family, seed=0, generator_version="t") if family else None
    return ImageRecord(
        image_id=image_id,
        dataset_id="toy",
        identity_id=identity,
        relpath=f"{identity}/{image_id}.png",
        sha256=hashlib.sha256((content or image_id).encode()).hexdigest(),
        split=split,
        source_image_id=source,
        occlusion=occ,
    )


def toy_images(
    n_identities: int, per_identity: int, split: SplitName = SplitName.CAL
) -> list[ImageRecord]:
    return [
        rec(f"i{p}_{k}", f"p{p}", split) for p in range(n_identities) for k in range(per_identity)
    ]
