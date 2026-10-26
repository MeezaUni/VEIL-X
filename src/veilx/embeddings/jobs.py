"""Deterministic work items for embedding extraction.

* clean item: key ``<image_id>``;
* occluded item: key ``<image_id>|<family>|<level>|<seed>`` - one per occluded pair side of a
  leave-one-family-out plan; the occluded image is re-rendered from the source pixels and its
  recipe hash must equal the one recorded in the plan (so the embedded image is provably the
  planned one);
* stress item: key ``<image_id>|stress|<f1>:<level>:<seed>+<f2>:<level>:<seed>`` (exploratory).

Item order is the plan order (pairs sorted by id, side a then b), so extraction is
reproducible and resumable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.data.manifest import PairRecord
from veilx.errors import ManifestError
from veilx.occlusion.engine import apply_occlusion, apply_stress

U8 = NDArray[np.uint8]


@dataclass(frozen=True)
class Item:
    key: str
    image_id: str
    family: str | None = None
    level: str | None = None
    seed: int | None = None
    recipe_sha256: str | None = None
    stress_steps: tuple[tuple[str, str, int], ...] | None = None


def occluded_key(image_id: str, family: str, level: str, seed: int) -> str:
    return f"{image_id}|{family}|{level}|{seed}"


def stress_key(image_id: str, steps: Sequence[Sequence[Any]]) -> str:
    return f"{image_id}|stress|" + "+".join(f"{f}:{lv}:{s}" for f, lv, s in steps)


def clean_items(image_ids: Iterable[str]) -> list[Item]:
    return [Item(key=i, image_id=i) for i in sorted(set(image_ids))]


def side_image(pair: PairRecord, side: str) -> str:
    return pair.image_a if side == "a" else pair.image_b


def occluded_items(
    rows: Iterable[Mapping[str, Any]], pairs_by_id: Mapping[str, PairRecord]
) -> list[Item]:
    """Items for every occluded side of a LOFO plan file (rows as written by the prepare
    script: ``pair_id``, ``a``, ``b`` with OcclusionSpec dicts or ``None``)."""
    out: list[Item] = []
    seen: set[str] = set()
    for row in rows:
        pair = pairs_by_id[row["pair_id"]]
        for side in ("a", "b"):
            spec = row[side]
            if spec is None:
                continue
            img = side_image(pair, side)
            key = occluded_key(img, spec["family"], spec["severity_level"], int(spec["seed"]))
            if key in seen:
                raise ManifestError(f"duplicate occluded item {key}")
            seen.add(key)
            out.append(
                Item(
                    key=key,
                    image_id=img,
                    family=spec["family"],
                    level=spec["severity_level"],
                    seed=int(spec["seed"]),
                    recipe_sha256=spec["recipe_sha256"],
                )
            )
    return out


def stress_items(
    rows: Iterable[Mapping[str, Any]], pairs_by_id: Mapping[str, PairRecord]
) -> list[Item]:
    out: list[Item] = []
    for row in rows:
        pair = pairs_by_id[row["pair_id"]]
        for side, steps_name in (("a", "steps_a"), ("b", "steps_b")):
            steps = tuple((str(f), str(lv), int(s)) for f, lv, s in row[steps_name])
            img = side_image(pair, side)
            out.append(Item(key=stress_key(img, steps), image_id=img, stress_steps=steps))
    return out


def render(item: Item, source_rgb: U8, source_pixel_sha256: str) -> U8:
    """Return the exact image to embed for ``item`` (clean, occluded or stress)."""
    if item.stress_steps is not None:
        return apply_stress(source_rgb, item.stress_steps).image  # type: ignore[arg-type]
    if item.family is None:
        return source_rgb
    res = apply_occlusion(
        source_rgb,
        item.family,
        item.level,  # type: ignore[arg-type]
        int(item.seed),  # type: ignore[arg-type]
        source_pixel_sha256=source_pixel_sha256,
    )
    if item.recipe_sha256 is not None and res.spec.recipe_sha256 != item.recipe_sha256:
        raise ManifestError(f"{item.key}: re-rendered recipe differs from the plan")
    return res.image
