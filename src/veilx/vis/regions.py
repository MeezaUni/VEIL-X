"""Visibility targets, the face-cell mask and template facial regions on the estimator grid.

All geometry is defined on the ArcFace 112x112 template (aligned crops), so regions are fixed
masks shared by every image.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from veilx.occlusion.geometry import Landmarks, ellipse_mask, face_region

GRID = 28
SIZE = 112
F64 = NDArray[np.float64]
REGION_NAMES = ("left_eye", "right_eye", "nose", "mouth", "lower_face")


def pool(mask112: NDArray[np.bool_] | F64, grid: int = GRID) -> F64:
    """Average-pool a 112x112 map to ``grid`` x ``grid``."""
    k = SIZE // grid
    m = np.asarray(mask112, dtype=np.float64)
    return m.reshape(grid, k, grid, k).mean(axis=(1, 3))


def visibility_target(occluded112: NDArray[np.bool_], grid: int = GRID) -> F64:
    """Soft per-cell visible fraction (1 = fully visible) from an occlusion mask."""
    return 1.0 - pool(occluded112, grid)


@lru_cache(maxsize=4)
def face_cells(grid: int = GRID) -> NDArray[np.bool_]:
    """Cells whose majority lies inside the template face ellipse (brows to chin)."""
    lm = Landmarks.template(SIZE, SIZE)
    return pool(face_region(lm, SIZE, SIZE), grid) >= 0.5


@lru_cache(maxsize=4)
def region_cells(grid: int = GRID) -> dict[str, NDArray[np.bool_]]:
    """Five fixed template regions (cells with >= 50 % of their area inside the region)."""
    lm = Landmarks.template(SIZE, SIZE)
    d = lm.eye_distance
    pts = lm.points
    mouth = (pts[3] + pts[4]) / 2
    face = face_region(lm, SIZE, SIZE)
    yy = np.arange(SIZE)[:, None] * np.ones((1, SIZE))
    regs112 = {
        "left_eye": ellipse_mask(SIZE, SIZE, pts[0, 0], pts[0, 1], 0.35 * d, 0.25 * d),
        "right_eye": ellipse_mask(SIZE, SIZE, pts[1, 0], pts[1, 1], 0.35 * d, 0.25 * d),
        "nose": ellipse_mask(SIZE, SIZE, pts[2, 0], pts[2, 1] - 0.15 * d, 0.25 * d, 0.40 * d),
        "mouth": ellipse_mask(SIZE, SIZE, mouth[0], mouth[1], 0.45 * d, 0.22 * d),
        "lower_face": face & (yy >= mouth[1] + 0.22 * d),
    }
    return {k: pool(v, grid) >= 0.5 for k, v in regs112.items()}


def global_visibility(vis_map: F64, grid: int = GRID) -> F64:
    """Mean visibility over face cells; works on (H, W) or (N, H, W)."""
    f = face_cells(grid)
    return np.asarray(np.asarray(vis_map)[..., f].mean(axis=-1), dtype=np.float64)


def region_visibility(vis_map: F64, grid: int = GRID) -> dict[str, F64]:
    v = np.asarray(vis_map)
    return {
        k: np.asarray(v[..., m].mean(axis=-1), dtype=np.float64)
        for k, m in region_cells(grid).items()
    }
