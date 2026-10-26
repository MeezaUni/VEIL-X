"""Facial landmark geometry for occlusion placement.

When no landmark detector is available (Phase 1b: none is authorised), aligned face crops use
the standard 5-point ArcFace template for 112x112 crops, scaled to the image size. Whether a
dataset is actually aligned to this template must be checked empirically (see
``docs/data_integrity_report.md``); the landmark source is recorded with every occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

# InsightFace / ArcFace reference landmarks for 112x112 aligned crops (x, y):
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ARCFACE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class Landmarks:
    points: NDArray[np.float64]  # (5, 2) x, y
    source: str  # "arcface_template" or a detector id

    @classmethod
    def template(cls, height: int, width: int) -> Landmarks:
        return cls(ARCFACE_112 * np.array([width / 112.0, height / 112.0]), "arcface_template")

    @property
    def left_eye(self) -> NDArray[np.float64]:
        return np.asarray(self.points[0], dtype=np.float64)

    @property
    def right_eye(self) -> NDArray[np.float64]:
        return np.asarray(self.points[1], dtype=np.float64)

    @property
    def nose(self) -> NDArray[np.float64]:
        return np.asarray(self.points[2], dtype=np.float64)

    @property
    def mouth_left(self) -> NDArray[np.float64]:
        return np.asarray(self.points[3], dtype=np.float64)

    @property
    def mouth_right(self) -> NDArray[np.float64]:
        return np.asarray(self.points[4], dtype=np.float64)

    @property
    def eye_center(self) -> NDArray[np.float64]:
        return (self.left_eye + self.right_eye) / 2

    @property
    def mouth_center(self) -> NDArray[np.float64]:
        return (self.mouth_left + self.mouth_right) / 2

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.right_eye - self.left_eye))


@lru_cache(maxsize=16)
def pixel_grid(h: int, w: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cached, read-only (yy, xx) coordinate grids (performance only; values unchanged)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy.flags.writeable = False
    xx.flags.writeable = False
    return yy, xx


def ellipse_mask(
    h: int, w: int, cx: float, cy: float, rx: float, ry: float, angle: float = 0.0
) -> BoolMask:
    """Pixels whose centres lie inside the (rotated) ellipse.

    Evaluated only inside the bounding square of radius max(rx, ry) (+1 px margin): every
    pixel centre outside it is farther than both semi-axes from the centre, so the result is
    identical to a full-image evaluation.
    """
    out = np.zeros((h, w), dtype=bool)
    r = max(rx, ry)
    y0, y1 = max(0, int(np.floor(cy - r)) - 1), min(h, int(np.ceil(cy + r)) + 1)
    x0, x1 = max(0, int(np.floor(cx - r)) - 1), min(w, int(np.ceil(cx + r)) + 1)
    if y0 >= y1 or x0 >= x1:
        return out
    yy, xx = pixel_grid(h, w)
    c, s = np.cos(angle), np.sin(angle)
    dx, dy = xx[y0:y1, x0:x1] + 0.5 - cx, yy[y0:y1, x0:x1] + 0.5 - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    out[y0:y1, x0:x1] = (u / max(rx, 1e-6)) ** 2 + (v / max(ry, 1e-6)) ** 2 <= 1.0
    return out


def face_region(lm: Landmarks, h: int, w: int) -> BoolMask:
    """Approximate facial area (brows to chin) used to normalise severity and visibility."""
    d = lm.eye_distance
    cx = float(lm.points[:, 0].mean())
    cy = 0.45 * float(lm.eye_center[1]) + 0.55 * float(lm.mouth_center[1])
    return ellipse_mask(h, w, cx, cy, 1.15 * d, 1.55 * d)


def polygon_mask(h: int, w: int, pts: NDArray[np.float64]) -> BoolMask:
    """Even-odd rule point-in-polygon test at pixel centres (vectorised)."""
    yy, xx = pixel_grid(h, w)
    x, y = xx + 0.5, yy + 0.5
    inside = np.zeros((h, w), dtype=bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        cond = (y1 > y) != (y2 > y)
        xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xint)
    return inside
