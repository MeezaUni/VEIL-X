"""Image-based visibility heuristic (non-learned comparator, Phase 3 §3).

Per-cell appearance anomaly: each 4x4-pixel cell's mean RGB is compared with the per-cell mean
and standard deviation over clean ``p3_train`` crops; the anomaly z is the Euclidean norm of
the standardised difference, and P(visible) = sigmoid(a - b * z) with the two parameters fitted
on ``p3_val`` (all other statistics come from training images).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from veilx.fusion.logreg import LogisticModel
from veilx.vis.regions import GRID, SIZE

F64 = NDArray[np.float64]


def cell_means(images: NDArray[np.uint8], grid: int = GRID) -> F64:
    """(N, 112, 112, 3) -> (N, grid, grid, 3) mean RGB per cell, in [0, 1]."""
    k = SIZE // grid
    x = images.astype(np.float64) / 255.0
    return x.reshape(len(x), grid, k, grid, k, 3).mean(axis=(2, 4))


@dataclass
class AnomalyHeuristic:
    mu: F64  # (G, G, 3)
    sd: F64  # (G, G, 3)
    link: LogisticModel | None = None

    @classmethod
    def from_clean(cls, images: NDArray[np.uint8]) -> AnomalyHeuristic:
        c = cell_means(images)
        return cls(c.mean(0), np.maximum(c.std(0), 1e-3))

    def anomaly(self, images: NDArray[np.uint8]) -> F64:
        z = (cell_means(images) - self.mu) / self.sd
        return np.asarray(np.sqrt((z**2).sum(-1)), dtype=np.float64)

    def fit_link(self, images: NDArray[np.uint8], target: F64) -> None:
        a = self.anomaly(images).reshape(-1, 1)
        y = (target.reshape(-1) >= 0.5).astype(np.int64)
        self.link = LogisticModel(("anomaly",), 1e6).fit(a, y)

    def predict(self, images: NDArray[np.uint8]) -> F64:
        if self.link is None:
            raise RuntimeError("fit_link on validation data first")
        a = self.anomaly(images)
        return self.link.predict_proba(a.reshape(-1, 1)).reshape(a.shape)
