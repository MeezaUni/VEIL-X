"""Phase 4 conservative visibility and unknown-occlusion detection (phase4_protocol.md 4.2-4.4).

Every quantity here is fitted on DEVELOPMENT data only: temperature and lower-bound shift on
p3_val renders of SEEN families; the novelty Gaussian on p3_train renders; the novelty threshold
on p3_val in-distribution pairs. Nothing is tuned on held-out or test families.

* :func:`fit_temperature` - one scalar T minimising face-cell log-loss of sigmoid(logit / T).
* :func:`lower_bound_shift` - smallest c >= 0 such that the predicted global visible fraction
  overestimates the truth by more than ``margin`` on at most ``max_rate`` of occluded renders.
* :func:`conservative_map` - ``clip(sigmoid(logit / T) - c, 0, 1)``.
* :class:`MahalanobisNovelty` - Gaussian on bottleneck features with fixed shrinkage; pair score =
  max over the two sides; threshold = (1 - fpr) quantile of in-distribution PAIR scores.
* :func:`tta_disagreement` - mean |p - p_flip| over face cells (secondary, reported only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.vis.regions import face_cells

F64 = NDArray[np.float64]
T_GRID = np.round(np.arange(0.5, 6.01, 0.05), 4)
C_GRID = np.round(np.arange(0.0, 0.6001, 0.005), 4)
SHRINKAGE = 0.1  # registered with the protocol (fixed, not tuned)


def _sigmoid(x: F64) -> F64:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def fit_temperature(logits: NDArray[np.floating], target_visible: NDArray[np.bool_]) -> float:
    """Grid search over T on face cells (logits/targets shaped (N, G, G))."""
    fc = face_cells(logits.shape[-1])
    z = np.asarray(logits, np.float64)[:, fc].ravel()
    y = np.asarray(target_visible)[:, fc].ravel().astype(np.float64)
    best_t, best = 1.0, np.inf
    for t in T_GRID:
        p = np.clip(_sigmoid(z / t), 1e-7, 1 - 1e-7)
        ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        if ll < best:
            best_t, best = float(t), ll
    return best_t


def global_visible(prob_map: NDArray[np.floating]) -> F64:
    fc = face_cells(prob_map.shape[-1])
    return np.asarray(np.asarray(prob_map, np.float64)[:, fc].mean(axis=1), dtype=np.float64)


def lower_bound_shift(
    pred_global: NDArray[np.floating],
    true_global: NDArray[np.floating],
    *,
    margin: float = 0.10,
    max_rate: float = 0.05,
) -> float:
    """Smallest grid c with mean((pred - c) - true > margin) <= max_rate."""
    over = np.asarray(pred_global, np.float64) - np.asarray(true_global, np.float64)
    for c in C_GRID:
        if float(((over - c) > margin).mean()) <= max_rate:
            return float(c)
    return float(C_GRID[-1])


def conservative_map(logits: NDArray[np.floating], temperature: float, shift: float) -> F64:
    p = _sigmoid(np.asarray(logits, np.float64) / temperature)
    return np.asarray(np.clip(p - shift, 0.0, 1.0), dtype=np.float64)


def tta_disagreement(logits: NDArray[np.floating], logits_flip: NDArray[np.floating]) -> F64:
    fc = face_cells(np.asarray(logits).shape[-1])
    d = np.abs(
        _sigmoid(np.asarray(logits, np.float64)) - _sigmoid(np.asarray(logits_flip, np.float64))
    )
    return np.asarray(d[:, fc].mean(axis=1), dtype=np.float64)


@dataclass
class MahalanobisNovelty:
    mean: F64
    precision: F64
    threshold: float | None = None

    @classmethod
    def fit(
        cls, features: NDArray[np.floating], shrinkage: float = SHRINKAGE
    ) -> MahalanobisNovelty:
        x = np.asarray(features, np.float64)
        if x.ndim != 2 or len(x) <= x.shape[1]:
            raise ValueError("need more samples than feature dimensions")
        mu = x.mean(axis=0)
        cov = np.cov(x - mu, rowvar=False)
        d = cov.shape[0]
        cov_s = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * np.eye(d)
        return cls(mu, np.linalg.inv(cov_s))

    def score(self, features: NDArray[np.floating]) -> F64:
        z = np.asarray(features, np.float64) - self.mean
        return np.asarray(np.sqrt(np.einsum("ij,jk,ik->i", z, self.precision, z)), dtype=np.float64)

    def set_threshold(self, id_pair_scores: NDArray[np.floating], fpr: float) -> float:
        """Threshold = (1 - fpr) quantile of in-distribution PAIR scores (pair FPR = fpr)."""
        self.threshold = float(np.quantile(np.asarray(id_pair_scores, np.float64), 1.0 - fpr))
        return self.threshold

    def is_novel(self, pair_scores: NDArray[np.floating]) -> NDArray[np.bool_]:
        if self.threshold is None:
            raise RuntimeError("set_threshold on in-distribution validation pairs first")
        return np.asarray(np.asarray(pair_scores) > self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MahalanobisNovelty:
        t = d.get("threshold")
        return cls(
            np.asarray(d["mean"], np.float64),
            np.asarray(d["precision"], np.float64),
            None if t is None else float(t),
        )


def pair_max(a: NDArray[np.floating], b: NDArray[np.floating]) -> F64:
    return np.asarray(
        np.maximum(np.asarray(a, np.float64), np.asarray(b, np.float64)), dtype=np.float64
    )
