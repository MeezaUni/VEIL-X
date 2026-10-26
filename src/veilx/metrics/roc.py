"""Score-based verification metrics.

Conventions (used everywhere in VEIL-X):

* ``labels``: 1 = genuine (same identity), 0 = impostor.
* ``scores``: higher means "more likely genuine"; a pair is **accepted as a match iff
  score >= threshold**.
* FAR = FMR = (# impostor scores >= t) / (# impostors); TAR = (# genuine scores >= t) / (# genuine);
  FNMR = 1 - TAR. (Standard biometric definitions; see docs/metrics.md.)

ROC-AUC is the Mann-Whitney probability P(s_genuine > s_impostor) + 0.5 P(tie), which equals the
trapezoidal area under the empirical ROC (both are implemented; tests check they agree).
Operating points refuse to run when the impostor sample cannot support the requested FAR
(protocol rule: at least ``min_impostor_factor / FAR`` impostor pairs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from veilx.errors import InsufficientSamplesError

FloatArr = NDArray[np.float64]
DEFAULT_MIN_IMPOSTOR_FACTOR = 10.0


def _as_arrays(scores: ArrayLike, labels: ArrayLike) -> tuple[FloatArr, NDArray[np.int64]]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1 or s.shape != y.shape:
        raise ValueError("scores and labels must be 1-D arrays of equal length")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite (encode alignment failures explicitly)")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (impostor) or 1 (genuine)")
    y = y.astype(np.int64)
    if y.sum() == 0 or y.sum() == y.size:
        raise InsufficientSamplesError("both genuine and impostor pairs are required")
    return s, y


def roc_curve(scores: ArrayLike, labels: ArrayLike) -> tuple[FloatArr, FloatArr, FloatArr]:
    """Empirical ROC at every distinct threshold, from (0, 0) at t=+inf to (1, 1).

    Returns ``(far, tar, thresholds)``; point k uses threshold ``thresholds[k]`` (accept >= t).
    """
    s, y = _as_arrays(scores, labels)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    distinct = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    tp = np.cumsum(y_sorted)[distinct]
    fp = (distinct + 1) - tp
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    far = np.r_[0.0, fp / n_neg]
    tar = np.r_[0.0, tp / n_pos]
    thr = np.r_[np.inf, s_sorted[distinct]]
    return far, tar, thr


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Mann-Whitney AUC with mid-rank tie handling."""
    s, y = _as_arrays(scores, labels)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    s_sorted = s[order]
    i = 0
    while i < s.size:  # average ranks over ties
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc_trapezoid(scores: ArrayLike, labels: ArrayLike) -> float:
    far, tar, _ = roc_curve(scores, labels)
    return float(np.sum(np.diff(far) * (tar[1:] + tar[:-1]) / 2.0))


@dataclass(frozen=True)
class EERResult:
    eer: float
    threshold: float  # approximate threshold at the crossing (score units)


def equal_error_rate(scores: ArrayLike, labels: ArrayLike) -> EERResult:
    """EER: the error where FAR equals FNMR, linearly interpolated between ROC points."""
    far, tar, thr = roc_curve(scores, labels)
    fnmr = 1.0 - tar
    diff = far - fnmr  # starts at -1 (t=+inf), ends at +1
    k = int(np.flatnonzero(diff >= 0)[0])
    if diff[k] == 0 or k == 0:
        t = thr[k] if np.isfinite(thr[k]) else thr[k + 1]
        return EERResult(float(far[k]), float(t))
    w = -diff[k - 1] / (diff[k] - diff[k - 1])
    eer = far[k - 1] + w * (far[k] - far[k - 1])
    t_prev = thr[k - 1] if np.isfinite(thr[k - 1]) else thr[k]
    return EERResult(float(eer), float(t_prev + w * (thr[k] - t_prev)))


def required_impostors(far: float, factor: float = DEFAULT_MIN_IMPOSTOR_FACTOR) -> int:
    if not 0 < far < 1:
        raise ValueError("far must be in (0, 1)")
    return math.ceil(factor / far)


def threshold_at_far(
    impostor_scores: ArrayLike,
    far: float,
    *,
    min_impostor_factor: float = DEFAULT_MIN_IMPOSTOR_FACTOR,
) -> float:
    """Smallest threshold whose empirical FAR on these impostors is <= ``far`` (accept >= t)."""
    imp = np.sort(np.asarray(impostor_scores, dtype=np.float64))[::-1]
    need = required_impostors(far, min_impostor_factor)
    if imp.size < need:
        raise InsufficientSamplesError(
            f"FAR={far} needs >= {need} impostor scores (factor {min_impostor_factor:g}); "
            f"got {imp.size}"
        )
    k_allowed = math.floor(far * imp.size + 1e-12)
    if k_allowed >= imp.size:
        return -math.inf
    # Threshold must be strictly above the (k_allowed+1)-th largest impostor score.
    return float(np.nextafter(imp[k_allowed], math.inf))


@dataclass(frozen=True)
class ThresholdEvaluation:
    threshold: float
    n_genuine: int
    n_impostor: int
    true_accepts: int
    false_accepts: int
    false_rejects: int
    true_rejects: int

    @property
    def tar(self) -> float:
        return self.true_accepts / self.n_genuine

    @property
    def fnmr(self) -> float:
        return self.false_rejects / self.n_genuine

    @property
    def far(self) -> float:
        return self.false_accepts / self.n_impostor


def evaluate_at_threshold(
    scores: ArrayLike, labels: ArrayLike, threshold: float
) -> ThresholdEvaluation:
    s, y = _as_arrays(scores, labels)
    acc = s >= threshold
    return ThresholdEvaluation(
        threshold=float(threshold),
        n_genuine=int(y.sum()),
        n_impostor=int((1 - y).sum()),
        true_accepts=int((acc & (y == 1)).sum()),
        false_accepts=int((acc & (y == 0)).sum()),
        false_rejects=int((~acc & (y == 1)).sum()),
        true_rejects=int((~acc & (y == 0)).sum()),
    )


@dataclass(frozen=True)
class OperatingPoint:
    mode: Literal["benchmark_threshold_on_test", "deployment_threshold_from_cal"]
    target_far: float
    threshold: float
    tar: float
    realised_far: float
    n_genuine: int
    n_impostor: int
    threshold_source_n_impostor: int


def tar_at_far(
    scores: ArrayLike,
    labels: ArrayLike,
    far: float,
    *,
    min_impostor_factor: float = DEFAULT_MIN_IMPOSTOR_FACTOR,
) -> OperatingPoint:
    """Benchmark mode: threshold chosen on the evaluated set itself (literature convention)."""
    s, y = _as_arrays(scores, labels)
    t = threshold_at_far(s[y == 0], far, min_impostor_factor=min_impostor_factor)
    ev = evaluate_at_threshold(s, y, t)
    return OperatingPoint(
        "benchmark_threshold_on_test",
        far,
        t,
        ev.tar,
        ev.far,
        ev.n_genuine,
        ev.n_impostor,
        ev.n_impostor,
    )


def deployment_operating_point(
    cal_scores: ArrayLike,
    cal_labels: ArrayLike,
    test_scores: ArrayLike,
    test_labels: ArrayLike,
    far: float,
    *,
    min_impostor_factor: float = DEFAULT_MIN_IMPOSTOR_FACTOR,
) -> OperatingPoint:
    """Deployment mode: threshold fixed on calibration data, realised FAR/TAR measured on test."""
    cs, cy = _as_arrays(cal_scores, cal_labels)
    t = threshold_at_far(cs[cy == 0], far, min_impostor_factor=min_impostor_factor)
    ts, ty = _as_arrays(test_scores, test_labels)
    need = required_impostors(far, min_impostor_factor)
    if int((ty == 0).sum()) < need:
        raise InsufficientSamplesError(
            f"realised FAR at {far} needs >= {need} test impostors; got {int((ty == 0).sum())}"
        )
    ev = evaluate_at_threshold(ts, ty, t)
    return OperatingPoint(
        "deployment_threshold_from_cal",
        far,
        t,
        ev.tar,
        ev.far,
        ev.n_genuine,
        ev.n_impostor,
        int((cy == 0).sum()),
    )
