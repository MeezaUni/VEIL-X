"""Visibility-estimation metrics (explicit definitions; all on face cells of the 28x28 grid).

* A cell is **occluded** in the target when its visible fraction is < 0.5, and **predicted
  occluded** when P(visible) < 0.5.
* ``iou`` / ``dice``: per image over occluded cells, |P and T| / |P or T| and
  2|P and T| / (|P| + |T|); images whose target has no occluded face cell are excluded
  (reported count); averaged over images.
* ``brier``: mean squared error between P(visible) and the binary cell target.
* ``ece``: 10 equal-width bins of P(visible); sum over bins of (bin share) x |mean prob - mean
  binary target|.
* ``global_mae``: |predicted face-visible fraction - reference visibility| per image, mean.
* ``region_mae``: per region, |predicted - target region visibility|, mean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.vis.regions import face_cells, global_visibility, region_visibility

F64 = NDArray[np.float64]


def segmentation_scores(prob: F64, target: F64) -> dict[str, Any]:
    """prob, target: (N, G, G) P(visible) and soft visible-fraction targets."""
    f = face_cells(prob.shape[-1])
    p_occ = prob[:, f] < 0.5
    t_occ = target[:, f] < 0.5
    inter = (p_occ & t_occ).sum(1)
    union = (p_occ | t_occ).sum(1)
    tot = p_occ.sum(1) + t_occ.sum(1)
    has = t_occ.any(1)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    dice = np.where(tot > 0, 2 * inter / np.maximum(tot, 1), 1.0)
    return {
        "iou": float(iou[has].mean()) if has.any() else None,
        "dice": float(dice[has].mean()) if has.any() else None,
        "n_images_with_occlusion": int(has.sum()),
        "clean_false_occlusion_rate": float(p_occ[~has].mean()) if (~has).any() else None,
    }


def calibration_scores(prob: F64, target: F64, bins: int = 10) -> dict[str, float]:
    f = face_cells(prob.shape[-1])
    p = prob[:, f].ravel()
    y = (target[:, f] >= 0.5).ravel().astype(np.float64)
    brier = float(np.mean((p - y) ** 2))
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"brier": brier, "ece": float(ece)}


def visibility_errors(
    prob: F64, target: F64, reference_global: F64 | None = None
) -> dict[str, Any]:
    """Global and regional absolute errors, plus the signed global bias (predicted - reference;
    > 0 means visibility is OVERestimated, the direction that loosens stratified rules).
    ``reference_global`` defaults to the target map's face-cell mean (pass the engine's
    pixel-level ``visibility_ratio`` to include discretisation error)."""
    g_pred = global_visibility(prob)
    g_ref = global_visibility(target) if reference_global is None else np.asarray(reference_global)
    rp, rt = region_visibility(prob), region_visibility(target)
    return {
        "global_mae": float(np.mean(np.abs(g_pred - g_ref))),
        "global_bias": float(np.mean(g_pred - g_ref)),
        "region_mae": {k: float(np.mean(np.abs(rp[k] - rt[k]))) for k in rp},
    }


def all_scores(prob: F64, target: F64, reference_global: F64 | None = None) -> dict[str, Any]:
    return {
        "n": int(prob.shape[0]),
        **segmentation_scores(prob, target),
        **calibration_scores(prob, target),
        **visibility_errors(prob, target, reference_global),
    }
