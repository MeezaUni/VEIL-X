"""Pairwise co-visibility from per-image visibility maps (registered definitions, Phase 3 §4).

Inputs are P(visible) maps on the estimator grid (predicted, heuristic or oracle). Every
definition is a function of the pair alone, so strata built from it satisfy acceptance
locality (A7) when edges are fixed on validation data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from veilx.vis.regions import face_cells, global_visibility, region_visibility

F64 = NDArray[np.float64]
DEFINITIONS = ("cv_min", "cv_prod", "cv_spatial", "cv_region")


def pair_features(map_a: F64, map_b: F64) -> dict[str, F64]:
    """map_a, map_b: (N, G, G). Returns per-pair co-visibility features."""
    va, vb = global_visibility(map_a), global_visibility(map_b)
    f = face_cells(map_a.shape[-1])
    spatial = np.minimum(map_a[:, f], map_b[:, f]).mean(axis=1)
    ra, rb = region_visibility(map_a), region_visibility(map_b)
    region = np.mean([np.minimum(ra[k], rb[k]) for k in ra], axis=0)
    return {
        "vis_a": va,
        "vis_b": vb,
        "vmin": np.minimum(va, vb),
        "vmax": np.maximum(va, vb),
        "cv_min": np.minimum(va, vb),
        "cv_prod": va * vb,
        "cv_spatial": np.asarray(spatial, dtype=np.float64),
        "cv_region": np.asarray(region, dtype=np.float64),
        "asym": np.abs(va - vb),
    }
