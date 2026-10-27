"""Phase 3 numpy components: visibility targets/regions/metrics, co-visibility, logistic fusion,
heuristic, training-sample isolation (synthetic inputs only)."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.fusion.logreg import LogisticModel, log_loss, select_c
from veilx.vis.covis import pair_features
from veilx.vis.heuristic import AnomalyHeuristic
from veilx.vis.metrics import all_scores, calibration_scores, segmentation_scores
from veilx.vis.regions import (
    GRID,
    REGION_NAMES,
    face_cells,
    global_visibility,
    pool,
    region_cells,
    visibility_target,
)
from veilx.vis.samples import check_isolation, draw_condition, make_sample


def test_pool_and_target() -> None:
    m = np.zeros((112, 112), bool)
    m[:56] = True  # top half occluded
    t = visibility_target(m)
    assert t.shape == (GRID, GRID)
    np.testing.assert_allclose(t[:14], 0.0)
    np.testing.assert_allclose(t[14:], 1.0)
    assert pool(np.ones((112, 112))).mean() == 1.0


def test_regions_are_inside_the_face_and_distinct() -> None:
    f = face_cells()
    regs = region_cells()
    assert set(regs) == set(REGION_NAMES)
    for name, r in regs.items():
        assert r.any(), name
        assert (r & ~f).sum() <= 0.25 * r.sum(), name  # regions lie mostly on face cells
    assert not (regs["left_eye"] & regs["right_eye"]).any()
    assert global_visibility(np.ones((GRID, GRID))) == pytest.approx(1.0)


def test_metrics_perfect_and_inverted_predictions() -> None:
    m = np.zeros((112, 112), bool)
    m[50:90, 30:80] = True
    t = visibility_target(m)[None]
    perfect = segmentation_scores((t >= 0.5).astype(float), t)
    assert perfect["iou"] == pytest.approx(1.0) and perfect["dice"] == pytest.approx(1.0)
    inverted = segmentation_scores((t < 0.5).astype(float), t)
    assert inverted["iou"] == pytest.approx(0.0)
    cal = calibration_scores(np.full_like(t, 0.5), t)
    assert cal["brier"] == pytest.approx(0.25)
    s = all_scores(t, t)
    assert s["global_mae"] == pytest.approx(0.0) and s["n"] == 1


def test_covisibility_definitions() -> None:
    a = np.ones((2, GRID, GRID))
    b = np.ones((2, GRID, GRID))
    b[1] = 0.0
    f = pair_features(a, b)
    np.testing.assert_allclose(f["cv_min"], [1.0, 0.0])
    np.testing.assert_allclose(f["cv_prod"], [1.0, 0.0])
    np.testing.assert_allclose(f["cv_spatial"], [1.0, 0.0])
    np.testing.assert_allclose(f["asym"], [0.0, 1.0])
    # spatial co-visibility is lower than min(global) when the visible regions do not overlap
    a2 = np.zeros((1, GRID, GRID))
    a2[0, :14] = 1.0
    b2 = np.zeros((1, GRID, GRID))
    b2[0, 14:] = 1.0
    g = pair_features(a2, b2)
    assert g["cv_spatial"][0] == pytest.approx(0.0) and g["cv_min"][0] > 0.2


def test_logistic_recovers_signal_and_regularises() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(1.5 * x[:, 0] - 0.5)))).astype(np.int64)
    m = LogisticModel(("a", "b"), 1e4).fit(x, y)
    assert m.converged
    assert m.coef[0] / m.scale[0] == pytest.approx(1.5, abs=0.15)
    assert abs(m.coef[1]) < 0.1
    strong = LogisticModel(("a", "b"), 1e-4).fit(x, y)
    assert abs(strong.coef[0]) < abs(m.coef[0])
    back = LogisticModel.from_dict(m.to_dict())
    np.testing.assert_allclose(back.predict_proba(x), m.predict_proba(x))
    best, scores = select_c(x[:3000], y[:3000], x[3000:], y[3000:], ("a", "b"))
    assert best.C in scores and min(scores.values()) == scores[best.C]
    assert log_loss(np.array([0.5]), np.array([1]))[0] == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        LogisticModel(("a",), 1.0).fit(x, y)


def test_heuristic_flags_unusual_cells() -> None:
    rng = np.random.default_rng(1)
    clean = rng.integers(100, 140, (50, 112, 112, 3), dtype=np.uint8)
    h = AnomalyHeuristic.from_clean(clean)
    occ = clean[:10].copy()
    occ[:, :56] = 0  # dark band = occluder
    target = np.ones((10, GRID, GRID))
    target[:, :14] = 0.0
    h.fit_link(occ, target)
    p = h.predict(occ)
    assert p[:, :14].mean() < 0.5 < p[:, 14:].mean()


def test_samples_respect_held_out_family() -> None:
    allowed = ("F1_mask", "F2_sunglasses")
    with pytest.raises(ValueError):
        check_isolation(allowed, "F1_mask")
    check_isolation(allowed, "F4_random_patch")
    rng = np.random.default_rng(2)
    fams = {draw_condition(rng, allowed)[0] for _ in range(400)}
    assert fams <= {None, *allowed} and None in fams
    img = np.full((112, 112, 3), 128, np.uint8)
    out, tgt, meta = make_sample(img, "F2_sunglasses", "high", 7)
    assert out.shape == img.shape and tgt.min() < 1.0 and 0 < meta["visibility_ratio"] < 1
    clean, t2, _ = make_sample(img, None, None, 7)
    assert (t2 == 1).all() and clean is img
