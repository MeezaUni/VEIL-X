"""Detection decoding and template alignment (numpy maths; synthetic inputs only)."""

from __future__ import annotations

import numpy as np
import pytest

from veilx.detect.align import TEMPLATE, align_to_template, transform_points, umeyama, warp
from veilx.detect.scrfd import (
    Detection,
    anchor_centers,
    decode,
    distance2bbox,
    distance2kps,
    nms,
    select_central,
)


def _similarity(scale: float, angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty]])


def test_umeyama_recovers_known_similarity() -> None:
    m = _similarity(1.7, 0.3, 12.0, -5.0)
    src = np.random.default_rng(0).uniform(0, 200, (5, 2))
    est = umeyama(src, transform_points(m, src))
    np.testing.assert_allclose(est, m, atol=1e-9)


def test_umeyama_rejects_bad_input_and_avoids_reflection() -> None:
    with pytest.raises(ValueError):
        umeyama(np.zeros((5, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        umeyama(np.ones((5, 2)), np.ones((5, 2)))
    src = TEMPLATE.copy()
    mirrored = src * np.array([-1.0, 1.0])
    est = umeyama(src, mirrored)
    assert np.linalg.det(est[:, :2]) > 0  # a proper similarity, never a reflection


def test_alignment_maps_landmarks_onto_template() -> None:
    m = _similarity(2.0, -0.2, 40.0, 30.0)
    img = np.random.default_rng(1).integers(0, 256, (250, 250, 3), dtype=np.uint8)
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
    landmarks = transform_points(inv, TEMPLATE)  # where the template points sit in the photo
    crop, est = align_to_template(img, landmarks)
    assert crop.shape == (112, 112, 3) and crop.dtype == np.uint8
    np.testing.assert_allclose(transform_points(est, landmarks), TEMPLATE, atol=1e-6)
    np.testing.assert_allclose(est, m, atol=1e-6)


def test_warp_identity_is_lossless() -> None:
    img = np.random.default_rng(2).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    np.testing.assert_array_equal(warp(img, np.array([[1.0, 0, 0], [0, 1.0, 0]])), img)


def test_anchor_centres_and_distance_decoding() -> None:
    c = anchor_centers(2, 3, 8)
    assert c.shape == (12, 2)
    np.testing.assert_array_equal(c[:4], [[0, 0], [0, 0], [8, 0], [8, 0]])  # 2 anchors per cell
    np.testing.assert_array_equal(c[6], [0, 8])
    box = distance2bbox(np.array([[10.0, 20.0]], np.float32), np.array([[1, 2, 3, 4]], np.float32))
    np.testing.assert_array_equal(box, [[9, 18, 13, 24]])
    kps = distance2kps(np.array([[10.0, 20.0]], np.float32), np.arange(10, dtype=np.float32)[None])
    np.testing.assert_array_equal(kps[0, :4], [10, 21, 12, 23])


def test_decode_keeps_only_confident_anchors() -> None:
    h = w = 64
    outs = []
    for stride in (8, 16, 32):
        n = (h // stride) * (w // stride) * 2
        outs.append(np.zeros((n, 1), np.float32))
    outs[0][5, 0] = 0.9  # one confident anchor at stride 8
    for stride in (8, 16, 32):
        n = (h // stride) * (w // stride) * 2
        outs.append(np.ones((n, 4), np.float32))
    for stride in (8, 16, 32):
        n = (h // stride) * (w // stride) * 2
        outs.append(np.zeros((n, 10), np.float32))
    scores, boxes, kps = decode(outs, (h, w), 0.5)
    assert scores.tolist() == pytest.approx([0.9])
    cx, cy = anchor_centers(8, 8, 8)[5]
    np.testing.assert_allclose(boxes[0], [cx - 8, cy - 8, cx + 8, cy + 8])
    assert kps.shape == (1, 5, 2)


def test_nms_and_central_selection() -> None:
    dets = np.array([[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [50, 50, 60, 60, 0.7]], np.float32)
    assert nms(dets, 0.4) == [0, 2]
    faces = [
        Detection(np.array([0, 0, 40, 40], np.float32), 0.99, np.zeros((5, 2), np.float32)),
        Detection(np.array([100, 100, 150, 150], np.float32), 0.6, np.zeros((5, 2), np.float32)),
    ]
    assert select_central(faces, (250, 250)) == 1  # the centred face, not the most confident
    assert select_central([], (250, 250)) is None
