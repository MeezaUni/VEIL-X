"""Model preprocessing and specs (numpy only; runs in CI without torch)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from veilx.models.preprocess import drop_opaque_alpha, to_model_input
from veilx.models.spec import (
    ADAFACE_IR101,
    ARCFACE_IR101,
    BACKBONES,
    PRIMARY_BACKBONE,
    PreprocessSpec,
)


def _img(n: int = 2, c: int = 3, size: int = 112) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (n, size, size, c), dtype=np.uint8)


def test_rgb_normalisation_matches_authors_transform() -> None:
    # ToTensor() then Normalize(0.5, 0.5): (x/255 - 0.5)/0.5, CHW, RGB kept.
    x = _img()
    out = to_model_input(x, PreprocessSpec())
    assert out.shape == (2, 3, 112, 112) and out.dtype == np.float32
    ref = (x.astype(np.float32) / 255.0 - 0.5) / 0.5
    np.testing.assert_allclose(out, ref.transpose(0, 3, 1, 2), atol=1e-6)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_bgr_order_reverses_channels() -> None:
    x = _img()
    rgb = to_model_input(x, PreprocessSpec())
    bgr = to_model_input(x, PreprocessSpec(channel_order="BGR"))
    np.testing.assert_array_equal(rgb[:, ::-1], bgr)


def test_known_pixel_values() -> None:
    x = np.zeros((1, 112, 112, 3), np.uint8)
    x[..., 0] = 255  # pure red
    out = to_model_input(x, PreprocessSpec())
    assert out[0, 0, 0, 0] == pytest.approx(1.0)
    assert out[0, 1, 0, 0] == pytest.approx(-1.0)
    assert out[0, 2, 0, 0] == pytest.approx(-1.0)


def test_rejects_wrong_size_dtype_and_rank() -> None:
    with pytest.raises(ValueError, match="aligned crops"):
        to_model_input(_img(size=128), PreprocessSpec())
    with pytest.raises(TypeError):
        to_model_input(_img().astype(np.float32), PreprocessSpec())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        to_model_input(_img()[0], PreprocessSpec())


def test_alpha_dropped_only_when_opaque() -> None:
    x = _img(c=4)
    x[..., 3] = 255
    np.testing.assert_array_equal(drop_opaque_alpha(x), x[..., :3])
    x[0, 0, 0, 3] = 254
    with pytest.raises(ValueError, match="non-opaque"):
        to_model_input(x, PreprocessSpec())
    with pytest.raises(ValueError, match="channels"):
        drop_opaque_alpha(_img(c=2))


def test_specs_are_pinned_and_distinct() -> None:
    assert ADAFACE_IR101.backbone_id == PRIMARY_BACKBONE
    assert set(BACKBONES) == {ADAFACE_IR101.backbone_id, ARCFACE_IR101.backbone_id}
    for s in BACKBONES.values():
        assert len(s.revision) == 40 and len(s.weights_sha256) == 64
        assert s.embedding_dim == 512 and s.similarity == "cosine"
        assert s.preprocess.channel_order == "RGB" and not s.preprocess.flip_tta
    assert ADAFACE_IR101.fingerprint() != ARCFACE_IR101.fingerprint()


def test_fingerprint_changes_with_preprocessing() -> None:
    bgr = replace(ADAFACE_IR101, preprocess=PreprocessSpec(channel_order="BGR"))
    assert bgr.fingerprint() != ADAFACE_IR101.fingerprint()
    assert PreprocessSpec().fingerprint() == PreprocessSpec().fingerprint()
