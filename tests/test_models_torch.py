"""Strict loader and architectures (requires torch; skipped in the CPU-only CI environment).

Uses randomly initialised architectures saved to temporary safetensors files - no real
weights are needed or touched.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from veilx.data.archive import sha256_file  # noqa: E402
from veilx.errors import ModelLoadError  # noqa: E402
from veilx.models.architectures import (  # noqa: E402
    CVLfaceIR101,
    InsightfaceIResNet100,
    build_architecture,
)
from veilx.models.extractor import FrozenExtractor  # noqa: E402
from veilx.models.loader import load_backbone, tensor_fingerprint  # noqa: E402
from veilx.models.spec import ADAFACE_IR101, ARCFACE_IR101, BackboneSpec  # noqa: E402


def _write_ckpt(tmp: Path, spec: BackboneSpec, mutate: str | None = None) -> BackboneSpec:
    torch.manual_seed(0)
    model = build_architecture(spec.architecture)
    state = {spec.state_dict_prefix + k: v.clone() for k, v in model.state_dict().items()}
    if mutate == "missing":
        state.pop(next(iter(state)))
    elif mutate == "unexpected":
        state[spec.state_dict_prefix + "extra.weight"] = torch.zeros(1)
    elif mutate == "shape":
        k = next(k for k in state if k.endswith("fc.weight") or k.endswith("output_layer.3.weight"))
        state[k] = torch.zeros(4, 4)
    elif mutate == "prefix":
        state = {k.replace(spec.state_dict_prefix, ""): v for k, v in state.items()}
    path = tmp / spec.weights_file
    save_file(state, str(path))
    return replace(spec, weights_sha256=sha256_file(path), weights_bytes=path.stat().st_size)


@pytest.mark.parametrize("spec", [ADAFACE_IR101, ARCFACE_IR101], ids=lambda s: s.backbone_id)
def test_strict_load_roundtrip(tmp_path: Path, spec: BackboneSpec) -> None:
    s = _write_ckpt(tmp_path, spec)
    model, fp = load_backbone(s, tmp_path)
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())
    assert fp == tensor_fingerprint(model.state_dict())
    out = model(torch.zeros(2, 3, 112, 112))
    assert tuple(out.shape) == (2, 512)


@pytest.mark.parametrize("mutate", ["missing", "unexpected", "shape", "prefix"])
def test_strict_load_rejects_mismatch(tmp_path: Path, mutate: str) -> None:
    s = _write_ckpt(tmp_path, ARCFACE_IR101, mutate)
    with pytest.raises(ModelLoadError):
        load_backbone(s, tmp_path)


def test_hash_and_size_pinning(tmp_path: Path) -> None:
    s = _write_ckpt(tmp_path, ARCFACE_IR101)
    with pytest.raises(ModelLoadError, match="SHA-256"):
        load_backbone(replace(s, weights_sha256="0" * 64), tmp_path)
    with pytest.raises(ModelLoadError, match="size"):
        load_backbone(replace(s, weights_bytes=1), tmp_path)
    with pytest.raises(ModelLoadError, match="not found"):
        load_backbone(s, tmp_path / "nowhere")


def test_architectures_have_expected_depth() -> None:
    ir = CVLfaceIR101()
    assert len(ir.body) == 3 + 13 + 30 + 3
    rn = InsightfaceIResNet100()
    assert [len(x) for x in (rn.layer1, rn.layer2, rn.layer3, rn.layer4)] == [3, 13, 30, 3]
    with pytest.raises(ValueError):
        build_architecture("vit")


def test_extractor_outputs_unit_norm_and_is_deterministic(tmp_path: Path) -> None:
    s = _write_ckpt(tmp_path, ADAFACE_IR101)
    ex = FrozenExtractor(s, tmp_path, device="cpu")
    crops = np.random.default_rng(1).integers(0, 256, (3, 112, 112, 3), dtype=np.uint8)
    b1, b2 = ex.embed(crops), ex.embed(crops)
    np.testing.assert_allclose(np.linalg.norm(b1.embeddings, axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(b1.embeddings, b2.embeddings)
    assert b1.feature_norms.shape == (3,) and (b1.feature_norms > 0).all()
    assert ex.fingerprint.endswith("-fp32")
    with pytest.raises(ValueError, match="CUDA"):
        FrozenExtractor(s, tmp_path, device="cpu", precision="fp16")
