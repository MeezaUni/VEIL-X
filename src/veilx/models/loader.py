"""Strict, safe checkpoint loading.

* safetensors only (no pickle execution); the file's SHA-256 must equal the pinned value;
* every checkpoint key must carry the expected prefix;
* the key set must equal the architecture's ``state_dict`` exactly - any missing or
  unexpected key raises (the authors' own loader uses ``strict=False``; we never do);
* shapes and dtypes must match per tensor, then ``load_state_dict(strict=True)``;
* the loaded model is put in ``eval`` mode with gradients disabled, and a fingerprint of
  the loaded tensors is recomputed and compared with the checkpoint's.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from veilx.data.archive import sha256_file
from veilx.errors import ModelLoadError
from veilx.models.architectures import build_architecture
from veilx.models.spec import BackboneSpec


def tensor_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over (name, dtype, shape, raw bytes) of every tensor, in sorted name order."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().to("cpu").contiguous()
        h.update(f"{k}|{t.dtype}|{tuple(t.shape)}|".encode())
        # 0-d tensors cannot be byte-viewed directly, hence the flatten.
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    bad = [k for k in state if not k.startswith(prefix)]
    if bad:
        raise ModelLoadError(f"{len(bad)} checkpoint keys lack prefix {prefix!r}: {bad[:5]}")
    return {k[len(prefix) :]: v for k, v in state.items()}


def check_state_compatible(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ModelLoadError(
            f"key mismatch: {len(missing)} missing {missing[:5]}, "
            f"{len(unexpected)} unexpected {unexpected[:5]}"
        )
    bad = [
        f"{k}: ckpt {tuple(state[k].shape)}/{state[k].dtype} vs model {tuple(v.shape)}/{v.dtype}"
        for k, v in expected.items()
        if tuple(state[k].shape) != tuple(v.shape) or state[k].dtype != v.dtype
    ]
    if bad:
        raise ModelLoadError(f"{len(bad)} shape/dtype mismatches: {bad[:3]}")


def load_backbone(
    spec: BackboneSpec, weights_dir: Path, *, verify_hash: bool = True
) -> tuple[nn.Module, str]:
    """Return ``(model in eval mode on CPU, loaded-tensor fingerprint)``."""
    from safetensors.torch import load_file

    path = weights_dir / spec.weights_file
    if not path.is_file():
        raise ModelLoadError(f"{spec.backbone_id}: weights not found ({spec.weights_file})")
    if path.stat().st_size != spec.weights_bytes:
        raise ModelLoadError(f"{spec.backbone_id}: weight size {path.stat().st_size} != pinned")
    if verify_hash and sha256_file(path) != spec.weights_sha256:
        raise ModelLoadError(f"{spec.backbone_id}: weight SHA-256 differs from pinned value")
    state = strip_prefix(load_file(str(path), device="cpu"), spec.state_dict_prefix)
    model = build_architecture(spec.architecture, spec.embedding_dim)
    check_state_compatible(model, state)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:  # unreachable under strict=True
        raise ModelLoadError(f"{spec.backbone_id}: strict load reported {result}")
    model.eval()
    model.requires_grad_(False)
    loaded = tensor_fingerprint(model.state_dict())
    if loaded != tensor_fingerprint(state):
        raise ModelLoadError(f"{spec.backbone_id}: loaded tensors differ from checkpoint")
    return model, loaded
