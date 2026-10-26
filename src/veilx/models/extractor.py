"""Frozen embedding extractor implementing :class:`veilx.interfaces.EmbeddingExtractor`.

Inference only (``torch.inference_mode``, eval mode, no gradients). Precision is ``fp32`` by
default; ``bf16``/``fp16`` autocast is optional and must be validated against fp32 before use
(``scripts/verify_models.py``). Raw embeddings are L2-normalised in float32; the
pre-normalisation norm is returned as a quality proxy. Non-finite outputs or zero norms raise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from veilx.errors import ModelLoadError
from veilx.interfaces import EmbeddingBatch
from veilx.models.loader import load_backbone
from veilx.models.preprocess import to_model_input
from veilx.models.spec import BackboneSpec

Precision = Literal["fp32", "bf16", "fp16"]
_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class FrozenExtractor:
    def __init__(
        self,
        spec: BackboneSpec,
        weights_dir: Path,
        *,
        device: str | None = None,
        precision: Precision = "fp32",
        verify_hash: bool = True,
    ) -> None:
        self.spec = spec
        self.backbone_id = spec.backbone_id
        self.embedding_dim = spec.embedding_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if precision != "fp32" and self.device.type != "cuda":
            raise ValueError("mixed precision is only supported on CUDA here")
        self.precision: Precision = precision
        model, self.weights_fingerprint = load_backbone(spec, weights_dir, verify_hash=verify_hash)
        self.model = model.to(self.device)
        torch.backends.cudnn.benchmark = False

    @property
    def fingerprint(self) -> str:
        """Cache key: backbone spec (weights + preprocessing) and numeric precision."""
        return f"{self.spec.fingerprint()}-{self.precision}"

    def raw(self, batch_nchw: NDArray[np.float32]) -> NDArray[np.float32]:
        with torch.inference_mode():
            x = torch.from_numpy(batch_nchw).to(self.device, non_blocking=True)
            if self.precision == "fp32":
                out = self.model(x)
            else:
                with torch.autocast("cuda", dtype=_AMP_DTYPES[self.precision]):
                    out = self.model(x)
        arr: NDArray[np.float32] = out.float().cpu().numpy()
        return arr

    def embed(self, crops: NDArray[np.uint8]) -> EmbeddingBatch:
        raw = self.raw(to_model_input(crops, self.spec.preprocess))
        if raw.shape != (crops.shape[0], self.embedding_dim):
            raise ModelLoadError(f"unexpected output shape {raw.shape}")
        if not np.isfinite(raw).all():
            raise FloatingPointError(f"{self.backbone_id}: non-finite embedding values")
        norms = np.linalg.norm(raw, axis=1)
        if (norms <= 0).any():
            raise FloatingPointError(f"{self.backbone_id}: zero-norm embedding")
        emb = (raw / norms[:, None]).astype(np.float32)
        return EmbeddingBatch(
            backbone_id=self.backbone_id, embeddings=emb, feature_norms=norms.astype(np.float32)
        )
