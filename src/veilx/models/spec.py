"""Pinned specifications of the approved frozen backbones.

Every fact here was read from the authors' released files at the pinned revision (config.json,
pretrained_model/model.yaml, models/*/__init__.py ``make_test_transform``) or from the Hub's
LFS record of the weight file; see docs/model_card_backbones.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

ARCFACE_TEMPLATE_112: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


@dataclass(frozen=True)
class PreprocessSpec:
    """Exact input contract. ``(x / 255 - mean) / std`` per channel, channels in ``channel_order``,
    NCHW float32. No resize is ever performed silently: inputs must already be
    ``input_size`` x ``input_size`` crops aligned to ``alignment_template``."""

    input_size: int = 112
    channel_order: Literal["RGB", "BGR"] = "RGB"
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    alignment_template: tuple[tuple[float, float], ...] = ARCFACE_TEMPLATE_112
    flip_tta: bool = False  # single view; horizontal-flip fusion is not used

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class BackboneSpec:
    backbone_id: str
    architecture: Literal["cvlface_ir101", "insightface_iresnet100"]
    hf_repo: str
    revision: str
    weights_file: str
    weights_sha256: str
    weights_bytes: int
    state_dict_prefix: str
    training_data: str
    loss: str
    paper: str
    licence: str
    embedding_dim: int
    similarity: Literal["cosine"]
    preprocess: PreprocessSpec

    def fingerprint(self) -> str:
        """Stable id of (weights, architecture, preprocessing) used to key embedding caches."""
        blob = json.dumps(
            {
                "id": self.backbone_id,
                "arch": self.architecture,
                "w": self.weights_sha256,
                "pre": self.preprocess.fingerprint(),
                "dim": self.embedding_dim,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


_WEBFACE_LICENCE = (
    "CVLface code MIT; weights trained on WebFace4M - no licence file in the model repo, the "
    "card asks users to follow the training-dataset licence (WebFace: non-commercial research "
    "only). Used here for non-commercial academic research; weights never redistributed."
)

ADAFACE_IR101 = BackboneSpec(
    backbone_id="adaface_ir101_webface4m",
    architecture="cvlface_ir101",
    hf_repo="minchul/cvlface_adaface_ir101_webface4m",
    revision="f2b38d9e24bfe301490d8dd081d8924b102333dd",
    weights_file="model.safetensors",
    weights_sha256="21adb6220e8799a0e658f16946df9649c7269f432fe9810a7b9c4ad1241080a8",
    weights_bytes=260_980_552,
    state_dict_prefix="model.net.",
    training_data="WebFace4M",
    loss="AdaFace",
    paper="arXiv:2204.00964",
    licence=_WEBFACE_LICENCE,
    embedding_dim=512,
    similarity="cosine",
    preprocess=PreprocessSpec(),
)

ARCFACE_IR101 = BackboneSpec(
    backbone_id="arcface_ir101_webface4m",
    architecture="insightface_iresnet100",
    hf_repo="minchul/cvlface_arcface_ir101_webface4m",
    revision="5846a5e96fbe1094952a414c78ce5b057c92d013",
    weights_file="model.safetensors",
    weights_sha256="e90e0c8e24c94409fe53c2254a1343ef54f48a3a66b885b8ff69aefa21660ee7",
    weights_bytes=260_997_808,
    state_dict_prefix="model.net.",
    training_data="WebFace4M",
    loss="ArcFace",
    paper="arXiv:1801.07698",
    licence=_WEBFACE_LICENCE,
    embedding_dim=512,
    similarity="cosine",
    preprocess=PreprocessSpec(),
)

BACKBONES: dict[str, BackboneSpec] = {s.backbone_id: s for s in (ADAFACE_IR101, ARCFACE_IR101)}
PRIMARY_BACKBONE = ADAFACE_IR101.backbone_id
