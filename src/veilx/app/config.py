"""Frozen configuration of the Phase 5 local research application (configs/app/veilx_app.yaml).

The schema enforces local-only operation (loopback host, no public share link, no analytics) and
the research-only flag. :func:`load_frozen_decision` reads the PRIVATE Phase 3 freeze, verifies its
SHA-256 (the same content hash the evaluation scripts use), and returns the exact A3 rule of the
chosen fold. Nothing here is re-fitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from veilx.fusion.logreg import LogisticModel


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionSection(_Strict):
    default_fold: str
    folds: tuple[str, ...]
    arm: Literal["A3_quality"]
    target: Literal["primary"]
    ordering: Literal["v2"]
    fused_score: Literal["M1"]
    phase3_frozen_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase3_frozen_file: str
    # App-only safety guard (Phase 6, D-055): 1st percentile of each fold's calibration q.
    # Below it the frozen M1 score extrapolates (p -> 1 for non-faces); the app abstains.
    q_support_floor: dict[str, float] | None = None


class DiagnosticsSection(_Strict):
    phase4_frozen_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    phase4_frozen_file: str
    vec_dir: str
    vec_sha256: dict[str, str]


class DetectorSection(_Strict):
    model: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    det_thresh: float = Field(gt=0, lt=1)
    nms_thresh: float = Field(gt=0, lt=1)
    min_score: float = Field(gt=0, lt=1)
    max_residual: float = Field(gt=0)


class FallbackSection(_Strict):
    aspect_ratio_range: tuple[float, float]
    reference_size: int = Field(ge=112)
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]


class InputSection(_Strict):
    max_bytes: int = Field(gt=0)
    min_side: int = Field(ge=16)
    max_side: int = Field(ge=112)
    formats: tuple[str, ...]


class ServerSection(_Strict):
    host: str
    share: Literal[False]
    analytics: Literal[False]

    @field_validator("host")
    @classmethod
    def _loopback(cls, v: str) -> str:
        if v not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the research app may only bind to a loopback address")
        return v


class AppConfig(_Strict):
    schema_version: Literal[1]
    research_only: Literal[True]
    decision: DecisionSection
    diagnostics: DiagnosticsSection
    detector: DetectorSection
    fallback: FallbackSection
    inputs: InputSection
    server: ServerSection

    @field_validator("decision")
    @classmethod
    def _default_in_folds(cls, v: DecisionSection) -> DecisionSection:
        if v.default_fold not in v.folds:
            raise ValueError("default_fold must be one of the frozen folds")
        return v


def load_app_config(path: Path) -> AppConfig:
    return AppConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def content_sha256(frozen: dict[str, Any]) -> str:
    """Same content hash as scripts/evaluate_phase3.py / evaluate_phase4.py (``_sha``)."""
    body = {k: v for k, v in frozen.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenRule:
    """The exact A3 decision system of one Phase 3 fold."""

    fold: str
    model: LogisticModel  # M1: features (s, q, s*q)
    q_edges: tuple[float, float]  # quality tertile edges (p3_val)
    rules: dict[str, tuple[float, float] | None]  # stratum -> (tau_low, tau_high)
    clean_far1_threshold: float  # M1 clean-cal FAR-1 % threshold (reported, not a decision)


def load_frozen_decision(path: Path, expected_sha256: str, fold: str) -> FrozenRule:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("sha256") != expected_sha256 or content_sha256(frozen) != expected_sha256:
        raise RuntimeError("Phase 3 frozen configuration does not match the registered SHA-256")
    f = frozen["folds"][fold]
    rec = f["arms"]["A3_quality|primary|v2"]
    if rec["model"] != "M1" or rec["strata"] != "quality":
        raise RuntimeError("unexpected A3 record in the frozen configuration")
    rules = {
        k: (None if v is None else (float(v[0]), float(v[1]))) for k, v in rec["rules"].items()
    }
    q = f["edges"]["q"]
    return FrozenRule(
        fold=fold,
        model=LogisticModel.from_dict(f["models"]["M1"]),
        q_edges=(float(q[0]), float(q[1])),
        rules=rules,
        clean_far1_threshold=float(f["h4_thresholds"]["M1"]),
    )
