"""Reproducible experiment configuration for future GPU runs (Phase 2+).

Training is **not authorised** in Phase 1b. :func:`assert_training_authorized` refuses to start
any run unless BOTH the config sets ``authorized: true`` AND the environment variable
``VEILX_PHASE2_AUTHORIZED=1`` is present, so a committed config alone can never launch training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from veilx.errors import ConfigError, ProtocolViolationError

AUTH_ENV = "VEILX_PHASE2_AUTHORIZED"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointPolicy(_Strict):
    directory: str = Field(description="relative to the private data root")
    keep_last: int = Field(ge=1, default=3)
    save_every_steps: int = Field(ge=1, default=1000)

    @model_validator(mode="after")
    def _relative(self) -> CheckpointPolicy:
        p = Path(self.directory)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("checkpoint directory must be relative to the data root")
        return self


class PrecisionPolicy(_Strict):
    mode: Literal["fp32", "fp16", "bf16"] = "fp32"
    allow_tf32: bool = False
    deterministic: bool = True


class TrainingRunConfig(_Strict):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]{2,63}$")
    phase: Literal["phase2", "phase3"]
    authorized: bool = False
    purpose: str = Field(min_length=10)
    seed: int = Field(ge=0)
    device: Literal["cuda", "cpu"] = "cuda"
    backbone_id: str
    backbone_frozen: Literal[True] = True  # protocol D-003: no backbone fine-tuning
    heldout_family: str | None = None
    train_pairs: str
    val_pairs: str
    batch_size: int = Field(ge=1)
    grad_accumulation: int = Field(ge=1, default=1)
    max_epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    precision: PrecisionPolicy = PrecisionPolicy()
    checkpoint: CheckpointPolicy

    @model_validator(mode="after")
    def _no_test_data(self) -> TrainingRunConfig:
        for f in (self.train_pairs, self.val_pairs):
            if "test" in Path(f).name.lower() or "cal" in Path(f).name.lower():
                raise ValueError("training/validation inputs may not be test or calibration pairs")
        return self


def load_training_config(path: Path) -> TrainingRunConfig:
    try:
        return TrainingRunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ConfigError(f"{path.name}: invalid training config ({exc})") from exc


def assert_training_authorized(cfg: TrainingRunConfig) -> None:
    if not cfg.authorized or os.environ.get(AUTH_ENV) != "1":
        raise ProtocolViolationError(
            f"training run {cfg.run_id!r} is not authorised (Phase 2 requires a separate PI "
            f"authorisation: set authorized: true in the config AND {AUTH_ENV}=1)"
        )
