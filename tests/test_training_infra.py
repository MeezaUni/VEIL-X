"""Phase 2 infrastructure (config guard, checkpoints, precision); no training is performed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from veilx.errors import DataAccessError, ProtocolViolationError
from veilx.training.checkpoint import CheckpointManager
from veilx.training.config import (
    AUTH_ENV,
    TrainingRunConfig,
    assert_training_authorized,
    load_training_config,
)
from veilx.training.precision import autocast, describe_accelerator, torch_available

from .conftest import ROOT

TEMPLATE = ROOT / "configs" / "experiments" / "phase2_template.yaml"


def template_dict() -> dict[str, Any]:
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


def test_template_is_valid_and_not_authorised(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_training_config(TEMPLATE)
    assert cfg.authorized is False and cfg.backbone_frozen is True
    monkeypatch.setenv(AUTH_ENV, "1")
    with pytest.raises(ProtocolViolationError, match="not authorised"):
        assert_training_authorized(cfg)


def test_authorisation_needs_config_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = TrainingRunConfig.model_validate({**template_dict(), "authorized": True})
    monkeypatch.delenv(AUTH_ENV, raising=False)
    with pytest.raises(ProtocolViolationError):
        assert_training_authorized(cfg)
    monkeypatch.setenv(AUTH_ENV, "1")
    assert_training_authorized(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_pairs", "pairs/test_balanced.jsonl"),
        ("val_pairs", "pairs/cal_independent.jsonl"),
        ("backbone_frozen", False),
        ("run_id", "Bad ID"),
        ("checkpoint", {"directory": "../escape"}),
    ],
)
def test_config_rejects_protocol_violations(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig.model_validate({**template_dict(), field: value})


def test_checkpoint_roundtrip_rotation_and_integrity(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path / "ck", keep_last=2)
    for step in (10, 20, 30):
        mgr.save(step, f"state-{step}".encode(), {"loss": step / 10})
    items = mgr.list()
    assert [c.step for c in items] == [20, 30]  # rotated
    latest = mgr.latest()
    assert latest is not None and mgr.load(latest) == b"state-30"
    latest.path.write_bytes(b"tampered")
    with pytest.raises(DataAccessError, match="SHA-256"):
        mgr.load(latest)
    assert not list((tmp_path / "ck").glob("*.tmp"))
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path / "x", keep_last=0)


def test_precision_helpers_without_or_with_torch() -> None:
    from veilx.training.config import PrecisionPolicy

    with autocast(PrecisionPolicy(mode="fp32"), "cpu"):
        pass
    info = describe_accelerator()
    assert ("torch" in info) and isinstance(info["cuda_available"], bool)
    if torch_available():
        with autocast(PrecisionPolicy(mode="bf16"), "cpu"):
            pass
