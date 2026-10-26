"""YAML loading with clean, protocol-aware error messages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from veilx.config.schema import DatasetRegistry, ResearchProtocol
from veilx.errors import ConfigError

_M = TypeVar("_M", bound=BaseModel)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path.name}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML ({exc.__class__.__name__})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    return data


def _validate(model: type[_M], data: dict[str, Any], source: str) -> _M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        lines = [
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigError(
            f"{source}: {len(lines)} validation error(s)\n" + "\n".join(lines)
        ) from exc


def load_protocol(path: Path) -> ResearchProtocol:
    """Load and validate ``research_protocol.yaml``."""
    return _validate(ResearchProtocol, _read_yaml(path), path.name)


def load_dataset_registry(path: Path) -> DatasetRegistry:
    """Load and validate ``datasets.yaml``."""
    return _validate(DatasetRegistry, _read_yaml(path), path.name)


def config_sha256(path: Path) -> str:
    """SHA-256 of the raw config bytes (used in provenance records)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
