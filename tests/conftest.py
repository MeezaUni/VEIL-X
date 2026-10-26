from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "research_protocol.yaml"
DATASETS = ROOT / "configs" / "datasets.yaml"


@pytest.fixture(scope="session")
def protocol_dict() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))


@pytest.fixture
def mutable_protocol(protocol_dict: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(protocol_dict)


@pytest.fixture(scope="session")
def registry_dict() -> dict[str, Any]:
    return yaml.safe_load(DATASETS.read_text(encoding="utf-8"))
