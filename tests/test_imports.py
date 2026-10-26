from __future__ import annotations

import importlib
import sys

import pytest

MODULES = [
    "veilx",
    "veilx.errors",
    "veilx.logging_utils",
    "veilx.seeding",
    "veilx.provenance",
    "veilx.guards",
    "veilx.interfaces",
    "veilx.config",
    "veilx.config.schema",
    "veilx.config.loader",
    "veilx.data",
    "veilx.data.manifest",
    "veilx.data.splits",
    "veilx.stats",
    "veilx.stats.feasibility",
]

pytestmark = pytest.mark.structural


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_python_version_supported() -> None:
    assert (3, 11) <= sys.version_info[:2] < (3, 13)


def test_no_heavy_ml_dependency_imported() -> None:
    for name in MODULES:
        importlib.import_module(name)
    # Phase 0 must not pull in model frameworks.
    assert "torch" not in sys.modules
    assert "insightface" not in sys.modules
    assert "onnxruntime" not in sys.modules
