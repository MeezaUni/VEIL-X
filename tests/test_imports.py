from __future__ import annotations

import importlib
import subprocess
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
    # Core modules must not pull in model frameworks. Checked in a fresh interpreter, because other
    # test files (model tests in the GPU environment) legitimately import torch into this process.
    code = (
        f"import importlib, sys; [importlib.import_module(n) for n in {MODULES!r}]; "
        "sys.exit(any(m in sys.modules for m in ('torch', 'insightface', 'onnxruntime')))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
