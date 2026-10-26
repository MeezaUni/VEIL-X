"""Provenance records for every run.

Records git commit and dirty state, interpreter and package versions, config hashes,
seeds and the invoking command. Deliberately **excludes** hostname, username and absolute
paths (privacy): paths are recorded relative to the project root when possible.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from veilx import __version__
from veilx.config.loader import config_sha256

_PACKAGES = ("numpy", "pydantic", "PyYAML", "torch", "onnxruntime", "insightface")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name  # never leak absolute locations


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def collect_provenance(
    project_root: Path,
    *,
    configs: list[Path] | None = None,
    seed: int | None = None,
    argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"], project_root)
    toplevel = _git(["rev-parse", "--show-toplevel"], project_root)
    status = _git(["status", "--porcelain"], project_root)
    return {
        "veilx_version": __version__,
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "git": {
            "commit": commit,
            "dirty": bool(status) if status is not None else None,
            "repo_is_project_root": (
                Path(toplevel).resolve() == project_root.resolve() if toplevel else None
            ),
        },
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "os": platform.system(),
        "packages": package_versions(),
        "configs": {_relative(p, project_root): config_sha256(p) for p in (configs or [])},
        "seed": seed,
        "argv": [a if not Path(a).is_absolute() else Path(a).name for a in (argv or [])],
        "extra": extra or {},
    }


def write_provenance(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
