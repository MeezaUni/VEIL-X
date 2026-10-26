"""Location of private data (archives, images, manifests) — always outside version control.

The data root comes from the ``VEILX_DATA_ROOT`` environment variable. If unset, it defaults to
``<project>/data`` (git-ignored). A data root inside the project is accepted only if git
ignores it; anything else raises, so biometric data can never be staged by accident.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from veilx.errors import DataAccessError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_VAR = "VEILX_DATA_ROOT"


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @property
    def raw(self) -> Path:  # downloaded archives
        return self.root / "raw"

    @property
    def extracted(self) -> Path:  # extracted images (one directory per archive part)
        return self.root / "extracted"

    @property
    def manifests(self) -> Path:  # private manifests and pair lists (contain identity ids)
        return self.root / "manifests"

    @property
    def private_reports(self) -> Path:  # private reports (may list image ids)
        return self.root / "reports"

    @property
    def models(self) -> Path:  # third-party model weights (licence-restricted; never committed)
        return self.root / "models"

    @property
    def embeddings(self) -> Path:  # extracted embeddings (biometric templates; private)
        return self.root / "embeddings"

    def ensure(self) -> DataLayout:
        for p in (
            self.raw,
            self.extracted,
            self.manifests,
            self.private_reports,
            self.models,
            self.embeddings,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _is_git_ignored(path: Path) -> bool:
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "check-ignore", "-q", "--no-index", str(path)],  # noqa: S607
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
    except OSError:
        return False
    return res.returncode == 0


def resolve_data_root(explicit: Path | None = None) -> DataLayout:
    root = explicit or Path(os.environ.get(ENV_VAR, PROJECT_ROOT / "data"))
    root = root.resolve()
    try:
        inside = root.is_relative_to(PROJECT_ROOT)
    except ValueError:
        inside = False
    # Probe with an extension-less name so that only *directory* rules count (a name such as
    # "x.png" would be ignored anywhere by the image-extension rules).
    if inside and not _is_git_ignored(root / "veilx_data_root_probe"):
        raise DataAccessError(
            "data root lies inside the repository but is not git-ignored; refusing to use it"
        )
    return DataLayout(root)
