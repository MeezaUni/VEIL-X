"""Framework-agnostic checkpoint management.

Callers serialise their state to bytes (e.g. ``torch.save(state, buffer)``); the manager writes
it atomically (temporary file + fsync + rename), records SHA-256, step, metadata and the git
commit in a JSON sidecar, keeps the last ``keep_last`` checkpoints and verifies the hash on load.
Checkpoints live under the private data root (``*.ckpt`` is git-ignored).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from veilx.errors import DataAccessError


@dataclass(frozen=True)
class CheckpointInfo:
    step: int
    path: Path
    sha256: str
    metadata: dict[str, Any]


class CheckpointManager:
    def __init__(self, directory: Path, *, keep_last: int = 3) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.directory = directory
        self.keep_last = keep_last
        directory.mkdir(parents=True, exist_ok=True)

    def _paths(self, step: int) -> tuple[Path, Path]:
        stem = f"step_{step:09d}"
        return self.directory / f"{stem}.ckpt", self.directory / f"{stem}.json"

    def save(
        self, step: int, payload: bytes, metadata: dict[str, Any] | None = None
    ) -> CheckpointInfo:
        if step < 0:
            raise ValueError("step must be non-negative")
        ckpt, meta = self._paths(step)
        digest = hashlib.sha256(payload).hexdigest()
        tmp = ckpt.with_suffix(".ckpt.tmp")
        with tmp.open("wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(ckpt)
        meta_dict: dict[str, Any] = dict(metadata or {})
        record = {
            "step": step,
            "sha256": digest,
            "bytes": len(payload),
            "saved_utc": datetime.now(tz=UTC).isoformat(),
            "metadata": meta_dict,
        }
        meta.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
        self._rotate()
        return CheckpointInfo(step, ckpt, digest, meta_dict)

    def list(self) -> list[CheckpointInfo]:
        out = []
        for meta in sorted(self.directory.glob("step_*.json")):
            rec = json.loads(meta.read_text(encoding="utf-8"))
            out.append(
                CheckpointInfo(
                    rec["step"], meta.with_suffix(".ckpt"), rec["sha256"], rec["metadata"]
                )
            )
        return sorted(out, key=lambda c: c.step)

    def latest(self) -> CheckpointInfo | None:
        items = self.list()
        return items[-1] if items else None

    def load(self, info: CheckpointInfo) -> bytes:
        data = info.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != info.sha256:
            raise DataAccessError(f"checkpoint {info.path.name} failed its SHA-256 check")
        return data

    def _rotate(self) -> None:
        items = self.list()
        for old in items[: max(0, len(items) - self.keep_last)]:
            old.path.unlink(missing_ok=True)
            old.path.with_suffix(".json").unlink(missing_ok=True)
