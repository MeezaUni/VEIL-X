"""Session temp directories of the local app: creation, PID marking and stale cleanup.

Each app session creates ``<system temp>/veilx_app_<random>/`` and writes its PID to ``owner.pid``.
Uploads (Gradio cache) and synthetic examples live there only. It is deleted on normal exit. At
startup, directories left behind by sessions whose process no longer exists (e.g. a killed
process) are removed. Only directories with the ``veilx_app_`` prefix are ever touched.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

PREFIX = "veilx_app_"
PID_FILE = "owner.pid"


def _alive(pid: int) -> bool:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is present in the app environment
        return True  # unknown: keep the directory (never delete what might be in use)
    return bool(psutil.pid_exists(pid))


def create_session_dir(base: Path | None = None) -> Path:
    d = Path(tempfile.mkdtemp(prefix=PREFIX, dir=None if base is None else str(base)))
    (d / PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    return d


def cleanup_stale(base: Path | None = None) -> list[Path]:
    """Remove ``veilx_app_*`` directories whose recorded owner process is gone."""
    root = Path(tempfile.gettempdir()) if base is None else base
    removed = []
    for d in root.glob(f"{PREFIX}*"):
        if not d.is_dir():
            continue
        pid_path = d / PID_FILE
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue  # unknown owner (e.g. created by an older version): leave it
        if pid != os.getpid() and not _alive(pid):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    return removed


def remove_session_dir(d: Path) -> None:
    if d.name.startswith(PREFIX):
        shutil.rmtree(d, ignore_errors=True)
