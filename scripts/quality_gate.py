"""Run the full local quality gate: Ruff lint, Ruff format check, mypy, pytest, privacy scan,
and config validation. Exit code is non-zero if any step fails.

    python scripts/quality_gate.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS: list[tuple[str, list[str]]] = [
    ("ruff lint", [PY, "-m", "ruff", "check", "."]),
    ("ruff format", [PY, "-m", "ruff", "format", "--check", "."]),
    ("mypy", [PY, "-m", "mypy"]),
    ("pytest", [PY, "-m", "pytest", "-q"]),
    ("config", [PY, "scripts/validate_config.py"]),
    ("privacy scan", [PY, "scripts/scan_sensitive.py"]),
]

# torch-dependent tests are skipped in the CPU-only environment (and in CI); when the local
# GPU environment exists they are run there as an extra step.
_GPU_PY = next(
    (
        p
        for p in (
            ROOT / ".venv-gpu" / "Scripts" / "python.exe",
            ROOT / ".venv-gpu" / "bin" / "python",
        )
        if p.exists()
    ),
    None,
)
if _GPU_PY is not None:
    STEPS.append(
        ("pytest (torch)", [str(_GPU_PY), "-m", "pytest", "-q", "tests/test_models_torch.py"])
    )


def main() -> int:
    failed: list[str] = []
    for name, cmd in STEPS:
        print(f"=== {name}: {' '.join(Path(c).name if Path(c).is_absolute() else c for c in cmd)}")
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"--- {name}: {'PASS' if rc == 0 else f'FAIL (exit {rc})'}")
        if rc != 0:
            failed.append(name)
    print("QUALITY GATE:", "PASS" if not failed else f"FAIL ({', '.join(failed)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
