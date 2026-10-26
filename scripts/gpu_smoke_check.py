"""GPU infrastructure smoke check (Phase 1b). NO MODEL, NO TRAINING, NO DATA.

Verifies that the prepared Phase 2 infrastructure works on this machine:
accelerator facts, deterministic settings, fp16/bf16 autocast on a random matrix product,
GradScaler construction, and a checkpoint round-trip of a random tensor dict through
``CheckpointManager`` (into a temporary directory).

    .venv-gpu\\Scripts\\python.exe scripts\\gpu_smoke_check.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

from veilx.training.checkpoint import CheckpointManager
from veilx.training.config import PrecisionPolicy
from veilx.training.precision import (
    autocast,
    configure_determinism,
    describe_accelerator,
    grad_scaler,
    torch_available,
)


def main() -> int:
    report: dict[str, object] = {
        "note": "infrastructure check only; no model, no training, no data"
    }
    report["accelerator"] = describe_accelerator()
    if not torch_available():
        print(json.dumps(report, indent=2))
        return 1
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configure_determinism(PrecisionPolicy(mode="fp32", deterministic=True))
    gen = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(512, 512, device=device, generator=gen)
    b = torch.randn(512, 512, device=device, generator=gen)
    ref = (a @ b).float()
    checks: dict[str, object] = {}
    modes = ["fp16", "bf16"] if device == "cuda" else ["bf16"]
    for mode in modes:
        with autocast(PrecisionPolicy(mode=mode), device):  # type: ignore[arg-type]
            out = a @ b
        rel = float((out.float() - ref).norm() / ref.norm())
        checks[f"autocast_{mode}"] = {
            "dtype": str(out.dtype),
            "relative_error_vs_fp32": round(rel, 5),
        }
    scaler = grad_scaler(PrecisionPolicy(mode="fp16"), device)
    checks["grad_scaler_enabled"] = bool(scaler.is_enabled())
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(Path(tmp) / "ck", keep_last=2)
        state = {"w": torch.randn(4, 4, generator=torch.Generator().manual_seed(1)), "step": 3}
        buf = io.BytesIO()
        torch.save(state, buf)
        info = mgr.save(3, buf.getvalue(), {"purpose": "smoke"})
        loaded = torch.load(io.BytesIO(mgr.load(info)), weights_only=True)
        checks["checkpoint_roundtrip_equal"] = bool(torch.equal(loaded["w"], state["w"]))
    if device == "cuda":
        checks["peak_memory_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    report["checks"] = checks
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
