"""Mixed-precision and determinism helpers (PyTorch imported lazily; optional dependency)."""

from __future__ import annotations

import contextlib
import importlib.util
import os
from typing import Any

from veilx.training.config import PrecisionPolicy


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def describe_accelerator() -> dict[str, Any]:
    """Facts about the local accelerator (no model involved)."""
    if not torch_available():
        return {"torch": None, "cuda_available": False}
    import torch

    info: dict[str, Any] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            device=p.name,
            total_memory_gib=round(p.total_memory / 2**30, 2),
            capability=f"{p.major}.{p.minor}",
            bf16_supported=torch.cuda.is_bf16_supported(),
            cuda_runtime=torch.version.cuda,
        )
    return info


def configure_determinism(policy: PrecisionPolicy) -> None:
    import torch

    if policy.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = policy.allow_tf32
    torch.backends.cudnn.allow_tf32 = policy.allow_tf32


def autocast(policy: PrecisionPolicy, device: str) -> contextlib.AbstractContextManager[Any]:
    """``torch.autocast`` for fp16/bf16, a no-op context for fp32."""
    if policy.mode == "fp32":
        return contextlib.nullcontext()
    import torch

    dtype = torch.float16 if policy.mode == "fp16" else torch.bfloat16
    ctx: contextlib.AbstractContextManager[Any] = torch.autocast(device_type=device, dtype=dtype)
    return ctx


def grad_scaler(policy: PrecisionPolicy, device: str) -> Any:
    """GradScaler for fp16 (loss scaling); disabled (pass-through) otherwise."""
    import torch

    return torch.amp.GradScaler(device, enabled=policy.mode == "fp16")
