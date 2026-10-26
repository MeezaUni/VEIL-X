"""Deterministic seeding.

* :func:`set_global_seed` seeds ``random`` and NumPy (and PyTorch if it is installed).
* :func:`derive_seed` produces stable sub-seeds from a base seed and string keys, so that
  e.g. the occlusion seed for (image, family) does not depend on iteration order.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random

import numpy as np

_MAX_SEED = 2**32 - 1


def derive_seed(base_seed: int, *keys: str | int) -> int:
    """Return a 32-bit seed deterministically derived from ``base_seed`` and ``keys``."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    material = ":".join([str(base_seed), *map(str, keys)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> dict[str, bool]:
    """Seed all available RNGs. Returns which libraries were seeded.

    ``PYTHONHASHSEED`` only affects interpreters started *after* it is set; it is exported
    here so that subprocesses inherit it, and VEIL-X code never relies on ``hash()`` ordering.
    """
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG, for third-party code
    seeded = {"python": True, "numpy": True, "torch": False}
    if importlib.util.find_spec("torch") is not None:  # pragma: no cover - torch absent in Phase 0
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        seeded["torch"] = True
    return seeded


def make_rng(base_seed: int, *keys: str | int) -> np.random.Generator:
    """Return an independent NumPy Generator for a named purpose."""
    return np.random.default_rng(derive_seed(base_seed, *keys))
