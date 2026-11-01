"""Phase 4: calibration-leakage guards, memory-bounded streaming, checkpoint reproducibility."""

from __future__ import annotations

import importlib.util
from concurrent.futures import Future
from typing import Any

import numpy as np
import pytest

from veilx.errors import ProtocolViolationError
from veilx.guards import require_disjoint_identities, require_eval_only_unused

from .conftest import ROOT


def test_disjoint_identity_guard() -> None:
    require_disjoint_identities({"train": {"a", "b"}, "cal": {"c"}, "test": {"d"}})
    with pytest.raises(ProtocolViolationError, match="cal and test"):
        require_disjoint_identities({"train": {"a"}, "cal": {"c", "x"}, "test": {"x"}})


def test_eval_only_family_guard() -> None:
    require_eval_only_unused(("F1_mask", "F2_sunglasses"))
    with pytest.raises(ProtocolViolationError):
        require_eval_only_unused(("F1_mask", "E1_hair_strands"))


class _CountingPool:
    """Fake executor recording the maximum number of unresolved futures."""

    def __init__(self) -> None:
        self.pending: list[Future[Any]] = []
        self.max_pending = 0

    def submit(self, fn: Any, arg: Any) -> Future[Any]:
        fut: Future[Any] = Future()
        fut.set_result({"n": len(arg)})
        self.pending.append(fut)
        self.max_pending = max(
            self.max_pending, len([f for f in self.pending if not getattr(f, "_taken", False)])
        )
        return fut


def test_extraction_stream_bounds_in_flight_batches() -> None:
    """Memory-safe inference: at most ``inflight`` rendered batches exist at any time."""
    spec = importlib.util.spec_from_file_location(
        "extract_phase4", ROOT / "scripts" / "extract_phase4.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pool = _CountingPool()
    items = [(str(i), str(i), None, None, 0) for i in range(1000)]
    seen = 0
    for out in mod._stream(pool, items, 64, 3):
        for f in pool.pending:
            if f.done() and not getattr(f, "_taken", False):
                f._taken = True  # type: ignore[attr-defined]
                break
        seen += out["n"]
    assert seen == 1000
    assert pool.max_pending <= 3


def test_visnet_checkpoint_roundtrip_is_bit_identical(tmp_path: Any) -> None:
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from veilx.vis.model import VisNet, to_input

    torch.manual_seed(0)
    a = VisNet().eval()
    torch.manual_seed(0)
    b = VisNet().eval()
    for (ka, va), (_, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(va, vb), ka  # seeded initialisation is deterministic
    path = tmp_path / "w.safetensors"
    path.write_bytes(st.save({k: v.contiguous() for k, v in a.state_dict().items()}))
    c = VisNet().eval()
    c.load_state_dict(st.load_file(str(path)))
    x = torch.from_numpy(
        to_input(np.random.default_rng(0).integers(0, 256, (2, 112, 112, 3), dtype=np.uint8))
    )
    with torch.inference_mode():
        la, fa = a.forward_with_features(x)
        lc, fc = c.forward_with_features(x)
    assert torch.equal(la, lc) and torch.equal(fa, fc)
    assert fa.shape == (2, 256) and torch.equal(a(x), la)
