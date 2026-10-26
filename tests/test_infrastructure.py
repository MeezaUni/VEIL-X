"""Seeding, logging, provenance, guards and interface contracts."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pytest

from veilx.config.schema import SplitName
from veilx.errors import ProtocolViolationError, VeilxError
from veilx.guards import require_final_for_test, require_heldout_isolation
from veilx.interfaces import (
    DatasetAdapter,
    Decision,
    EmbeddingBatch,
    EmbeddingExtractor,
    RiskCalibrator,
)
from veilx.logging_utils import configure_logging, get_logger
from veilx.provenance import collect_provenance
from veilx.seeding import derive_seed, make_rng, set_global_seed

from .conftest import PROTOCOL, ROOT

pytestmark = pytest.mark.structural


def test_global_seed_reproducible() -> None:
    set_global_seed(123)
    a = (random.random(), np.random.rand())
    set_global_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_derived_seeds_stable_and_distinct() -> None:
    assert derive_seed(1, "occlusion", "img7") == derive_seed(1, "occlusion", "img7")
    assert derive_seed(1, "occlusion", "img7") != derive_seed(1, "occlusion", "img8")
    assert make_rng(5, "x").integers(0, 10**9) == make_rng(5, "x").integers(0, 10**9)
    with pytest.raises(ValueError):
        derive_seed(-1)
    with pytest.raises(ValueError):
        set_global_seed(2**33)


def test_json_logging() -> None:
    buf = io.StringIO()
    configure_logging("INFO", stream=buf)
    get_logger("test").info("hello", extra={"fields": {"n": 3}})
    record = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert record["msg"] == "hello"
    assert record["fields"] == {"n": 3}
    assert record["logger"] == "veilx.test"


def test_provenance_has_no_private_host_fields(tmp_path: Path) -> None:
    prov = collect_provenance(ROOT, configs=[PROTOCOL], seed=1, argv=[str(tmp_path / "x.py"), "-v"])
    assert "configs/research_protocol.yaml" in prov["configs"]
    flat = json.dumps(prov)
    assert "hostname" not in flat and "node" not in prov
    assert str(tmp_path) not in flat  # absolute paths are reduced to names
    assert prov["argv"] == ["x.py", "-v"]


def test_test_access_guard() -> None:
    with pytest.raises(ProtocolViolationError):
        require_final_for_test(SplitName.TEST, final=False)
    require_final_for_test(SplitName.TEST, final=True)
    require_final_for_test(SplitName.VAL, final=False)


def test_heldout_guard() -> None:
    with pytest.raises(ProtocolViolationError):
        require_heldout_isolation("F1_mask", ("F1_mask", "F2_sunglasses"))
    require_heldout_isolation("F1_mask", ("F2_sunglasses",))


def test_errors_share_base() -> None:
    assert issubclass(ProtocolViolationError, VeilxError)


def test_interfaces_are_runtime_checkable() -> None:
    class DummyExtractor:
        backbone_id = "dummy"
        embedding_dim = 4

        def embed(self, crops: np.ndarray) -> EmbeddingBatch:
            n = crops.shape[0]
            e = np.ones((n, 4)) / 2.0
            return EmbeddingBatch("dummy", e, np.ones(n))

    assert isinstance(DummyExtractor(), EmbeddingExtractor)
    assert not isinstance(DummyExtractor(), DatasetAdapter)
    assert not isinstance(object(), RiskCalibrator)
    assert {d.value for d in Decision} == {"match", "non_match", "abstain"}
