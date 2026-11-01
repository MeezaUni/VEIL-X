"""Phase 5 app with the real frozen models (local only; skipped without torch or private data)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from .conftest import ROOT

torch = pytest.importorskip("torch")
DATA = Path(os.environ.get("VEILX_DATA_ROOT", "C:/Users/MAK/veilx_data"))
pytestmark = pytest.mark.skipif(
    not (DATA / "reports" / "phase3_frozen.json").exists(), reason="private data root not available"
)


@pytest.fixture(scope="module")
def pipe():  # type: ignore[no-untyped-def]
    from veilx.app.config import load_app_config
    from veilx.app.pipeline import VerificationPipeline

    return VerificationPipeline(load_app_config(ROOT / "configs/app/veilx_app.yaml"), DATA)


def _digiface_pair() -> tuple[np.ndarray, np.ndarray, str]:
    from veilx.data.images import load_rgb
    from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl

    p = next(iter(read_jsonl(DATA / "manifests/phase3/pairs/p3_val_pairs.jsonl", PairRecord)))
    rel = {
        r.image_id: r.relpath
        for r in read_jsonl(DATA / "manifests/digiface_final.jsonl", ImageRecord)
        if r.image_id in (p.image_a, p.image_b)
    }
    return (
        load_rgb(DATA / "extracted" / rel[p.image_a]),
        load_rgb(DATA / "extracted" / rel[p.image_b]),
        p.label,
    )


def test_frozen_models_load_with_verified_hashes(pipe) -> None:  # type: ignore[no-untyped-def]
    for fold in pipe.cfg.decision.folds:
        r = pipe.rule(fold)
        assert set(r.rules) == {"q_low", "q_mid", "q_high"}
    pipe.faces.detector()  # SHA-256 of the detector weights is checked on load
    assert pipe.backbone("adaface_ir101_webface4m").embedding_dim == 512


def test_real_pair_runs_and_diagnostics_are_isolated(pipe) -> None:  # type: ignore[no-untyped-def]
    a, b, _ = _digiface_pair()
    r1 = pipe.verify(a, b, pre_aligned=True, with_diagnostics=True)
    r2 = pipe.verify(a, b, pre_aligned=True, with_diagnostics=False)
    assert r1.decision.label in ("MATCH", "NON-MATCH", "ABSTAIN")
    assert (
        r1.decision.code == r2.decision.code
        and abs(r1.decision.fused_score - r2.decision.fused_score) < 1e-9
    )
    d = r1.diagnostics
    assert (
        d is not None
        and d["visibility_maps"].shape == (2, 28, 28)
        and "not used in the decision" in d["note"]
    )


def test_cpu_gpu_consistency(pipe) -> None:  # type: ignore[no-untyped-def]
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from veilx.app.pipeline import VerificationPipeline

    a, b, _ = _digiface_pair()
    cpu = VerificationPipeline(pipe.cfg, DATA, device="cpu")
    g = pipe.verify(a, b, pre_aligned=True, with_diagnostics=False)
    c = cpu.verify(a, b, pre_aligned=True, with_diagnostics=False)
    assert abs(g.s_adaface - c.s_adaface) < 5e-3 and abs(g.quality - c.quality) < 0.1
    assert g.decision.code == c.decision.code
