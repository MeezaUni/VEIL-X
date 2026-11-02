"""Gradio interface construction (needs gradio; skipped in the CPU environment)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from veilx.app.config import load_app_config

from .conftest import ROOT

gr = pytest.importorskip("gradio")


def test_inputs_pass_original_uploads_and_cache_expires_quickly(tmp_path: Path) -> None:
    """Uploads must reach the app untouched (so the app deletes them itself), and anything Gradio
    keeps after a failed event must expire within minutes (Phase 6 privacy check)."""
    from veilx.app.pipeline import VerificationPipeline
    from veilx.app.ui import build_app

    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    demo = build_app(VerificationPipeline(cfg, tmp_path, device="cpu"), tmp_path, None)
    images = [
        b for b in demo.blocks.values() if isinstance(b, gr.Image) and b.interactive is not False
    ]
    uploads = [b for b in images if b.label in ("Image A", "Image B")]
    assert len(uploads) == 2
    assert all(b.type == "filepath" and b.image_mode is None for b in uploads)
    assert demo.delete_cache is not None and demo.delete_cache[1] <= 180
    checkboxes = [b.label for b in demo.blocks.values() if isinstance(b, gr.Checkbox)]
    assert any(lbl.startswith("I am authorised") for lbl in checkboxes)
    assert any(lbl.startswith("I understand that the active configuration") for lbl in checkboxes)
    assert any("LFW-style framing" in lbl for lbl in checkboxes)


def test_uploads_reach_the_app_validator_and_examples_live_in_memory(tmp_path: Path) -> None:
    """A corrupt upload must reach the app's own validator (clear message, file deleted) instead
    of failing inside Gradio's PIL decode; examples must not depend on files that Gradio's cache
    cleanup may delete; building the app must not write anything into the source tree."""
    from gradio.data_classes import FileData

    from veilx.app.pipeline import VerificationPipeline
    from veilx.app.ui import build_app

    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    ex = tmp_path / "ex.png"
    Image.new("RGB", (112, 112), (120, 90, 60)).save(ex)
    demo = build_app(
        VerificationPipeline(cfg, tmp_path, device="cpu"), tmp_path, [(str(ex), str(ex))]
    )
    ex.unlink()  # e.g. removed by delete_cache: the example buttons must still work
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    uploads = [
        b
        for b in demo.blocks.values()
        if isinstance(b, gr.Image) and b.label in ("Image A", "Image B")
    ]
    assert all(b.preprocess(FileData(path=str(bad))) == str(bad) for b in uploads)
    assert all(b.preprocess(None) is None for b in uploads)
    buttons = [b.value for b in demo.blocks.values() if isinstance(b, gr.Button)]
    assert any(str(v).startswith("Different people") for v in buttons)
    assert any(str(v).startswith("Not a face") for v in buttons)
    compare = [
        b for b in demo.blocks.values() if isinstance(b, gr.Button) and b.value == "Compare faces"
    ]
    assert (
        len(compare) == 1 and compare[0].interactive is False
    )  # enabled only after both confirmations
    assert not list((ROOT / "src").rglob("*.pyi"))
