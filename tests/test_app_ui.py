"""Gradio interface construction (needs gradio; skipped in the CPU environment)."""

from __future__ import annotations

from pathlib import Path

import pytest

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
