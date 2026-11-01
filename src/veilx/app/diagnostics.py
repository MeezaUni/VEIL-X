"""Explanatory diagnostics (NOT used for any decision): VE-C visibility maps and novelty flags.

These come from the frozen Phase 4 conservative visibility estimator of the same fold as the
decision rule (temperature, lower-bound shift and Mahalanobis threshold from the Phase 4 freeze).
Phase 4 (D-050) found that learned visibility, even ground-truth visibility, did not improve on
quality-stratified decisions, so these signals are shown to the user as explanations only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.app.config import AppConfig, content_sha256
from veilx.vis.conservative import MahalanobisNovelty, conservative_map, global_visible
from veilx.vis.regions import region_visibility


class DiagnosticsEngine:
    def __init__(self, cfg: AppConfig, data_root: Path, device: str) -> None:
        frozen = json.loads((data_root / cfg.diagnostics.phase4_frozen_file).read_text("utf-8"))
        if content_sha256(frozen) != cfg.diagnostics.phase4_frozen_sha256:
            raise RuntimeError("Phase 4 frozen configuration does not match the registered SHA-256")
        self.frozen = frozen
        self.vec_dir = data_root / cfg.diagnostics.vec_dir
        self.device = device
        self._models: dict[str, Any] = {}

    def _model(self, fold: str) -> Any:
        if fold not in self._models:
            from safetensors.torch import load_file

            from veilx.vis.model import VisNet

            m = VisNet().to(self.device).eval()
            m.load_state_dict(
                load_file(str(self.vec_dir / fold / "best.safetensors"), device=self.device)
            )
            self._models[fold] = m
        return self._models[fold]

    def run(self, crops: NDArray[np.uint8], fold: str) -> dict[str, Any]:
        import torch

        from veilx.vis.model import to_input

        f = self.frozen["folds"][fold]
        nov = MahalanobisNovelty.from_dict(f["novelty"])
        m = self._model(fold)
        with torch.inference_mode():
            x = torch.from_numpy(to_input(crops)).to(self.device)
            lg, ft = m.forward_with_features(x)
        logits = lg.float().cpu().numpy().astype(np.float64)
        maps = conservative_map(logits, f["temperature"], f["shift"])
        scores = nov.score(ft.float().cpu().numpy().astype(np.float64))
        thr = float(f["novelty"]["threshold"])
        regions = region_visibility(maps)
        return {
            "visibility_maps": maps,  # (2, 28, 28) predicted P(visible), conservative
            "visible_fraction": global_visible(maps).tolist(),
            "regions": {k: v.tolist() for k, v in regions.items()},
            "novelty_scores": scores.tolist(),
            "novelty_threshold": thr,
            "novel": [bool(s > thr) for s in scores],
            "pair_novel": bool(max(scores) > thr),
            "note": "explanatory only; not used in the decision",
        }
