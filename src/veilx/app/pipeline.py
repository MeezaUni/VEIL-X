"""End-to-end frozen verification pipeline of the local research app (no UI code here).

Stages (each timed):
1. validate - input checks (:mod:`veilx.app.inputs`);
2. detect / align - status per image (:mod:`veilx.app.face`);
3. embed - frozen AdaFace IR-101 (decision) and ArcFace IR-101 (diagnostic only);
4. decide - frozen A3 rule on (AdaFace cosine, min feature norm) (:mod:`veilx.app.decision`);
5. diagnostics - VE-C visibility maps and Mahalanobis novelty (:mod:`veilx.app.diagnostics`).
   These are computed AFTER the decision and never feed it.

Models load lazily on first use. Nothing is written to disk; embeddings live only in the returned
result object, and the UI never persists them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.app.config import AppConfig, FrozenRule, load_frozen_decision
from veilx.app.decision import Decision, decide
from veilx.app.face import FacePreparer, PreparedFace
from veilx.metrics.selective import ABSTAIN

U8 = NDArray[np.uint8]


@dataclass
class VerificationResult:
    decision: Decision
    face_a: PreparedFace
    face_b: PreparedFace
    fold: str
    device: str
    s_adaface: float | None = None
    s_arcface: float | None = None
    norm_a: float | None = None
    norm_b: float | None = None
    quality: float | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


class VerificationPipeline:
    def __init__(self, cfg: AppConfig, data_root: Path, *, device: str | None = None) -> None:
        self.cfg = cfg
        self.data_root = data_root
        self.faces = FacePreparer(cfg, data_root)
        self._device = device
        self._backbones: dict[str, Any] = {}
        self._rules: dict[str, FrozenRule] = {}
        self._diag: Any = None

    # -------------------------------------------------------------- lazy resources
    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def rule(self, fold: str) -> FrozenRule:
        if fold not in self.cfg.decision.folds:
            raise ValueError(f"unknown frozen fold {fold}")
        if fold not in self._rules:
            self._rules[fold] = load_frozen_decision(
                self.data_root / self.cfg.decision.phase3_frozen_file,
                self.cfg.decision.phase3_frozen_sha256,
                fold,
            )
        return self._rules[fold]

    def backbone(self, name: str) -> Any:
        if name not in self._backbones:
            from veilx.models.extractor import FrozenExtractor
            from veilx.models.spec import BACKBONES

            spec = BACKBONES[name]
            self._backbones[name] = FrozenExtractor(
                spec,
                self.data_root / "models" / spec.hf_repo.replace("/", "__") / spec.revision,
                device=self.device,
                precision="fp16" if self.device == "cuda" else "fp32",
            )
        return self._backbones[name]

    def diagnostics_engine(self) -> Any:
        if self._diag is None:
            from veilx.app.diagnostics import DiagnosticsEngine

            self._diag = DiagnosticsEngine(self.cfg, self.data_root, self.device)
        return self._diag

    def warm_up(self) -> None:
        self.faces.detector()
        for name in ("adaface_ir101_webface4m", "arcface_ir101_webface4m"):
            self.backbone(name)
        self.rule(self.cfg.decision.default_fold)

    # -------------------------------------------------------------- verification
    def verify(
        self,
        img_a: U8,
        img_b: U8,
        *,
        fold: str | None = None,
        pre_aligned: bool = False,
        select_a: int | None = None,
        select_b: int | None = None,
        with_diagnostics: bool = True,
    ) -> VerificationResult:
        fold = fold or self.cfg.decision.default_fold
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        fa = self.faces.prepare(img_a, pre_aligned=pre_aligned, select=select_a)
        fb = self.faces.prepare(img_b, pre_aligned=pre_aligned, select=select_b)
        t["detect_align"] = 1000 * (time.perf_counter() - t0)
        rule = self.rule(fold)
        unusable = [
            f"image {n}: {f.status.replace('_', ' ')} ({'; '.join(f.notes)})"
            for n, f in (("A", fa), ("B", fb))
            if not f.usable
        ]
        if unusable:
            dec = Decision("ABSTAIN", ABSTAIN, float("nan"), "n/a", None, None, tuple(unusable))
            return VerificationResult(dec, fa, fb, fold, self.device, timings_ms=t)
        t1 = time.perf_counter()
        crops = np.stack([fa.crop, fb.crop])  # type: ignore[list-item]
        ada = self.backbone("adaface_ir101_webface4m").embed(crops)
        t["embed_adaface"] = 1000 * (time.perf_counter() - t1)
        t2 = time.perf_counter()
        arc = self.backbone("arcface_ir101_webface4m").embed(crops)
        t["embed_arcface"] = 1000 * (time.perf_counter() - t2)
        s_ada = float(ada.embeddings[0].astype(np.float64) @ ada.embeddings[1].astype(np.float64))
        s_arc = float(arc.embeddings[0].astype(np.float64) @ arc.embeddings[1].astype(np.float64))
        na, nb = (float(v) for v in ada.feature_norms)
        q = min(na, nb)
        t3 = time.perf_counter()
        dec = decide(rule, s_ada, q)
        t["decide"] = 1000 * (time.perf_counter() - t3)
        res = VerificationResult(dec, fa, fb, fold, self.device, s_ada, s_arc, na, nb, q, t)
        if fa.is_fallback or fb.is_fallback:
            res.notes.append(
                "at least one image used the registered fixed-crop FALLBACK (not a detected face)"
            )
        if with_diagnostics:  # strictly after the decision; cannot change it
            t4 = time.perf_counter()
            res.diagnostics = self.diagnostics_engine().run(crops, fold)
            t["diagnostics"] = 1000 * (time.perf_counter() - t4)
        t["total"] = 1000 * (time.perf_counter() - t0)
        return res
