"""Face preparation for the local app: detection, explicit status, alignment or a labelled fallback.

The status of every image is one of:

* ``aligned``              exactly one face, detector score and 5-point residual acceptable, aligned
                           to the ArcFace 112x112 template (the offline LFW path);
* ``selected_face``        several faces; the user explicitly selected one by index;
* ``multiple_faces``       several faces and no selection. **REFUSED**: the app never picks a person
                           on its own (the decision becomes ABSTAIN);
* ``fallback_fixed_crop``  no face, or one low-quality face, AND the user declared LFW-style framing
                           (loosely cropped, centred, single-face portrait) AND the image is
                           near-square: the registered Phase 4F policy. Detector-free fixed crop
                           after resizing to the LFW reference size. **Not** a successful detection;
                           always labelled a fallback;
* ``no_reliable_face``     no face, or one low-quality face, and the fallback is not applicable (no
                           framing declaration, non-square image, or a face selected in a
                           multi-face image). **REFUSED**: a fixed crop of an arbitrary photograph
                           is not a face (Phase 6, D-054; the offline evaluation always applied the
                           fallback to LFW images, which is ``allow_fallback=True``);
* ``pre_aligned``          the user declares the input an already aligned 112x112 crop (DigiFace
                           format); no detection is run;
* ``rejected``             invalid input (see :mod:`veilx.app.inputs`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.app.config import AppConfig
from veilx.detect.align import align_to_template, alignment_residual, warp

U8 = NDArray[np.uint8]


@dataclass
class FaceBox:
    box: tuple[float, float, float, float]
    score: float
    residual: float


@dataclass
class PreparedFace:
    status: str
    crop: U8 | None  # 112x112 RGB, or None when the image cannot be used
    faces: list[FaceBox] = field(default_factory=list)
    chosen: int | None = None
    is_fallback: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.crop is not None


class FacePreparer:
    """Lazy SCRFD detector (CPU onnxruntime, as in the offline pipeline) plus frozen geometry."""

    def __init__(self, cfg: AppConfig, data_root: Path) -> None:
        self.cfg = cfg
        self.model_path = data_root / cfg.detector.model
        self._det: Any = None
        self.m_fixed = np.asarray(cfg.fallback.matrix, dtype=np.float64)

    def detector(self) -> Any:
        if self._det is None:
            from veilx.data.archive import sha256_file
            from veilx.detect.scrfd import SCRFDDetector

            if sha256_file(self.model_path) != self.cfg.detector.sha256:
                raise RuntimeError("detector weights differ from the frozen SHA-256")
            self._det = SCRFDDetector(
                self.model_path,
                det_thresh=self.cfg.detector.det_thresh,
                nms_thresh=self.cfg.detector.nms_thresh,
            )
        return self._det

    def fixed_crop(self, rgb: U8) -> U8:
        """Registered fallback. LFW-size inputs (250x250) are warped exactly as offline; other sizes
        are first resized to the reference size (box filter), an assumption of LFW-like framing."""
        from PIL import Image

        ref = self.cfg.fallback.reference_size
        img = rgb
        if rgb.shape[:2] != (ref, ref):
            img = np.asarray(
                Image.fromarray(rgb).resize((ref, ref), Image.Resampling.BOX), np.uint8
            )
        return warp(np.ascontiguousarray(img), self.m_fixed)

    def fallback_applicable(self, rgb: U8) -> bool:
        """Geometric precondition of the fixed crop: a near-square image (LFW is 250x250)."""
        h, w = rgb.shape[:2]
        lo, hi = self.cfg.fallback.aspect_ratio_range
        return bool(lo <= w / h <= hi)

    def _refuse_or_fallback(
        self, rgb: U8, faces: list[FaceBox], chosen: int | None, why: str, allow: bool
    ) -> PreparedFace:
        multi = len(faces) > 1
        if allow and not multi and self.fallback_applicable(rgb):
            return PreparedFace(
                "fallback_fixed_crop",
                self.fixed_crop(rgb),
                faces,
                chosen=chosen,
                is_fallback=True,
                notes=[f"{why}; registered fixed-crop fallback used (NOT a detection)"],
            )
        if multi:
            cause = "the fallback never applies to a face selected in a multi-face image"
        elif not allow:
            cause = "fixed-crop fallback not enabled (LFW-style framing not declared)"
        else:
            cause = "fixed-crop fallback not applicable: image is not near-square"
        return PreparedFace(
            "no_reliable_face", None, faces, chosen=chosen, notes=[f"{why}; {cause}"]
        )

    def prepare(
        self,
        rgb: U8,
        *,
        pre_aligned: bool = False,
        select: int | None = None,
        allow_fallback: bool = False,
    ) -> PreparedFace:
        if pre_aligned:
            if rgb.shape[:2] != (112, 112):
                return PreparedFace(
                    "rejected", None, notes=["pre-aligned mode requires a 112x112 image"]
                )
            return PreparedFace(
                "pre_aligned", rgb, notes=["declared pre-aligned by the user; detection skipped"]
            )
        dets = self.detector().detect(rgb)
        faces = [
            FaceBox(
                (float(d.box[0]), float(d.box[1]), float(d.box[2]), float(d.box[3])),
                float(d.score),
                alignment_residual(d.landmarks.astype(np.float64)),
            )
            for d in dets
        ]
        if len(dets) > 1 and select is None:
            return PreparedFace(
                "multiple_faces",
                None,
                faces,
                notes=[f"{len(dets)} faces detected; select one explicitly (no automatic choice)"],
            )
        if dets:
            k = 0 if select is None else int(select)
            if not 0 <= k < len(dets):
                return PreparedFace(
                    "rejected", None, faces, notes=[f"selected face {k} does not exist"]
                )
            f = faces[k]
            if (
                f.score >= self.cfg.detector.min_score
                and f.residual <= self.cfg.detector.max_residual
            ):
                crop, _ = align_to_template(rgb, dets[k].landmarks.astype(np.float64))
                status = "selected_face" if len(dets) > 1 else "aligned"
                return PreparedFace(status, crop, faces, chosen=k)
            reason = (
                f"detector score {f.score:.2f} < {self.cfg.detector.min_score}"
                if f.score < self.cfg.detector.min_score
                else f"alignment residual {f.residual:.2f} > {self.cfg.detector.max_residual:.2f}"
            )
            return self._refuse_or_fallback(
                rgb, faces, k, f"low-quality detection ({reason})", allow_fallback
            )
        return self._refuse_or_fallback(rgb, faces, None, "no face detected", allow_fallback)
