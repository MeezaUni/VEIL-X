# ruff: noqa: E501, RUF001  (UI layout and user-facing text)
"""Local Gradio interface of the VEIL-X research demonstration (built by :func:`build_app`).

Privacy behaviour:
- uploads arrive as Gradio temp files; their bytes are read and validated in memory, and the temp
  file is deleted immediately (best effort, within the app's private temp directory only). The
  bytes are kept only in the browser session's server-side state, so the same pair can be compared
  again (e.g. with other options) without uploading it again; *Clear* forgets them;
- nothing is logged, embeddings are never written, and there is no flagging, no public link and no
  analytics (see ``scripts/run_app.py``).
"""

from __future__ import annotations

import io
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from veilx.app.config import AppConfig
from veilx.app.decision import rule_description
from veilx.app.explain import (
    build_explanation,
    load_reference,
    metrics_html,
    score_bar_html,
    steps_html,
    verdict_html,
)
from veilx.app.inputs import InputRejectedError, ValidatedImage, load_image_bytes
from veilx.app.pipeline import VerificationPipeline, VerificationResult
from veilx.app.report import (
    DISCLAIMER,
    FAMILY_NAMES,
    LIMITATIONS_MD,
    WHY_NOT_VISIBILITY_MD,
    abstention_text,
    annotate_faces,
    decision_html,
    fold_card_md,
    rule_bar,
    visibility_overlay,
)

REFERENCE_PATH = Path(__file__).resolve().parents[3] / "configs" / "app" / "explain_reference.json"

STATUS_TEXT = {
    "aligned": "one face detected and aligned (detector score and alignment residual acceptable)",
    "selected_face": "several faces detected; the face you selected was aligned",
    "multiple_faces": "several faces detected and none selected: REFUSED (the app does not choose a person)",
    "fallback_fixed_crop": "FALLBACK: no reliable detection; detector-free fixed crop used because you declared LFW-style framing (not a detected face)",
    "no_reliable_face": "no reliable face (none detected, or detection quality too low) and the fixed-crop fallback is not applicable: REFUSED",
    "pre_aligned": "declared pre-aligned 112×112 crop; detection skipped",
    "rejected": "rejected",
}


CONSENT_REQUIRED = (
    "<b>Please confirm both statements</b> (authorisation to process the images, and the calibration "
    "limitations of the active experimental configuration). The uploaded files were discarded; "
    "upload them again."
)

# What the default example pairs show (scripts/run_app.make_examples: the first p3_val pairs of
# synthetic DigiFace-1M identities, the last one with a synthetic mask on image B).
EXAMPLE_INFO = [
    ("Different people", "two synthetic people, clear images"),
    ("Same person", "one synthetic person, two images"),
    ("Different people (2)", "another pair of synthetic people"),
    ("Same person, masked", "image B wears a synthetic mask"),
]
NOISE_INFO = ("Not a face (noise)", "random pixels: tests the safety guard")


def safe_error_html(exc: BaseException) -> str:
    """User-facing message for an unexpected failure: no stack trace, no file paths."""
    name = type(exc).__name__
    if "OutOfMemory" in name or isinstance(exc, MemoryError):
        return (
            "<b>No decision: resources exhausted</b> (out of memory). Close other GPU programs or "
            "restart with <code>--device cpu</code>."
        )
    return f"<b>No decision: internal processing error</b> ({name}). Details are not shown."


def read_and_discard(path: str | None, temp_root: Path | None) -> bytes | None:
    """Read an uploaded temp file and delete it if it lives in the app's temp directory."""
    if not path:
        return None
    p = Path(path)
    data = p.read_bytes()
    if temp_root is not None:
        try:
            if p.resolve().is_relative_to(temp_root.resolve()):
                p.unlink(missing_ok=True)
        except OSError:
            pass
    return data


def disposable_copies(paths: tuple[str, str], temp_root: Path | None) -> tuple[str, str]:
    """Copies of example files for one comparison (the comparison deletes its inputs after reading)."""
    if temp_root is None:
        return paths
    out = temp_root / "uploads"
    out.mkdir(exist_ok=True)
    copies = []
    for src in paths:
        dst = out / f"{uuid.uuid4().hex}{Path(src).suffix}"
        shutil.copyfile(src, dst)
        copies.append(str(dst))
    return copies[0], copies[1]


def model_info_md(cfg: AppConfig) -> str:
    return f"""
### Frozen inference configuration
| Component | Value |
|---|---|
| Detector | SCRFD det_10g (InsightFace buffalo_l v0.7), CPU onnxruntime; score ≥ {cfg.detector.min_score}; 5-point residual ≤ {cfg.detector.max_residual:.2f} px |
| Fallback | Detector-free fixed crop (LFW View-1 DevTrain mean landmarks), after resizing to {cfg.fallback.reference_size}×{cfg.fallback.reference_size}; registered Phase 4F policy |
| Alignment | 5-point similarity to the ArcFace 112×112 template |
| Recognition | Frozen AdaFace IR-101 (WebFace4M) — **decision**; frozen ArcFace IR-101 (WebFace4M) — **diagnostic only** |
| Fused score | M1 logistic(cosine, min feature norm, product) — Phase 3 freeze `{cfg.decision.phase3_frozen_sha256[:12]}` |
| Decision rule | A3 quality-stratified three-way rule (primary target: selective FNMR ≤ 5 %, selective FMR ≤ 5 %, δ = 0.05, v2 ordering), calibrated on synthetic DigiFace `p3_cal` |
| Frozen systems | {", ".join(cfg.decision.folds)} (one per Phase 3 fold; default {cfg.decision.default_fold}, D-051) |
| Diagnostics | Phase 4 VE-C visibility maps + Mahalanobis novelty — freeze `{cfg.diagnostics.phase4_frozen_sha256[:12]}`; **not used in decisions** |
| Precision | GPU: fp16 autocast (as evaluated offline); CPU: fp32 |

Evidence: Phase 3 confirmatory `bb5d9244bb3c`; Phase 4 confirmatory `903f6a5c7de3`; decisions D-046…D-051.
"""


def rules_md(pipe: VerificationPipeline) -> str:
    out = [
        "### Frozen A3 rules per system (quality = smaller pre-normalisation AdaFace feature norm of the pair)"
    ]
    for fold in pipe.cfg.decision.folds:
        try:
            rows = rule_description(pipe.rule(fold))
        except Exception as exc:
            out.append(f"- {fold}: unavailable ({type(exc).__name__})")
            continue
        out.append(f"\n**{fold}**\n\n| stratum | quality range | rule |\n|---|---|---|")
        out += [f"| {r['stratum']} | {r['quality']} | {r['rule']} |" for r in rows]
    out.append(
        "\nIn the medium and high quality strata the frozen thresholds coincide, so the rule never abstains there; abstention is only possible for low-quality pairs."
    )
    return "\n".join(out)


# ------------------------------------------------------------------------- shared result parts
def status_md(res: VerificationResult, a: ValidatedImage, b: ValidatedImage) -> str:
    status = []
    for name, f, img in (("A", res.face_a, a), ("B", res.face_b, b)):
        det = (
            "; ".join(
                f"face {i}: score {x.score:.2f}, residual {x.residual:.2f}"
                for i, x in enumerate(f.faces)
            )
            or "no detections"
        )
        status.append(
            f"**Image {name}** — {STATUS_TEXT.get(f.status, f.status)}. Detections: {det}."
            + (f" Notes: {'; '.join(f.notes + img.notes)}." if f.notes or img.notes else "")
        )
    if res.notes:
        status.append("**Note:** " + " ".join(res.notes))
    return "\n\n".join(status)


def quantity_rows(res: VerificationResult) -> list[list[Any]]:
    dec = res.decision
    if res.s_adaface is None:
        return []
    return [
        ["AdaFace cosine similarity", f"{res.s_adaface:.4f}", "decision input"],
        [
            "Feature norm A / B (quality proxy)",
            f"{res.norm_a:.2f} / {res.norm_b:.2f}",
            "decision input (min)",
        ],
        [
            "Pair quality q = min norm",
            f"{res.quality:.2f}",
            f"decision input → stratum {dec.stratum}",
        ],
        [
            "Fused score M1 (logistic; not a probability of identity)",
            f"{dec.fused_score:.4f}",
            "decision score",
        ],
        [
            "Rule thresholds (τ_low, τ_high)",
            f"{dec.tau_low:.3f}, {dec.tau_high:.3f}" if dec.tau_low is not None else "n/a",
            "frozen A3",
        ],
        ["ArcFace cosine similarity", f"{res.s_arcface:.4f}", "diagnostic only"],
        [
            "|AdaFace − ArcFace| disagreement",
            f"{abs(res.s_adaface - (res.s_arcface or 0.0)):.4f}",
            "diagnostic only",
        ],
    ]


def novelty_md(d: dict[str, Any]) -> str:
    def flag(n: bool) -> str:
        return "UNUSUAL (outside the estimator's training distribution)" if n else "typical"

    return (
        "**Explanatory diagnostics — NOT used in the decision.**\n\n"
        f"- Predicted visible face fraction: A {d['visible_fraction'][0]:.2f}, B {d['visible_fraction'][1]:.2f}\n"
        f"- Novelty (Mahalanobis on the visibility estimator's features; threshold {d['novelty_threshold']:.2f}): "
        f"A {d['novelty_scores'][0]:.2f} → {flag(d['novel'][0])}; B {d['novelty_scores'][1]:.2f} → {flag(d['novel'][1])}\n"
        "- Region estimates (predicted visible fraction, A / B): "
        + ", ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in d["regions"].items())
        + "\n\nThe estimator can miss occluders that look like faces and was trained only on synthetic occluders; "
        "a 'typical' novelty value does not mean the image is free of occlusion."
    )


def timing_text(res: VerificationResult) -> str:
    return (
        " · ".join(f"{k} {v:.0f} ms" for k, v in res.timings_ms.items()) + f" · device {res.device}"
    )


def detection_image(img: ValidatedImage, f: Any) -> Any:
    return annotate_faces(img.rgb, f.faces, f.chosen) if f.faces else img.rgb


def verify_bytes(
    pipe: VerificationPipeline,
    raw_a: bytes,
    raw_b: bytes,
    fold: str,
    pre: bool,
    sel_a: float | None,
    sel_b: float | None,
    diag: bool,
    allow_fallback: bool,
) -> tuple[VerificationResult, ValidatedImage, ValidatedImage]:
    """Validate two in-memory images and run the frozen pipeline. Raises InputRejectedError for
    invalid input; any other exception is a processing error (shown via :func:`safe_error_html`)."""
    a = load_image_bytes(raw_a, pipe.cfg.inputs)
    b = load_image_bytes(raw_b, pipe.cfg.inputs)
    res = pipe.verify(
        a.rgb,
        b.rgb,
        fold=fold,
        pre_aligned=pre,
        select_a=None if sel_a is None or sel_a < 0 else int(sel_a),
        select_b=None if sel_b is None or sel_b < 0 else int(sel_b),
        with_diagnostics=diag,
        allow_fallback=allow_fallback,
    )
    return res, a, b


def run_comparison(
    pipe: VerificationPipeline,
    temp_root: Path | None,
    path_a: str | None,
    path_b: str | None,
    fold: str,
    pre: bool,
    sel_a: float,
    sel_b: float,
    diag: bool,
    authorised: bool,
    calibration_ack: bool = False,
    allow_fallback: bool = False,
) -> tuple[Any, ...]:
    """One comparison request from two file paths: read and discard the files, then verify (or
    refuse). Returns the decision HTML, the rule-bar image and the detailed outputs."""
    empty = (None,) * 7
    try:
        raw_a, raw_b = read_and_discard(path_a, temp_root), read_and_discard(path_b, temp_root)
        if not (authorised and calibration_ack):
            return (
                CONSENT_REQUIRED,
                None,
                "No decision: confirmation missing.",
                "",
                None,
                None,
                "",
                *empty,
            )
        if raw_a is None or raw_b is None:
            return ("<b>Please provide two images.</b>", None, "", "", None, None, "", *empty)
        try:
            res, a, b = verify_bytes(
                pipe, raw_a, raw_b, fold, pre, sel_a, sel_b, diag, allow_fallback
            )
        except InputRejectedError:
            raise
        except Exception as exc:  # safe message; no stack trace or paths in the page
            return (
                safe_error_html(exc),
                None,
                "No decision: processing error.",
                "",
                None,
                None,
                "",
                *empty,
            )
    except InputRejectedError as exc:
        return (
            f"<b>Input rejected</b> ({exc.code}): {exc}",
            None,
            "No decision: invalid input.",
            "",
            None,
            None,
            "",
            *empty,
        )
    dec = res.decision
    vis_a = vis_b = None
    nov = "Diagnostics not computed."
    if res.diagnostics:
        d = res.diagnostics
        vis_a = visibility_overlay(res.face_a.crop, d["visibility_maps"][0])  # type: ignore[arg-type]
        vis_b = visibility_overlay(res.face_b.crop, d["visibility_maps"][1])  # type: ignore[arg-type]
        nov = novelty_md(d)
    return (
        decision_html(dec.label, dec.fused_score, dec.stratum),
        rule_bar(dec.fused_score, dec.tau_low, dec.tau_high),
        abstention_text(dec.reasons, dec.label),
        status_md(res, a, b),
        detection_image(a, res.face_a),
        detection_image(b, res.face_b),
        timing_text(res),
        res.face_a.crop,
        res.face_b.crop,
        quantity_rows(res) or None,
        vis_a,
        vis_b,
        nov,
        f"Frozen system: {res.fold}",
    )


# ------------------------------------------------------------------------------------ the page
CSS = """
.gradio-container {max-width: 1280px !important; margin: 0 auto !important;}
.vx-header {display:flex; gap:18px; align-items:center; padding:18px 22px; border-radius:14px;
  background: linear-gradient(120deg, #13294b 0%, #1d4e89 60%, #2f6db3 100%); color:#fff; margin-bottom:6px;}
.vx-header, .vx-header * {color:#fff !important;}
.vx-header b {font-weight:800;}
.vx-logo {font-size:30px; font-weight:800; letter-spacing:1px; padding:8px 14px; border:2px solid rgba(255,255,255,.55); border-radius:10px;}
.vx-title {font-size:21px; font-weight:700; line-height:1.25;}
.vx-sub {font-size:14px; opacity:.92; margin-top:4px;}
.vx-badges {display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;}
.vx-badges span {font-size:12px; padding:3px 10px; border-radius:999px; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3);}
.vx-h {font-size:16px; font-weight:700; margin:2px 0 4px;}
.vx-hint {font-size:13px; color: var(--body-text-color-subdued);}
.vx-upl-ok {font-size:13px; color:#1b7f5a; min-height:18px;}
.vx-upl-err {font-size:13px; color:#b3261e; font-weight:600;}
.vx-ex-btn {min-height:58px !important; white-space:pre-line !important; line-height:1.25 !important; font-size:13px !important;}
.vx-compare {font-size:17px !important; min-height:50px !important;}
.vx-empty {border:2px dashed var(--border-color-primary); border-radius:14px; padding:26px; color: var(--body-text-color-subdued);}
.vx-empty-t {font-size:18px; font-weight:700; color: var(--body-text-color); margin-bottom:8px;}
.vx-empty ol {margin:8px 0 10px 20px;}
.vx-alert {border-radius:12px; padding:14px 16px; border:1px solid; font-size:15px;}
.vx-alert-warn {background:rgba(154,91,0,.10); border-color:rgba(154,91,0,.5);}
.vx-alert-err {background:rgba(179,38,30,.08); border-color:rgba(179,38,30,.5);}
.vx-verdict {border-radius:14px; padding:16px 18px; border-left:8px solid; background: var(--block-background-fill);}
.vx-v-match {border-color:#1d4e89; background:rgba(29,78,137,.09);}
.vx-v-nonmatch {border-color:#4a4a4a; background:rgba(74,74,74,.08);}
.vx-v-abstain {border-color:#9a5b00; background:rgba(154,91,0,.10);}
.vx-v-error {border-color:#b3261e; background:rgba(179,38,30,.08);}
.vx-v-label {font-size:30px; font-weight:800; letter-spacing:.5px;}
.vx-v-match .vx-v-label {color:#1d4e89;} .vx-v-nonmatch .vx-v-label {color:#4a4a4a;}
.vx-v-abstain .vx-v-label {color:#9a5b00;} .vx-v-error .vx-v-label {color:#b3261e;}
.dark .vx-v-match .vx-v-label {color:#8db8ee;} .dark .vx-v-nonmatch .vx-v-label {color:#cfcfcf;}
.dark .vx-v-abstain .vx-v-label {color:#f0b45a;} .dark .vx-v-error .vx-v-label {color:#f08a82;}
.vx-v-head {font-size:18px; font-weight:700; margin-top:2px;}
.vx-v-sum {font-size:15px; margin-top:6px; line-height:1.45;}
.vx-v-meta {font-size:12px; margin-top:10px; color: var(--body-text-color-subdued);}
.vx-cards {display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px;}
@media (max-width: 760px) {.vx-cards {grid-template-columns: repeat(2, minmax(0,1fr));} .vx-header {flex-direction:column; align-items:flex-start;}}
.vx-card {border:1px solid var(--border-color-primary); border-radius:12px; padding:10px 12px; background: var(--block-background-fill);}
.vx-card-t {font-size:12px; text-transform:uppercase; letter-spacing:.6px; color: var(--body-text-color-subdued);}
.vx-card-v {font-size:24px; font-weight:800; margin:2px 0; font-variant-numeric: tabular-nums;}
.vx-card-s {font-size:12px; color: var(--body-text-color-subdued);}
.vx-bar-wrap {padding:26px 6px 4px;}
.vx-bar {position:relative; height:30px; border-radius:8px; overflow:visible; background: var(--border-color-primary);}
.vx-zone {position:absolute; top:0; height:100%; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:700; overflow:hidden; white-space:nowrap;}
.vx-zone:first-child {border-radius:8px 0 0 8px;} .vx-zone:last-child {border-radius:0 8px 8px 0;}
.vx-zone:only-child {border-radius:8px;}
.vx-z-match {background:rgba(29,78,137,.35);} .vx-z-nonmatch {background:rgba(74,74,74,.28);} .vx-z-abstain {background:rgba(154,91,0,.35);}
.vx-marker {position:absolute; top:-8px; bottom:-8px; width:4px; margin-left:-2px; background: var(--body-text-color); border-radius:2px;}
.vx-marker-l {position:absolute; top:-22px; left:50%; transform:translateX(-50%); font-size:12px; font-weight:700; white-space:nowrap;
  background: var(--body-text-color); color: var(--background-fill-primary); padding:1px 6px; border-radius:6px;}
.vx-ticks {position:relative; height:18px; margin-top:6px;}
.vx-tick {position:absolute; transform:translateX(-50%); font-size:11px; color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums;}
.vx-tick:first-child {transform:none;} .vx-tick:last-child {transform:translateX(-100%);}
.vx-bar-note {font-size:12px; color: var(--body-text-color-subdued); margin-top:2px;}
.vx-steps {display:flex; flex-direction:column; gap:10px;}
.vx-step {display:flex; gap:12px; align-items:flex-start; border:1px solid var(--border-color-primary); border-radius:12px; padding:10px 12px; background: var(--block-background-fill);}
.vx-step-icon {flex:0 0 26px; height:26px; border-radius:50%; color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; font-size:14px;}
.vx-step-title {font-weight:700; font-size:15px; margin-bottom:2px;}
.vx-step ul, .vx-change ul {margin:2px 0 0 18px; padding:0;}
.vx-step li, .vx-change li {font-size:14px; line-height:1.45; margin:2px 0;}
.vx-skip {opacity:.6;}
.vx-stop {border-color:rgba(179,38,30,.45);}
.vx-warn {border-color:rgba(154,91,0,.5);}
.vx-change {border-radius:12px; padding:10px 14px; background:rgba(29,78,137,.07); border:1px solid rgba(29,78,137,.3);}
.vx-note {font-size:12px; color: var(--body-text-color-subdued);}
.vx-how {display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px;}
@media (max-width: 900px) {.vx-how {grid-template-columns: 1fr;}}
.vx-how-c {border:1px solid var(--border-color-primary); border-radius:12px; padding:12px 14px; background: var(--block-background-fill);}
.vx-how-n {display:inline-block; width:26px; height:26px; border-radius:50%; background:#1d4e89; color:#fff; text-align:center; line-height:26px; font-weight:800; margin-right:6px;}
.vx-how-c b {font-size:15px;}
.vx-how-c p {font-size:14px; margin:6px 0 0; line-height:1.45;}
footer {display:none !important;}
@media (max-width: 760px) {.tab-container {flex-wrap:wrap !important; min-width:0 !important; height:auto !important;}}
"""

HEADER_HTML = """
<div class='vx-header'>
  <div class='vx-logo'>VEIL-X</div>
  <div>
    <div class='vx-title'>Face verification that explains itself, and says “not sure” when it should</div>
    <div class='vx-sub'>Compare two face images. VEIL-X answers <b>MATCH</b>, <b>NON-MATCH</b> or <b>ABSTAIN</b> and shows exactly why.</div>
    <div class='vx-badges'><span>Runs only on this computer</span><span>Images never stored</span><span>Research demo, not identity verification</span></div>
  </div>
</div>
"""

EMPTY_HTML = """
<div class='vx-empty'>
  <div class='vx-empty-t'>Your result will appear here</div>
  <ol><li>Upload two face photos, or pick an example on the left.</li>
  <li>Tick the two confirmations.</li>
  <li>Press <b>Compare faces</b>.</li></ol>
  <div>Every result comes with a step-by-step explanation: the exact numbers, the rule that applied and what would change the answer.</div>
</div>
"""

HOW_HTML = """
<div class='vx-how'>
 <div class='vx-how-c'><span class='vx-how-n'>1</span><b>Find and align the face</b><p>The SCRFD detector finds each face and five landmarks (eyes, nose tip, mouth corners). The face is rotated and scaled to a standard 112×112 position. If an image holds several faces, <i>you</i> choose one: VEIL-X never picks a person.</p></div>
 <div class='vx-how-c'><span class='vx-how-n'>2</span><b>Describe each face</b><p>AdaFace turns each aligned face into 512 numbers (an <i>embedding</i>). The vector's length before normalisation (its <i>feature norm</i>) measures how clearly the face can be seen: this is the <b>quality</b>.</p></div>
 <div class='vx-how-c'><span class='vx-how-n'>3</span><b>Compare</b><p><b>Similarity</b> is the cosine between the two embeddings (1 = same direction, 0 = unrelated). A pair is only as good as its worse image, so <b>pair quality q</b> is the smaller of the two norms.</p></div>
 <div class='vx-how-c'><span class='vx-how-n'>4</span><b>Score</b><p>A small calibrated formula (logistic regression on similarity, quality and their product) gives a <b>decision score</b> from 0 to 1. It ranks pairs; it is <i>not</i> a probability that it is the same person.</p></div>
 <div class='vx-how-c'><span class='vx-how-n'>5</span><b>Decide with a quality-aware rule</b><p>Pairs fall into a low, medium or high quality group. Medium and high quality use one cut-off: MATCH above it, NON-MATCH below. Low quality has an <b>uncertain zone</b>: inside it VEIL-X answers ABSTAIN instead of guessing.</p></div>
 <div class='vx-how-c'><span class='vx-how-n'>6</span><b>Safety guard and explanations</b><p>If quality is below anything seen during calibration (for example, a non-face image), VEIL-X always abstains. A second model (ArcFace) and a visibility map are shown to help you understand the result, but they <b>never change the decision</b>.</p></div>
</div>
"""

GLOSSARY_MD = """
### What the three answers mean
| Answer | Meaning |
|---|---|
| **MATCH** | The score passed the cut-off for this quality group: the two images likely show the same person. |
| **NON-MATCH** | The score fell below the cut-off: the two images likely show different people. |
| **ABSTAIN** | VEIL-X declines to decide: the images are low quality and the score is in the uncertain zone, the quality is below the calibrated range, or a face could not be used. |

Abstaining is a feature: when the evidence is weak, saying “not sure” is better than a confident mistake.
"""

PRIVACY_MD = """
### What happens to your images
- Images are processed **only on this computer**. The app listens on 127.0.0.1 and makes no network requests.
- An uploaded file is read into memory and its temporary file is **deleted immediately**. The image stays in
  memory only for your browser session, so you can compare again with other options; **Clear** forgets it.
- Face embeddings and results are **never written to disk**, logged or sent anywhere. Result images are
  temporary display files that expire within minutes.
"""


def raw_upload_preprocess(payload: Any) -> str | None:
    """Pass an upload on as its raw temp-file path.

    Gradio's own ``Image.preprocess`` decodes the file with PIL first, so a corrupt or non-image
    upload failed inside Gradio with a generic "Error" badge. With this override (set on the two
    input components), the app's validator (:func:`load_image_bytes`) sees every upload, gives a
    clear message and deletes the file. (A ``gr.Image`` subclass is avoided: Gradio writes a
    ``.pyi`` stub next to the module of every component subclass.)
    """
    return None if payload is None or payload.path is None else str(payload.path)


def build_app(
    pipe: VerificationPipeline,
    temp_root: Path | None = None,
    example_pairs: list[tuple[str, str]] | None = None,
    reference: dict[str, Any] | None = None,
) -> Any:
    import gradio as gr

    cfg = pipe.cfg
    ref = reference if reference is not None else load_reference(REFERENCE_PATH)
    floors = cfg.decision.q_support_floor or {}
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=["system-ui", "Segoe UI", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    )

    # ---- examples, held in memory from startup (title, description, image A, image B). Files in
    # the temp directory may be removed while the app runs, so nothing is read from disk later.
    def _png(rgb: Any) -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    def _example(raw: bytes) -> tuple[bytes, ValidatedImage]:
        return raw, load_image_bytes(raw, cfg.inputs)

    examples: list[tuple[str, str, tuple[bytes, ValidatedImage], tuple[bytes, ValidatedImage]]] = []
    for k, (pa, pb) in enumerate(example_pairs or []):
        title, desc = (
            EXAMPLE_INFO[k] if k < len(EXAMPLE_INFO) else (f"Example {k + 1}", "synthetic pair")
        )
        try:
            examples.append(
                (title, desc, _example(Path(pa).read_bytes()), _example(Path(pb).read_bytes()))
            )
        except (OSError, InputRejectedError):
            continue
    if example_pairs is not None:
        rng = np.random.default_rng(0)
        na, nb = (_png(rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)) for _ in range(2))
        examples.append((*NOISE_INFO, _example(na), _example(nb)))

    def loaded(img: ValidatedImage, raw: bytes, origin: str) -> dict[str, Any]:
        h, w = img.rgb.shape[:2]
        return {"data": raw, "w": w, "h": h, "origin": origin}

    def is_crop(st: dict[str, Any] | None) -> bool:
        return st is not None and (st["w"], st["h"]) == (112, 112)

    def ready_text(which: str, st: dict[str, Any]) -> str:
        crop = " · looks like an aligned 112×112 face crop" if is_crop(st) else ""
        return f"<div class='vx-upl-ok'>✓ Image {which} ready ({st['origin']}, {st['w']}×{st['h']}){crop}</div>"

    def auto_pre(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
        present = [s for s in (a, b) if s is not None]
        return bool(present) and all(is_crop(s) for s in present)

    def on_upload(
        path: str | None, this_other: dict[str, Any] | None, which: str
    ) -> tuple[Any, ...]:
        try:
            raw = read_and_discard(path, temp_root)
            if raw is None:
                return None, "", gr.update(), gr.update(), *stale()
            img = load_image_bytes(raw, cfg.inputs)
        except InputRejectedError as exc:
            msg = f"<div class='vx-upl-err'>✕ Image {which} not accepted: {exc}. Use a JPEG, PNG, BMP or WebP photo.</div>"
            return None, msg, gr.update(value=None), gr.update(), *stale()
        except OSError:
            msg = f"<div class='vx-upl-err'>✕ Image {which} could not be read. Please upload it again.</div>"
            return None, msg, gr.update(value=None), gr.update(), *stale()
        st = loaded(img, raw, "uploaded")
        pair = (st, this_other) if which == "A" else (this_other, st)
        return st, ready_text(which, st), gr.update(), gr.update(value=auto_pre(*pair)), *stale()

    def stale() -> tuple[Any, Any]:
        return gr.update(value=EMPTY_HTML, visible=True), gr.update(visible=False)

    def on_clear_one() -> tuple[Any, ...]:
        return None, "", *stale()

    def on_example(k: int) -> tuple[Any, ...]:
        title, _, (raw_a, img_a), (raw_b, img_b) = examples[k]
        sts = [loaded(img_a, raw_a, f"example: {title}"), loaded(img_b, raw_b, f"example: {title}")]
        msg = (
            f"<div class='vx-alert vx-alert-warn'><b>Example loaded: {title}.</b> These are synthetic, already aligned "
            "112×112 faces, so <i>already aligned</i> was switched on in Advanced options. Tick the two confirmations "
            "and press <b>Compare faces</b>.</div>"
        )
        return (
            sts[0],
            sts[1],
            img_a.rgb,
            img_b.rgb,
            ready_text("A", sts[0]),
            ready_text("B", sts[1]),
            True,
            gr.update(value=msg, visible=True),
            gr.update(visible=False),
        )

    def on_clear_all() -> tuple[Any, ...]:
        return None, None, None, None, "", "", False, *stale()

    def gate(a: bool, b: bool) -> tuple[Any, Any]:
        ok = bool(a and b)
        hint = (
            ""
            if ok
            else "<div class='vx-hint'>Tick both confirmations above to enable <b>Compare faces</b>.</div>"
        )
        return gr.update(interactive=ok), hint

    def alert(text: str, kind: str = "warn") -> dict[Any, Any]:
        return {
            placeholder: gr.update(
                value=f"<div class='vx-alert vx-alert-{kind}'>{text}</div>", visible=True
            ),
            results: gr.update(visible=False),
        }

    def compare(
        st_a: dict[str, Any] | None,
        st_b: dict[str, Any] | None,
        fold: str,
        pre: bool,
        sel_a: float | None,
        sel_b: float | None,
        diag: bool,
        authorised: bool,
        calibration_ack: bool,
        allow_fb: bool,
    ) -> dict[Any, Any]:
        if not (authorised and calibration_ack):
            return alert(
                "<b>Please tick both confirmations</b> (authorisation and the calibration limitations) before comparing."
            )
        missing = [n for n, s in (("A", st_a), ("B", st_b)) if s is None]
        if missing:
            return alert(
                f"<b>Please add image {' and '.join(missing)}</b>: upload a photo or pick an example."
            )
        assert st_a is not None and st_b is not None
        try:
            res, a, b = verify_bytes(
                pipe, st_a["data"], st_b["data"], fold, pre, sel_a, sel_b, diag, allow_fb
            )
        except InputRejectedError as exc:
            return alert(f"<b>Input rejected</b> ({exc.code}): {exc}", "err")
        except Exception as exc:  # safe message; no stack trace or paths in the page
            return alert(safe_error_html(exc), "err")
        rule = pipe.rule(fold)
        floor = floors.get(fold)
        ex = build_explanation(res, rule, floor, ref)
        dec = res.decision
        meta = (
            f"Calibration system {res.fold} ({FAMILY_NAMES.get(res.fold, res.fold)} held out) · {res.device.upper()} · "
            f"{res.timings_ms.get('total', sum(res.timings_ms.values())):.0f} ms · research output, not a certified decision"
        )
        guard = floor is not None and res.quality is not None and res.quality < floor
        d = res.diagnostics
        multi = any(f.status == "multiple_faces" for f in (res.face_a, res.face_b))
        out: dict[Any, Any] = {
            placeholder: gr.update(visible=False),
            results: gr.update(visible=True),
            out_verdict: verdict_html(ex, meta),
            out_cards: metrics_html(res, rule, floor, ref),
            out_bar: score_bar_html(dec.fused_score, dec.tau_low, dec.tau_high, guard),
            out_why: steps_html(ex),
            out_det_a: detection_image(a, res.face_a),
            out_det_b: detection_image(b, res.face_b),
            out_crop_a: res.face_a.crop,
            out_crop_b: res.face_b.crop,
            vis_box: gr.update(visible=bool(d)),
            out_vis_a: visibility_overlay(res.face_a.crop, d["visibility_maps"][0]) if d else None,  # type: ignore[arg-type]
            out_vis_b: visibility_overlay(res.face_b.crop, d["visibility_maps"][1]) if d else None,  # type: ignore[arg-type]
            out_vis_md: novelty_md(d) if d else "",
            out_table: quantity_rows(res) or None,
            out_tech: status_md(res, a, b)
            + "\n\n"
            + abstention_text(dec.reasons, dec.label)
            + f"\n\n**Timings:** {timing_text(res)}",
            advanced: gr.update(open=True) if multi else gr.update(),
        }
        return out

    with gr.Blocks(
        title="VEIL-X · face verification (local research demo)",
        theme=theme,
        css=CSS,
        analytics_enabled=False,
        delete_cache=(60, 120),  # leftovers (result images, a failed upload) live at most ~3 min
    ) as demo:
        st_a = gr.State(None, time_to_live=3600)
        st_b = gr.State(None, time_to_live=3600)
        gr.HTML(HEADER_HTML)
        with gr.Tabs():
            with gr.Tab("Verify faces", id="verify"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=340):
                        gr.HTML("<div class='vx-h'>1 · Choose two images</div>")
                        with gr.Row():
                            with gr.Column(min_width=150):
                                in_a = gr.Image(
                                    label="Image A",
                                    type="filepath",
                                    image_mode=None,
                                    sources=["upload"],
                                    height=230,
                                    show_download_button=False,
                                    show_fullscreen_button=False,
                                )
                                msg_a = gr.HTML()
                            with gr.Column(min_width=150):
                                in_b = gr.Image(
                                    label="Image B",
                                    type="filepath",
                                    image_mode=None,
                                    sources=["upload"],
                                    height=230,
                                    show_download_button=False,
                                    show_fullscreen_button=False,
                                )
                                msg_b = gr.HTML()
                        if examples:
                            gr.HTML(
                                "<div class='vx-hint'>No photos at hand? Try an example (synthetic faces, no real people):</div>"
                            )
                            ex_buttons = []
                            with gr.Row():
                                for title, desc, _, _ in examples:
                                    ex_buttons.append(
                                        gr.Button(
                                            f"{title}\n{desc}",
                                            size="sm",
                                            elem_classes=["vx-ex-btn"],
                                        )
                                    )
                        with gr.Accordion("Advanced options", open=False) as advanced:
                            pre = gr.Checkbox(
                                False,
                                label="Images are already aligned 112×112 face crops: skip face detection (switched on automatically for 112×112 images)",
                            )
                            with gr.Row():
                                sel_a = gr.Number(
                                    -1,
                                    precision=0,
                                    label="Face to use in image A (-1 = automatic: only when exactly one face is found)",
                                )
                                sel_b = gr.Number(
                                    -1, precision=0, label="Face to use in image B (-1 = automatic)"
                                )
                            allow_fb = gr.Checkbox(
                                False,
                                label="Both images are loosely cropped, centred single-face portraits (LFW-style framing): allow the registered fixed-crop FALLBACK when no reliable face is detected",
                            )
                            diag = gr.Checkbox(
                                True,
                                label="Show where faces may be covered (visibility map and novelty check; never used in the decision)",
                            )
                            fold = gr.Dropdown(
                                list(cfg.decision.folds),
                                value=cfg.decision.default_fold,
                                label="Calibration system (frozen A3 system; default per D-051)",
                            )
                            with gr.Accordion("About this calibration system", open=False):
                                fold_card = gr.Markdown(
                                    fold_card_md(
                                        cfg.decision.default_fold, cfg.decision.phase3_frozen_sha256
                                    )
                                )
                        gr.HTML("<div class='vx-h'>2 · Confirm</div>")
                        authorised = gr.Checkbox(
                            False,
                            label="I am authorised to process these two images, and I understand this is a research demonstration (not identity verification)",
                        )
                        calib_ack = gr.Checkbox(
                            False,
                            label="I understand that the active configuration is an experimental calibration on synthetic faces and synthetic occluders, that its 5 % error target was not maintained under population shift, and that its output is not a certified or deployment-ready decision",
                        )
                        gr.HTML("<div class='vx-h'>3 · Compare</div>")
                        with gr.Row():
                            go = gr.Button(
                                "Compare faces",
                                variant="primary",
                                interactive=False,
                                scale=3,
                                elem_classes=["vx-compare"],
                            )
                            clear = gr.Button(
                                "Clear", variant="secondary", scale=1, elem_classes=["vx-compare"]
                            )
                        gate_hint = gr.HTML(
                            "<div class='vx-hint'>Tick both confirmations above to enable <b>Compare faces</b>.</div>"
                        )
                    with gr.Column(scale=7, min_width=360):
                        placeholder = gr.HTML(EMPTY_HTML)
                        with gr.Column(visible=False) as results:
                            out_verdict = gr.HTML()
                            out_cards = gr.HTML()
                            out_bar = gr.HTML()
                            gr.HTML(
                                "<div class='vx-h' style='margin-top:6px'>Why this result?</div>"
                            )
                            out_why = gr.HTML()
                            with (
                                gr.Accordion("Faces used (detection and alignment)", open=True),
                                gr.Row(),
                            ):
                                out_det_a = gr.Image(
                                    label="A: faces found (numbered)",
                                    interactive=False,
                                    height=170,
                                    min_width=100,
                                    show_download_button=False,
                                )
                                out_crop_a = gr.Image(
                                    label="A: aligned face used",
                                    interactive=False,
                                    height=170,
                                    min_width=100,
                                    show_download_button=False,
                                )
                                out_det_b = gr.Image(
                                    label="B: faces found (numbered)",
                                    interactive=False,
                                    height=170,
                                    min_width=100,
                                    show_download_button=False,
                                )
                                out_crop_b = gr.Image(
                                    label="B: aligned face used",
                                    interactive=False,
                                    height=170,
                                    min_width=100,
                                    show_download_button=False,
                                )
                            with gr.Accordion(
                                "Where might the faces be covered? (explanation only)", open=False
                            ) as vis_box:
                                gr.Markdown(
                                    "Blue = predicted visible face, orange = predicted covered. Computed **after** the decision; it cannot change it."
                                )
                                with gr.Row():
                                    out_vis_a = gr.Image(
                                        label="Visibility A",
                                        interactive=False,
                                        height=170,
                                        show_download_button=False,
                                    )
                                    out_vis_b = gr.Image(
                                        label="Visibility B",
                                        interactive=False,
                                        height=170,
                                        show_download_button=False,
                                    )
                                out_vis_md = gr.Markdown()
                            with gr.Accordion("Technical details", open=False):
                                out_table = gr.Dataframe(
                                    headers=["quantity", "value", "role"],
                                    interactive=False,
                                    wrap=True,
                                )
                                out_tech = gr.Markdown()
                gr.Markdown(DISCLAIMER, elem_classes=["vx-note"])
            with gr.Tab("How it works", id="how"):
                gr.HTML(HOW_HTML)
                gr.Markdown(GLOSSARY_MD)
                with gr.Accordion("Exact thresholds of every calibration system", open=False):
                    gr.Markdown(rules_md(pipe))
                with gr.Accordion("Models and experiment details", open=False):
                    gr.Markdown(model_info_md(cfg))
            with gr.Tab("Limitations", id="limits"):
                gr.Markdown(LIMITATIONS_MD)
                gr.Markdown(WHY_NOT_VISIBILITY_MD)
            with gr.Tab("Privacy & about", id="about"):
                gr.Markdown(PRIVACY_MD)
                gr.Markdown(DISCLAIMER)
                if ref is not None:
                    gr.Markdown(
                        f"Reference ranges used in explanations come from {ref['n_pairs']:,} development pairs "
                        "(synthetic DigiFace-1M faces, clean and occluded). They give context only and never affect a decision."
                    )

        for comp in (in_a, in_b):
            comp.preprocess = raw_upload_preprocess
        stale_out = [placeholder, results]
        in_a.upload(
            lambda p, o: on_upload(p, o, "A"), [in_a, st_b], [st_a, msg_a, in_a, pre, *stale_out]
        )
        in_b.upload(
            lambda p, o: on_upload(p, o, "B"), [in_b, st_a], [st_b, msg_b, in_b, pre, *stale_out]
        )
        in_a.clear(on_clear_one, None, [st_a, msg_a, *stale_out])
        in_b.clear(on_clear_one, None, [st_b, msg_b, *stale_out])
        if examples:
            for k, btn in enumerate(ex_buttons):
                btn.click(
                    lambda k=k: on_example(k),
                    None,
                    [st_a, st_b, in_a, in_b, msg_a, msg_b, pre, *stale_out],
                )
        clear.click(on_clear_all, None, [st_a, st_b, in_a, in_b, msg_a, msg_b, pre, *stale_out])
        authorised.change(gate, [authorised, calib_ack], [go, gate_hint])
        calib_ack.change(gate, [authorised, calib_ack], [go, gate_hint])
        fold.change(
            lambda f: fold_card_md(f, cfg.decision.phase3_frozen_sha256), [fold], [fold_card]
        )
        go.click(
            compare,
            [st_a, st_b, fold, pre, sel_a, sel_b, diag, authorised, calib_ack, allow_fb],
            [
                placeholder,
                results,
                out_verdict,
                out_cards,
                out_bar,
                out_why,
                out_det_a,
                out_det_b,
                out_crop_a,
                out_crop_b,
                vis_box,
                out_vis_a,
                out_vis_b,
                out_vis_md,
                out_table,
                out_tech,
                advanced,
            ],
            scroll_to_output=True,
        )
    return demo


def env_for_local_only(temp_root: Path) -> None:
    """Environment for strictly local operation; must run before gradio is imported."""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    # Gradio's cache (uploads, result images) gets its own folder: ``delete_cache`` empties it
    # regularly and must not remove the session's owner.pid marker next to it.
    cache = temp_root / "gradio_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
    os.environ.setdefault("GRADIO_SHARE", "False")
    np.seterr(all="ignore")
