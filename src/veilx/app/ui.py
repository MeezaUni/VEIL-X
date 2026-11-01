# ruff: noqa: E501, RUF001  (UI layout and user-facing text)
"""Local Gradio interface of the VEIL-X research demonstration (built by :func:`build_app`).

Privacy behaviour:
- uploads arrive as Gradio temp files; their bytes are read, validated in memory, and the temp file
  is deleted immediately (best effort, within the app's private temp directory only);
- nothing is logged, embeddings are never written, and there is no flagging, no public link and no
  analytics (see ``scripts/run_app.py``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from veilx.app.config import AppConfig
from veilx.app.decision import rule_description
from veilx.app.inputs import InputRejectedError, load_image_bytes
from veilx.app.pipeline import VerificationPipeline
from veilx.app.report import (
    DISCLAIMER,
    LIMITATIONS_MD,
    WHY_NOT_VISIBILITY_MD,
    abstention_text,
    annotate_faces,
    decision_html,
    rule_bar,
    visibility_overlay,
)

STATUS_TEXT = {
    "aligned": "one face detected and aligned (detector score and alignment residual acceptable)",
    "selected_face": "several faces detected; the face you selected was aligned",
    "multiple_faces": "several faces detected and none selected: REFUSED (the app does not choose a person)",
    "fallback_fixed_crop": "FALLBACK: no reliable detection; detector-free fixed crop used (not a detected face)",
    "pre_aligned": "declared pre-aligned 112×112 crop; detection skipped",
    "rejected": "rejected",
}


CONSENT_REQUIRED = (
    "<b>Please confirm</b> that you are authorised to process both images and that you understand "
    "this is a research demonstration. The uploaded files were discarded; upload them again."
)


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
) -> tuple[Any, ...]:
    """One comparison request: read and discard the uploads, then verify (or refuse)."""
    cfg = pipe.cfg
    empty = (None,) * 7
    try:
        raw_a, raw_b = read_and_discard(path_a, temp_root), read_and_discard(path_b, temp_root)
        if not authorised:
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
        a = load_image_bytes(raw_a, cfg.inputs)
        b = load_image_bytes(raw_b, cfg.inputs)
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
    res = pipe.verify(
        a.rgb,
        b.rgb,
        fold=fold,
        pre_aligned=pre,
        select_a=None if sel_a is None or sel_a < 0 else int(sel_a),
        select_b=None if sel_b is None or sel_b < 0 else int(sel_b),
        with_diagnostics=diag,
    )
    dec = res.decision
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
    rows: list[list[Any]] = []
    if res.s_adaface is not None:
        rows = [
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
    nov = "Diagnostics not computed."
    vis_a = vis_b = None
    if res.diagnostics:
        d = res.diagnostics
        vis_a = visibility_overlay(res.face_a.crop, d["visibility_maps"][0])  # type: ignore[arg-type]
        vis_b = visibility_overlay(res.face_b.crop, d["visibility_maps"][1])  # type: ignore[arg-type]

        def flag(n: bool) -> str:
            return "UNUSUAL (outside the estimator's training distribution)" if n else "typical"

        nov = (
            "**Explanatory diagnostics — NOT used in the decision.**\n\n"
            f"- Predicted visible face fraction: A {d['visible_fraction'][0]:.2f}, B {d['visible_fraction'][1]:.2f}\n"
            f"- Novelty (Mahalanobis on the visibility estimator's features; threshold {d['novelty_threshold']:.2f}): "
            f"A {d['novelty_scores'][0]:.2f} → {flag(d['novel'][0])}; B {d['novelty_scores'][1]:.2f} → {flag(d['novel'][1])}\n"
            "- Region estimates (predicted visible fraction): "
            + ", ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in d["regions"].items())
            + "\n\nThe estimator can miss occluders that look like faces and was trained only on synthetic occluders; "
            "a 'typical' novelty value does not mean the image is free of occlusion."
        )
    timing = (
        " · ".join(f"{k} {v:.0f} ms" for k, v in res.timings_ms.items()) + f" · device {res.device}"
    )
    return (
        decision_html(dec.label, dec.fused_score, dec.stratum),
        rule_bar(dec.fused_score, dec.tau_low, dec.tau_high),
        abstention_text(dec.reasons, dec.label),
        "\n\n".join(status),
        annotate_faces(a.rgb, res.face_a.faces, res.face_a.chosen) if res.face_a.faces else a.rgb,
        annotate_faces(b.rgb, res.face_b.faces, res.face_b.chosen) if res.face_b.faces else b.rgb,
        timing,
        res.face_a.crop,
        res.face_b.crop,
        rows or None,
        vis_a,
        vis_b,
        nov,
        f"Frozen system: {res.fold}",
    )


def build_app(
    pipe: VerificationPipeline,
    temp_root: Path | None = None,
    example_pairs: list[tuple[str, str]] | None = None,
) -> Any:
    import gradio as gr

    cfg = pipe.cfg
    theme = gr.themes.Base(
        font=["system-ui", "Segoe UI", "sans-serif"], font_mono=["Consolas", "monospace"]
    )

    def run(*args: Any) -> tuple[Any, ...]:
        return run_comparison(pipe, temp_root, *args)

    with gr.Blocks(
        title="VEIL-X research demo (local)",
        theme=theme,
        analytics_enabled=False,
        delete_cache=(600, 900),
    ) as demo:
        gr.Markdown("# VEIL-X — occlusion-aware face verification (research demonstration)")
        gr.Markdown(DISCLAIMER)
        with gr.Tab("Compare two images"):
            with gr.Row():
                in_a = gr.Image(label="Image A", type="filepath", sources=["upload"], height=280)
                in_b = gr.Image(label="Image B", type="filepath", sources=["upload"], height=280)
            with gr.Accordion("Options", open=False):
                fold = gr.Dropdown(
                    list(cfg.decision.folds),
                    value=cfg.decision.default_fold,
                    label="Frozen A3 system (Phase 3 fold; default per D-051)",
                )
                pre = gr.Checkbox(
                    False,
                    label="Inputs are pre-aligned 112×112 crops (e.g. DigiFace) — skip detection",
                )
                with gr.Row():
                    sel_a = gr.Number(
                        -1,
                        precision=0,
                        label="Face index to use in image A (-1 = none; needed if several faces)",
                    )
                    sel_b = gr.Number(-1, precision=0, label="Face index to use in image B")
                diag = gr.Checkbox(
                    True,
                    label="Compute explanatory diagnostics (visibility map, novelty) — never used in the decision",
                )
            authorised = gr.Checkbox(
                False,
                label="I am authorised to process these two images, and I understand this is a research demonstration (not identity verification)",
            )
            go = gr.Button("Compare locally with the frozen rule", variant="primary")
            out_decision = gr.HTML()
            out_system = gr.Markdown()
            gr.Markdown("**Decision-rule region of this pair's quality stratum (frozen A3)**")
            out_rule = gr.Image(
                show_label=False,
                interactive=False,
                height=90,
                show_download_button=False,
                show_fullscreen_button=False,
            )
            out_abstain = gr.Markdown()
            gr.Markdown("### Detector and alignment status")
            out_status = gr.Markdown()
            with gr.Row():
                out_det_a = gr.Image(
                    label="Detections in A (face indices shown)", interactive=False, height=240
                )
                out_det_b = gr.Image(label="Detections in B", interactive=False, height=240)
                out_crop_a = gr.Image(
                    label="Crop used for A (aligned or fallback)", interactive=False, height=140
                )
                out_crop_b = gr.Image(label="Crop used for B", interactive=False, height=140)
            gr.Markdown("### Similarity and quality diagnostics")
            out_table = gr.Dataframe(
                headers=["quantity", "value", "role"], interactive=False, wrap=True
            )
            gr.Markdown(
                "### Predicted visibility (explanatory only — not used in the decision)\nBlue = predicted visible face, orange = predicted not visible."
            )
            with gr.Row():
                out_vis_a = gr.Image(label="Predicted visibility A", interactive=False, height=160)
                out_vis_b = gr.Image(label="Predicted visibility B", interactive=False, height=160)
            out_novelty = gr.Markdown()
            out_timing = gr.Markdown()
            go.click(
                run,
                [in_a, in_b, fold, pre, sel_a, sel_b, diag, authorised],
                [
                    out_decision,
                    out_rule,
                    out_abstain,
                    out_status,
                    out_det_a,
                    out_det_b,
                    out_timing,
                    out_crop_a,
                    out_crop_b,
                    out_table,
                    out_vis_a,
                    out_vis_b,
                    out_novelty,
                    out_system,
                ],
            )
        with gr.Tab("Calibration & limitations"):
            gr.Markdown(LIMITATIONS_MD)
        with gr.Tab("Why visibility is not decisive"):
            gr.Markdown(WHY_NOT_VISIBILITY_MD)
        with gr.Tab("Model & experiment information"):
            gr.Markdown(model_info_md(cfg))
            gr.Markdown(rules_md(pipe))
        with gr.Tab("Synthetic examples"):
            gr.Markdown(
                "Optional reproducible examples using **synthetic DigiFace-1M** faces (research licence R-UDA) read from "
                "your local data root. No face images are bundled with VEIL-X. Examples are pre-aligned 112×112 crops: "
                "after loading, tick *pre-aligned* in Options."
            )
            if example_pairs:
                ex_btn = gr.Dropdown(
                    [f"example {i + 1}" for i in range(len(example_pairs))],
                    label="Synthetic example pair",
                )
                load = gr.Button("Load into the comparison tab")

                def load_example(name: str) -> tuple[str, str]:
                    i = int(name.split()[-1]) - 1
                    return example_pairs[i]

                load.click(load_example, [ex_btn], [in_a, in_b])
            else:
                gr.Markdown(
                    "*No local synthetic examples available (set VEILX_DATA_ROOT to a prepared data root).*"
                )
    return demo


def env_for_local_only(temp_root: Path) -> None:
    """Environment for strictly local operation; must run before gradio is imported."""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_TEMP_DIR"] = str(temp_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
    os.environ.setdefault("GRADIO_SHARE", "False")
    np.seterr(all="ignore")
