# ruff: noqa: E501
"""Phase 6 (6H): is SCRFD on the onnxruntime CUDA provider a safe drop-in for the CPU detector?

Development data only: LFW View 1 DevTrain (the Phase 5 parity images and pairs).

The same frozen ``det_10g.onnx`` is run with the CPU and the CUDA execution provider. Measured:
- detection count;
- box, score and landmark differences;
- status agreement under the app policy;
- crop parity (aligned 112x112 pixels);
- end-to-end AdaFace cosine and A3 decision agreement on the DevTrain pairs;
- warm latency per image;
- GPU memory used by the onnxruntime session (nvidia-smi delta).

**Adoption rule** (stated before running):
- GPU detection is adopted as an *option* only if:
  - statuses agree on 100 % of images;
  - crops are pixel-identical, or differ by at most 2 grey levels, on >= 99 % of aligned images;
  - A3 decisions agree on 100 % of pairs.
- Otherwise the CPU detector remains the only supported path.
- In either case the CPU detector stays the default, because it produced every published result.

    .venv-gpu\\Scripts\\python.exe scripts\\phase6_gpu_detector.py [--n-images 300] [--n-pairs 300]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gpu_used_mib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def main() -> int:
    from veilx.app.config import load_app_config
    from veilx.app.face import FacePreparer
    from veilx.app.pipeline import VerificationPipeline
    from veilx.data.images import load_rgb
    from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
    from veilx.data.paths import resolve_data_root
    from veilx.detect.scrfd import SCRFDDetector
    from veilx.provenance import collect_provenance

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--n-pairs", type=int, default=300)
    args = ap.parse_args()
    lay = resolve_data_root()
    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    lfw = lay.manifests / "lfw"
    imgs = {r.image_id: r.relpath for r in read_jsonl(lfw / "images.jsonl", ImageRecord)}
    pairs = list(read_jsonl(lfw / "view1_devtrain_pairs.jsonl", PairRecord))[: args.n_pairs]
    ids = sorted({p.image_a for p in pairs} | {p.image_b for p in pairs})
    ids = ids[: max(args.n_images, 0)] if args.n_images < len(ids) else ids
    cache = {i: load_rgb(lay.extracted / "lfw_kaggle_v4" / imgs[i]) for i in ids}

    cpu = FacePreparer(cfg, lay.root)
    cpu.detector()  # SHA-256 checked
    m0 = gpu_used_mib()
    gpu = FacePreparer(cfg, lay.root)
    gpu._det = SCRFDDetector(
        gpu.model_path,
        det_thresh=cfg.detector.det_thresh,
        nms_thresh=cfg.detector.nms_thresh,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
    )
    active = gpu._det.session.get_providers()
    gpu._det.detect(next(iter(cache.values())))  # warm-up (CUDA kernels, cuDNN)
    m1 = gpu_used_mib()

    stats: dict[str, Any] = {
        "n_images": len(cache),
        "gpu_session_providers": active,
        "count_mismatch": 0,
        "status_mismatch": 0,
        "aligned_compared": 0,
        "crop_identical": 0,
        "crop_max_diff_le_2": 0,
        "crop_max_abs_diff": 0,
        "max_box_diff_px": 0.0,
        "max_landmark_diff_px": 0.0,
        "max_score_diff": 0.0,
    }
    t_cpu, t_gpu = [], []
    for im in cache.values():
        t = time.perf_counter()
        dc = cpu.detector().detect(im)
        t_cpu.append(1000 * (time.perf_counter() - t))
        t = time.perf_counter()
        dg = gpu._det.detect(im)
        t_gpu.append(1000 * (time.perf_counter() - t))
        if len(dc) != len(dg):
            stats["count_mismatch"] += 1
        else:
            for a, b in zip(dc, dg, strict=True):
                stats["max_box_diff_px"] = max(
                    stats["max_box_diff_px"],
                    float(np.abs(np.asarray(a.box) - np.asarray(b.box)).max()),
                )
                stats["max_landmark_diff_px"] = max(
                    stats["max_landmark_diff_px"], float(np.abs(a.landmarks - b.landmarks).max())
                )
                stats["max_score_diff"] = max(
                    stats["max_score_diff"], abs(float(a.score) - float(b.score))
                )
        pc = cpu.prepare(im, allow_fallback=True)
        pg = gpu.prepare(im, allow_fallback=True)
        if pc.status != pg.status:
            stats["status_mismatch"] += 1
        elif pc.crop is not None and pg.crop is not None:
            d = int(np.abs(pc.crop.astype(int) - pg.crop.astype(int)).max())
            stats["aligned_compared"] += 1
            stats["crop_identical"] += int(d == 0)
            stats["crop_max_diff_le_2"] += int(d <= 2)
            stats["crop_max_abs_diff"] = max(stats["crop_max_abs_diff"], d)

    # end-to-end decisions on the DevTrain pairs (same pipeline, detector swapped)
    pipe_c = VerificationPipeline(cfg, lay.root, device="cuda")
    pipe_g = VerificationPipeline(cfg, lay.root, device="cuda")
    pipe_g.faces = gpu
    pipe_g._backbones = pipe_c._backbones  # share the frozen backbones (memory)
    agree = n_dec = 0
    dcos: list[float] = []
    for p in pairs:
        if p.image_a not in cache or p.image_b not in cache:
            continue
        a, b = cache[p.image_a], cache[p.image_b]
        rc = pipe_c.verify(a, b, with_diagnostics=False, allow_fallback=True)
        rg = pipe_g.verify(a, b, with_diagnostics=False, allow_fallback=True)
        n_dec += 1
        agree += int(rc.decision.code == rg.decision.code)
        if rc.s_adaface is not None and rg.s_adaface is not None:
            dcos.append(abs(rc.s_adaface - rg.s_adaface))
    stats |= {
        "pairs_compared": n_dec,
        "decision_agreement": agree / max(n_dec, 1),
        "max_abs_adaface_cosine_diff": max(dcos) if dcos else None,
        "latency_ms_per_image": {
            "cpu_median": float(np.median(t_cpu)),
            "cpu_p95": float(np.percentile(t_cpu, 95)),
            "cuda_median": float(np.median(t_gpu)),
            "cuda_p95": float(np.percentile(t_gpu, 95)),
        },
        "gpu_memory_delta_mib_nvidia_smi": (m1 - m0) if m0 is not None and m1 is not None else None,
    }
    aligned = max(stats["aligned_compared"], 1)
    adopt = (
        "CUDAExecutionProvider" in active
        and stats["status_mismatch"] == 0
        and stats["count_mismatch"] == 0
        and stats["crop_max_diff_le_2"] / aligned >= 0.99
        and stats["decision_agreement"] == 1.0
    )
    stats["adoption_rule_met"] = bool(adopt)
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 6 GPU vs CPU SCRFD parity and latency (LFW View 1 DevTrain; development data)",
        "results": stats,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase6_gpu_detector.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
