"""LFW detection and alignment (private outputs; aggregate report only is public).

Two preprocessing paths, always reported separately:

* ``det``    detector-inclusive: SCRFD (InsightFace det_10g, 640x640, thresh 0.5, NMS 0.4) on
             every image; the detection nearest the image centre is the labelled face (LFW
             images are centred on their subject); five landmarks -> similarity transform to
             the ArcFace 112x112 template. Images with no detection are FAILURES: they get no
             crop, are counted, and every pair using them is handled explicitly at evaluation
             (forced non-match / abstain) - never dropped silently.
* ``fixed``  detector-free: ONE similarity transform for every image, estimated from the mean
             landmark positions of the View-1 **DevTrain** images (the LFW development view),
             so no View-2 evaluation pair influences it.

    .venv-gpu\\Scripts\\python.exe scripts\\lfw_detect_align.py [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from veilx.data.images import encode_png_rgb, load_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.detect.align import TEMPLATE, align_to_template, umeyama, warp
from veilx.provenance import collect_provenance

ROOT = Path(__file__).resolve().parents[1]
DET_MODEL = "insightface_buffalo_l_v0.7/det_10g.onnx"
DET_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
_W: dict[str, Any] = {}


def crop_name(image_id: str) -> str:
    return image_id.replace(":", "__") + ".png"


def _init(model: str, root: str, out_det: str) -> None:
    from veilx.detect.scrfd import SCRFDDetector

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    _W["det"] = SCRFDDetector(Path(model))
    _W["root"], _W["out"] = Path(root), Path(out_det)


def _detect(job: tuple[str, str]) -> dict[str, Any]:
    from veilx.detect.scrfd import select_central

    image_id, relpath = job
    img = load_rgb(_W["root"] / relpath)
    t = time.perf_counter()
    dets = _W["det"].detect(img)
    ms = 1000 * (time.perf_counter() - t)
    k = select_central(dets, img.shape[:2])
    rec: dict[str, Any] = {"image_id": image_id, "n_faces": len(dets), "latency_ms": round(ms, 2)}
    if k is None:
        return rec | {"ok": False, "failure": "no_face_detected"}
    d = dets[k]
    crop, m = align_to_template(img, d.landmarks.astype(np.float64))
    (_W["out"] / crop_name(image_id)).write_bytes(encode_png_rgb(crop))
    return rec | {
        "ok": True,
        "chosen": k,
        "chosen_is_top_score": int(k == 0),
        "score": round(d.score, 4),
        "box": [round(float(x), 2) for x in d.box],
        "landmarks": [[round(float(x), 3) for x in p] for p in d.landmarks],
        "scale": round(float(np.hypot(m[0, 0], m[1, 0])), 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    layout = resolve_data_root()
    from veilx.data.archive import sha256_file

    model = layout.models / DET_MODEL
    if sha256_file(model) != DET_SHA256:
        raise RuntimeError("detector weights differ from the recorded hash")
    img_root = layout.extracted / "lfw_kaggle_v4"
    out_det = layout.extracted / "lfw_aligned" / "det"
    out_fixed = layout.extracted / "lfw_aligned" / "fixed"
    out_det.mkdir(parents=True, exist_ok=True)
    out_fixed.mkdir(parents=True, exist_ok=True)
    recs = list(read_jsonl(layout.manifests / "lfw" / "images.jsonl", ImageRecord))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(str(model), str(img_root), str(out_det))
    ) as pool:
        dets = list(pool.map(_detect, [(r.image_id, r.relpath) for r in recs], chunksize=32))
    wall = time.perf_counter() - t0
    by_id = {d["image_id"]: d for d in dets}
    out = layout.manifests / "lfw" / "detections.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for d in dets:
            fh.write(json.dumps(d) + "\n")
    # ---- fixed transform from View-1 DevTrain images only
    dev = list(read_jsonl(layout.manifests / "lfw" / "view1_devtrain_pairs.jsonl", PairRecord))
    dev_ids = sorted({p.image_a for p in dev} | {p.image_b for p in dev})
    lms = np.array([by_id[i]["landmarks"] for i in dev_ids if by_id[i]["ok"]], dtype=np.float64)
    mean_lm = lms.mean(axis=0)
    m_fixed = umeyama(mean_lm, TEMPLATE)
    for r in recs:
        crop = warp(load_rgb(img_root / r.relpath), m_fixed)
        (out_fixed / crop_name(r.image_id)).write_bytes(encode_png_rgb(crop))
    (layout.manifests / "lfw" / "fixed_transform.json").write_text(
        json.dumps(
            {
                "matrix": m_fixed.tolist(),
                "mean_landmarks": mean_lm.tolist(),
                "n_devtrain_images": len(lms),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # ---- aggregate report (no names, no ids)
    view2 = list(read_jsonl(layout.manifests / "lfw" / "view2_pairs.jsonl", PairRecord))
    v2_ids = {p.image_a for p in view2} | {p.image_b for p in view2}
    ok = np.array([d["ok"] for d in dets])
    lat = np.array([d["latency_ms"] for d in dets])
    nf = np.array([d["n_faces"] for d in dets])
    sc = np.array([d["score"] for d in dets if d["ok"]])
    failed_pairs = sum(1 for p in view2 if not (by_id[p.image_a]["ok"] and by_id[p.image_b]["ok"]))
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "SCRFD detection/alignment on Kaggle LFW v4 (deep-funneled); aggregate only",
        "detector": {
            "weights": "InsightFace buffalo_l v0.7 det_10g.onnx",
            "sha256": DET_SHA256,
            "input": "640x640",
            "det_thresh": 0.5,
            "nms_thresh": 0.4,
            "provider": "onnxruntime CPUExecutionProvider",
            "selection": "detection nearest the image centre",
        },
        "n_images": len(dets),
        "detected": int(ok.sum()),
        "failures_no_face": int((~ok).sum()),
        "view2_images_failed": sum(1 for i in v2_ids if not by_id[i]["ok"]),
        "view2_pairs_with_a_failed_side": failed_pairs,
        "multi_face_images": int((nf > 1).sum()),
        "faces_per_image_histogram": {
            str(k): int((nf == k).sum()) for k in sorted(set(nf.tolist()))
        },
        "chosen_not_top_score": int(
            sum(1 for d in dets if d["ok"] and not d["chosen_is_top_score"])
        ),
        "score_quantiles": np.quantile(sc, [0.01, 0.05, 0.5]).round(4).tolist(),
        "latency_ms_cpu": {
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
            "p99": round(float(np.percentile(lat, 99)), 1),
        },
        "wall_seconds": round(wall, 1),
        "workers": args.workers,
        "fixed_transform_from": f"mean landmarks of {len(lms)} View-1 DevTrain images",
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    pub = ROOT / "reports" / "experiments" / "lfw_detection.json"
    with pub.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
