# ruff: noqa: E501
"""Phase 5 end-to-end latency and memory benchmark of the application pipeline (development data).

Per device (cuda, cpu):
- cold start: load times of the detector, AdaFace, ArcFace and the VE-C diagnostics model, plus the
  latency of the first comparison;
- warm per-stage latency (median / p95 over N pairs): detect + align, AdaFace embedding, ArcFace
  embedding, fused score + A3 decision, diagnostics, total. Two input paths: LFW View 1 DevTrain
  photographs (detector path) and DigiFace p3_val crops (pre-aligned path).
Peak process RSS is sampled by a background thread; peak GPU memory comes from torch.

Batch size is one pair (the interactive case). The earlier "~2 ms per image" figure was for
batch-256 backbone throughput only and is NOT an end-to-end latency.

    .venv-gpu\\Scripts\\python.exe scripts\\phase5_benchmark.py [--n 40] [--n-cpu 10]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PeakRSS:
    def __init__(self) -> None:
        import psutil

        self.proc = psutil.Process()
        self.peak = 0
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.02)

    def __enter__(self) -> PeakRSS:
        self.t.start()
        return self

    def __exit__(self, *a: object) -> None:
        self._stop = True
        self.t.join()


def bench_device(
    device: str,
    cfg: Any,
    root: Path,
    lfw_pairs: list[tuple[np.ndarray, np.ndarray]],
    dig_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    import torch

    from veilx.app.pipeline import VerificationPipeline

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    pipe = VerificationPipeline(cfg, root, device=device)
    cold: dict[str, float] = {}
    t = time.perf_counter()
    pipe.faces.detector()
    cold["load_detector_ms"] = 1000 * (time.perf_counter() - t)
    for name in ("adaface_ir101_webface4m", "arcface_ir101_webface4m"):
        t = time.perf_counter()
        pipe.backbone(name)
        cold[f"load_{name.split('_')[0]}_ms"] = 1000 * (time.perf_counter() - t)
    t = time.perf_counter()
    pipe.rule(cfg.decision.default_fold)
    pipe.diagnostics_engine()._model(cfg.decision.default_fold)
    cold["load_rule_and_diagnostics_ms"] = 1000 * (time.perf_counter() - t)
    a, b = lfw_pairs[0]
    t = time.perf_counter()
    first = pipe.verify(a, b)
    cold["first_comparison_ms"] = 1000 * (time.perf_counter() - t)
    out: dict[str, Any] = {"cold": cold, "first_comparison_stages_ms": first.timings_ms}
    for path, pairs, pre in (
        ("lfw_detector_path", lfw_pairs, False),
        ("digiface_pre_aligned_path", dig_pairs, True),
    ):
        rec: dict[str, list[float]] = {}
        statuses: dict[str, int] = {}
        for a, b in pairs:
            r = pipe.verify(a, b, pre_aligned=pre)
            for k, v in r.timings_ms.items():
                rec.setdefault(k, []).append(v)
            for f in (r.face_a, r.face_b):
                statuses[f.status] = statuses.get(f.status, 0) + 1
        out[path] = {
            "n_pairs": len(pairs),
            "face_statuses": statuses,
            "stages_ms": {
                k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "n": len(v)}
                for k, v in rec.items()
            },
        }
    if device == "cuda":
        out["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
    del pipe
    return out


def main() -> int:
    import torch

    from veilx.app.config import load_app_config
    from veilx.data.images import load_rgb
    from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
    from veilx.data.paths import resolve_data_root
    from veilx.provenance import collect_provenance

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-cpu", type=int, default=10)
    args = ap.parse_args()
    lay = resolve_data_root()
    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    lfw = lay.manifests / "lfw"
    imgs = {r.image_id: r.relpath for r in read_jsonl(lfw / "images.jsonl", ImageRecord)}
    dev = list(read_jsonl(lfw / "view1_devtrain_pairs.jsonl", PairRecord))[: args.n]
    lfw_pairs = [
        (
            load_rgb(lay.extracted / "lfw_kaggle_v4" / imgs[p.image_a]),
            load_rgb(lay.extracted / "lfw_kaggle_v4" / imgs[p.image_b]),
        )
        for p in dev
    ]
    vp = list(read_jsonl(lay.manifests / "phase3/pairs/p3_val_pairs.jsonl", PairRecord))[: args.n]
    need = {i for p in vp for i in (p.image_a, p.image_b)}
    rel = {
        r.image_id: r.relpath
        for r in read_jsonl(lay.manifests / "digiface_final.jsonl", ImageRecord)
        if r.image_id in need
    }
    dig_pairs = [
        (load_rgb(lay.extracted / rel[p.image_a]), load_rgb(lay.extracted / rel[p.image_b]))
        for p in vp
    ]
    res: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    with PeakRSS() as mon:
        res["cuda"] = bench_device("cuda", cfg, lay.root, lfw_pairs, dig_pairs)
        res["cuda"]["peak_process_rss_gib_so_far"] = mon.peak / 2**30
        res["cpu"] = bench_device(
            "cpu", cfg, lay.root, lfw_pairs[: args.n_cpu], dig_pairs[: args.n_cpu]
        )
        res["peak_process_rss_gib_total"] = mon.peak / 2**30
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 5 end-to-end application latency and memory (one pair per request; development images; RTX 4060 Laptop GPU)",
        "results": res,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase5_benchmark.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1)
    for dev_ in ("cuda", "cpu"):
        print(
            dev_,
            json.dumps({k: res[dev_][k] for k in ("cold",)}),
            {
                p: {k: round(v["median"], 1) for k, v in res[dev_][p]["stages_ms"].items()}
                for p in ("lfw_detector_path", "digiface_pre_aligned_path")
            },
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
