"""Phase 4 computational cost per component (docs/phase4_protocol.md section 5, secondary).

Times the GPU forward pass on synthetic uint8 batches. No data is read, so no test access.
Components: AdaFace and ArcFace IR-101 (fp16), Phase 3 VE, VE-C with bottleneck features,
VE-C flip-TTA, and the Mahalanobis score (CPU, 256-d). Reports the median ms per image over 20
timed batches after warm-up, plus peak GPU memory.

    .venv-gpu\\Scripts\\python.exe scripts\\phase4_cost.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import torch
    from safetensors.torch import load_file

    from veilx.data.paths import resolve_data_root
    from veilx.models.extractor import FrozenExtractor
    from veilx.models.spec import BACKBONES
    from veilx.provenance import collect_provenance
    from veilx.vis.conservative import MahalanobisNovelty
    from veilx.vis.model import VisNet, to_input

    lay = resolve_data_root()
    rng = np.random.default_rng(0)
    batch = rng.integers(0, 256, (256, 112, 112, 3), dtype=np.uint8)
    dev = torch.device("cuda")

    def bench(fn: Any, reps: int = 20) -> dict[str, float]:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return {
            "ms_per_image": 1000 * float(np.median(ts)) / len(batch),
            "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
        }

    out: dict[str, Any] = {}
    for bid, spec in BACKBONES.items():
        ex = FrozenExtractor(
            spec,
            lay.models / spec.hf_repo.replace("/", "__") / spec.revision,
            device="cuda",
            precision="fp16",
        )
        out[bid] = bench(lambda ex=ex: ex.embed(batch))
        del ex
    for name, path in (
        ("ve_phase3", "models/phase3/ve/F1_mask/best.safetensors"),
        ("vec_phase4", "models/phase4/vec/F1_mask/best.safetensors"),
    ):
        m = VisNet().to(dev).eval()
        m.load_state_dict(load_file(str(lay.root / path), device="cuda"))
        x = torch.from_numpy(to_input(batch)).to(dev)

        def fwd(m: Any = m, x: Any = x) -> None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                m.forward_with_features(x)

        def tta(m: Any = m, x: Any = x) -> None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                m(x)
                m(torch.flip(x, dims=[3]))

        out[name] = bench(fwd)
        out[f"{name}_flip_tta"] = bench(tta)
    nov = MahalanobisNovelty.fit(rng.normal(size=(5000, 256)))
    feats = rng.normal(size=(256, 256))
    t0 = time.perf_counter()
    for _ in range(50):
        nov.score(feats)
    out["mahalanobis_cpu"] = {"ms_per_image": 1000 * (time.perf_counter() - t0) / 50 / 256}
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 4 computational cost (synthetic inputs; RTX 4060 Laptop GPU; batch 256)",
        "gpu": torch.cuda.get_device_name(0),
        "results": out,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase4_cost.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
