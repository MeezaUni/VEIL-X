"""Verify the frozen backbones before any extraction (Phase 2, gate 3). Inference only.

For each approved backbone:

1. **Reference equality** - build the authors' own module from the downloaded reference code
   (pinned revision), load the same safetensors strictly, and compare outputs with the VEIL-X
   re-implementation (fp32, CPU and CUDA) on random tensors and real validation crops.
2. **Channel order** - verification AUC on validation pairs with RGB vs BGR input (the
   authors' config states RGB; this checks it empirically). Validation split only.
3. **Precision** - cosine agreement of bf16/fp16 autocast embeddings with fp32.
4. **Determinism** - two identical CUDA passes must give bit-identical embeddings.
5. **CPU fallback** - CPU vs CUDA fp32 agreement.

Writes aggregated numbers (no embeddings, no image ids) to
``reports/models/model_verification.json``.

    .venv-gpu\\Scripts\\python.exe scripts\\verify_models.py [--pairs 2000]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

from veilx.data.images import load_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.metrics.roc import roc_auc
from veilx.models.extractor import FrozenExtractor
from veilx.models.preprocess import to_model_input
from veilx.models.spec import BACKBONES, BackboneSpec, PreprocessSpec
from veilx.provenance import collect_provenance

ROOT = Path(__file__).resolve().parents[1]


def _reference_module(spec: BackboneSpec, ref_dir: Path) -> torch.nn.Module:
    """Instantiate the authors' architecture from their released file (no repo code is run
    beyond the module definition; fvcore, used only for FLOP counting, is stubbed)."""
    sub = "iresnet" if spec.architecture == "cvlface_ir101" else "iresnet_insightface"
    stub = types.ModuleType("fvcore.nn")
    stub.flop_count = None  # type: ignore[attr-defined]
    sys.modules.setdefault("fvcore", types.ModuleType("fvcore"))
    sys.modules.setdefault("fvcore.nn", stub)
    path = ref_dir / "models" / sub / "model.py"
    mspec = importlib.util.spec_from_file_location(f"ref_{sub}", path)
    assert mspec and mspec.loader
    mod = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(mod)
    net = (
        mod.IR_101(input_size=(112, 112), output_dim=512) if sub == "iresnet" else mod.iresnet100()
    )
    from safetensors.torch import load_file

    state = {
        k[len(spec.state_dict_prefix) :]: v
        for k, v in load_file(str(ref_dir / spec.weights_file)).items()
    }
    res = net.load_state_dict(state, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    return net.eval()


def _load_val(n_pairs: int, layout: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pairs = []
    for p in read_jsonl(layout.manifests / "pairs" / "val.jsonl", PairRecord):
        pairs.append(p)
        if len(pairs) >= n_pairs:
            break
    need = {p.image_a for p in pairs} | {p.image_b for p in pairs}
    recs = {
        r.image_id: r
        for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
        if r.image_id in need
    }
    ids = sorted(need)
    index = {k: i for i, k in enumerate(ids)}
    imgs = np.stack([load_rgb(layout.extracted / recs[k].relpath) for k in ids])
    a = np.array([index[p.image_a] for p in pairs])
    b = np.array([index[p.image_b] for p in pairs])
    y = np.array([1 if p.label == "genuine" else 0 for p in pairs])
    return imgs, a, b, y


def _embed_all(ex: FrozenExtractor, imgs: np.ndarray, bs: int = 128) -> np.ndarray:
    return np.concatenate([ex.embed(imgs[i : i + bs]).embeddings for i in range(0, len(imgs), bs)])


def _auc(emb: np.ndarray, a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc(np.sum(emb[a] * emb[b], axis=1), y))


def verify(spec: BackboneSpec, models_root: Path, data: tuple[np.ndarray, ...]) -> dict[str, Any]:
    imgs, a, b, y = data
    wdir = models_root / spec.hf_repo.replace("/", "__") / spec.revision
    out: dict[str, Any] = {"backbone_id": spec.backbone_id, "spec_fingerprint": spec.fingerprint()}
    t0 = time.perf_counter()
    ex = FrozenExtractor(spec, wdir, device="cuda", precision="fp32")
    out["load_seconds"] = round(time.perf_counter() - t0, 2)
    out["weights_tensor_fingerprint"] = ex.weights_fingerprint
    out["n_parameters"] = int(sum(p.numel() for p in ex.model.parameters()))

    # 1. reference equality ------------------------------------------------------------
    ref = _reference_module(spec, wdir)
    rnd = torch.from_numpy(np.random.default_rng(0).standard_normal((8, 3, 112, 112), np.float32))
    real = torch.from_numpy(to_model_input(imgs[:64], spec.preprocess))
    eq: dict[str, float] = {}
    with torch.inference_mode():
        for name, x in (("random", rnd), ("real", real)):
            mine_cpu = ex.model.to("cpu")(x)
            eq[f"{name}_cpu_max_abs_diff"] = float((ref(x) - mine_cpu).abs().max())
            ref_gpu, mine_gpu = ref.to("cuda"), ex.model.to("cuda")
            xg = x.to("cuda")
            eq[f"{name}_cuda_max_abs_diff"] = float((ref_gpu(xg) - mine_gpu(xg)).abs().max())
            eq[f"{name}_cpu_vs_cuda_max_abs_diff"] = float(
                (mine_cpu - mine_gpu(xg).cpu()).abs().max()
            )
            ref = ref.to("cpu")
    out["reference_equality"] = eq
    del ref
    torch.cuda.empty_cache()

    # 2. channel order -------------------------------------------------------------------
    emb_rgb = _embed_all(ex, imgs)
    ex.spec = _with_order(spec, "BGR")
    emb_bgr = _embed_all(ex, imgs)
    ex.spec = spec
    out["channel_order_val_auc"] = {
        "RGB": _auc(emb_rgb, a, b, y),
        "BGR": _auc(emb_bgr, a, b, y),
        "n_pairs": len(y),
        "n_genuine": int(y.sum()),
    }

    # 3. precision agreement + 4. determinism ---------------------------------------------
    again = _embed_all(ex, imgs)
    out["determinism_cuda_fp32_bit_identical"] = bool(np.array_equal(emb_rgb, again))
    prec: dict[str, Any] = {}
    for p in ("bf16", "fp16"):
        ex.precision = p  # type: ignore[assignment]
        e = _embed_all(ex, imgs)
        cos = np.sum(e * emb_rgb, axis=1)
        sd = np.abs(np.sum(e[a] * e[b], axis=1) - np.sum(emb_rgb[a] * emb_rgb[b], axis=1))
        prec[p] = {
            "min_cosine_to_fp32": float(cos.min()),
            "mean_cosine_to_fp32": float(cos.mean()),
            "max_abs_pair_score_diff": float(sd.max()),
            "val_auc": _auc(e, a, b, y),
        }
    ex.precision = "fp32"
    out["precision_agreement"] = prec

    # 5. throughput (fp32 and bf16, CUDA) --------------------------------------------------
    thr: dict[str, float] = {}
    batch = imgs[:256] if len(imgs) >= 256 else imgs
    for p in ("fp32", "bf16"):
        ex.precision = p  # type: ignore[assignment]
        ex.embed(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(3):
            ex.embed(batch)
        torch.cuda.synchronize()
        thr[f"{p}_images_per_s_batch{len(batch)}"] = round(
            3 * len(batch) / (time.perf_counter() - t), 1
        )
        thr[f"{p}_peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    ex.precision = "fp32"
    out["throughput_cuda"] = thr
    del ex
    torch.cuda.empty_cache()
    return out


def _with_order(spec: BackboneSpec, order: str) -> BackboneSpec:
    from dataclasses import replace

    return replace(spec, preprocess=replace(spec.preprocess, channel_order=order))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=2000)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available; this verification requires the GPU environment")
        return 1
    layout = resolve_data_root()
    data = _load_val(args.pairs, layout)
    print(f"loaded {len(data[0])} validation crops for {len(data[3])} pairs", flush=True)
    results = [verify(s, layout.models, data) for s in BACKBONES.values()]
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "model verification on DigiFace validation pairs (not an evaluation result)",
        "split": "val",
        "preprocess_default": PreprocessSpec().__dict__,
        "results": results,
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    out = ROOT / "reports" / "models" / "model_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
