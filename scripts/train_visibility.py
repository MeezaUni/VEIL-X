"""Train the Phase 3 visibility estimator for one held-out family (docs/phase3_protocol.md §3).

Training images: p3_train only; every sample is clean (25 %) or occluded with one SEEN family
(the fold's held-out family is excluded and checked). Validation: a fixed set of 10,000
deterministic p3_val renders (seen families). Early stopping on validation loss; the best
weights are kept as safetensors. Resumable: the latest epoch state is checkpointed atomically
(SHA-256 verified) and restored with ``torch.load(weights_only=True)``.

Rendering runs in torch-free worker processes (bounded queue); the GPU process trains.
Requires ``authorized: true`` in the config AND ``VEILX_PHASE2_AUTHORIZED=1``.

    .venv-gpu\\Scripts\\python.exe scripts\\train_visibility.py \\
        --config configs/experiments/phase3/ve_F1_mask.yaml [--pilot]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.images import load_rgb
from veilx.data.manifest import ImageRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.seeding import derive_seed
from veilx.training.config import assert_training_authorized, load_training_config
from veilx.vis.regions import GRID, face_cells
from veilx.vis.samples import check_isolation, draw_condition, make_sample

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
P3 = PROTO.phase3
assert P3 is not None
FAMILIES = tuple(PROTO.family_ids)
SAMPLES_PER_EPOCH = 50_000
VAL_SAMPLES = 10_000

_W: dict[str, Any] = {}


def _init(root: str, recs: list[tuple[str, str]]) -> None:
    _W["root"], _W["recs"] = Path(root), recs


def _render(jobs: list[tuple[int, str | None, str | None, int]]) -> tuple[np.ndarray, np.ndarray]:
    imgs = np.empty((len(jobs), 112, 112, 3), np.uint8)
    tgts = np.empty((len(jobs), GRID, GRID), np.float32)
    for k, (idx, fam, lvl, seed) in enumerate(jobs):
        relpath, psha = _W["recs"][idx]
        img, tgt, _ = make_sample(load_rgb(_W["root"] / relpath), fam, lvl, seed, psha)
        imgs[k], tgts[k] = img, tgt
    return imgs, tgts


def _jobs(
    n_images: int, n: int, allowed: tuple[str, ...], seed: int, tag: str
) -> list[tuple[int, str | None, str | None, int]]:
    out = []
    for i in range(n):
        rng = np.random.default_rng(derive_seed(seed, tag, i))
        fam, lvl = draw_condition(rng, allowed)
        out.append((int(rng.integers(n_images)), fam, lvl, derive_seed(seed, tag, "occl", i)))
    return out


def _batches(pool: ProcessPoolExecutor, jobs: list[Any], bs: int, inflight: int):  # type: ignore[no-untyped-def]
    chunks = iter([jobs[i : i + bs] for i in range(0, len(jobs), bs)])
    q: deque[Future[Any]] = deque()
    while True:
        while len(q) < inflight:
            nxt = next(chunks, None)
            if nxt is None:
                break
            q.append(pool.submit(_render, nxt))
        if not q:
            return
        yield q.popleft().result()


def main() -> int:
    import torch
    from safetensors.torch import save as st_save
    from torch import nn

    from veilx.training.checkpoint import CheckpointManager
    from veilx.vis.metrics import all_scores
    from veilx.vis.model import VisNet, n_parameters, to_input

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pilot", action="store_true", help="2 short epochs, separate output dir")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = load_training_config(args.config)
    assert_training_authorized(cfg)
    heldout = cfg.heldout_family
    allowed = tuple(f for f in FAMILIES if f != heldout)
    check_isolation(allowed, heldout)
    layout = resolve_data_root()
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    sub = json.loads((layout.manifests / "phase3" / "subsplits.json").read_text("utf-8"))
    tr_ids = sorted(k for k, v in sub.items() if v == "p3_train")
    va_ids = sorted(k for k, v in sub.items() if v == "p3_val")
    wanted = set(tr_ids) | set(va_ids)
    man = {  # stream: keep only paths/hashes of the images this run uses
        r.image_id: (r.relpath, r.pixel_sha256 or r.sha256)
        for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
        if r.image_id in wanted
    }
    recs = [man[i] for i in tr_ids + va_ids]
    n_tr = len(tr_ids)
    per_epoch = 2048 if args.pilot else SAMPLES_PER_EPOCH
    n_val = 1024 if args.pilot else VAL_SAMPLES
    max_epochs = 2 if args.pilot else cfg.max_epochs
    patience = P3.ve_patience  # type: ignore[union-attr]
    out_dir = layout.root / cfg.checkpoint.directory / ("pilot" if args.pilot else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(out_dir / "state", keep_last=2)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = VisNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    steps_per_epoch = (per_epoch + cfg.batch_size - 1) // cfg.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs * steps_per_epoch)
    amp = cfg.precision.mode in ("fp16", "bf16") and dev.type == "cuda"
    amp_dtype = torch.float16 if cfg.precision.mode == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    w_face = torch.from_numpy(np.where(face_cells(), 1.0, 0.25).astype(np.float32)).to(dev)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    hist: list[dict[str, Any]] = []
    start_epoch, best_loss, bad = 0, float("inf"), 0
    latest = ckpt.latest()
    if latest is not None:  # resume
        state = torch.load(io.BytesIO(ckpt.load(latest)), map_location=dev, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_loss, bad = (
            int(state["epoch"]) + 1,
            float(state["best"]),
            int(state["bad"]),
        )
        hist = json.loads((out_dir / "history.json").read_text("utf-8"))
        print(f"resumed after epoch {start_epoch - 1}", flush=True)
    val_jobs = [
        (n_tr + j[0] % len(va_ids), *j[1:]) for j in _jobs(len(va_ids), n_val, allowed, seed, "val")
    ]
    t_all = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(str(layout.extracted), recs)
    ) as pool:
        vx, vt = [], []
        for imgs, tg in _batches(pool, val_jobs, 256, 2 * args.workers):
            vx.append(imgs)
            vt.append(tg)
        val_x, val_t = np.concatenate(vx), np.concatenate(vt)

        def evaluate(dtype_amp: bool) -> tuple[float, np.ndarray]:
            model.eval()
            losses, probs = [], []
            with torch.inference_mode():
                for i in range(0, len(val_x), 256):
                    x = torch.from_numpy(to_input(val_x[i : i + 256])).to(dev)
                    t = torch.from_numpy(val_t[i : i + 256]).to(dev)
                    with torch.autocast(dev.type, dtype=amp_dtype, enabled=dtype_amp):
                        lg = model(x).float()
                    losses.append(float((bce(lg, t) * w_face).mean()) * len(x))
                    probs.append(torch.sigmoid(lg).cpu().numpy())
            return sum(losses) / len(val_x), np.concatenate(probs)

        for epoch in range(start_epoch, max_epochs):
            model.train()
            t0 = time.perf_counter()
            jobs = _jobs(n_tr, per_epoch, allowed, seed, f"train-e{epoch}")
            tot, n = 0.0, 0
            for imgs, tg in _batches(pool, jobs, cfg.batch_size, 2 * args.workers):
                x = torch.from_numpy(to_input(imgs)).to(dev, non_blocking=True)
                t = torch.from_numpy(tg).to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp):
                    loss = (bce(model(x).float(), t) * w_face).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += float(loss.detach()) * len(x)
                n += len(x)
            vloss, vprob = evaluate(amp)
            sc = all_scores(vprob.astype(np.float64), val_t.astype(np.float64))
            rec = {
                "epoch": epoch,
                "train_loss": tot / n,
                "val_loss": vloss,
                "val_iou": sc["iou"],
                "val_global_mae": sc["global_mae"],
                "val_brier": sc["brier"],
                "lr": sched.get_last_lr()[0],
                "seconds": round(time.perf_counter() - t0, 1),
                "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if dev.type == "cuda"
                else None,
            }
            hist.append(rec)
            improved = vloss < best_loss - 1e-5
            if improved:
                best_loss, bad = vloss, 0
                (out_dir / "best.safetensors").write_bytes(
                    st_save(
                        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
                    )
                )
            else:
                bad += 1
            buf = io.BytesIO()
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best": best_loss,
                    "bad": bad,
                },
                buf,
            )
            (out_dir / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
            ckpt.save(epoch, buf.getvalue(), {"fold": heldout, "val_loss": vloss})
            print(json.dumps(rec), flush=True)
            if bad >= patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
    # AMP validation of the selected weights: fp32 vs fp16 predictions on the val set
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(out_dir / "best.safetensors"), device=str(dev)))
    l32, p32 = evaluate(False)
    l16, p16 = evaluate(True)
    summary = {
        "evidence_type": "real_measured_model_result",
        "scope": "visibility-estimator training on DigiFace p3_train (synthetic occlusions)",
        "fold_heldout_family": heldout,
        "seen_families": list(allowed),
        "pilot": args.pilot,
        "n_parameters": n_parameters(model),
        "samples_per_epoch": per_epoch,
        "val_samples": n_val,
        "epochs_run": len(hist),
        "best_val_loss": best_loss,
        "best_epoch": int(min(hist, key=lambda r: r["val_loss"])["epoch"]),
        "amp_check": {
            "val_loss_fp32": l32,
            "val_loss_fp16": l16,
            "max_abs_prob_diff": float(np.abs(p32 - p16).max()),
        },
        "history": hist,
        "wall_seconds": round(time.perf_counter() - t_all, 1),
        "config": cfg.model_dump(),
        "best_weights_sha256": __import__("hashlib")
        .sha256((out_dir / "best.safetensors").read_bytes())
        .hexdigest(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )
    if not args.pilot:
        pub = ROOT / "reports" / "experiments" / f"phase3_ve_training_{heldout}.json"
        with pub.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(summary, fh, indent=1, default=str)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "fold_heldout_family",
                    "epochs_run",
                    "best_epoch",
                    "best_val_loss",
                    "amp_check",
                    "wall_seconds",
                )
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.exit(main())
