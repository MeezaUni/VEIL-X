r"""Launch the VEIL-X local research application (Phase 5). Local only: loopback, no share link.

    $env:VEILX_DATA_ROOT = "C:\path\to\veilx_data"
    .venv-gpu\Scripts\python.exe scripts\run_app.py [--port 7860] [--device cuda|cpu]

The app binds to 127.0.0.1 only. There is no Gradio share link, analytics are disabled and the
Hugging Face hub is offline. Uploads land in a PRIVATE temporary directory created for this
session; each upload is deleted as soon as it has been read, and the whole directory is removed on
exit. Directories left by a killed session are removed at the next start (owner PID check).
Optional synthetic examples (DigiFace, development identities) are written to the same temp
directory, never into the repository.
"""

from __future__ import annotations

import argparse
import atexit
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_examples(data_root: Path, out: Path, n: int = 4) -> list[tuple[str, str]]:
    """Synthetic DigiFace example pairs (p3_val: development identities; pre-aligned crops), plus
    one pair with a synthetic mask occluder on image B. Written to the session temp dir only."""
    from veilx.data.images import encode_png_rgb, load_rgb, pixel_sha256
    from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
    from veilx.occlusion.engine import apply_occlusion

    pairs = list(read_jsonl(data_root / "manifests/phase3/pairs/p3_val_pairs.jsonl", PairRecord))[
        :n
    ]
    need = {i for p in pairs for i in (p.image_a, p.image_b)}
    rel = {
        r.image_id: r.relpath
        for r in read_jsonl(data_root / "manifests/digiface_final.jsonl", ImageRecord)
        if r.image_id in need
    }
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for k, p in enumerate(pairs):
        a, b = (
            load_rgb(data_root / "extracted" / rel[p.image_a]),
            load_rgb(data_root / "extracted" / rel[p.image_b]),
        )
        if k == n - 1:  # one occluded example (synthetic mask, high severity)
            b = apply_occlusion(
                b, "F1_mask", "high", 20261102, source_pixel_sha256=pixel_sha256(b)
            ).image
        pa, pb = out / f"example{k + 1}_a.png", out / f"example{k + 1}_b.png"
        pa.write_bytes(encode_png_rgb(a))
        pb.write_bytes(encode_png_rgb(b))
        res.append((str(pa), str(pb)))
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    from veilx.app.tempfiles import cleanup_stale, create_session_dir, remove_session_dir

    stale = cleanup_stale()  # directories of killed sessions (owner process gone)
    temp_root = create_session_dir()
    atexit.register(remove_session_dir, temp_root)
    if stale:
        print(f"removed {len(stale)} stale VEIL-X temp director{'y' if len(stale) == 1 else 'ies'}")

    from veilx.app.ui import env_for_local_only

    env_for_local_only(temp_root)  # before gradio is imported
    from veilx.app.config import load_app_config
    from veilx.app.pipeline import VerificationPipeline
    from veilx.app.ui import build_app
    from veilx.data.paths import resolve_data_root

    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    data_root = resolve_data_root().root
    pipe = VerificationPipeline(cfg, data_root, device=args.device)
    if not args.no_warmup:
        pipe.warm_up()
    examples = None
    if not args.no_examples:
        try:
            examples = make_examples(data_root, temp_root / "examples")
        except (OSError, KeyError, StopIteration):
            examples = None
    demo = build_app(pipe, temp_root, examples).queue(max_size=8)
    import uvicorn

    from veilx.app.server import build_asgi

    app = build_asgi(demo, cfg, args.port, temp_root)
    print(
        f"VEIL-X research demo: http://{cfg.server.host}:{args.port}  (local only; Ctrl+C to stop)",
        flush=True,
    )
    uvicorn.run(app, host=cfg.server.host, port=args.port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
