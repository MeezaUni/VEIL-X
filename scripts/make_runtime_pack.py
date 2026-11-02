# ruff: noqa: E501
"""Build the private VEIL-X runtime pack: the frozen files the app needs that are NOT in the public
repository.

The pack lets someone else run the application with ``run_veilx.bat``.

**Contents** (all SHA-256 checked against ``configs/app/veilx_app.yaml`` before packing):
- frozen decision system: ``reports/phase3_frozen.json``;
- frozen diagnostics: ``reports/phase4_frozen.json`` and the five VE-C weight files;
- SCRFD face detector: ``det_10g.onnx``;
- the 4 synthetic demo pairs (DigiFace-1M, computer-generated faces) as ``demo_examples/``.

**Not included:**
- the AdaFace / ArcFace weights (``run_veilx.bat`` downloads them from Hugging Face; pass
  ``--with-backbones`` for a fully offline pack, about 600 MB);
- any dataset, manifest, embedding or real-person image.

Licences: the DigiFace-1M images and the models carry non-commercial research terms. Share the
pack privately (for example with an evaluator), not publicly.

    .venv-gpu\\Scripts\\python.exe scripts\\make_runtime_pack.py [--out VEIL-X_runtime_pack.zip] [--with-backbones]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from veilx.app.config import content_sha256, load_app_config
    from veilx.data.archive import sha256_file
    from veilx.data.paths import resolve_data_root
    from veilx.models.spec import BACKBONES

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "deliverables" / "VEIL-X_runtime_pack.zip"))
    ap.add_argument("--with-backbones", action="store_true")
    args = ap.parse_args()
    import json

    root = resolve_data_root().root
    cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
    files: list[Path] = []
    for rel, sha in (
        (cfg.decision.phase3_frozen_file, cfg.decision.phase3_frozen_sha256),
        (cfg.diagnostics.phase4_frozen_file, cfg.diagnostics.phase4_frozen_sha256),
    ):
        data = json.loads((root / rel).read_text("utf-8"))
        if content_sha256({k: v for k, v in data.items() if k != "sha256"}) != sha:
            raise SystemExit(f"{rel}: content hash differs from the registered value")
        files.append(root / rel)
    det = root / cfg.detector.model
    if sha256_file(det) != cfg.detector.sha256:
        raise SystemExit("detector weights differ from the registered SHA-256")
    files.append(det)
    for fold, sha in cfg.diagnostics.vec_sha256.items():
        f = root / cfg.diagnostics.vec_dir / fold / "best.safetensors"
        if sha256_file(f) != sha:
            raise SystemExit(f"VE-C weights of {fold} differ from the registered SHA-256")
        files.append(f)
    if args.with_backbones:
        for spec in BACKBONES.values():
            prefix = spec.backbone_id.split("_")[0]
            matches = list(
                (root / "models").glob(f"*{prefix}_ir101_webface4m/*/{spec.weights_file}")
            )
            if not matches or sha256_file(matches[0]) != spec.weights_sha256:
                raise SystemExit(f"{spec.backbone_id}: weights missing or hash mismatch")
            files.append(matches[0])
        prov = root / "models" / "model_provenance.json"
        if prov.exists():
            files.append(prov)
    # demo examples, generated exactly as the app does
    spec_ = importlib.util.spec_from_file_location("run_app", ROOT / "scripts" / "run_app.py")
    assert spec_ is not None and spec_.loader is not None
    run_app = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(run_app)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ex = run_app.make_examples(root, Path(tmp) / "demo_examples")
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, "veilx_data/" + f.relative_to(root).as_posix())
            for a, b in ex:
                for p in (Path(a), Path(b)):
                    z.write(p, "veilx_data/demo_examples/" + p.name)
            z.writestr(
                "veilx_data/README.txt",
                "VEIL-X runtime pack (private). Keep this folder next to run_veilx.bat as 'veilx_data'.\n"
                "Contains frozen decision/diagnostic files, the SCRFD detector and synthetic demo "
                "images (DigiFace-1M, computer-generated). Non-commercial research use only; do not "
                "redistribute publicly.\n",
            )
    print(
        f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB, {len(files)} model/config files, {2 * len(ex)} demo images)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
