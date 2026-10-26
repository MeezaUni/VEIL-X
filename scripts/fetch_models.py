"""Download the approved frozen face-recognition checkpoints at PINNED revisions (Phase 2).

Only ``model.safetensors`` (tensor-only format; no pickle) and the authors' reference code /
configs are fetched; the pickled ``pretrained_model/model.pt`` is deliberately NOT downloaded.
Each weight file is checked against the SHA-256 that the Hugging Face Hub records for its LFS
object, and the result is written to ``<data_root>/models/model_provenance.json``.

Weights stay in the private data root (licence-restricted, never committed).

    .venv-gpu\\Scripts\\python.exe scripts\\fetch_models.py   (data root from VEILX_DATA_ROOT)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

from veilx.data.archive import sha256_file
from veilx.data.paths import resolve_data_root

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# repo -> (pinned commit, expected safetensors SHA-256 from the Hub LFS pointer, size bytes)
APPROVED_MODELS: dict[str, tuple[str, str, int]] = {
    "minchul/cvlface_adaface_ir101_webface4m": (
        "f2b38d9e24bfe301490d8dd081d8924b102333dd",
        "21adb6220e8799a0e658f16946df9649c7269f432fe9810a7b9c4ad1241080a8",
        260_980_552,
    ),
    "minchul/cvlface_arcface_ir101_webface4m": (
        "5846a5e96fbe1094952a414c78ce5b057c92d013",
        "e90e0c8e24c94409fe53c2254a1343ef54f48a3a66b885b8ff69aefa21660ee7",
        260_997_808,
    ),
}
# Reference code and configs (small text files) used to cross-check the strict loader.
CODE_SUFFIXES = (".py", ".yaml", ".json", ".md", ".txt")


def main() -> int:
    from huggingface_hub import HfApi, hf_hub_download

    layout = resolve_data_root().ensure()
    api = HfApi()
    records = []
    for repo, (rev, sha_expected, size_expected) in APPROVED_MODELS.items():
        info = api.model_info(repo, revision=rev, files_metadata=True)
        if info.sha != rev:
            raise RuntimeError(f"{repo}: hub resolved {info.sha}, expected pinned {rev}")
        local = layout.models / repo.replace("/", "__") / rev
        wanted = [
            s.rfilename
            for s in info.siblings
            if s.rfilename == "model.safetensors" or s.rfilename.endswith(CODE_SUFFIXES)
        ]
        for name in wanted:
            hf_hub_download(repo, name, revision=rev, local_dir=local)
        weights = local / "model.safetensors"
        digest = sha256_file(weights)
        size = weights.stat().st_size
        if digest != sha_expected or size != size_expected:
            raise RuntimeError(f"{repo}: weight hash/size mismatch ({digest}, {size})")
        files = {
            str(p.relative_to(local)).replace("\\", "/"): sha256_file(p)
            for p in sorted(local.rglob("*"))
            if p.is_file() and ".cache" not in p.parts
        }
        records.append(
            {
                "repo": repo,
                "revision": rev,
                "url": f"https://huggingface.co/{repo}/tree/{rev}",
                "weights": "model.safetensors",
                "weights_sha256": digest,
                "weights_bytes": size,
                "pickle_checkpoint_downloaded": False,
                "card_data": info.card_data.to_dict() if info.card_data else {},
                "files_sha256": files,
                "local_dir": str(local),
                "downloaded_utc": datetime.now(tz=UTC).isoformat(),
            }
        )
        print(f"{repo}@{rev[:10]}: {size:,} bytes sha256 {digest[:16]}... OK", flush=True)
    out = layout.models / "model_provenance.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
