"""Prepare the Kaggle LFW copy (D-043) in the private data root. Never commits images or names.

Stages:
  extract   archive inspection (sizes, member pattern, traversal, types, CRC) and safe extraction
  scan      decode every image; pixel SHA-256; exact duplicates; dimensions/modes; official
            View-2 pairs (pairs.csv) parsed unmodified and checked (10 folds x 300 + 300, every
            referenced image present); View-1 DevTrain/DevTest pairs parsed for development use
  summary   aggregate, non-identifying statistics -> reports/data/lfw_kaggle_v4_summary.json

    python scripts/prepare_lfw.py extract scan summary     (data root from VEILX_DATA_ROOT)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from veilx.config import load_dataset_registry
from veilx.config.schema import AccessStatus, DatasetRole, SplitName
from veilx.data.archive import inspect_zip, safe_extract, sha256_file
from veilx.data.lfw import (
    LFWPair,
    image_id,
    image_relpath,
    parse_pairs_csv,
    require_images,
    to_pair_records,
)
from veilx.data.manifest import ImageRecord, write_jsonl
from veilx.data.paths import resolve_data_root
from veilx.errors import DataAccessError, ManifestError

ROOT = Path(__file__).resolve().parents[1]
DATASET = "kaggle_jessicali9530_lfw"
ARCHIVE = "lfw_kaggle_v4/lfw-dataset.zip"
EXPECTED_SHA256 = "fff0f15b1f848f0d834f8df58ca2e3273bda21e37bd9a895acf66de60521ba10"
EXPECTED_BYTES = 117_895_655
EXPECTED_IMAGES, EXPECTED_IDENTITIES = 13_233, 5_749
MEMBER_RE = re.compile(
    r"lfw-deepfunneled/lfw-deepfunneled/([A-Za-z_\-]+)/\1_\d{4}\.jpg|[A-Za-z_]+\.csv"
)
IMG_DIR = "lfw-deepfunneled/lfw-deepfunneled"


def _check_access() -> None:
    reg = load_dataset_registry(ROOT / "configs" / "datasets.yaml")
    d = reg.get(DATASET)
    if d.access_status is AccessStatus.BLOCKED or DatasetRole.TEST_REAL not in d.roles:
        raise DataAccessError(f"{DATASET} is not approved for test_real use")


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def stage_extract() -> None:
    _check_access()
    layout = resolve_data_root().ensure()
    arc = layout.raw / ARCHIVE
    if sha256_file(arc) != EXPECTED_SHA256:
        raise DataAccessError("LFW archive hash differs from the recorded download")
    dest = layout.extracted / "lfw_kaggle_v4"
    rep = inspect_zip(
        arc,
        expected_size=EXPECTED_BYTES,
        allowed_suffixes=frozenset({".jpg", ".csv"}),
        member_pattern=MEMBER_RE,
    )
    print(f"members={rep.n_members} crc_ok={rep.crc_ok} rejected={len(rep.rejected)}")
    if not dest.exists():
        safe_extract(arc, layout.extracted, rep, wrap_dir="lfw_kaggle_v4")
    _dump(layout.private_reports / "lfw_archive_report.json", rep.to_dict())


def _pairs(text: str, n_folds: int, per_class: int) -> list[LFWPair]:
    return parse_pairs_csv(text, n_folds=n_folds, per_class=per_class)


def _dev_pairs(root: Path, which: str) -> list[LFWPair]:
    """View 1 DevTrain/DevTest: match + mismatch CSVs stacked as one 'fold'."""
    rows = []
    for kind in ("match", "mismatch"):
        with (root / f"{kind}pairs{which}.csv").open(encoding="utf-8") as fh:
            body = [r for r in csv.reader(fh) if any(c.strip() for c in r)][1:]
        rows.append(body)
    if len(rows[0]) != len(rows[1]):
        raise ManifestError(f"View 1 {which}: unequal match/mismatch counts")
    text = "\n".join(",".join(r) for r in rows[0] + rows[1])
    return parse_pairs_csv("name,x,y\n" + text, n_folds=1, per_class=len(rows[0]))


def _decode(path: Path) -> tuple[str, str, tuple[int, int], str]:
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        mode = im.mode
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h = hashlib.sha256()
    h.update(f"{arr.shape}".encode())
    h.update(arr.tobytes())
    return hashlib.sha256(data).hexdigest(), h.hexdigest(), (arr.shape[1], arr.shape[0]), mode


def stage_scan() -> None:
    _check_access()
    layout = resolve_data_root()
    root = layout.extracted / "lfw_kaggle_v4"
    img_root = root / IMG_DIR
    files = sorted(img_root.glob("*/*.jpg"))
    names = sorted({f.parent.name for f in files})
    if len(files) != EXPECTED_IMAGES or len(names) != EXPECTED_IDENTITIES:
        raise ManifestError(
            f"expected 13,233 images / 5,749 identities, got {len(files)}/{len(names)}"
        )
    t0 = time.perf_counter()
    with ThreadPoolExecutor(16) as pool:
        decoded = list(pool.map(_decode, files))
    recs = []
    for f, (fsha, psha, _dims, _mode) in zip(files, decoded, strict=True):
        name, idx = f.parent.name, int(f.stem.rsplit("_", 1)[1])
        recs.append(
            ImageRecord(
                image_id=image_id(name, idx),
                dataset_id="lfw_kaggle_v4",
                identity_id=f"lfw:{name}",
                relpath=f"{IMG_DIR}/{image_relpath(name, idx)}",
                sha256=fsha,
                split=SplitName.TEST,
                pixel_sha256=psha,
                source_group=psha,
            )
        )
    dup = Counter(r.pixel_sha256 for r in recs)
    dup_groups = [h for h, c in dup.items() if c > 1]
    by_hash: dict[str, set[str]] = {}
    for r in recs:
        by_hash.setdefault(r.pixel_sha256 or "", set()).add(r.identity_id)
    cross_identity = sum(1 for h in dup_groups if len(by_hash[h]) > 1)
    out = layout.manifests / "lfw"
    write_jsonl(out / "images.jsonl", recs)
    text = (root / "pairs.csv").read_text(encoding="utf-8")
    view2 = _pairs(text, 10, 300)
    require_images(img_root, view2)
    write_jsonl(out / "view2_pairs.jsonl", to_pair_records(view2))
    folds = {p.position + (p.fold - 1) * 600: p.fold for p in view2}
    dev = {w: _dev_pairs(root, w) for w in ("DevTrain", "DevTest")}
    for w, ps in dev.items():
        require_images(img_root, ps)
        write_jsonl(
            out / f"view1_{w.lower()}_pairs.jsonl",
            [
                r.model_copy(update={"pair_id": r.pair_id.replace("lfw-f01", f"lfw-{w.lower()}")})
                for r in to_pair_records(ps)
            ],
        )
    view2_ids = {p.name_a for p in view2} | {p.name_b for p in view2}
    devtrain_ids = {p.name_a for p in dev["DevTrain"]} | {p.name_b for p in dev["DevTrain"]}
    stats = {
        "n_images": len(recs),
        "n_identities": len(names),
        "images_per_identity_histogram": dict(
            sorted(Counter(Counter(r.identity_id for r in recs).values()).items())
        ),
        "dims": dict(Counter(f"{w}x{h}" for _, _, (w, h), _ in decoded)),
        "modes": dict(Counter(m for *_, m in decoded)),
        "exact_duplicate_groups": len(dup_groups),
        "exact_duplicate_groups_spanning_identities": cross_identity,
        "view2": {
            "n_pairs": len(view2),
            "n_folds": len(set(folds.values())),
            "genuine": sum(p.genuine for p in view2),
            "impostor": sum(not p.genuine for p in view2),
            "n_identities": len(view2_ids),
            "n_images": len(
                {(p.name_a, p.idx_a) for p in view2} | {(p.name_b, p.idx_b) for p in view2}
            ),
        },
        "view1": {
            w: {"n_pairs": len(ps), "genuine": sum(p.genuine for p in ps)} for w, ps in dev.items()
        },
        "view1_devtrain_identities_also_in_view2": len(devtrain_ids & view2_ids),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    _dump(layout.private_reports / "lfw_scan.json", stats)
    print(json.dumps(stats, indent=1))


def stage_summary() -> None:
    layout = resolve_data_root()
    arc = json.loads((layout.private_reports / "lfw_archive_report.json").read_text("utf-8"))
    scan = json.loads((layout.private_reports / "lfw_scan.json").read_text("utf-8"))
    summary = {
        "evidence_type": "real_data_statistics",
        "dataset": DATASET,
        "source": "https://www.kaggle.com/datasets/jessicali9530/lfw-dataset (version 4)",
        "authorisation": "user-reported academic approval (D-043); not a dataset-owner licence",
        "note": "Aggregated counts only; no images, names or ids.",
        "archive": {
            k: arc[k]
            for k in ("size_bytes", "sha256", "n_members", "n_files", "crc_ok", "rejected")
            if k in arc
        },
        "scan": scan,
    }
    _dump(ROOT / "reports" / "data" / "lfw_kaggle_v4_summary.json", summary)
    print("wrote reports/data/lfw_kaggle_v4_summary.json")


STAGES = {"extract": stage_extract, "scan": stage_scan, "summary": stage_summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="+", choices=list(STAGES))
    for s in ap.parse_args().stages:
        print(f"=== {s}", flush=True)
        STAGES[s]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
