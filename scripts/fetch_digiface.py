"""Download the APPROVED DigiFace-1M parts from the official host (Microsoft, linked from
github.com/microsoft/DigiFace1M). Approved parts only:

* Phase 1b: subjects_0-1999_72_imgs.zip        (2,000 identities x 72 images)
            subjects_100000-133332_5_imgs.zip  (~33,333 identities x 5 images)
* Phase 2:  subjects_133333-166665_5_imgs.zip  (~33,333 identities x 5 images)
            subjects_166666-199998_5_imgs.zip  (~33,333 identities x 5 images)

Expected sizes are the official host's Content-Length values (recorded 2026-10-26).
Already-complete archives are skipped; the download log is merged, never overwritten.
DigiFace publishes no checksums; this script records SHA-256 fingerprints after download.

    python scripts/fetch_digiface.py [--workers 32]   (data root from VEILX_DATA_ROOT)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from veilx.data.archive import sha256_file
from veilx.data.download import parallel_download
from veilx.data.paths import resolve_data_root

BASE_URL = "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/"
APPROVED_PARTS: dict[str, int] = {
    "subjects_0-1999_72_imgs.zip": 2_938_553_163,
    "subjects_100000-133332_5_imgs.zip": 3_399_664_847,
    "subjects_133333-166665_5_imgs.zip": 3_398_240_548,
    "subjects_166666-199998_5_imgs.zip": 3_400_932_852,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    layout = resolve_data_root().ensure()
    log_path = layout.private_reports / "download_log.json"
    previous = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    done = {e["file"] for e in previous}
    log = list(previous)
    for name, size in APPROVED_PARTS.items():
        dest = layout.raw / name
        if name in done and dest.exists() and dest.stat().st_size == size:
            print(f"{name}: already downloaded and logged", flush=True)
            continue
        print(f"downloading {name} ({size / 2**30:.2f} GiB)", flush=True)
        res = parallel_download(BASE_URL + name, layout.raw / name, size, workers=args.workers)
        digest = sha256_file(res.path)
        entry = {
            "file": name,
            "url": BASE_URL + name,
            "size_bytes": res.size_bytes,
            "expected_size_bytes": size,
            "sha256_first_download": digest,
            "published_checksum": None,
            "seconds": round(res.seconds, 1),
            "resumed_segments": res.resumed_segments,
            "completed_utc": datetime.now(tz=UTC).isoformat(),
        }
        print(json.dumps(entry), flush=True)
        log.append(entry)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
