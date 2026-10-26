"""Resumable, segmented parallel HTTPS download from an official dataset host.

The DigiFace host throttles each connection (~0.3 MB/s observed) but supports HTTP range
requests, so the file is fetched as fixed-size segments by a thread pool and written at their
offsets into a pre-allocated ``.part`` file. Completed segments are recorded in a JSON state
file, so an interrupted download resumes without re-fetching finished segments.

Every segment response must be ``206 Partial Content`` with a matching ``Content-Range``;
the final file must have exactly the expected size. Integrity is then checked end-to-end by
the caller (SHA-256 fingerprint + zip CRC-32 of every member, ``veilx.data.archive``).
Only ``https`` URLs are accepted.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from veilx.errors import DataAccessError

_CR = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size_bytes: int
    segments: int
    seconds: float
    resumed_segments: int


def _fetch_segment(url: str, start: int, end: int, total: int, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https enforced)
        if resp.status != 206:
            raise DataAccessError(f"expected 206 Partial Content, got {resp.status}")
        m = _CR.match(resp.headers.get("Content-Range", ""))
        if not m or (int(m[1]), int(m[2]), int(m[3])) != (start, end, total):
            raise DataAccessError(f"unexpected Content-Range {resp.headers.get('Content-Range')!r}")
        data = resp.read()
    if len(data) != end - start + 1:
        raise DataAccessError(f"short segment {start}-{end}: {len(data)} bytes")
    return bytes(data)


def parallel_download(
    url: str,
    dest: Path,
    expected_size: int,
    *,
    workers: int = 32,
    segment_bytes: int = 16 * 1024 * 1024,
    retries: int = 8,
    timeout: float = 120.0,
    progress_every: float = 60.0,
) -> DownloadResult:
    if not url.startswith("https://"):
        raise DataAccessError("only https downloads are allowed")
    if dest.exists() and dest.stat().st_size == expected_size:
        return DownloadResult(dest, expected_size, 0, 0.0, 0)
    part = dest.with_name(dest.name + ".part")
    state_path = dest.with_name(dest.name + ".segments.json")
    n_seg = (expected_size + segment_bytes - 1) // segment_bytes
    done: set[int] = set()
    if part.exists() and state_path.exists() and part.stat().st_size == expected_size:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("expected_size") == expected_size
            and state.get("segment_bytes") == segment_bytes
        ):
            done = set(state.get("done", []))
    else:
        with part.open("wb") as fh:
            fh.truncate(expected_size)
    resumed = len(done)
    lock = threading.Lock()
    t0 = last = time.monotonic()

    def save_state() -> None:
        state_path.write_text(
            json.dumps(
                {
                    "expected_size": expected_size,
                    "segment_bytes": segment_bytes,
                    "done": sorted(done),
                }
            ),
            encoding="utf-8",
        )

    def work(k: int) -> int:
        start = k * segment_bytes
        end = min(expected_size, start + segment_bytes) - 1
        for attempt in range(retries):
            try:
                data = _fetch_segment(url, start, end, expected_size, timeout)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(min(60.0, 2.0**attempt))
        with lock, part.open("r+b") as fh:
            fh.seek(start)
            fh.write(data)
        return k

    todo = [k for k in range(n_seg) if k not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, k) for k in todo]
        for fut in as_completed(futures):
            k = fut.result()  # re-raises segment failures after retries
            with lock:
                done.add(k)
                save_state()
                now = time.monotonic()
                if now - last >= progress_every:
                    last = now
                    mib = len(done) * segment_bytes / 2**20
                    print(
                        f"  {dest.name}: {len(done)}/{n_seg} segments "
                        f"({mib:,.0f} MiB, {mib / (now - t0):.1f} MiB/s)",
                        flush=True,
                    )
    if len(done) != n_seg or part.stat().st_size != expected_size:
        raise DataAccessError(f"{dest.name}: incomplete download ({len(done)}/{n_seg} segments)")
    part.replace(dest)
    state_path.unlink(missing_ok=True)
    return DownloadResult(dest, expected_size, n_seg, time.monotonic() - t0, resumed)
