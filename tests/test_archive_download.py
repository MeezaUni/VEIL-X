"""Archive safety checks, safe extraction and the segmented downloader (all synthetic)."""

from __future__ import annotations

import io
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest

from veilx.data import download as dl
from veilx.data.archive import inspect_zip, safe_extract, sha256_file
from veilx.errors import DataAccessError

from .helpers import noise_png

PATTERN = re.compile(r"\d{1,6}/\d{1,3}\.png")


def make_zip(path: Path, members: dict[str, bytes], *, symlink: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "target")
    return path


def good_members() -> dict[str, bytes]:
    return {f"{i}/{k}.png": noise_png(i * 10 + k) for i in (0, 1) for k in range(3)}


def test_clean_archive_extracts_wrapped(tmp_path: Path) -> None:
    z = make_zip(tmp_path / "part.zip", good_members())
    rep = inspect_zip(z, expected_size=z.stat().st_size, member_pattern=PATTERN)
    assert rep.rejected == [] and rep.crc_ok and rep.n_files == 6
    assert rep.sha256 == sha256_file(z) and rep.size_matches
    out = safe_extract(z, tmp_path / "x", rep, wrap_dir="part")
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png"))[0] == "0/0.png"
    assert not any(p.name.startswith(".tmp_") for p in (tmp_path / "x").iterdir())
    with pytest.raises(DataAccessError, match="already exists"):
        safe_extract(z, tmp_path / "x", rep, wrap_dir="part")


@pytest.mark.parametrize(
    ("name", "fragment"),
    [
        ("../evil.png", "traversal"),
        ("/abs.png", "absolute"),
        ("0/notes.txt", "file type"),
        ("a\\b.png", "backslash"),
        ("x/y/z.png", "pattern"),
    ],
)
def test_malicious_members_rejected(tmp_path: Path, name: str, fragment: str) -> None:
    z = make_zip(tmp_path / "bad.zip", {**good_members(), name: noise_png(1)})
    rep = inspect_zip(z, member_pattern=PATTERN)
    # zipfile normalises "\\" to "/" on Windows; such names are then caught by the pattern check
    expected = (fragment, "pattern") if fragment == "backslash" else (fragment,)
    assert any(f in r for r in rep.rejected for f in expected), rep.rejected
    with pytest.raises(DataAccessError):
        safe_extract(z, tmp_path / "x", rep, wrap_dir="bad")
    assert not (tmp_path / "x" / "bad").exists()


def test_symlink_and_size_mismatch_rejected(tmp_path: Path) -> None:
    z = make_zip(tmp_path / "s.zip", good_members(), symlink="0/link.png")
    assert any("symlink" in r for r in inspect_zip(z).rejected)
    z2 = make_zip(tmp_path / "ok.zip", good_members())
    rep = inspect_zip(z2, expected_size=123)
    assert rep.size_matches is False
    with pytest.raises(DataAccessError, match="Content-Length"):
        safe_extract(z2, tmp_path / "x", rep, wrap_dir="ok")


def test_corrupted_member_fails_crc(tmp_path: Path) -> None:
    z = make_zip(tmp_path / "c.zip", good_members())
    raw = bytearray(z.read_bytes())
    data = noise_png(0)
    pos = raw.find(data[20:60])  # stored member bytes (ZIP_STORED default)
    raw[pos + 5] ^= 0xFF
    z.write_bytes(bytes(raw))
    rep = inspect_zip(z)
    assert rep.crc_ok is False and any("CRC" in r for r in rep.rejected)


def test_compression_bomb_guard(tmp_path: Path) -> None:
    z = tmp_path / "bomb.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0/0.png", b"\x00" * 5_000_000)
    rep = inspect_zip(z, max_ratio=200.0)
    assert any("ratio" in r for r in rep.rejected)


# --------------------------------------------------------------------------- downloader


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, status: int, headers: dict[str, str]) -> None:
        super().__init__(data)
        self.status = status
        self.headers = headers

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()


def fake_server(payload: bytes, *, fail_first: int = 0, bad_range: bool = False) -> Any:
    calls = {"n": 0}

    def urlopen(req: Any, timeout: float) -> FakeResponse:
        calls["n"] += 1
        if calls["n"] <= fail_first:
            raise OSError("transient")
        start, end = map(int, req.headers["Range"].removeprefix("bytes=").split("-"))
        cr = f"bytes {start}-{end + (1 if bad_range else 0)}/{len(payload)}"
        return FakeResponse(payload[start : end + 1], 206, {"Content-Range": cr})

    return urlopen


def test_parallel_download_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = bytes(range(256)) * 1000
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_server(payload, fail_first=2))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    res = dl.parallel_download(
        "https://example.org/f.zip",
        tmp_path / "f.zip",
        len(payload),
        workers=4,
        segment_bytes=10_000,
    )
    assert (tmp_path / "f.zip").read_bytes() == payload and res.segments == 26
    assert not (tmp_path / "f.zip.segments.json").exists()
    again = dl.parallel_download("https://example.org/f.zip", tmp_path / "f.zip", len(payload))
    assert again.segments == 0  # already complete


def test_download_rejects_bad_range_and_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x" * 5000
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_server(payload, bad_range=True))
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    with pytest.raises(DataAccessError, match="Content-Range"):
        dl.parallel_download(
            "https://e.org/f",
            tmp_path / "f",
            len(payload),
            workers=1,
            segment_bytes=1000,
            retries=1,
        )
    with pytest.raises(DataAccessError, match="https"):
        dl.parallel_download("http://e.org/f", tmp_path / "g", 10)
