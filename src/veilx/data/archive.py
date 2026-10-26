"""Archive verification and safe extraction for downloaded dataset parts.

Checks before anything touches the destination:

* streaming SHA-256 of the archive (recorded; DigiFace publishes no checksum, so this is a
  trust-on-first-download fingerprint, never an authenticity proof);
* expected byte size (from the official host's ``Content-Length``) if supplied;
* every member: no absolute paths, drive letters, ``..`` components, backslashes, symlinks or
  special files; only allowed suffixes; a single expected top-level directory;
* zip-bomb guards: total uncompressed size limit and per-member compression-ratio limit;
* CRC-32 of every member (``ZipFile.testzip``).

Extraction goes to a temporary sibling directory that is renamed into place only after all
members were written, so an interrupted extraction never leaves a half-valid dataset.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import stat
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from veilx.errors import DataAccessError

_CHUNK = 8 * 1024 * 1024


@dataclass
class ArchiveReport:
    archive: str
    size_bytes: int
    sha256: str
    expected_size_bytes: int | None
    size_matches: bool | None
    n_members: int = 0
    n_files: int = 0
    n_dirs: int = 0
    uncompressed_bytes: int = 0
    max_compression_ratio: float = 0.0
    crc_ok: bool | None = None
    top_level: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    extracted_to: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _member_problem(info: zipfile.ZipInfo, allowed_suffixes: frozenset[str]) -> str | None:
    name = info.filename
    if "\\" in name:
        return "backslash in member name"
    p = PurePosixPath(name)
    if p.is_absolute() or name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return "absolute path"
    if any(part in ("..", "") for part in p.parts):
        return "path traversal or empty component"
    file_type = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)  # 0 when only perms are stored
    if file_type == stat.S_IFLNK:
        return "symlink"
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        return "special file"
    if not info.is_dir() and p.suffix.lower() not in allowed_suffixes:
        return f"unexpected file type {p.suffix or '(none)'}"
    return None


def inspect_zip(
    archive: Path,
    *,
    expected_size: int | None = None,
    allowed_suffixes: frozenset[str] = frozenset({".png"}),
    expected_top_level: str | None = None,
    member_pattern: re.Pattern[str] | None = None,
    max_total_bytes: int = 64 * 1024**3,
    max_ratio: float = 200.0,
    check_crc: bool = True,
) -> ArchiveReport:
    size = archive.stat().st_size
    report = ArchiveReport(
        archive=archive.name,
        size_bytes=size,
        sha256=sha256_file(archive),
        expected_size_bytes=expected_size,
        size_matches=None if expected_size is None else size == expected_size,
    )
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        report.n_members = len(infos)
        tops: set[str] = set()
        for info in infos:
            problem = _member_problem(info, allowed_suffixes)
            if (
                problem is None
                and member_pattern is not None
                and not info.is_dir()
                and not member_pattern.fullmatch(info.filename)
            ):
                problem = "name does not match the expected member pattern"
            if problem:
                report.rejected.append(f"{info.filename}: {problem}")
                continue
            tops.add(PurePosixPath(info.filename).parts[0])
            if info.is_dir():
                report.n_dirs += 1
                continue
            report.n_files += 1
            report.uncompressed_bytes += info.file_size
            if info.compress_size:
                report.max_compression_ratio = max(
                    report.max_compression_ratio, info.file_size / info.compress_size
                )
        report.top_level = sorted(tops)
        if expected_top_level is not None and report.top_level != [expected_top_level]:
            report.rejected.append(
                f"top-level entries {report.top_level[:5]} != [{expected_top_level}]"
            )
        if report.uncompressed_bytes > max_total_bytes:
            report.rejected.append(f"uncompressed size {report.uncompressed_bytes} exceeds limit")
        if report.max_compression_ratio > max_ratio:
            report.rejected.append(f"compression ratio {report.max_compression_ratio:.0f} too high")
        if check_crc and not report.rejected:
            bad = zf.testzip()
            report.crc_ok = bad is None
            if bad is not None:
                report.rejected.append(f"CRC failure in {bad}")
    return report


def safe_extract(
    archive: Path, dest_dir: Path, report: ArchiveReport, *, wrap_dir: str | None = None
) -> Path:
    """Extract a previously inspected archive; refuses if the inspection found any problem.

    With ``wrap_dir`` every member is placed under ``dest_dir/wrap_dir`` (for archives whose
    members sit at the top level, as DigiFace's do); otherwise the archive must contain a
    single top-level directory. CRC verification must have passed.
    """
    if report.rejected:
        raise DataAccessError(
            f"{archive.name}: refusing to extract ({len(report.rejected)} problems)"
        )
    if report.size_matches is False:
        raise DataAccessError(f"{archive.name}: size does not match the official Content-Length")
    if report.crc_ok is not True:
        raise DataAccessError(f"{archive.name}: CRC verification has not passed")
    if wrap_dir is None and len(report.top_level) != 1:
        raise DataAccessError(f"{archive.name}: expected exactly one top-level directory")
    name = wrap_dir if wrap_dir is not None else report.top_level[0]
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise DataAccessError("invalid destination directory name")
    final = dest_dir / name
    if final.exists():
        raise DataAccessError(f"{final.name} already exists; remove it explicitly to re-extract")
    tmp = dest_dir / f".tmp_{name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    member_root = tmp / name if wrap_dir is not None else tmp
    member_root.mkdir(parents=True)
    root_resolved = member_root.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = (member_root / info.filename).resolve()
            if not target.is_relative_to(root_resolved):  # defence in depth
                raise DataAccessError(f"member escapes destination: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
    (tmp / name).rename(final)
    tmp.rmdir()
    report.extracted_to = final.name
    return final
