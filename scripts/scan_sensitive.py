"""Pre-commit privacy scan for the VEIL-X repository.

Checks files that are staged, or untracked-but-not-ignored, and refuses anything that could
carry biometric data, personal documents, weights, archives or secrets. Run before every
commit:

    python scripts/scan_sensitive.py          # exit 0 = clean, 1 = findings, 2 = unsafe repo

It also verifies that git's top level is this project folder (never a parent repository).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_000_000

FORBIDDEN_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".npy",
    ".npz",
    ".parquet",
    ".jsonl",
    ".h5",
    ".hdf5",
    ".lmdb",
    ".rec",
    ".idx",
    ".pkl",
    ".pickle",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".rar",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".safetensors",
    ".bin",
    ".engine",
    ".tflite",
    ".pb",
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".lnk",
    ".url",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\..*)?|kaggle\.json|credentials.*|secrets.*)$", re.I)
ALLOWED_PREFIXES = ("tests/fixtures/synthetic/",)
TEXT_PATTERNS = {
    "absolute user path": re.compile(
        r"[A-Za-z]:\\\\?Users\\\\?[^\\\s\"']+|/home/[a-z_][a-z0-9_-]*/", re.I
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "token-like secret": re.compile(r"\b(?:ghp|gho|github_pat|hf)_[A-Za-z0-9_]{20,}\b"),
    "aws key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".cfg", ".ini", ""}
# Files that legitimately document the patterns above.
PATTERN_EXEMPT = {"scripts/scan_sensitive.py", "tests/test_scan_sensitive.py"}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def candidate_files() -> list[str]:
    staged = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR").splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({p for p in staged + untracked if p})


def check_file(rel: str, root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    if rel.startswith(ALLOWED_PREFIXES):
        return findings
    path = root / rel
    if Path(rel).suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append(f"{rel}: forbidden file type")
    if FORBIDDEN_NAMES.search(rel):
        findings.append(f"{rel}: forbidden file name")
    if path.is_file():
        size = path.stat().st_size
        if size > MAX_BYTES:
            findings.append(f"{rel}: {size:,} bytes exceeds {MAX_BYTES:,}")
        if Path(rel).suffix.lower() in TEXT_SUFFIXES and rel not in PATTERN_EXEMPT:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in TEXT_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{rel}: contains {label}")
    return findings


def main() -> int:
    toplevel = Path(_git("rev-parse", "--show-toplevel").strip()).resolve()
    if toplevel != ROOT.resolve():
        print(f"UNSAFE: git top level is not the project folder ({toplevel.name})", file=sys.stderr)
        return 2
    files = candidate_files()
    findings = [f for rel in files for f in check_file(rel)]
    print(f"scanned {len(files)} candidate file(s)")
    for f in findings:
        print(f"  FINDING {f}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
