"""Documentation link check (Phase 6, 6K).

Every backticked or linked repository path in the Markdown documents (README.md, docs/**) must
exist. Placeholders such as ``<fold>``, globs (``*``) and private-data-root paths are skipped.
URLs are not fetched; they are listed by host for a manual review.

    python scripts/check_docs.py
Exit code 0 = no broken repository path.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("docs/", "scripts/", "reports/", "src/", "configs/", "tests/")
PATH_RE = re.compile(
    r"(?:`|\]\()((?:docs|scripts|reports|src|configs|tests)/[A-Za-z0-9_./\-]+?)(?:`|\)|#|\s|:)"
)
URL_RE = re.compile(r"https?://([A-Za-z0-9.\-]+)")
PRIVATE = ("reports/phase3_frozen.json", "reports/phase4_frozen.json")  # under VEILX_DATA_ROOT


def check(files: list[Path]) -> tuple[list[tuple[str, str]], Counter[str]]:
    broken: list[tuple[str, str]] = []
    hosts: Counter[str] = Counter()
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        hosts.update(URL_RE.findall(text))
        for m in PATH_RE.finditer(text):
            p = m.group(1).rstrip(".,")
            if any(c in p for c in "<>*{}") or p in PRIVATE or p.endswith("/"):
                continue
            if not (ROOT / p).exists():
                broken.append((str(f.relative_to(ROOT)), p))
    return broken, hosts


def main() -> int:
    files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    broken, hosts = check(files)
    print(f"checked {len(files)} Markdown files")
    for f, p in broken:
        print(f"  BROKEN {f}: {p}")
    print(
        "external hosts referenced (not fetched):",
        ", ".join(f"{h} ({n})" for h, n in hosts.most_common()),
    )
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
