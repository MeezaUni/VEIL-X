"""The privacy scanner and .gitignore must block biometric and personal artefacts."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from .conftest import ROOT

pytestmark = pytest.mark.structural


def _load_scanner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "scan_sensitive", ROOT / "scripts" / "scan_sensitive.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCAN = _load_scanner()


@pytest.mark.parametrize(
    "name",
    [
        "face.jpg",
        "x/emb.npy",
        "w/model.onnx",
        "doc/passport.pdf",
        ".env",
        "kaggle.json",
        "data.zip",
        "rof.pkl",
        "id.docx",
    ],
)
def test_scanner_flags_forbidden(tmp_path: Path, name: str) -> None:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    assert SCAN.check_file(name, root=tmp_path)


def test_scanner_flags_absolute_user_paths_and_tokens(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("see C:\\Users\\someone\\Desktop\\x", encoding="utf-8")
    (tmp_path / "b.py").write_text("t = 'ghp_" + "a" * 30 + "'", encoding="utf-8")
    assert SCAN.check_file("a.md", root=tmp_path)
    assert SCAN.check_file("b.py", root=tmp_path)


def test_scanner_allows_clean_text(tmp_path: Path) -> None:
    (tmp_path / "ok.md").write_text("# clean\nrelative/path.txt\n", encoding="utf-8")
    assert SCAN.check_file("ok.md", root=tmp_path) == []


@pytest.mark.parametrize(
    "path",
    [
        "data/x.jpg",
        "data/manifest.jsonl",
        "pairs/cal_pairs.jsonl",
        "src/veilx/data/sample.png",
        "cache/e.npy",
        "weights/a.onnx",
        ".env",
        "results/r.json",
        "personal.pdf",
        "kaggle.json",
        ".venv/x",
        ".remember/now.md",
        "some/face.png",
    ],
)
def test_gitignore_blocks(path: str) -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, capture_output=True
    )
    assert res.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize(
    "path",
    [
        "src/veilx/config/schema.py",
        "src/veilx/data/manifest.py",
        "src/veilx/data/splits.py",
        "src/veilx/data/__init__.py",
        "docs/decision_log.md",
        "configs/research_protocol.yaml",
        "README.md",
    ],
)
def test_gitignore_keeps_project_files(path: str) -> None:
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, capture_output=True
    )
    assert res.returncode == 1, f"{path} is unexpectedly ignored"


def test_no_project_source_file_is_ignored() -> None:
    """Every code/config/doc file must be trackable (guards against over-broad ignore rules)."""
    candidates = [
        p.relative_to(ROOT).as_posix()
        for d in ("src", "tests", "scripts", "configs", "docs", "reports")
        for p in (ROOT / d).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in {".py", ".md", ".yaml", ".yml", ".toml", ".json"}
    ]
    assert candidates
    res = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=ROOT,
        input="\n".join(candidates),
        capture_output=True,
        text=True,
    )
    assert res.stdout.strip() == "", f"ignored project files:\n{res.stdout}"


def test_repository_root_is_project_folder() -> None:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert Path(top).resolve() == ROOT.resolve()
