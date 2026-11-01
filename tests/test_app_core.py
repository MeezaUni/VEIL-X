"""Phase 5 research app: config, inputs, frozen decision, faces, isolation, privacy (no models)."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from veilx.app.config import (
    AppConfig,
    FrozenRule,
    content_sha256,
    load_app_config,
    load_frozen_decision,
)
from veilx.app.decision import decide, rule_description
from veilx.app.face import FacePreparer
from veilx.app.inputs import InputRejectedError, load_image_bytes, validate_array
from veilx.app.report import (
    DISCLAIMER,
    LIMITATIONS_MD,
    WHY_NOT_VISIBILITY_MD,
    abstention_text,
    decision_html,
)
from veilx.app.server import csp, local_only_headers
from veilx.app.tempfiles import PID_FILE, cleanup_stale, create_session_dir
from veilx.fusion.logreg import LogisticModel

from .conftest import ROOT

CFG_PATH = ROOT / "configs" / "app" / "veilx_app.yaml"


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_app_config(CFG_PATH)


def _png(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def _rule() -> FrozenRule:
    m = LogisticModel(("s", "q", "s*q"), 10.0)
    m.mean, m.scale = np.zeros(3), np.ones(3)
    m.coef, m.intercept = np.array([10.0, 0.0, 0.0]), -3.0  # p = sigmoid(10 s - 3)
    rules = {"q_low": (0.2, 0.8), "q_mid": (0.5, 0.5), "q_high": (0.5, 0.5)}
    return FrozenRule("F1_mask", m, (19.0, 22.0), rules, 0.5)  # type: ignore[arg-type]


# ----------------------------------------------------------------- configuration


def test_repo_config_is_frozen_local_and_research_only(cfg: AppConfig) -> None:
    assert cfg.research_only and cfg.server.host == "127.0.0.1"
    assert cfg.server.share is False and cfg.server.analytics is False
    assert cfg.decision.arm == "A3_quality" and cfg.decision.fused_score == "M1"
    assert cfg.decision.default_fold in cfg.decision.folds


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("server", "host"), "0.0.0.0"),
        (("server", "share"), True),
        (("server", "analytics"), True),
        (("research_only",), False),
        (("decision", "default_fold"), "F9_none"),
    ],
)
def test_config_rejects_non_local_or_unfrozen_settings(
    path: tuple[str, ...], value: object
) -> None:
    raw = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    d = raw
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    with pytest.raises(ValueError):
        AppConfig.model_validate(raw)


def test_frozen_decision_sha_is_enforced(tmp_path: Path) -> None:
    m = _rule().model.to_dict()
    body = {
        "folds": {
            "F1_mask": {
                "models": {"M1": m},
                "edges": {"q": [19.0, 22.0]},
                "h4_thresholds": {"M1": 0.5},
                "arms": {
                    "A3_quality|primary|v2": {
                        "model": "M1",
                        "strata": "quality",
                        "rules": {"q_low": [0.2, 0.8], "q_mid": [0.5, 0.5], "q_high": None},
                    }
                },
            }
        }
    }
    body["sha256"] = content_sha256(body)
    p = tmp_path / "frozen.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    r = load_frozen_decision(p, body["sha256"], "F1_mask")
    assert r.rules["q_high"] is None and r.q_edges == (19.0, 22.0)
    body["folds"]["F1_mask"]["edges"]["q"] = [10.0, 22.0]  # tamper
    p.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        load_frozen_decision(p, body["sha256"], "F1_mask")


# ----------------------------------------------------------------- input validation


def test_valid_image_and_conversions(cfg: AppConfig) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 100, 3), dtype=np.uint8)
    v = load_image_bytes(_png(img), cfg.inputs)
    assert v.rgb.shape == (120, 100, 3) and v.source_format == "PNG"
    grey = load_image_bytes(_png(img[..., 0]), cfg.inputs)
    assert grey.rgb.shape == (120, 100, 3) and any("RGB" in n for n in grey.notes)
    f = validate_array(rng.random((80, 80, 3)), cfg.inputs)
    assert f.rgb.dtype == np.uint8


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "empty_file"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "corrupt_or_unreadable"),
        (b"not an image at all", "corrupt_or_unreadable"),
    ],
)
def test_invalid_files_are_rejected_with_codes(cfg: AppConfig, data: bytes, code: str) -> None:
    with pytest.raises(InputRejectedError) as e:
        load_image_bytes(data, cfg.inputs)
    assert e.value.code == code


def test_truncated_unsupported_small_blank(cfg: AppConfig) -> None:
    rng = np.random.default_rng(1)
    good = _png(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
    for data, code in (
        (good[: len(good) // 2], "corrupt_or_unreadable"),
        (_png(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8), "GIF"), "unsupported_format"),
        (_png(rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)), "image_too_small"),
        (_png(np.full((100, 100, 3), 128, np.uint8)), "blank_image"),
    ):
        with pytest.raises(InputRejectedError) as e:
            load_image_bytes(data, cfg.inputs)
        assert e.value.code == code


# ----------------------------------------------------------------- frozen decision


def test_all_three_decision_states() -> None:
    r = _rule()
    assert decide(r, 0.6, 25.0).label == "MATCH"
    assert decide(r, 0.1, 25.0).label == "NON-MATCH"
    ab = decide(r, 0.3, 15.0)  # q_low, p = sigmoid(0) = 0.5 inside [0.2, 0.8)
    assert ab.label == "ABSTAIN" and "abstention band" in ab.reasons[0] and ab.stratum == "q_low"
    forced = decide(r, 0.9, 25.0, forced_reasons=("image A: multiple faces",))
    assert forced.label == "ABSTAIN" and forced.reasons == ("image A: multiple faces",)
    rows = rule_description(r)
    assert "no abstention band" in rows[1]["rule"] and "else ABSTAIN" in rows[0]["rule"]


def test_decision_matches_offline_phase3_decide() -> None:
    spec = importlib.util.spec_from_file_location(
        "evaluate_phase3", ROOT / "scripts" / "evaluate_phase3.py"
    )
    assert spec is not None and spec.loader is not None
    ev3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev3)
    r = _rule()
    rng = np.random.default_rng(2)
    s, q = rng.uniform(-0.2, 1.0, 500), rng.uniform(12, 28, 500)
    t = {"s": s, "q": q}
    p = r.model.predict_proba(np.column_stack([s, q, s * q]))
    rec = {"rules": {k: list(v) for k, v in r.rules.items()}}
    off = ev3.decide(rec, p, ev3.strata_labels("quality", t, {"q": list(r.q_edges)}, "cv_region"))
    app = np.array([decide(r, float(a), float(b)).code for a, b in zip(s, q, strict=True)])
    assert np.array_equal(off, app)


# ----------------------------------------------------------------- face handling (fake detector)


class _FakeDet:
    def __init__(self, dets: list[Any]) -> None:
        self.dets = dets

    def detect(self, img: np.ndarray) -> list[Any]:
        return self.dets


def _det(score: float, shift: float = 0.0) -> Any:
    from veilx.detect.align import TEMPLATE

    lm = (TEMPLATE * 1.5 + 40 + shift).astype(np.float32)
    return SimpleNamespace(
        box=np.array([20 + shift, 20, 150 + shift, 180], np.float32), score=score, landmarks=lm
    )


def _prep(cfg: AppConfig, dets: list[Any]) -> FacePreparer:
    fp = FacePreparer(cfg, Path())
    fp._det = _FakeDet(dets)
    return fp


def test_face_statuses(cfg: AppConfig) -> None:
    img = np.random.default_rng(3).integers(0, 255, (250, 250, 3), dtype=np.uint8)
    none = _prep(cfg, []).prepare(img)
    assert (
        none.status == "fallback_fixed_crop"
        and none.is_fallback
        and none.crop.shape == (112, 112, 3)
    )  # type: ignore[union-attr]
    one = _prep(cfg, [_det(0.9)]).prepare(img)
    assert one.status == "aligned" and not one.is_fallback and one.crop is not None
    low = _prep(cfg, [_det(0.4)]).prepare(img)
    assert low.status == "fallback_fixed_crop" and low.is_fallback
    two = _prep(cfg, [_det(0.9), _det(0.9, 60)])
    refused = two.prepare(img)
    assert refused.status == "multiple_faces" and refused.crop is None and not refused.usable
    chosen = two.prepare(img, select=1)
    assert chosen.status == "selected_face" and chosen.chosen == 1
    assert two.prepare(img, select=5).status == "rejected"


def test_pre_aligned_mode(cfg: AppConfig) -> None:
    fp = _prep(cfg, [])
    ok = fp.prepare(np.zeros((112, 112, 3), np.uint8) + 7, pre_aligned=True)
    assert ok.status == "pre_aligned" and ok.crop is not None
    assert fp.prepare(np.zeros((120, 112, 3), np.uint8), pre_aligned=True).status == "rejected"


# ----------------------------------------------------------------- pipeline isolation (fake models)


class _FakeBackbone:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def embed(self, crops: np.ndarray) -> Any:
        rng = np.random.default_rng(self.seed + int(crops.sum()) % 1000)
        e = rng.normal(size=(len(crops), 8))
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return SimpleNamespace(
            embeddings=e.astype(np.float32), feature_norms=np.array([18.0, 25.0], np.float32)
        )


class _BadDiagnostics:
    """Diagnostics that return extreme values; must never change the decision."""

    def run(self, crops: np.ndarray, fold: str) -> dict[str, Any]:
        return {"visibility_maps": np.zeros((2, 28, 28)), "pair_novel": True}


def _pipeline(cfg: AppConfig, dets: list[Any]) -> Any:
    from veilx.app.pipeline import VerificationPipeline

    p = VerificationPipeline(cfg, Path(), device="cpu")
    p.faces._det = _FakeDet(dets)
    p._backbones = {
        "adaface_ir101_webface4m": _FakeBackbone(1),
        "arcface_ir101_webface4m": _FakeBackbone(2),
    }
    p._rules = {f: _rule() for f in cfg.decision.folds}
    p._diag = _BadDiagnostics()
    return p


def test_diagnostics_cannot_change_decision(cfg: AppConfig) -> None:
    img = np.random.default_rng(4).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    p = _pipeline(cfg, [])
    with_d = p.verify(img, img.copy(), pre_aligned=True, with_diagnostics=True)
    without = p.verify(img, img.copy(), pre_aligned=True, with_diagnostics=False)
    assert (
        with_d.decision == without.decision
        and with_d.diagnostics is not None
        and without.diagnostics is None
    )


def test_multiple_faces_abstain_with_reason_and_fallback_is_flagged(cfg: AppConfig) -> None:
    img = np.random.default_rng(5).integers(0, 255, (250, 250, 3), dtype=np.uint8)
    multi = _pipeline(cfg, [_det(0.9), _det(0.9, 60)]).verify(img, img)
    assert multi.decision.label == "ABSTAIN" and "multiple faces" in multi.decision.reasons[0]
    fb = _pipeline(cfg, []).verify(img, img, with_diagnostics=False)
    assert fb.face_a.is_fallback and any("FALLBACK" in n for n in fb.notes)


# ----------------------------------------------------------------- wording, privacy, local-only


def test_disclosures_and_wording() -> None:
    text = (DISCLAIMER + LIMITATIONS_MD + WHY_NOT_VISIBILITY_MD).lower()
    for must in (
        "5 % selective-error target was not maintained",
        "not because it is universally safe",
        "no formal out-of-distribution risk guarantee",
        "have not been independently validated",
        "do not establish deployment readiness",
        "not a probability of identity",
        "research",
    ):
        assert must in text, must
    html = decision_html("MATCH", 0.9, "q_high").lower()
    assert (
        "green" not in html
        and "#0f0" not in html
        and "safe" not in html
        and "certified" not in html.replace("not certified", "")
    )
    assert "not a probability of identity" in html
    assert "not a causal explanation" in abstention_text(("x",), "ABSTAIN")


def test_server_is_loopback_only_with_strict_csp() -> None:
    with pytest.raises(ValueError):
        local_only_headers("0.0.0.0", 7860)
    h = local_only_headers("127.0.0.1", 7860)
    assert h["Cache-Control"] == "no-store" and h["Referrer-Policy"] == "no-referrer"
    policy = csp("127.0.0.1", 7860)
    assert "default-src 'self'" in policy and "https:" not in policy and "cdnjs" not in policy
    launcher = (ROOT / "scripts" / "run_app.py").read_text(encoding="utf-8")
    assert "build_asgi" in launcher and "share=True" not in launcher and "0.0.0.0" not in launcher


def test_temp_dirs_and_upload_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import veilx.app.tempfiles as tf
    from veilx.app.ui import read_and_discard

    mine = create_session_dir(tmp_path)
    assert (mine / PID_FILE).read_text(encoding="utf-8") == str(os.getpid())
    dead = tmp_path / "veilx_app_dead"
    dead.mkdir()
    (dead / PID_FILE).write_text("999999", encoding="utf-8")
    unknown = tmp_path / "veilx_app_unknown"
    unknown.mkdir()
    monkeypatch.setattr(tf, "_alive", lambda pid: False)
    removed = cleanup_stale(tmp_path)
    assert dead in removed and not dead.exists() and mine.exists() and unknown.exists()
    up = mine / "upload.png"
    up.write_bytes(b"abc")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"xyz")
    assert read_and_discard(str(up), mine) == b"abc" and not up.exists()  # deleted after reading
    assert (
        read_and_discard(str(outside), mine) == b"xyz" and outside.exists()
    )  # never touches other files


def test_comparison_requires_authorisation_and_discards_uploads(
    cfg: AppConfig, tmp_path: Path
) -> None:
    from veilx.app.ui import CONSENT_REQUIRED, run_comparison

    p = _pipeline(cfg, [])
    img = np.random.default_rng(5).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    paths = []
    for n in ("a.png", "b.png"):
        Image.fromarray(img).save(tmp_path / n)
        paths.append(str(tmp_path / n))
    out = run_comparison(p, tmp_path, *paths, "F1_mask", True, -1, -1, False, False)
    assert out[0] == CONSENT_REQUIRED and out[1] is None  # refused, no decision
    assert not any(Path(x).exists() for x in paths)  # uploads deleted even when refused
    for n in ("a.png", "b.png"):
        Image.fromarray(img).save(tmp_path / n)
    out = run_comparison(p, tmp_path, *paths, "F1_mask", True, -1, -1, False, True)
    assert any(lab in out[0] for lab in ("MATCH", "NON-MATCH", "ABSTAIN"))
    assert not any(Path(x).exists() for x in paths)


def test_examples_survive_a_comparison(cfg: AppConfig, tmp_path: Path) -> None:
    from veilx.app.ui import disposable_copies, run_comparison

    ex = tmp_path / "examples"
    ex.mkdir()
    img = np.random.default_rng(6).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    srcs = (str(ex / "a.png"), str(ex / "b.png"))
    for s in srcs:
        Image.fromarray(img).save(s)
    a, b = disposable_copies(srcs, tmp_path)
    run_comparison(_pipeline(cfg, []), tmp_path, a, b, "F1_mask", True, -1, -1, False, True)
    assert all(Path(s).exists() for s in srcs) and not Path(a).exists() and not Path(b).exists()
