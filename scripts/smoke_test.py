# ruff: noqa: E501
"""Reproducibility smoke test (Phase 6, 6G). Needs NO biometric data.

Level 1 (any environment, no data root):
- the package imports;
- protocol and app configs load and validate;
- the A3 decision rule reproduces MATCH / NON-MATCH / ABSTAIN on a synthetic rule;
- the input validator rejects corrupt and unsupported files;
- the server refuses non-loopback hosts;
- a synthetic LTT calibration on simulated scores is deterministic for a fixed seed.

Level 2 (only with ``VEILX_DATA_ROOT`` and the GPU environment; still no face images):
- every frozen artefact is SHA-256 verified (backbones, SCRFD, VE-C weights, the Phase 3 / 4
  freezes);
- the full pipeline runs twice on two **synthetic noise images** in pre-aligned mode, with
  identical outputs (repeatability) and a valid three-way label.

    python scripts/smoke_test.py            # level 1 (+ level 2 if data root and torch exist)
    python scripts/smoke_test.py --level 1
Exit code 0 = all executed checks passed.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    def deco(fn: Callable[[], str]) -> Callable[[], str]:
        try:
            msg = fn()
            RESULTS.append((name, True, msg))
        except Exception as exc:  # report and continue
            RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        return fn

    return deco


def level1() -> None:
    @check("imports")
    def _() -> str:
        import veilx  # noqa: F401
        from veilx.app import config, decision, inputs, report, server  # noqa: F401

        return "veilx and veilx.app (model-free modules) import"

    @check("configs load")
    def _() -> str:
        from veilx.app.config import load_app_config
        from veilx.config import load_protocol

        p = load_protocol(ROOT / "configs" / "research_protocol.yaml")
        c = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
        assert c.server.host == "127.0.0.1" and c.research_only
        return f"protocol {p.protocol_version}; app default system {c.decision.default_fold}"

    @check("three-way decision rule")
    def _() -> str:
        from veilx.app.config import FrozenRule
        from veilx.app.decision import decide
        from veilx.fusion.logreg import LogisticModel

        m = LogisticModel(("s", "q", "s*q"), 10.0)
        m.mean, m.scale = np.zeros(3), np.ones(3)
        m.coef, m.intercept = np.array([8.0, 0.0, 0.0]), -2.4  # p = sigmoid(8 s - 2.4)
        rules: dict[str, tuple[float, float] | None] = {
            "q_low": (0.2, 0.8),
            "q_mid": (0.5, 0.5),
            "q_high": (0.5, 0.5),
        }
        rule = FrozenRule("SMOKE", m, (10.0, 20.0), rules, 0.5)
        labels = {decide(rule, s, 5.0).label for s in (0.0, 0.3, 0.6)} | {
            decide(rule, 0.6, 25.0).label
        }
        assert labels == {"MATCH", "NON-MATCH", "ABSTAIN"}, labels
        return "MATCH, NON-MATCH and ABSTAIN all produced"

    @check("input validation")
    def _() -> str:
        from PIL import Image

        from veilx.app.config import load_app_config
        from veilx.app.inputs import InputRejectedError, load_image_bytes

        lim = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml").inputs
        buf = io.BytesIO()
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (128, 128, 3), np.uint8)).save(
            buf, "PNG"
        )
        ok = load_image_bytes(buf.getvalue(), lim)
        codes = []
        for bad in (b"", b"not an image", buf.getvalue()[:200]):
            try:
                load_image_bytes(bad, lim)
            except InputRejectedError as exc:
                codes.append(exc.code)
        assert ok.rgb.shape == (128, 128, 3) and len(codes) == 3
        return f"valid PNG accepted; rejected: {', '.join(codes)}"

    @check("local-only server policy")
    def _() -> str:
        from veilx.app.server import local_only_headers

        h = local_only_headers("127.0.0.1", 7860)
        try:
            local_only_headers("0.0.0.0", 7860)  # noqa: S104  (must be refused)
            raise AssertionError("non-loopback host accepted")
        except ValueError:
            pass
        assert "default-src 'self'" in h["Content-Security-Policy"]
        return "loopback only; CSP default-src 'self'"

    @check("deterministic LTT calibration (simulated scores)")
    def _() -> str:
        from veilx.selective.ltt import LTTCalibrator, symmetric_rules

        def run() -> tuple[float, float] | None:
            rng = np.random.default_rng(20261102)
            y = rng.integers(0, 2, 4000)
            p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.18, 4000), 0, 1)
            cal = LTTCalibrator(
                symmetric_rules(0.5, np.linspace(0.45, 0.0, 46)),
                target_fm=0.05,
                target_fnm=0.05,
                delta=0.05,
                require_valid=False,  # simulated scores: no PairSet, so no guarantee is claimed
            )
            r = cal.fit(p, y, pair_set=None)
            return None if r.rule is None else (r.rule.tau_low, r.rule.tau_high)

        a, b = run(), run()
        assert a is not None and a == b
        return f"identical thresholds on rerun: ({a[0]:.3f}, {a[1]:.3f}); guarantee not claimed (simulated)"


def level2() -> None:
    data = Path(os.environ["VEILX_DATA_ROOT"])

    @check("frozen artefact hashes")
    def _() -> str:
        import json

        from veilx.app.config import content_sha256, load_app_config
        from veilx.data.archive import sha256_file
        from veilx.models.spec import BACKBONES

        cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
        n = 0
        assert sha256_file(data / cfg.detector.model) == cfg.detector.sha256
        n += 1
        for fold, sha in cfg.diagnostics.vec_sha256.items():
            assert sha256_file(data / cfg.diagnostics.vec_dir / fold / "best.safetensors") == sha, (
                fold
            )
            n += 1
        for spec in BACKBONES.values():
            path = next(
                (data / "models").glob(
                    f"*{spec.backbone_id.split('_')[0]}_ir101_webface4m/*/{spec.weights_file}"
                )
            )
            assert sha256_file(path) == spec.weights_sha256, spec.backbone_id
            n += 1
        for f, sha in (
            (cfg.decision.phase3_frozen_file, cfg.decision.phase3_frozen_sha256),
            (cfg.diagnostics.phase4_frozen_file, cfg.diagnostics.phase4_frozen_sha256),
        ):
            d = json.loads((data / f).read_text("utf-8"))
            assert content_sha256({k: v for k, v in d.items() if k != "sha256"}) == sha, f
            n += 1
        return f"{n} artefacts verified"

    @check("pipeline repeatability on synthetic noise (pre-aligned)")
    def _() -> str:
        from veilx.app.config import load_app_config
        from veilx.app.pipeline import VerificationPipeline

        cfg = load_app_config(ROOT / "configs" / "app" / "veilx_app.yaml")
        pipe = VerificationPipeline(cfg, data)
        rng = np.random.default_rng(7)
        a = rng.integers(0, 255, (112, 112, 3), np.uint8)
        b = rng.integers(0, 255, (112, 112, 3), np.uint8)
        r1 = pipe.verify(a, b, pre_aligned=True, with_diagnostics=True)
        r2 = pipe.verify(a, b, pre_aligned=True, with_diagnostics=True)
        assert r1.decision.label in ("MATCH", "NON-MATCH", "ABSTAIN")
        assert (
            r1.decision.code == r2.decision.code
            and r1.decision.fused_score == r2.decision.fused_score
        )
        return f"device {pipe.device}; label {r1.decision.label}; score {r1.decision.fused_score:.4f} (identical on rerun)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--level", type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    level1()
    ran2 = False
    if args.level == 2 and os.environ.get("VEILX_DATA_ROOT"):
        try:
            import torch  # noqa: F401

            level2()
            ran2 = True
        except ImportError:
            pass
    for name, ok, msg in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    if not ran2:
        print("[SKIP] level 2 (needs VEILX_DATA_ROOT and the GPU environment with torch)")
    return 0 if all(ok for _, ok, _ in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
