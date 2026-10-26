"""End-to-end run of scripts/prepare_digiface.py on a synthetic DigiFace-shaped fixture.

The fixture uses smooth random-noise images (no faces), the official directory layout and a
planted cross-identity duplicate. Verifies stage chaining, leakage/isolation invariants and that
the public summary contains no identifiers.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from veilx.data.images import encode_png_rgb
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl

from .conftest import ROOT


def smooth(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (7, 7, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((112, 112), Image.Resampling.BICUBIC))


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    data = tmp_path_factory.mktemp("veilx_data")
    public = tmp_path_factory.mktemp("public")
    five = data / "extracted" / "subjects_100000-133332_5_imgs"
    seventy_two = data / "extracted" / "subjects_0-1999_72_imgs"
    for ident in range(100_000, 100_120):
        d = five / str(ident)
        d.mkdir(parents=True)
        for k in range(5):
            (d / f"{k}.png").write_bytes(encode_png_rgb(smooth(ident * 10 + k)))
    for ident in range(3):
        d = seventy_two / str(ident)
        d.mkdir(parents=True)
        for k in range(72):
            (d / f"{k}.png").write_bytes(encode_png_rgb(smooth(10_000_000 + ident * 100 + k)))
    # planted duplicate: identity 100001 image 4 repeats identity 100000 image 0 (label conflict)
    (five / "100001" / "4.png").write_bytes(encode_png_rgb(smooth(100_000 * 10)))
    env = {
        **os.environ,
        "VEILX_DATA_ROOT": str(data),
        "VEILX_PUBLIC_REPORTS": str(public),
        "VEILX_FAR_IMPOSTORS": "300",
        "VEILX_DEDUP_N_TRAIN": "60",
        "VEILX_DEDUP_N_VAL": "40",
        "PYTHONUNBUFFERED": "1",
    }
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_digiface.py"),
            "scan",
            "fingerprint",
            "dedup",
            "pairs",
            "occlusion",
            "summary",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return data, public


def test_final_manifest_is_identity_disjoint_and_deduplicated(prepared: tuple[Path, Path]) -> None:
    data, _ = prepared
    final = list(read_jsonl(data / "manifests" / "digiface_final.jsonl", ImageRecord))
    splits = defaultdict(set)
    for r in final:
        splits[r.identity_id].add(r.split)
        assert r.pixel_sha256 and r.source_group
    assert all(len(s) == 1 for s in splits.values())
    ids = {r.image_id for r in final}
    # the duplicated picture is ambiguous (two identities): both copies are removed
    assert "digiface1m:100000:0" not in ids and "digiface1m:100001:4" not in ids
    assert len(final) == 120 * 5 + 3 * 72 - 2


def test_pairs_and_calibration_independence(prepared: tuple[Path, Path]) -> None:
    data, _ = prepared
    final = {
        r.image_id: r for r in read_jsonl(data / "manifests" / "digiface_final.jsonl", ImageRecord)
    }
    pdir = data / "manifests" / "pairs"
    for tag in ("train", "val", "test_balanced", "test_far", "cal_independent"):
        pairs = list(read_jsonl(pdir / f"{tag}.jsonl", PairRecord))
        assert pairs, tag
        for p in pairs:
            a, b = final[p.image_a], final[p.image_b]
            assert a.split is b.split is p.split
            assert (a.identity_id == b.identity_id) == (p.label == "genuine")
    cal = list(read_jsonl(pdir / "cal_independent.jsonl", PairRecord))
    used = [final[x].identity_id for p in cal for x in {p.image_a, p.image_b}]
    seen_ids = [final[p.image_a].identity_id for p in cal if p.label == "genuine"]
    seen_ids += [
        i
        for p in cal
        if p.label == "impostor"
        for i in (final[p.image_a].identity_id, final[p.image_b].identity_id)
    ]
    assert len(seen_ids) == len(set(seen_ids)), "calibration identities reused"
    assert len(used) >= len(cal)
    prov = json.loads((pdir / "cal_independent.provenance.json").read_text(encoding="utf-8"))
    assert prov["summary"]["identity_disjoint"] is True


def test_occlusion_plans_respect_heldout_isolation(prepared: tuple[Path, Path]) -> None:
    data, _ = prepared
    odir = data / "manifests" / "occlusion"
    files = sorted(odir.glob("F*__*.jsonl.gz"))
    assert len(files) == 5 * 7
    for f in files:
        heldout, role, _ = f.name.removesuffix(".jsonl.gz").split("__")
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                fams = {s["family"] for s in (row["a"], row["b"]) if s}
                if role == "test_ood":
                    assert not fams or heldout in fams
                else:
                    assert heldout not in fams
                for s in (row["a"], row["b"]):
                    if s:
                        assert 0.0 <= s["visibility_ratio"] <= 1.0 and s["recipe_sha256"]
                        assert s["physical"] is (s["family"] != "F5_extreme_lighting")
    assert (odir / "stress__test_balanced.jsonl.gz").exists()


def test_public_summary_contains_no_identifiers(prepared: tuple[Path, Path]) -> None:
    _, public = prepared
    text = (public / "digiface_phase1b_summary.json").read_text(encoding="utf-8")
    summary = json.loads(text)
    assert summary["final_manifest"]["images_per_part"]
    assert "digiface1m:" not in text and "relpath" not in text and "pair_id" not in text
    assert "subjects_100000" in text  # part names are public metadata, ids are not
