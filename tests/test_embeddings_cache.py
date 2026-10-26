"""Embedding shard store and extraction work items (numpy only; synthetic data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from veilx.data.manifest import PairRecord
from veilx.embeddings.jobs import (
    Item,
    clean_items,
    occluded_items,
    occluded_key,
    render,
    stress_items,
)
from veilx.embeddings.store import is_complete, read_meta, read_shard, write_shard
from veilx.errors import ManifestError
from veilx.occlusion.engine import apply_occlusion


def _emb(n: int, d: int = 8) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.default_rng(0).standard_normal((n, d)).astype(np.float32)
    norm = np.linalg.norm(x, axis=1).astype(np.float32)
    return x / norm[:, None], norm


def test_shard_roundtrip_and_meta(tmp_path: Path) -> None:
    emb, norm = _emb(5)
    path = tmp_path / "g.npz"
    write_shard(path, [f"k{i}" for i in range(5)], emb, norm, {"job_sha256": "j", "fp": "a"})
    sh = read_shard(path)
    np.testing.assert_array_equal(sh.emb, emb)
    np.testing.assert_array_equal(sh.norm, norm)
    assert sh.index()["k3"] == 3 and sh.meta["n"] == 5 and sh.meta["dim"] == 8
    assert is_complete(path, {"job_sha256": "j", "fp": "a"})
    assert not is_complete(path, {"job_sha256": "other"})  # stale -> recompute
    assert not list(tmp_path.glob("*.tmp"))


def test_shard_rejects_bad_input_and_tampering(tmp_path: Path) -> None:
    emb, norm = _emb(3)
    with pytest.raises(ManifestError, match="duplicate"):
        write_shard(tmp_path / "d.npz", ["a", "a", "b"], emb, norm, {})
    with pytest.raises(ValueError, match="shape"):
        write_shard(tmp_path / "s.npz", ["a", "b"], emb, norm, {})
    bad = emb.copy()
    bad[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        write_shard(tmp_path / "n.npz", ["a", "b", "c"], bad, norm, {})
    path = tmp_path / "t.npz"
    write_shard(path, ["a", "b", "c"], emb, norm, {})
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip one byte
    path.write_bytes(bytes(raw))
    with pytest.raises(ManifestError, match="SHA-256"):
        read_shard(path)


def test_incomplete_shard_is_not_complete(tmp_path: Path) -> None:
    emb, norm = _emb(2)
    path = tmp_path / "g.npz"
    write_shard(path, ["a", "b"], emb, norm, {"job_sha256": "j"})
    path.with_name("g.meta.json").unlink()  # crash before the completion marker
    assert read_meta(path) is None and not is_complete(path, {"job_sha256": "j"})
    with pytest.raises(ManifestError, match="incomplete"):
        read_shard(path)


def _pair(pid: str, a: str, b: str) -> PairRecord:
    return PairRecord(
        pair_id=pid,
        image_a=a,
        image_b=b,
        label="genuine",
        stratum="clean_clean",
        split="test",
        protocol="veilx",
    )


def test_items_are_deterministic_and_keyed() -> None:
    assert [i.key for i in clean_items(["b", "a", "b"])] == ["a", "b"]
    pairs = {"p1": _pair("p1", "x:1", "x:2"), "p2": _pair("p2", "y:1", "y:2")}
    spec = {"family": "F1_mask", "severity_level": "low", "seed": 7, "recipe_sha256": "r"}
    rows = [{"pair_id": "p1", "a": None, "b": spec}, {"pair_id": "p2", "a": spec, "b": spec}]
    items = occluded_items(rows, pairs)
    assert [i.key for i in items] == [
        occluded_key("x:2", "F1_mask", "low", 7),
        occluded_key("y:1", "F1_mask", "low", 7),
        occluded_key("y:2", "F1_mask", "low", 7),
    ]
    with pytest.raises(ManifestError, match="duplicate"):
        occluded_items([*rows, rows[0]], pairs)
    st = stress_items(
        [
            {
                "pair_id": "p1",
                "steps_a": [["F1_mask", "medium", 1]],
                "steps_b": [["F2_sunglasses", "medium", 2]],
            }
        ],
        pairs,
    )
    assert st[0].key == "x:1|stress|F1_mask:medium:1" and st[1].image_id == "x:2"


def test_render_reproduces_planned_occlusion_and_checks_recipe() -> None:
    img = np.random.default_rng(3).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    planned = apply_occlusion(img, "F2_sunglasses", "medium", 11, source_pixel_sha256="h")
    item = Item(
        key="k",
        image_id="i",
        family="F2_sunglasses",
        level="medium",
        seed=11,
        recipe_sha256=planned.spec.recipe_sha256,
    )
    np.testing.assert_array_equal(render(item, img, "h"), planned.image)
    np.testing.assert_array_equal(render(Item(key="i", image_id="i"), img, "h"), img)
    wrong = Item(
        key="k",
        image_id="i",
        family="F2_sunglasses",
        level="medium",
        seed=11,
        recipe_sha256="0" * 64,
    )
    with pytest.raises(ManifestError, match="recipe"):
        render(wrong, img, "h")
    stressed = render(
        Item(
            key="s",
            image_id="i",
            stress_steps=(("F1_mask", "medium", 1), ("F2_sunglasses", "medium", 2)),
        ),
        img,
        "h",
    )
    assert stressed.shape == img.shape and not np.array_equal(stressed, img)
