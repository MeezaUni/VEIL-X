"""Pixel hashing, perceptual hashing, duplicate validation and grouping (synthetic images)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from veilx.config.schema import SplitName
from veilx.data.dedup import (
    candidate_pairs,
    connected_groups,
    pair_distance,
    random_duplicate,
    select_rule,
    signals,
    validate_rule,
)
from veilx.data.images import (
    decode_rgb,
    encode_png_rgb,
    fingerprint,
    hamming64,
    phash64,
    pixel_sha256,
)
from veilx.data.integrity import (
    FingerprintTable,
    finalize_records,
    find_duplicates,
    fingerprint_records,
)
from veilx.data.paths import resolve_data_root
from veilx.errors import DataAccessError

from .helpers import rec


def smooth_image(seed: int, size: int = 64) -> np.ndarray:
    """Low-frequency synthetic image (pHash is meaningful on smooth content)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC))


def test_pixel_hash_ignores_encoding_but_not_content() -> None:
    a = smooth_image(1)
    png = encode_png_rgb(a)
    assert pixel_sha256(decode_rgb(png)) == pixel_sha256(a)
    b = a.copy()
    b[0, 0, 0] ^= 1
    assert pixel_sha256(b) != pixel_sha256(a)
    assert pixel_sha256(a[:32]) != pixel_sha256(a)


def test_phash_and_ncc_separate_duplicates_from_different_images() -> None:
    rng = np.random.default_rng(0)
    pos, neg = [], []
    for s in range(40):
        a = smooth_image(s)
        dup, _ = random_duplicate(a, rng)
        pos.append(pair_distance(signals(a), signals(dup)))
        neg.append(pair_distance(signals(a), signals(smooth_image(s + 1000))))
    pos_a, neg_a = np.array(pos), np.array(neg)
    assert np.median(pos_a[:, 0]) < np.median(neg_a[:, 0])
    assert np.median(pos_a[:, 1]) > np.median(neg_a[:, 1])


def test_flip_aware_distance() -> None:
    a = smooth_image(3)
    s, f = signals(a), signals(np.ascontiguousarray(a[:, ::-1]))
    ham, ncc = pair_distance(s, f)
    assert ham == 0 and ncc == pytest.approx(1.0, abs=1e-5)
    assert hamming64(phash64(a), phash64(a)) == 0


def test_fingerprint_reports_corruption() -> None:
    fp, thumb = fingerprint(b"not an image")
    assert not fp.ok and thumb is None and fp.error
    fp2, thumb2 = fingerprint(encode_png_rgb(smooth_image(5)))
    assert fp2.ok and fp2.width == 64 and thumb2 is not None and thumb2.shape == (256,)


def test_rule_selection_prefers_low_same_identity_fpr() -> None:
    pos = np.array([[2, 0.99], [4, 0.97], [6, 0.95]] * 50, dtype=float)
    same = np.array([[10, 0.9], [30, 0.5]] * 50, dtype=float)
    imp = np.array([[32, 0.1]] * 100, dtype=float)
    rule = select_rule(pos, same, imp, target_recall=0.99)
    assert rule is not None and rule.recall == 1.0 and rule.same_identity_fpr == 0.0
    v = validate_rule((pos, same, imp), (pos, same, imp), target_recall=0.99)
    assert v.accepted
    impossible = validate_rule((pos, pos, imp), (pos, pos, imp), target_recall=0.99)
    assert not impossible.accepted  # duplicates indistinguishable from same-identity images


def test_frontier_and_review_rule() -> None:
    from veilx.data.dedup import frontier, review_rule

    pos = np.array([[2, 0.99]] * 90 + [[20, 0.7]] * 10, dtype=float)  # 10 % hard duplicates
    same = np.array([[18, 0.8]] * 5 + [[40, 0.2]] * 95, dtype=float)
    imp = np.array([[40, 0.1]] * 100, dtype=float)
    front = frontier(pos, same, imp, fpr_budgets=(0.0, 0.05))
    assert front[0].same_identity_fpr == 0.0 and front[0].recall == pytest.approx(0.9)
    assert front[1].recall == 1.0 and front[1].same_identity_fpr <= 0.05
    rule, val = review_rule((pos, same, imp), (pos, same, imp), max_same_identity_fpr=0.001)
    assert rule is not None and val is not None and rule.same_identity_fpr == 0.0
    assert rule.recall == pytest.approx(0.9)


def test_exhaustive_candidate_search_matches_brute_force() -> None:
    imgs = [smooth_image(s) for s in range(30)]
    rng = np.random.default_rng(1)
    imgs += [random_duplicate(imgs[k], rng)[0] for k in (0, 5, 9)]
    sig = [signals(a) for a in imgs]
    ph = np.array([s.phash for s in sig], dtype=np.uint64)
    phf = np.array([s.phash_flip for s in sig], dtype=np.uint64)
    th = np.stack([s.thumb for s in sig])
    thf = np.stack([s.thumb_flip for s in sig])
    found = {
        (i, j)
        for i, j, _, _ in candidate_pairs(ph, phf, th, thf, hamming_max=12, ncc_min=0.9, block=7)
    }
    brute = {
        (i, j)
        for i in range(len(sig))
        for j in range(i + 1, len(sig))
        if (d := pair_distance(sig[i], sig[j]))[0] <= 12 and d[1] >= 0.9
    }
    assert found == brute


def test_connected_groups() -> None:
    g = connected_groups(6, [(0, 3), (3, 5), (1, 2)])
    assert g.tolist() == [0, 1, 1, 0, 4, 0]


def test_fingerprint_table_roundtrip_and_duplicate_policy(tmp_path: Path) -> None:
    imgs = {
        "a": smooth_image(1),
        "b": smooth_image(2),
        "c": smooth_image(1),
        "d": smooth_image(3),
        "e": smooth_image(2),
    }
    for k, a in imgs.items():
        (tmp_path / f"{k}.png").write_bytes(encode_png_rgb(a))
    (tmp_path / "bad.png").write_bytes(b"corrupt")
    records = [
        rec("a", "p1", SplitName.TRAIN).model_copy(update={"relpath": "a.png"}),
        rec("b", "p2", SplitName.TRAIN).model_copy(update={"relpath": "b.png"}),
        rec("c", "p1", SplitName.TRAIN).model_copy(
            update={"relpath": "c.png"}
        ),  # exact dup, same id
        rec("d", "p3", SplitName.TEST).model_copy(update={"relpath": "d.png"}),
        rec("e", "p4", SplitName.TRAIN).model_copy(
            update={"relpath": "e.png"}
        ),  # dup of b, other id
        rec("bad", "p5", SplitName.TRAIN).model_copy(update={"relpath": "bad.png"}),
    ]
    table = fingerprint_records(records, tmp_path, workers=2)
    assert table.ok.tolist() == [True, True, True, True, True, False] and "bad" in table.errors
    table.save(tmp_path / "fp.npz")
    back = FingerprintTable.load(tmp_path / "fp.npz")
    assert back.pixel_sha == table.pixel_sha and back.errors == table.errors
    from veilx.data.dedup import DedupValidation

    no_near = DedupValidation(0.99, "train", "val", None, None, False, "n/a")
    findings = find_duplicates(records, table, no_near)
    assert findings.exact_groups == 2 and findings.groups_spanning_identities == 1
    final = finalize_records(records, table, findings)
    ids = [r.image_id for r in final]
    assert ids == ["a", "c", "d"]  # b/e removed (ambiguous label), bad removed (decode failure)
    assert final[0].source_group == final[1].source_group  # a and c share one root photo


def test_data_root_must_not_be_tracked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from veilx.data import paths

    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert resolve_data_root().root == tmp_path.resolve()
    with pytest.raises(DataAccessError):
        resolve_data_root(paths.PROJECT_ROOT / "src")  # inside the repo, not ignored
    assert resolve_data_root(paths.PROJECT_ROOT / "data").root.name == "data"
