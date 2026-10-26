"""Pair scoring and rate summaries on synthetic embeddings (no biometric data)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from veilx.data.manifest import PairRecord
from veilx.embeddings.jobs import occluded_key
from veilx.embeddings.store import Shard
from veilx.errors import ManifestError
from veilx.eval.scoring import EmbeddingLookup, score_pairs
from veilx.eval.summaries import rate, selective_summary, verification_summary
from veilx.metrics.selective import ABSTAIN, MATCH, NON_MATCH


def _shard(keys: list[str], emb: np.ndarray, fp: str = "fp") -> Shard:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return Shard(
        np.asarray(keys),
        emb.astype(np.float32),
        np.full(len(keys), 2.0, np.float32),
        {"backbone_fingerprint": fp},
    )


def _pair(pid: str, a: str, b: str, label: str) -> PairRecord:
    return PairRecord(
        pair_id=pid,
        image_a=a,
        image_b=b,
        label=label,
        stratum="clean_clean",
        split="test",
        protocol="veilx",
    )


def test_clean_and_occluded_scores_and_covariates() -> None:
    e = np.eye(4)
    clean = _shard(["d:1:0", "d:1:1", "d:2:0"], np.stack([e[0], e[0] + 0.1 * e[1], e[2]]))
    okey = occluded_key("d:1:1", "F1_mask", "high", 5)
    occ = _shard([okey], np.stack([e[1]]))
    lk = EmbeddingLookup([clean, occ])
    pairs = [_pair("p1", "d:1:0", "d:1:1", "genuine"), _pair("p2", "d:1:0", "d:2:0", "impostor")]
    ident = {"d:1:0": "1", "d:1:1": "1", "d:2:0": "2"}
    sp = score_pairs(pairs, ident, lk)
    assert sp.score[0] == pytest.approx(1 / np.sqrt(1.01)) and sp.score[1] == pytest.approx(0.0)
    assert (sp.covis == 1.0).all() and list(sp.stratum) == ["clean_clean"] * 2
    spec = {"family": "F1_mask", "severity_level": "high", "seed": 5, "visibility_ratio": 0.6}
    rows = {
        "p1": {"a": None, "b": spec, "stratum": "clean_occluded"},
        "p2": {"a": None, "b": None, "stratum": "clean_clean"},
    }
    so = score_pairs(pairs, ident, lk, rows)
    assert so.score[0] == pytest.approx(0.0) and so.covis[0] == pytest.approx(0.6)
    assert so.fam_b[0] == "F1_mask" and so.lvl_b[0] == "high" and so.fam_a[0] == ""
    with pytest.raises(ManifestError, match="no embedding"):
        score_pairs([_pair("p3", "d:9:0", "d:1:0", "impostor")], ident | {"d:9:0": "9"}, lk)
    with pytest.raises(ManifestError, match="rows"):
        score_pairs(pairs, ident, lk, {"p1": rows["p1"]})


def test_lookup_rejects_mixed_runs_and_duplicate_keys() -> None:
    e = np.eye(2)
    with pytest.raises(ManifestError, match="different backbone"):
        EmbeddingLookup([_shard(["a"], e[:1], "x"), _shard(["b"], e[1:], "y")])
    with pytest.raises(ManifestError, match="two shards"):
        EmbeddingLookup([_shard(["a"], e[:1]), _shard(["a"], e[1:])])


def _scored(n_id: int = 60, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    keys, vecs, pairs, ident = [], [], [], {}
    for i in range(n_id):
        base = rng.standard_normal(16)
        for j in range(2):
            k = f"d:{i}:{j}"
            keys.append(k)
            vecs.append(base + 0.6 * rng.standard_normal(16))
            ident[k] = str(i)
        pairs.append(_pair(f"g{i}", f"d:{i}:0", f"d:{i}:1", "genuine"))
        pairs.append(_pair(f"i{i}", f"d:{i}:0", f"d:{(i + 1) % n_id}:1", "impostor"))
    return score_pairs(pairs, ident, EmbeddingLookup([_shard(keys, np.stack(vecs))]))


def test_rate_intervals_follow_protocol_methods() -> None:
    sp = _scored()
    g, i = sp.label == 1, sp.label == 0
    r = rate(sp.score < 0.5, g, sp, "genuine")
    assert r["n"] == 60 and r["method"] == "jackknife_logit" and r["lo"] <= r["value"] <= r["hi"]
    r0 = rate(np.zeros(len(sp), bool), i, sp, "impostor")
    assert r0["k"] == 0 and r0["hi"] > 0 and "clopper" in r0["method"]
    assert rate(sp.score > 0, np.zeros(len(sp), bool), sp, "genuine")["value"] is None


def test_verification_and_selective_summaries() -> None:
    sp = _scored()
    v = verification_summary(sp, {"t": 0.5})
    assert 0.5 < v["roc_auc"] <= 1.0 and v["n_genuine"] == v["n_impostor"] == 60
    assert set(v["operating_points"]["t"]) == {"threshold", "fmr", "fnmr"}
    d = np.where(sp.score >= 0.6, MATCH, np.where(sp.score <= 0.2, NON_MATCH, ABSTAIN))
    s = selective_summary(d, sp, 0.05, 0.05)
    assert 0 <= s["coverage"] <= 1
    assert s["verdict_fm"] in {
        "within_target",
        "point_estimate_above_target",
        "violated_ci_excludes_target",
    }
    none = selective_summary(np.full(len(sp), ABSTAIN), sp, 0.05, 0.05)
    assert none["coverage"] == 0 and none["verdict_fm"] == "no_accepted_pairs"


def test_roc_points_and_histogram() -> None:
    from veilx.eval.summaries import roc_points, score_histogram

    sp = _scored()
    r = roc_points(sp)
    assert len(r["fpr"]) == len(r["tpr"]) and r["tpr"][-1] == 1.0
    assert all(a <= b for a, b in itertools.pairwise(r["tpr"]))  # monotone in FPR
    h = score_histogram(sp)
    assert sum(h["genuine"]) == 60 and sum(h["impostor"]) == 60
    assert len(h["edges"]) == len(h["genuine"]) + 1
