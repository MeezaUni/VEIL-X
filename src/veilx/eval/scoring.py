"""Pair scores with per-pair covariates.

A pair side is the clean image (key ``image_id``) or, under an occlusion plan, the occluded
render (key from :func:`veilx.embeddings.jobs.occluded_key`). The score is the cosine
similarity of the L2-normalised embeddings. Covariates per pair:

* identities of both sides (deletion units for the jackknife);
* ground-truth visibility of each side (1.0 when clean) from the plan's synthetic mask -
  this is ORACLE visibility, not an estimate; ``covis = min(vis_a, vis_b)``;
* occlusion family and severity level per side; plan stratum;
* pre-normalisation feature norms (backbone quality proxy).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.data.manifest import PairRecord
from veilx.embeddings.jobs import occluded_key
from veilx.embeddings.store import Shard
from veilx.errors import ManifestError

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
Str = NDArray[np.str_]


class EmbeddingLookup:
    """Key -> (unit embedding, norm) over one or more shards of the SAME backbone run."""

    def __init__(self, shards: Sequence[Shard]) -> None:
        fps = {s.meta["backbone_fingerprint"] for s in shards}
        if len(fps) != 1:
            raise ManifestError(f"shards from different backbone runs: {sorted(fps)}")
        self.backbone_fingerprint = fps.pop()
        self.emb = np.concatenate([s.emb for s in shards])
        self.norm = np.concatenate([s.norm for s in shards]).astype(np.float64)
        keys = np.concatenate([s.keys for s in shards])
        self.index: dict[str, int] = {}
        for i, k in enumerate(keys.tolist()):
            if k in self.index:
                raise ManifestError(f"key {k} present in two shards")
            self.index[k] = i

    def rows(self, keys: Sequence[str]) -> I64:
        try:
            return np.fromiter((self.index[k] for k in keys), dtype=np.int64, count=len(keys))
        except KeyError as exc:  # never silently skip a pair
            raise ManifestError(f"no embedding for {exc.args[0]}") from exc


@dataclass(frozen=True)
class ScoredPairs:
    pair_id: Str
    score: F64
    label: I64  # 1 genuine, 0 impostor
    id_a: Str
    id_b: Str
    vis_a: F64
    vis_b: F64
    fam_a: Str  # "" when clean
    fam_b: Str
    lvl_a: Str
    lvl_b: Str
    stratum: Str
    norm_a: F64
    norm_b: F64

    def __len__(self) -> int:
        return int(self.score.size)

    @property
    def covis(self) -> F64:
        return np.minimum(self.vis_a, self.vis_b)

    @property
    def min_norm(self) -> F64:
        return np.minimum(self.norm_a, self.norm_b)

    def subset(self, mask: NDArray[np.bool_]) -> ScoredPairs:
        return ScoredPairs(**{k: getattr(self, k)[mask] for k in self.__dataclass_fields__})


def _side(spec: Mapping[str, Any] | None, image_id: str) -> tuple[str, float, str, str]:
    if spec is None:
        return image_id, 1.0, "", ""
    key = occluded_key(image_id, spec["family"], spec["severity_level"], int(spec["seed"]))
    return key, float(spec["visibility_ratio"]), str(spec["family"]), str(spec["severity_level"])


def score_pairs(
    pairs: Sequence[PairRecord],
    identity_of: Mapping[str, str],
    lookup: EmbeddingLookup,
    plan_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> ScoredPairs:
    """Score ``pairs`` clean (``plan_rows`` None) or under a plan (rows keyed by pair id)."""
    if plan_rows is not None and len(plan_rows) != len(pairs):
        raise ManifestError(f"plan has {len(plan_rows)} rows for {len(pairs)} pairs")
    ka, kb, va, vb, fa, fb, la, lb, st = [], [], [], [], [], [], [], [], []
    for p in pairs:
        row = plan_rows[p.pair_id] if plan_rows is not None else None
        a = _side(row["a"] if row else None, p.image_a)
        b = _side(row["b"] if row else None, p.image_b)
        ka.append(a[0])
        kb.append(b[0])
        va.append(a[1])
        vb.append(b[1])
        fa.append(a[2])
        fb.append(b[2])
        la.append(a[3])
        lb.append(b[3])
        st.append(str(row["stratum"]) if row else "clean_clean")
    ra, rb = lookup.rows(ka), lookup.rows(kb)
    score = np.einsum("ij,ij->i", lookup.emb[ra], lookup.emb[rb]).astype(np.float64)
    return ScoredPairs(
        pair_id=np.asarray([p.pair_id for p in pairs]),
        score=score,
        label=np.asarray([1 if p.label == "genuine" else 0 for p in pairs], dtype=np.int64),
        id_a=np.asarray([identity_of[p.image_a] for p in pairs]),
        id_b=np.asarray([identity_of[p.image_b] for p in pairs]),
        vis_a=np.asarray(va, dtype=np.float64),
        vis_b=np.asarray(vb, dtype=np.float64),
        fam_a=np.asarray(fa),
        fam_b=np.asarray(fb),
        lvl_a=np.asarray(la),
        lvl_b=np.asarray(lb),
        stratum=np.asarray(st),
        norm_a=lookup.norm[ra],
        norm_b=lookup.norm[rb],
    )
