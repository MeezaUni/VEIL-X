"""Exact and near-duplicate detection with a validated decision rule (protocol_design.md §8).

Signals (``veilx.data.images``): flip-aware pHash Hamming distance and coarse NCC. A pair is a
**near-duplicate candidate** iff ``hamming <= h`` AND ``ncc >= t``. Face-embedding similarity is
deliberately *not* used: two photographs of the same person are not duplicates.

Validation (non-test data only):

* positives  = an image vs a randomly transformed copy of itself (JPEG re-encode, rescale,
  crop <= 10 %, brightness/contrast, noise, horizontal flip);
* hard negatives = two *different* images of the same identity (must not be flagged);
* negatives  = images of two different identities.

``(h, t)`` is selected on **train** identities (maximise recall >= target, then minimise the
same-identity false-positive rate), then **confirmed on val** identities. The threshold is
accepted only if the val check passes; otherwise it stays unset and is reported as such.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance

from veilx.data.images import U8, phash64, thumb_vector

UInt64Arr = NDArray[np.uint64]
F32 = NDArray[np.float32]


# --------------------------------------------------------------------------- transforms


def random_duplicate(arr: U8, rng: np.random.Generator) -> tuple[U8, str]:
    """A plausible re-encoded/edited copy of ``arr`` (a known duplicate)."""
    h, w = arr.shape[:2]
    im = Image.fromarray(arr, mode="RGB")
    ops: list[str] = []
    if rng.random() < 0.5:
        s = rng.uniform(0.5, 1.5)
        im = im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.Resampling.BILINEAR)
        im = im.resize((w, h), Image.Resampling.BILINEAR)
        ops.append(f"rescale{s:.2f}")
    if rng.random() < 0.5:
        f = rng.uniform(0.0, 0.10)
        dx, dy = int(w * f * rng.random()), int(h * f * rng.random())
        im = im.crop((dx, dy, w - int(w * f) + dx, h - int(h * f) + dy)).resize((w, h))
        ops.append(f"crop{f:.2f}")
    if rng.random() < 0.5:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.15))
        ops.append("photometric")
    if rng.random() < 0.3:
        im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        ops.append("flip")
    out = np.asarray(im, dtype=np.float64)
    if rng.random() < 0.5:
        out = out + rng.normal(0, rng.uniform(1, 5), out.shape)
        ops.append("noise")
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        buf = io.BytesIO()
        q = int(rng.integers(30, 96))
        Image.fromarray(out_u8, mode="RGB").save(buf, format="JPEG", quality=q)
        out_u8 = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
        ops.append(f"jpeg{q}")
    return out_u8, "+".join(ops) or "identity"


# --------------------------------------------------------------------------- signals


@dataclass(frozen=True)
class Signals:
    phash: int
    phash_flip: int
    thumb: F32
    thumb_flip: F32


def signals(arr: U8) -> Signals:
    flip = arr[:, ::-1, :]
    return Signals(
        phash64(arr),
        phash64(np.ascontiguousarray(flip)),
        thumb_vector(arr),
        thumb_vector(np.ascontiguousarray(flip)),
    )


def pair_distance(a: Signals, b: Signals) -> tuple[int, float]:
    """Flip-aware (min Hamming, max NCC)."""
    ham = min((a.phash ^ b.phash).bit_count(), (a.phash ^ b.phash_flip).bit_count())
    ncc = max(float(a.thumb @ b.thumb), float(a.thumb @ b.thumb_flip))
    return ham, ncc


# --------------------------------------------------------------------------- rule selection


@dataclass(frozen=True)
class RuleEval:
    hamming_max: int
    ncc_min: float
    recall: float
    same_identity_fpr: float
    impostor_fpr: float
    n_pos: int
    n_same: int
    n_imp: int


def _evaluate(
    h: int, t: float, pos: NDArray[np.float64], same: NDArray[np.float64], imp: NDArray[np.float64]
) -> RuleEval:
    def flag(x: NDArray[np.float64]) -> NDArray[np.bool_]:
        return (x[:, 0] <= h) & (x[:, 1] >= t)

    return RuleEval(
        h,
        t,
        float(flag(pos).mean()),
        float(flag(same).mean()),
        float(flag(imp).mean()),
        len(pos),
        len(same),
        len(imp),
    )


HAMMING_GRID = tuple(range(0, 25))
NCC_GRID = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98)


def select_rule(
    pos: NDArray[np.float64],
    same: NDArray[np.float64],
    imp: NDArray[np.float64],
    target_recall: float,
) -> RuleEval | None:
    """Among rules with recall >= target, minimise same-identity FPR, then impostor FPR,
    then prefer the stricter rule (smaller h, larger t)."""
    ok = [
        e
        for h in HAMMING_GRID
        for t in NCC_GRID
        if (e := _evaluate(h, t, pos, same, imp)).recall >= target_recall
    ]
    if not ok:
        return None
    return min(ok, key=lambda e: (e.same_identity_fpr, e.impostor_fpr, e.hamming_max, -e.ncc_min))


def frontier(
    pos: NDArray[np.float64],
    same: NDArray[np.float64],
    imp: NDArray[np.float64],
    fpr_budgets: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05),
) -> list[RuleEval]:
    """Best achievable duplicate recall for each same-identity false-positive budget."""
    evals = [_evaluate(h, t, pos, same, imp) for h in HAMMING_GRID for t in NCC_GRID]
    out = []
    for b in fpr_budgets:
        ok = [e for e in evals if e.same_identity_fpr <= b]
        if ok:
            out.append(
                max(ok, key=lambda e: (e.recall, -e.impostor_fpr, -e.hamming_max, e.ncc_min))
            )
    return out


def review_rule(
    train: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    val: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    *,
    max_same_identity_fpr: float = 0.001,
) -> tuple[RuleEval | None, RuleEval | None]:
    """Strict rule for nominating candidates for MANUAL REVIEW (never for automatic removal):
    maximal train recall subject to train same-identity FPR <= budget; evaluated on val."""
    best = frontier(*train, fpr_budgets=(max_same_identity_fpr,))
    if not best:
        return None, None
    rule = best[0]
    return rule, _evaluate(rule.hamming_max, rule.ncc_min, *val)


@dataclass(frozen=True)
class DedupValidation:
    target_recall: float
    selected_on: str
    confirmed_on: str
    train_rule: RuleEval | None
    val_eval: RuleEval | None
    accepted: bool
    acceptance_criteria: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_rule(
    train: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    val: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    *,
    target_recall: float,
    max_val_recall_shortfall: float = 0.01,
    max_val_same_identity_fpr: float = 0.01,
) -> DedupValidation:
    rule = select_rule(*train, target_recall=target_recall)
    val_eval = _evaluate(rule.hamming_max, rule.ncc_min, *val) if rule else None
    accepted = bool(
        rule is not None
        and val_eval is not None
        and val_eval.recall >= target_recall - max_val_recall_shortfall
        and val_eval.same_identity_fpr <= max_val_same_identity_fpr
    )
    return DedupValidation(
        target_recall=target_recall,
        selected_on="train",
        confirmed_on="val",
        train_rule=rule,
        val_eval=val_eval,
        accepted=accepted,
        acceptance_criteria=(
            f"val recall >= {target_recall - max_val_recall_shortfall:.3f} and val "
            f"same-identity FPR <= {max_val_same_identity_fpr:.3f}"
        ),
    )


# --------------------------------------------------------------------------- exhaustive search


def candidate_pairs(
    phash: UInt64Arr,
    phash_flip: UInt64Arr,
    thumbs: F32,
    thumbs_flip: F32,
    *,
    hamming_max: int,
    ncc_min: float,
    block: int = 256,
) -> list[tuple[int, int, int, float]]:
    """Exhaustive search (exact, no hashing shortcuts) for pairs i < j meeting the rule."""
    n = phash.size
    out: list[tuple[int, int, int, float]] = []
    for start in range(0, n, block):
        stop = min(n, start + block)
        a = phash[start:stop, None]
        ham = np.minimum(
            np.bitwise_count(a ^ phash[None, :]), np.bitwise_count(a ^ phash_flip[None, :])
        )
        rows, cols = np.nonzero(ham <= hamming_max)
        keep = cols > rows + start  # i < j only
        rows, cols = rows[keep] + start, cols[keep]
        if rows.size == 0:
            continue
        ncc = np.maximum(
            np.einsum("ij,ij->i", thumbs[rows], thumbs[cols]),
            np.einsum("ij,ij->i", thumbs[rows], thumbs_flip[cols]),
        )
        sel = ncc >= ncc_min
        out.extend(
            (int(i), int(j), int(ham[i - start, j]), float(c))
            for i, j, c in zip(rows[sel], cols[sel], ncc[sel], strict=True)
        )
    return out


def connected_groups(n: int, edges: Sequence[tuple[int, int]]) -> NDArray[np.int64]:
    """Union-find; returns a group representative (smallest index) for every node."""
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)], dtype=np.int64)
