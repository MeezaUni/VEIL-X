"""Leave-one-family-out (LOFO) condition plans for pair lists.

For a fold with held-out family ``F_k`` and a *role*:

* ``train`` / ``val`` / ``cal`` / ``test_id`` — occluded sides use only the seen families
  (all except ``F_k``); the guard :func:`veilx.guards.require_heldout_isolation` is applied;
* ``test_ood`` — every occluded side involves ``F_k``: clean-occluded and same-family pairs use
  ``F_k``; cross-family pairs combine ``F_k`` with one seen family.

Strata (clean-clean, clean-occluded, occluded-occluded same/cross family) are assigned in equal
quotas after a seeded shuffle; severity levels are uniform over low/medium/high. Each side's seed
is derived from (global seed, fold, role, pair id, side) so plans are reproducible and
independent of iteration order. Stress conditions (two physical families on one image) are
generated separately and are never part of LOFO statistics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from veilx.data.manifest import PairRecord
from veilx.guards import require_heldout_isolation
from veilx.occlusion.families import LEVELS, PHYSICAL, Level
from veilx.seeding import derive_seed, make_rng

Role = Literal["train", "val", "cal", "test_id", "test_ood"]
Stratum = Literal[
    "clean_clean",
    "clean_occluded",
    "occluded_occluded_same_family",
    "occluded_occluded_cross_family",
]
STRATA: tuple[Stratum, ...] = (
    "clean_clean",
    "clean_occluded",
    "occluded_occluded_same_family",
    "occluded_occluded_cross_family",
)


@dataclass(frozen=True)
class SideCondition:
    family: str | None
    level: Level | None
    seed: int


@dataclass(frozen=True)
class PairCondition:
    pair_id: str
    fold: str  # held-out family id
    role: Role
    stratum: Stratum
    side_a: SideCondition
    side_b: SideCondition

    def families(self) -> set[str]:
        return {f for f in (self.side_a.family, self.side_b.family) if f is not None}


@dataclass(frozen=True)
class StressCondition:
    pair_id: str
    steps_a: tuple[tuple[str, Level, int], ...]
    steps_b: tuple[tuple[str, Level, int], ...]


def plan_conditions(
    pairs: Sequence[PairRecord],
    families: Sequence[str],
    heldout: str,
    role: Role,
    *,
    global_seed: int,
) -> list[PairCondition]:
    if heldout not in families:
        raise ValueError(f"unknown held-out family {heldout!r}")
    seen = [f for f in families if f != heldout]
    pairs = sorted(pairs, key=lambda p: p.pair_id)  # plans must not depend on input order
    rng = make_rng(global_seed, "plan", heldout, role)
    order = rng.permutation(len(pairs))
    out: list[PairCondition] = []
    for rank, idx in enumerate(order):
        p = pairs[int(idx)]
        stratum = STRATA[rank % len(STRATA)]

        def side(tag: str, fam: str | None, p: PairRecord = p) -> SideCondition:
            seed = derive_seed(global_seed, "occl", heldout, role, p.pair_id, tag)
            lvl = LEVELS[seed % 3] if fam else None
            return SideCondition(fam, lvl, seed)

        if role == "test_ood":
            pick_seen = seen[int(rng.integers(len(seen)))]
            fams = {
                "clean_clean": (None, None),
                "clean_occluded": (None, heldout),
                "occluded_occluded_same_family": (heldout, heldout),
                "occluded_occluded_cross_family": (heldout, pick_seen),
            }[stratum]
        else:
            f1 = seen[int(rng.integers(len(seen)))]
            others = [f for f in seen if f != f1]
            f2 = others[int(rng.integers(len(others)))]
            fams = {
                "clean_clean": (None, None),
                "clean_occluded": (None, f1),
                "occluded_occluded_same_family": (f1, f1),
                "occluded_occluded_cross_family": (f1, f2),
            }[stratum]
        if rng.random() < 0.5:  # randomise which side carries the occlusion
            fams = (fams[1], fams[0])
        cond = PairCondition(
            p.pair_id, heldout, role, stratum, side("a", fams[0]), side("b", fams[1])
        )
        if role != "test_ood":
            require_heldout_isolation(heldout, tuple(cond.families()))
        out.append(cond)
    out.sort(key=lambda c: c.pair_id)
    return out


STRESS_COMBOS: tuple[tuple[str, str], ...] = (
    ("F1_mask", "F2_sunglasses"),
    ("F1_mask", "F3_hand_object"),
    ("F2_sunglasses", "F4_random_patch"),
    ("F3_hand_object", "F4_random_patch"),
)


def plan_stress(
    pairs: Sequence[PairRecord], *, global_seed: int, levels: Sequence[Level] = ("medium",)
) -> list[StressCondition]:
    """Combined-occlusion stress test: one physical combination on each side (not LOFO)."""
    assert all(PHYSICAL[f] for combo in STRESS_COMBOS for f in combo)
    rng = make_rng(global_seed, "stress")
    out = []
    for p in pairs:

        def steps(tag: str, p: PairRecord = p) -> tuple[tuple[str, Level, int], ...]:
            combo = STRESS_COMBOS[int(rng.integers(len(STRESS_COMBOS)))]
            lvl = levels[int(rng.integers(len(levels)))]
            return tuple(
                (f, lvl, derive_seed(global_seed, "stress", p.pair_id, tag, f)) for f in combo
            )

        out.append(StressCondition(p.pair_id, steps("a"), steps("b")))
    return out


def assert_plan_isolation(plan: Sequence[PairCondition]) -> None:
    """Independent re-check used by tests and the isolation script."""
    for c in plan:
        fams = c.families()
        if c.role == "test_ood":
            if fams and c.fold not in fams:
                raise AssertionError(f"{c.pair_id}: OOD pair without the held-out family")
        elif c.fold in fams:
            raise AssertionError(f"{c.pair_id}: held-out family {c.fold} used in role {c.role}")
