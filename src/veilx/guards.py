"""Runtime guards for pre-registered protocol rules."""

from __future__ import annotations

from veilx.config.schema import SplitName
from veilx.errors import ProtocolViolationError


def require_final_for_test(split: SplitName, *, final: bool) -> None:
    """Refuse any access to the test split unless the run is explicitly marked final."""
    if split is SplitName.TEST and not final:
        raise ProtocolViolationError(
            "test split requested without --final; tuning on test data is prohibited"
        )


def require_heldout_isolation(
    heldout_family: str, families_used_for_fitting: tuple[str, ...]
) -> None:
    """Refuse fitting/calibration that saw the family it will be evaluated on."""
    if heldout_family in families_used_for_fitting:
        raise ProtocolViolationError(
            f"held-out family {heldout_family} was used for fitting/selection/calibration"
        )


def require_disjoint_identities(named_sets: dict[str, set[str]]) -> None:
    """Refuse calibration leakage: every pair of named identity sets (e.g. training, selection,
    calibration and confirmatory test identities) must be disjoint."""
    names = sorted(named_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = named_sets[a] & named_sets[b]
            if common:
                raise ProtocolViolationError(
                    f"identity leakage: {len(common)} identities shared by {a} and {b}"
                )


def require_eval_only_unused(families_used_for_fitting: tuple[str, ...]) -> None:
    """Refuse any fitting, selection or calibration that touched an evaluation-only family."""
    from veilx.config.schema import EVAL_ONLY_FAMILY_IDS

    bad = [f for f in families_used_for_fitting if f in EVAL_ONLY_FAMILY_IDS]
    if bad:
        raise ProtocolViolationError(f"evaluation-only families used for fitting: {bad}")
