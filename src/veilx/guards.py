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
