from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from veilx.config import (
    APPROVED_FAMILY_IDS,
    AccessStatus,
    ConstantStatus,
    DatasetRole,
    load_dataset_registry,
    load_protocol,
)
from veilx.config.schema import DatasetRegistry, ResearchProtocol
from veilx.errors import ConfigError

from .conftest import DATASETS, PROTOCOL

pytestmark = pytest.mark.structural


def _validate(d: dict[str, Any]) -> ResearchProtocol:
    return ResearchProtocol.model_validate(d)


def test_shipped_protocol_is_valid() -> None:
    proto = load_protocol(PROTOCOL)
    assert proto.family_ids == APPROVED_FAMILY_IDS
    assert proto.status is ConstantStatus.PROVISIONAL


def test_registered_constants_match_approved_values() -> None:
    c = load_protocol(PROTOCOL).constants
    assert c.clean_noninferiority_margin_pp.value == 0.5
    assert c.cost_ratio_false_match_to_false_non_match.value == 10
    assert c.confidence_level.value == 0.95
    assert c.accepted_decision_error_target.value == 0.05
    assert c.far_operating_points.value == (0.01, 0.001)
    assert c.bootstrap_ci_level.value == 0.95


def test_v040_real_data_sections_and_frozen_constants() -> None:
    proto = load_protocol(PROTOCOL)
    assert proto.protocol_version == "0.4.0"
    dp = proto.data_preparation
    assert dp is not None and dp.images_final == 643_995 and dp.identities_final == 101_999
    assert len(dp.parts) == 4 and set(dp.archive_sha256) == set(dp.archive_bytes)
    assert sum(dp.identities_per_split.values()) == dp.identities_final
    assert sum(dp.images_per_split.values()) == dp.images_final
    # train keeps the protocol-wide cap; val/test use the bounded cap (D-038)
    assert proto.pairs.genuine_pairs_per_identity_cap == dp.genuine_pairs_per_identity_cap["train"]
    assert (
        dp.genuine_pairs_per_identity_cap["val"]
        == dp.genuine_pairs_per_identity_cap["test_balanced"]
        == 3
    )
    roles = {x.split(":")[0] for x in dp.occlusion_plan_sets}
    assert roles == {"val", "cal", "test_id", "test_ood"}  # no train plans in Phase 2
    cp = proto.calibration_plan
    assert cp is not None and cp.independent_genuine_pairs + 2 * cp.independent_impostor_pairs <= (
        cp.calibration_identities
    )
    c = proto.constants
    assert c.confidence_level.status is ConstantStatus.FROZEN
    assert c.cost_ratio_false_match_to_false_non_match.status is ConstantStatus.FROZEN
    assert c.clean_noninferiority_margin_pp.status is ConstantStatus.PROVISIONAL
    assert c.accepted_decision_error_target.status is ConstantStatus.PROVISIONAL
    assert proto.deduplication.near_duplicate.threshold is None  # validation failed: stays unset
    assert proto.bootstrap.interval_method_by_statistic == {
        "genuine_pair_rates": "jackknife_logit",
        "impostor_pair_rates": "jackknife_t",
    }


def test_interval_method_needs_validation_record(mutable_protocol: dict[str, Any]) -> None:
    mutable_protocol["bootstrap"]["interval_method_validation"] = None
    with pytest.raises(ValueError, match="validation record"):
        _validate(mutable_protocol)


def test_phash_rule_needs_both_thresholds(mutable_protocol: dict[str, Any]) -> None:
    nd = mutable_protocol["deduplication"]["near_duplicate"]
    nd.update(threshold=14, validated_on_split="val", validation_record="x", ncc_min=None)
    with pytest.raises(ValueError, match="ncc_min"):
        _validate(mutable_protocol)
    nd["ncc_min"] = 0.9
    _validate(mutable_protocol)


def test_lighting_is_flagged_non_physical() -> None:
    proto = load_protocol(PROTOCOL)
    lighting = proto.challenge_families[-1]
    assert lighting.id == "F5_extreme_lighting"
    assert lighting.physical_occlusion is False
    assert lighting.interpretation == "separate_photometric"


def test_lofo_seen_families_exclude_heldout() -> None:
    proto = load_protocol(PROTOCOL)
    for fam in proto.family_ids:
        seen = proto.seen_families(fam)
        assert fam not in seen
        assert len(seen) == len(proto.family_ids) - 1
    with pytest.raises(ValueError):
        proto.seen_families("F9_unknown")


def test_guarantee_scope_and_assumptions_are_explicit() -> None:
    cal = load_protocol(PROTOCOL).calibration
    text = " ".join(cal.assumptions).lower()
    assert "exchangeab" in text and "independen" in text and "held-out" in text
    assert "not claimed" in cal.guarantee_scope_statement.lower()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("splits", "bands", "test"), [85, 100]),  # gap 80-85
        (("splits", "bands", "cal"), [70, 81]),  # overlap
        (("splits", "unit"), "image"),  # splits must be identity-level
        (("detector", "failure_policy_selective"), "drop"),  # failures may not be dropped
        (("test_access", "require_final_flag"), False),
        (("backbones", "primary", "frozen"), False),  # no fine-tuning
        (("bootstrap", "unit"), "pair"),
        (("bootstrap", "resamples"), 100),
        (("project", "scope"), "one-to-many"),
    ],
)
def test_protocol_rule_violations_rejected(
    mutable_protocol: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    node = mutable_protocol
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        _validate(mutable_protocol)


def test_family_set_is_fixed(mutable_protocol: dict[str, Any]) -> None:
    mutable_protocol["challenge_families"] = mutable_protocol["challenge_families"][:4]
    with pytest.raises(ValueError):
        _validate(mutable_protocol)


def test_lighting_cannot_be_physical(mutable_protocol: dict[str, Any]) -> None:
    mutable_protocol["challenge_families"][4]["physical_occlusion"] = True
    with pytest.raises(ValueError):
        _validate(mutable_protocol)


def test_isolation_scope_must_cover_required_stages(mutable_protocol: dict[str, Any]) -> None:
    mutable_protocol["heldout"]["isolation_scope"].remove("calibration")
    with pytest.raises(ValueError, match="calibration"):
        _validate(mutable_protocol)


def test_dedup_threshold_cannot_be_fixed_without_validation(
    mutable_protocol: dict[str, Any],
) -> None:
    nd = mutable_protocol["deduplication"]["near_duplicate"]
    nd["threshold"] = 14
    nd["ncc_min"] = 0.9
    with pytest.raises(ValueError, match="validation"):
        _validate(mutable_protocol)
    nd["validated_on_split"] = "val"
    nd["validation_record"] = "docs/dedup.md"
    _validate(mutable_protocol)


def test_dedup_validation_cannot_use_test(mutable_protocol: dict[str, Any]) -> None:
    nd = mutable_protocol["deduplication"]["near_duplicate"]
    nd.update(threshold=0.9, validated_on_split="test", validation_record="x")
    with pytest.raises(ValueError):
        _validate(mutable_protocol)


def test_frozen_protocol_rejects_provisional_constants(mutable_protocol: dict[str, Any]) -> None:
    mutable_protocol["status"] = "frozen"
    with pytest.raises(ValueError, match="provisional"):
        _validate(mutable_protocol)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("confidence_level", 1.2),
        ("accepted_decision_error_target", 0.7),
        ("far_operating_points", [0.01, 0.01]),
        ("far_operating_points", [1.5]),
        ("cost_ratio_false_match_to_false_non_match", -1),
    ],
)
def test_constant_ranges(mutable_protocol: dict[str, Any], name: str, value: Any) -> None:
    mutable_protocol["constants"][name]["value"] = value
    with pytest.raises(ValueError):
        _validate(mutable_protocol)


def test_loader_reports_clean_error(tmp_path: Path, protocol_dict: dict[str, Any]) -> None:
    bad = dict(protocol_dict, unexpected_key=1)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError, match="validation error"):
        load_protocol(p)
    with pytest.raises(ConfigError, match="not found"):
        load_protocol(tmp_path / "missing.yaml")


# ----------------------------------------------------------------- dataset registry


def test_shipped_registry_is_valid() -> None:
    reg = load_dataset_registry(DATASETS)
    assert reg.get("digiface1m").access_status is AccessStatus.CONFIRMED
    kaggle = reg.get("kaggle_trainingdatapro_human_face_images")
    assert kaggle.access_status is AccessStatus.BLOCKED
    assert kaggle.roles == (DatasetRole.NONE,)


def test_only_confirmed_datasets_are_usable() -> None:
    reg = load_dataset_registry(DATASETS)
    assert [d.id for d in reg.usable_for(DatasetRole.CAL)] == ["digiface1m"]
    assert reg.usable_for(DatasetRole.TEST_REAL) == ()  # ROF still conditional


def test_test_only_dataset_cannot_be_used_for_calibration(registry_dict: dict[str, Any]) -> None:
    d = {**registry_dict, "datasets": [dict(e) for e in registry_dict["datasets"]]}
    rof = next(e for e in d["datasets"] if e["id"] == "rof")
    rof["roles"] = ["test_real", "cal"]
    with pytest.raises(ValueError, match="test-only"):
        DatasetRegistry.model_validate(d)


def test_blocked_dataset_cannot_have_roles(registry_dict: dict[str, Any]) -> None:
    d = {**registry_dict, "datasets": [dict(e) for e in registry_dict["datasets"]]}
    kag = next(e for e in d["datasets"] if e["id"] == "kaggle_trainingdatapro_human_face_images")
    kag["roles"] = ["visibility_training"]
    with pytest.raises(ValueError, match="blocked"):
        DatasetRegistry.model_validate(d)


def test_verification_roles_require_identity_labels(registry_dict: dict[str, Any]) -> None:
    d = {**registry_dict, "datasets": [dict(e) for e in registry_dict["datasets"]]}
    occ = next(e for e in d["datasets"] if e["id"] == "faceocc")
    occ["roles"] = ["test_synth"]
    with pytest.raises(ValueError, match="identity labels"):
        DatasetRegistry.model_validate(d)
