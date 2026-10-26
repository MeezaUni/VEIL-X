"""Typed schema for the VEIL-X research protocol (``configs/research_protocol.yaml``) and the
dataset registry (``configs/datasets.yaml``).

The schema encodes protocol rules as validation errors, so a config that would break the
pre-registered design cannot be loaded. Key rules:

* identity split bands are contiguous, non-overlapping and cover [0, 100);
* exactly the five approved challenge families exist, and lighting is flagged as
  non-physical so it is interpreted separately;
* the held-out family must be isolated from every fitting/selection/calibration stage;
* the near-duplicate threshold cannot be fixed without a recorded non-test validation;
* a ``frozen`` protocol cannot contain ``provisional`` constants;
* test access always requires an explicit final flag.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- shared


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConstantStatus(StrEnum):
    PROVISIONAL = "provisional"
    FROZEN = "frozen"


class SplitName(StrEnum):
    TRAIN = "train"
    VAL = "val"
    CAL = "cal"
    TEST = "test"


APPROVED_FAMILY_IDS: tuple[str, ...] = (
    "F1_mask",
    "F2_sunglasses",
    "F3_hand_object",
    "F4_random_patch",
    "F5_extreme_lighting",
)

REQUIRED_ISOLATION_STAGES: frozenset[str] = frozenset(
    {
        "feature_selection",
        "model_fitting",
        "threshold_selection",
        "hyperparameter_optimization",
        "calibration",
        "visibility_estimator_training",
        "occluder_asset_selection",
        "deduplication_threshold_validation",
    }
)

# --------------------------------------------------------------------------- sections


class ProjectSection(_Strict):
    name: Literal["VEIL-X"]
    usage: Literal["non-commercial-academic-research"]
    scope: Literal["one-to-one-verification-only"]


class SeedSection(_Strict):
    global_seed: int = Field(ge=0, le=2**32 - 1)
    experiment_seeds: tuple[int, ...] = Field(min_length=1)

    @field_validator("experiment_seeds")
    @classmethod
    def _unique_non_negative(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(v)) != len(v) or any(s < 0 for s in v):
            raise ValueError("experiment_seeds must be unique and non-negative")
        return v


class BackboneSpec(_Strict):
    id: str
    family: Literal["adaface", "arcface"]
    frozen: Literal[True]
    weights_licence: str
    weights_status: Literal["not_downloaded", "downloaded_unverified", "verified"]


class BackboneSection(_Strict):
    primary: BackboneSpec
    secondary: BackboneSpec


class DetectorSection(_Strict):
    id: str
    weights_licence: str
    failure_policy_selective: Literal["abstain"]
    failure_policy_score: Literal["forced_non_match"]
    report_failure_rate_per_family: Literal[True]


class ChallengeFamily(_Strict):
    id: str
    label: str
    physical_occlusion: bool
    interpretation: Literal["occlusion", "separate_photometric"]
    real_counterparts: tuple[str, ...] = ()


class HeldoutSection(_Strict):
    scheme: Literal["leave_one_family_out"]
    isolation_scope: frozenset[str]

    @field_validator("isolation_scope")
    @classmethod
    def _covers_required(cls, v: frozenset[str]) -> frozenset[str]:
        missing = REQUIRED_ISOLATION_STAGES - v
        if missing:
            raise ValueError(f"isolation_scope is missing required stages: {sorted(missing)}")
        return v


class SplitSection(_Strict):
    unit: Literal["identity"]
    hash_function: Literal["sha256"]
    key_template: Literal["{dataset}:{identity_id}"]
    bands: dict[SplitName, tuple[int, int]]
    variants_inherit_source_split: Literal[True]

    @field_validator("bands")
    @classmethod
    def _contiguous_cover(
        cls, v: dict[SplitName, tuple[int, int]]
    ) -> dict[SplitName, tuple[int, int]]:
        if set(v) != set(SplitName):
            raise ValueError("bands must define exactly train, val, cal and test")
        intervals = sorted(v.values())
        if intervals[0][0] != 0 or intervals[-1][1] != 100:
            raise ValueError("bands must cover [0, 100)")
        for (_, hi), (nlo, _) in pairwise(intervals):
            if hi != nlo:
                raise ValueError("bands must be contiguous and non-overlapping")
        if any(lo >= hi for lo, hi in intervals):
            raise ValueError("each band must be non-empty (lo < hi)")
        return v


class PairSection(_Strict):
    strata: tuple[
        Literal[
            "clean_clean",
            "clean_occluded",
            "occluded_occluded_same_family",
            "occluded_occluded_cross_family",
        ],
        ...,
    ]
    genuine_fraction_constructed_sets: float = Field(gt=0, lt=1)
    genuine_pairs_per_identity_cap: int = Field(ge=1)
    min_impostor_pairs_factor: float = Field(ge=1)
    primary_calibration_pairing: Literal["identity_disjoint_matching"]
    sensitivity_calibration_pairing: tuple[Literal["one_pair_per_identity_pair", "all_pairs"], ...]
    official_benchmark_pairs: Literal["report_as_is_and_separately"]


class CalibrationSection(_Strict):
    method: Literal["learn_then_test"]
    prediction_target: str = Field(min_length=20)
    controlled_risk: Literal["selective_error_among_accepted"]
    secondary_controlled_risks: tuple[Literal["selective_fmr", "selective_fnmr"], ...]
    pvalue: Literal["binomial_exact", "hoeffding_bentkus"]
    fwer_procedure: Literal["fixed_sequence", "bonferroni"]
    lambda_grid: Literal["tau_low_tau_high_cmin"]
    stratification_arms: tuple[Literal["global", "visibility_strata", "oracle_family"], ...]
    assumptions: tuple[str, ...] = Field(min_length=3)
    guarantee_scope_statement: str = Field(min_length=40)


class MetricSection(_Strict):
    primary_selective_metric: Literal["augrc"]
    report: tuple[str, ...]
    tar_at_far_modes: tuple[
        Literal["benchmark_threshold_on_test", "deployment_threshold_from_cal"], ...
    ]


class BootstrapSection(_Strict):
    unit: Literal["identity"]
    resamples: int = Field(ge=1000)
    paired_differences: Literal[True]
    validated_by_simulation_before_use: Literal[True]


class ConstantSpec(_Strict):
    value: float | int | tuple[float, ...]
    status: ConstantStatus
    rationale: str = Field(min_length=10)
    feasibility: str | None = None


class ConstantsSection(_Strict):
    clean_noninferiority_margin_pp: ConstantSpec
    cost_ratio_false_match_to_false_non_match: ConstantSpec
    confidence_level: ConstantSpec
    accepted_decision_error_target: ConstantSpec
    far_operating_points: ConstantSpec
    bootstrap_ci_level: ConstantSpec

    @model_validator(mode="after")
    def _ranges(self) -> ConstantsSection:
        def scalar(c: ConstantSpec) -> float:
            if isinstance(c.value, tuple):
                raise ValueError("expected a scalar constant")
            return float(c.value)

        if not 0 < scalar(self.clean_noninferiority_margin_pp) < 100:
            raise ValueError("clean_noninferiority_margin_pp must be in (0, 100)")
        if scalar(self.cost_ratio_false_match_to_false_non_match) <= 0:
            raise ValueError("cost ratio must be positive")
        for c in (self.confidence_level, self.bootstrap_ci_level):
            if not 0.5 < scalar(c) < 1:
                raise ValueError("confidence levels must be in (0.5, 1)")
        if not 0 < scalar(self.accepted_decision_error_target) < 0.5:
            raise ValueError("accepted_decision_error_target must be in (0, 0.5)")
        fars = self.far_operating_points.value
        if not isinstance(fars, tuple) or not fars:
            raise ValueError("far_operating_points must be a non-empty list")
        if any(not 0 < f < 1 for f in fars) or len(set(fars)) != len(fars):
            raise ValueError("FAR operating points must be unique and in (0, 1)")
        return self

    def all_specs(self) -> dict[str, ConstantSpec]:
        return {name: getattr(self, name) for name in type(self).model_fields}


class NearDuplicateSection(_Strict):
    method: Literal["phash_and_embedding_cosine"]
    threshold: float | None
    validated_on_split: Literal["train", "val"] | None
    validation_record: str | None
    target_duplicate_recall: ConstantSpec

    @model_validator(mode="after")
    def _threshold_requires_validation(self) -> NearDuplicateSection:
        if self.threshold is not None and (
            self.validated_on_split is None or not self.validation_record
        ):
            raise ValueError(
                "near-duplicate threshold cannot be fixed without validated_on_split "
                "and validation_record (non-test data only)"
            )
        return self


class DeduplicationSection(_Strict):
    exact: Literal["sha256_of_decoded_pixels"]
    near_duplicate: NearDuplicateSection
    removal_side: Literal["non_test"]


class TestAccessSection(_Strict):
    require_final_flag: Literal[True]
    run_log: str
    real_benchmarks_final_only: Literal[True]


# --------------------------------------------------------------------------- root


class ResearchProtocol(_Strict):
    schema_version: Literal[1]
    protocol_id: str
    protocol_version: str
    status: ConstantStatus
    project: ProjectSection
    seeds: SeedSection
    backbones: BackboneSection
    detector: DetectorSection
    challenge_families: tuple[ChallengeFamily, ...]
    heldout: HeldoutSection
    splits: SplitSection
    pairs: PairSection
    calibration: CalibrationSection
    metrics: MetricSection
    bootstrap: BootstrapSection
    constants: ConstantsSection
    deduplication: DeduplicationSection
    test_access: TestAccessSection

    @field_validator("challenge_families")
    @classmethod
    def _approved_families(cls, v: tuple[ChallengeFamily, ...]) -> tuple[ChallengeFamily, ...]:
        ids = tuple(f.id for f in v)
        if ids != APPROVED_FAMILY_IDS:
            raise ValueError(f"challenge families must be exactly {APPROVED_FAMILY_IDS}, got {ids}")
        for fam in v:
            if fam.physical_occlusion and fam.interpretation != "occlusion":
                raise ValueError(f"{fam.id}: physical occlusions use interpretation 'occlusion'")
            if not fam.physical_occlusion and fam.interpretation != "separate_photometric":
                raise ValueError(
                    f"{fam.id}: non-physical challenges must be interpreted separately"
                )
        lighting = v[-1]
        if lighting.physical_occlusion:
            raise ValueError("F5_extreme_lighting must be flagged physical_occlusion: false")
        return v

    @model_validator(mode="after")
    def _frozen_means_all_frozen(self) -> ResearchProtocol:
        if self.status is ConstantStatus.FROZEN:
            provisional = [
                n
                for n, c in self.constants.all_specs().items()
                if c.status is ConstantStatus.PROVISIONAL
            ]
            if self.deduplication.near_duplicate.threshold is None:
                provisional.append("deduplication.near_duplicate.threshold")
            if provisional:
                raise ValueError(
                    f"protocol is 'frozen' but these remain provisional: {provisional}"
                )
        return self

    # -- helpers ---------------------------------------------------------------
    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(f.id for f in self.challenge_families)

    def seen_families(self, heldout_family: str) -> tuple[str, ...]:
        """Families usable for fitting/selection/calibration when ``heldout_family`` is tested."""
        if heldout_family not in self.family_ids:
            raise ValueError(f"unknown family {heldout_family!r}")
        return tuple(f for f in self.family_ids if f != heldout_family)

    def provisional_constants(self) -> list[str]:
        names = [
            n
            for n, c in self.constants.all_specs().items()
            if c.status is ConstantStatus.PROVISIONAL
        ]
        if self.deduplication.near_duplicate.threshold is None:
            names.append("deduplication.near_duplicate.threshold (unset until validated)")
        return names


# --------------------------------------------------------------------------- dataset registry


class AccessStatus(StrEnum):
    CONFIRMED = "confirmed"  # licence read from owner, access route known
    CONDITIONAL = "conditional"  # usable only after a named condition is met
    UNRESOLVED = "unresolved"  # licence/access not yet established
    BLOCKED = "blocked"  # must not be used


class DatasetRole(StrEnum):
    TRAIN = "train"
    VAL = "val"
    CAL = "cal"
    TEST_SYNTH = "test_synth"
    TEST_REAL = "test_real"
    VISIBILITY_TRAINING = "visibility_training"
    NONE = "none"


class DatasetEntry(_Strict):
    id: str
    name: str
    url: str
    licence: str
    licence_source: str
    licence_verified_on: str | None
    access_status: AccessStatus
    conditions: tuple[str, ...] = ()
    identity_labels: Literal["yes", "no", "unknown"]
    n_identities: int | None = None
    n_images: int | None = None
    images_per_identity: str | None = None
    download_size: str | None = None
    roles: tuple[DatasetRole, ...]
    test_only: bool
    contamination_risks: tuple[str, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _role_rules(self) -> DatasetEntry:
        active = [r for r in self.roles if r is not DatasetRole.NONE]
        if self.access_status is AccessStatus.BLOCKED and active:
            raise ValueError(f"{self.id}: blocked datasets cannot have active roles")
        fitting = {
            DatasetRole.TRAIN,
            DatasetRole.VAL,
            DatasetRole.CAL,
            DatasetRole.VISIBILITY_TRAINING,
        }
        if self.test_only and fitting.intersection(active):
            raise ValueError(
                f"{self.id}: test-only datasets cannot be used for fitting/calibration"
            )
        verification_roles = {
            DatasetRole.TRAIN,
            DatasetRole.VAL,
            DatasetRole.CAL,
            DatasetRole.TEST_SYNTH,
            DatasetRole.TEST_REAL,
        }
        if self.identity_labels != "yes" and verification_roles.intersection(active):
            raise ValueError(
                f"{self.id}: verification roles require identity labels (identity_labels='yes')"
            )
        if self.access_status is AccessStatus.CONDITIONAL and not self.conditions:
            raise ValueError(f"{self.id}: conditional access must list its conditions")
        return self


class DatasetRegistry(_Strict):
    schema_version: Literal[1]
    checked_on: str
    datasets: tuple[DatasetEntry, ...]

    @field_validator("datasets")
    @classmethod
    def _unique_ids(cls, v: tuple[DatasetEntry, ...]) -> tuple[DatasetEntry, ...]:
        ids = [d.id for d in v]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset ids must be unique")
        return v

    def get(self, dataset_id: str) -> DatasetEntry:
        for d in self.datasets:
            if d.id == dataset_id:
                return d
        raise KeyError(dataset_id)

    def usable_for(self, role: DatasetRole) -> tuple[DatasetEntry, ...]:
        """Datasets whose access is confirmed (not merely conditional) for ``role``."""
        return tuple(
            d
            for d in self.datasets
            if role in d.roles and d.access_status is AccessStatus.CONFIRMED
        )
