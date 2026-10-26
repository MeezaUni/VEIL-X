"""Interface contracts for every VEIL-X module (Phase 0: definitions only).

Each protocol states *what* a component must provide so that baselines and proposed
methods are interchangeable and individually ablatable. No implementation lives here.
Array shape conventions: ``N`` images, ``P`` pairs, ``D`` embedding dim, ``K`` grid cells.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from veilx.data.manifest import ImageRecord, PairRecord

FloatArray = NDArray[np.float32] | NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


# --------------------------------------------------------------------------- decisions


class Decision(StrEnum):
    MATCH = "match"
    NON_MATCH = "non_match"
    ABSTAIN = "abstain"


class AbstainReason(StrEnum):
    INSUFFICIENT_COVISIBILITY = "insufficient_covisibility"
    LOW_QUALITY = "low_quality"
    AMBIGUOUS_SCORE = "ambiguous_score"
    ALIGNMENT_FAILED = "alignment_failed"


# --------------------------------------------------------------------------- data


@runtime_checkable
class DatasetAdapter(Protocol):
    """Turns a locally stored, licence-approved dataset into :class:`ImageRecord` rows."""

    dataset_id: str

    def check_access(self) -> None:
        """Raise ``DataAccessError`` unless the registry marks the dataset usable."""

    def iter_images(self) -> Iterator[ImageRecord]:
        """Yield records with dataset-relative paths and pixel hashes (no pixels kept)."""

    def official_pairs(self) -> Sequence[PairRecord]:
        """Official benchmark pairs, if the dataset defines a protocol (else empty)."""


@dataclass(frozen=True)
class AlignmentResult:
    ok: bool
    crop: NDArray[np.uint8] | None  # (112, 112, 3) RGB when ok
    landmarks: FloatArray | None  # (5, 2) in crop coordinates
    detector_score: float | None
    failure_reason: str | None = None


@runtime_checkable
class FaceAligner(Protocol):
    detector_id: str

    def align(self, image: NDArray[np.uint8]) -> AlignmentResult: ...


# --------------------------------------------------------------------------- features


@dataclass(frozen=True)
class EmbeddingBatch:
    backbone_id: str
    embeddings: FloatArray  # (N, D), L2-normalised
    feature_norms: FloatArray  # (N,), pre-normalisation norm (quality proxy)
    spatial: FloatArray | None = None  # (N, C, H, W) last conv block, optional


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Frozen backbone wrapper. Must be deterministic in eval mode."""

    backbone_id: str
    embedding_dim: int

    def embed(self, crops: NDArray[np.uint8]) -> EmbeddingBatch: ...


@dataclass(frozen=True)
class VisibilityBatch:
    estimator_id: str
    cell_visibility: FloatArray  # (N, K) in [0, 1]
    region_visibility: FloatArray  # (N, R) in [0, 1], R named regions
    region_names: tuple[str, ...]
    trained_on_families: tuple[str, ...]  # provenance for held-out isolation checks


@runtime_checkable
class VisibilityEstimator(Protocol):
    estimator_id: str
    trained_on_families: tuple[str, ...]

    def estimate(self, crops: NDArray[np.uint8]) -> VisibilityBatch: ...


@runtime_checkable
class QualityEstimator(Protocol):
    quality_id: str

    def quality(self, crops: NDArray[np.uint8], embeddings: EmbeddingBatch) -> FloatArray: ...


@dataclass(frozen=True)
class PairFeatures:
    names: tuple[str, ...]
    values: FloatArray  # (P, F)
    pair_ids: tuple[str, ...]


@runtime_checkable
class PairFeatureGenerator(Protocol):
    """Builds the pair feature vector. Feature sets are named so ablations are explicit:
    ``quality_only``, ``covisibility_only``, ``quality_plus_covisibility``, ... ."""

    feature_set: str
    feature_names: tuple[str, ...]

    def generate(
        self,
        pairs: Sequence[PairRecord],
        embeddings: Mapping[str, EmbeddingBatch],
        visibility: Mapping[str, VisibilityBatch] | None,
        quality: Mapping[str, FloatArray] | None,
    ) -> PairFeatures: ...


# --------------------------------------------------------------------------- risk control


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    stratification: str
    parameters: Mapping[str, float]  # e.g. tau_low, tau_high, c_min (per stratum if stratified)
    target_risk: float
    delta: float
    n_calibration_pairs: int
    n_accepted_calibration_pairs: int
    calibration_families: tuple[str, ...]
    assumptions: tuple[str, ...]
    guarantee_scope: str  # states that validity requires cal/test exchangeability
    feasible: bool  # False if no lambda passed the test (system then abstains on all)


@runtime_checkable
class RiskCalibrator(Protocol):
    calibrator_id: str

    def fit(
        self,
        scores: FloatArray,
        labels: IntArray,
        strata: Sequence[str] | None,
        calibration_families: tuple[str, ...],
    ) -> CalibrationResult: ...

    def decide(
        self, scores: FloatArray, strata: Sequence[str] | None
    ) -> tuple[list[Decision], list[AbstainReason | None]]: ...


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class MetricValue:
    name: str
    value: float
    ci_low: float | None
    ci_high: float | None
    n_pairs: int
    n_identities: int
    note: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    method_id: str
    split: str
    condition: str  # e.g. "heldout=F1_mask", "real=ROF_masked"
    metrics: tuple[MetricValue, ...]
    composition: Mapping[str, int]  # genuine/impostor counts, per stratum
    coverage: Mapping[str, float]  # overall, genuine, impostor
    extra: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        method_id: str,
        decisions: Sequence[Decision],
        scores: FloatArray,
        labels: IntArray,
        identity_pairs: Sequence[tuple[str, str]],
        condition: str,
    ) -> EvaluationResult: ...


@runtime_checkable
class ResultReporter(Protocol):
    def report(self, results: Sequence[EvaluationResult], provenance: Mapping[str, object]) -> str:
        """Persist results + provenance; return a relative path or run id."""
