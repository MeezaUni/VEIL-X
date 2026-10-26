"""Verification, selective and calibration metrics (definitions: docs/metrics.md)."""

from __future__ import annotations

from veilx.metrics.calibration import (
    ReliabilityDiagram,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
    reliability_svg,
)
from veilx.metrics.roc import (
    EERResult,
    OperatingPoint,
    ThresholdEvaluation,
    deployment_operating_point,
    equal_error_rate,
    evaluate_at_threshold,
    required_impostors,
    roc_auc,
    roc_auc_trapezoid,
    roc_curve,
    tar_at_far,
    threshold_at_far,
)
from veilx.metrics.selective import (
    ABSTAIN,
    MATCH,
    NON_MATCH,
    DecisionSummary,
    RiskCoverageCurve,
    augrc,
    aurc,
    decisions_from_thresholds,
    risk_coverage_curve,
    summarize_decisions,
    verification_failures,
)

__all__ = [
    "ABSTAIN",
    "MATCH",
    "NON_MATCH",
    "DecisionSummary",
    "EERResult",
    "OperatingPoint",
    "ReliabilityDiagram",
    "RiskCoverageCurve",
    "ThresholdEvaluation",
    "augrc",
    "aurc",
    "brier_score",
    "decisions_from_thresholds",
    "deployment_operating_point",
    "equal_error_rate",
    "evaluate_at_threshold",
    "expected_calibration_error",
    "reliability_diagram",
    "reliability_svg",
    "required_impostors",
    "risk_coverage_curve",
    "roc_auc",
    "roc_auc_trapezoid",
    "roc_curve",
    "summarize_decisions",
    "tar_at_far",
    "threshold_at_far",
    "verification_failures",
]
