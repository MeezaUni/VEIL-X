"""Selective decision rules and risk control (see docs/ltt_audit.md)."""

from __future__ import annotations

from veilx.selective.ltt import (
    ASSUMPTIONS,
    CandidateStats,
    EmpiricalCalibrator,
    LTTCalibrator,
    RiskControlResult,
    StratifiedResult,
    ThresholdRule,
    binomial_pvalue,
    fit_stratified,
    symmetric_rules,
)

__all__ = [
    "ASSUMPTIONS",
    "CandidateStats",
    "EmpiricalCalibrator",
    "LTTCalibrator",
    "RiskControlResult",
    "StratifiedResult",
    "ThresholdRule",
    "binomial_pvalue",
    "fit_stratified",
    "symmetric_rules",
]
