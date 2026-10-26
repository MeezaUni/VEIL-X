"""Pair construction (genuine/impostor), independence accounting and provenance."""

from __future__ import annotations

from veilx.pairs.builder import (
    PAIR_GENERATOR_VERSION,
    PairSet,
    PairSetSummary,
    PairSpec,
    assert_independent,
    build_pairs,
    summarize,
)

__all__ = [
    "PAIR_GENERATOR_VERSION",
    "PairSet",
    "PairSetSummary",
    "PairSpec",
    "assert_independent",
    "build_pairs",
    "summarize",
]
