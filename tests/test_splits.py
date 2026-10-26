from __future__ import annotations

import random

import pytest

from veilx.config import load_protocol
from veilx.config.schema import SplitName
from veilx.data.splits import assign_split, identity_bucket, split_counts

from .conftest import PROTOCOL

pytestmark = pytest.mark.structural

BANDS = load_protocol(PROTOCOL).splits.bands


def test_assignment_is_deterministic_and_order_independent() -> None:
    ids = [f"id{i}" for i in range(500)]
    first = {i: assign_split("digiface1m", i, BANDS) for i in ids}
    shuffled = ids[:]
    random.Random(7).shuffle(shuffled)
    second = {i: assign_split("digiface1m", i, BANDS) for i in shuffled}
    assert first == second


def test_dataset_namespacing() -> None:
    # Same identity string in two datasets is hashed independently.
    buckets = {identity_bucket(ds, "42") for ds in ("a", "b", "c", "d", "e", "f")}
    assert len(buckets) > 1


def test_band_proportions_roughly_match() -> None:
    counts = split_counts("toy", (str(i) for i in range(20_000)), BANDS)
    n = sum(counts.values())
    expected = {SplitName.TRAIN: 0.6, SplitName.VAL: 0.1, SplitName.CAL: 0.1, SplitName.TEST: 0.2}
    for split, frac in expected.items():
        assert abs(counts[split] / n - frac) < 0.02


def test_every_bucket_is_covered() -> None:
    for b in range(100):
        assert any(lo <= b < hi for lo, hi in BANDS.values())
