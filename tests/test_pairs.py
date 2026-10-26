from __future__ import annotations

import pytest

from veilx.config.schema import SplitName
from veilx.errors import InfeasibleRequestError, LeakageError, ProtocolViolationError
from veilx.pairs import PairSpec, assert_independent, build_pairs

from .helpers import rec, toy_images


def spec(**kw) -> PairSpec:  # type: ignore[no-untyped-def]
    base = {
        "mode": "independent",
        "n_pairs": 10,
        "genuine_fraction": 0.5,
        "split": SplitName.CAL,
        "seed": 1,
    }
    base.update(kw)
    return PairSpec(**base)  # type: ignore[arg-type]


def test_independent_pairs_are_identity_disjoint() -> None:
    images = toy_images(40, 3)
    ps = build_pairs(images, spec(n_pairs=12))
    s = ps.summary
    assert (s.n_genuine, s.n_impostor) == (6, 6)
    assert s.identity_disjoint and s.effective_independent_pairs == 12
    assert s.max_pairs_per_identity == 1 and s.max_pairs_per_image == 1
    assert s.n_identities == 6 + 2 * 6
    assert_independent(ps)


def test_determinism_and_provenance() -> None:
    images = toy_images(40, 3)
    a, b = build_pairs(images, spec()), build_pairs(images, spec())
    c = build_pairs(images, spec(seed=2))
    assert a.pairs == b.pairs
    assert a.provenance["pairs_digest"] == b.provenance["pairs_digest"]
    assert a.provenance["pairs_digest"] != c.provenance["pairs_digest"]
    assert a.provenance["manifest_digest"] == c.provenance["manifest_digest"]
    assert a.provenance["spec"]["seed"] == 1  # type: ignore[index]


def test_infeasible_independent_request_is_explicit() -> None:
    images = toy_images(10, 2)
    with pytest.raises(InfeasibleRequestError, match="identity-disjoint"):
        build_pairs(images, spec(n_pairs=8))  # needs 4 + 2*4 = 12 identities
    build_pairs(images, spec(n_pairs=6))  # 3 + 2*3 = 9 <= 10


def test_single_image_identities_cannot_be_genuine() -> None:
    images = toy_images(10, 1) + toy_images(1, 2)[:0]
    with pytest.raises(InfeasibleRequestError, match="genuine"):
        build_pairs(images, spec(n_pairs=2))
    ps = build_pairs(images, spec(n_pairs=4, genuine_fraction=0.0))
    assert ps.summary.n_genuine == 0 and ps.summary.n_impostor == 4


def test_variants_and_duplicates_never_form_genuine_pairs() -> None:
    images = [
        rec("a", "q1"),
        rec("a_mask", "q1", source="a", family="F1_mask"),  # variant of the same photo
        rec("a_copy", "q1", content="a"),  # byte-identical duplicate content
        *toy_images(4, 1),
    ]
    with pytest.raises(InfeasibleRequestError):
        build_pairs(images, spec(n_pairs=1, genuine_fraction=1.0))
    images.append(rec("b", "q1"))
    ps = build_pairs(images, spec(n_pairs=1, genuine_fraction=1.0))
    pair = ps.pairs[0]
    assert {pair.image_a, pair.image_b} & {"b"}


def test_cross_identity_duplicate_content_breaks_independence() -> None:
    """A photo mislabelled under two identities must not count as two independent observations."""
    images = [rec("a", "q1"), rec("b", "q2", content="a"), *toy_images(10, 2)]
    from veilx.pairs.builder import source_keys, summarize

    keys = source_keys(images)
    assert keys["a"] == keys["b"]
    from veilx.data.manifest import PairRecord

    pairs = [
        PairRecord(
            pair_id="x1",
            image_a="a",
            image_b="i0_0",
            label="impostor",
            stratum="clean_clean",
            split=SplitName.CAL,
        ),
        PairRecord(
            pair_id="x2",
            image_a="b",
            image_b="i1_0",
            label="impostor",
            stratum="clean_clean",
            split=SplitName.CAL,
        ),
    ]
    s = summarize(pairs, images, "independent")
    assert s.identity_disjoint and s.max_pairs_per_source_image == 2
    from veilx.pairs import PairSet

    with pytest.raises(ProtocolViolationError):
        assert_independent(PairSet(tuple(pairs), s))


def test_bernoulli_class_assignment() -> None:
    images = toy_images(400, 2)
    ps = build_pairs(images, spec(n_pairs=100, class_assignment="bernoulli", seed=4))
    assert 0 < ps.summary.n_genuine < 100
    assert ps.provenance["spec"]["class_assignment"] == "bernoulli"  # type: ignore[index]


def test_clustered_pairs_are_not_independent() -> None:
    images = toy_images(8, 6)
    ps = build_pairs(images, spec(mode="clustered", n_pairs=60, genuine_cap_per_identity=4))
    s = ps.summary
    assert not s.identity_disjoint and s.effective_independent_pairs is None
    assert s.max_identity_disjoint_subset <= 8
    assert "NOT independent" in s.note
    with pytest.raises(ProtocolViolationError):
        assert_independent(ps)


def test_clustered_genuine_cap_and_capacity() -> None:
    images = toy_images(5, 6)
    with pytest.raises(InfeasibleRequestError, match="genuine"):
        build_pairs(
            images,
            spec(mode="clustered", n_pairs=40, genuine_fraction=1.0, genuine_cap_per_identity=2),
        )  # max 10
    with pytest.raises(InfeasibleRequestError, match="impostor"):
        build_pairs(toy_images(2, 1), spec(mode="clustered", n_pairs=2, genuine_fraction=0.0))


def test_clustered_impostor_identity_cap() -> None:
    images = toy_images(6, 3)
    ps = build_pairs(
        images,
        spec(mode="clustered", n_pairs=6, genuine_fraction=0.0, max_impostor_pairs_per_identity=2),
    )
    assert ps.summary.max_pairs_per_identity <= 2
    with pytest.raises(InfeasibleRequestError):
        build_pairs(
            images,
            spec(
                mode="clustered",
                n_pairs=10,
                genuine_fraction=0.0,
                max_impostor_pairs_per_identity=2,
            ),
        )


def test_large_identities_use_sampling_path() -> None:
    images = toy_images(30, 72)
    ps = build_pairs(images, spec(mode="clustered", n_pairs=200, genuine_cap_per_identity=5))
    assert ps.summary.n_genuine == 100
    assert len({(p.image_a, p.image_b) for p in ps.pairs}) == 200


def test_split_purity_enforced() -> None:
    images = [*toy_images(10, 2), rec("x", "q1", SplitName.TEST)]
    with pytest.raises(LeakageError):
        build_pairs(images, spec(n_pairs=2))


@pytest.mark.parametrize(
    "bad",
    [
        {"n_pairs": 0},
        {"genuine_fraction": 1.5},
        {"genuine_cap_per_identity": 0},
        {"max_impostor_pairs_per_identity": 0},
    ],
)
def test_spec_validation(bad: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        spec(**bad)
