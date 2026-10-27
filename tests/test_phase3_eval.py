"""Phase 3 evaluation code path on tiny synthetic feature stores (no data, no torch)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from veilx.data.manifest import PairRecord
from veilx.embeddings.jobs import occluded_key
from veilx.fusion.features import LADDER
from veilx.fusion.logreg import LogisticModel

from .conftest import ROOT


@pytest.fixture(scope="module")
def ev() -> Any:
    spec = importlib.util.spec_from_file_location(
        "evaluate_phase3", ROOT / "scripts" / "evaluate_phase3.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _store(ev: Any, keys: list[str], rng: np.random.Generator, occluded: bool) -> Any:
    n = len(keys)
    g = object.__new__(ev.Group)
    g.keys = np.array(keys)
    emb = rng.normal(size=(n, 512)).astype(np.float16)
    g.arr = {
        "emb_ada": emb,
        "emb_arc": (emb + 0.1 * rng.normal(size=emb.shape)).astype(np.float16),
        "norm_ada": rng.uniform(15, 25, n).astype(np.float32),
        "norm_arc": rng.uniform(15, 25, n).astype(np.float32),
        "z": rng.uniform(0, 4, (n, 28, 28)).astype(np.float16),
    }
    vmap = rng.integers(100, 256, (n, 28, 28)).astype(np.uint8)
    if occluded:
        g.arr |= {"ve": vmap, "gt": vmap, "vis": rng.uniform(0.3, 0.9, n).astype(np.float32)}
    else:
        g.arr["ve_F1"] = np.full((n, 28, 28), 250, np.uint8)
    g.index = {k: i for i, k in enumerate(keys)}
    return g


def test_table_arms_and_endpoints_run(ev: Any) -> None:
    rng = np.random.default_rng(0)
    n_id = 120
    imgs = [f"digiface:i{i:04d}:{j}" for i in range(n_id) for j in range(2)]
    ident = {im: im.split(":")[1] for im in imgs}
    pairs, rows = [], {}
    for i in range(n_id):
        a, b = f"digiface:i{i:04d}:0", f"digiface:i{i:04d}:1"
        c = f"digiface:i{(i + 1) % n_id:04d}:0"
        for k, (x, y, lab) in enumerate(((a, b, "genuine"), (b, c, "impostor"))):
            pid = f"p{i:04d}{k}"
            pairs.append(
                PairRecord.model_construct(
                    pair_id=pid,
                    image_a=x,
                    image_b=y,
                    label=lab,
                    stratum="clean_clean",
                    split="test",
                )
            )
            spec = {
                "family": "F1_mask",
                "severity_level": "medium",
                "seed": i * 2 + k,
                "visibility_ratio": 0.6,
            }
            rows[pid] = {"stratum": "clean_occluded", "a": None, "b": spec if i % 2 == 0 else None}
    occ_keys = [
        occluded_key(p.image_b, "F1_mask", "medium", rows[p.pair_id]["b"]["seed"])
        for p in pairs
        if rows[p.pair_id]["b"] is not None
    ]
    clean, occ = _store(ev, imgs, rng, False), _store(ev, occ_keys, rng, True)
    link = LogisticModel(("anomaly",), 1e6).fit(
        rng.uniform(0, 4, (400, 1)), (rng.random(400) < 0.5).astype(np.int64)
    )
    t = ev.build_table(pairs, rows, clean, occ, "F1_mask", link, ident)
    tc = ev.build_table(pairs, None, clean, None, "F1_mask", link, ident)
    assert len(t["s"]) == len(pairs) and t["occluded"].sum() == len(occ_keys)
    assert np.all(tc["gt_engine_covis"] == 1.0) and not tc["occluded"].any()
    cv_def = "cv_spatial"
    y = t["label"]
    models = {}
    for m in LADDER:
        x, names = ev.design(t, m, cv_def)
        models[m] = LogisticModel(names, 1.0).fit(x, y)
    sc = {m: mdl.predict_proba(ev.design(t, m, cv_def)[0]) for m, mdl in models.items()}
    scc = {m: mdl.predict_proba(ev.design(tc, m, cv_def)[0]) for m, mdl in models.items()}
    edges = {
        "q": [18.0, 22.0],
        "cv": [0.4, 0.7],
        "q_med": 20.0,
        "cv_med": 0.6,
        "oracle": 0.6,
        "disagree_p95": 0.2,
    }
    arms = {}
    for arm, (mname, kind) in ev.ARMS.items():
        st = sorted(set(ev.strata_labels(kind, t, edges, cv_def).tolist()))
        for tn in ev.TARGETS:
            for o in ev.ORDERINGS:
                arms[f"{arm}|{tn}|{o}"] = {
                    "model": mname,
                    "strata": kind,
                    "rules": {s: [0.3, 0.7] for s in st},
                }
    ff = {
        "arms": arms,
        "edges": edges,
        "eer_threshold": {"M0": 0.5, "M4": 0.5},
        "h4_thresholds": {"M0": 0.5, "M4": 0.5},
    }
    block = ev.evaluate_occluded(t, sc, ff, cv_def)
    assert set(block) >= {"models", "arms", "c3_A4_minus_A3", "c1_logloss_M1_minus_M4"}
    assert "abstention_reasons_nonexclusive" in block["arms"]["A4_predvis|primary|v2"]
    assert 0 <= block["arms"]["A1_clean|primary|v2"]["coverage"] <= 1
    h4 = ev.evaluate_clean_h4(tc, scc, ff)
    assert set(h4) >= {"fnmr_M4", "fnmr_M0", "delta_fnmr_M4_minus_M0", "non_inferior_at_0.5pp"}
    sub = ev._subset(t, t["occluded"])
    assert len(sub["s"]) == len(occ_keys)


def test_strata_labels_cover_all_pairs(ev: Any) -> None:
    t = {
        "s": np.zeros(4),
        "q": np.array([10.0, 19.0, 21.0, 30.0]),
        "ve_cv_min": np.array([0.1, 0.5, 0.8, 0.9]),
        "gt_engine_covis": np.array([1.0, 0.7, 0.2, 1.0]),
    }
    edges = {"q": [18.0, 22.0], "cv": [0.4, 0.7], "q_med": 20.0, "cv_med": 0.6, "oracle": 0.5}
    assert ev.strata_labels("quality", t, edges, "cv_min").tolist() == [
        "q_low",
        "q_mid",
        "q_mid",
        "q_high",
    ]
    assert ev.strata_labels("predvis", t, edges, "cv_min").tolist() == [
        "v_low",
        "v_mid",
        "v_high",
        "v_high",
    ]
    assert ev.strata_labels("oracle", t, edges, "cv_min").tolist() == [
        "clean",
        "vis_high",
        "vis_low",
        "clean",
    ]
    assert ev.strata_labels("q_x_v", t, edges, "cv_min").tolist() == [
        "qL_vL",
        "qL_vL",
        "qH_vH",
        "qH_vH",
    ]
