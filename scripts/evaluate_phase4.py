# ruff: noqa: E501  (long result-record literals)
"""Phase 4 develop / rehearse / final (docs/phase4_protocol.md). Run in .venv-gpu.

develop          DEVELOPMENT data only (p3_train / p3_val / p3_cal). Per fold:
                 - VE-C temperature T and lower-bound shift c (p3_val, seen families);
                 - Mahalanobis novelty (Gaussian on p3_train renders; threshold = 95th percentile
                   of p3_val in-distribution PAIR scores);
                 - fusion: logistic M1 / M4c / M5c / M5_noCV, plus MLP (selection on val);
                 - arms B1, C4, N4, H4 calibrated on p3_cal (orders from p3_val).
                 B0 / A3 / A4 / A5 are the FROZEN Phase 3 rules (34409779...).
                 Frozen with SHA-256.
final --rehearse the identical evaluation path on p3_val tables (no test access, not logged).
final --final    the single logged p4_test run: Test-OOD (5 folds) and E1 (5 folds) primary
                 cells; Test-ID secondary; P1 safety, P2 benefit, clean margin, secondary metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from veilx.config import load_protocol
from veilx.data.manifest import ImageRecord, PairRecord, read_jsonl
from veilx.data.paths import resolve_data_root
from veilx.embeddings.jobs import occluded_key
from veilx.fusion.features import LADDER_P4, covis_columns, design
from veilx.fusion.logreg import LogisticModel, log_loss, select_c
from veilx.metrics.roc import equal_error_rate, roc_auc, threshold_at_far
from veilx.metrics.selective import augrc
from veilx.provenance import collect_provenance
from veilx.selective.hybrid import hybrid_score, hybrid_strata, novelty_forced_strata, tertile
from veilx.selective.ltt import LTTCalibrator, fit_stratified
from veilx.stats.jackknife import jackknife_interval
from veilx.stats.paired import holm, jackknife_ratio_difference, normal_p_value
from veilx.vis.conservative import (
    MahalanobisNovelty,
    conservative_map,
    fit_temperature,
    global_visible,
    lower_bound_shift_maps,
    pair_max,
    tta_disagreement,
)

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
P4 = PROTO.phase4
assert P4 is not None
FAMILIES = tuple(PROTO.family_ids)
_spec = importlib.util.spec_from_file_location(
    "evaluate_phase3", ROOT / "scripts" / "evaluate_phase3.py"
)
assert _spec is not None and _spec.loader is not None
ev3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev3)
TARGETS, DELTA, ORDERINGS = ev3.TARGETS, ev3.DELTA, ev3.ORDERINGS
P3_ARMS = {
    "B0_raw_global": "A2_global_M0",
    "A3_quality": "A3_quality",
    "A4_original": "A4_predvis",
    "A5_oracle": "A5_oracle",
}
P4_ARMS = ("B1_quality_global", "C4_conservative", "N4_novelty_aware", "H4_quality_first_hybrid")
CANDIDATES = ("C4_conservative", "N4_novelty_aware", "H4_quality_first_hybrid")
CV = "cv_region"  # Phase 3 registered selection (D-047); unchanged in Phase 4
SUMMARY_KEYS = (
    "coverage",
    "coverage_genuine",
    "coverage_impostor",
    "selective_fmr",
    "selective_fnmr",
    "verdict_fm",
    "verdict_fnm",
)


def _sha(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------- feature joins


def attach(group: Any, src: Any, mapping: dict[str, str], *, allow_missing: bool = False) -> None:
    """Copy columns of ``src`` into ``group`` aligned by key (src may hold more keys). With
    ``allow_missing`` keys absent from ``src`` get zeros; callers must never read those rows
    (follow-up tables only touch images present in ``src``)."""
    keys = [str(k) for k in group.keys.tolist()]
    if not allow_missing:
        idx = np.array([src.index[k] for k in keys], dtype=np.int64)
        for dst, col in mapping.items():
            group.arr[dst] = src.arr[col][idx]
        return
    have = np.array([k in src.index for k in keys], dtype=bool)
    idx = np.array([src.index[k] for k, h in zip(keys, have, strict=True) if h], dtype=np.int64)
    for dst, col in mapping.items():
        ref = src.arr[col]
        out = np.zeros((len(keys), *ref.shape[1:]), ref.dtype)
        out[have] = ref[idx]
        group.arr[dst] = out
    group.arr["_p4_present"] = have


def side_rows(
    pairs: list[PairRecord], rows: dict[str, Any] | None, clean: Any, occ: Any
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pairs)
    store, row = np.zeros((2, n), np.int8), np.zeros((2, n), np.int64)
    for k, p in enumerate(pairs):
        r = rows[p.pair_id] if rows is not None else None
        for j, (side, img) in enumerate((("a", p.image_a), ("b", p.image_b))):
            spec = r[side] if r is not None else None
            if spec is None:
                row[j, k] = clean.index[img]
            else:
                store[j, k] = 1
                row[j, k] = occ.index[
                    occluded_key(img, spec["family"], spec["severity_level"], int(spec["seed"]))
                ]
    return store, row


def gather(
    clean: Any, occ: Any, col: str, store: np.ndarray, row: np.ndarray, j: int, sl: slice
) -> np.ndarray:
    st, rw = store[j, sl], row[j, sl]
    ref = clean.arr[col]
    out = np.empty((len(rw), *ref.shape[1:]), np.float64)
    out[st == 0] = ref[rw[st == 0]]
    if (st == 1).any():
        out[st == 1] = occ.arr[col][rw[st == 1]]
    return out


def extend_table(
    t: dict[str, np.ndarray],
    pairs: list[PairRecord],
    rows: dict[str, Any] | None,
    clean: Any,
    occ: Any,
    temp: float,
    shift: float,
    nov: MahalanobisNovelty | None,
) -> None:
    """Add VE-C conservative co-visibility (prefix vc_), novelty pair score and TTA disagreement."""
    store, row = side_rows(pairs, rows, clean, occ)
    present = clean.arr.get("_p4_present")
    if present is not None and not present[row[store == 0]].all():
        raise RuntimeError("table uses clean images missing from the Phase 4 extraction")
    n = len(pairs)
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for i in range(0, n, 8192):
        sl = slice(i, min(n, i + 8192))
        lg = [gather(clean, occ, "vec", store, row, j, sl) for j in (0, 1)]
        maps = [conservative_map(x, temp, shift) for x in lg]
        for k, v in covis_columns(maps[0], maps[1], "vc").items():
            parts[k].append(v)
        lgf = [gather(clean, occ, "vecf", store, row, j, sl) for j in (0, 1)]
        parts["tta"].append(
            pair_max(tta_disagreement(lg[0], lgf[0]), tta_disagreement(lg[1], lgf[1]))
        )
        if nov is not None:
            ft = [gather(clean, occ, "feat", store, row, j, sl) for j in (0, 1)]
            sa, sb = nov.score(ft[0]), nov.score(ft[1])
            parts["nov"].append(pair_max(sa, sb))
            parts["nov_a"].append(sa)
            parts["nov_b"].append(sb)
            parts["side_b_occluded"].append(store[1, sl] == 1)
            parts["side_a_occluded"].append(store[0, sl] == 1)
    for k, v in parts.items():
        t[k] = np.concatenate(v)


# ----------------------------------------------------------------- MLP fusion


def train_mlp(
    xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray, seed: int
) -> dict[str, Any]:
    import torch

    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt = (
        torch.tensor((xtr - mu) / sd, dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32),
    )
    xv = torch.tensor((xva - mu) / sd, dtype=torch.float32)
    best_v, bad, best_state = 1e9, 0, None
    g = torch.Generator().manual_seed(seed)
    for _ep in range(200):
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 1024):
            b = perm[i : i + 1024]
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b])[:, 0], yt[b]).backward()
            opt.step()
        with torch.no_grad():
            pv = torch.sigmoid(net(xv)[:, 0]).numpy().astype(np.float64)
        v = float(log_loss(pv, yva).mean())
        if v < best_v - 1e-5:
            best_v, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    assert best_state is not None
    return {
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "val_log_loss": best_v,
        "state": {k: v.numpy().tolist() for k, v in best_state.items()},
    }


def mlp_predict(m: dict[str, Any], x: np.ndarray) -> np.ndarray:
    z = (x - np.asarray(m["mu"])) / np.asarray(m["sd"])
    s = m["state"]
    h = np.maximum(z @ np.asarray(s["0.weight"]).T + np.asarray(s["0.bias"]), 0.0)
    o = h @ np.asarray(s["2.weight"]).T + np.asarray(s["2.bias"])
    return np.asarray(1 / (1 + np.exp(-o[:, 0])), np.float64)


# ----------------------------------------------------------------- calibration of new arms


def calibrate(
    sv: np.ndarray,
    stv: np.ndarray,
    yv: np.ndarray,
    sc: np.ndarray,
    stc: np.ndarray,
    yc: np.ndarray,
    cal_set: Any,
    tname: str,
    ordk: str,
) -> dict[str, Any]:
    tfm, tfnm = TARGETS[tname]
    rules = ev3._grid(sv)
    names = sorted(set(stv.tolist()) | set(stc.tolist()))
    rec: dict[str, Any] = {"rules": {}}
    if names == ["all"]:
        o = ev3._order(ordk, rules, sv, yv, len(sc), tfm, tfnm, DELTA)
        r = (
            LTTCalibrator(o, target_fm=tfm, target_fnm=tfnm, delta=DELTA).fit(
                sc, yc, pair_set=cal_set
            )
            if o
            else None
        )
        rec["rules"]["all"] = (
            None if r is None or r.rule is None else [r.rule.tau_low, r.rule.tau_high]
        )
        rec["n_ordered"] = {"all": len(o)}
    else:
        ds = DELTA / len(names)
        by = {
            nm: ev3._order(
                ordk, rules, sv[stv == nm], yv[stv == nm], int((stc == nm).sum()), tfm, tfnm, ds
            )
            for nm in names
        }
        res = fit_stratified(
            LTTCalibrator,
            rules,
            sc,
            yc,
            stc.tolist(),
            cal_set,
            target_fm=tfm,
            target_fnm=tfnm,
            delta=DELTA,
            candidates_by_stratum=by,
        )
        for nm in names:
            rr = res.per_stratum.get(nm)
            rec["rules"][nm] = (
                None if rr is None or rr.rule is None else [rr.rule.tau_low, rr.rule.tau_high]
            )
        rec["n_ordered"] = {nm: len(by[nm]) for nm in names}
    rec["n_strata_certified"] = sum(v is not None for v in rec["rules"].values())
    return rec


def p4_scores(t: dict[str, np.ndarray], f4: dict[str, Any]) -> dict[str, np.ndarray]:
    sc = {
        m: LogisticModel.from_dict(d).predict_proba(design(t, m, CV, LADDER_P4)[0])
        for m, d in f4["models"].items()
    }
    sc["MLP"] = mlp_predict(f4["mlp"], design(t, "M5c", CV, LADDER_P4)[0])
    return sc


def p4_strata(
    arm: str, t: dict[str, np.ndarray], f4: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    """(strata labels, novelty flags) for a Phase 4 arm; also the score key used."""
    novel = t["nov"] > f4["novelty"]["threshold"]
    cv_e = tuple(f4["edges"]["vc"])
    if arm == "B1_quality_global":
        return np.full(len(t["s"]), "all"), novel, "M1"
    if arm == "C4_conservative":
        return tertile(t[f"vc_{CV}"], cv_e, "v"), novel, "M4c"  # type: ignore[arg-type]
    if arm == "N4_novelty_aware":
        return novelty_forced_strata(novel, t[f"vc_{CV}"], cv_e), novel, "M4c"  # type: ignore[arg-type]
    if arm == "H4_quality_first_hybrid":
        return (
            hybrid_strata(~novel, t[f"vc_{CV}"], t["q"], cv_e, tuple(f4["edges"]["q"])),
            novel,
            "H4",
        )  # type: ignore[arg-type]
    raise ValueError(arm)


def arm_score(key: str, sc: dict[str, np.ndarray], novel: np.ndarray) -> np.ndarray:
    return hybrid_score(~novel, sc["M4c"], sc["M1"]) if key == "H4" else sc[key]


# ----------------------------------------------------------------- data context


class Ctx:
    def __init__(self) -> None:
        self.c3 = ev3.Ctx()
        self.layout = self.c3.layout
        self.dev = self.layout.root / "features" / "phase4" / "p3dev"
        fz3 = json.loads((self.layout.private_reports / "phase3_frozen.json").read_text("utf-8"))
        if _sha({k: v for k, v in fz3.items() if k != "sha256"}) != fz3["sha256"]:
            raise RuntimeError("phase3 frozen config modified")
        self.fz3 = fz3

    def clean(self, fold: str) -> Any:
        c = self.c3.clean(fold)
        d = ev3.Group(
            self.dev, "p3_clean", tuple(f"{x}_{fold[:2]}" for x in ("vec", "vecf", "feat"))
        )
        # the Phase 3 clean store also holds p3_test images, absent from the development
        # extraction: attach partially; development code reads only rows marked present
        attach(
            c,
            d,
            {"vec": f"vec_{fold[:2]}", "vecf": f"vecf_{fold[:2]}", "feat": f"feat_{fold[:2]}"},
            allow_missing=True,
        )
        return c

    def occ(self, fold: str, role: str) -> Any:
        o = self.c3.occ(fold, role)
        d = ev3.Group(self.dev, f"{fold}__{role}__{ev3.TAGS[role]}", ("vec", "vecf", "feat"))
        attach(o, d, {"vec": "vec", "vecf": "vecf", "feat": "feat"})
        return o


# ----------------------------------------------------------------- develop


def develop() -> int:
    ctx = Ctx()
    cal_set = ctx.c3.cal_set()
    sub = ctx.c3.sub
    frozen: dict[str, Any] = {
        "protocol_version": PROTO.protocol_version,
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "phase3_frozen_sha256": ctx.fz3["sha256"],
        "cv_definition": CV,
        "folds": {},
    }
    for fold in FAMILIES:
        f3 = ctx.fz3["folds"][fold]
        link = LogisticModel.from_dict(f3["heuristic_link"])
        clean = ctx.clean(fold)
        occ = {r: ctx.occ(fold, r) for r in ("train", "val", "cal")}
        # 1) temperature and lower-bound shift on p3_val (seen families + clean val images).
        # Lighting renders are EXCLUDED: lighting is a corruption handled by quality (protocol
        # 2.7 / 4.2) and their Phase 3 GT counts darkened pixels as occluded, contradicting the
        # VE-C target (D-049).
        rng = np.random.default_rng(P4.seed)  # type: ignore[union-attr]
        ov = occ["val"]
        present = clean.arr["_p4_present"]
        fam_v = np.array([str(k).split("|")[1] for k in ov.keys.tolist()])
        phys = np.flatnonzero(fam_v != "F5_extreme_lighting")
        pick = rng.choice(phys, min(6000, len(phys)), replace=False)
        val_imgs = [
            i
            for i, s in sub.items()
            if s == "p3_val" and i in clean.index and present[clean.index[i]]
        ]
        cpick = rng.choice(len(val_imgs), min(2000, len(val_imgs)), replace=False)
        cidx = np.array([clean.index[val_imgs[i]] for i in cpick])
        lg = np.concatenate(
            [ov.arr["vec"][pick].astype(np.float64), clean.arr["vec"][cidx].astype(np.float64)]
        )
        vis_t = np.concatenate([ov.arr["gt"][pick] >= 128, np.ones((len(cidx), 28, 28), bool)])
        temp = fit_temperature(lg, vis_t)
        occl = ov.arr["gt"][pick].astype(np.float64) / 255
        g_true = global_visible(occl)
        shift = lower_bound_shift_maps(
            conservative_map(ov.arr["vec"][pick], temp, 0.0), g_true, margin=0.10, max_rate=0.05
        )
        # sanity (protocol 7): VE-C vs Phase 3 VE overestimation on val seen PHYSICAL families
        gvec = global_visible(conservative_map(ov.arr["vec"][pick], temp, shift))
        gve3 = global_visible(ov.arr["ve"][pick].astype(np.float64) / 255)
        bias_vec, bias_ve3 = float((gvec - g_true).mean()), float((gve3 - g_true).mean())
        over_vec = float(((gvec - g_true) > 0.10).mean())
        over_ve3 = float(((gve3 - g_true) > 0.10).mean())
        # 2) novelty Gaussian on p3_train renders (seen families + clean train images)
        tr_imgs = [
            i
            for i, s in sub.items()
            if s == "p3_train" and i in clean.index and present[clean.index[i]]
        ]
        feats = np.concatenate(
            [
                occ["train"].arr["feat"].astype(np.float64),
                clean.arr["feat"][[clean.index[i] for i in tr_imgs]].astype(np.float64),
            ]
        )
        nov = MahalanobisNovelty.fit(feats)
        del feats
        # 3) tables with conservative co-visibility, novelty and TTA
        tabs: dict[str, dict[str, np.ndarray]] = {}
        for role in ("train", "val", "cal"):
            pairs, rows = ctx.c3.pairs[ev3.TAGS[role]], ctx.c3.plan(fold, role)
            tabs[role] = ev3.build_table(pairs, rows, clean, occ[role], fold, link, ctx.c3.ident)
            extend_table(tabs[role], pairs, rows, clean, occ[role], temp, shift, nov)
        for role, tag in (("val_clean", "p3_val_pairs"), ("cal_clean", "p3_cal_independent")):
            pairs = ctx.c3.pairs[tag]
            tabs[role] = ev3.build_table(pairs, None, clean, None, fold, link, ctx.c3.ident)
            extend_table(tabs[role], pairs, None, clean, None, temp, shift, nov)
        thr = nov.set_threshold(tabs["val"]["nov"], P4.novelty_id_fpr)  # type: ignore[union-attr]
        f: dict[str, Any] = {
            "temperature": temp,
            "shift": shift,
            "novelty": nov.to_dict(),
            "sanity_val_seen": {
                "bias_vec": bias_vec,
                "bias_ve3": bias_ve3,
                "share_over_0.10_vec": over_vec,
                "share_over_0.10_ve3": over_ve3,
            },
            "models": {},
            "val": {},
            "arms": {},
        }
        # 4) fusion
        models: dict[str, LogisticModel] = {}
        for m in LADDER_P4:
            xtr, names = design(tabs["train"], m, CV, LADDER_P4)
            xva, _ = design(tabs["val"], m, CV, LADDER_P4)
            best, scores = select_c(xtr, tabs["train"]["label"], xva, tabs["val"]["label"], names)
            models[m] = best
            p = best.predict_proba(xva)
            f["models"][m] = best.to_dict()
            f["val"][m] = {
                "c_scores": scores,
                "log_loss": float(log_loss(p, tabs["val"]["label"]).mean()),
                "auc": roc_auc(p, tabs["val"]["label"]),
            }
        xtr, _ = design(tabs["train"], "M5c", CV, LADDER_P4)
        xva, _ = design(tabs["val"], "M5c", CV, LADDER_P4)
        f["mlp"] = train_mlp(xtr, tabs["train"]["label"], xva, tabs["val"]["label"], P4.seed)  # type: ignore[union-attr]
        rel = (f["val"]["M5c"]["log_loss"] - f["mlp"]["val_log_loss"]) / f["val"]["M5c"]["log_loss"]
        f["mlp_selection"] = {
            "relative_val_log_loss_gain_vs_M5c": rel,
            "deployable_fused_score": "MLP" if rel > 0.05 else "M5c",
        }
        # 5) strata edges (val)
        f["edges"] = {
            "vc": [float(x) for x in np.quantile(tabs["val"][f"vc_{CV}"], [1 / 3, 2 / 3])],
            "q": list(f3["edges"]["q"]),
        }
        f["novelty"]["threshold"] = thr
        f["novelty_val"] = {
            "id_pair_fpr_val": float((tabs["val"]["nov"] > thr).mean()),
            "clean_pair_flag_rate_val": float((tabs["val_clean"]["nov"] > thr).mean()),
        }
        # 6) new arms
        scv, scc = p4_scores(tabs["val"], f), p4_scores(tabs["cal"], f)
        for arm in P4_ARMS:
            stv, nv, key = p4_strata(arm, tabs["val"], f)
            stc, nc, _ = p4_strata(arm, tabs["cal"], f)
            sv, scal = arm_score(key, scv, nv), arm_score(key, scc, nc)
            for tname in TARGETS:
                for ordk in ORDERINGS:
                    rec = calibrate(
                        sv,
                        stv,
                        tabs["val"]["label"],
                        scal,
                        stc,
                        tabs["cal"]["label"],
                        cal_set,
                        tname,
                        ordk,
                    )
                    rec |= {"score": key, "arm": arm}
                    f["arms"][f"{arm}|{tname}|{ordk}"] = rec
        # 7) clean-margin thresholds (clean cal, FAR 1 %) and EER thresholds (val)
        sclean = p4_scores(tabs["cal_clean"], f)
        ncl = tabs["cal_clean"]["nov"] > thr
        sclean["H4"] = hybrid_score(~ncl, sclean["M4c"], sclean["M1"])
        neg = tabs["cal_clean"]["label"] == 0
        f["h4_thresholds"] = {m: threshold_at_far(v[neg], 0.01) for m, v in sclean.items()}
        scv["H4"] = hybrid_score(~(tabs["val"]["nov"] > thr), scv["M4c"], scv["M1"])
        f["eer_threshold"] = {
            m: float(equal_error_rate(v, tabs["val"]["label"]).threshold) for m, v in scv.items()
        }
        frozen["folds"][fold] = f
        print(
            fold,
            {
                "T": temp,
                "c": shift,
                "nov_thr": round(thr, 2),
                "mlp_gain": round(rel, 4),
                "certified_v2_primary": {
                    a.split("|")[0]: r["n_strata_certified"]
                    for a, r in f["arms"].items()
                    if a.endswith("|primary|v2")
                },
                "sanity": f["sanity_val_seen"],
            },
            flush=True,
        )
        del clean, occ, tabs
    frozen["sha256"] = _sha(frozen)
    ctx.layout.private_reports.joinpath("phase4_frozen.json").write_text(
        json.dumps(frozen, default=str), encoding="utf-8"
    )
    pub = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 4 develop (p3_train/val/cal only; no test access)",
        "frozen_sha256": frozen["sha256"],
        "phase3_frozen_sha256": ctx.fz3["sha256"],
        "folds": {
            fo: {
                k: v[k]
                for k in (
                    "temperature",
                    "shift",
                    "sanity_val_seen",
                    "val",
                    "mlp_selection",
                    "edges",
                    "novelty_val",
                )
            }
            | {
                "novelty_threshold": v["novelty"]["threshold"],
                "arms_certified": {a: r["n_strata_certified"] for a, r in v["arms"].items()},
            }
            for fo, v in frozen["folds"].items()
        },
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase4_develop.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(pub, fh, indent=1, default=str)
    print("frozen", frozen["sha256"])
    return 0


# ----------------------------------------------------------------- evaluation core (shared)


def evaluate_cell(
    t: dict[str, np.ndarray], f3: dict[str, Any], f4: dict[str, Any]
) -> dict[str, Any]:
    """All arms on one table. Returns arm summaries and per-pair decisions for paired tests."""
    y = t["label"]
    s3 = {
        m: LogisticModel.from_dict(d).predict_proba(design(t, m, CV)[0])
        for m, d in f3["models"].items()
    }
    s4 = p4_scores(t, f4)
    out: dict[str, Any] = {"n_pairs": len(y), "arms": {}, "decisions": {}}
    for name, a3 in P3_ARMS.items():
        for tname in TARGETS:
            for ordk in ORDERINGS:
                rec = f3["arms"][f"{a3}|{tname}|{ordk}"]
                d = ev3.decide(
                    rec, s3[rec["model"]], ev3.strata_labels(rec["strata"], t, f3["edges"], CV)
                )
                out["arms"][f"{name}|{tname}|{ordk}"] = _summ(d, t, s3[rec["model"]], tname)
                out["decisions"][f"{name}|{tname}|{ordk}"] = d
    for arm in P4_ARMS:
        st, novel, key = p4_strata(arm, t, f4)
        sc = arm_score(key, s4, novel)
        for tname in TARGETS:
            for ordk in ORDERINGS:
                rec = f4["arms"][f"{arm}|{tname}|{ordk}"]
                d = ev3.decide(rec, sc, st)
                out["arms"][f"{arm}|{tname}|{ordk}"] = _summ(d, t, sc, tname)
                out["decisions"][f"{arm}|{tname}|{ordk}"] = d
    novel = t["nov"] > f4["novelty"]["threshold"]
    out["novel_pair_rate"] = float(novel.mean())
    occm = t["occluded"]
    s4["H4"] = hybrid_score(~novel, s4["M4c"], s4["M1"])
    allsc = {
        "M0": s3["M0"],
        "M1": s3["M1"],
        "M4_phase3": s3["M4"],
        "M4_oracle": s3["M4_oracle"],
    } | {k: v for k, v in s4.items() if k != "M1"}
    out["models"] = {}
    for m, p in allsc.items():
        rec = {"auc": roc_auc(p, y), "log_loss": float(log_loss(p, y).mean())}
        if occm.any() and len(np.unique(y[occm])) == 2:
            rec["auc_occluded"] = roc_auc(p[occm], y[occm])
            et = (
                f4["eer_threshold"].get(m)
                if m in f4["eer_threshold"]
                else f3["eer_threshold"].get({"M4_phase3": "M4"}.get(m, m))
            )
            if et is not None:
                fail = ((p >= et) != (y == 1)).astype(np.int64)
                rec["augrc_occluded"] = augrc(np.abs(p - et)[occm], fail[occm])
        out["models"][m] = rec
    # co-visibility ablations (paired identity jackknife of per-pair log-loss differences)
    out["ablations"] = {}
    for nm, (a, b) in {
        "M1_minus_M4c": ("M1", "M4c"),
        "M5_noCV_minus_M5c": ("M5_noCV", "M5c"),
        "M5c_minus_MLP": ("M5c", "MLP"),
    }.items():
        dl = log_loss(allsc[a], y) - log_loss(allsc[b], y)
        m = occm if occm.any() else np.ones(len(y), bool)
        j = jackknife_interval(
            dl[m], np.ones(int(m.sum())), t["id_a"][m], t["id_b"][m], scale="linear"
        )
        out["ablations"][nm] = {
            "estimate": j.estimate,
            "ci": [j.ci_low, j.ci_high],
            "p_value": normal_p_value(j.estimate, j.se),
        }
    return out


def _summ(d: np.ndarray, t: dict[str, np.ndarray], score: np.ndarray, tname: str) -> dict[str, Any]:
    s = ev3.selective_summary(d, ev3.as_sp(t, score), *TARGETS[tname])
    return {k: s[k] for k in SUMMARY_KEYS}


def paired_vs_a3(
    cell: dict[str, Any], t: dict[str, np.ndarray], key_suffix: str = "primary|v2"
) -> dict[str, Any]:
    y = t["label"]
    b = ev3._sel(cell["decisions"][f"A3_quality|{key_suffix}"], y)
    out = {}
    for x in (*CANDIDATES, "A4_original", "A5_oracle", "B1_quality_global"):
        a = ev3._sel(cell["decisions"][f"{x}|{key_suffix}"], y)
        comp = {}
        for nm, (na, da, nb, db) in {
            "sFNMR": (a["fnm"], a["acc_g"], b["fnm"], b["acc_g"]),
            "sFMR": (a["fm"], a["acc_i"], b["fm"], b["acc_i"]),
            "coverage": (a["acc"], np.ones(len(y)), b["acc"], np.ones(len(y))),
        }.items():
            if da.sum() > 0 and db.sum() > 0:
                comp[nm] = jackknife_ratio_difference(na, da, nb, db, t["id_a"], t["id_b"]).__dict__
        out[x] = comp
    return out


def clean_margin(
    tc: dict[str, np.ndarray], f3: dict[str, Any], f4: dict[str, Any]
) -> dict[str, Any]:
    y = tc["label"]
    g = y == 1
    s0 = LogisticModel.from_dict(f3["models"]["M0"]).predict_proba(design(tc, "M0", CV)[0])
    f0 = (s0 < f3["h4_thresholds"]["M0"]).astype(float)
    s4 = p4_scores(tc, f4)
    s4["H4"] = hybrid_score(~(tc["nov"] > f4["novelty"]["threshold"]), s4["M4c"], s4["M1"])
    out = {
        "fnmr_M0": float(f0[g].mean()),
        "fmr_M0": float((s0 >= f3["h4_thresholds"]["M0"])[~g].mean()),
    }
    for m in ("M1", "M4c", "M5c", "MLP", "H4"):
        fx = (s4[m] < f4["h4_thresholds"][m]).astype(float)
        dd = jackknife_ratio_difference(
            fx * g, g.astype(float), f0 * g, g.astype(float), tc["id_a"], tc["id_b"]
        )
        out[m] = {
            "fnmr": float(fx[g].mean()),
            "fmr": float((s4[m] >= f4["h4_thresholds"][m])[~g].mean()),
            "delta_fnmr_vs_M0": dd.__dict__,
            "non_inferior_at_margin": bool(dd.ci_high < P4.clean_margin_pp / 100),
        }  # type: ignore[union-attr]
    return out


def endpoints(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """P1 safety / P2 benefit over the 10 primary cells (Test-OOD x5, E1 x5)."""
    prim = [k for k in cells if k.endswith("|test_ood") or k.endswith("|e1")]
    res: dict[str, Any] = {"primary_cells": prim}

    def viol(key: str) -> int:
        n = 0
        for c in prim:
            a = cells[c]["arms"][key]
            n += int(a["verdict_fnm"].startswith("violated")) + int(
                a["verdict_fm"].startswith("violated")
            )
        return n

    res["violations_primary_v2"] = {
        x: viol(f"{x}|primary|v2")
        for x in (
            "A3_quality",
            "A4_original",
            "A5_oracle",
            "B0_raw_global",
            "B1_quality_global",
            *CANDIDATES,
        )
    }
    for x in CANDIDATES:
        worse = {}
        for err in ("sFNMR", "sFMR"):
            ps = [cells[c]["paired_vs_a3"][x].get(err, {}).get("p_value", 1.0) for c in prim]
            est = [cells[c]["paired_vs_a3"][x].get(err, {}).get("estimate", 0.0) for c in prim]
            adj = holm(ps)
            worse[err] = [c for c, e, p in zip(prim, est, adj, strict=True) if e > 0 and p < 0.05]
        safe = (
            res["violations_primary_v2"][x] <= res["violations_primary_v2"]["A3_quality"]
            and not worse["sFNMR"]
            and not worse["sFMR"]
        )
        cps = [cells[c]["paired_vs_a3"][x].get("coverage", {}) for c in prim]
        gains = [c.get("estimate", 0.0) for c in cps]
        adjc = holm([c.get("p_value", 1.0) for c in cps])
        sig_pos = sum(1 for g, p in zip(gains, adjc, strict=True) if g > 0 and p < 0.05)
        better = bool(safe and np.mean(gains) > 0 and sig_pos >= 6)
        res[x] = {
            "significantly_worse_cells": worse,
            "safe": bool(safe),
            "mean_coverage_gain": float(np.mean(gains)),
            "cells_significant_coverage_gain": sig_pos,
            "better_than_A3": better,
        }
    better = [x for x in CANDIDATES if res[x]["better_than_A3"]]
    if better:
        better.sort(key=lambda x: (res["violations_primary_v2"][x], -res[x]["mean_coverage_gain"]))
        res["recommendation"] = better[0]
    else:
        res["recommendation"] = "A3_quality"
    return res


# ----------------------------------------------------------------- final / rehearse


def _p4_group(directory: Path, name: str, cols: tuple[str, ...], fold: str | None) -> Any:
    g = ev3.Group(directory, name, cols)
    if fold is not None:  # per-fold columns -> generic names
        for base in ("ve", "vec", "vecf", "feat"):
            if f"{base}_{fold[:2]}" in g.arr:
                g.arr[base] = g.arr[f"{base}_{fold[:2]}"]
    g.arr.setdefault("z", np.zeros((len(g.keys), 28, 28), np.float16))
    return g


def final(args: argparse.Namespace) -> int:
    lay = resolve_data_root()
    fz4 = json.loads((lay.private_reports / "phase4_frozen.json").read_text("utf-8"))
    if _sha({k: v for k, v in fz4.items() if k != "sha256"}) != fz4["sha256"]:
        raise RuntimeError("phase4 frozen config modified")
    if args.final == args.rehearse:
        raise RuntimeError("choose exactly one of --final (p4_test, logged) or --rehearse (p3_val)")
    ctx = Ctx()
    if ctx.fz3["sha256"] != fz4["phase3_frozen_sha256"]:
        raise RuntimeError("phase3 freeze changed since phase4 develop")
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    if args.final:
        with (ROOT / PROTO.test_access.run_log).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "utc": datetime.now(tz=UTC).isoformat(),
                        "script": "scripts/evaluate_phase4.py final",
                        "splits": ["p4_test"],
                        "frozen_config_sha256": fz4["sha256"],
                        "git_commit": (prov.get("git") or {}).get("commit"),
                        "note": "Phase 4 confirmatory run",
                    }
                )
                + "\n"
            )
    cells: dict[str, dict[str, Any]] = {}
    clean_res: dict[str, Any] = {}
    nov_eval: dict[str, Any] = {}
    bias: dict[str, Any] = {}
    if args.rehearse:
        for fold in FAMILIES:
            f3, f4 = ctx.fz3["folds"][fold], fz4["folds"][fold]
            link = LogisticModel.from_dict(f3["heuristic_link"])
            clean, occ = ctx.clean(fold), ctx.occ(fold, "val")
            pairs, rows = ctx.c3.pairs["p3_val_pairs"], ctx.c3.plan(fold, "val")
            nov = MahalanobisNovelty.from_dict(f4["novelty"])
            t = ev3.build_table(pairs, rows, clean, occ, fold, link, ctx.c3.ident)
            extend_table(t, pairs, rows, clean, occ, f4["temperature"], f4["shift"], nov)
            for role in ("test_ood", "e1"):  # stand-ins: the val table plays both roles
                c = evaluate_cell(t, f3, f4)
                c["paired_vs_a3"] = paired_vs_a3(c, t)
                c.pop("decisions")
                cells[f"{fold}|{role}"] = c
            tc = ev3.build_table(pairs, None, clean, None, fold, link, ctx.c3.ident)
            extend_table(tc, pairs, None, clean, None, f4["temperature"], f4["shift"], nov)
            clean_res[fold] = clean_margin(tc, f3, f4)
            print("rehearse", fold, flush=True)
    else:
        p4dir = lay.root / "features" / "phase4" / "p4"
        man = lay.manifests / "phase4"
        pairs = list(read_jsonl(man / "p4_test_pairs.jsonl", PairRecord))
        ident = {r.image_id: r.identity_id for r in read_jsonl(man / "images.jsonl", ImageRecord)}
        base = ("emb_ada", "emb_arc", "norm_ada", "norm_arc")
        e1_rows = ev3._rows(man / "occlusion" / "E1_hair_strands__test_e1__p4_test_pairs.jsonl.gz")
        for fold in FAMILIES:
            f3, f4 = ctx.fz3["folds"][fold], fz4["folds"][fold]
            link = LogisticModel.from_dict(f3["heuristic_link"])
            nov = MahalanobisNovelty.from_dict(f4["novelty"])
            percol = tuple(f"{x}_{fold[:2]}" for x in ("ve", "vec", "vecf", "feat"))
            clean = _p4_group(p4dir, "p4_clean", (*base, *percol), fold)
            groups = {
                "test_id": _p4_group(
                    p4dir,
                    f"{fold}__test_id__p4_test_pairs",
                    (*base, "ve", "vec", "vecf", "feat", "gt", "vis"),
                    None,
                ),
                "test_ood": _p4_group(
                    p4dir,
                    f"{fold}__test_ood__p4_test_pairs",
                    (*base, "ve", "vec", "vecf", "feat", "gt", "vis"),
                    None,
                ),
                "e1": _p4_group(p4dir, "e1", (*base, *percol, "gt", "vis"), fold),
            }
            plans = {
                "test_id": ev3._rows(
                    man / "occlusion" / f"{fold}__test_id__p4_test_pairs.jsonl.gz"
                ),
                "test_ood": ev3._rows(
                    man / "occlusion" / f"{fold}__test_ood__p4_test_pairs.jsonl.gz"
                ),
                "e1": e1_rows,
            }
            nov_eval[fold] = {}
            for role, occ in groups.items():
                t = ev3.build_table(pairs, plans[role], clean, occ, fold, link, ident)
                extend_table(t, pairs, plans[role], clean, occ, f4["temperature"], f4["shift"], nov)
                c = evaluate_cell(t, f3, f4)
                c["paired_vs_a3"] = paired_vs_a3(c, t)
                c.pop("decisions")
                cells[f"{fold}|{role}"] = c
                # visibility bias of Phase 3 VE and VE-C on this group's occluded renders (image level)
                gtg = global_visible(occ.arr["gt"].astype(np.float64) / 255)
                b3 = global_visible(occ.arr["ve"].astype(np.float64) / 255) - gtg
                bc = (
                    global_visible(conservative_map(occ.arr["vec"], f4["temperature"], f4["shift"]))
                    - gtg
                )
                fams = np.array([str(k).split("|")[1] for k in occ.keys.tolist()])
                bias[f"{fold}|{role}"] = {
                    fam: {
                        "n": int((fams == fam).sum()),
                        "bias_ve_phase3": float(b3[fams == fam].mean()),
                        "bias_vec": float(bc[fams == fam].mean()),
                        "share_over_0.10_ve_phase3": float((b3[fams == fam] > 0.10).mean()),
                        "share_over_0.10_vec": float((bc[fams == fam] > 0.10).mean()),
                    }
                    for fam in sorted(set(fams.tolist()))
                }
                # novelty at image level: occluded sides of this role vs ID references
                side_scores = nov.score(occ.arr["feat"].astype(np.float64))
                nov_eval[fold][role] = {
                    "image_flag_rate_at_pair_threshold": float(
                        (side_scores > f4["novelty"]["threshold"]).mean()
                    ),
                    "pair_flag_rate": c["novel_pair_rate"],
                    "scores": side_scores,
                }
                print(fold, role, "done", flush=True)
            # novelty AUROC: ID (test_id occluded renders, seen families) vs held-out / E1 renders
            id_s = nov_eval[fold]["test_id"]["scores"]
            clean_s = nov.score(clean.arr["feat"].astype(np.float64))
            for role in ("test_ood", "e1"):
                o = nov_eval[fold][role]["scores"]
                lab = np.r_[np.ones(len(o)), np.zeros(len(id_s))]
                nov_eval[fold][role]["auroc_vs_seen_family_renders"] = roc_auc(np.r_[o, id_s], lab)
                lab2 = np.r_[np.ones(len(o)), np.zeros(len(clean_s))]
                nov_eval[fold][role]["auroc_vs_clean_images"] = roc_auc(np.r_[o, clean_s], lab2)
            for role in nov_eval[fold]:
                nov_eval[fold][role].pop("scores", None)
            tc = ev3.build_table(pairs, None, clean, None, fold, link, ident)
            extend_table(tc, pairs, None, clean, None, f4["temperature"], f4["shift"], nov)
            clean_res[fold] = clean_margin(tc, f3, f4)
            clean_res[fold]["clean_pair_novelty_flag_rate"] = float(
                (tc["nov"] > f4["novelty"]["threshold"]).mean()
            )
            del clean, groups
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 4 CONFIRMATORY run on p4_test (DigiFace identities never downloaded before Phase 4; synthetic occlusions)",
        "run_id": run_id,
        "frozen_sha256": fz4["sha256"],
        "phase3_frozen_sha256": ctx.fz3["sha256"],
        "cells": cells,
        "clean_margin": clean_res,
        "novelty": nov_eval,
        "visibility_bias": bias,
        "endpoints": endpoints(cells),
        "provenance": prov,
    }
    if args.rehearse:
        report["scope"] = "REHEARSAL on p3_val (code-path check only; not a result)"
        out = lay.private_reports / "phase4_final_rehearsal.json"
    else:
        out = ROOT / "reports" / "experiments" / "phase4_final.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("endpoints", json.dumps(report["endpoints"], default=str)[:2000])
    return 0


def followup(args: argparse.Namespace) -> int:
    """FOLLOW-UP on p3_test (Phase 3 confirmatory set, observed once): frozen Phase 4 rules,
    Test-OOD and Test-ID cells (no E1 renders exist for p3_test). Logged; descriptive only."""
    if not args.final:
        raise RuntimeError("p3_test follow-up access requires --final")
    lay = resolve_data_root()
    fz4 = json.loads((lay.private_reports / "phase4_frozen.json").read_text("utf-8"))
    if _sha({k: v for k, v in fz4.items() if k != "sha256"}) != fz4["sha256"]:
        raise RuntimeError("phase4 frozen config modified")
    ctx = Ctx()
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    with (ROOT / PROTO.test_access.run_log).open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "utc": datetime.now(tz=UTC).isoformat(),
                    "script": "scripts/evaluate_phase4.py followup",
                    "splits": ["p3_test"],
                    "frozen_config_sha256": fz4["sha256"],
                    "git_commit": (prov.get("git") or {}).get("commit"),
                    "note": "Phase 4 FOLLOW-UP on p3_test (observed in Phase 3)",
                }
            )
            + "\n"
        )
    cells: dict[str, dict[str, Any]] = {}
    clean_res: dict[str, Any] = {}
    pairs = ctx.c3.pairs["p3_test_pairs"]
    for fold in FAMILIES:
        f3, f4 = ctx.fz3["folds"][fold], fz4["folds"][fold]
        link = LogisticModel.from_dict(f3["heuristic_link"])
        nov = MahalanobisNovelty.from_dict(f4["novelty"])
        clean = ctx.c3.clean(fold)
        dclean = ev3.Group(
            ctx.dev, "p3test_clean", tuple(f"{x}_{fold[:2]}" for x in ("vec", "vecf", "feat"))
        )
        attach(
            clean,
            dclean,
            {"vec": f"vec_{fold[:2]}", "vecf": f"vecf_{fold[:2]}", "feat": f"feat_{fold[:2]}"},
            allow_missing=True,
        )
        for role in ("test_id", "test_ood"):
            occ = ctx.occ(fold, role)
            rows = ctx.c3.plan(fold, role)
            t = ev3.build_table(pairs, rows, clean, occ, fold, link, ctx.c3.ident)
            extend_table(t, pairs, rows, clean, occ, f4["temperature"], f4["shift"], nov)
            c = evaluate_cell(t, f3, f4)
            c["paired_vs_a3"] = paired_vs_a3(c, t)
            c.pop("decisions")
            cells[f"{fold}|{role}"] = c
        tc = ev3.build_table(pairs, None, clean, None, fold, link, ctx.c3.ident)
        extend_table(tc, pairs, None, clean, None, f4["temperature"], f4["shift"], nov)
        clean_res[fold] = clean_margin(tc, f3, f4)
        print("followup", fold, flush=True)
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 4 FOLLOW-UP on p3_test (observed once in Phase 3; frozen Phase 4 rules; descriptive)",
        "run_id": run_id,
        "frozen_sha256": fz4["sha256"],
        "cells": cells,
        "clean_margin": clean_res,
        "endpoints_descriptive_test_ood_only": endpoints(cells),
        "provenance": prov,
    }
    with (ROOT / "reports" / "experiments" / "phase4_followup_p3test.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(report, fh, indent=1, default=str)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["develop", "final", "followup"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--rehearse", action="store_true")
    args = ap.parse_args()
    if args.stage == "develop":
        return develop()
    return final(args) if args.stage == "final" else followup(args)


if __name__ == "__main__":
    sys.exit(main())
