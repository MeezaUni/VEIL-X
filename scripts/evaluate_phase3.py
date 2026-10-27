"""Phase 3 develop / final evaluation (docs/phase3_protocol.md §§4-7). Run in .venv-gpu.

develop  (p3_train / p3_val / p3_cal only; no p3_test access)
    heuristic links per fold (p3_val, seen families), co-visibility definition selection
    (p3_val, registered rule), fusion ladder per fold (train fit, C on val), MLP diagnostic,
    strata edges (val), LTT candidate orders v1/v2 (val), all calibration arms (p3_cal),
    frozen with SHA-256.
final --final  (p3_test, once, logged; --rehearse runs the same path on p3_val)
    C3 primary (A4 vs A3 on Test-OOD, Holm over folds, verdict rule), C1 primary (M1 vs M4
    log-loss, Holm over 10 cells), H4 clean margin, secondary arms/orderings/targets, AUGRC,
    risk-coverage, per family / severity, abstention-reason mix.
followup --final  (frozen config; Phase 2 DigiFace test held-out-family plans + LFW View 2
    derived occlusions, logged; --rehearse runs LFW View 1 DevTest only)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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
from veilx.embeddings.store import read_array_shard
from veilx.eval.scoring import ScoredPairs
from veilx.eval.summaries import selective_summary
from veilx.fusion.features import LADDER, SOURCES, covis_columns, design
from veilx.fusion.logreg import LogisticModel, log_loss, select_c
from veilx.metrics.roc import equal_error_rate, roc_auc, threshold_at_far
from veilx.metrics.selective import ABSTAIN, MATCH, NON_MATCH, augrc, risk_coverage_curve
from veilx.pairs.builder import PairSet, summarize
from veilx.provenance import collect_provenance
from veilx.selective.ltt import (
    LTTCalibrator,
    ThresholdRule,
    fit_stratified,
    order_candidates_by_validation_pvalue,
    order_candidates_on_validation,
)
from veilx.stats.jackknife import jackknife_interval
from veilx.stats.paired import holm, jackknife_ratio_difference, normal_p_value
from veilx.vis.covis import DEFINITIONS
from veilx.vis.heuristic import fit_link_cells

ROOT = Path(__file__).resolve().parents[1]
PROTO = load_protocol(ROOT / "configs" / "research_protocol.yaml")
P3 = PROTO.phase3
assert P3 is not None
FAMILIES = tuple(PROTO.family_ids)
TARGETS = {"primary": (0.05, 0.05), "far1": (0.01, 0.05)}
DELTA = 0.05
ORDERINGS = ("v2", "v1")
N_GRID = 61
# §4: errors of the frozen AdaFace cosine at its Phase 2 cal_far_1pct threshold (read, not retyped)
ERR_THRESHOLD_ADAFACE = json.loads(
    (ROOT / "reports" / "experiments" / "phase2_final.json").read_text("utf-8")
)["results"]["adaface_ir101_webface4m"]["thresholds"]["cal_far_1pct"]
TAGS = {
    "train": "p3_train_pairs",
    "val": "p3_val_pairs",
    "cal": "p3_cal_independent",
    "test_id": "p3_test_pairs",
    "test_ood": "p3_test_pairs",
}
ARMS: dict[str, tuple[str, str]] = {  # arm -> (score model, strata kind)
    "A1_clean": ("M0", "none"),
    "A2_global_M0": ("M0", "none"),
    "A2_global_M4": ("M4", "none"),
    "A2_global_M0arc": ("M0_arc", "none"),
    "A3_quality": ("M1", "quality"),
    "A4_predvis": ("M4", "predvis"),
    "A5_oracle": ("M4_oracle", "oracle"),
    "A6_quality_x_predvis": ("M4", "q_x_v"),
}


# ----------------------------------------------------------------- feature store access


class Group:
    def __init__(self, directory: Path, name: str, cols: tuple[str, ...]) -> None:
        parts = sorted(directory.glob(f"{name}.part*.npz"))
        if not parts:
            raise RuntimeError(f"no shards for {name}")
        keys, arrs = [], defaultdict(list)
        for k, p in enumerate(parts):
            if p.name != f"{name}.part{k:04d}.npz":
                raise RuntimeError(f"missing part {k} of {name}")
            kk, aa, _ = read_array_shard(p)  # SHA-256 verified
            keys.append(kk)
            for c in cols:
                arrs[c].append(aa[c])
        self.keys = np.concatenate(keys)
        self.arr = {c: np.concatenate(v) for c, v in arrs.items()}
        self.index = {str(k): i for i, k in enumerate(self.keys.tolist())}


def _emb(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def build_table(
    pairs: list[PairRecord],
    rows: dict[str, dict[str, Any]] | None,
    clean: Group,
    occ: Group | None,
    fold: str,
    link: LogisticModel,
    ident: dict[str, str],
) -> dict[str, np.ndarray]:
    """Pair table for clean pairs (rows=None) or pairs under a LOFO plan."""
    n = len(pairs)
    side_store = np.zeros((2, n), np.int8)  # 0 clean, 1 occluded
    side_row = np.zeros((2, n), np.int64)
    vis_engine = np.ones((2, n))
    fam = np.full((2, n), "", dtype=object)
    lvl = np.full((2, n), "", dtype=object)
    stratum = np.full(n, "clean_clean", dtype=object)
    for k, p in enumerate(pairs):
        row = rows[p.pair_id] if rows is not None else None
        if row is not None:
            stratum[k] = row["stratum"]
        for j, (side, img) in enumerate((("a", p.image_a), ("b", p.image_b))):
            spec = row[side] if row is not None else None
            if spec is None:
                side_row[j, k] = clean.index[img]
            else:
                assert occ is not None
                side_store[j, k] = 1
                side_row[j, k] = occ.index[
                    occluded_key(img, spec["family"], spec["severity_level"], int(spec["seed"]))
                ]
                vis_engine[j, k] = spec["visibility_ratio"]
                fam[j, k], lvl[j, k] = spec["family"], spec["severity_level"]

    def gather(col_clean: str, col_occ: str, j: int, idx: slice) -> np.ndarray:
        st, rw = side_store[j, idx], side_row[j, idx]
        src = clean.arr[col_clean][rw]
        if occ is not None and (st == 1).any():
            src = src.copy()
            src[st == 1] = occ.arr[col_occ][rw[st == 1]]
        return src

    out: dict[str, list[np.ndarray]] = defaultdict(list)
    for i in range(0, n, 8192):
        sl = slice(i, min(n, i + 8192))
        ea, eb = (
            _emb(gather("emb_ada", "emb_ada", 0, sl)),
            _emb(gather("emb_ada", "emb_ada", 1, sl)),
        )
        aa, ab = (
            _emb(gather("emb_arc", "emb_arc", 0, sl)),
            _emb(gather("emb_arc", "emb_arc", 1, sl)),
        )
        out["s"].append(np.einsum("ij,ij->i", ea, eb).astype(np.float64))
        out["s_arc"].append(np.einsum("ij,ij->i", aa, ab).astype(np.float64))
        out["q"].append(
            np.minimum(
                gather("norm_ada", "norm_ada", 0, sl), gather("norm_ada", "norm_ada", 1, sl)
            ).astype(np.float64)
        )
        out["q_arc"].append(
            np.minimum(
                gather("norm_arc", "norm_arc", 0, sl), gather("norm_arc", "norm_arc", 1, sl)
            ).astype(np.float64)
        )
        ve_col = f"ve_{fold[:2]}"
        maps = {"ve": [gather(ve_col, "ve", j, sl).astype(np.float64) / 255 for j in (0, 1)]}
        z = [gather("z", "z", j, sl).astype(np.float64) for j in (0, 1)]
        maps["heur"] = [link.predict_proba(zz.reshape(-1, 1)).reshape(zz.shape) for zz in z]
        gts = []
        for j in (0, 1):
            g = np.ones((sl.stop - sl.start, 28, 28))
            st = side_store[j, sl] == 1
            if occ is not None and st.any():
                g[st] = occ.arr["gt"][side_row[j, sl][st]].astype(np.float64) / 255
            gts.append(g)
        maps["gt"] = gts
        for src in SOURCES:
            for kcol, v in covis_columns(maps[src][0], maps[src][1], src).items():
                out[kcol].append(v)
    t = {k: np.concatenate(v) for k, v in out.items()}
    t["label"] = np.array([1 if p.label == "genuine" else 0 for p in pairs], np.int64)
    t["id_a"] = np.array([ident[p.image_a] for p in pairs])
    t["id_b"] = np.array([ident[p.image_b] for p in pairs])
    t["pair_id"] = np.array([p.pair_id for p in pairs])
    t["gt_engine_covis"] = np.minimum(vis_engine[0], vis_engine[1])
    t["occluded"] = side_store.sum(0) > 0
    t["stratum"] = stratum.astype(str)
    t["fam"] = np.array([a or b for a, b in zip(fam[0], fam[1], strict=True)])
    rank = {"": 0, "low": 1, "medium": 2, "high": 3}
    t["max_level"] = np.array([max(rank[a], rank[b]) for a, b in zip(lvl[0], lvl[1], strict=True)])
    return t


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {(r := json.loads(line))["pair_id"]: r for line in fh}


# ----------------------------------------------------------------- calibration helpers


def _grid(scores: np.ndarray) -> list[ThresholdRule]:
    taus = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, N_GRID)))
    return [ThresholdRule(float(lo), float(hi)) for k, lo in enumerate(taus) for hi in taus[k:]]


def _order(
    kind: str,
    rules: list[ThresholdRule],
    s: np.ndarray,
    y: np.ndarray,
    n_cal: int,
    tfm: float,
    tfnm: float,
    delta: float,
) -> list[ThresholdRule]:
    if len(s) == 0 or np.unique(y).size < 2:
        return []
    if kind == "v1":
        return order_candidates_on_validation(
            rules, s, y, n_cal_expected=max(n_cal, 1), target=min(tfm, tfnm), delta=delta
        )
    return order_candidates_by_validation_pvalue(
        rules, s, y, n_cal_expected=max(n_cal, 1), target_fm=tfm, target_fnm=tfnm, delta=delta
    )


def strata_labels(
    kind: str, t: dict[str, np.ndarray], edges: dict[str, Any], cv_def: str
) -> np.ndarray:
    n = len(t["s"])
    if kind == "none":
        return np.full(n, "all")
    if kind == "quality":
        lo, hi = edges["q"]
        return np.where(t["q"] < lo, "q_low", np.where(t["q"] < hi, "q_mid", "q_high"))
    if kind == "predvis":
        lo, hi = edges["cv"]
        v = t[f"ve_{cv_def}"]
        return np.where(v < lo, "v_low", np.where(v < hi, "v_mid", "v_high"))
    if kind == "oracle":
        g = t["gt_engine_covis"]
        return np.where(
            g >= 0.999999, "clean", np.where(g >= edges["oracle"], "vis_high", "vis_low")
        )
    if kind == "q_x_v":
        qm, vm = edges["q_med"], edges["cv_med"]
        qs = np.where(t["q"] < qm, "qL", "qH")
        vs = np.where(t[f"ve_{cv_def}"] < vm, "vL", "vH")
        return np.char.add(np.char.add(qs.astype(str), "_"), vs.astype(str))
    raise ValueError(kind)


def decide(arm_rec: dict[str, Any], score: np.ndarray, strata: np.ndarray) -> np.ndarray:
    d = np.full(score.shape, ABSTAIN, dtype=np.int64)
    for name, rule in arm_rec["rules"].items():
        if rule is None:
            continue
        m = strata == name
        d[m] = ThresholdRule(rule[0], rule[1]).decide(score[m])
    return d


def as_sp(
    t: dict[str, np.ndarray], score: np.ndarray, mask: np.ndarray | None = None
) -> ScoredPairs:
    n = len(score)
    one, e = np.ones(n), np.asarray([""] * n)
    sp = ScoredPairs(
        pair_id=t["pair_id"],
        score=score,
        label=t["label"],
        id_a=t["id_a"],
        id_b=t["id_b"],
        vis_a=t["gt_engine_covis"],
        vis_b=one,
        fam_a=t["fam"],
        fam_b=e,
        lvl_a=e,
        lvl_b=e,
        stratum=t["stratum"],
        norm_a=t["q"],
        norm_b=one,
    )
    return sp if mask is None else sp.subset(mask)


def _sha(o: Any) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------- shared loading


class Ctx:
    def __init__(self) -> None:
        self.layout = resolve_data_root()
        self.p3 = self.layout.manifests / "phase3"
        self.feat = self.layout.root / "features" / "phase3"
        self.sub = json.loads((self.p3 / "subsplits.json").read_text("utf-8"))
        self.images = [
            r
            for r in read_jsonl(self.layout.manifests / "digiface_final.jsonl", ImageRecord)
            if r.image_id in self.sub
        ]
        self.ident = {r.image_id: r.identity_id for r in self.images}
        self.pairs = {
            tag: list(read_jsonl(self.p3 / "pairs" / f"{tag}.jsonl", PairRecord))
            for tag in set(TAGS.values())
        }
        self.clean_cols = ("emb_ada", "emb_arc", "norm_ada", "norm_arc", "z")

    def clean(self, fold: str) -> Group:
        return Group(self.feat, "p3_clean", (*self.clean_cols, f"ve_{fold[:2]}"))

    def occ(self, fold: str, role: str) -> Group:
        return Group(
            self.feat,
            f"{fold}__{role}__{TAGS[role]}",
            ("emb_ada", "emb_arc", "norm_ada", "norm_arc", "z", "ve", "gt", "vis"),
        )

    def plan(self, fold: str, role: str) -> dict[str, dict[str, Any]]:
        return _rows(self.p3 / "occlusion" / f"{fold}__{role}__{TAGS[role]}.jsonl.gz")

    def cal_set(self) -> PairSet:
        pairs = self.pairs["p3_cal_independent"]
        imgs = [r for r in self.images if self.sub[r.image_id] == "p3_cal"]
        return PairSet(
            tuple(pairs),
            summarize(pairs, imgs, "independent"),
            {"spec": {"class_assignment": "fixed"}},
        )


def fit_link(occ_val: Group, seed: int) -> LogisticModel:
    return fit_link_cells(occ_val.arr["z"], occ_val.arr["gt"], seed)


# ----------------------------------------------------------------- develop


def develop() -> int:
    import torch

    ctx = Ctx()
    cal_set = ctx.cal_set()
    if not cal_set.summary.identity_disjoint:
        raise RuntimeError("p3 calibration set is not identity-disjoint")
    frozen: dict[str, Any] = {
        "protocol_version": PROTO.protocol_version,
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "targets": TARGETS,
        "delta": DELTA,
        "folds": {},
    }
    tables: dict[str, dict[str, Any]] = {}
    # pass 1: links + val/train/cal tables per fold (cv-definition selection needs all val tables)
    for fold in FAMILIES:
        clean = ctx.clean(fold)
        occ_val = ctx.occ(fold, "val")
        link = fit_link(occ_val, P3.seed)  # type: ignore[union-attr]
        tabs = {}
        for role in ("train", "val", "cal"):
            occ = occ_val if role == "val" else ctx.occ(fold, role)
            tabs[role] = build_table(
                ctx.pairs[TAGS[role]], ctx.plan(fold, role), clean, occ, fold, link, ctx.ident
            )
            del occ
        for role, tag in (("val_clean", "p3_val_pairs"), ("cal_clean", "p3_cal_independent")):
            tabs[role] = build_table(ctx.pairs[tag], None, clean, None, fold, link, ctx.ident)
        tables[fold] = {"tabs": tabs, "link": link}
        print(f"{fold}: tables built", flush=True)
        del clean, occ_val
    # registered co-visibility selection on p3_val occluded pairs (pooled folds), learned maps
    sel: dict[str, float] = {}
    for d in DEFINITIONS:
        v = np.concatenate(
            [
                tables[f]["tabs"]["val"][f"ve_{d}"][tables[f]["tabs"]["val"]["occluded"]]
                for f in FAMILIES
            ]
        )
        t = {f: tables[f]["tabs"]["val"] for f in FAMILIES}
        s = np.concatenate([t[f]["s"][t[f]["occluded"]] for f in FAMILIES])
        y = np.concatenate([t[f]["label"][t[f]["occluded"]] for f in FAMILIES])
        err = (
            ((s >= ERR_THRESHOLD_ADAFACE) & (y == 0)) | ((s < ERR_THRESHOLD_ADAFACE) & (y == 1))
        ).astype(np.int64)
        sel[d] = roc_auc(-v, err)
    cv_def = max(sel, key=lambda k: sel[k])
    frozen["cv_selection"] = {"val_auroc_error_by_low_cv": sel, "selected": cv_def}
    print("cv selection", sel, "->", cv_def, flush=True)
    for fold in FAMILIES:
        tabs, link = tables[fold]["tabs"], tables[fold]["link"]
        f: dict[str, Any] = {"heuristic_link": link.to_dict(), "models": {}, "val": {}, "arms": {}}
        models: dict[str, LogisticModel] = {}
        for m in LADDER:
            xtr, names = design(tabs["train"], m, cv_def)
            xva, _ = design(tabs["val"], m, cv_def)
            best, scores = select_c(xtr, tabs["train"]["label"], xva, tabs["val"]["label"], names)
            models[m] = best
            p = best.predict_proba(xva)
            f["models"][m] = best.to_dict()
            f["val"][m] = {
                "c_scores": scores,
                "log_loss": float(log_loss(p, tabs["val"]["label"]).mean()),
                "auc": roc_auc(p, tabs["val"]["label"]),
            }
        # MLP diagnostic (protocol §5): 2-layer MLP on M5 features vs logistic, val log-loss
        xtr, _ = design(tabs["train"], "M5", cv_def)
        xva, _ = design(tabs["val"], "M5", cv_def)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        torch.manual_seed(P3.seed)  # type: ignore[union-attr]
        net = torch.nn.Sequential(
            torch.nn.Linear(xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(tabs["train"]["label"], dtype=torch.float32)
        xv = torch.tensor((xva - mu) / sd, dtype=torch.float32)
        best_v, bad = 1e9, 0
        for _ep in range(200):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 1024):
                b = perm[i : i + 1024]
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b])[:, 0], yt[b])
                loss.backward()
                opt.step()
            with torch.no_grad():
                pv = torch.sigmoid(net(xv)[:, 0]).numpy().astype(np.float64)
            v = float(log_loss(pv, tabs["val"]["label"]).mean())
            if v < best_v - 1e-5:
                best_v, bad = v, 0
            else:
                bad += 1
                if bad >= 5:
                    break
        lr_v = f["val"]["M5"]["log_loss"]
        f["mlp_diagnostic"] = {
            "val_log_loss_mlp": best_v,
            "val_log_loss_logistic_M5": lr_v,
            "relative_improvement": (lr_v - best_v) / lr_v,
            "use_mlp": bool((lr_v - best_v) / lr_v > 0.05),
        }
        # strata edges from val
        tv = tabs["val"]
        edges = {
            "q": [float(x) for x in np.quantile(tv["q"], [1 / 3, 2 / 3])],
            "cv": [float(x) for x in np.quantile(tv[f"ve_{cv_def}"], [1 / 3, 2 / 3])],
            "q_med": float(np.median(tv["q"])),
            "cv_med": float(np.median(tv[f"ve_{cv_def}"])),
            "oracle": float(np.median(tv["gt_engine_covis"][tv["gt_engine_covis"] < 0.999999])),
            "disagree_p95": float(np.quantile(np.abs(tv["s"] - tv["s_arc"]), 0.95)),
        }
        f["edges"] = edges
        # calibration arms
        for arm, (mname, skind) in ARMS.items():
            mdl = models[mname]
            vt, ct = (
                (tabs["val_clean"], tabs["cal_clean"])
                if arm == "A1_clean"
                else (tabs["val"], tabs["cal"])
            )
            sv = mdl.predict_proba(design(vt, mname, cv_def)[0])
            sc = mdl.predict_proba(design(ct, mname, cv_def)[0])
            stv, stc = (
                strata_labels(skind, vt, edges, cv_def),
                strata_labels(skind, ct, edges, cv_def),
            )
            rules = _grid(sv)
            names = sorted(set(stv.tolist()) | set(stc.tolist()))
            for tname, (tfm, tfnm) in TARGETS.items():
                for ordk in ORDERINGS:
                    rec: dict[str, Any] = {"model": mname, "strata": skind, "rules": {}, "cal": {}}
                    if skind == "none":
                        o = _order(ordk, rules, sv, vt["label"], len(sc), tfm, tfnm, DELTA)
                        if o:
                            r = LTTCalibrator(o, target_fm=tfm, target_fnm=tfnm, delta=DELTA).fit(
                                sc, ct["label"], pair_set=cal_set
                            )
                            rec["rules"]["all"] = (
                                None if r.rule is None else [r.rule.tau_low, r.rule.tau_high]
                            )
                        else:
                            rec["rules"]["all"] = None
                        rec["n_ordered"] = {"all": len(o)}
                    else:
                        ds = DELTA / len(names)
                        by = {
                            nm: _order(
                                ordk,
                                rules,
                                sv[stv == nm],
                                vt["label"][stv == nm],
                                int((stc == nm).sum()),
                                tfm,
                                tfnm,
                                ds,
                            )
                            for nm in names
                        }
                        res = fit_stratified(
                            LTTCalibrator,
                            rules,
                            sc,
                            ct["label"],
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
                                None
                                if rr is None or rr.rule is None
                                else [rr.rule.tau_low, rr.rule.tau_high]
                            )
                        rec["n_ordered"] = {nm: len(by[nm]) for nm in names}
                    rec["n_strata_certified"] = sum(v is not None for v in rec["rules"].values())
                    f["arms"][f"{arm}|{tname}|{ordk}"] = rec
        # thresholds for H4 (clean cal, FAR 1 %) and EER thresholds for AUGRC (val)
        f["h4_thresholds"] = {}
        f["eer_threshold"] = {}
        for m in ("M0", "M4", "M1", "M4_oracle", "M4_heur", "M5", "M0_arc"):
            pc = models[m].predict_proba(design(tabs["cal_clean"], m, cv_def)[0])
            f["h4_thresholds"][m] = threshold_at_far(pc[tabs["cal_clean"]["label"] == 0], 0.01)
            pv = models[m].predict_proba(design(tabs["val"], m, cv_def)[0])
            f["eer_threshold"][m] = float(equal_error_rate(pv, tabs["val"]["label"]).threshold)
        frozen["folds"][fold] = f
        print(
            fold,
            "arms certified (v2 primary):",
            {
                a.split("|")[0]: r["n_strata_certified"]
                for a, r in f["arms"].items()
                if a.endswith("|primary|v2")
            },
            flush=True,
        )
    frozen["sha256"] = _sha(frozen)
    ctx.layout.private_reports.joinpath("phase3_frozen.json").write_text(
        json.dumps(frozen, indent=1, default=str), encoding="utf-8"
    )
    pub = {
        "evidence_type": "real_measured_model_result",
        "scope": "Phase 3 develop (p3_train/val/cal only; no p3_test access)",
        "frozen_sha256": frozen["sha256"],
        "cv_selection": frozen["cv_selection"],
        "folds": {
            fo: {
                "val": v["val"],
                "mlp_diagnostic": v["mlp_diagnostic"],
                "edges": v["edges"],
                "arms_certified": {a: r["n_strata_certified"] for a, r in v["arms"].items()},
            }
            for fo, v in frozen["folds"].items()
        },
        "provenance": collect_provenance(ROOT, argv=sys.argv),
    }
    with (ROOT / "reports" / "experiments" / "phase3_develop.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as fh:
        json.dump(pub, fh, indent=1, default=str)
    print("frozen", frozen["sha256"])
    return 0


# ----------------------------------------------------------------- final


def _sel(d: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    acc = d != ABSTAIN
    return {
        "fnm": ((d == NON_MATCH) & (y == 1)).astype(float),
        "acc_g": (acc & (y == 1)).astype(float),
        "fm": ((d == MATCH) & (y == 0)).astype(float),
        "acc_i": (acc & (y == 0)).astype(float),
        "acc": acc.astype(float),
    }


def evaluate_occluded(
    t: dict[str, np.ndarray], sc: dict[str, np.ndarray], ff: dict[str, Any], cv_def: str
) -> dict[str, Any]:
    """Models, all arms, abstention reasons, C3 (A4 - A3) and C1 (M1 - M4) on one table."""
    y = t["label"]
    block: dict[str, Any] = {"n_pairs": len(y), "models": {}, "arms": {}}
    occm = t["occluded"]
    for m, p in sc.items():
        ll = log_loss(p, y)
        eer_t = ff["eer_threshold"].get(m)
        mrec = {
            "auc": roc_auc(p, y),
            "log_loss": float(ll.mean()),
            "auc_occluded": roc_auc(p[occm], y[occm]),
        }
        if eer_t is not None:
            fail = ((p >= eer_t) != (y == 1)).astype(np.int64)
            conf = np.abs(p - eer_t)
            mrec["augrc_occluded"] = augrc(conf[occm], fail[occm])
            rc = risk_coverage_curve(conf[occm], fail[occm])
            idx = np.linspace(0, len(rc.coverage) - 1, 41).astype(int)
            mrec["risk_coverage"] = {
                "coverage": rc.coverage[idx].tolist(),
                "selective_risk": [
                    None if not np.isfinite(v) else float(v) for v in rc.selective_risk[idx]
                ],
            }
        block["models"][m] = mrec
    for key, rec in ff["arms"].items():
        _arm, tname, _ordering = key.split("|")
        st = strata_labels(rec["strata"], t, ff["edges"], cv_def)
        d = decide(rec, sc[rec["model"]], st)
        summ = selective_summary(d, as_sp(t, sc[rec["model"]]), *TARGETS[tname])
        block["arms"][key] = {
            k: summ[k]
            for k in (
                "coverage",
                "coverage_genuine",
                "coverage_impostor",
                "selective_fmr",
                "selective_fnmr",
                "verdict_fm",
                "verdict_fnm",
            )
        }
        if key in ("A4_predvis|primary|v2", "A3_quality|primary|v2"):
            block["arms"][key]["by_level"] = {
                str(lv): {
                    k: v
                    for k, v in selective_summary(
                        d[t["max_level"] == lv],
                        as_sp(t, sc[rec["model"]], t["max_level"] == lv),
                        *TARGETS[tname],
                    ).items()
                    if k in ("coverage", "selective_fnmr", "selective_fmr")
                }
                for lv in range(4)
                if (t["max_level"] == lv).any()
            }
            if key.startswith("A4"):
                ab = d == ABSTAIN
                uncert = np.array([rec["rules"].get(s) is None for s in st])
                lo_v, _ = ff["edges"]["cv"]
                reasons = {
                    "uncertifiable_stratum": int((ab & uncert).sum()),
                    "insufficient_covisibility_vlow": int((ab & (t[f"ve_{cv_def}"] < lo_v)).sum()),
                    "poor_quality_qlow": int((ab & (t["q"] < ff["edges"]["q"][0])).sum()),
                    "conflicting_models": int(
                        (ab & (np.abs(t["s"] - t["s_arc"]) > ff["edges"]["disagree_p95"])).sum()
                    ),
                    "uncertain_score_in_band": int((ab & ~uncert).sum()),
                    "n_abstained": int(ab.sum()),
                }
                block["arms"][key]["abstention_reasons_nonexclusive"] = reasons
    # C3 paired comparison A4 vs A3 (primary, v2)
    r4, r3 = ff["arms"]["A4_predvis|primary|v2"], ff["arms"]["A3_quality|primary|v2"]
    d4 = decide(r4, sc["M4"], strata_labels("predvis", t, ff["edges"], cv_def))
    d3 = decide(r3, sc["M1"], strata_labels("quality", t, ff["edges"], cv_def))
    a, b = _sel(d4, y), _sel(d3, y)
    comp = {}
    for nm, (na, da, nb, db) in {
        "sFNMR": (a["fnm"], a["acc_g"], b["fnm"], b["acc_g"]),
        "sFMR": (a["fm"], a["acc_i"], b["fm"], b["acc_i"]),
        "coverage": (a["acc"], np.ones(len(y)), b["acc"], np.ones(len(y))),
    }.items():
        if da.sum() > 0 and db.sum() > 0:
            pdiff = jackknife_ratio_difference(na, da, nb, db, t["id_a"], t["id_b"])
            comp[nm] = pdiff.__dict__
    block["c3_A4_minus_A3"] = comp
    # C1: M1 - M4 log-loss on occluded pairs
    ll1, ll4 = log_loss(sc["M1"], y), log_loss(sc["M4"], y)
    d_ll = (ll1 - ll4)[occm]
    j = jackknife_interval(
        d_ll, np.ones(d_ll.size), t["id_a"][occm], t["id_b"][occm], scale="linear"
    )
    block["c1_logloss_M1_minus_M4"] = {
        "estimate": j.estimate,
        "se": j.se,
        "ci": [j.ci_low, j.ci_high],
        "p_value": normal_p_value(j.estimate, j.se),
        "n_identities": j.n_identities,
    }
    return block


def evaluate_clean_h4(
    tc: dict[str, np.ndarray], sc: dict[str, np.ndarray], ff: dict[str, Any]
) -> dict[str, Any]:
    """Clean FNMR / FMR of M4 vs M0 at their own frozen clean-cal FAR-1 % thresholds."""
    y = tc["label"]
    g = y == 1
    f4 = (sc["M4"] < ff["h4_thresholds"]["M4"]).astype(float)
    f0 = (sc["M0"] < ff["h4_thresholds"]["M0"]).astype(float)
    dd = jackknife_ratio_difference(
        f4 * g, g.astype(float), f0 * g, g.astype(float), tc["id_a"], tc["id_b"]
    )
    return {
        "fnmr_M4": float(f4[g].mean()),
        "fnmr_M0": float(f0[g].mean()),
        "fmr_M4": float((sc["M4"] >= ff["h4_thresholds"]["M4"])[~g].mean()),
        "fmr_M0": float((sc["M0"] >= ff["h4_thresholds"]["M0"])[~g].mean()),
        "delta_fnmr_M4_minus_M0": dd.__dict__,
        "non_inferior_at_0.5pp": bool(dd.ci_high < 0.005),
    }


def final(args: argparse.Namespace) -> int:
    ctx = Ctx()
    fz = json.loads((ctx.layout.private_reports / "phase3_frozen.json").read_text("utf-8"))
    if _sha({k: v for k, v in fz.items() if k != "sha256"}) != fz["sha256"]:
        raise RuntimeError("phase3 frozen config modified")
    if args.final == args.rehearse:
        raise RuntimeError("choose exactly one of --final (p3_test, logged) or --rehearse (p3_val)")
    # rehearsal exercises the identical code path on p3_val (no test access, nothing logged)
    src_role = {"test_id": "val", "test_ood": "val"} if args.rehearse else {}
    test_tag = "p3_val_pairs" if args.rehearse else "p3_test_pairs"
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    if args.final:
        with (ROOT / PROTO.test_access.run_log).open("a", encoding="utf-8", newline="\n") as fh:
            rec = {
                "run_id": run_id,
                "utc": datetime.now(tz=UTC).isoformat(),
                "script": "scripts/evaluate_phase3.py final",
                "splits": ["p3_test"],
                "frozen_config_sha256": fz["sha256"],
                "git_commit": (prov.get("git") or {}).get("commit"),
                "note": "Phase 3 confirmatory run",
            }
            fh.write(json.dumps(rec) + "\n")
    cv_def = fz["cv_selection"]["selected"]
    res: dict[str, Any] = {"folds": {}}
    c3_p, c1_p, c1_keys = [], [], []
    for fold in FAMILIES:
        ff = fz["folds"][fold]
        link = LogisticModel.from_dict(ff["heuristic_link"])
        models = {m: LogisticModel.from_dict(d) for m, d in ff["models"].items()}
        clean = ctx.clean(fold)
        fr: dict[str, Any] = {}
        tabs = {}
        for role in ("test_id", "test_ood"):
            r_src = src_role.get(role, role)
            occ = ctx.occ(fold, r_src)
            tabs[role] = build_table(
                ctx.pairs[test_tag], ctx.plan(fold, r_src), clean, occ, fold, link, ctx.ident
            )
            del occ
        tabs["test_clean"] = build_table(
            ctx.pairs[test_tag], None, clean, None, fold, link, ctx.ident
        )
        scores = {
            r: {m: models[m].predict_proba(design(t, m, cv_def)[0]) for m in models}
            for r, t in tabs.items()
        }
        for role in ("test_id", "test_ood"):
            block = evaluate_occluded(tabs[role], scores[role], ff, cv_def)
            if role == "test_ood":
                c3_p.append(block["c3_A4_minus_A3"].get("sFNMR", {}).get("p_value", 1.0))
            c1_p.append(block["c1_logloss_M1_minus_M4"]["p_value"])
            c1_keys.append((fold, role))
            fr[role] = block
        fr["h4_clean"] = evaluate_clean_h4(tabs["test_clean"], scores["test_clean"], ff)
        res["folds"][fold] = fr
        print(fold, "done", flush=True)
        del clean, tabs, scores
    adj3 = holm(c3_p)
    adj1 = holm(c1_p)
    viol = {"A4": 0, "A3": 0}
    worse = 0
    cov = []
    for k, fold in enumerate(FAMILIES):
        b = res["folds"][fold]["test_ood"]
        for a in ("A4", "A3"):
            key = "A4_predvis|primary|v2" if a == "A4" else "A3_quality|primary|v2"
            viol[a] += int("violated" in b["arms"][key]["verdict_fnm"]) + int(
                "violated" in b["arms"][key]["verdict_fm"]
            )
        c = b["c3_A4_minus_A3"]
        b["c3_holm_p_sFNMR"] = adj3[k]
        if "sFNMR" in c and c["sFNMR"]["estimate"] > 0 and adj3[k] < 0.05:
            worse += 1
        if "coverage" in c:
            cov.append(c["coverage"]["estimate"])
    for (fold, role), pa in zip(c1_keys, adj1, strict=True):
        res["folds"][fold][role]["c1_holm_p"] = pa
    c1_sig = [
        (fo, ro)
        for (fo, ro), pa in zip(c1_keys, adj1, strict=True)
        if pa < 0.05 and res["folds"][fo][ro]["c1_logloss_M1_minus_M4"]["estimate"] > 0
    ]
    res["verdicts"] = {
        "C3": {
            "violations_A4": viol["A4"],
            "violations_A3": viol["A3"],
            "folds_A4_significantly_worse": worse,
            "mean_coverage_diff_A4_minus_A3": float(np.mean(cov)) if cov else None,
            "supported": bool(
                viol["A4"] < viol["A3"] and worse == 0 and cov and np.mean(cov) >= -0.05
            ),
        },
        "C1": {
            "cells_significant_positive": len(c1_sig),
            "random_patch_ood_significant": ("F4_random_patch", "test_ood") in c1_sig,
            "supported": bool(len(c1_sig) >= 6 and ("F4_random_patch", "test_ood") in c1_sig),
        },
        "H4": {
            "folds_non_inferior": sum(
                res["folds"][f]["h4_clean"]["non_inferior_at_0.5pp"] for f in FAMILIES
            )
        },
    }
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": (
            "Phase 3 CONFIRMATORY run on p3_test (DigiFace synthetic faces, synthetic occlusions)"
        ),
        "run_id": run_id,
        "frozen_sha256": fz["sha256"],
        "cv_definition": cv_def,
        "results": res,
        "provenance": prov,
    }
    if args.rehearse:
        report["scope"] = "REHEARSAL on p3_val (code-path check only; not a result)"
        out = ctx.layout.private_reports / "phase3_final_rehearsal.json"
    else:
        out = ROOT / "reports" / "experiments" / "phase3_final.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("verdicts", json.dumps(res["verdicts"], default=str))
    return 0


# ----------------------------------------------------------------- follow-up (gate 3G)


def _subset(t: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[m] for k, v in t.items()}


def _lfw_tables(
    layout: Any, pairs_names: tuple[str, ...]
) -> tuple[list[PairRecord], dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    lfw = layout.manifests / "lfw"
    with (lfw / "detections.jsonl").open(encoding="utf-8") as fh:
        ok = {d["image_id"] for line in fh if (d := json.loads(line))["ok"]}
    pairs = [
        p
        for name in pairs_names
        for p in read_jsonl(lfw / f"{name}.jsonl", PairRecord)
        if p.image_a in ok and p.image_b in ok
    ]
    ident = {r.image_id: r.identity_id for r in read_jsonl(lfw / "images.jsonl", ImageRecord)}
    want = {p.pair_id for p in pairs}
    with gzip.open(lfw / "occlusion_plan.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [r for line in fh if (r := json.loads(line))["pair_id"] in want]
    vis = json.loads((lfw / "occlusion_visibility.json").read_text("utf-8"))
    return pairs, ident, rows, vis


def _lfw_fold(
    pairs: list[PairRecord], rows: list[dict[str, Any]], vis: dict[str, Any], fold: str
) -> tuple[list[PairRecord], dict[str, dict[str, Any]]]:
    """One pseudo-pair per LFW plan row of family ``fold`` (pair x level x one/both)."""
    by_id = {p.pair_id: p for p in pairs}
    out_pairs, out_rows = [], {}
    for r in rows:
        if r["family"] != fold:
            continue
        p = by_id[r["pair_id"]]
        pid = f"{p.pair_id}|{r['level']}|{r['mode']}"

        def spec(sd: dict[str, Any] | None, r: dict[str, Any] = r) -> dict[str, Any] | None:
            if sd is None:
                return None
            key = occluded_key(sd["image_id"], fold, r["level"], int(sd["seed"]))
            return {
                "family": fold,
                "severity_level": r["level"],
                "seed": int(sd["seed"]),
                "visibility_ratio": float(vis[key]["visibility_ratio"]),
            }

        out_pairs.append(p.model_copy(update={"pair_id": pid}))
        out_rows[pid] = {"stratum": r["mode"], "a": spec(r["a"]), "b": spec(r["b"])}
    return out_pairs, out_rows


def followup(args: argparse.Namespace) -> int:
    """Follow-up sets with the FROZEN Phase 3 configuration (no fitting of any kind)."""
    layout = resolve_data_root()
    fz = json.loads((layout.private_reports / "phase3_frozen.json").read_text("utf-8"))
    if _sha({k: v for k, v in fz.items() if k != "sha256"}) != fz["sha256"]:
        raise RuntimeError("phase3 frozen config modified")
    if args.final == args.rehearse:
        raise RuntimeError("choose exactly one of --final or --rehearse (LFW View 1 DevTest)")
    cv_def = fz["cv_selection"]["selected"]
    fdir = layout.root / "features" / "phase3" / "followup"
    ccols = ("emb_ada", "emb_arc", "norm_ada", "norm_arc", "z")
    ocols = (*ccols, "ve", "gt", "vis")
    run_id = uuid.uuid4().hex[:12]
    prov = collect_provenance(ROOT, argv=sys.argv)
    sets = ("lfw",) if args.rehearse else ("p2test", "lfw")
    if args.final:
        with (ROOT / PROTO.test_access.run_log).open("a", encoding="utf-8", newline="\n") as fh:
            rec = {
                "run_id": run_id,
                "utc": datetime.now(tz=UTC).isoformat(),
                "script": "scripts/evaluate_phase3.py followup",
                "splits": ["test (Phase 2 test_balanced)", "lfw_view2"],
                "frozen_config_sha256": fz["sha256"],
                "git_commit": (prov.get("git") or {}).get("commit"),
                "note": "Phase 3 follow-up with frozen models/rules; LFW View 2 re-used (3rd look)",
            }
            fh.write(json.dumps(rec) + "\n")
    res: dict[str, Any] = {}
    if "p2test" in sets:
        pairs = list(read_jsonl(layout.manifests / "pairs" / "test_balanced.jsonl", PairRecord))
        need = {i for p in pairs for i in (p.image_a, p.image_b)}
        ident = {
            r.image_id: r.identity_id
            for r in read_jsonl(layout.manifests / "digiface_final.jsonl", ImageRecord)
            if r.image_id in need
        }
        res["phase2_digiface_test"] = {}
        for fold in FAMILIES:
            ff = fz["folds"][fold]
            link = LogisticModel.from_dict(ff["heuristic_link"])
            models = {m: LogisticModel.from_dict(d) for m, d in ff["models"].items()}
            clean = Group(fdir, "p2_clean", (*ccols, f"ve_{fold[:2]}"))
            occ = Group(fdir, f"p2__{fold}__test_ood", ocols)
            plan = _rows(
                layout.manifests / "occlusion" / f"{fold}__test_ood__test_balanced.jsonl.gz"
            )
            t = build_table(pairs, plan, clean, occ, fold, link, ident)
            tc = build_table(pairs, None, clean, None, fold, link, ident)
            sc = {m: mdl.predict_proba(design(t, m, cv_def)[0]) for m, mdl in models.items()}
            scc = {m: mdl.predict_proba(design(tc, m, cv_def)[0]) for m, mdl in models.items()}
            res["phase2_digiface_test"][fold] = {
                "test_ood": evaluate_occluded(t, sc, ff, cv_def),
                "h4_clean": evaluate_clean_h4(tc, scc, ff),
            }
            print("p2test", fold, "done", flush=True)
            del clean, occ, t, tc
    lfw_names = ("view1_devtest_pairs",) if args.rehearse else ("view2_pairs",)
    pairs, ident, rows, vis = _lfw_tables(layout, lfw_names)
    res["lfw_derived"] = {}
    for fold in FAMILIES:
        ff = fz["folds"][fold]
        link = LogisticModel.from_dict(ff["heuristic_link"])
        models = {m: LogisticModel.from_dict(d) for m, d in ff["models"].items()}
        clean = Group(fdir, "lfw_clean", (*ccols, f"ve_{fold[:2]}"))
        occ = Group(fdir, f"lfw__{fold}", ocols)
        fpairs, frows = _lfw_fold(pairs, rows, vis, fold)
        t = build_table(fpairs, frows, clean, occ, fold, link, ident)
        tc = build_table(pairs, None, clean, None, fold, link, ident)
        sc = {m: mdl.predict_proba(design(t, m, cv_def)[0]) for m, mdl in models.items()}
        scc = {m: mdl.predict_proba(design(tc, m, cv_def)[0]) for m, mdl in models.items()}
        block = {"all": evaluate_occluded(t, sc, ff, cv_def)}
        for mode in ("one", "both"):
            m = t["stratum"] == mode
            block[mode] = evaluate_occluded(
                _subset(t, m), {k: v[m] for k, v in sc.items()}, ff, cv_def
            )
        block["h4_clean"] = evaluate_clean_h4(tc, scc, ff)
        res["lfw_derived"][fold] = block
        print("lfw", fold, "done", flush=True)
        del clean, occ, t, tc
    report = {
        "evidence_type": "real_measured_model_result",
        "scope": (
            "Phase 3 FOLLOW-UP with the frozen Phase 3 models and rules: Phase 2 DigiFace "
            "test_balanced (held-out-family plans) and LFW View 2 derived synthetic occlusions "
            "(View 2 re-used; no tuning on it)"
        ),
        "run_id": run_id,
        "frozen_sha256": fz["sha256"],
        "cv_definition": cv_def,
        "results": res,
        "provenance": prov,
    }
    if args.rehearse:
        report["scope"] = "REHEARSAL on LFW View 1 DevTest (code-path check only)"
        out = layout.private_reports / "phase3_followup_rehearsal.json"
    else:
        out = ROOT / "reports" / "experiments" / "phase3_followup.json"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
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
