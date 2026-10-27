"""Static SVG figures for the Phase 2 reports (aggregated numbers only; no images, no ids).

Reads ``reports/experiments/phase2_final.json`` (and the extraction logs) and writes
``reports/figures/*.svg``. Colours follow the dataviz reference palette: categorical slots
1-3 only (validated all-pairs), one sequential blue ramp for heatmaps, recessive grid, thin
marks, legends plus direct labels.

    .venv\\Scripts\\python.exe scripts\\make_figures.py      (requires the [viz] extra)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("svg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
OUT = ROOT / "reports" / "figures"

SLOT = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (blue, orange, aqua)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
NAMES = {"adaface_ir101_webface4m": "AdaFace IR-101", "arcface_ir101_webface4m": "ArcFace IR-101"}
FAMS = ("F1_mask", "F2_sunglasses", "F3_hand_object", "F4_random_patch", "F5_extreme_lighting")
SHORT = {f: f.split("_", 1)[1].replace("_", " ") for f in FAMS}
SCOPE = "DigiFace-1M (synthetic faces), synthetic occlusions - not a real-world benchmark"

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "svg.hashsalt": "veilx",  # deterministic SVG ids
    }
)


FMT = "svg"


def _save(fig: Any, name: str, note: str = SCOPE) -> None:
    fig.text(0.01, 0.005, note, fontsize=7, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"Date": None} if FMT == "svg" else None
    fig.savefig(OUT / f"{name}.{FMT}", metadata=meta, dpi=110)
    plt.close(fig)
    print(f"wrote {name}.{FMT}")


def _val(r: dict[str, Any] | None) -> float:
    return float("nan") if not r or r.get("value") is None else float(r["value"])


def _err(r: dict[str, Any]) -> tuple[float, float]:
    v = _val(r)
    lo = r.get("lo") if r.get("lo") is not None else v
    hi = r.get("hi") if r.get("hi") is not None else v
    return max(v - lo, 0.0), max(hi - v, 0.0)


# ----------------------------------------------------------------- figures


def fig_roc_clean(res: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for k, (bid, r) in enumerate(res.items()):
        roc = r["clean"]["roc_points"]
        ax.plot(roc["fpr"], roc["tpr"], color=SLOT[k], label=NAMES[bid])
        auc = r["clean"]["test_balanced"]["roc_auc"]
        ax.annotate(
            f"{NAMES[bid]} (AUC {auc:.4f})",
            (roc["fpr"][8], roc["tpr"][8]),
            textcoords="offset points",
            xytext=(6, -12 - 12 * k),
            fontsize=8,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("False match rate (log)")
    ax.set_ylabel("True match rate")
    ax.set_title("Clean verification ROC, test_balanced")
    ax.legend(loc="lower right")
    _save(fig, "roc_clean")


def fig_scores_clean(res: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, len(res), figsize=(8.4, 3.2), sharey=True)
    for ax, (bid, r) in zip(np.atleast_1d(axes), res.items(), strict=True):
        h = r["clean"]["score_histogram"]
        e = np.asarray(h["edges"])
        for k, cls in enumerate(("genuine", "impostor")):
            c = np.asarray(h[cls], dtype=float)
            ax.step(e[:-1], c / c.sum(), where="post", color=SLOT[k], label=cls)
        t = r["thresholds"]["cal_far_1pct"]
        ax.axvline(t, color=INK2, lw=1, ls="--")
        ax.text(t, ax.get_ylim()[1] * 0.92, " cal FAR 1%", fontsize=7, color=INK2)
        ax.set_title(NAMES[bid])
        ax.set_xlabel("Cosine similarity")
    np.atleast_1d(axes)[0].set_ylabel("Fraction of pairs")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    _save(fig, "score_distributions_clean")


def fig_lofo_auc(res: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, len(res), figsize=(9, 3.4), sharey=True)
    x = np.arange(len(FAMS))
    for ax, (bid, r) in zip(np.atleast_1d(axes), res.items(), strict=True):
        clean = r["clean"]["test_balanced"]["roc_auc"]
        ax.axhline(clean, color=SLOT[0], lw=1.5, label="clean")
        for k, role in enumerate(("test_id", "test_ood"), start=1):
            y = [r["folds"][f][role]["overall"]["roc_auc"] for f in FAMS]
            ax.plot(
                x + (k - 1.5) * 0.12,
                y,
                "o",
                ms=6,
                color=SLOT[k],
                mec=SURFACE,
                mew=1,
                label=role.replace("_", "-").upper().replace("TEST-", "Test-"),
            )
        ax.set_xticks(x, [SHORT[f] for f in FAMS], rotation=20, ha="right")
        ax.set_title(NAMES[bid])
        ax.set_xlabel("Held-out family (fold)")
    np.atleast_1d(axes)[0].set_ylabel("ROC-AUC")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    _save(fig, "lofo_auc_by_fold")


def _pooled(r: dict[str, Any], role: str, group: str, metric: str) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = {}
    for f in FAMS:
        for name, blk in r["folds"][f][role]["subgroups"].get(group, {}).items():
            m = blk[metric]
            acc = out.setdefault(name, [0, 0])
            acc[0] += m["k"]
            acc[1] += m["n"]
    return {k: (v[0], v[1]) for k, v in out.items()}


def fig_error_vs_visibility(res: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, len(res), figsize=(9, 6), sharex=True)
    axes = np.atleast_2d(axes)
    for j, (bid, r) in enumerate(res.items()):
        for i, metric in enumerate(("fnmr", "fmr")):
            ax = axes[i, j]
            for k, role in enumerate(("test_id", "test_ood"), start=1):
                p = _pooled(r, role, "covis_bin", metric)
                names = sorted(p, key=lambda s: float(s.strip("[)").split(",")[0]))
                y = [p[n][0] / p[n][1] if p[n][1] else np.nan for n in names]
                ax.plot(range(len(names)), y, "-o", ms=5, color=SLOT[k], mec=SURFACE, label=role)
            labels = ["clean" if n.startswith("[1.00") else n for n in names]
            ax.set_xticks(range(len(names)), labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel(f"{metric.upper()} at cal FAR 1%")
            if i == 0:
                ax.set_title(NAMES[bid])
            if i == 1:
                ax.set_xlabel("Pair co-visibility bin (oracle mask), pooled over folds")
    axes[0, 0].legend(loc="upper right")
    _save(fig, "error_vs_covisibility")


def fig_severity_heatmap(res: dict[str, Any]) -> None:
    levels = ("low", "medium", "high")
    fig, axes = plt.subplots(1, len(res), figsize=(9, 3.4))
    for ax, (bid, r) in zip(np.atleast_1d(axes), res.items(), strict=True):
        m = np.full((len(FAMS), len(levels)), np.nan)
        for i, f in enumerate(FAMS):  # held-out family in its own fold (Test-OOD)
            sub = r["folds"][f]["test_ood"]["subgroups"]["max_level"]
            for j, lv in enumerate(levels):
                if lv in sub:
                    m[i, j] = _val(sub[lv]["fnmr"])
        im = ax.imshow(m, cmap=SEQ, aspect="auto", vmin=0, vmax=np.nanmax(m))
        for i in range(len(FAMS)):
            for j in range(len(levels)):
                if np.isfinite(m[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{100 * m[i, j]:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#ffffff" if m[i, j] > 0.6 * np.nanmax(m) else INK,
                    )
        ax.set_xticks(range(3), levels)
        ax.set_yticks(range(len(FAMS)), [SHORT[f] for f in FAMS])
        ax.grid(False)
        ax.set_title(f"{NAMES[bid]}: Test-OOD FNMR by max severity in pair", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "severity_fnmr_heatmap")


def fig_ltt(res: dict[str, Any], target: str, version: str = "v1") -> None:
    suffix = "" if version == "v1" else "_v2"
    arms = tuple(a + suffix for a in ("global", "stratified_vis", "clean_cal"))
    for bid, r in res.items():
        fig, axes = plt.subplots(3, 2, figsize=(9, 7.6), sharex=True)
        x = np.arange(len(FAMS))
        for j, role in enumerate(("test_id", "test_ood")):
            for i, key in enumerate(("selective_fmr", "selective_fnmr", "coverage")):
                ax = axes[i, j]
                for k, arm in enumerate(arms):
                    s = [r["folds"][f][role]["selective"][target][arm] for f in FAMS]
                    xs = x + (k - 1) * 0.18
                    if key == "coverage":
                        ax.plot(
                            xs,
                            [v["coverage"] for v in s],
                            "o",
                            ms=5,
                            color=SLOT[k],
                            mec=SURFACE,
                            label=arm,
                        )
                    else:
                        y = [_val(v[key]) for v in s]
                        e = np.array([_err(v[key]) for v in s]).T
                        ax.errorbar(
                            xs,
                            y,
                            yerr=e,
                            fmt="o",
                            ms=5,
                            color=SLOT[k],
                            mec=SURFACE,
                            elinewidth=1,
                            capsize=0,
                            label=arm,
                        )
                if key != "coverage":
                    tgt = r["folds"][FAMS[0]][role]["selective"][target][arms[0]][
                        "target_fm" if key == "selective_fmr" else "target_fnm"
                    ]
                    ax.axhline(tgt, color=INK2, lw=1, ls="--")
                    ax.text(len(FAMS) - 0.5, tgt, " target", fontsize=7, color=INK2, va="bottom")
                ax.set_ylabel(key.replace("_", " "))
                if i == 0:
                    ax.set_title(
                        "Test-ID (seen families)"
                        if role == "test_id"
                        else "Test-OOD (held-out family)"
                    )
                if i == 2:
                    ax.set_xticks(x, [SHORT[f] for f in FAMS], rotation=20, ha="right")
        axes[0, 0].legend(loc="upper left", fontsize=7)
        label = "pre-registered order" if version == "v1" else "post-hoc p-value order (D-042)"
        fig.suptitle(
            f"{NAMES[bid]}: realised selective risk of frozen LTT rules ({target}; {label})",
            fontsize=10,
            color=INK,
        )
        _save(fig, f"ltt_realised_{bid.split('_')[0]}_{target}_{version}")


def fig_throughput() -> None:
    rows = []
    for dev in ("cuda", "cpu"):
        p = EXP / f"extraction_pilot_{dev}.json"
        if p.exists():
            g = json.loads(p.read_text("utf-8"))["groups"][0]
            for bid, v in g["model_images_per_s"].items():
                rows.append(
                    (f"{NAMES[bid]}\n{'GPU fp16' if dev == 'cuda' else 'CPU fp32'}", v, dev)
                )
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    y = np.arange(len(rows))
    ax.barh(
        y,
        [r[1] for r in rows],
        color=[SLOT[0] if r[2] == "cuda" else SLOT[1] for r in rows],
        height=0.6,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for k, r in enumerate(rows):
        ax.text(r[1], k, f" {r[1]:.0f} img/s", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Model throughput (images/s, log; batch 256 GPU / 128 CPU)")
    ax.set_title("Embedding throughput, RTX 4060 Laptop vs CPU")
    _save(
        fig, "throughput", note="Model forward time only; excludes decoding and occlusion rendering"
    )


LFW_SCOPE = (
    "LFW (Kaggle v4) real photographs; "
    "occlusions are SYNTHETIC overlays (derived, not official LFW)"
)
PATHS = (
    ("clean_det_forced_nonmatch", "detector (all 6,000; failures = non-match)"),
    ("clean_det_excluding_failed", "detector (failed pairs excluded)"),
    ("clean_fixed_crop", "fixed crop (no detector)"),
)


def fig_lfw_clean(lfw: dict[str, Any]) -> None:
    res = lfw["results"]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(PATHS))
    for k, (bid, r) in enumerate(res.items()):
        acc = [r[p]["official_10fold_accuracy"]["mean"] for p, _ in PATHS]
        sd = [r[p]["official_10fold_accuracy"]["std"] or 0 for p, _ in PATHS]
        ax.errorbar(
            acc,
            y + (k - 0.5) * 0.2,
            xerr=sd,
            fmt="o",
            ms=6,
            color=SLOT[k],
            mec=SURFACE,
            elinewidth=1,
            capsize=0,
            label=NAMES[bid],
        )
        for j, a in enumerate(acc):
            ax.text(a, j + (k - 0.5) * 0.2 + 0.08, f" {100 * a:.2f}%", fontsize=7, color=INK)
    ax.set_yticks(y, [lbl for _, lbl in PATHS], fontsize=8)
    ax.set_xlabel("Official View-2 10-fold accuracy (mean ± s.d. over folds)")
    ax.set_title("Clean LFW, official protocol")
    ax.legend(loc="lower left", fontsize=7)
    _save(fig, "lfw_clean_official_accuracy", note=LFW_SCOPE)


def fig_lfw_vs_digiface(lfw: dict[str, Any], dig: dict[str, Any]) -> None:
    """FNMR at the DigiFace calibration threshold (FAR 1 %): DigiFace Test-OOD vs LFW derived."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    x = np.arange(len(FAMS))
    for ax, bid in zip(axes, lfw["results"], strict=True):
        d = [
            dig["results"][bid]["folds"][f]["test_ood"]["overall"]["operating_points"][
                "cal_far_1pct"
            ]["fnmr"]["value"]
            for f in FAMS
        ]
        lv = [
            lfw["results"][bid]["derived_occlusion_det"][f]["pooled_at_digiface_cal_far_1pct"][
                "fnmr"
            ]["value"]
            for f in FAMS
        ]
        ax.plot(
            x - 0.1,
            d,
            "o",
            ms=6,
            color=SLOT[0],
            mec=SURFACE,
            label="DigiFace Test-OOD (synthetic faces)",
        )
        ax.plot(
            x + 0.1, lv, "o", ms=6, color=SLOT[1], mec=SURFACE, label="LFW derived (real photos)"
        )
        ax.set_xticks(x, [SHORT[f] for f in FAMS], rotation=20, ha="right")
        ax.set_title(NAMES[bid])
    axes[0].set_ylabel("FNMR at DigiFace cal FAR-1% threshold")
    axes[0].legend(loc="upper left", fontsize=7)
    _save(fig, "lfw_vs_digiface_occlusion_fnmr", note=LFW_SCOPE + "; mixes differ (see docs)")


def fig_lfw_severity(lfw: dict[str, Any]) -> None:
    levels = ("low", "medium", "high")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (bid, r) in zip(axes, lfw["results"].items(), strict=True):
        m = np.array(
            [
                [
                    r["derived_occlusion_det"][f]["cells"][f"{lv}:both"][
                        "at_digiface_cal_far_1pct"
                    ]["fnmr"]["value"]
                    for lv in levels
                ]
                for f in FAMS
            ]
        )
        im = ax.imshow(m, cmap=SEQ, aspect="auto", vmin=0, vmax=float(np.nanmax(m)))
        for i in range(len(FAMS)):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{100 * m[i, j]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#ffffff" if m[i, j] > 0.6 * np.nanmax(m) else INK,
                )
        ax.set_xticks(range(3), levels)
        ax.set_yticks(range(len(FAMS)), [SHORT[f] for f in FAMS])
        ax.grid(False)
        ax.set_title(f"{NAMES[bid]}: LFW FNMR, both sides occluded", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "lfw_derived_occlusion_heatmap", note=LFW_SCOPE)


def fig_lfw_detection(det: dict[str, Any]) -> None:
    h = det["faces_per_image_histogram"]
    keys = sorted(h, key=int)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(
        range(len(keys)),
        [h[k] for k in keys],
        color=[SLOT[1] if k == "0" else SLOT[0] for k in keys],
        edgecolor=SURFACE,
        linewidth=2,
    )
    for i, k in enumerate(keys):
        ax.text(i, h[k], f"{h[k]:,}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)), keys)
    ax.set_xlabel("Faces detected per LFW image (0 = detection failure)")
    ax.set_ylabel("Images (log)")
    ax.set_title(f"SCRFD on LFW: {det['failures_no_face']} failures of {det['n_images']:,}")
    _save(
        fig,
        "lfw_detection_outcomes",
        note="Centre-nearest face is used; failures are counted, never dropped",
    )


def main() -> int:
    global OUT, FMT
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=EXP / "phase2_final.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--format", default="svg", choices=["svg", "png"])
    args = ap.parse_args()
    OUT, FMT = args.out, args.format
    if FMT != "svg":
        mpl.use("agg", force=True)
    final = json.loads(args.input.read_text("utf-8"))
    res = final["results"]
    fig_roc_clean(res)
    fig_scores_clean(res)
    fig_lofo_auc(res)
    fig_error_vs_visibility(res)
    fig_severity_heatmap(res)
    for t in final["targets"]:
        for version in ("v1", "v2"):
            fig_ltt(res, t, version)
    fig_throughput()
    lfw_path = EXP / "lfw_final.json"
    if lfw_path.exists():
        lfw = json.loads(lfw_path.read_text("utf-8"))
        fig_lfw_clean(lfw)
        fig_lfw_vs_digiface(lfw, final)
        fig_lfw_severity(lfw)
    det_path = EXP / "lfw_detection.json"
    if det_path.exists():
        fig_lfw_detection(json.loads(det_path.read_text("utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
