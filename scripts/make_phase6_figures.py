# ruff: noqa: E501
"""Phase 6 figures for the paper, from committed reports only. Writes SVGs to reports/figures/.

1. ``phase6_a3_population_shift.svg``: A3 sFNMR per fold on the calibration-matched population
   (Phase 3 Test-OOD), the Phase 4 population (Test-ID) and the Phase 6 replication (Test-ID).
2. ``phase6_lowq_tail.svg``: A3 sFMR on p3_val below vs at-or-above the calibration-support floor.
3. ``phase6_m1_extrapolation.svg``: frozen M1 p(s, q) for fold F1 against q, by similarity, with
   the support floor and the q_low MATCH threshold.

    .venv\\Scripts\\python.exe scripts\\make_phase6_figures.py      (requires the [viz] extra)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "reports" / "experiments"
OUT = ROOT / "reports" / "figures"
FOLDS = ("F1_mask", "F2_sunglasses", "F3_hand_object", "F4_random_patch", "F5_extreme_lighting")
SHORT = ("mask", "sunglasses", "hand/object", "random patch", "lighting")
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # reference palette slots 1-3 (validated)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def a3_cells(cells: dict, role: str) -> list[tuple[float, float, float, bool]]:
    out = []
    for f in FOLDS:
        a = (
            cells[f"{f}|{role}"]["arms"]["A3_quality|primary|v2"]["selective_fnmr"]
            if f"{f}|{role}" in cells
            else cells[f][role]["arms"]["A3_quality|primary|v2"]["selective_fnmr"]
        )
        out.append((100 * a["value"], 100 * a["lo"], 100 * a["hi"], 100 * a["lo"] > 5.0))
    return out


def fig_population() -> None:
    p3 = a3_cells(
        json.loads((EXP / "phase3_final.json").read_text("utf-8"))["results"]["folds"], "test_ood"
    )
    p4 = a3_cells(json.loads((EXP / "phase4_final.json").read_text("utf-8"))["cells"], "test_id")
    p6_path = EXP / "phase6_final.json"
    p6 = (
        a3_cells(json.loads(p6_path.read_text("utf-8"))["cells"], "test_id")
        if p6_path.exists()
        else None
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=SURFACE)
    style(ax)
    x = range(len(FOLDS))
    series = [("Phase 3: calibration-matched population, held-out family", p3, SERIES[0], -0.2)]
    series.append(("Phase 4: 72-image identities, seen families", p4, SERIES[1], 0.0))
    if p6:
        series.append(
            ("Phase 6 replication: new 72-image identities, seen families", p6, SERIES[2], 0.2)
        )
    for label, vals, col, dx in series:
        xs = [i + dx for i in x]
        ys = [v for v, _, _, _ in vals]
        err = [[v - lo for v, lo, _, _ in vals], [hi - v for v, _, hi, _ in vals]]
        ax.errorbar(
            xs,
            ys,
            yerr=err,
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=1.4,
            capsize=3,
            markersize=6,
            markeredgecolor=SURFACE,
            label=label,
            zorder=3,
        )
    ax.axhline(5.0, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(len(FOLDS) - 0.55, 5.1, "5 % target", color=INK, fontsize=8.5, ha="right", va="bottom")
    ax.set_xticks(list(x), SHORT, color=INK2)
    ax.set_ylabel("A3 selective FNMR (%), 95 % CI", color=INK2, fontsize=9)
    top = max(hi for _, vals, _, _ in series for _, _, hi, _ in vals)
    ax.set_ylim(0, max(9.5, top + 2.8))  # headroom: legend above the data
    ax.set_title(
        "Quality-stratified A3: miss rate on accepted genuine pairs, by fold and population",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "phase6_a3_population_shift.svg", facecolor=SURFACE)
    plt.close(fig)


def fig_lowq() -> None:
    r = json.loads((EXP / "phase6_lowq_audit.json").read_text("utf-8"))["results"]["folds"]
    below = [100 * r[f]["val_by_band"]["q < floor"]["sFMR"] for f in FOLDS]
    above = [100 * r[f]["val_by_band"]["q >= floor"]["sFMR"] for f in FOLDS]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    style(ax)
    w = 0.36
    xs = list(range(len(FOLDS)))
    b1 = ax.bar(
        [i - w / 2 - 0.01 for i in xs],
        below,
        w,
        color=SERIES[1],
        label="q below calibration-support floor (~1 % of pairs)",
    )
    b2 = ax.bar(
        [i + w / 2 + 0.01 for i in xs], above, w, color=SERIES[0], label="q at or above the floor"
    )
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.6,
                f"{b.get_height():.1f}",
                ha="center",
                fontsize=8,
                color=INK2,
            )
    ax.axhline(5.0, color=INK, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(len(FOLDS) - 0.5, 5.6, "5 % target", color=INK, fontsize=8.5, ha="right")
    ax.set_xticks(xs, SHORT, color=INK2)
    ax.set_ylabel("A3 selective FMR on p3_val (%)", color=INK2, fontsize=9)
    ax.set_title(
        "Frozen A3 false-match rate collapses below the calibration support of quality\n(development data; exploratory)",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "phase6_lowq_tail.svg", facecolor=SURFACE)
    plt.close(fig)


def fig_m1() -> None:
    r = json.loads((EXP / "phase6_lowq_audit.json").read_text("utf-8"))["results"]
    qs = r["m1_surface_q_grid"]
    surf = r["folds"]["F1_mask"]["m1_surface"]
    floor = r["folds"]["F1_mask"]["q_support_floor"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    style(ax)
    ax.axvspan(min(qs), floor, color=GRID, alpha=0.6, linewidth=0)
    ax.text(min(qs) + 0.2, 0.04, "below calibration\nsupport (1st pct.)", fontsize=8, color=INK2)
    for k, col in zip(("s=+0.0", "s=+0.1", "s=+0.2"), SERIES, strict=True):
        ax.plot(
            qs, surf[k], color=col, linewidth=2, marker="o", markersize=4, label=f"cosine {k[2:]}"
        )
    ax.hlines(0.833, min(qs), 19.895, color=INK, linewidth=1.0, linestyles=(0, (4, 3)))
    ax.text(
        19.895,
        0.85,
        "q_low MATCH threshold 0.833 (q_low: q < 19.9)",
        fontsize=8,
        color=INK,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("pair quality q (AdaFace feature norm)", color=INK2, fontsize=9)
    ax.set_ylabel("frozen M1 fused score p", color=INK2, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Frozen M1 (fold F1) extrapolates towards MATCH as quality drops,\neven at impostor-like similarity",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "phase6_m1_extrapolation.svg", facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.fonttype": "none", "svg.hashsalt": "veilx"}
    )
    fig_population()
    fig_lowq()
    fig_m1()
    print("written:", *sorted(p.name for p in OUT.glob("phase6_*.svg")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
