# ruff: noqa: E501, RUF001  (long user-facing text; typographic dashes and multiplication signs)
"""User-facing text, numeric summaries and simple image renderings for the research app.

All numbers quoted here are copied from committed reports (Phase 3: ``docs/phase3_results.md``;
Phase 4: ``docs/phase4_results.md``) and carry their run identifiers. No claim of certification,
safety or deployment readiness is made anywhere. Tests check the wording.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

DISCLAIMER = (
    "**Research demonstration only — not an identity-verification product.** VEIL-X runs locally on this "
    "computer. Use only images you are authorised to process. The output is the response of a frozen research "
    "decision rule; it does not prove whether two images show the same person and must not be used for "
    "identification, surveillance or any decision affecting people."
)

LIMITATIONS_MD = """
### Calibration and domain-shift limitations (read before interpreting any result)

- **The 5 % selective-error target was NOT maintained on the new Phase 4 population.** On the untouched
  Phase 4 partition (DigiFace identities 2000–5999, run `903f6a5c7de3`) every decision rule, including the one
  used here (A3), exceeded the 5 % selective false-non-match target even on occlusion types seen during
  calibration (A3: 5.4–6.3 % on seen families; up to 7.6 % on held-out families). A pre-registered
  replication on 4,000 new identities (Phase 6, run `1c291322aa27`) reproduced the failure (4 of 5 folds).
- **A3 was selected because it performed better than the alternatives** (fewest violations; no candidate was
  safe relative to it, D-050), **not because it is universally safe.**
- **Calibration depends on the population and on preprocessing.** The rule was calibrated on synthetic DigiFace
  pairs (Phase 3 `p3_cal`) with ArcFace-template alignment. Genuine pairs of the Phase 4 population were about
  4× harder (clean miss rate of raw similarity 4.81 % vs 1.12 %), and the guarantee did not transfer. Real
  photographs, other cameras, other demographics and other alignment pipelines are different populations.
- **Unseen physical occlusions have not been independently validated.** All occluders in VEIL-X were synthetic
  (rendered or composited). No physical-occlusion dataset has been evaluated.
- **No formal out-of-distribution risk guarantee exists.** Learn-then-Test certificates hold only for the
  calibration distribution.
- **LFW and DigiFace results do not establish deployment readiness.** LFW was used under academic authorisation
  for research evaluation only; DigiFace faces are synthetic.
- **The frozen score is unreliable for very low-quality inputs.** Below the calibration support of quality it
  extrapolates (non-face images reach a score of 1.000; the false-match rate there was 18–42 % on development data
  and 23 % on new data). This app therefore abstains below that support (support guard); the research rule does not.
- **A numerical score never proves that two images show the same person.** The fused score is a logistic output
  fitted on synthetic data — not a probability of identity and not a guaranteed confidence.
"""

WHY_NOT_VISIBILITY_MD = """
### Why learned visibility is shown but not used in the decision

| Finding (measured) | Source |
|---|---|
| Phase 3 learned-visibility calibration (A4) failed on an unseen occluder (random patch): selective false-non-match **8.52 %** vs 2.12 % for quality strata (A3) | Phase 3 confirmatory run `bb5d9244bb3c` |
| Root cause: the estimator learned *occluder classes*; on unseen patches it called **88 %** of covered face cells visible, with confident predictions; another person's face patch is called visible by every model | Phase 4 diagnostics (development data) |
| The conservative estimator (VE-C) **fixed the overestimation** (held-out random patch: share overestimated by > 0.10 fell from **94.6 % to 21.9 %**) | Phase 4 run `903f6a5c7de3` |
| …but **no visibility-based rule beat A3**: conservative, novelty-aware and quality-first hybrid rules all had significantly higher selective false-non-match than A3 in 6–9 of 10 primary cells; even **ground-truth visibility** was worse than A3 on every never-seen hair-occluder cell | Phase 4 run `903f6a5c7de3` |
| Co-visibility features **worsened** the fused model's log-loss on all never-seen-occluder cells | Phase 4 run `903f6a5c7de3` |

The visibility map and novelty flag are therefore **explanatory diagnostics only**. They are computed after the
decision and cannot change it.
"""


def decision_html(label: str, fused: float, stratum: str) -> str:
    """Neutral colours only: no green 'safe' / 'certified' state."""
    colours = {"MATCH": "#1d4e89", "NON-MATCH": "#4a4a4a", "ABSTAIN": "#9a5b00"}
    meaning = {
        "MATCH": "the frozen rule accepts the pair as a match",
        "NON-MATCH": "the frozen rule rejects the pair",
        "ABSTAIN": "the frozen rule declines to decide",
    }
    c = colours.get(label, "#4a4a4a")
    score = "n/a" if not np.isfinite(fused) else f"{fused:.3f}"
    return (
        f"<div style='border:2px solid {c};border-radius:8px;padding:12px'>"
        f"<div style='font-size:1.6em;font-weight:700;color:{c}'>{html.escape(label)}</div>"
        f"<div>Research output of the frozen A3 rule: {html.escape(meaning.get(label, ''))}.</div>"
        f"<div style='font-size:0.9em;margin-top:6px'>Fused score {score} (logistic output — not a probability of "
        f"identity) · quality stratum {html.escape(stratum)}</div>"
        "<div style='font-size:0.85em;margin-top:6px'>Not certified. The 5 % target was not maintained under "
        "population shift; see the limitations panel.</div></div>"
    )


def rule_bar(
    fused: float, tau_low: float | None, tau_high: float | None, width: int = 520, height: int = 70
) -> NDArray[np.uint8]:
    """Horizontal [0, 1] score axis with the stratum's NON-MATCH / ABSTAIN / MATCH regions."""
    img = Image.new("RGB", (width, height), (250, 250, 250))
    d = ImageDraw.Draw(img)
    x0, x1, y0, y1 = 10, width - 10, 18, 42

    def px(v: float) -> int:
        return int(x0 + (x1 - x0) * min(max(v, 0.0), 1.0))

    if tau_low is not None and tau_high is not None:
        d.rectangle([x0, y0, px(tau_low), y1], fill=(200, 200, 200))
        if tau_high > tau_low:
            d.rectangle([px(tau_low), y0, px(tau_high), y1], fill=(240, 200, 140))
        d.rectangle([px(tau_high), y0, x1, y1], fill=(160, 190, 225))
        d.text((x0, 2), "NON-MATCH", fill=(60, 60, 60))
        if tau_high > tau_low:
            d.text((px(tau_low) + 2, 2), "ABSTAIN", fill=(120, 70, 0))
        d.text((max(px(tau_high) + 2, x1 - 60), 2), "MATCH", fill=(20, 60, 120))
    else:
        d.rectangle([x0, y0, x1, y1], fill=(240, 200, 140))
        d.text((x0, 2), "ABSTAIN (no usable rule / input)", fill=(120, 70, 0))
    if np.isfinite(fused):
        x = px(fused)
        d.polygon([(x, y1 + 2), (x - 6, y1 + 14), (x + 6, y1 + 14)], fill=(0, 0, 0))
        d.text((min(x + 8, width - 80), y1 + 6), f"score {fused:.3f}", fill=(0, 0, 0))
    d.text((x0, height - 12), "0", fill=(90, 90, 90))
    d.text((x1 - 6, height - 12), "1", fill=(90, 90, 90))
    return np.asarray(img, dtype=np.uint8)


def visibility_overlay(crop: NDArray[np.uint8], vis_map: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Blend a 28x28 predicted-visibility map onto the 112x112 crop: blue = predicted visible,
    orange = predicted not visible. Diagnostic only."""
    v = np.repeat(np.repeat(np.clip(np.asarray(vis_map, np.float64), 0, 1), 4, 0), 4, 1)[..., None]
    colour = v * np.array([60.0, 110.0, 200.0]) + (1 - v) * np.array([235.0, 120.0, 20.0])
    out = 0.55 * crop.astype(np.float64) + 0.45 * colour
    return np.asarray(np.clip(out, 0, 255), dtype=np.uint8)


def annotate_faces(
    rgb: NDArray[np.uint8], faces: list[Any], chosen: int | None
) -> NDArray[np.uint8]:
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f.box
        colour = (20, 80, 200) if i == chosen else (230, 120, 0)
        d.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        d.text((x1 + 3, y1 + 3), f"face {i}", fill=colour)
    return np.asarray(img, dtype=np.uint8)


def abstention_text(reasons: tuple[str, ...], label: str) -> str:
    if label != "ABSTAIN":
        return "No abstention: the frozen rule returned a decision for this pair."
    if not reasons:
        return "ABSTAIN (no recorded reason)."
    items = "\n".join(f"- {r}" for r in reasons)
    return (
        "**Recorded abstention reasons** (descriptive; they state which rule condition applied, not a causal "
        f"explanation of the images):\n{items}"
    )


# Phase 6 (6C): what each frozen A3 system is. A3 cells: coverage / sFNMR / sFMR in %;
# x = 95 % identity-jackknife CI excludes the 5 % target; + = point estimate above target.
# Sources: docs/phase3_results.md §5 (run bb5d9244bb3c), reports/experiments/phase4_tables.md
# (run 903f6a5c7de3); Phase 6 replication run 1c291322aa27 (docs/phase6_results.md).
FAMILY_NAMES = {
    "F1_mask": "mask",
    "F2_sunglasses": "sunglasses",
    "F3_hand_object": "hand / object",
    "F4_random_patch": "random patch",
    "F5_extreme_lighting": "extreme lighting",
}
A3_EVIDENCE = {
    "F1_mask": (
        "93.7 / 1.94 / 2.56",
        "94.0 / 5.89 x / 1.27",
        "90.9 / 7.55 x / 2.59",
        "93.2 / 5.76 x / 1.11",
        "93.8 / 5.90 x / 1.40",
    ),
    "F2_sunglasses": (
        "96.8 / 1.87 / 1.24",
        "89.7 / 6.34 x / 1.67",
        "95.7 / 5.72 x / 1.27",
        "91.1 / 5.46 x / 1.08",
        "89.8 / 6.49 x / 1.73",
    ),
    "F3_hand_object": (
        "94.3 / 1.59 / 1.23",
        "90.2 / 6.16 x / 1.59",
        "92.4 / 5.84 x / 1.24",
        "90.6 / 5.09 + / 1.03",
        "90.0 / 6.41 x / 1.77",
    ),
    "F4_random_patch": (
        "88.6 / 2.12 / 2.00",
        "93.2 / 5.36 x / 1.63",
        "86.0 / 7.26 x / 2.04",
        "91.1 / 4.65 / 1.15",
        "93.0 / 5.15 + / 1.75",
    ),
    "F5_extreme_lighting": (
        "96.1 / 1.08 / 1.27",
        "89.4 / 6.05 x / 2.00",
        "94.7 / 4.33 / 1.47",
        "90.7 / 4.65 / 1.26",
        "89.6 / 6.39 x / 2.02",
    ),
}


def fold_card_md(fold: str, freeze_sha: str) -> str:
    """User-facing description of the ACTIVE experimental configuration (one frozen A3 system)."""
    held = FAMILY_NAMES[fold]
    seen = ", ".join(v for k, v in FAMILY_NAMES.items() if k != fold)
    p3, p4id, p4ood, p4e1, p6id = A3_EVIDENCE[fold]
    return f"""**Active experimental configuration: frozen A3 system `{fold}`** (Phase 3 freeze `{freeze_sha[:12]}…`). Research configuration; **not a deployment calibration.**

- **Calibrated on:** 6,146 identity-disjoint pairs (3,073 genuine) of *synthetic* DigiFace-1M identities (`p3_cal_independent`; 9,219 identities, of which 9,033 are 5-image and only 186 are 72-image identities), clean and occluded with the four *seen* synthetic occluder families ({seen}). The **{held}** family was held out of this system's calibration.
- **Score and rule:** fused score M1 from the AdaFace cosine and AdaFace feature norm; three quality tertiles (edges from `p3_val`); Learn-then-Test, target selective FNMR ≤ 5 % and selective FMR ≤ 5 % at δ = 0.05.
- **Why it is not universal:** it is one of five leave-one-family-out systems that exist only to test generalisation to an unseen occluder; none was fitted on all families, on real photographs or on any deployment population. `F1_mask` is the default only because it is the first registered fold (D-051).

| Measured A3 result (coverage / sFNMR / sFMR, %) | Value |
|---|---|
| Phase 3 confirmatory, held-out {held} (`bb5d9244bb3c`) | {p3} |
| Phase 4 new population, seen families (`903f6a5c7de3`) | {p4id} |
| Phase 4 new population, held-out {held} | {p4ood} |
| Phase 4 new population, never-seen hair occluder | {p4e1} |
| Phase 6 replication, new identities, seen families (`1c291322aa27`) | {p6id} |

x = 95 % CI excludes the 5 % target; + = point estimate above target. On the 72-image population the miss-rate target was **not** held, in Phase 4 and again in the Phase 6 replication."""
