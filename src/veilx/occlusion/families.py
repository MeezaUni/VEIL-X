"""Procedural challenge generators (no external assets; therefore no asset licences).

Every generator has the signature ``(image, landmarks, level, rng) -> (image, mask, params)``:
``mask`` marks pixels whose facial content is physically covered (for lighting: pixels whose
appearance is photometrically destroyed — clipped/crushed or strongly darkened; flagged
non-physical by the engine). ``params`` are the sampled numeric parameters, recorded for
reproducibility. All randomness comes from ``rng`` (seeded by the engine).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from veilx.occlusion.geometry import BoolMask, Landmarks, ellipse_mask, face_region, polygon_mask

U8 = NDArray[np.uint8]
F64 = NDArray[np.float64]
Level = Literal["low", "medium", "high"]
LEVELS: tuple[Level, ...] = ("low", "medium", "high")
Generator = Callable[
    [U8, Landmarks, Level, np.random.Generator], tuple[U8, BoolMask, dict[str, float]]
]


def _composite(
    img: U8,
    mask: BoolMask,
    colour: F64,
    rng: np.random.Generator,
    *,
    texture: float,
    alpha: float = 1.0,
    shade: F64 | None = None,
) -> U8:
    out = img.astype(np.float64)
    h, w = mask.shape
    layer = np.broadcast_to(colour, (h, w, 3)).astype(np.float64).copy()
    if shade is not None:
        layer *= shade[..., None]
    if texture > 0:
        layer += rng.normal(0, texture, (h, w, 1))
    m = mask[..., None]
    out = np.where(m, (1 - alpha) * out + alpha * layer, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _vertical_shade(h: int, w: int, lo: float, hi: float) -> F64:
    return np.repeat(np.linspace(hi, lo, h)[:, None], w, axis=1)


# --------------------------------------------------------------------------- F1 masks

_MASK_COLOURS = np.array(
    [[175, 205, 225], [235, 235, 235], [40, 40, 45], [150, 150, 155], [120, 160, 140]],
    dtype=np.float64,
)


def surgical_mask(
    img: U8, lm: Landmarks, level: Level, rng: np.random.Generator
) -> tuple[U8, BoolMask, dict[str, float]]:
    h, w = img.shape[:2]
    d = lm.eye_distance
    eye_y, nose_y, mouth_y = lm.eye_center[1], lm.nose[1], lm.mouth_center[1]
    top_frac = {"low": 0.25, "medium": -0.30, "high": -0.65}[level] + rng.uniform(-0.08, 0.08)
    top = (
        nose_y + top_frac * (nose_y - eye_y)
        if top_frac < 0
        else nose_y + top_frac * (mouth_y - nose_y)
    )
    bottom = mouth_y + rng.uniform(0.65, 0.85) * d
    half_top = rng.uniform(0.95, 1.10) * d
    half_mid = half_top * rng.uniform(1.00, 1.10)
    cx = float(lm.eye_center[0] + rng.uniform(-0.04, 0.04) * d)
    mid_y = (top + bottom) / 2
    pts = [
        (cx - half_top, top),
        (cx + half_top, top),
        (cx + half_mid, mid_y),
        (cx + 0.55 * half_mid, bottom - 0.05 * d),
        (cx, bottom),
        (cx - 0.55 * half_mid, bottom - 0.05 * d),
        (cx - half_mid, mid_y),
    ]
    mask = polygon_mask(h, w, np.array(pts))
    colour = _MASK_COLOURS[int(rng.integers(len(_MASK_COLOURS)))]
    shade = _vertical_shade(h, w, 0.85, 1.0)
    for k in range(3):  # pleats
        y = top + (k + 1) * (bottom - top) / 4
        band = (np.abs(np.arange(h)[:, None] + 0.5 - y) < max(0.6, 0.015 * h)) & np.ones(
            (1, w), bool
        )
        shade = np.where(band, shade * 0.88, shade)
    out = _composite(img, mask, colour, rng, texture=3.0, shade=shade)
    return (
        out,
        mask,
        {
            "top_y": float(top),
            "bottom_y": float(bottom),
            "half_width": float(half_mid),
            "colour_index": float(np.argmax((colour == _MASK_COLOURS).all(1))),
        },
    )


# --------------------------------------------------------------------------- F2 sunglasses


def sunglasses(
    img: U8, lm: Landmarks, level: Level, rng: np.random.Generator
) -> tuple[U8, BoolMask, dict[str, float]]:
    h, w = img.shape[:2]
    d = lm.eye_distance
    rx = d * {"low": 0.30, "medium": 0.38, "high": 0.48}[level] * rng.uniform(0.92, 1.08)
    ry = rx * rng.uniform(0.65, 0.85)
    tilt = float(np.arctan2(*(lm.right_eye - lm.left_eye)[::-1]))
    dy = rng.uniform(-0.05, 0.05) * d
    lenses = ellipse_mask(h, w, lm.left_eye[0], lm.left_eye[1] + dy, rx, ry, tilt) | ellipse_mask(
        h, w, lm.right_eye[0], lm.right_eye[1] + dy, rx, ry, tilt
    )
    ec = lm.eye_center
    bridge = (
        np.abs(np.arange(h)[:, None] + 0.5 - (ec[1] + dy - 0.2 * ry)) < max(0.8, 0.012 * h)
    ) & (np.abs(np.arange(w)[None, :] + 0.5 - ec[0]) < 0.5 * d - 0.6 * rx + 1.0)
    mask = lenses | bridge
    tint = rng.uniform(15, 60)
    out = _composite(
        img,
        mask,
        np.array([tint, tint, tint * rng.uniform(0.9, 1.2)]),
        rng,
        texture=2.0,
        alpha=float(rng.uniform(0.88, 1.0)),
    )
    return (
        out,
        mask,
        {"lens_rx": float(rx), "lens_ry": float(ry), "tint": float(tint), "dy": float(dy)},
    )


# --------------------------------------------------------------------------- F3 hands / objects

_SKIN = np.array(
    [[224, 172, 140], [198, 134, 96], [141, 85, 54], [255, 220, 190], [110, 70, 45]], float
)
_OBJECTS = np.array(
    [[30, 30, 35], [200, 40, 40], [40, 90, 200], [220, 220, 210], [90, 140, 60]], float
)


def hand_or_object(
    img: U8, lm: Landmarks, level: Level, rng: np.random.Generator
) -> tuple[U8, BoolMask, dict[str, float]]:
    h, w = img.shape[:2]
    d = lm.eye_distance
    r0 = d * {"low": 0.40, "medium": 0.62, "high": 0.85}[level] * rng.uniform(0.9, 1.1)
    face = face_region(lm, h, w)
    ys, xs = np.nonzero(face)
    k = int(rng.integers(len(ys)))
    cx, cy = float(xs[k]) + 0.5, float(ys[k]) + 0.5
    n = 24
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    harmonics = sum(
        rng.uniform(0, 0.12) * np.cos(m * theta + rng.uniform(0, 2 * np.pi)) for m in (2, 3, 5)
    )
    radius = r0 * (1 + harmonics)
    pts = np.stack([cx + radius * np.cos(theta), cy + radius * np.sin(theta)], axis=1)
    mask = polygon_mask(h, w, pts)
    is_hand = rng.random() < 0.6
    if is_hand:  # finger-like capsules from the blob towards a random direction
        base = rng.uniform(0, 2 * np.pi)
        for f in range(int(rng.integers(3, 5))):
            ang = base + (f - 1.5) * 0.28
            length = r0 * rng.uniform(0.9, 1.4)
            for t in np.linspace(0.6, 1.0 + length / r0, 8):
                mask |= ellipse_mask(
                    h, w, cx + t * r0 * np.cos(ang), cy + t * r0 * np.sin(ang), 0.16 * r0, 0.16 * r0
                )
        colour = _SKIN[int(rng.integers(len(_SKIN)))]
    else:
        colour = _OBJECTS[int(rng.integers(len(_OBJECTS)))]
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 0.8 + 0.2 * np.clip(1 - np.hypot(xx - cx, yy - cy) / (2 * r0), 0, 1)
    out = _composite(img, mask, colour, rng, texture=4.0, shade=shade)
    return out, mask, {"cx": cx, "cy": cy, "radius": float(r0), "is_hand": float(is_hand)}


# --------------------------------------------------------------------------- F4 random patches

_PATCH_COVER = {"low": (0.10, 0.22), "medium": (0.25, 0.42), "high": (0.45, 0.65)}


def random_patches(
    img: U8, lm: Landmarks, level: Level, rng: np.random.Generator
) -> tuple[U8, BoolMask, dict[str, float]]:
    h, w = img.shape[:2]
    face = face_region(lm, h, w)
    lo, hi = _PATCH_COVER[level]
    target = rng.uniform(lo, hi)
    mask = np.zeros((h, w), dtype=bool)
    out = img.copy()
    n_patches = 0
    for _ in range(12):
        if mask[face].mean() >= target:
            break
        ys, xs = np.nonzero(face)
        k = int(rng.integers(len(ys)))
        cx, cy = float(xs[k]), float(ys[k])
        size = np.sqrt(target * face.sum() / rng.uniform(1.0, 3.0))
        shape = int(rng.integers(3))
        if shape == 0:
            pw, ph = size * rng.uniform(0.6, 1.4), size * rng.uniform(0.6, 1.4)
            pts = np.array(
                [
                    (cx - pw / 2, cy - ph / 2),
                    (cx + pw / 2, cy - ph / 2),
                    (cx + pw / 2, cy + ph / 2),
                    (cx - pw / 2, cy + ph / 2),
                ]
            )
            m = polygon_mask(h, w, pts)
        elif shape == 1:
            m = ellipse_mask(
                h,
                w,
                cx,
                cy,
                size * rng.uniform(0.4, 0.8),
                size * rng.uniform(0.4, 0.8),
                rng.uniform(0, np.pi),
            )
        else:
            ang = rng.uniform(0, 2 * np.pi) + np.array([0, 2.1, 4.2])
            m = polygon_mask(h, w, np.stack([cx + size * np.cos(ang), cy + size * np.sin(ang)], 1))
        colour = rng.uniform(0, 255, 3)
        out = _composite(out, m, colour, rng, texture=float(rng.choice([0.0, 25.0])))
        mask |= m
        n_patches += 1
    return out, mask, {"target_cover": float(target), "n_patches": float(n_patches)}


# --------------------------------------------------------------------------- F5 extreme lighting


def extreme_lighting(
    img: U8, lm: Landmarks, level: Level, rng: np.random.Generator
) -> tuple[U8, BoolMask, dict[str, float]]:
    """Photometric challenge. ``mask`` = facial pixels whose information is destroyed
    (crushed <= 12, clipped >= 250 where the source was not, or darkened to <= 35 % of the
    source luminance). This is NOT a physical occlusion and is reported separately."""
    h, w = img.shape[:2]
    src = img.astype(np.float64)
    mode = int(rng.integers(3))  # 0 shadow, 1 under-exposure, 2 over-exposure
    if mode == 0:
        factor = {"low": 0.55, "medium": 0.35, "high": 0.15}[level] * rng.uniform(0.9, 1.1)
        ang = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        proj = (xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)
        off = rng.uniform(-0.15, 0.15) * w
        soft = max(1.0, rng.uniform(0.01, 0.05) * w)
        gain = factor + (1 - factor) * 1 / (1 + np.exp((proj - off) / soft))
        out = src * gain[..., None]
        params = {"mode": 0.0, "factor": float(factor), "angle": float(ang), "offset": float(off)}
    elif mode == 1:
        g = {"low": 0.35, "medium": 0.22, "high": 0.12}[level] * rng.uniform(0.9, 1.1)
        out = 255.0 * (src / 255.0) ** 1.3 * g
        params = {"mode": 1.0, "gain": float(g)}
    else:
        g = {"low": 1.8, "medium": 2.6, "high": 3.6}[level] * rng.uniform(0.9, 1.1)
        out = src * g
        params = {"mode": 2.0, "gain": float(g)}
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    lum_s, lum_o = src.mean(axis=2), out_u8.astype(np.float64).mean(axis=2)
    crushed = out_u8.max(axis=2) <= 12
    clipped = (out_u8.min(axis=2) >= 250) & (src.min(axis=2) < 250)
    darkened = lum_o <= 0.35 * np.maximum(lum_s, 1.0)
    return out_u8, crushed | clipped | darkened, params


GENERATORS: dict[str, Generator] = {
    "F1_mask": surgical_mask,
    "F2_sunglasses": sunglasses,
    "F3_hand_object": hand_or_object,
    "F4_random_patch": random_patches,
    "F5_extreme_lighting": extreme_lighting,
}
PHYSICAL = {
    "F1_mask": True,
    "F2_sunglasses": True,
    "F3_hand_object": True,
    "F4_random_patch": True,
    "F5_extreme_lighting": False,
}
