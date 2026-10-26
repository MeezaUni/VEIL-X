"""Image decoding and content fingerprints (no face models involved).

* ``pixel_sha256`` — SHA-256 over (height, width, channels) + decoded RGB bytes: identical
  pictures stored with different encodings or metadata get the same hash (exact duplicates).
* ``phash64`` — 64-bit perceptual hash: 32x32 greyscale (Pillow BILINEAR), 2-D DCT-II, the 8x8
  low-frequency block thresholded at its median (DC excluded from the median). Used only to
  *nominate* near-duplicate candidates.
* ``thumb_vector`` — zero-mean, unit-norm 16x16 greyscale vector; the dot product of two such
  vectors is a normalised cross-correlation (NCC) on a coarse grid, used as a second,
  independent near-duplicate signal.

Neither signal is a face-identity signal: two different photographs of the same (synthetic)
person are *not* duplicates and the validation procedure explicitly measures that they are
rarely flagged (``veilx.data.dedup``).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

U8 = NDArray[np.uint8]


def _dct_matrix(n: int) -> NDArray[np.float64]:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0, :] = np.sqrt(1.0 / n)
    return np.asarray(m, dtype=np.float64)


_D32 = _dct_matrix(32)


def decode_rgb(data: bytes) -> U8:
    """Fully decode an image to RGB uint8 (raises on truncated/corrupt data)."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def load_rgb(path: Path) -> U8:
    return decode_rgb(path.read_bytes())


def encode_png_rgb(arr: U8) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def pixel_sha256(arr: U8) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())
    return h.hexdigest()


def _grey(arr: U8, size: int) -> NDArray[np.float64]:
    im = (
        Image.fromarray(arr, mode="RGB")
        .convert("L")
        .resize((size, size), Image.Resampling.BILINEAR)
    )
    return np.asarray(im, dtype=np.float64)


def phash64(arr: U8) -> int:
    g = _grey(arr, 32)
    coeffs = _D32 @ g @ _D32.T
    block = coeffs[:8, :8].ravel()
    med = np.median(block[1:])
    bits = (block > med).astype(np.uint8)
    return int("".join(map(str, bits.tolist())), 2)


def thumb_vector(arr: U8) -> NDArray[np.float32]:
    v = _grey(arr, 16).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class Fingerprint:
    ok: bool
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    pixel_sha256: str | None = None
    phash: int | None = None
    error: str | None = None


def fingerprint(data: bytes) -> tuple[Fingerprint, NDArray[np.float32] | None]:
    try:
        with Image.open(io.BytesIO(data)) as im:
            mode = im.mode
        arr = decode_rgb(data)
    except Exception as exc:  # Pillow raises many exception types for corrupt data
        return Fingerprint(False, error=f"{type(exc).__name__}: {exc}"[:200]), None
    h, w = arr.shape[:2]
    return (
        Fingerprint(True, w, h, mode, pixel_sha256(arr), phash64(arr)),
        thumb_vector(arr),
    )
