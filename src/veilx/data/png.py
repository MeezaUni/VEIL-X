"""Dependency-free PNG integrity checking (and a tiny encoder for synthetic test fixtures).

``check_png`` verifies, without decoding pixels into memory-heavy arrays:

* the 8-byte PNG signature;
* every chunk's length and CRC-32;
* a well-formed IHDR as the first chunk (dimensions, bit depth, colour type, interlace);
* the presence of IDAT data and a terminating IEND chunk (truncation detection);
* that the concatenated IDAT zlib stream decompresses completely and, for non-interlaced
  images, to exactly ``height * (1 + row_bytes)`` bytes.

``encode_png`` writes small RGB/greyscale images; it exists so that tests can build synthetic,
non-biometric fixture directories (random noise, not faces).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_VALID_DEPTHS = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}


@dataclass(frozen=True)
class PngCheck:
    ok: bool
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    color_type: int | None = None
    error: str | None = None


def check_png_bytes(data: bytes) -> PngCheck:
    if not data.startswith(PNG_SIGNATURE):
        return PngCheck(False, error="bad signature")
    pos = len(PNG_SIGNATURE)
    ihdr: tuple[int, int, int, int, int] | None = None
    idat = bytearray()
    seen_iend = False
    first = True
    while pos < len(data):
        if pos + 8 > len(data):
            return PngCheck(False, error="truncated chunk header")
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        end = pos + 8 + length + 4
        if end > len(data):
            return PngCheck(False, error=f"truncated chunk {ctype!r}")
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : end])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            return PngCheck(False, error=f"CRC mismatch in {ctype!r}")
        if first:
            if ctype != b"IHDR" or length != 13:
                return PngCheck(False, error="first chunk is not a valid IHDR")
            w, h, depth, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", body)
            if w == 0 or h == 0 or color not in _CHANNELS or depth not in _VALID_DEPTHS[color]:
                return PngCheck(False, error="invalid IHDR values")
            ihdr = (w, h, depth, color, interlace)
            first = False
        elif ctype == b"IDAT":
            idat.extend(body)
        elif ctype == b"IEND":
            seen_iend = True
            break
        pos = end
    if ihdr is None:
        return PngCheck(False, error="missing IHDR")
    w, h, depth, color, interlace = ihdr
    if not idat:
        return PngCheck(False, w, h, depth, color, error="missing IDAT")
    if not seen_iend:
        return PngCheck(False, w, h, depth, color, error="missing IEND (truncated file)")
    try:
        dobj = zlib.decompressobj()
        raw = dobj.decompress(bytes(idat))
        if not dobj.eof:
            return PngCheck(False, w, h, depth, color, error="incomplete zlib stream")
    except zlib.error as exc:
        return PngCheck(False, w, h, depth, color, error=f"zlib error: {exc}")
    if interlace == 0:
        row_bytes = (w * _CHANNELS[color] * depth + 7) // 8
        if len(raw) != h * (1 + row_bytes):
            return PngCheck(False, w, h, depth, color, error="decoded size mismatch")
    return PngCheck(True, w, h, depth, color)


def check_png(path: Path) -> PngCheck:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return PngCheck(False, error=f"unreadable: {exc.__class__.__name__}")
    return check_png_bytes(data)


def _chunk(ctype: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(ctype + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", crc)


def encode_png(pixels: NDArray[np.uint8]) -> bytes:
    """Encode an (H, W) greyscale or (H, W, 3) RGB uint8 array as PNG (filter type 0)."""
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise ValueError("expected uint8 (H, W) or (H, W, 3)")
    if pixels.ndim == 3 and pixels.shape[2] != 3:
        raise ValueError("only RGB is supported for 3-D input")
    h, w = pixels.shape[:2]
    color = 0 if pixels.ndim == 2 else 2
    rows = pixels.reshape(h, -1)
    raw = b"".join(b"\x00" + rows[i].tobytes() for i in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
