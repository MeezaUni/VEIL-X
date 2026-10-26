from __future__ import annotations

import struct

import numpy as np
import pytest

from veilx.data.png import PNG_SIGNATURE, check_png_bytes, encode_png

from .helpers import noise_png


def test_roundtrip_rgb_and_grey() -> None:
    ok = check_png_bytes(noise_png(1, size=7))
    assert ok.ok and (ok.width, ok.height, ok.color_type) == (7, 7, 2)
    grey = check_png_bytes(encode_png(np.zeros((3, 5), dtype=np.uint8)))
    assert grey.ok and (grey.width, grey.height, grey.color_type) == (5, 3, 0)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda b: b"GIF89a" + b[6:], "signature"),
        (lambda b: b[: len(b) // 2], "truncated"),
        (lambda b: b[:-12], "IEND"),
        (lambda b: b[:30] + bytes([b[30] ^ 0xFF]) + b[31:], "CRC"),
        (lambda b: b"", "signature"),
    ],
)
def test_corruptions_detected(mutate, fragment: str) -> None:  # type: ignore[no-untyped-def]
    res = check_png_bytes(mutate(noise_png(2)))
    assert not res.ok
    assert fragment.lower() in (res.error or "").lower()


def test_bad_zlib_stream_detected() -> None:
    import zlib

    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0)

    def chunk(t: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + t + body + struct.pack(">I", zlib.crc32(t + body))

    data = PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"notzlib") + chunk(b"IEND", b"")
    assert not check_png_bytes(data).ok
    # well-formed zlib but wrong decoded size
    data = (
        PNG_SIGNATURE
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )
    res = check_png_bytes(data)
    assert not res.ok and "size" in (res.error or "")


def test_encode_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2), dtype=np.float32))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2, 4), dtype=np.uint8))
