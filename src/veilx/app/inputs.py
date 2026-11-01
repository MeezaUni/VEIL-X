"""Input validation for the local research app: every rejection has an explicit reason code.

Images are decoded in memory only. Nothing is written to disk by this module, and EXIF
orientation is applied. Alpha, palette and greyscale images are converted to RGB, a lossless
interpretation that is recorded in ``notes``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from veilx.app.config import InputSection


class InputRejectedError(ValueError):
    """Raised with a stable ``code`` for user-facing reporting."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ValidatedImage:
    rgb: NDArray[np.uint8]
    source_format: str
    notes: list[str] = field(default_factory=list)


def load_image_bytes(data: bytes, limits: InputSection) -> ValidatedImage:
    if not data:
        raise InputRejectedError("empty_file", "the file is empty")
    if len(data) > limits.max_bytes:
        raise InputRejectedError("file_too_large", f"file larger than {limits.max_bytes} bytes")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in limits.formats:
                raise InputRejectedError(
                    "unsupported_format", f"format {fmt or 'unknown'} is not accepted"
                )
            w, h = probe.size
            if max(w, h) > limits.max_side:
                raise InputRejectedError(
                    "image_too_large", f"image side {max(w, h)} > {limits.max_side}"
                )
            probe.verify()  # structural check (truncated / corrupt files)
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            notes: list[str] = []
            oriented = ImageOps.exif_transpose(img)
            if oriented.size != img.size:
                notes.append("EXIF orientation applied")
            if oriented.mode != "RGB":
                notes.append(f"converted {oriented.mode} to RGB")
                oriented = oriented.convert("RGB")
            arr = np.asarray(oriented, dtype=np.uint8).copy()
    except InputRejectedError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InputRejectedError(
            "corrupt_or_unreadable", f"the file could not be decoded ({type(exc).__name__})"
        ) from exc
    return _check_array(arr, limits, fmt, notes)


def validate_array(arr: NDArray[np.generic], limits: InputSection) -> ValidatedImage:
    """For numpy inputs (e.g. from the UI component): dtype / shape / size checks."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise InputRejectedError("bad_shape", f"expected an RGB image, got shape {a.shape}")
    if a.shape[2] == 4:
        a = a[..., :3]
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.size and float(np.nanmax(a)) <= 1.0:
            a = np.clip(np.rint(a * 255.0), 0, 255)
        a = np.clip(a, 0, 255).astype(np.uint8)
    return _check_array(np.ascontiguousarray(a), limits, "ARRAY", [])


def _check_array(
    arr: NDArray[np.uint8], limits: InputSection, fmt: str, notes: list[str]
) -> ValidatedImage:
    h, w = arr.shape[:2]
    if min(h, w) < limits.min_side:
        raise InputRejectedError("image_too_small", f"image side {min(h, w)} < {limits.min_side}")
    if max(h, w) > limits.max_side:
        raise InputRejectedError("image_too_large", f"image side {max(h, w)} > {limits.max_side}")
    if arr.std() < 1.0:
        raise InputRejectedError("blank_image", "the image is (nearly) uniform")
    return ValidatedImage(arr, fmt, notes)
