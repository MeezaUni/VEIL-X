"""Atomic, verifiable embedding shards.

A shard is ``<name>.npz`` (``keys``, ``emb`` (N, D) float32 unit vectors, ``norm`` (N,) float32
pre-normalisation norms) plus ``<name>.meta.json``. The meta file is written last and records
the npz SHA-256, so its presence marks a complete shard; a shard whose meta does not match the
requested fingerprint (backbone weights + preprocessing + precision + job definition) is stale
and is recomputed - this is the cache-invalidation rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from veilx.data.archive import sha256_file
from veilx.errors import ManifestError


@dataclass(frozen=True)
class Shard:
    keys: NDArray[np.str_]
    emb: NDArray[np.float32]
    norm: NDArray[np.float32]
    meta: dict[str, Any]

    def index(self) -> dict[str, int]:
        return {str(k): i for i, k in enumerate(self.keys.tolist())}


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".npz") + ".meta.json")


def write_shard(
    path: Path,
    keys: list[str],
    emb: NDArray[np.float32],
    norm: NDArray[np.float32],
    meta: dict[str, Any],
) -> None:
    n = len(keys)
    if emb.ndim != 2 or emb.shape[0] != n or norm.shape != (n,):
        raise ValueError(f"shape mismatch: keys {n}, emb {emb.shape}, norm {norm.shape}")
    if len(set(keys)) != n:
        raise ManifestError("duplicate keys in shard")
    if not (np.isfinite(emb).all() and np.isfinite(norm).all()):
        raise FloatingPointError("non-finite values in shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    _meta_path(path).unlink(missing_ok=True)  # invalidate first: meta is the completion marker
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            keys=np.asarray(keys, dtype=np.str_),
            emb=emb.astype(np.float32),
            norm=norm.astype(np.float32),
        )
    tmp.replace(path)
    full = dict(meta) | {"n": n, "dim": int(emb.shape[1]), "npz_sha256": sha256_file(path)}
    mtmp = _meta_path(path).with_name(_meta_path(path).name + ".tmp")
    mtmp.write_text(json.dumps(full, indent=2, sort_keys=True), encoding="utf-8")
    mtmp.replace(_meta_path(path))


def read_meta(path: Path) -> dict[str, Any] | None:
    mp = _meta_path(path)
    if not (mp.exists() and path.exists()):
        return None
    meta: dict[str, Any] = json.loads(mp.read_text(encoding="utf-8"))
    return meta


def is_complete(path: Path, expect: dict[str, Any]) -> bool:
    """True when the shard exists and its meta matches every expected field."""
    meta = read_meta(path)
    return meta is not None and all(meta.get(k) == v for k, v in expect.items())


def read_shard(path: Path, *, verify: bool = True) -> Shard:
    meta = read_meta(path)
    if meta is None:
        raise ManifestError(f"incomplete shard {path.name}")
    if verify and sha256_file(path) != meta["npz_sha256"]:
        raise ManifestError(f"shard {path.name} does not match its recorded SHA-256")
    with np.load(path, allow_pickle=False) as z:
        keys, emb, norm = z["keys"], z["emb"], z["norm"]
    if emb.shape != (meta["n"], meta["dim"]) or len(keys) != meta["n"]:
        raise ManifestError(f"shard {path.name} shape differs from its meta")
    return Shard(keys, emb, norm, meta)


def write_array_shard(
    path: Path, keys: list[str], arrays: dict[str, NDArray[Any]], meta: dict[str, Any]
) -> None:
    """Atomic multi-array shard (same completion-marker rule as :func:`write_shard`). Every
    array's first dimension must equal ``len(keys)``; float arrays must be finite."""
    n = len(keys)
    if len(set(keys)) != n:
        raise ManifestError("duplicate keys in shard")
    for name, a in arrays.items():
        if a.shape[0] != n:
            raise ValueError(f"{name}: first dimension {a.shape[0]} != {n} keys")
        if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
            raise FloatingPointError(f"{name}: non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    _meta_path(path).unlink(missing_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, keys=np.asarray(keys, dtype=np.str_), **arrays)  # type: ignore[arg-type]
    tmp.replace(path)
    full = dict(meta) | {
        "n": n,
        "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in arrays.items()},
        "npz_sha256": sha256_file(path),
    }
    mtmp = _meta_path(path).with_name(_meta_path(path).name + ".tmp")
    mtmp.write_text(json.dumps(full, indent=2, sort_keys=True), encoding="utf-8")
    mtmp.replace(_meta_path(path))


def read_array_shard(
    path: Path, *, verify: bool = True
) -> tuple[NDArray[np.str_], dict[str, NDArray[Any]], dict[str, Any]]:
    meta = read_meta(path)
    if meta is None:
        raise ManifestError(f"incomplete shard {path.name}")
    if verify and sha256_file(path) != meta["npz_sha256"]:
        raise ManifestError(f"shard {path.name} does not match its recorded SHA-256")
    with np.load(path, allow_pickle=False) as z:
        keys = z["keys"]
        arrays = {k: z[k] for k in z.files if k != "keys"}
    if len(keys) != meta["n"]:
        raise ManifestError(f"shard {path.name} key count differs from its meta")
    return keys, arrays, meta
