"""Atomic IO, run identity, embedding SHA256, ID-set consistency checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..utils import atomic_write_json, ensure_dir, sha256_file


def make_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{os.getpid()}"


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.parent / f".{path.stem}.{os.getpid()}.tmp"
    try:
        np.save(tmp, arr)  # writes tmp.npy
        saved = Path(str(tmp) + ".npy")
        os.replace(saved, path)
    finally:
        for p in (tmp, Path(str(tmp) + ".npy")):
            if p.exists() and p.resolve() != path.resolve():
                try:
                    p.unlink()
                except OSError:
                    pass


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp.parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def stamp_run_id(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    out = df.copy()
    out["run_id"] = run_id
    return out


def assert_same_id_set(name_a: str, ids_a: Iterable[str], name_b: str, ids_b: Iterable[str]) -> None:
    a = set(map(str, ids_a))
    b = set(map(str, ids_b))
    if a != b:
        only_a = sorted(a - b)[:5]
        only_b = sorted(b - a)[:5]
        raise RuntimeError(
            f"ID set mismatch: {name_a}(n={len(a)}) vs {name_b}(n={len(b)}); "
            f"only_in_{name_a}={only_a} only_in_{name_b}={only_b}"
        )


def assert_same_n(name_a: str, n_a: int, name_b: str, n_b: int) -> None:
    if int(n_a) != int(n_b):
        raise RuntimeError(f"Count mismatch: {name_a}={n_a} vs {name_b}={n_b}")


def write_run_meta(cfg: dict[str, Any], **extra: Any) -> Path:
    meta = {
        "run_id": cfg["run_id"],
        "method_name": cfg.get("method_name"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pooling": cfg.get("pooling"),
        "use_local_patches": cfg.get("use_local_patches", False),
        "normalization": cfg.get("normalization"),
        **extra,
    }
    path = Path(cfg["paths"]["metadata_dir"]) / "run_meta.json"
    atomic_write_json(path, meta)
    return path


def load_aligned_embeddings(cfg: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Load embeddings + index; fail if n/IDs disagree; return emb_sha256."""
    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    meta_dir = Path(cfg["paths"]["metadata_dir"])
    name = (cfg.get("extract") or {}).get("save_name", "deepseek_ocr2_mean_l2.npy")
    emb_path = emb_dir / name
    idx_path = meta_dir / "embedding_index.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)

    X = np.load(emb_path)
    idx = pd.read_parquet(idx_path)
    idx = idx[idx["status"] == "ok"].copy()
    if "embedding_row" not in idx.columns:
        raise RuntimeError("embedding_index missing embedding_row")
    idx = idx.sort_values("embedding_row").reset_index(drop=True)
    rows = idx["embedding_row"].to_numpy(dtype=np.int64)
    assert_same_n("embedding_index_rows", len(idx), "embeddings.npy_rows", int(X.shape[0]))
    if not np.array_equal(rows, np.arange(len(idx))):
        # allow non-contiguous mapping only if max < n
        if rows.min() < 0 or rows.max() >= X.shape[0]:
            raise RuntimeError("embedding_row out of bounds")
        X = X[rows]
        idx = idx.reset_index(drop=True)
        idx["embedding_row"] = np.arange(len(idx))
    else:
        X = X.astype(np.float32)

    if "run_id" in idx.columns:
        rid = str(idx["run_id"].iloc[0])
        if rid != str(cfg.get("run_id")):
            raise RuntimeError(f"run_id mismatch: index={rid} cfg={cfg.get('run_id')}")

    emb_sha = file_sha256(emb_path)
    cfg["embedding_sha256"] = emb_sha
    cfg["embedding_path"] = str(emb_path)
    cfg["n_embeddings"] = int(X.shape[0])
    return X.astype(np.float32), idx, emb_sha
