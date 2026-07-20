"""Feature storage: parquet indexes + float32 memory-mapped embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import atomic_write_json, ensure_dir


class EmbeddingStore:
    """Append-friendly float32 mmap store with parquet row index."""

    def __init__(
        self,
        mmap_path: str | Path,
        index_path: str | Path,
        dim: int,
        meta_path: str | Path | None = None,
    ):
        self.mmap_path = Path(mmap_path)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path) if meta_path else self.mmap_path.with_suffix(".meta.json")
        self.dim = int(dim)
        ensure_dir(self.mmap_path.parent)
        ensure_dir(self.index_path.parent)
        self._index = self._load_index()
        self._n = len(self._index)
        self._fp = None
        self._arr = None
        self._open_mmap()

    def _load_index(self) -> pd.DataFrame:
        if self.index_path.exists():
            return pd.read_parquet(self.index_path)
        return pd.DataFrame(
            columns=[
                "image_id",
                "row_index",
                "embedding_dim",
                "selected_layer",
                "token_count",
                "embedding_norm_before_normalization",
            ]
        )

    def _open_mmap(self) -> None:
        if self.mmap_path.exists() and self.mmap_path.stat().st_size > 0:
            nbytes = self.mmap_path.stat().st_size
            n = nbytes // (4 * self.dim)
            self._n = max(self._n, n)
            if self._n > 0:
                self._arr = np.memmap(
                    self.mmap_path, dtype=np.float32, mode="r+", shape=(self._n, self.dim)
                )
            else:
                self._arr = None
        else:
            # Create empty file with 0 rows; first write will extend
            self.mmap_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.mmap_path.exists():
                self.mmap_path.touch()
            self._arr = None
            self._n = 0
        atomic_write_json(
            self.meta_path,
            {"dim": self.dim, "n_rows": int(len(self._index)), "dtype": "float32"},
        )

    @property
    def done_ids(self) -> set[str]:
        if self._index.empty:
            return set()
        return set(self._index["image_id"].astype(str).tolist())

    def append_many(self, rows: list[dict[str, Any]], vectors: np.ndarray) -> None:
        if len(rows) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        assert len(rows) == vectors.shape[0]

        start = len(self._index)
        end = start + len(rows)
        # Extend file
        new_size = end * self.dim * 4
        with open(self.mmap_path, "r+b" if self.mmap_path.stat().st_size else "w+b") as f:
            f.truncate(new_size)
        arr = np.memmap(self.mmap_path, dtype=np.float32, mode="r+", shape=(end, self.dim))
        arr[start:end] = vectors
        arr.flush()
        self._arr = arr
        self._n = end

        for i, r in enumerate(rows):
            r = dict(r)
            r["row_index"] = start + i
            r["embedding_dim"] = self.dim
            new_df = pd.DataFrame([r])
            if self._index.empty:
                self._index = new_df
            else:
                self._index = pd.concat([self._index, new_df], ignore_index=True)

        self.flush_index()

    def flush_index(self) -> None:
        tmp = self.index_path.with_suffix(".parquet.tmp")
        self._index.to_parquet(tmp, index=False)
        tmp.replace(self.index_path)
        atomic_write_json(
            self.meta_path,
            {"dim": self.dim, "n_rows": int(len(self._index)), "dtype": "float32"},
        )

    def load_matrix(self, ids: Iterable[str] | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        if self._index.empty or not self.mmap_path.exists() or self.mmap_path.stat().st_size == 0:
            return np.zeros((0, self.dim), dtype=np.float32), self._index.copy()
        n = len(self._index)
        arr = np.memmap(self.mmap_path, dtype=np.float32, mode="r", shape=(n, self.dim))
        df = self._index.copy()
        if ids is not None:
            id_set = set(ids)
            mask = df["image_id"].astype(str).isin(id_set)
            df = df.loc[mask].reset_index(drop=True)
            mat = np.asarray(arr[df["row_index"].to_numpy()], dtype=np.float32)
            return mat, df
        return np.asarray(arr[:n], dtype=np.float32), df


def atomic_replace_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def merge_status_parquet(
    path: str | Path,
    new_rows: pd.DataFrame,
    key: str = "image_id",
) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=[key], keep="last")
    else:
        combined = new_rows
    atomic_replace_parquet(combined, path)
    return combined
