"""Load vision/text feature matrices via their sample indexes (never by filesystem order)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FeatureStore:
    kind: str
    matrix_path: str
    index_path: str
    matrix: np.ndarray  # float32 mmap or array [N, D]
    # basename (image filename) -> row_index
    basename_to_row: dict[str, int]
    dim: int
    n: int
    selection_reason: str

    def get_by_basename(self, basename: str) -> np.ndarray:
        if basename not in self.basename_to_row:
            raise KeyError(f"{self.kind}: no feature for basename={basename!r}")
        row = self.basename_to_row[basename]
        vec = np.asarray(self.matrix[row], dtype=np.float32)
        if vec.shape != (self.dim,):
            raise RuntimeError(f"{self.kind} bad shape {vec.shape} for {basename}")
        return vec.copy()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_vision_store(vision_root: str | Path) -> FeatureStore:
    root = Path(vision_root)
    # Prefer official concat 1792-d (Projector-before global+local means).
    candidates = [
        (
            root / "deepseek_ocr2_global_local_concat_fp32.npy",
            root / "sample_index.jsonl",
            1792,
            "official concat [N,1792] + sample_index.jsonl (Projector-before, no L2)",
        ),
    ]
    matrix_path = index_path = None
    dim = None
    reason = None
    for mp, ip, d, why in candidates:
        if mp.is_file() and ip.is_file():
            matrix_path, index_path, dim, reason = mp, ip, d, why
            break
    if matrix_path is None:
        raise FileNotFoundError(
            f"vision concat feature+index not found under {root}; "
            f"refusing to guess among other files"
        )

    mat = np.load(matrix_path, mmap_mode="r")
    if mat.ndim != 2 or mat.shape[1] != dim:
        raise RuntimeError(f"vision matrix expected [N,{dim}], got {mat.shape}")
    if mat.dtype != np.float32:
        # will cast on read; warn via reason
        reason += f"; on-disk dtype={mat.dtype}"

    index = _load_jsonl(index_path)
    if len(index) != mat.shape[0]:
        raise RuntimeError(
            f"vision index length {len(index)} != matrix rows {mat.shape[0]}"
        )

    basename_to_row: dict[str, int] = {}
    for row in index:
        # Prefer basename from image_path; fall back to sample_id (also a filename here).
        if row.get("image_path"):
            basename = Path(str(row["image_path"])).name
        else:
            basename = str(row["sample_id"])
        ri = int(row["row_index"])
        if basename in basename_to_row and basename_to_row[basename] != ri:
            raise RuntimeError(f"duplicate vision basename mapping: {basename}")
        basename_to_row[basename] = ri
        if not (0 <= ri < mat.shape[0]):
            raise RuntimeError(f"vision row_index OOB: {ri}")

    return FeatureStore(
        kind="vision",
        matrix_path=str(matrix_path),
        index_path=str(index_path),
        matrix=mat,
        basename_to_row=basename_to_row,
        dim=int(dim),
        n=int(mat.shape[0]),
        selection_reason=reason,
    )


def load_text_store(text_root: str | Path) -> FeatureStore:
    root = Path(text_root)
    matrix_path = root / "qwen3_embedding_0.6b_last_token_raw_fp32.npy"
    index_path = root / "text_sample_index.jsonl"
    if not matrix_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"text feature+index not found: {matrix_path} / {index_path}"
        )
    mat = np.load(matrix_path, mmap_mode="r")
    dim = 1024
    if mat.ndim != 2 or mat.shape[1] != dim:
        raise RuntimeError(f"text matrix expected [N,{dim}], got {mat.shape}")

    index = _load_jsonl(index_path)
    if len(index) != mat.shape[0]:
        raise RuntimeError(f"text index length {len(index)} != matrix rows {mat.shape[0]}")

    basename_to_row: dict[str, int] = {}
    for row in index:
        # Prefer explicit image_name; else strip .json from sample_id
        if row.get("image_name"):
            basename = str(row["image_name"])
        else:
            sid = str(row["sample_id"])
            basename = sid[:-5] if sid.endswith(".json") else sid
        ri = int(row["row_index"])
        if basename in basename_to_row and basename_to_row[basename] != ri:
            raise RuntimeError(f"duplicate text basename mapping: {basename}")
        basename_to_row[basename] = ri

    return FeatureStore(
        kind="text",
        matrix_path=str(matrix_path),
        index_path=str(index_path),
        matrix=mat,
        basename_to_row=basename_to_row,
        dim=dim,
        n=int(mat.shape[0]),
        selection_reason=(
            "qwen3_embedding_0.6b_last_token_raw_fp32.npy [N,1024] + text_sample_index.jsonl "
            "(last-token pooling, no L2)"
        ),
    )
