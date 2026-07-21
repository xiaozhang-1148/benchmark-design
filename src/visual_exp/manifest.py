"""Build experiment manifest (audit fields only; not used for clustering)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ..feature_store import atomic_replace_parquet
from ..utils import ensure_dir, iter_images, make_image_id, safe_image_meta, sha256_file


def _one(path: Path, root: Path) -> dict[str, Any]:
    abs_path = str(path.resolve())
    meta = safe_image_meta(path)
    w, h = meta["width"], meta["height"]
    row: dict[str, Any] = {
        "image_path": abs_path,
        "relative_path": str(path.relative_to(root)),
        "file_hash": None,
        "width": w,
        "height": h,
        "aspect_ratio": (float(w) / float(h)) if w and h and h > 0 else None,
        "file_size": meta["file_size"],
        "token_count": None,
        "n_local_patches": None,
        "embedding_row": None,
        "norm_before_l2": None,
        "status": "pending",
        "error_message": meta["error_message"],
    }
    if meta["error_message"]:
        row["status"] = "corrupt"
        row["image_id"] = make_image_id(abs_path)
        return row
    try:
        digest = sha256_file(path)
        row["file_hash"] = digest
        row["image_id"] = make_image_id(abs_path, digest)
    except Exception as e:  # noqa: BLE001
        row["status"] = "corrupt"
        row["error_message"] = f"{type(e).__name__}: {e}"
        row["image_id"] = make_image_id(abs_path)
    return row


def build_manifest(cfg: dict[str, Any], *, limit: int | None = None) -> pd.DataFrame:
    root = Path(cfg["data"]["input_dir"])
    exts = set(cfg["data"].get("image_extensions") or [".jpg", ".jpeg", ".png"])
    recursive = bool(cfg["data"].get("recursive", True))
    paths = list(iter_images(root, exts, recursive=recursive))
    if limit is not None:
        paths = paths[: int(limit)]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_one, p, root) for p in paths]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="manifest"):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    # stable order
    df = df.sort_values("image_id").reset_index(drop=True)
    out = Path(cfg["paths"]["metadata_dir"]) / "manifest.parquet"
    ensure_dir(out.parent)
    atomic_replace_parquet(df, out)
    print(f"[manifest] n={len(df)} corrupt={int((df.status=='corrupt').sum())} -> {out}")
    return df
