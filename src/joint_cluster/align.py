"""Align embeddings with GT pages via image path stem (= page_id)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import sha256_file
from .gt_features import FEATURE_NAMES, compute_all_page_gt


def load_embeddings(run_dir: str | Path, embedding_name: str) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    run = Path(run_dir)
    emb_path = run / "embeddings" / embedding_name
    if not emb_path.exists():
        # fallback: shards already merged elsewhere
        raise FileNotFoundError(emb_path)
    X = np.load(emb_path).astype(np.float32)
    idx = pd.read_parquet(run / "metadata" / "embedding_index.parquet")
    man = pd.read_parquet(run / "metadata" / "manifest.parquet")
    idx = idx[idx["status"] == "ok"].copy()
    idx = idx.sort_values("embedding_row").reset_index(drop=True)
    if len(idx) != X.shape[0]:
        raise RuntimeError(f"embedding rows {X.shape[0]} != index ok {len(idx)}")
    return X, idx, man


def build_aligned_table(
    cfg: dict[str, Any],
    *,
    gt_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """
    page_id = filename stem (matches GT JSON image_id).
    Join embeddings via manifest.image_path stem.
    """
    X, idx, man = load_embeddings(cfg["paths"]["embedding_run_dir"], cfg["paths"]["embedding_name"])
    man = man.copy()
    man["page_id"] = man["image_path"].map(lambda p: Path(str(p)).stem)
    man["embedding_image_id"] = man["image_id"].astype(str)
    idx = idx.copy()
    idx["embedding_image_id"] = idx["image_id"].astype(str)

    # row alignment: index embedding_row -> X
    emb_meta = idx.merge(
        man[["embedding_image_id", "page_id", "image_path", "relative_path", "file_hash", "width", "height"]],
        on="embedding_image_id",
        how="left",
        validate="one_to_one",
    )
    if emb_meta["page_id"].isna().any():
        raise RuntimeError("manifest missing for some embeddings")

    if gt_df is None:
        gt_cfg = cfg.get("gt") or {}
        gt_df = compute_all_page_gt(
            cfg["paths"]["images_dir"],
            exclude_structure_tokens=gt_cfg.get("exclude_structure_tokens", ["{", "}", "$"]),
            exclude_layout_from_plain=bool(gt_cfg.get("exclude_layout_from_plain", True)),
        )

    # duplicate image mark (same file_hash appears >1)
    hash_counts = emb_meta["file_hash"].value_counts()
    emb_meta["is_duplicate_image"] = emb_meta["file_hash"].map(lambda h: bool(hash_counts.get(h, 0) > 1))

    merged = emb_meta.merge(gt_df, on="page_id", how="left", indicator=True)
    missing_gt = merged["_merge"] == "left_only"
    merged["gt_join_status"] = np.where(missing_gt, "missing_gt", "ok")
    merged.loc[missing_gt, "ast_parse_status"] = "missing_gt"
    for c in FEATURE_NAMES:
        if c not in merged.columns:
            merged[c] = np.nan

    # embedding QA
    dim = int(cfg["image"]["expected_dim"])
    if X.shape[1] != dim:
        raise RuntimeError(f"embedding dim {X.shape[1]} != {dim}")
    finite = np.isfinite(X).all(axis=1)
    norms = np.linalg.norm(X, axis=1)
    zero = norms < 1e-12
    # exact duplicates among embeddings
    # round for speed
    rounded = np.round(X, 6)
    _, inv, counts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    dup_emb = counts[inv] > 1

    merged["embedding_status"] = "ok"
    merged.loc[~finite, "embedding_status"] = "nan_inf"
    merged.loc[finite & zero, "embedding_status"] = "zero"
    merged["embedding_norm"] = norms
    merged["is_duplicate_embedding"] = dup_emb
    merged["embedding_row"] = np.arange(len(merged), dtype=np.int64)

    # usable for clustering: emb ok + gt parse ok (empty pages allowed as 0-structure)
    usable = (
        (merged["embedding_status"] == "ok")
        & (merged["ast_parse_status"].isin(["ok", "empty"]))
        & merged[list(FEATURE_NAMES)].notna().all(axis=1)
    )
    # empty pages have max_ast_depth=0 etc — ok
    # fail / missing_gt excluded
    merged["cluster_fit"] = usable

    qa = {
        "n_embeddings": int(X.shape[0]),
        "n_gt_pages": int(len(gt_df)),
        "n_aligned": int(len(merged)),
        "n_missing_gt": int(missing_gt.sum()),
        "n_parse_fail": int((merged["ast_parse_status"] == "fail").sum()),
        "n_empty_gt": int((merged["ast_parse_status"] == "empty").sum()),
        "n_emb_nan_inf": int((~finite).sum()),
        "n_emb_zero": int(zero.sum()),
        "n_duplicate_embedding_rows": int(dup_emb.sum()),
        "n_duplicate_image_rows": int(merged["is_duplicate_image"].sum()),
        "n_cluster_fit": int(usable.sum()),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_max": float(norms.max()),
        "embedding_sha256": sha256_file(Path(cfg["paths"]["embedding_run_dir"]) / "embeddings" / cfg["paths"]["embedding_name"]),
    }

    # consistency checks on fit set
    fit = merged.loc[usable]
    bad_nodes = ((fit["ast_tree_count"] == 0) & (fit["total_ast_node_count"] > 0)).sum()
    bad_depth = ((fit["ast_tree_count"] == 0) & (fit["max_ast_depth"].fillna(0) > 0)).sum()
    qa["n_inconsistent_nodes_without_trees"] = int(bad_nodes)
    qa["n_inconsistent_depth_without_trees"] = int(bad_depth)
    if bad_nodes or bad_depth:
        raise RuntimeError(
            f"GT inconsistency: nodes_wo_trees={bad_nodes} depth_wo_trees={bad_depth}"
        )

    X_fit = X[merged.loc[usable, "embedding_row"].to_numpy()]
    return merged, X_fit, qa
