"""Independent quantitative analysis for visual / layout / recognition features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from .config import get_seed, load_config
from .feature_store import EmbeddingStore, atomic_replace_parquet
from .utils import atomic_write_json, ensure_dir, l2_normalize

try:
    import faiss
except Exception:  # noqa: BLE001
    faiss = None


# PCA layout cols — no blank_ratio, no constant reading-order placeholders, no fallback
LAYOUT_FEATURE_COLS = [
    "content_area_ratio",
    "mean_center_x",
    "mean_center_y",
    "center_spread_x",
    "center_spread_y",
    "margin_top",
    "margin_bottom",
    "margin_left",
    "margin_right",
    "text_area_ratio",
    "formula_area_ratio",
    "figure_area_ratio",
    "table_area_ratio",
    "median_block_area",
    "iqr_block_area",
    "mean_block_aspect_ratio",
    "std_block_aspect_ratio",
    "max_block_area_ratio",
    "estimated_row_count",
    "estimated_col_count",
    "mean_horizontal_gap",
    "mean_vertical_gap",
    "left_align_score",
    "center_align_score",
    "two_column_score",
    "block_overlap_ratio",
    "reading_order_inversion_count",
    "reading_order_row_jump_count",
    "reading_order_mean_jump_distance",
    "block_count_log1p",
    "text_block_count_log1p",
    "formula_block_count_log1p",
    "figure_block_count_log1p",
    "table_block_count_log1p",
] + [f"occupancy_{i}_{j}" for i in range(4) for j in range(4)]

# Content-only recognition PCA cols (NO quality)
RECOG_FEATURE_COLS = [
    "chinese_ratio",
    "latin_ratio",
    "digit_ratio",
    "math_operator_ratio",
    "bracket_ratio",
    "punctuation_ratio",
    "whitespace_ratio",
    "formula_character_ratio",
    "formula_line_ratio",
    "non_empty_line_ratio",
    "mean_line_length",
    "math_structure_per_line",
    "equals_count",
    "frac_count",
    "sqrt_count",
    "subscript_count",
    "superscript_count",
    "vector_symbol_count",
    "angle_symbol_count",
    "latex_env_count",
    "formula_span_count",
    "output_character_count",
    "line_count",
]

RECOG_CONTENT_DIAG_COLS = [
    "chinese_ratio",
    "latin_ratio",
    "digit_ratio",
    "math_operator_ratio",
    "formula_character_ratio",
    "formula_line_ratio",
    "math_structure_per_line",
    "equals_count",
    "frac_count",
    "output_character_count",
    "line_count",
]

RECOG_QUALITY_DIAG_COLS = [
    "repetition_ratio",
    "output_token_count",
    "hit_max_tokens",
    "invalid_char_rate",
    "max_ngram_repeat_count",
]


def quality_report(X: np.ndarray, name: str) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X)
    norms = np.linalg.norm(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    var = np.nanvar(X, axis=0)
    return {
        "channel": name,
        "n_samples": int(X.shape[0]),
        "n_dim": int(X.shape[1]) if X.ndim == 2 else 0,
        "missing_cells": int(np.isnan(X).sum()),
        "nan_inf_cells": int((~finite).sum()),
        "constant_dims": int(np.sum(var < 1e-12)),
        "near_zero_var_dims": int(np.sum(var < 1e-6)),
        "norm_mean": float(np.mean(norms)) if len(norms) else None,
        "norm_std": float(np.std(norms)) if len(norms) else None,
        "norm_p50": float(np.median(norms)) if len(norms) else None,
    }


def fit_pca(X: np.ndarray, n_components: int, seed: int) -> dict[str, Any]:
    n_samples, n_dim = X.shape
    k = int(min(n_components, n_samples, n_dim))
    if k < 2:
        return {"error": "too few samples/dims for PCA"}
    if n_samples > 20000 and n_dim > 64:
        pca = IncrementalPCA(n_components=k, batch_size=min(1024, n_samples))
        for i in range(0, n_samples, 1024):
            pca.partial_fit(X[i : i + 1024])
        coords = pca.transform(X)[:, :2]
        evr = pca.explained_variance_ratio_
    else:
        pca = PCA(n_components=k, random_state=seed, svd_solver="randomized" if n_dim > 100 else "auto")
        coords = pca.fit_transform(X)[:, :2]
        evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    def dims_for(thr):
        idx = np.searchsorted(cum, thr)
        return int(min(idx + 1, len(cum)))

    return {
        "explained_variance_ratio": evr.tolist(),
        "cumulative_explained_variance": cum.tolist(),
        "dims_80": dims_for(0.80),
        "dims_90": dims_for(0.90),
        "dims_95": dims_for(0.95),
        "pca_coords": coords.astype(np.float32),
        "pca_model": pca,
        "n_components_fit": int(k),
        "n_samples": int(n_samples),
    }


def knn_stats(X: np.ndarray, metric: str, k: int = 15) -> dict[str, Any]:
    n = X.shape[0]
    k_eff = min(k + 1, n)
    if n < 2:
        return {"error": "too few samples"}
    if faiss is not None and metric in {"cosine", "inner_product"} and n >= 100:
        xb = X.astype(np.float32).copy()
        faiss.normalize_L2(xb)
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        D, I = index.search(xb, k_eff)
        dists = 1.0 - D
        nn_idx = I[:, 1:k_eff]
        nn_dist = dists[:, 1:k_eff]
    else:
        m = "cosine" if metric == "cosine" else "euclidean"
        nn = NearestNeighbors(n_neighbors=k_eff, metric=m, algorithm="auto")
        nn.fit(X)
        dists, idxs = nn.kneighbors(X)
        nn_idx = idxs[:, 1:]
        nn_dist = dists[:, 1:]
    k15 = nn_dist[:, min(k - 1, nn_dist.shape[1] - 1)]
    return {
        "nn_indices": nn_idx.astype(np.int32),
        "nn_distances": nn_dist.astype(np.float32),
        "kth_neighbor_distance": k15.astype(np.float32),
        "nearest_neighbor_index": nn_idx[:, 0].astype(np.int32),
        "nearest_neighbor_distance": nn_dist[:, 0].astype(np.float32),
    }


def robust_scale_explicit(
    df: pd.DataFrame, cols: list[str], out_scaler: Path
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    use_cols = [c for c in cols if c in df.columns]
    X = df[use_cols].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    missing = X.isna().sum()
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X.values)
    joblib.dump({"scaler": scaler, "columns": use_cols}, out_scaler)
    meta = pd.DataFrame({"column": use_cols, "missing_count": [int(missing[c]) for c in use_cols]})
    return Xs.astype(np.float32), meta, use_cols


def _drop_constant_duplicate_cols(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    use = [c for c in cols if c in df.columns]
    dropped_const: list[str] = []
    kept: list[str] = []
    for c in use:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            dropped_const.append(c)
        else:
            kept.append(c)
    dup_pairs: list[tuple[str, str]] = []
    final: list[str] = []
    seen_vals: list[tuple[str, pd.Series]] = []
    for c in kept:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        is_dup = False
        for prev_c, prev_s in seen_vals:
            if s.equals(prev_s):
                dup_pairs.append((c, prev_c))
                is_dup = True
                break
        if not is_dup:
            final.append(c)
            seen_vals.append((c, s))
    return final, dropped_const, dup_pairs


def pairwise_upper_spearman(D1: np.ndarray, D2: np.ndarray) -> float:
    iu = np.triu_indices_from(D1, k=1)
    a = D1[iu]
    b = D2[iu]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    corr, _ = spearmanr(a[mask], b[mask])
    return float(corr)


def distance_matrix(X: np.ndarray, metric: str, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        X = X[idx]
    else:
        idx = np.arange(n)
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = Xn @ Xn.T
        D = 1.0 - S
    else:
        sq = np.sum(X * X, axis=1, keepdims=True)
        D = np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))
    return D.astype(np.float32), idx.astype(np.int64)


def _pca_target_spearman(coords: np.ndarray, df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for t in targets:
        if t not in df.columns:
            continue
        b = pd.to_numeric(df[t], errors="coerce").to_numpy()
        for pi, name in [(0, "pca1"), (1, "pca2")]:
            a = coords[:, pi]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() <= 5:
                corr = np.nan
            else:
                corr, _ = spearmanr(a[mask], b[mask])
            rows.append({"pca": name, "feature": t, "spearman": float(corr) if corr == corr else None})
    return pd.DataFrame(rows)


def analyze_all(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = get_seed(cfg)
    out_dir = Path(cfg["paths"]["outputs_dir"])
    analysis_dir = out_dir / "analysis"
    ensure_dir(analysis_dir)
    k = int(cfg["analysis"].get("knn_k", 15))
    pca_k = int(cfg["analysis"].get("pca_n_components", 64))
    metrics: dict[str, Any] = {"channels": {}, "filters": {}}

    # ---- OCR quality summary (never PCA) ----
    q_path = out_dir / "ocr_quality.parquet"
    quality_df = pd.read_parquet(q_path) if q_path.exists() else None
    if quality_df is not None:
        vc = quality_df["ocr_quality_status"].value_counts().to_dict()
        metrics["ocr_quality"] = {
            "n": int(len(quality_df)),
            "status_counts": {str(a): int(b) for a, b in vc.items()},
            "truncation_rate": float(quality_df["hit_max_tokens"].mean()) if "hit_max_tokens" in quality_df else None,
            "repetitive_rate": float((quality_df["ocr_quality_status"] == "repetitive").mean()),
            "valid_rate": float((quality_df["ocr_quality_status"] == "valid").mean()),
        }

    # ---- Visual ----
    dim = 896
    meta_p = out_dir / "visual_embeddings.f32.meta.json"
    if meta_p.exists():
        dim = int(json.loads(meta_p.read_text()).get("dim", 896))
    vis_store = EmbeddingStore(
        mmap_path=out_dir / "visual_embeddings.f32.mmap",
        index_path=out_dir / "visual_index.parquet",
        dim=dim,
    )
    Xv, vis_idx = vis_store.load_matrix()
    if len(Xv):
        Xv = np.stack([l2_normalize(v) for v in Xv], axis=0)
        qv = quality_report(Xv, "visual")
        pca_v = fit_pca(Xv, pca_k, seed)
        knn_v = knn_stats(Xv, metric="cosine", k=k)
        np.save(analysis_dir / "visual_pca_coords.npy", pca_v.get("pca_coords", np.zeros((0, 2))))
        if "pca_model" in pca_v:
            joblib.dump(pca_v["pca_model"], analysis_dir / "visual_pca.joblib")
            pca_v = {kk: vv for kk, vv in pca_v.items() if kk != "pca_model"}
        np.save(analysis_dir / "visual_knn_kth.npy", knn_v.get("kth_neighbor_distance", np.array([])))
        np.save(analysis_dir / "visual_nn_indices.npy", knn_v.get("nn_indices", np.zeros((0, 1), dtype=np.int32)))
        np.save(analysis_dir / "visual_nn_distances.npy", knn_v.get("nn_distances", np.zeros((0, 1), dtype=np.float32)))
        vis_idx.to_parquet(analysis_dir / "visual_index_aligned.parquet", index=False)
        metrics["channels"]["visual"] = {
            "quality": qv,
            "pca": {k2: pca_v[k2] for k2 in pca_v if k2 != "pca_coords"},
            "knn_summary": {
                "kth_mean": float(np.mean(knn_v["kth_neighbor_distance"])) if "kth_neighbor_distance" in knn_v else None,
                "kth_p50": float(np.median(knn_v["kth_neighbor_distance"])) if "kth_neighbor_distance" in knn_v else None,
            },
            "n_used": int(len(Xv)),
        }
        if "pca_coords" in pca_v:
            coords_df = vis_idx[["image_id"]].copy()
            coords_df["pca1"] = pca_v["pca_coords"][:, 0]
            coords_df["pca2"] = pca_v["pca_coords"][:, 1]
            coords_df["kth_nn_dist"] = knn_v.get("kth_neighbor_distance")
            atomic_replace_parquet(coords_df, analysis_dir / "visual_coords.parquet")

    # ---- Layout (available only) ----
    lay_path = out_dir / "layout_features.parquet"
    if lay_path.exists():
        lay_all = pd.read_parquet(lay_path)
        lay = lay_all[lay_all["layout_available"].astype(bool)].copy() if "layout_available" in lay_all else lay_all
        lay_cols_use, dropped_c, dup_pairs = _drop_constant_duplicate_cols(lay, LAYOUT_FEATURE_COLS)
        Xl, lay_meta, lay_cols = robust_scale_explicit(lay, lay_cols_use, analysis_dir / "layout_scaler.joblib")
        lay_meta.to_parquet(analysis_dir / "layout_missing.parquet", index=False)
        ql = quality_report(Xl, "layout")
        pca_l = fit_pca(Xl, min(pca_k, Xl.shape[1]), seed)
        knn_l = knn_stats(Xl, metric="euclidean", k=k)
        np.save(analysis_dir / "layout_X_scaled.npy", Xl)
        np.save(analysis_dir / "layout_pca_coords.npy", pca_l.get("pca_coords", np.zeros((0, 2))))
        if "pca_model" in pca_l:
            joblib.dump(pca_l["pca_model"], analysis_dir / "layout_pca.joblib")
            pca_l = {kk: vv for kk, vv in pca_l.items() if kk != "pca_model"}
        np.save(analysis_dir / "layout_knn_kth.npy", knn_l.get("kth_neighbor_distance", np.array([])))
        np.save(analysis_dir / "layout_nn_indices.npy", knn_l.get("nn_indices", np.zeros((0, 1), dtype=np.int32)))
        np.save(analysis_dir / "layout_nn_distances.npy", knn_l.get("nn_distances", np.zeros((0, 1), dtype=np.float32)))
        ids_l = lay[["image_id"]].copy()
        ids_l.to_parquet(analysis_dir / "layout_index_aligned.parquet", index=False)
        if "pca_coords" in pca_l:
            cdf = ids_l.copy()
            cdf["pca1"] = pca_l["pca_coords"][:, 0]
            cdf["pca2"] = pca_l["pca_coords"][:, 1]
            cdf["kth_nn_dist"] = knn_l.get("kth_neighbor_distance")
            atomic_replace_parquet(cdf, analysis_dir / "layout_coords.parquet")
        corr = pd.DataFrame(Xl, columns=lay_cols).corr(method="spearman")
        corr.to_parquet(analysis_dir / "layout_feature_spearman.parquet")
        # High-corr pairs table
        pairs = []
        for i, c1 in enumerate(lay_cols):
            for j, c2 in enumerate(lay_cols):
                if j <= i:
                    continue
                v = corr.iloc[i, j]
                if pd.notna(v) and abs(float(v)) >= 0.7:
                    pairs.append({"feature_a": c1, "feature_b": c2, "spearman": float(v)})
        pd.DataFrame(pairs).to_parquet(analysis_dir / "layout_high_corr_pairs.parquet", index=False)
        metrics["channels"]["layout"] = {
            "quality": ql,
            "pca": {k2: pca_l[k2] for k2 in pca_l if k2 != "pca_coords"},
            "knn_summary": {
                "kth_mean": float(np.mean(knn_l["kth_neighbor_distance"])) if "kth_neighbor_distance" in knn_l else None,
            },
            "n_layout_available": int(len(lay)),
            "n_layout_total": int(len(lay_all)),
            "n_used": int(len(lay)),
            "dropped_constant_cols": dropped_c,
            "dropped_duplicate_pairs": [{"dropped": a, "same_as": b} for a, b in dup_pairs],
            "feature_cols": lay_cols,
        }
        metrics["filters"]["layout"] = "layout_available==True"

    # ---- Recognition (valid OCR quality only) ----
    rec_path = out_dir / "recognition_features.parquet"
    if rec_path.exists():
        rec_all = pd.read_parquet(rec_path)
        if quality_df is not None:
            valid_ids = set(
                quality_df.loc[quality_df["ocr_quality_status"] == "valid", "image_id"].astype(str)
            )
            rec = rec_all[rec_all["image_id"].astype(str).isin(valid_ids)].copy()
        else:
            rec = rec_all.copy()
        rec_cols_use, dropped_c, dup_pairs = _drop_constant_duplicate_cols(rec, RECOG_FEATURE_COLS)
        # log1p for heavy counts inside scaling matrix only
        rec_scaled_src = rec.copy()
        for c in ("equals_count", "frac_count", "sqrt_count", "subscript_count", "superscript_count",
                  "vector_symbol_count", "angle_symbol_count", "latex_env_count", "formula_span_count",
                  "output_character_count", "line_count", "mean_line_length"):
            if c in rec_scaled_src.columns and c in rec_cols_use:
                rec_scaled_src[c] = np.log1p(pd.to_numeric(rec_scaled_src[c], errors="coerce").fillna(0))
        Xr, rec_meta, rec_cols = robust_scale_explicit(
            rec_scaled_src, rec_cols_use, analysis_dir / "recognition_scaler.joblib"
        )
        rec_meta.to_parquet(analysis_dir / "recognition_missing.parquet", index=False)
        qr = quality_report(Xr, "recognition")
        pca_r = fit_pca(Xr, min(pca_k, max(Xr.shape[1], 1)), seed)
        knn_r = knn_stats(Xr, metric="euclidean", k=k)
        np.save(analysis_dir / "recognition_X_scaled.npy", Xr)
        np.save(analysis_dir / "recognition_pca_coords.npy", pca_r.get("pca_coords", np.zeros((0, 2))))
        if "pca_model" in pca_r:
            joblib.dump(pca_r["pca_model"], analysis_dir / "recognition_pca.joblib")
            pca_r = {kk: vv for kk, vv in pca_r.items() if kk != "pca_model"}
        np.save(analysis_dir / "recognition_knn_kth.npy", knn_r.get("kth_neighbor_distance", np.array([])))
        np.save(analysis_dir / "recognition_nn_indices.npy", knn_r.get("nn_indices", np.zeros((0, 1), dtype=np.int32)))
        np.save(analysis_dir / "recognition_nn_distances.npy", knn_r.get("nn_distances", np.zeros((0, 1), dtype=np.float32)))
        ids_r = rec[["image_id"]].copy()
        ids_r.to_parquet(analysis_dir / "recognition_index_aligned.parquet", index=False)
        if "pca_coords" in pca_r:
            cdf = ids_r.copy()
            cdf["pca1"] = pca_r["pca_coords"][:, 0]
            cdf["pca2"] = pca_r["pca_coords"][:, 1]
            cdf["kth_nn_dist"] = knn_r.get("kth_neighbor_distance")
            atomic_replace_parquet(cdf, analysis_dir / "recognition_coords.parquet")
            coords = pca_r["pca_coords"]
            # Content Spearman (rows=PC, cols=content)
            content_sp = _pca_target_spearman(coords, rec, RECOG_CONTENT_DIAG_COLS)
            content_sp.to_parquet(analysis_dir / "recognition_pca_target_spearman.parquet", index=False)
            # Quality Spearman on same valid subset join (diagnostic only)
            if quality_df is not None:
                qj = quality_df.set_index("image_id").loc[rec["image_id"].astype(str)].reset_index()
                q_sp = _pca_target_spearman(coords, qj, RECOG_QUALITY_DIAG_COLS)
                q_sp.to_parquet(analysis_dir / "recognition_quality_spearman.parquet", index=False)
        metrics["channels"]["recognition"] = {
            "quality": qr,
            "pca": {k2: pca_r[k2] for k2 in pca_r if k2 != "pca_coords"},
            "n_recognition_total": int(len(rec_all)),
            "n_used_valid": int(len(rec)),
            "dropped_constant_cols": dropped_c,
            "dropped_duplicate_pairs": [{"dropped": a, "same_as": b} for a, b in dup_pairs],
            "feature_cols": rec_cols,
            "excludes_quality_fields": True,
        }
        metrics["filters"]["recognition"] = "ocr_quality_status==valid"

    # ---- Cross-channel distance Spearman (common ids among filtered sets) ----
    try:
        ids_v = set(pd.read_parquet(analysis_dir / "visual_index_aligned.parquet")["image_id"].astype(str))
        ids_l = set(pd.read_parquet(analysis_dir / "layout_index_aligned.parquet")["image_id"].astype(str))
        ids_r = set(pd.read_parquet(analysis_dir / "recognition_index_aligned.parquet")["image_id"].astype(str))
        common = sorted(ids_v & ids_l & ids_r)
        max_n = int(cfg["analysis"].get("distance_matrix_sample_size", 500))
        rng = np.random.default_rng(seed)
        if len(common) > max_n:
            common = list(rng.choice(common, size=max_n, replace=False))
        if len(common) >= 20:
            Xv_c, v_df = vis_store.load_matrix(common)
            order = {iid: i for i, iid in enumerate(v_df["image_id"].astype(str))}
            common = [c for c in common if c in order]
            Xv_c = Xv_c[[order[c] for c in common]]
            lay = pd.read_parquet(lay_path).set_index("image_id").loc[common]
            rec = pd.read_parquet(rec_path).set_index("image_id").loc[common]
            Xl_c, _, _ = robust_scale_explicit(
                lay.reset_index(), LAYOUT_FEATURE_COLS, analysis_dir / "layout_scaler_common.joblib"
            )
            Xr_c, _, _ = robust_scale_explicit(
                rec.reset_index(), RECOG_FEATURE_COLS, analysis_dir / "recognition_scaler_common.joblib"
            )
            Dv, _ = distance_matrix(Xv_c, "cosine", max_n, seed)
            Dl, _ = distance_matrix(Xl_c, "euclidean", max_n, seed)
            Dr, _ = distance_matrix(Xr_c, "euclidean", max_n, seed)
            n = min(Dv.shape[0], Dl.shape[0], Dr.shape[0])
            Dv, Dl, Dr = Dv[:n, :n], Dl[:n, :n], Dr[:n, :n]
            mat = np.array(
                [
                    [1.0, pairwise_upper_spearman(Dv, Dl), pairwise_upper_spearman(Dv, Dr)],
                    [pairwise_upper_spearman(Dl, Dv), 1.0, pairwise_upper_spearman(Dl, Dr)],
                    [pairwise_upper_spearman(Dr, Dv), pairwise_upper_spearman(Dr, Dl), 1.0],
                ]
            )
            metrics["cross_channel_distance_spearman"] = {
                "labels": ["visual", "layout", "recognition"],
                "matrix": mat.tolist(),
                "n_common_sampled": n,
            }
            np.save(analysis_dir / "cross_channel_spearman.npy", mat)
            pd.DataFrame({"image_id": common[:n]}).to_parquet(analysis_dir / "cross_channel_ids.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        metrics["cross_channel_error"] = f"{type(e).__name__}: {e}"

    atomic_write_json(Path(cfg["paths"]["reports_dir"]) / "feature_metrics.json", metrics)
    atomic_write_json(analysis_dir / "feature_metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    m = analyze_all(cfg)
    print(f"[analyze_features] channels={list(m.get('channels', {}).keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
