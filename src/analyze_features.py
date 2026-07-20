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


LAYOUT_FEATURE_COLS = [
    "blank_ratio",
    "content_area_ratio",
    "mean_block_area",
    "std_block_area",
    "mean_block_aspect_ratio",
    "std_block_aspect_ratio",
    "mean_center_x",
    "mean_center_y",
    "upper_region_density",
    "middle_region_density",
    "lower_region_density",
    "left_region_density",
    "center_region_density",
    "right_region_density",
    "block_count_log1p",
    "text_block_count_log1p",
    "formula_block_count_log1p",
    "figure_block_count_log1p",
    "table_block_count_log1p",
    "reading_order_length_log1p",
    "reading_order_vertical_violation_count_log1p",
] + [f"occupancy_{i}_{j}" for i in range(4) for j in range(4)]

RECOG_FEATURE_COLS = [
    "digit_ratio",
    "latin_ratio",
    "chinese_ratio",
    "math_symbol_ratio",
    "whitespace_ratio",
    "formula_character_ratio",
    "repetition_ratio",
    "output_token_count_log1p",
    "output_character_count_log1p",
    "line_count_log1p",
    "markdown_heading_count_log1p",
    "formula_count_log1p",
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
    }


def knn_stats(X: np.ndarray, metric: str, k: int = 15) -> dict[str, Any]:
    n = X.shape[0]
    k_eff = min(k + 1, n)  # include self
    if n < 2:
        return {"error": "too few samples"}
    if faiss is not None and metric in {"cosine", "inner_product"} and n >= 100:
        xb = X.astype(np.float32).copy()
        faiss.normalize_L2(xb)
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        D, I = index.search(xb, k_eff)
        # convert similarity to distance
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


def robust_scale_explicit(df: pd.DataFrame, cols: list[str], out_scaler: Path) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
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
        # euclidean
        sq = np.sum(X * X, axis=1, keepdims=True)
        D = np.sqrt(np.maximum(sq + sq.T - 2 * X @ X.T, 0.0))
    return D.astype(np.float32), idx.astype(np.int64)


def analyze_all(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = get_seed(cfg)
    out_dir = Path(cfg["paths"]["outputs_dir"])
    analysis_dir = out_dir / "analysis"
    ensure_dir(analysis_dir)
    k = int(cfg["analysis"].get("knn_k", 15))
    pca_k = int(cfg["analysis"].get("pca_n_components", 64))
    metrics: dict[str, Any] = {"channels": {}}

    # ---- Visual ----
    vis_store = EmbeddingStore(
        mmap_path=out_dir / "visual_embeddings.f32.mmap",
        index_path=out_dir / "visual_index.parquet",
        dim=896,
    )
    # fix dim from meta if needed
    meta_p = out_dir / "visual_embeddings.f32.meta.json"
    if meta_p.exists():
        dim = json.loads(meta_p.read_text()).get("dim", 896)
        vis_store = EmbeddingStore(
            mmap_path=out_dir / "visual_embeddings.f32.mmap",
            index_path=out_dir / "visual_index.parquet",
            dim=dim,
        )
    Xv, vis_idx = vis_store.load_matrix()
    if len(Xv):
        # ensure L2
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
        metrics["channels"]["visual"] = {"quality": qv, "pca": {k2: pca_v[k2] for k2 in pca_v if k2 != "pca_coords"}, "knn_summary": {
            "kth_mean": float(np.mean(knn_v["kth_neighbor_distance"])) if "kth_neighbor_distance" in knn_v else None,
            "kth_p50": float(np.median(knn_v["kth_neighbor_distance"])) if "kth_neighbor_distance" in knn_v else None,
        }}
        # coords table
        if "pca_coords" in pca_v:
            coords_df = vis_idx[["image_id"]].copy()
            coords_df["pca1"] = pca_v["pca_coords"][:, 0]
            coords_df["pca2"] = pca_v["pca_coords"][:, 1]
            coords_df["kth_nn_dist"] = knn_v.get("kth_neighbor_distance")
            atomic_replace_parquet(coords_df, analysis_dir / "visual_coords.parquet")

    # ---- Layout ----
    lay_path = out_dir / "layout_features.parquet"
    if lay_path.exists():
        lay = pd.read_parquet(lay_path)
        Xl, lay_meta, lay_cols = robust_scale_explicit(lay, LAYOUT_FEATURE_COLS, analysis_dir / "layout_scaler.joblib")
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
        # layout spearman corr among features
        corr = pd.DataFrame(Xl, columns=lay_cols).corr(method="spearman")
        corr.to_parquet(analysis_dir / "layout_feature_spearman.parquet")
        metrics["channels"]["layout"] = {
            "quality": ql,
            "pca": {k2: pca_l[k2] for k2 in pca_l if k2 != "pca_coords"},
            "knn_summary": {
                "kth_mean": float(np.mean(knn_l["kth_neighbor_distance"])) if "kth_neighbor_distance" in knn_l else None,
            },
            "n_layout_available": int(lay["layout_available"].sum()) if "layout_available" in lay else None,
        }

    # ---- Recognition ----
    rec_path = out_dir / "recognition_features.parquet"
    if rec_path.exists():
        rec = pd.read_parquet(rec_path)
        Xr, rec_meta, rec_cols = robust_scale_explicit(rec, RECOG_FEATURE_COLS, analysis_dir / "recognition_scaler.joblib")
        rec_meta.to_parquet(analysis_dir / "recognition_missing.parquet", index=False)
        qr = quality_report(Xr, "recognition")
        pca_r = fit_pca(Xr, min(pca_k, Xr.shape[1]), seed)
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
        # PCA vs length-like features
        if "pca_coords" in fit_pca.__dict__ or True:
            coords = np.load(analysis_dir / "recognition_pca_coords.npy")
            if coords.shape[0] == len(rec):
                targets = ["output_token_count", "formula_count", "digit_ratio", "math_symbol_ratio", "repetition_ratio"]
                rows = []
                for t in targets:
                    if t not in rec.columns:
                        continue
                    for pi, name in [(0, "pca1"), (1, "pca2")]:
                        a = coords[:, pi]
                        b = rec[t].astype(float).to_numpy()
                        mask = np.isfinite(a) & np.isfinite(b)
                        corr, _ = spearmanr(a[mask], b[mask]) if mask.sum() > 5 else (np.nan, None)
                        rows.append({"pca": name, "feature": t, "spearman": float(corr) if corr == corr else None})
                pd.DataFrame(rows).to_parquet(analysis_dir / "recognition_pca_target_spearman.parquet", index=False)
        metrics["channels"]["recognition"] = {
            "quality": qr,
            "pca": {k2: pca_r[k2] for k2 in pca_r if k2 != "pca_coords"},
            "logprob_available_rate": float(rec["logprob_available"].mean()) if "logprob_available" in rec else None,
        }

    # ---- Cross-channel distance Spearman (common ids) ----
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
            # align order
            order = {iid: i for i, iid in enumerate(v_df["image_id"].astype(str))}
            common = [c for c in common if c in order]
            Xv_c = Xv_c[[order[c] for c in common]]
            lay = pd.read_parquet(lay_path).set_index("image_id").loc[common]
            rec = pd.read_parquet(rec_path).set_index("image_id").loc[common]
            Xl_c, _, _ = robust_scale_explicit(lay.reset_index(), LAYOUT_FEATURE_COLS, analysis_dir / "layout_scaler_common.joblib")
            Xr_c, _, _ = robust_scale_explicit(rec.reset_index(), RECOG_FEATURE_COLS, analysis_dir / "recognition_scaler_common.joblib")
            Dv, _ = distance_matrix(Xv_c, "cosine", max_n, seed)
            Dl, _ = distance_matrix(Xl_c, "euclidean", max_n, seed)
            Dr, _ = distance_matrix(Xr_c, "euclidean", max_n, seed)
            # ensure same n
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
    proj = Path(cfg["paths"].get("project_reports_dir", "reports"))
    ensure_dir(proj)
    atomic_write_json(proj / "feature_metrics.json", metrics)
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
