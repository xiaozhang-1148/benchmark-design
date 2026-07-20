"""Clustering analysis for spatial layout patterns."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.gpu import is_gpu_available
from heatmap_analysis.gpu_clustering import evaluate_kmeans_k_gpu, kmeans_fit, kmeans_predict, pca_reduce
from heatmap_analysis.heatmap import flatten_heatmap
from heatmap_analysis.utils import ensure_dir, save_json, standardize_features
from heatmap_analysis.visualization import plot_heatmap

logger = logging.getLogger("heatmap_analysis.clustering")


def _clustering_features(cfg: AnalysisConfig, metrics: pd.DataFrame, rel_map: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    ids = [i for i in metrics["image_id"].astype(str) if i in rel_map]
    rel_key = getattr(cfg.heatmap, "clustering_grid_version", "smoothed_relative")
    if rel_key == "raw_relative":
        rel_key = "d_rel"
    else:
        rel_key = "d_rel_smooth"

    cache_dir = cfg.cache_dir / "per_image"
    rel_vecs = []
    for iid in ids:
        d = np.load(cache_dir / f"{iid}.npz")
        grid = d["d_rel"] if rel_key == "d_rel" else d["d_rel_smooth"]
        rel_vecs.append(flatten_heatmap(grid))
    rel_mat = np.stack(rel_vecs)

    if cfg.clustering.feature_mode == "relative_layout":
        return ids, rel_mat

    mdf = metrics.set_index("image_id").loc[ids]
    scalar_cols = [
        "ink_coverage",
        "centroid_x",
        "centroid_y",
        "spatial_entropy",
        "hotspot_concentration",
        "active_cell_ratio",
    ]
    scalars = mdf[scalar_cols].values.astype(np.float64)
    scalars_std, _, _ = standardize_features(scalars)
    combined = np.hstack([rel_mat, scalars_std])
    return ids, combined


def _effective_k_range(n: int, k_min: int, k_max: int) -> range:
    if n < 4:
        return range(2, 3)
    upper = min(k_max, n - 1, max(2, n // 3))
    lower = min(k_min, upper)
    return range(lower, upper + 1)


def evaluate_kmeans_k(X: np.ndarray, k_range: range, seed: int) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_stability(X: np.ndarray, k: int, seed: int, n_iter: int) -> dict:
    """Assess label stability across bootstrap resamples."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    base = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    ari_scores = []
    for t in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        sub = X[idx]
        if len(set(idx)) < k:
            continue
        labels = KMeans(n_clusters=k, random_state=seed + t + 1, n_init=5).fit_predict(sub)
        # Map bootstrap labels back
        mapped = np.full(n, -1)
        for pos, orig in enumerate(idx):
            mapped[orig] = labels[pos]
        valid = mapped >= 0
        if valid.sum() < k:
            continue
        ari_scores.append(adjusted_rand_score(base[valid], mapped[valid]))
    return {
        "bootstrap_iterations": n_iter,
        "mean_ari": float(np.mean(ari_scores)) if ari_scores else 0.0,
        "std_ari": float(np.std(ari_scores)) if ari_scores else 0.0,
    }


def select_k(metrics_df: pd.DataFrame, k_fixed: int | None) -> int:
    if k_fixed is not None:
        return k_fixed
    if metrics_df.empty:
        return 2
    best = metrics_df.loc[metrics_df["silhouette"].idxmax()]
    return int(best["k"])


def _fit_kmeans_at_k(
    X_pca: np.ndarray,
    k: int,
    seed: int,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids) in PCA feature space."""
    if k < 2 or k >= X_pca.shape[0]:
        raise ValueError(f"invalid k={k} for n={X_pca.shape[0]}")
    if use_gpu:
        return kmeans_fit(X_pca, k, seed, use_gpu=True, n_init=10)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_pca)
    return labels.astype(np.int32), km.cluster_centers_.astype(np.float64)


def _heatmap_centroids_from_raw(
    X_raw: np.ndarray,
    grid_size: int,
    k: int,
    seed: int,
    use_gpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """KMeans directly on flattened heatmaps; returns (labels, centers_rel grid k×H×W)."""
    n_feat = grid_size * grid_size
    if X_raw.shape[1] < n_feat:
        raise ValueError("X_raw does not contain full heatmap features")
    X_hm = X_raw[:, :n_feat]
    if use_gpu and is_gpu_available():
        labels, centers = kmeans_fit(X_hm, k, seed, use_gpu=True, n_init=10)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_hm)
        centers = km.cluster_centers_
    centers_grid = centers.reshape(k, grid_size, grid_size)
    row_sum = centers_grid.sum(axis=(1, 2), keepdims=True)
    centers_rel = np.divide(
        centers_grid,
        np.maximum(row_sum, 1e-12),
        where=row_sum > 1e-12,
        out=np.zeros_like(centers_grid),
    )
    return labels.astype(np.int32), centers_rel.astype(np.float64)


def _export_fixed_k_centers(
    cfg: AnalysisConfig,
    out_root: Path,
    k: int,
    X_pca: np.ndarray,
    X_raw: np.ndarray,
    ids: list[str],
    rel_map: dict[str, np.ndarray],
    abs_map: dict[str, np.ndarray],
    mdf: pd.DataFrame,
    n: int,
    use_gpu: bool,
) -> None:
    """Export cluster labels, PCA centroids, and mean heatmap centers for a fixed k."""
    if k < 2 or k >= n:
        logger.warning("Skip extra k=%d (n=%d)", k, n)
        return

    labels, centroids_pca = _fit_kmeans_at_k(X_pca, k, cfg.clustering.random_seed, use_gpu)
    gs = cfg.heatmap.grid_size

    kdir = ensure_dir(out_root / f"k_fixed_{k:02d}")
    label_df = pd.DataFrame({"image_id": ids, "cluster": labels})
    label_df.to_csv(kdir / "cluster_labels.csv", index=False)

    centers_rel = np.zeros((k, gs, gs), dtype=np.float64)
    centers_abs = np.zeros((k, gs, gs), dtype=np.float64)
    cluster_sizes = np.zeros(k, dtype=np.int64)

    for cid in range(k):
        cids = label_df[label_df["cluster"] == cid]["image_id"].astype(str).tolist()
        cids = [i for i in cids if i in rel_map]
        cluster_sizes[cid] = len(cids)
        if not cids:
            continue
        rel_stack = np.stack([rel_map[i] for i in cids])
        abs_stack = np.stack([abs_map[i] for i in cids])
        centers_rel[cid] = np.mean(rel_stack, axis=0)
        centers_abs[cid] = np.mean(abs_stack, axis=0)

        cdir = ensure_dir(kdir / f"cluster_{cid:02d}")
        np.savez_compressed(
            cdir / "cluster_stats.npz",
            mean_rel=centers_rel[cid],
            mean_abs=centers_abs[cid],
            n_samples=len(cids),
            centroid_pca=centroids_pca[cid],
        )
        plot_heatmap(
            centers_rel[cid],
            cdir / "center_mean_rel.png",
            f"k={k} cluster {cid} center (mean rel)",
            cmap=cfg.report.colormap,
            n_samples=len(cids),
        )
        plot_heatmap(
            centers_abs[cid],
            cdir / "center_mean_abs.png",
            f"k={k} cluster {cid} center (mean abs)",
            cmap=cfg.report.colormap,
            n_samples=len(cids),
        )

    centers_rel_kmeans = None
    if cfg.clustering.feature_mode == "relative_layout":
        try:
            _, centers_rel_kmeans = _heatmap_centroids_from_raw(
                X_raw, gs, k, cfg.clustering.random_seed, use_gpu
            )
        except ValueError:
            centers_rel_kmeans = None

    save_payload: dict = {
        "centroids_pca": centroids_pca,
        "centers_rel_mean": centers_rel,
        "centers_abs_mean": centers_abs,
        "cluster_sizes": cluster_sizes,
        "k": k,
        "n_samples": n,
        "grid_size": gs,
    }
    if centers_rel_kmeans is not None:
        save_payload["centers_rel_kmeans"] = centers_rel_kmeans
        for cid in range(k):
            plot_heatmap(
                centers_rel_kmeans[cid],
                kdir / f"cluster_{cid:02d}" / "center_kmeans_rel.png",
                f"k={k} cluster {cid} KMeans centroid (rel layout)",
                cmap=cfg.report.colormap,
            )

    np.savez_compressed(kdir / "cluster_centers.npz", **save_payload)

    # CSV table of flattened centers for downstream analysis
    rows = []
    for cid in range(k):
        row = {
            "cluster": cid,
            "n_samples": int(cluster_sizes[cid]),
            "centroid_pca_norm": float(np.linalg.norm(centroids_pca[cid])),
        }
        flat_rel = centers_rel[cid].ravel()
        for i, v in enumerate(flat_rel):
            row[f"center_rel_{i}"] = float(v)
        rows.append(row)
    pd.DataFrame(rows).to_csv(kdir / "cluster_centers.csv", index=False)

    save_json(
        kdir / "summary.json",
        {
            "k": k,
            "n_samples": n,
            "cluster_sizes": cluster_sizes.tolist(),
            "feature_mode": cfg.clustering.feature_mode,
        },
    )
    logger.info("Exported fixed k=%d cluster centers to %s", k, kdir)


def run_clustering(cfg: AnalysisConfig) -> dict:
    metrics = pd.read_csv(cfg.output.output_dir / "tables" / "per_image_metrics.csv")
    cache_dir = cfg.cache_dir / "per_image"
    rel_map = {}
    abs_map = {}
    for npz in cache_dir.glob("*.npz"):
        d = np.load(npz)
        rel_map[npz.stem] = d["d_rel_smooth"] if "d_rel_smooth" in d else d["d_rel"]
        abs_map[npz.stem] = d["d_abs"]

    template_groups: dict[str | None, pd.DataFrame] = {"__all__": metrics}
    if cfg.clustering.separate_by_template and "template_id" in metrics.columns:
        template_groups = {}
        for k, g in metrics.groupby("template_id", dropna=False):
            key = "default" if k is None or (isinstance(k, float) and np.isnan(k)) or str(k) == "nan" else str(k)
            template_groups[key] = g

    all_results: dict = {}
    for tmpl_key, mdf in template_groups.items():
        ids, X_raw = _clustering_features(cfg, mdf, rel_map)
        n = len(ids)
        if n < cfg.clustering.min_samples_for_clustering:
            logger.warning("Too few samples (%d) for clustering in group %s", n, tmpl_key)
            continue

        gpu_ok = cfg.gpu.enabled and cfg.gpu.clustering and is_gpu_available()
        cluster_model = None
        pca = None
        scaler = None
        used_gpu_cluster = False

        if gpu_ok and n >= 64 and cfg.clustering.algorithm == "kmeans":
            X_pca, cumvar, _ = pca_reduce(
                X_raw,
                cfg.clustering.pca_variance,
                cfg.clustering.pca_n_components,
                cfg.clustering.random_seed,
                use_gpu=True,
            )
            X_pca = StandardScaler().fit_transform(X_pca)
            k_range = _effective_k_range(n, cfg.clustering.k_min, cfg.clustering.k_max)
            k_metrics = evaluate_kmeans_k_gpu(X_pca, k_range, cfg.clustering.random_seed, use_gpu=True)
            k = select_k(k_metrics, cfg.clustering.k_fixed)
            labels = kmeans_predict(X_pca, k, cfg.clustering.random_seed, use_gpu=True)
            used_gpu_cluster = True
            logger.info("Clustering group %s: GPU PCA+KMeans", tmpl_key)
        else:
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            n_comp = cfg.clustering.pca_n_components
            if n_comp is None:
                pca = PCA(n_components=cfg.clustering.pca_variance, random_state=cfg.clustering.random_seed)
            else:
                pca = PCA(n_components=min(n_comp, n - 1, X.shape[1]), random_state=cfg.clustering.random_seed)
            X_pca = pca.fit_transform(X)
            cumvar = float(np.sum(pca.explained_variance_ratio_))

            k_range = _effective_k_range(n, cfg.clustering.k_min, cfg.clustering.k_max)
            k_metrics = evaluate_kmeans_k(X_pca, k_range, cfg.clustering.random_seed)
            k = select_k(k_metrics, cfg.clustering.k_fixed)

            if cfg.clustering.algorithm == "hierarchical":
                cluster_model = AgglomerativeClustering(n_clusters=k)
                labels = cluster_model.fit_predict(X_pca)
            elif cfg.clustering.algorithm == "hdbscan":
                try:
                    import hdbscan

                    cluster_model = hdbscan.HDBSCAN(min_cluster_size=max(5, n // 20))
                    labels = cluster_model.fit_predict(X_pca)
                    k = len(set(labels)) - (1 if -1 in labels else 0)
                except ImportError:
                    logger.warning("hdbscan not installed, falling back to kmeans")
                    cluster_model = KMeans(n_clusters=k, random_state=cfg.clustering.random_seed, n_init=10)
                    labels = cluster_model.fit_predict(X_pca)
            else:
                cluster_model = KMeans(n_clusters=k, random_state=cfg.clustering.random_seed, n_init=10)
                labels = cluster_model.fit_predict(X_pca)

        stability = bootstrap_stability(X_pca, k, cfg.clustering.random_seed, cfg.clustering.bootstrap_iterations)

        if used_gpu_cluster:
            labels_alt = kmeans_predict(X_pca, k, cfg.clustering.random_seed + 999, use_gpu=True, n_init=3)
        else:
            labels_alt = KMeans(n_clusters=k, random_state=cfg.clustering.random_seed + 999, n_init=10).fit_predict(
                X_pca
            )
        seed_ari = adjusted_rand_score(labels, labels_alt)

        label_df = pd.DataFrame({"image_id": ids, "cluster": labels, "template_group": tmpl_key})
        out_root = ensure_dir(cfg.output.output_dir / "clustering" / str(tmpl_key))
        label_df.to_csv(out_root / "cluster_labels.csv", index=False)
        k_metrics.to_csv(out_root / "k_selection_metrics.csv", index=False)

        models_dir = ensure_dir(cfg.output.output_dir / "models")
        if scaler is not None:
            joblib.dump(scaler, models_dir / f"scaler_{tmpl_key}.joblib")
        if pca is not None:
            joblib.dump(pca, models_dir / f"pca_{tmpl_key}.joblib")
        if cluster_model is not None and cfg.clustering.algorithm == "kmeans":
            joblib.dump(cluster_model, models_dir / f"kmeans_{tmpl_key}.joblib")
        if used_gpu_cluster:
            save_json(
                models_dir / f"gpu_cluster_{tmpl_key}.json",
                {"backend": "cupy", "pca_variance": cumvar, "k": k},
            )

        # Per-cluster outputs
        small_cluster_warn = []
        for cid in sorted(set(labels)):
            if cid < 0:
                continue
            cids = label_df[label_df["cluster"] == cid]["image_id"].tolist()
            if len(cids) < max(3, n * 0.02):
                small_cluster_warn.append({"cluster": int(cid), "n": len(cids)})

            rel_stack = np.stack([rel_map[i] for i in cids if i in rel_map])
            abs_stack = np.stack([abs_map[i] for i in cids if i in abs_map])
            cdir = ensure_dir(out_root / f"cluster_{cid:02d}")
            np.savez_compressed(
                cdir / "cluster_stats.npz",
                mean_rel=np.mean(rel_stack, axis=0),
                mean_abs=np.mean(abs_stack, axis=0),
                std_rel=np.std(rel_stack, axis=0),
                n_samples=len(cids),
            )
            cm = mdf[mdf["image_id"].astype(str).isin(cids)]
            cm.describe().to_csv(cdir / "metrics_summary.csv")

            # Representative & outlier samples by distance to centroid
            centroid = np.mean(rel_stack, axis=0).ravel()
            dists = [np.linalg.norm(rel_map[i].ravel() - centroid) for i in cids]
            order = np.argsort(dists)
            n_repr = min(cfg.report.representative_samples, len(cids))
            repr_ids = [cids[i] for i in order[:n_repr]]
            outlier_ids = [cids[i] for i in order[-n_repr:]]
            pd.DataFrame({"representative": repr_ids}).to_csv(cdir / "representative_samples.csv", index=False)
            pd.DataFrame({"outlier": outlier_ids}).to_csv(cdir / "outlier_samples.csv", index=False)

        # Anomaly detection
        iso = IsolationForest(random_state=cfg.clustering.random_seed, contamination="auto")
        anomaly = iso.fit_predict(X_pca)
        anomaly_ids = [ids[i] for i, a in enumerate(anomaly) if a < 0]
        pca_outliers = []
        if X_pca.shape[1] >= 2:
            dist = np.linalg.norm(X_pca - np.mean(X_pca, axis=0), axis=1)
            thresh = np.mean(dist) + 2 * np.std(dist)
            pca_outliers = [ids[i] for i, d in enumerate(dist) if d > thresh]

        blank_ids = mdf[mdf["is_blank"] == True]["image_id"].astype(str).tolist()  # noqa: E712
        high_cov = mdf.nlargest(min(10, n), "ink_coverage")["image_id"].astype(str).tolist()

        anomaly_report = {
            "isolation_forest": anomaly_ids,
            "pca_outliers": pca_outliers,
            "blank_samples": blank_ids,
            "high_ink_coverage": high_cov,
        }
        save_json(out_root / "anomalies.json", anomaly_report)

        result = {
            "template_group": tmpl_key,
            "n_samples": n,
            "selected_k": k,
            "pca_cumulative_variance": cumvar,
            "k_selection": k_metrics.to_dict(orient="records"),
            "stability": stability,
            "seed_consistency_ari": float(seed_ari),
            "small_cluster_warnings": small_cluster_warn,
            "feature_mode": cfg.clustering.feature_mode,
            "clustering_grid_version": getattr(cfg.heatmap, "clustering_grid_version", "smoothed_relative"),
            "algorithm": cfg.clustering.algorithm,
        }
        save_json(out_root / "clustering_report.json", result)

        for extra_k in cfg.clustering.extra_k_outputs:
            if extra_k == k:
                continue
            try:
                _export_fixed_k_centers(
                    cfg,
                    out_root,
                    int(extra_k),
                    X_pca,
                    X_raw,
                    ids,
                    rel_map,
                    abs_map,
                    mdf,
                    n,
                    used_gpu_cluster,
                )
            except ValueError as e:
                logger.warning("Could not export extra k=%s: %s", extra_k, e)

        all_results[str(tmpl_key)] = result
        logger.info("Clustering group %s: k=%d, n=%d", tmpl_key, k, n)

    return all_results
