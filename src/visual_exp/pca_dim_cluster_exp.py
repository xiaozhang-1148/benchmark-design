"""Multi-method clustering on L2 → PCA(64/128) features.

Methods:
  - KMeans (Euclidean on PCA coordinates)
  - Spherical KMeans (KMeans on L2-renormalized PCA features ≈ cosine)
  - PCA + HDBSCAN
  - Leiden (kNN graph community detection)

For KMeans / Spherical KMeans: sweep many K and write metrics tables + PCA figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from ..utils import atomic_write_json, ensure_dir
from .io_util import atomic_write_parquet, stamp_run_id

try:
    import hdbscan
except Exception as e:  # noqa: BLE001
    hdbscan = None
    _HDBSCAN_ERR = e
else:
    _HDBSCAN_ERR = None

try:
    import igraph as ig
    import leidenalg
except Exception as e:  # noqa: BLE001
    ig = None
    leidenalg = None
    _LEIDEN_ERR = e
else:
    _LEIDEN_ERR = None

_CLUSTER_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]

# Default K sweep for Euclidean / spherical KMeans.
DEFAULT_K_LIST: tuple[int, ...] = tuple(range(2, 21))

# PCA figures for these K values (+ best-by-silhouette).
DEFAULT_FIGURE_K: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _silhouette_sample(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    rng: np.random.Generator,
    sample: int = 5000,
) -> float:
    uniq = set(int(x) for x in labels.tolist()) - {-1}
    if len(uniq) < 2:
        return float("nan")
    n = X.shape[0]
    if sample < n:
        ii = rng.choice(n, size=sample, replace=False)
        lab = labels[ii]
        if len(set(int(x) for x in lab.tolist()) - {-1}) < 2:
            return float("nan")
        return float(silhouette_score(X[ii], lab, metric=metric))
    return float(silhouette_score(X, labels, metric=metric))


def _size_stats(labels: np.ndarray, k: int | None = None) -> dict[str, Any]:
    valid = labels[labels >= 0]
    if valid.size == 0:
        return {
            "n_clusters": 0,
            "n_noise": int(np.sum(labels < 0)),
            "min_cluster_size": 0,
            "max_cluster_size": 0,
            "mean_cluster_size": 0.0,
            "cluster_sizes": [],
        }
    if k is None:
        sizes = np.bincount(valid.astype(int))
        sizes = sizes[sizes > 0]
    else:
        sizes = np.bincount(valid.astype(int), minlength=k)
    return {
        "n_clusters": int(len(sizes) if k is None else k),
        "n_noise": int(np.sum(labels < 0)),
        "min_cluster_size": int(sizes.min()) if sizes.size else 0,
        "max_cluster_size": int(sizes.max()) if sizes.size else 0,
        "mean_cluster_size": float(sizes.mean()) if sizes.size else 0.0,
        "cluster_sizes": sizes.astype(int).tolist(),
    }


def _fit_kmeans(X: np.ndarray, k: int, seed: int, n_init: int) -> tuple[np.ndarray, float]:
    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        algorithm="lloyd",
        max_iter=300,
    )
    labels = km.fit_predict(X).astype(np.int32)
    return labels, float(km.inertia_)


def _plot_pca_clusters(
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    out_path: Path,
    dpi: int = 150,
    max_legend: int = 12,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    noise = labels < 0
    if noise.any():
        ax.scatter(
            xy[noise, 0],
            xy[noise, 1],
            s=4,
            c="#bdbdbd",
            alpha=0.3,
            linewidths=0,
            label=f"noise (n={int(noise.sum())})",
            zorder=1,
        )
    clusters = sorted(int(c) for c in set(labels.tolist()) if c >= 0)
    if len(clusters) <= max_legend:
        for i, c in enumerate(clusters):
            mask = labels == c
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=6,
                c=_CLUSTER_COLORS[i % len(_CLUSTER_COLORS)],
                alpha=0.55,
                linewidths=0,
                label=f"c{c} (n={int(mask.sum())})",
                zorder=2,
            )
        ax.legend(loc="best", fontsize=7, markerscale=2, framealpha=0.9)
    else:
        sc = ax.scatter(
            xy[~noise, 0],
            xy[~noise, 1],
            c=labels[~noise],
            s=5,
            cmap="tab20",
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, label="cluster id")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    _save(fig, out_path, dpi)


def _plot_metric_curves(metrics: pd.DataFrame, method: str, figures: Path, dpi: int) -> None:
    if metrics.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = metrics["k"].to_numpy()
    axes[0, 0].plot(x, metrics["inertia"], marker="o", ms=3)
    axes[0, 0].set_title("Inertia")
    axes[0, 1].plot(x, metrics["silhouette"], marker="o", ms=3, color="#2ca02c")
    axes[0, 1].set_title("Silhouette (sample)")
    axes[1, 0].plot(x, metrics["calinski_harabasz"], marker="o", ms=3, color="#ff7f0e")
    axes[1, 0].set_title("Calinski–Harabasz")
    axes[1, 1].plot(x, metrics["davies_bouldin"], marker="o", ms=3, color="#d62728")
    axes[1, 1].set_title("Davies–Bouldin (lower better)")
    for ax in axes.ravel():
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{method}: metrics vs K")
    fig.tight_layout()
    _save(fig, figures / f"{method}_metrics_vs_k.png", dpi)


def _run_k_sweep(
    X: np.ndarray,
    xy2: np.ndarray,
    ids: list[str],
    *,
    method: str,
    metric_for_sil: str,
    out_dir: Path,
    figures: Path,
    k_list: tuple[int, ...],
    figure_ks: set[int],
    seed: int,
    run_id: str,
    dpi: int,
) -> dict[str, Any]:
    """Sweep K for KMeans or spherical_kmeans; write metrics + selected PCA figures."""
    method_dir = ensure_dir(out_dir / method)
    assign_dir = ensure_dir(method_dir / "assignments")
    fig_dir = ensure_dir(figures / method)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    best_sil = (-np.inf, None, None)  # score, k, labels

    n = X.shape[0]
    for k in k_list:
        if k >= n:
            continue
        n_init = 15 if k <= 20 else (5 if k <= 200 else 3)
        print(f"[cluster_exp] {method} k={k} …")
        labels, inertia = _fit_kmeans(X, k, seed=seed, n_init=n_init)
        sil = _silhouette_sample(X, labels, metric=metric_for_sil, rng=rng)
        try:
            ch = float(calinski_harabasz_score(X, labels))
        except Exception:  # noqa: BLE001
            ch = float("nan")
        try:
            db = float(davies_bouldin_score(X, labels))
        except Exception:  # noqa: BLE001
            db = float("nan")
        stats = _size_stats(labels, k=k)
        row = {
            "method": method,
            "k": int(k),
            "inertia": inertia,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "min_cluster_size": stats["min_cluster_size"],
            "max_cluster_size": stats["max_cluster_size"],
            "mean_cluster_size": stats["mean_cluster_size"],
            "n_empty": int(sum(s == 0 for s in stats["cluster_sizes"])),
            "run_id": run_id,
        }
        rows.append(row)

        assign = stamp_run_id(
            pd.DataFrame({"image_id": ids, "cluster": labels.astype(int), "method": method, "k": k}),
            run_id,
        )
        atomic_write_parquet(assign, assign_dir / f"k{k}_assignments.parquet")

        if np.isfinite(sil) and sil > best_sil[0]:
            best_sil = (sil, int(k), labels)

        if k in figure_ks:
            _plot_pca_clusters(
                xy2,
                labels,
                title=f"L2→PCA · {method} (k={k})",
                out_path=fig_dir / f"pca_clusters_k{k}.png",
                dpi=dpi,
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(method_dir / "metrics_by_k.csv", index=False)
    atomic_write_parquet(metrics, method_dir / "metrics_by_k.parquet")
    _plot_metric_curves(metrics, method, fig_dir, dpi)

    best_k = int(best_sil[1]) if best_sil[1] is not None else int(metrics.iloc[0]["k"])
    best_labels = best_sil[2]
    if best_labels is not None and best_k not in figure_ks:
        _plot_pca_clusters(
            xy2,
            best_labels,
            title=f"L2→PCA · {method} (best silhouette k={best_k})",
            out_path=fig_dir / f"pca_clusters_best_sil_k{best_k}.png",
            dpi=dpi,
        )

    summary = {
        "method": method,
        "k_list": [int(r["k"]) for r in rows],
        "best_silhouette_k": best_k,
        "best_silhouette": float(best_sil[0]) if np.isfinite(best_sil[0]) else None,
        "metrics_csv": str(method_dir / "metrics_by_k.csv"),
        "assignments_dir": str(assign_dir),
        "figures_dir": str(fig_dir),
    }
    atomic_write_json(method_dir / "summary.json", summary)
    return summary


def _run_hdbscan(
    X: np.ndarray,
    xy2: np.ndarray,
    ids: list[str],
    *,
    out_dir: Path,
    figures: Path,
    seed: int,
    run_id: str,
    dpi: int,
    min_cluster_size: int,
) -> dict[str, Any]:
    method_dir = ensure_dir(out_dir / "hdbscan")
    fig_dir = ensure_dir(figures / "hdbscan")
    if hdbscan is None:
        summary = {"method": "hdbscan", "skipped": True, "error": str(_HDBSCAN_ERR)}
        atomic_write_json(method_dir / "summary.json", summary)
        print(f"[cluster_exp] HDBSCAN skipped: {_HDBSCAN_ERR}")
        return summary

    print(f"[cluster_exp] HDBSCAN min_cluster_size={min_cluster_size} …")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        core_dist_n_jobs=8,
    )
    labels = clusterer.fit_predict(X).astype(np.int32)
    stats = _size_stats(labels)
    rng = np.random.default_rng(seed)
    sil = _silhouette_sample(X, labels, metric="euclidean", rng=rng)

    assign = stamp_run_id(
        pd.DataFrame(
            {
                "image_id": ids,
                "cluster": labels.astype(int),
                "is_noise": labels < 0,
                "method": "hdbscan",
            }
        ),
        run_id,
    )
    atomic_write_parquet(assign, method_dir / "assignments.parquet")
    assign.to_csv(method_dir / "assignments.csv", index=False)

    _plot_pca_clusters(
        xy2,
        labels,
        title=f"L2→PCA · HDBSCAN (clusters={stats['n_clusters']}, noise={stats['n_noise']})",
        out_path=fig_dir / "pca_clusters.png",
        dpi=dpi,
    )

    summary = {
        "method": "hdbscan",
        "skipped": False,
        "min_cluster_size": int(min_cluster_size),
        "silhouette": sil,
        **{k: v for k, v in stats.items() if k != "cluster_sizes"},
        "cluster_sizes": stats["cluster_sizes"],
    }
    atomic_write_json(method_dir / "summary.json", summary)
    return summary


def _run_leiden(
    X: np.ndarray,
    xy2: np.ndarray,
    ids: list[str],
    *,
    out_dir: Path,
    figures: Path,
    seed: int,
    run_id: str,
    dpi: int,
    knn_k: int = 15,
    resolution: float = 1.0,
) -> dict[str, Any]:
    method_dir = ensure_dir(out_dir / "leiden")
    fig_dir = ensure_dir(figures / "leiden")
    if leidenalg is None or ig is None:
        summary = {"method": "leiden", "skipped": True, "error": str(_LEIDEN_ERR)}
        atomic_write_json(method_dir / "summary.json", summary)
        print(f"[cluster_exp] Leiden skipped: {_LEIDEN_ERR}")
        return summary

    print(f"[cluster_exp] Leiden knn={knn_k} resolution={resolution} …")
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, n), metric="euclidean", n_jobs=8)
    nn.fit(X)
    dists, inds = nn.kneighbors(X)

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(n):
        for j_pos in range(1, inds.shape[1]):
            j = int(inds[i, j_pos])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.append((a, b))
            # similarity weight from distance
            weights.append(float(1.0 / (1e-6 + dists[i, j_pos])))

    # Deduplicate undirected edges (keep max weight).
    edge_w: dict[tuple[int, int], float] = {}
    for (a, b), w in zip(edges, weights):
        prev = edge_w.get((a, b))
        edge_w[(a, b)] = w if prev is None else max(prev, w)

    g = ig.Graph(n=n, edges=list(edge_w.keys()), directed=False)
    g.es["weight"] = list(edge_w.values())
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.array(partition.membership, dtype=np.int32)
    stats = _size_stats(labels)
    rng = np.random.default_rng(seed)
    sil = _silhouette_sample(X, labels, metric="euclidean", rng=rng)

    assign = stamp_run_id(
        pd.DataFrame({"image_id": ids, "cluster": labels.astype(int), "method": "leiden"}),
        run_id,
    )
    atomic_write_parquet(assign, method_dir / "assignments.parquet")
    assign.to_csv(method_dir / "assignments.csv", index=False)

    _plot_pca_clusters(
        xy2,
        labels,
        title=f"L2→PCA · Leiden (kNN={knn_k}, res={resolution}, clusters={stats['n_clusters']})",
        out_path=fig_dir / "pca_clusters.png",
        dpi=dpi,
    )

    summary = {
        "method": "leiden",
        "skipped": False,
        "knn_k": int(knn_k),
        "resolution": float(resolution),
        "silhouette": sil,
        "modularity": float(partition.modularity),
        **{k: v for k, v in stats.items() if k != "cluster_sizes"},
        "cluster_sizes": stats["cluster_sizes"],
    }
    atomic_write_json(method_dir / "summary.json", summary)
    return summary


def _run_one_pca_dim(
    X_l2: np.ndarray,
    ids: list[str],
    *,
    pca_dim: int,
    out_dir: Path,
    k_list: tuple[int, ...],
    figure_ks: set[int],
    seed: int,
    run_id: str,
    dpi: int = 150,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    figures = ensure_dir(out_dir / "figures")
    n, d = X_l2.shape
    pca_dim = min(pca_dim, n - 1, d)

    print(f"[cluster_exp] ===== PCA dim={pca_dim} =====")
    pca = PCA(n_components=pca_dim, random_state=seed)
    Z = pca.fit_transform(X_l2).astype(np.float64)
    # Spherical features: L2-renormalize PCA rows.
    Z_sph = (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    # Euclidean KMeans uses centered/scaled PCA (keep raw PCA; optional mild scale).
    Z_euc = Z.astype(np.float32)

    np.save(out_dir / f"features_pca{pca_dim}.npy", Z_euc)
    np.save(out_dir / f"features_pca{pca_dim}_spherical_l2.npy", Z_sph)
    atomic_write_parquet(
        stamp_run_id(
            pd.DataFrame({"image_id": ids, "pc1": Z[:, 0], "pc2": Z[:, 1], "pc3": Z[:, 2] if pca_dim > 2 else 0.0}),
            run_id,
        ),
        out_dir / "pca_coordinates.parquet",
    )

    cum = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#1f4e79")
    ax.axhline(0.95, color="#9a031e", linestyle="--", linewidth=1, label="95%")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(f"L2 → PCA-{pca_dim} cumulative variance (total={cum[-1]:.4f})")
    ax.legend()
    _save(fig, figures / "pca_variance.png", dpi)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.4, linewidths=0, c="#4a4a4a")
    ax.set_title(f"L2 → PCA-{pca_dim}: PC1–PC2 (unlabeled)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _save(fig, figures / "pca_scatter_unlabeled.png", dpi)

    xy2 = Z[:, :2]
    mcs = max(20, int(round(0.005 * n)))

    kmeans_sum = _run_k_sweep(
        Z_euc,
        xy2,
        ids,
        method="kmeans",
        metric_for_sil="euclidean",
        out_dir=out_dir,
        figures=figures,
        k_list=k_list,
        figure_ks=figure_ks,
        seed=seed,
        run_id=run_id,
        dpi=dpi,
    )
    sph_sum = _run_k_sweep(
        Z_sph,
        xy2,
        ids,
        method="spherical_kmeans",
        metric_for_sil="cosine",
        out_dir=out_dir,
        figures=figures,
        k_list=k_list,
        figure_ks=figure_ks,
        seed=seed,
        run_id=run_id,
        dpi=dpi,
    )
    hdb_sum = _run_hdbscan(
        Z_euc,
        xy2,
        ids,
        out_dir=out_dir,
        figures=figures,
        seed=seed,
        run_id=run_id,
        dpi=dpi,
        min_cluster_size=mcs,
    )
    leiden_sum = _run_leiden(
        Z_euc,
        xy2,
        ids,
        out_dir=out_dir,
        figures=figures,
        seed=seed,
        run_id=run_id,
        dpi=dpi,
        knn_k=15,
        resolution=1.0,
    )

    summary = {
        "feature": f"L2 embedding → PCA-{pca_dim}",
        "pca_dim": int(pca_dim),
        "n": int(n),
        "input_dim": int(d),
        "explained_variance_total": float(cum[-1]),
        "explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
        "methods": {
            "kmeans": kmeans_sum,
            "spherical_kmeans": sph_sum,
            "hdbscan": hdb_sum,
            "leiden": leiden_sum,
        },
        "seed": int(seed),
        "run_id": run_id,
        "output_dir": str(out_dir),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    print(f"[cluster_exp] wrote {out_dir}")
    return summary


def _run_full_dim_spherical(
    X_l2: np.ndarray,
    ids: list[str],
    *,
    out_dir: Path,
    k_list: tuple[int, ...],
    figure_ks: set[int],
    seed: int,
    run_id: str,
    dpi: int = 150,
) -> dict[str, Any]:
    """Spherical KMeans on full L2 embeddings (no PCA reduction); layout matches 64/128."""
    ensure_dir(out_dir)
    figures = ensure_dir(out_dir / "figures")
    n, d = X_l2.shape
    print(f"[cluster_exp] ===== full dim={d} spherical_kmeans =====")

    X_sph = (X_l2 / (np.linalg.norm(X_l2, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    np.save(out_dir / f"features_{d}_spherical_l2.npy", X_sph)
    np.save(out_dir / f"features_{d}_l2.npy", X_sph)

    # 2D coords for figures only (same as PCA dirs).
    pca2 = PCA(n_components=2, random_state=seed)
    xy2 = pca2.fit_transform(X_sph)
    atomic_write_parquet(
        stamp_run_id(
            pd.DataFrame({"image_id": ids, "pc1": xy2[:, 0], "pc2": xy2[:, 1]}),
            run_id,
        ),
        out_dir / "pca_coordinates.parquet",
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(xy2[:, 0], xy2[:, 1], s=4, alpha=0.4, linewidths=0, c="#4a4a4a")
    ax.set_title(f"L2 {d}-D → PCA-2 (viz only, unlabeled)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _save(fig, figures / "pca_scatter_unlabeled.png", dpi)

    sph_sum = _run_k_sweep(
        X_sph,
        xy2,
        ids,
        method="spherical_kmeans",
        metric_for_sil="cosine",
        out_dir=out_dir,
        figures=figures,
        k_list=k_list,
        figure_ks=figure_ks,
        seed=seed,
        run_id=run_id,
        dpi=dpi,
    )
    # Patch figure titles to say full-dim rather than L2→PCA.
    summary = {
        "feature": f"L2 embedding ({d}-D, no PCA)",
        "pca_dim": None,
        "dim": int(d),
        "n": int(n),
        "input_dim": int(d),
        "explained_variance_total": None,
        "methods": {"spherical_kmeans": sph_sum},
        "seed": int(seed),
        "run_id": run_id,
        "output_dir": str(out_dir),
    }
    atomic_write_json(out_dir / "summary.json", summary)
    print(f"[cluster_exp] wrote {out_dir}")
    return summary


def run_pca_dim_cluster_exp(
    *,
    embedding_path: Path,
    index_path: Path,
    output_root: Path,
    pca_dims: tuple[int, ...] = (64, 128),
    k_list: tuple[int, ...] = DEFAULT_K_LIST,
    figure_ks: tuple[int, ...] = DEFAULT_FIGURE_K,
    seed: int = 42,
    run_id: str = "all_benchmark_v1",
    include_full_dim_spherical: bool = False,
) -> dict[str, Any]:
    output_root = ensure_dir(output_root)
    print(f"[cluster_exp] load {embedding_path}")
    X = np.load(embedding_path).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    idx = pd.read_parquet(index_path).sort_values("embedding_row").reset_index(drop=True)
    if len(idx) != X.shape[0]:
        raise RuntimeError(f"index n={len(idx)} != embedding n={X.shape[0]}")
    ids = idx["image_id"].astype(str).tolist()
    fig_set = set(int(k) for k in figure_ks)

    results = {}
    for dim in pca_dims:
        results[str(dim)] = _run_one_pca_dim(
            X,
            ids,
            pca_dim=int(dim),
            out_dir=output_root / str(dim),
            k_list=tuple(int(k) for k in k_list),
            figure_ks=fig_set,
            seed=seed,
            run_id=run_id,
        )

    if include_full_dim_spherical:
        d = int(X.shape[1])
        results[str(d)] = _run_full_dim_spherical(
            X,
            ids,
            out_dir=output_root / str(d),
            k_list=tuple(int(k) for k in k_list),
            figure_ks=fig_set,
            seed=seed,
            run_id=run_id,
        )

    # Merge with existing overview if present (preserve prior 64/128 entries).
    overview_path = output_root / "overview.json"
    prev: dict[str, Any] = {}
    if overview_path.is_file():
        try:
            prev = json.loads(overview_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}

    prev_results = dict(prev.get("results") or {})
    for k, v in results.items():
        if v.get("pca_dim") is None:
            prev_results[k] = {
                "explained_variance_total": None,
                "spherical_best_sil_k": v["methods"]["spherical_kmeans"].get("best_silhouette_k"),
                "methods": ["spherical_kmeans"],
            }
        else:
            prev_results[k] = {
                "explained_variance_total": v["explained_variance_total"],
                "kmeans_best_sil_k": v["methods"]["kmeans"].get("best_silhouette_k"),
                "spherical_best_sil_k": v["methods"]["spherical_kmeans"].get("best_silhouette_k"),
                "hdbscan_n_clusters": v["methods"]["hdbscan"].get("n_clusters"),
                "leiden_n_clusters": v["methods"]["leiden"].get("n_clusters"),
            }

    pca_dims_out = list(prev.get("pca_dims") or [])
    for dim in pca_dims:
        if int(dim) not in pca_dims_out:
            pca_dims_out.append(int(dim))
    if include_full_dim_spherical and int(X.shape[1]) not in pca_dims_out:
        # Keep pca_dims for PCA-only; track full dim separately.
        pass

    overview = {
        "embedding_path": str(embedding_path),
        "index_path": str(index_path),
        "output_root": str(output_root),
        "pca_dims": pca_dims_out or list(pca_dims),
        "full_dim_spherical": int(X.shape[1]) if include_full_dim_spherical else prev.get("full_dim_spherical"),
        "k_list": list(k_list) if k_list else prev.get("k_list"),
        "figure_ks": list(figure_ks) if figure_ks else prev.get("figure_ks"),
        "methods": prev.get("methods")
        or ["kmeans", "spherical_kmeans", "hdbscan", "leiden"],
        "n": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "results": prev_results,
    }
    atomic_write_json(overview_path, overview)
    return overview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="L2+PCA multi-method clustering: KMeans / Spherical / HDBSCAN / Leiden"
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/embeddings/deepseek_ocr2_mean_l2.npy"
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/metadata/embedding_index.parquet"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp"),
    )
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument(
        "--full-dim-spherical-only",
        action="store_true",
        help="Only run spherical KMeans on full 1280-D (skip PCA dims); write to output_root/1280/",
    )
    parser.add_argument("--k-list", type=int, nargs="+", default=list(DEFAULT_K_LIST))
    parser.add_argument("--figure-ks", type=int, nargs="+", default=list(DEFAULT_FIGURE_K))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default="all_benchmark_v1")
    args = parser.parse_args(argv)

    if args.full_dim_spherical_only:
        run_pca_dim_cluster_exp(
            embedding_path=args.embedding,
            index_path=args.index,
            output_root=args.output_root,
            pca_dims=tuple(),
            k_list=tuple(args.k_list),
            figure_ks=tuple(args.figure_ks),
            seed=args.seed,
            run_id=args.run_id,
            include_full_dim_spherical=True,
        )
    else:
        run_pca_dim_cluster_exp(
            embedding_path=args.embedding,
            index_path=args.index,
            output_root=args.output_root,
            pca_dims=tuple(args.pca_dims),
            k_list=tuple(args.k_list),
            figure_ks=tuple(args.figure_ks),
            seed=args.seed,
            run_id=args.run_id,
            include_full_dim_spherical=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
