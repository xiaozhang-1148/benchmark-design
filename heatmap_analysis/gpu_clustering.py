"""GPU-accelerated clustering (PCA + KMeans via CuPy)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from heatmap_analysis.gpu import get_xp, is_gpu_available, to_numpy

logger = logging.getLogger("heatmap_analysis.gpu_clustering")


def _standardize_xp(X, xp):
    mean = xp.mean(X, axis=0)
    std = xp.std(X, axis=0)
    std = xp.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std


def pca_reduce(X: np.ndarray, variance: float, n_components: int | None, seed: int, use_gpu: bool):
    """PCA via SVD; returns (X_reduced numpy, explained_variance_ratio sum, n_comp)."""
    xp = get_xp(use_gpu and is_gpu_available())
    Xg = xp.asarray(X, dtype=xp.float64)
    Xs, _, _ = _standardize_xp(Xg, xp)

    # SVD: Xs = U S Vt
    U, S, Vt = xp.linalg.svd(Xs, full_matrices=False)
    n = X.shape[0]
    eigvar = (S ** 2) / max(n - 1, 1)
    total = float(xp.sum(eigvar))
    if total <= 0:
        return X[:, : min(2, X.shape[1])], 0.0, min(2, X.shape[1])

    ratio = to_numpy(eigvar / total)
    cum = np.cumsum(ratio)
    if n_components is not None:
        k = min(n_components, len(S))
    else:
        k = int(np.searchsorted(cum, variance) + 1)
        k = max(2, min(k, len(S)))

    X_red = U[:, :k] * S[:k]
    cumvar = float(np.sum(ratio[:k]))
    return to_numpy(X_red), cumvar, k


def kmeans_predict(X: np.ndarray, k: int, seed: int, use_gpu: bool, n_init: int = 10) -> np.ndarray:
    """K-means labels only."""
    labels, _ = kmeans_fit(X, k, seed, use_gpu, n_init=n_init)
    return labels


def kmeans_fit(
    X: np.ndarray,
    k: int,
    seed: int,
    use_gpu: bool,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """K-means with best-of-n_init restarts; returns (labels, centroids)."""
    xp = get_xp(use_gpu and is_gpu_available())
    Xg = xp.asarray(X, dtype=xp.float64)
    n, d = Xg.shape
    rng = np.random.default_rng(seed)

    best_inertia = xp.inf
    best_labels = xp.zeros(n, dtype=xp.int32)
    best_centroids = xp.zeros((k, d), dtype=xp.float64)

    for init_i in range(n_init):
        idx = rng.choice(n, size=k, replace=False)
        centroids = Xg[idx].copy()

        for _ in range(300):
            dists = xp.sum((Xg[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = xp.argmin(dists, axis=1)

            H = xp.zeros((n, k), dtype=xp.float64)
            H[xp.arange(n), labels] = 1.0
            counts = H.sum(axis=0)
            new_centroids = (H.T @ Xg) / xp.maximum(counts[:, None], 1.0)

            empty = counts < 0.5
            if xp.any(empty):
                empty_idx = to_numpy(xp.flatnonzero(empty))
                for ei in empty_idx:
                    new_centroids[ei] = Xg[rng.integers(0, n)]

            shift = float(xp.max(xp.linalg.norm(new_centroids - centroids, axis=1)))
            centroids = new_centroids
            if shift < 1e-6:
                break

        dists = xp.sum((Xg[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        inertia = float(xp.sum(xp.min(dists, axis=1)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return to_numpy(best_labels).astype(np.int32), to_numpy(best_centroids)


def evaluate_kmeans_k_gpu(X: np.ndarray, k_range: range, seed: int, use_gpu: bool) -> pd.DataFrame:
    rows = []
    for k in k_range:
        labels = kmeans_predict(X, k, seed, use_gpu, n_init=5)
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
