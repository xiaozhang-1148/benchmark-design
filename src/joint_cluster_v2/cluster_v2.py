"""K-Means helpers for V2 fixed-K and aux scans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_kmeans(
    X: np.ndarray,
    k: int,
    *,
    seed: int = 42,
    n_init: int = 50,
    init: str = "k-means++",
) -> tuple[KMeans, np.ndarray, dict[str, Any]]:
    km = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=seed, algorithm="lloyd")
    lab = km.fit_predict(X)
    n = X.shape[0]
    sample_n = min(5000, n)
    if sample_n < n:
        rng = np.random.default_rng(seed)
        ii = rng.choice(n, size=sample_n, replace=False)
        sil = float(silhouette_score(X[ii], lab[ii], metric="euclidean"))
    else:
        sil = float(silhouette_score(X, lab, metric="euclidean"))
    sizes = np.bincount(lab, minlength=k)
    meta = {
        "k": k,
        "n_samples": n,
        "inertia": float(km.inertia_),
        "silhouette": sil,
        "n_iter": int(km.n_iter_),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "cluster_sizes": sizes.tolist(),
    }
    return km, lab.astype(int), meta


def distances_to_centers(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - centers[labels], axis=1)
