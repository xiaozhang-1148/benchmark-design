"""Tests for clustering reproducibility and template separation."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from heatmap_analysis.clustering import bootstrap_stability, evaluate_kmeans_k


def test_clustering_reproducible_with_seed():
    rng = np.random.default_rng(42)
    X = rng.random((30, 50))
    l1 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    l2 = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
    assert np.array_equal(l1, l2)


def test_kmeans_evaluation_returns_metrics():
    rng = np.random.default_rng(0)
    X = rng.random((25, 20))
    df = evaluate_kmeans_k(X, range(2, 5), seed=42)
    assert not df.empty
    assert "silhouette" in df.columns


def test_bootstrap_stability_runs():
    rng = np.random.default_rng(0)
    X = rng.random((40, 15))
    result = bootstrap_stability(X, k=3, seed=42, n_iter=5)
    assert "mean_ari" in result
