"""Tests for clustering reproducibility and template separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from heatmap_analysis.clustering import evaluate_kmeans_k, bootstrap_stability


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


def test_extra_k_centers_exported():
    from heatmap_analysis.config import load_config
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    if not cfg_path.exists():
        pytest.skip("config.example.yaml missing")
    cfg = load_config(cfg_path)
    assert 3 in cfg.clustering.extra_k_outputs
    assert 5 in cfg.clustering.extra_k_outputs
    out = cfg.output.output_dir / "clustering"
    if not out.exists():
        pytest.skip("run cluster on synthetic data first")
    k3 = list(out.rglob("k_fixed_03/cluster_centers.npz"))
    assert k3, "expected k=3 cluster centers"
    d = np.load(k3[0])
    assert d["centroids_pca"].shape[0] == 3
    assert d["centers_rel_mean"].shape[0] == 3
