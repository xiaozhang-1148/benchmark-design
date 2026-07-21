"""Fast smoke tests for the heatmap extract + clustering data path."""

from __future__ import annotations

from heatmap_analysis.cluster_study import _feature_matrix, fit_pca, load_dataset
from heatmap_analysis.clustering import evaluate_kmeans_k
from heatmap_analysis.pipeline import extract_all


def test_synthetic_extract_and_kmeans_metrics(synthetic_heatmap_cfg) -> None:
    cfg = synthetic_heatmap_cfg
    extract_all(cfg)

    ids, abs_map, rel_map, _ = load_dataset(cfg)
    assert len(ids) >= cfg.clustering.min_samples_for_clustering
    assert len(list((cfg.cache_dir / "per_image").glob("*.npz"))) == len(ids)

    for grid_map in (abs_map, rel_map):
        X_pca, cumvar, n_comp = fit_pca(cfg, _feature_matrix(grid_map, ids))
        assert n_comp >= 1
        assert cumvar > 0.5

        metrics = evaluate_kmeans_k(X_pca, cfg.clustering.k_values, cfg.clustering.random_seed)
        assert not metrics.empty
        assert set(metrics["k"]) <= set(cfg.clustering.k_values)
