"""Unit tests for fusion identity, PCA pipeline, and spherical K-means (no GPU required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.equal_fusion_cluster.fusion import equal_weight_fuse, verify_equal_weight_identity
from src.equal_fusion_cluster.pca_fuse import (
    load_pca_pickle,
    prepare_group_pca_fusion,
    save_group_pca_artifacts,
    transform_with_saved_pca,
)
from src.equal_fusion_cluster.spherical_kmeans import (
    cosine_silhouette,
    elbow_select_k,
    spherical_kmeans,
    sweep_spherical_kmeans_elbow,
)


def test_equal_weight_identity():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(50, 1792)).astype(np.float32)
    T = rng.normal(size=(50, 1024)).astype(np.float32)
    x, t, z = equal_weight_fuse(V, T)
    assert z.shape == (50, 2816)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)
    r = verify_equal_weight_identity(x, t, z, n_pairs=100, seed=1)
    assert r["passed"]


def test_per_group_pca_fusion_and_artifacts(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 80
    V = rng.normal(size=(n, 1792)).astype(np.float32)
    T = rng.normal(size=(n, 1024)).astype(np.float32)
    ids = [f"s{i:03d}" for i in range(n)]
    prep = prepare_group_pca_fusion(V, T, group_id="g/q", sample_ids=ids, max_components=64)
    assert prep.n_components == 64
    assert prep.z.shape == (n, 128)
    assert np.allclose(np.linalg.norm(prep.x_prime, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(prep.t_prime, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(prep.z, axis=1), 1.0, atol=1e-5)
    idc = verify_equal_weight_identity(prep.x_prime, prep.t_prime, prep.z, n_pairs=50, seed=0)
    assert idc["passed"]

    save_group_pca_artifacts(prep, tmp_path)
    for name in (
        "image_pca.pkl",
        "text_pca.pkl",
        "image_pca_explained_variance.json",
        "text_pca_explained_variance.json",
        "reduced_image_features.npy",
        "reduced_text_features.npy",
        "fused_features.npy",
    ):
        assert (tmp_path / name).exists()

    blob = load_pca_pickle(tmp_path / "image_pca.pkl")
    for key in (
        "mean_",
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "input_dim",
        "output_dim",
        "whiten",
        "sample_ids",
        "config",
    ):
        assert key in blob
    assert blob["whiten"] is False
    assert blob["input_dim"] == 1792 and blob["output_dim"] == 64
    assert blob["sample_ids"] == ids

    scores = transform_with_saved_pca(blob, prep.x_l2.astype(np.float64))
    assert np.allclose(scores, prep.x_pca, atol=1e-4)


def test_spherical_kmeans_recovers_clusters():
    rng = np.random.default_rng(1)
    bases = np.eye(4, 32, dtype=np.float32)
    bases = bases / np.linalg.norm(bases, axis=1, keepdims=True)
    zs, truth = [], []
    for c in range(4):
        for _ in range(25):
            v = bases[c] + rng.normal(0, 0.03, size=32).astype(np.float32)
            v = v / np.linalg.norm(v)
            zs.append(v)
            truth.append(c)
    z = np.stack(zs)
    ids = [f"i{i}" for i in range(len(zs))]
    res = spherical_kmeans(z, 4, sample_ids=ids, n_init=5, seed=7)
    assert res.centroids.shape == (4, 32)
    assert abs(np.linalg.norm(res.centroids, axis=1) - 1).max() < 1e-4
    pur = 0
    for c in range(4):
        mem = [truth[i] for i in range(len(ids)) if res.labels[i] == c]
        pur += max(mem.count(x) for x in set(mem))
    assert pur / len(ids) > 0.85


def test_elbow_selects_clear_knee():
    # Distortion drops sharply until K=4 then flattens; interior elbow near 4
    ks = list(range(2, 11))
    distortions = [10.0, 6.0, 3.5, 2.2, 2.0, 1.9, 1.85, 1.82, 1.8]
    k, meta = elbow_select_k(ks, distortions)
    assert k in {3, 4, 5}
    assert meta["method"] == "max_perpendicular_distance_to_chord"
    assert meta["interior_only"] is True
    # endpoints must not be selected when interior exists
    assert k not in (2, 10)


def test_cosine_silhouette_all_points():
    rng = np.random.default_rng(2)
    bases = np.eye(3, 16, dtype=np.float32)
    bases = bases / np.linalg.norm(bases, axis=1, keepdims=True)
    zs, labels = [], []
    for c in range(3):
        for _ in range(20):
            v = bases[c] + rng.normal(0, 0.02, size=16).astype(np.float32)
            v = v / np.linalg.norm(v)
            zs.append(v)
            labels.append(c)
    z = np.stack(zs)
    lab = np.asarray(labels, dtype=np.int64)
    mean, per = cosine_silhouette(z, lab, return_per_sample=True)
    assert per.shape == (60,)
    assert mean is not None and mean > 0.5


def test_sweep_elbow_range_clamped():
    rng = np.random.default_rng(3)
    bases = np.eye(3, 16, dtype=np.float32)
    bases /= np.linalg.norm(bases, axis=1, keepdims=True)
    zs = []
    for c in range(3):
        for _ in range(15):
            v = bases[c] + rng.normal(0, 0.03, size=16).astype(np.float32)
            zs.append(v / np.linalg.norm(v))
    z = np.stack(zs).astype(np.float32)
    ids = [f"s{i}" for i in range(len(zs))]
    out = sweep_spherical_kmeans_elbow(z, ids, k_min=2, k_max=30, n_init=3, seed=0)
    assert out["k_min"] == 2
    # N=45 → k_max=min(30,44)=30, curve len=29
    assert out["k_max"] == 30
    assert len(out["curve"]) == 29
    assert out["selected_k"] in {c["k"] for c in out["curve"]}
    assert out["result"].centroids.shape[0] == out["selected_k"]
    assert all(c.get("n_init") == 3 for c in out["curve"])
    assert all(len(out["results_by_k"][c["k"]].n_init_trials) == 3 for c in out["curve"])
    assert "spherical_kmeans_total_cosine_distortion" in out["curve"][0]
    sil = cosine_silhouette(z, out["result"].labels, return_per_sample=True)
    assert sil is not None
    mean, per = sil
    assert per.shape[0] == len(ids)
    assert isinstance(mean, float)
    assert out["silhouette_role"] == "diagnostic_only"
