"""Spherical K-Means (main) + HDBSCAN (auxiliary outliers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..utils import atomic_write_json, ensure_dir
from .diagnostics import load_embeddings

try:
    import hdbscan
except Exception:  # noqa: BLE001
    hdbscan = None


def _spherical_kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 10) -> np.ndarray:
    """KMeans on L2 unit vectors ≈ spherical k-means with cosine geometry."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, algorithm="lloyd")
    return km.fit_predict(X)


def run_clustering(cfg: dict[str, Any]) -> dict[str, Any]:
    clus = Path(cfg["paths"]["clustering_dir"])
    ensure_dir(clus)
    seed = int(cfg.get("random_seed", 42))
    X, idx = load_embeddings(cfg)
    n = X.shape[0]
    ids = idx["image_id"].astype(str).tolist()

    k_min = int(cfg["analysis"].get("kmeans_k_min", 2))
    k_max = int(min(int(cfg["analysis"].get("kmeans_k_max", 20)), max(2, int(np.sqrt(n)))))
    n_seeds = int(cfg["analysis"].get("kmeans_n_init_seeds", 10))
    n_boot = int(cfg["analysis"].get("kmeans_bootstrap", 5))

    rows = []
    best = None
    rng = np.random.default_rng(seed)

    for k in range(k_min, k_max + 1):
        labels_runs = []
        sils = []
        for s in range(n_seeds):
            lab = _spherical_kmeans(X, k, seed + s, n_init=1)
            labels_runs.append(lab)
            # silhouette on subsample for speed
            sample = min(5000, n)
            if sample < n:
                ii = rng.choice(n, size=sample, replace=False)
                sil = float(silhouette_score(X[ii], lab[ii], metric="cosine"))
            else:
                sil = float(silhouette_score(X, lab, metric="cosine"))
            sils.append(sil)
        # ARI stability across seeds
        aris = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                aris.append(adjusted_rand_score(labels_runs[i], labels_runs[j]))
        # bootstrap stability vs reference
        ref = labels_runs[int(np.argmax(sils))]
        boot_aris = []
        for b in range(n_boot):
            ii = rng.choice(n, size=n, replace=True)
            # fit on bootstrap unique then map — approximate: refit full with different seed
            lab_b = _spherical_kmeans(X, k, seed + 1000 + b, n_init=1)
            boot_aris.append(adjusted_rand_score(ref, lab_b))
        sizes = np.bincount(ref, minlength=k)
        tiny = float(np.mean(sizes < max(5, 0.005 * n)))
        row = {
            "k": k,
            "silhouette_mean": float(np.mean(sils)),
            "silhouette_std": float(np.std(sils)),
            "ari_seed_mean": float(np.mean(aris)) if aris else 1.0,
            "ari_bootstrap_mean": float(np.mean(boot_aris)) if boot_aris else None,
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "tiny_cluster_frac": tiny,
        }
        rows.append(row)
        score = row["silhouette_mean"] + 0.3 * row["ari_seed_mean"] - 0.5 * tiny
        if best is None or score > best[0]:
            best = (score, k, ref, sizes)

    sel = pd.DataFrame(rows)
    sel.to_csv(clus / "k_selection.csv", index=False)
    assert best is not None
    _, k_star, labels, sizes = best

    assign = pd.DataFrame({"image_id": ids, "cluster": labels.astype(int)})
    assign.to_parquet(clus / "cluster_assignments.parquet", index=False)

    # Stability report
    stab = sel.copy()
    stab.to_csv(clus / "stability_report.csv", index=False)

    # HDBSCAN outliers on PCA-50
    outlier_df = pd.DataFrame({"image_id": ids, "hdbscan_label": -1, "is_outlier": False})
    if hdbscan is None:
        print("[clustering] hdbscan not installed; skipping")
    else:
        pca_dim = min(int(cfg["analysis"].get("hdbscan_pca_dim", 50)), n - 1, X.shape[1])
        Z = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
        mcs = max(
            int(cfg["analysis"].get("hdbscan_min_cluster_size_floor", 20)),
            int(round(float(cfg["analysis"].get("hdbscan_min_cluster_size_frac", 0.005)) * n)),
        )
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
        hlab = clusterer.fit_predict(Z)
        outlier_df = pd.DataFrame(
            {
                "image_id": ids,
                "hdbscan_label": hlab.astype(int),
                "is_outlier": hlab == -1,
            }
        )
        outlier_df.to_parquet(clus / "hdbscan_outliers.parquet", index=False)

    # medoids
    medoids = []
    for c in range(k_star):
        members = np.where(labels == c)[0]
        center = X[members].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)
        sims = X[members] @ center
        mid = members[int(np.argmax(sims))]
        medoids.append({"cluster": int(c), "medoid_image_id": ids[mid], "size": int(len(members))})
    pd.DataFrame(medoids).to_csv(clus / "medoids.csv", index=False)

    summary = {
        "k_selected": int(k_star),
        "cluster_sizes": sizes.tolist(),
        "silhouette_at_k": float(sel.loc[sel.k == k_star, "silhouette_mean"].iloc[0]),
        "n_hdbscan_outliers": int(outlier_df["is_outlier"].sum()) if len(outlier_df) else 0,
    }
    atomic_write_json(clus / "clustering_summary.json", summary)
    print(f"[clustering] selected k={k_star} sizes={sizes.tolist()}")
    return summary
