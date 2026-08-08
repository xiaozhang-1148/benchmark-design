"""Multi-method clustering on L2 embeddings: KMeans, GMM, HDBSCAN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from ..utils import atomic_write_json, ensure_dir
from .io_util import (
    assert_same_id_set,
    atomic_write_parquet,
    load_aligned_embeddings,
    stamp_run_id,
    write_run_meta,
)

try:
    import hdbscan
except Exception:  # noqa: BLE001
    hdbscan = None


def _spherical_kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 10) -> np.ndarray:
    """KMeans on L2 unit vectors ≈ spherical k-means with cosine geometry."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, algorithm="lloyd")
    return km.fit_predict(X)


def _silhouette_sample(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str,
    rng: np.random.Generator,
    sample: int = 5000,
) -> float:
    n = X.shape[0]
    if len(set(labels.tolist())) < 2:
        return float("nan")
    if sample < n:
        ii = rng.choice(n, size=sample, replace=False)
        return float(silhouette_score(X[ii], labels[ii], metric=metric))
    return float(silhouette_score(X, labels, metric=metric))


def _cluster_size_stats(labels: np.ndarray, k: int, n: int) -> dict[str, Any]:
    sizes = np.bincount(labels.astype(int), minlength=k)
    tiny_thr = max(5, int(0.005 * n))
    return {
        "cluster_sizes": sizes.tolist(),
        "min_cluster_size": int(sizes.min()) if sizes.size else 0,
        "max_cluster_size": int(sizes.max()) if sizes.size else 0,
        "tiny_cluster_frac": float(np.mean(sizes < tiny_thr)) if sizes.size else 0.0,
        "n_clusters": int(k),
    }


def _select_kmeans(
    X: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    n_seeds: int,
    n_boot: int,
    seed: int,
    run_id: str,
) -> tuple[pd.DataFrame, int, np.ndarray, np.ndarray]:
    n = X.shape[0]
    rows: list[dict[str, Any]] = []
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    rng = np.random.default_rng(seed)

    for k in range(k_min, k_max + 1):
        labels_runs: list[np.ndarray] = []
        sils: list[float] = []
        for s in range(n_seeds):
            lab = _spherical_kmeans(X, k, seed + s, n_init=1)
            labels_runs.append(lab)
            sils.append(_silhouette_sample(X, lab, metric="cosine", rng=rng))
        aris = [
            adjusted_rand_score(labels_runs[i], labels_runs[j])
            for i in range(len(labels_runs))
            for j in range(i + 1, len(labels_runs))
        ]
        ref = labels_runs[int(np.nanargmax(sils))]
        boot_aris = [
            adjusted_rand_score(ref, _spherical_kmeans(X, k, seed + 1000 + b, n_init=1))
            for b in range(n_boot)
        ]
        size_stats = _cluster_size_stats(ref, k, n)
        row = {
            "method": "kmeans",
            "k": k,
            "silhouette_mean": float(np.nanmean(sils)),
            "silhouette_std": float(np.nanstd(sils)),
            "ari_seed_mean": float(np.mean(aris)) if aris else 1.0,
            "ari_bootstrap_mean": float(np.mean(boot_aris)) if boot_aris else None,
            **{kk: vv for kk, vv in size_stats.items() if kk != "cluster_sizes"},
            "run_id": run_id,
        }
        rows.append(row)
        score = row["silhouette_mean"] + 0.3 * row["ari_seed_mean"] - 0.5 * row["tiny_cluster_frac"]
        if best is None or score > best[0]:
            best = (score, k, ref, np.array(size_stats["cluster_sizes"], dtype=np.int64))

    assert best is not None
    _, k_star, labels, sizes = best
    return pd.DataFrame(rows), int(k_star), labels.astype(int), sizes


def _select_gmm(
    Z: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    seed: int,
    run_id: str,
) -> tuple[pd.DataFrame, int, np.ndarray, np.ndarray]:
    n = Z.shape[0]
    rows: list[dict[str, Any]] = []
    best: tuple[float, int, np.ndarray, np.ndarray] | None = None
    rng = np.random.default_rng(seed)

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=seed,
            n_init=3,
            max_iter=200,
            reg_covar=1e-5,
        )
        lab = gmm.fit_predict(Z)
        sil = _silhouette_sample(Z, lab, metric="euclidean", rng=rng)
        size_stats = _cluster_size_stats(lab, k, n)
        bic = float(gmm.bic(Z))
        row = {
            "method": "gmm",
            "k": k,
            "silhouette_mean": sil,
            "bic": bic,
            **{kk: vv for kk, vv in size_stats.items() if kk != "cluster_sizes"},
            "run_id": run_id,
        }
        rows.append(row)
        score = (sil if np.isfinite(sil) else -1.0) - 0.5 * row["tiny_cluster_frac"] - bic / (n * Z.shape[1])
        if best is None or score > best[0]:
            best = (score, k, lab, np.array(size_stats["cluster_sizes"], dtype=np.int64))

    assert best is not None
    _, k_star, labels, sizes = best
    return pd.DataFrame(rows), int(k_star), labels.astype(int), sizes


def _run_hdbscan(
    Z: np.ndarray,
    ids: list[str],
    *,
    mcs: int,
    seed: int,
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if hdbscan is None:
        print("[clustering] hdbscan not installed; skipping")
        df = stamp_run_id(
            pd.DataFrame({"image_id": ids, "hdbscan_label": -1, "is_outlier": False, "cluster": -1}),
            run_id,
        )
        return df, {"n_clusters": 0, "n_outliers": 0, "skipped": True}

    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean", core_dist_n_jobs=-1)
    hlab = clusterer.fit_predict(Z)
    n_clusters = int(len(set(hlab.tolist()) - {-1}))
    n_outliers = int(np.sum(hlab == -1))
    df = stamp_run_id(
        pd.DataFrame(
            {
                "image_id": ids,
                "hdbscan_label": hlab.astype(int),
                "cluster": hlab.astype(int),
                "is_outlier": hlab == -1,
            }
        ),
        run_id,
    )
    summary = {
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "min_cluster_size": int(mcs),
        "skipped": False,
        "seed": int(seed),
    }
    return df, summary


def run_clustering(cfg: dict[str, Any]) -> dict[str, Any]:
    clus = Path(cfg["paths"]["clustering_dir"])
    ensure_dir(clus)
    seed = int(cfg.get("random_seed", 42))
    X, idx, emb_sha = load_aligned_embeddings(cfg)
    n = X.shape[0]
    ids = idx["image_id"].astype(str).tolist()
    run_id = str(cfg["run_id"])

    k_min = int(cfg["analysis"].get("kmeans_k_min", 2))
    k_max = int(min(int(cfg["analysis"].get("kmeans_k_max", 20)), max(2, int(np.sqrt(n)))))
    n_seeds = int(cfg["analysis"].get("kmeans_n_init_seeds", 10))
    n_boot = int(cfg["analysis"].get("kmeans_bootstrap", 5))
    pca_dim = min(int(cfg["analysis"].get("hdbscan_pca_dim", 50)), n - 1, X.shape[1])

    print(f"[clustering] n={n} dim={X.shape[1]} k_range=[{k_min},{k_max}] pca_dim={pca_dim}")

    # Shared PCA space for GMM / HDBSCAN (high-dim GMM is unstable/slow).
    print("[clustering] fitting PCA …")
    Z = PCA(n_components=pca_dim, random_state=seed).fit_transform(X).astype(np.float64)

    # --- KMeans (spherical / cosine) ---
    print("[clustering] KMeans scan …")
    kmeans_sel, k_kmeans, lab_kmeans, sizes_kmeans = _select_kmeans(
        X,
        k_min=k_min,
        k_max=k_max,
        n_seeds=n_seeds,
        n_boot=n_boot,
        seed=seed,
        run_id=run_id,
    )
    kmeans_sel.to_csv(clus / "kmeans_k_selection.csv", index=False)
    # Legacy aliases expected by galleries / report.
    kmeans_sel.to_csv(clus / "k_selection.csv", index=False)
    kmeans_sel.to_csv(clus / "stability_report.csv", index=False)
    kmeans_assign = stamp_run_id(
        pd.DataFrame({"image_id": ids, "cluster": lab_kmeans.astype(int), "method": "kmeans"}),
        run_id,
    )
    assert_same_id_set("embeddings", ids, "kmeans_assignments", kmeans_assign["image_id"])
    atomic_write_parquet(kmeans_assign, clus / "kmeans_assignments.parquet")
    atomic_write_parquet(kmeans_assign, clus / "cluster_assignments.parquet")

    medoids = []
    for c in range(k_kmeans):
        members = np.where(lab_kmeans == c)[0]
        center = X[members].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)
        sims = X[members] @ center
        mid = members[int(np.argmax(sims))]
        medoids.append(
            {
                "method": "kmeans",
                "cluster": int(c),
                "medoid_image_id": ids[mid],
                "size": int(len(members)),
                "run_id": run_id,
            }
        )
    pd.DataFrame(medoids).to_csv(clus / "medoids.csv", index=False)
    print(f"[clustering] KMeans selected k={k_kmeans} sizes={sizes_kmeans.tolist()}")

    # --- GMM (diag cov on PCA) ---
    print("[clustering] GMM scan …")
    gmm_sel, k_gmm, lab_gmm, sizes_gmm = _select_gmm(
        Z,
        k_min=k_min,
        k_max=k_max,
        seed=seed,
        run_id=run_id,
    )
    gmm_sel.to_csv(clus / "gmm_k_selection.csv", index=False)
    gmm_assign = stamp_run_id(
        pd.DataFrame({"image_id": ids, "cluster": lab_gmm.astype(int), "method": "gmm"}),
        run_id,
    )
    atomic_write_parquet(gmm_assign, clus / "gmm_assignments.parquet")
    print(f"[clustering] GMM selected k={k_gmm} sizes={sizes_gmm.tolist()}")

    # --- HDBSCAN ---
    mcs = max(
        int(cfg["analysis"].get("hdbscan_min_cluster_size_floor", 20)),
        int(round(float(cfg["analysis"].get("hdbscan_min_cluster_size_frac", 0.005)) * n)),
    )
    print(f"[clustering] HDBSCAN min_cluster_size={mcs} …")
    hdb_df, hdb_summary = _run_hdbscan(Z, ids, mcs=mcs, seed=seed, run_id=run_id)
    atomic_write_parquet(hdb_df, clus / "hdbscan_assignments.parquet")
    atomic_write_parquet(hdb_df, clus / "hdbscan_outliers.parquet")
    print(
        f"[clustering] HDBSCAN clusters={hdb_summary.get('n_clusters')} "
        f"outliers={hdb_summary.get('n_outliers')}"
    )

    # --- Method agreement ---
    compare_rows = [
        {
            "method_a": "kmeans",
            "method_b": "gmm",
            "ari": float(adjusted_rand_score(lab_kmeans, lab_gmm)),
            "k_a": int(k_kmeans),
            "k_b": int(k_gmm),
        }
    ]
    if not hdb_summary.get("skipped") and int(hdb_summary.get("n_clusters", 0)) >= 1:
        hlab = hdb_df["hdbscan_label"].to_numpy(dtype=int)
        compare_rows.append(
            {
                "method_a": "kmeans",
                "method_b": "hdbscan",
                "ari": float(adjusted_rand_score(lab_kmeans, hlab)),
                "k_a": int(k_kmeans),
                "k_b": int(hdb_summary["n_clusters"]),
            }
        )
        compare_rows.append(
            {
                "method_a": "gmm",
                "method_b": "hdbscan",
                "ari": float(adjusted_rand_score(lab_gmm, hlab)),
                "k_a": int(k_gmm),
                "k_b": int(hdb_summary["n_clusters"]),
            }
        )
    pd.DataFrame(compare_rows).to_csv(clus / "method_agreement.csv", index=False)

    # Combined long-form assignments for downstream analysis.
    long_frames = [
        kmeans_assign.assign(method="kmeans"),
        gmm_assign.assign(method="gmm"),
        hdb_df.rename(columns={"hdbscan_label": "cluster"})[["image_id", "cluster", "run_id"]].assign(
            method="hdbscan"
        ),
    ]
    atomic_write_parquet(pd.concat(long_frames, ignore_index=True), clus / "all_method_assignments.parquet")

    summary = {
        "run_id": run_id,
        "embedding_sha256": emb_sha,
        "n": int(n),
        "dim": int(X.shape[1]),
        "pca_dim": int(pca_dim),
        "methods": {
            "kmeans": {
                "k_selected": int(k_kmeans),
                "cluster_sizes": sizes_kmeans.tolist(),
                "silhouette_at_k": float(
                    kmeans_sel.loc[kmeans_sel.k == k_kmeans, "silhouette_mean"].iloc[0]
                ),
                "assignments": "kmeans_assignments.parquet",
            },
            "gmm": {
                "k_selected": int(k_gmm),
                "cluster_sizes": sizes_gmm.tolist(),
                "silhouette_at_k": float(gmm_sel.loc[gmm_sel.k == k_gmm, "silhouette_mean"].iloc[0]),
                "bic_at_k": float(gmm_sel.loc[gmm_sel.k == k_gmm, "bic"].iloc[0]),
                "assignments": "gmm_assignments.parquet",
            },
            "hdbscan": {
                **hdb_summary,
                "assignments": "hdbscan_assignments.parquet",
            },
        },
        # Legacy top-level fields (galleries / report).
        "k_selected": int(k_kmeans),
        "cluster_sizes": sizes_kmeans.tolist(),
        "silhouette_at_k": float(kmeans_sel.loc[kmeans_sel.k == k_kmeans, "silhouette_mean"].iloc[0]),
        "n_hdbscan_outliers": int(hdb_summary.get("n_outliers", 0)),
    }
    atomic_write_json(clus / "clustering_summary.json", summary)
    write_run_meta(cfg, stage="clustering_done", **{
        "k_selected": summary["k_selected"],
        "n_hdbscan_outliers": summary["n_hdbscan_outliers"],
        "k_gmm": int(k_gmm),
        "n_hdbscan_clusters": int(hdb_summary.get("n_clusters", 0)),
        "embedding_sha256": emb_sha,
        "n": int(n),
    })
    print(f"[clustering] done → {clus}")

    try:
        from .cluster_figures import export_cluster_projection_figures

        export_cluster_projection_figures(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"[clustering] cluster projection figures failed: {exc}")

    return summary
