"""PCA / UMAP figures colored by best clustering per method."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

from ..utils import atomic_write_json, ensure_dir
from .io_util import load_aligned_embeddings, stamp_run_id

try:
    import umap as umap_lib
except Exception as e:  # noqa: BLE001
    umap_lib = None
    _UMAP_ERR = e
else:
    _UMAP_ERR = None

# Distinct categorical palette (cluster 0..N); noise (-1) drawn in gray separately.
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


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_labels(clus: Path, method: str, ids: list[str]) -> tuple[np.ndarray, str]:
    """Return labels aligned to *ids* and a short title for the best result."""
    if method == "kmeans":
        path = clus / "kmeans_assignments.parquet"
        sel = clus / "kmeans_k_selection.csv"
        col = "cluster"
    elif method == "gmm":
        path = clus / "gmm_assignments.parquet"
        sel = clus / "gmm_k_selection.csv"
        col = "cluster"
    elif method == "hdbscan":
        path = clus / "hdbscan_assignments.parquet"
        sel = None
        col = "cluster"
    else:
        raise KeyError(method)

    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "cluster" not in df.columns and "hdbscan_label" in df.columns:
        col = "hdbscan_label"
    mapped = df.set_index("image_id")[col].reindex(ids)
    if mapped.isna().any():
        raise RuntimeError(f"{method}: missing labels for some image_ids")
    labels = mapped.to_numpy(dtype=int)

    if method == "hdbscan":
        n_clust = int(len(set(labels.tolist()) - {-1}))
        n_noise = int(np.sum(labels == -1))
        title = f"HDBSCAN (clusters={n_clust}, noise={n_noise})"
    else:
        k = int(pd.read_csv(sel).sort_values("silhouette_mean", ascending=False).iloc[0]["k"]) if sel and sel.is_file() else int(labels.max() + 1)
        # Prefer selected k from actual assignment cardinality when no noise.
        k_assign = int(len(set(labels.tolist()) - {-1}))
        title = f"{method.upper()} (best k={k_assign})"
        if sel and sel.is_file():
            row = pd.read_csv(sel)
            # match the assignment's k if present
            match = row[row["k"] == k_assign]
            if len(match):
                sil = float(match.iloc[0]["silhouette_mean"])
                title = f"{method.upper()} (best k={k_assign}, sil={sil:.3f})"
            else:
                _ = k  # keep lint quiet
    return labels, title


def _scatter_by_cluster(
    ax,
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    s: float = 4.0,
    alpha: float = 0.55,
) -> None:
    noise = labels == -1
    if noise.any():
        ax.scatter(
            xy[noise, 0],
            xy[noise, 1],
            s=s,
            c="#bdbdbd",
            alpha=0.35,
            linewidths=0,
            label="noise (-1)",
            zorder=1,
        )
    clusters = sorted(c for c in set(labels.tolist()) if c != -1)
    for i, c in enumerate(clusters):
        mask = labels == c
        color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=s,
            c=color,
            alpha=alpha,
            linewidths=0,
            label=f"c{c} (n={int(mask.sum())})",
            zorder=2,
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(clusters) <= 12:
        ax.legend(loc="best", fontsize=7, markerscale=2, frameon=True, framealpha=0.85)


def _compute_or_load_pca_umap(
    cfg: dict[str, Any],
    X: np.ndarray,
    ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (PCA_2d, UMAP_2d); cache under projections/."""
    proj = Path(cfg["paths"]["projections_dir"])
    ensure_dir(proj)
    seed = int(cfg.get("random_seed", 42))
    dpi = int(cfg["analysis"].get("figure_dpi", 150))
    run_id = str(cfg["run_id"])

    pca_path = proj / "pca_coordinates.parquet"
    umap_path = proj / "umap_coordinates.parquet"

    if pca_path.is_file():
        pca_df = pd.read_parquet(pca_path).set_index("image_id").reindex(ids)
        if pca_df[["pc1", "pc2"]].isna().any().any():
            raise RuntimeError("pca_coordinates missing rows")
        P = pca_df[["pc1", "pc2"]].to_numpy(dtype=np.float64)
    else:
        print("[cluster_figures] fitting PCA …")
        n_comp = min(int(cfg["analysis"].get("pca_n_components", 50)), X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=seed)
        Z = pca.fit_transform(X)
        P = Z[:, :2]
        out = stamp_run_id(
            pd.DataFrame({"image_id": ids, "pc1": P[:, 0], "pc2": P[:, 1], "pc3": Z[:, 2] if Z.shape[1] > 2 else 0.0}),
            run_id,
        )
        out.to_parquet(pca_path, index=False)
        # also write an unlabeled base scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(P[:, 0], P[:, 1], s=3, alpha=0.4, linewidths=0, c="#1f4e79")
        ax.set_title("PCA PC1–PC2 (unlabeled)")
        _save(fig, proj / "pca_scatter.png", dpi)

    if umap_path.is_file():
        umap_df = pd.read_parquet(umap_path).set_index("image_id").reindex(ids)
        if umap_df[["umap1", "umap2"]].isna().any().any():
            raise RuntimeError("umap_coordinates missing rows")
        U = umap_df[["umap1", "umap2"]].to_numpy(dtype=np.float64)
    else:
        if umap_lib is None:
            raise RuntimeError(f"umap-learn required: {_UMAP_ERR}")
        print("[cluster_figures] fitting UMAP …")
        reducer = umap_lib.UMAP(
            n_neighbors=int(cfg["analysis"].get("umap_n_neighbors", 15)),
            min_dist=float(cfg["analysis"].get("umap_min_dist", 0.1)),
            n_components=2,
            metric="cosine",
            random_state=int(cfg["analysis"].get("umap_random_state", 42)),
        )
        n = X.shape[0]
        max_fit = 30000
        rng = np.random.default_rng(seed)
        if n > max_fit:
            fit_idx = rng.choice(n, size=max_fit, replace=False)
            reducer.fit(X[fit_idx])
            U = reducer.transform(X)
        else:
            U = reducer.fit_transform(X)
        out = stamp_run_id(
            pd.DataFrame({"image_id": ids, "umap1": U[:, 0], "umap2": U[:, 1]}),
            run_id,
        )
        out.to_parquet(umap_path, index=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hexbin(U[:, 0], U[:, 1], gridsize=70, cmap="magma", mincnt=1)
        ax.set_title("UMAP density (unlabeled)")
        _save(fig, proj / "umap_scatter.png", dpi)

    return P, U


def export_cluster_projection_figures(cfg: dict[str, Any]) -> dict[str, Any]:
    """Color PCA/UMAP by each method's best clustering; write under clustering/figures/."""
    clus = Path(cfg["paths"]["clustering_dir"])
    fig_dir = ensure_dir(clus / "figures")
    dpi = int(cfg["analysis"].get("figure_dpi", 150))

    X, idx, emb_sha = load_aligned_embeddings(cfg)
    ids = idx["image_id"].astype(str).tolist()
    P, U = _compute_or_load_pca_umap(cfg, X, ids)

    methods = ("kmeans", "gmm", "hdbscan")
    panel_rows: list[tuple[str, np.ndarray, str]] = []
    written: list[str] = []

    for method in methods:
        path = clus / f"{method}_assignments.parquet"
        if not path.is_file():
            print(f"[cluster_figures] skip {method}: missing {path.name}")
            continue
        labels, title = _load_labels(clus, method, ids)
        panel_rows.append((method, labels, title))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        _scatter_by_cluster(axes[0], P, labels, title=f"PCA · {title}")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        _scatter_by_cluster(axes[1], U, labels, title=f"UMAP · {title}")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        fig.suptitle(f"Best {method.upper()} clustering on DeepSeek-OCR2 embeddings", fontsize=12)
        fig.tight_layout()
        out = fig_dir / f"{method}_best_pca_umap.png"
        _save(fig, out, dpi)
        written.append(str(out))
        print(f"[cluster_figures] wrote {out}")

    if panel_rows:
        n_m = len(panel_rows)
        fig, axes = plt.subplots(n_m, 2, figsize=(13, 4.2 * n_m))
        if n_m == 1:
            axes = np.array([axes])
        for r, (method, labels, title) in enumerate(panel_rows):
            _scatter_by_cluster(axes[r, 0], P, labels, title=f"PCA · {title}")
            axes[r, 0].set_xlabel("PC1")
            axes[r, 0].set_ylabel("PC2")
            _scatter_by_cluster(axes[r, 1], U, labels, title=f"UMAP · {title}")
            axes[r, 1].set_xlabel("UMAP1")
            axes[r, 1].set_ylabel("UMAP2")
        fig.suptitle("Best clustering per method (PCA / UMAP)", fontsize=13, y=1.01)
        fig.tight_layout()
        out = fig_dir / "all_methods_best_pca_umap.png"
        _save(fig, out, dpi)
        written.append(str(out))
        print(f"[cluster_figures] wrote {out}")

    summary = {
        "embedding_sha256": emb_sha,
        "n": int(X.shape[0]),
        "figures": written,
        "methods": [m for m, _, _ in panel_rows],
    }
    atomic_write_json(fig_dir / "cluster_figures_summary.json", summary)
    return summary
