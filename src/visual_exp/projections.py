"""PCA / UMAP projections and kNN galleries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ..utils import ensure_dir
from .io_util import atomic_write_parquet, load_aligned_embeddings, stamp_run_id

try:
    import umap as umap_lib
except Exception as e:  # noqa: BLE001
    umap_lib = None
    _UMAP_ERR = e
else:
    _UMAP_ERR = None


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_projections(cfg: dict[str, Any]) -> dict[str, Any]:
    proj = Path(cfg["paths"]["projections_dir"])
    gal = Path(cfg["paths"]["galleries_dir"]) / "nearest_neighbors"
    dpi = int(cfg["analysis"].get("figure_dpi", 150))
    seed = int(cfg.get("random_seed", 42))
    X, idx, emb_sha = load_aligned_embeddings(cfg)
    n, d = X.shape
    run_id = str(cfg["run_id"])
    man = pd.read_parquet(Path(cfg["paths"]["metadata_dir"]) / "manifest.parquet")
    id_to_path = dict(zip(man["image_id"].astype(str), man["image_path"].astype(str)))

    # PCA on full L2 embeddings
    k = min(int(cfg["analysis"].get("pca_n_components", 50)), n - 1, d)
    pca = PCA(n_components=k, random_state=seed)
    Z = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(
        {
            "image_id": idx["image_id"].astype(str),
            **{f"pc{i+1}": Z[:, i] for i in range(min(10, Z.shape[1]))},
        }
    )
    # merge audit for coloring
    for c in ("token_count", "aspect_ratio", "n_local_patches"):
        if c in idx.columns:
            pca_df[c] = idx[c].to_numpy()
        elif c in man.columns:
            m = man.set_index("image_id")
            pca_df[c] = pca_df["image_id"].map(m[c].to_dict())
    pca_df = stamp_run_id(pca_df, run_id)
    atomic_write_parquet(pca_df, proj / "pca_coordinates.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.5, linewidths=0, c="#1f4e79")
    axes[0].set_title("PCA PC1–PC2")
    if Z.shape[1] >= 3:
        axes[1].scatter(Z[:, 0], Z[:, 2], s=4, alpha=0.5, linewidths=0, c="#9a031e")
        axes[1].set_title("PCA PC1–PC3")
    else:
        axes[1].plot(np.arange(1, len(cum) + 1), cum)
        axes[1].set_title("PCA cumulative variance")
    _save(fig, proj / "pca_scatter.png", dpi)

    # UMAP (viz only)
    if umap_lib is None:
        raise RuntimeError(f"umap-learn required: {_UMAP_ERR}")
    reducer = umap_lib.UMAP(
        n_neighbors=int(cfg["analysis"].get("umap_n_neighbors", 15)),
        min_dist=float(cfg["analysis"].get("umap_min_dist", 0.1)),
        n_components=2,
        metric="cosine",
        random_state=int(cfg["analysis"].get("umap_random_state", 42)),
    )
    # fit on subset if large
    max_fit = 30000
    rng = np.random.default_rng(seed)
    if n > max_fit:
        fit_idx = rng.choice(n, size=max_fit, replace=False)
        reducer.fit(X[fit_idx])
        U = reducer.transform(X)
    else:
        U = reducer.fit_transform(X)
    umap_df = pca_df[["image_id"]].copy()
    umap_df["umap1"] = U[:, 0]
    umap_df["umap2"] = U[:, 1]
    for c in ("token_count", "aspect_ratio"):
        if c in pca_df.columns:
            umap_df[c] = pca_df[c]
    umap_df = stamp_run_id(umap_df, run_id)
    atomic_write_parquet(umap_df, proj / "umap_coordinates.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hexbin(U[:, 0], U[:, 1], gridsize=70, cmap="magma", mincnt=1)
    axes[0].set_title("UMAP density")
    for ax, key in zip(axes[1:], ["token_count", "aspect_ratio"]):
        vals = pd.to_numeric(umap_df.get(key), errors="coerce").to_numpy()
        sc = ax.scatter(U[:, 0], U[:, 1], c=vals, s=4, cmap="viridis", linewidths=0, alpha=0.7)
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"UMAP by {key}")
    _save(fig, proj / "umap_scatter.png", dpi)

    # kNN contact sheets
    k = min(int(cfg["analysis"].get("knn_k", 8)), max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(X)
    dists, inds = nn.kneighbors(X)
    # pick random queries
    q_idx = rng.choice(n, size=min(12, n), replace=False)
    _write_nn_gallery(
        [str(idx.iloc[i]["image_id"]) for i in range(n)],
        id_to_path,
        inds,
        dists,
        q_idx.tolist(),
        gal / "random_queries",
        neighbors=k,
    )
    return {
        "run_id": run_id,
        "embedding_sha256": emb_sha,
        "pca_dims": int(k),
        "umap_n": int(n),
        "pca_dims_95": int(np.searchsorted(cum, 0.95) + 1),
    }


def _write_nn_gallery(
    ids: list[str],
    id_to_path: dict[str, str],
    inds: np.ndarray,
    dists: np.ndarray,
    queries: list[int],
    out: Path,
    neighbors: int = 8,
) -> None:
    ensure_dir(out.parent)
    thumb = 140
    fig, axes = plt.subplots(len(queries), neighbors + 1, figsize=((neighbors + 1) * 1.8, len(queries) * 2.0))
    if len(queries) == 1:
        axes = np.array([axes])
    for ri, qi in enumerate(queries):
        row_ids = [ids[qi]] + [ids[int(inds[qi, j + 1])] for j in range(neighbors)]
        row_d = [0.0] + [float(dists[qi, j + 1]) for j in range(neighbors)]
        for j, (iid, dist) in enumerate(zip(row_ids, row_d)):
            ax = axes[ri, j]
            ax.set_xticks([])
            ax.set_yticks([])
            path = id_to_path.get(iid)
            if path and Path(path).exists():
                try:
                    im = Image.open(path).convert("RGB")
                    im.thumbnail((thumb, thumb))
                    ax.imshow(im)
                except Exception:
                    ax.text(0.5, 0.5, "err", ha="center")
            label = "QUERY" if j == 0 else f"d={dist:.3f}"
            ax.set_xlabel(f"{label}\n{iid[:8]}", fontsize=6)
    fig.suptitle("Cosine kNN gallery")
    _save(fig, out, dpi=120)
