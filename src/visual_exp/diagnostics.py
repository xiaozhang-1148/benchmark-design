"""Representation validity diagnostics (collapse, norms, technical confounders)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..utils import atomic_write_json, ensure_dir


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_embeddings(cfg: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    meta_dir = Path(cfg["paths"]["metadata_dir"])
    name = (cfg.get("extract") or {}).get("save_name", "deepseek_ocr2_mean_l2.npy")
    X = np.load(emb_dir / name)
    idx = pd.read_parquet(meta_dir / "embedding_index.parquet")
    idx = idx[idx["status"] == "ok"].sort_values("embedding_row").reset_index(drop=True)
    # align
    rows = idx["embedding_row"].to_numpy()
    if len(rows) == len(X) and np.array_equal(rows, np.arange(len(X))):
        return X.astype(np.float32), idx
    return X[rows].astype(np.float32), idx.reset_index(drop=True)


def run_diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
    diag = Path(cfg["paths"]["diagnostics_dir"])
    dpi = int(cfg["analysis"].get("figure_dpi", 150))
    seed = int(cfg.get("random_seed", 42))
    X, idx = load_embeddings(cfg)
    n, d = X.shape
    man = pd.read_parquet(Path(cfg["paths"]["metadata_dir"]) / "manifest.parquet")
    man = man.set_index("image_id")

    nan_inf = int((~np.isfinite(X)).sum())
    # duplicate vectors
    # hash rounded floats
    rounded = np.round(X, 6)
    uniq = np.unique(rounded, axis=0).shape[0]
    n_dup_groups = n - uniq

    norms = idx["norm_before_l2"].to_numpy(dtype=np.float64) if "norm_before_l2" in idx.columns else np.linalg.norm(X, axis=1)
    tokens = idx["token_count"].to_numpy(dtype=np.float64) if "token_count" in idx.columns else None

    rng = np.random.default_rng(seed)
    sample_n = min(int(cfg["analysis"].get("similarity_sample", 2000)), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    Xs = X[sample_idx]
    # pairwise cosine among random pairs
    sims = []
    for _ in range(min(5000, sample_n * 2)):
        i, j = rng.choice(sample_n, size=2, replace=False)
        sims.append(float(np.dot(Xs[i], Xs[j])))
    sims = np.asarray(sims)

    # kNN distances (k=1 excluding self) on sample
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    nn1 = dist[:, 1]

    # PCA variance + effective rank via covariance eigenvalues
    pca = PCA(n_components=min(50, n - 1, d), random_state=seed)
    coords = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    # effective rank
    ev = pca.explained_variance_
    p = ev / ev.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    # plots
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sims, bins=50, color="#3d5a80", edgecolor="white")
    ax.set_title("Random pairwise cosine similarity")
    ax.axvline(sims.mean(), color="red", linestyle="--", label=f"mean={sims.mean():.3f}")
    ax.legend()
    _save(fig, diag / "similarity_distribution.png", dpi)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(norms[np.isfinite(norms)], bins=50, color="#2a6f6f", edgecolor="white")
    ax.set_title("L2 norm before normalization")
    _save(fig, diag / "norm_distribution.png", dpi)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#1f4e79")
    ax.axhline(0.95, color="gray", linestyle=":")
    ax.set_ylim(0, 1.02)
    ax.set_title("PCA cumulative explained variance")
    _save(fig, diag / "pca_variance.png", dpi)

    # technical confounders vs PC1/PC2
    ids = idx["image_id"].astype(str).tolist()
    audit = {
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "token_count": [man.loc[i, "token_count"] if i in man.index else np.nan for i in ids],
        "aspect_ratio": [man.loc[i, "aspect_ratio"] if i in man.index else np.nan for i in ids],
        "width": [man.loc[i, "width"] if i in man.index else np.nan for i in ids],
        "file_size": [man.loc[i, "file_size"] if i in man.index else np.nan for i in ids],
        "n_local_patches": [man.loc[i, "n_local_patches"] if i in man.index else np.nan for i in ids],
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key in zip(axes.ravel(), ["token_count", "aspect_ratio", "width", "n_local_patches"]):
        vals = pd.to_numeric(pd.Series(audit[key]), errors="coerce").to_numpy()
        sc = ax.scatter(audit["pc1"], audit["pc2"], c=vals, s=6, cmap="viridis", linewidths=0, alpha=0.7)
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"PCA colored by {key}")
    _save(fig, diag / "technical_confounders.png", dpi)

    # Spearman of audit vars with PC1
    from scipy.stats import spearmanr

    confound_corr = {}
    for key in ["token_count", "aspect_ratio", "width", "file_size", "n_local_patches"]:
        vals = pd.to_numeric(pd.Series(audit[key]), errors="coerce").to_numpy()
        mask = np.isfinite(vals) & np.isfinite(audit["pc1"])
        if mask.sum() > 10:
            rho, _ = spearmanr(vals[mask], np.asarray(audit["pc1"])[mask])
            confound_corr[key] = float(rho)

    summary = {
        "n": int(n),
        "dim": int(d),
        "nan_inf_count": nan_inf,
        "n_unique_vectors": int(uniq),
        "n_duplicate_excess": int(n_dup_groups),
        "pairwise_cosine_mean": float(sims.mean()),
        "pairwise_cosine_p50": float(np.median(sims)),
        "pairwise_cosine_p95": float(np.percentile(sims, 95)),
        "nn1_cosine_distance_mean": float(nn1.mean()),
        "nn1_cosine_distance_p50": float(np.median(nn1)),
        "pca_dims_95": int(np.searchsorted(cum, 0.95) + 1),
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "effective_rank_top50": eff_rank,
        "confound_spearman_pc1": confound_corr,
        "collapse_warning": bool(sims.mean() > 0.95),
        "token_count_dominated_warning": bool(abs(confound_corr.get("token_count", 0)) > 0.7),
    }
    atomic_write_json(diag / "extraction_summary.json", {**(json_load_existing(diag / "extraction_summary.json")), **summary})
    atomic_write_json(diag / "diagnostics_summary.json", summary)
    print(f"[diagnostics] n={n} dim={d} cos_mean={summary['pairwise_cosine_mean']:.3f} collapse={summary['collapse_warning']}")
    return summary


def json_load_existing(path: Path) -> dict:
    import json

    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}
