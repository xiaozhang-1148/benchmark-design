"""Representation validity diagnostics with multi-PC technical confounder alerts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ..utils import atomic_write_json, ensure_dir
from .io_util import atomic_write_parquet, load_aligned_embeddings, stamp_run_id, write_run_meta


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_embeddings(cfg: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame]:
    X, idx, _ = load_aligned_embeddings(cfg)
    return X, idx


def run_diagnostics(cfg: dict[str, Any]) -> dict[str, Any]:
    diag = Path(cfg["paths"]["diagnostics_dir"])
    dpi = int(cfg["analysis"].get("figure_dpi", 150))
    seed = int(cfg.get("random_seed", 42))
    rho_thr = float(cfg["analysis"].get("confound_rho_threshold", 0.5))

    X, idx, emb_sha = load_aligned_embeddings(cfg)
    n, d = X.shape
    man = pd.read_parquet(Path(cfg["paths"]["metadata_dir"]) / "manifest.parquet")
    man = man.copy()
    man.index = man["image_id"].astype(str)

    nan_inf = int((~np.isfinite(X)).sum())
    rounded = np.round(X, 6)
    uniq = np.unique(rounded, axis=0).shape[0]
    n_dup_groups = n - uniq

    norms = (
        idx["norm_before_l2"].to_numpy(dtype=np.float64)
        if "norm_before_l2" in idx.columns
        else np.linalg.norm(X, axis=1)
    )

    rng = np.random.default_rng(seed)
    sample_n = min(int(cfg["analysis"].get("similarity_sample", 2000)), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    Xs = X[sample_idx]
    sims = []
    for _ in range(min(5000, sample_n * 2)):
        i, j = rng.choice(sample_n, size=2, replace=False)
        sims.append(float(np.dot(Xs[i], Xs[j])))
    sims = np.asarray(sims)

    nn = NearestNeighbors(n_neighbors=2, metric="cosine")
    nn.fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    nn1 = dist[:, 1]

    n_pc = min(50, n - 1, d)
    pca = PCA(n_components=n_pc, random_state=seed)
    coords = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    ev = pca.explained_variance_
    p = ev / ev.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    dims_80 = int(np.searchsorted(cum, 0.80) + 1)

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
    ax.axhline(0.8, color="gray", linestyle=":")
    ax.axhline(0.95, color="gray", linestyle="--")
    ax.set_ylim(0, 1.02)
    ax.set_title("PCA cumulative explained variance")
    _save(fig, diag / "pca_variance.png", dpi)

    ids = idx["image_id"].astype(str).tolist()

    def _col(name: str) -> np.ndarray:
        if name in idx.columns:
            return pd.to_numeric(idx[name], errors="coerce").to_numpy(dtype=np.float64)
        vals = []
        for i in ids:
            if i in man.index:
                vals.append(man.loc[i, name] if name in man.columns else np.nan)
            else:
                vals.append(np.nan)
        return pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=np.float64)

    tech_vars = {
        "token_count": _col("token_count"),
        "n_local_patches": _col("n_local_patches"),
        "aspect_ratio": _col("aspect_ratio"),
        "width": _col("width"),
        "height": _col("height"),
        "file_size": _col("file_size"),
        "norm_before_l2": norms.astype(np.float64),
    }

    # PCs to check: PC1–PC10 and all PCs within 80% cumulative variance
    pc_indices = sorted(set(range(min(10, n_pc))) | set(range(dims_80)))
    alerts: list[dict[str, Any]] = []
    confound_matrix: dict[str, dict[str, float]] = {}
    for pi in pc_indices:
        pc_name = f"PC{pi + 1}"
        confound_matrix[pc_name] = {}
        pc_vals = coords[:, pi]
        for tname, tvals in tech_vars.items():
            mask = np.isfinite(tvals) & np.isfinite(pc_vals)
            if mask.sum() < 10:
                continue
            # skip constant tech vars (e.g. token_count all 256)
            if np.nanstd(tvals[mask]) < 1e-12:
                continue
            rho, _ = spearmanr(tvals[mask], pc_vals[mask])
            rho_f = float(rho)
            confound_matrix[pc_name][tname] = rho_f
            if abs(rho_f) >= rho_thr:
                alerts.append(
                    {
                        "pc": pc_name,
                        "variable": tname,
                        "spearman_rho": rho_f,
                        "threshold": rho_thr,
                        "severity": float(pca.explained_variance_ratio_[pi]),
                    }
                )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_keys = ["token_count", "aspect_ratio", "width", "file_size", "n_local_patches", "norm_before_l2"]
    for ax, key in zip(axes.ravel(), plot_keys):
        vals = tech_vars[key]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=6, cmap="viridis", linewidths=0, alpha=0.7)
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(f"PCA by {key}")
    _save(fig, diag / "technical_confounders.png", dpi)

    # heatmap of |rho| for PC1-10 x tech
    heat_pcs = [f"PC{i+1}" for i in range(min(10, n_pc))]
    heat_vars = list(tech_vars.keys())
    mat = np.zeros((len(heat_pcs), len(heat_vars)))
    for i, pc in enumerate(heat_pcs):
        for j, tv in enumerate(heat_vars):
            mat[i, j] = abs(confound_matrix.get(pc, {}).get(tv, np.nan))
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat_vars)))
    ax.set_xticklabels(heat_vars, rotation=30, ha="right")
    ax.set_yticks(range(len(heat_pcs)))
    ax.set_yticklabels(heat_pcs)
    ax.set_title(f"|Spearman ρ| tech confounders (alert ≥ {rho_thr})")
    fig.colorbar(im, ax=ax)
    _save(fig, diag / "confounder_rho_heatmap.png", dpi)

    summary = {
        "run_id": cfg["run_id"],
        "embedding_sha256": emb_sha,
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
        "pca_dims_80": dims_80,
        "pca_dims_95": int(np.searchsorted(cum, 0.95) + 1),
        "pca_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "effective_rank_top50": eff_rank,
        "confound_rho_threshold": rho_thr,
        "confound_matrix": confound_matrix,
        "confound_alerts": alerts,
        "confound_alert_count": len(alerts),
        "collapse_warning": bool(sims.mean() > 0.95),
    }
    atomic_write_json(diag / "diagnostics_summary.json", summary)
    atomic_write_json(diag / "confound_alerts.json", {"alerts": alerts, "threshold": rho_thr})
    # merge into extraction_summary if present
    prev = {}
    ep = diag / "extraction_summary.json"
    if ep.exists():
        import json

        try:
            prev = json.loads(ep.read_text())
        except Exception:
            prev = {}
    atomic_write_json(ep, {**prev, **{k: summary[k] for k in ("n", "dim", "embedding_sha256", "run_id", "confound_alert_count")}})
    write_run_meta(cfg, stage="diagnostics_done", embedding_sha256=emb_sha, confound_alert_count=len(alerts))
    print(
        f"[diagnostics] n={n} dim={d} cos_mean={summary['pairwise_cosine_mean']:.3f} "
        f"confound_alerts={len(alerts)}"
    )
    for a in alerts[:12]:
        print(f"  ALERT {a['variable']}–{a['pc']} ρ={a['spearman_rho']:.3f}")
    return summary
