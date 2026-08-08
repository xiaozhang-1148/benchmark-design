"""K=1000 micro-clustering: extract representatives + coverage analysis."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..utils import atomic_write_json, ensure_dir
from .io_util import atomic_write_parquet, load_aligned_embeddings, stamp_run_id

try:
    import umap as umap_lib
except Exception:  # noqa: BLE001
    umap_lib = None

_MACRO_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_HIGH_SIM_THRESHOLDS = (0.99, 0.995, 0.999)


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _fit_kmeans(X: np.ndarray, k: int, seed: int, *, n_init: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Spherical KMeans on L2 embeddings; return (labels, centers)."""
    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        algorithm="lloyd",
        max_iter=300,
    )
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.astype(np.float64)
    # Renormalize centers for cosine geometry.
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    return labels.astype(np.int32), centers


def _select_medoids(X: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int) -> np.ndarray:
    """For each cluster, pick the sample with highest cosine similarity to center."""
    medoid_idx = np.full(k, -1, dtype=np.int64)
    for c in range(k):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue
        sims = X[members] @ centers[c]
        medoid_idx[c] = int(members[int(np.argmax(sims))])
    return medoid_idx


def _load_or_compute_xy(cfg: dict[str, Any], X: np.ndarray, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    proj = Path(cfg["paths"]["projections_dir"])
    ensure_dir(proj)
    seed = int(cfg.get("random_seed", 42))
    run_id = str(cfg["run_id"])

    pca_path = proj / "pca_coordinates.parquet"
    umap_path = proj / "umap_coordinates.parquet"

    if pca_path.is_file():
        pdf = pd.read_parquet(pca_path).set_index("image_id").reindex(ids)
        P = pdf[["pc1", "pc2"]].to_numpy(dtype=np.float64)
        if np.isnan(P).any():
            raise RuntimeError("pca_coordinates incomplete")
    else:
        n_comp = min(50, X.shape[0] - 1, X.shape[1])
        Z = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
        P = Z[:, :2]
        stamp_run_id(
            pd.DataFrame({"image_id": ids, "pc1": P[:, 0], "pc2": P[:, 1]}),
            run_id,
        ).to_parquet(pca_path, index=False)

    if umap_path.is_file():
        udf = pd.read_parquet(umap_path).set_index("image_id").reindex(ids)
        U = udf[["umap1", "umap2"]].to_numpy(dtype=np.float64)
        if np.isnan(U).any():
            raise RuntimeError("umap_coordinates incomplete")
    else:
        if umap_lib is None:
            raise RuntimeError("umap-learn is required for UMAP figures")
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
        stamp_run_id(
            pd.DataFrame({"image_id": ids, "umap1": U[:, 0], "umap2": U[:, 1]}),
            run_id,
        ).to_parquet(umap_path, index=False)

    return P, U


def _plot_representatives(
    *,
    P: np.ndarray,
    U: np.ndarray,
    rep_mask: np.ndarray,
    sizes: np.ndarray,
    macro_labels: np.ndarray,
    out_dir: Path,
    dpi: int,
) -> list[str]:
    """All points light gray; reps colored by macro K=4, sized by microcluster size."""
    written: list[str] = []
    size_scaled = 20.0 + 180.0 * (sizes / max(float(sizes.max()), 1.0))

    for name, xy in (("pca", P), ("umap", U)):
        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        ax.scatter(xy[:, 0], xy[:, 1], s=3, c="#d0d0d0", alpha=0.35, linewidths=0, zorder=1)
        for c in range(4):
            mask = rep_mask & (macro_labels == c)
            if not mask.any():
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=size_scaled[mask],
                c=_MACRO_COLORS[c],
                alpha=0.85,
                linewidths=0.4,
                edgecolors="white",
                label=f"macro K=4 cluster {c} (n={int(mask.sum())})",
                zorder=2,
            )
        ax.set_title(f"{name.upper()}: full corpus (gray) + 1000 microcluster reps (colored by visual K=4)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        path = out_dir / f"representatives_{name}.png"
        _save(fig, path, dpi)
        written.append(str(path))

    # Combined panel
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, name, xy in zip(axes, ("PCA", "UMAP"), (P, U), strict=True):
        ax.scatter(xy[:, 0], xy[:, 1], s=2.5, c="#d0d0d0", alpha=0.35, linewidths=0, zorder=1)
        for c in range(4):
            mask = rep_mask & (macro_labels == c)
            if not mask.any():
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=size_scaled[mask],
                c=_MACRO_COLORS[c],
                alpha=0.85,
                linewidths=0.35,
                edgecolors="white",
                label=f"K4-{c}",
                zorder=2,
            )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("1000 microcluster representatives over full embedding space", fontsize=12)
    fig.tight_layout()
    path = out_dir / "representatives_pca_umap.png"
    _save(fig, path, dpi)
    written.append(str(path))
    return written


def run_microcluster_1000(cfg: dict[str, Any], *, k_micro: int = 1000, k_macro: int = 4) -> dict[str, Any]:
    """
    KMeans(K=1000) representatives + coverage / redundancy / macro-K4 analysis.

    Writes:
      - examples_1000/: 1000 representative images + index.csv
      - emaples_cluster_1000/: metrics, tables, figures  (path spelling kept as requested)
    """
    run_dir = Path(cfg["paths"]["run_dir"])
    examples_dir = ensure_dir(run_dir / "examples_1000")
    # Intentional directory name matching the user request (typo preserved).
    analysis_dir = ensure_dir(run_dir / "emaples_cluster_1000")
    figures_dir = ensure_dir(analysis_dir / "figures")
    seed = int(cfg.get("random_seed", 42))
    dpi = int(cfg["analysis"].get("figure_dpi", 150))
    run_id = str(cfg["run_id"])

    print(f"[micro1000] loading embeddings …")
    X, idx, emb_sha = load_aligned_embeddings(cfg)
    X = np.asarray(X, dtype=np.float32)
    n, dim = X.shape
    if k_micro > n:
        raise ValueError(f"k_micro={k_micro} > n={n}")
    ids = idx["image_id"].astype(str).tolist()
    man = pd.read_parquet(Path(cfg["paths"]["metadata_dir"]) / "manifest.parquet")
    id_to_path = dict(zip(man["image_id"].astype(str), man["image_path"].astype(str)))

    # --- Macro visual K=4 (spherical KMeans) ---
    print(f"[micro1000] fitting macro KMeans k={k_macro} …")
    macro_labels, _ = _fit_kmeans(X, k_macro, seed=seed + 7, n_init=20)
    full_macro_counts = np.bincount(macro_labels, minlength=k_macro)
    full_macro_ratio = full_macro_counts / float(n)

    # --- Micro KMeans K=1000 ---
    print(f"[micro1000] fitting micro KMeans k={k_micro} (n={n}, dim={dim}) …")
    micro_labels, centers = _fit_kmeans(X, k_micro, seed=seed, n_init=5)
    sizes = np.bincount(micro_labels, minlength=k_micro).astype(np.int64)
    empty = int(np.sum(sizes == 0))
    if empty:
        print(f"[micro1000] warning: {empty} empty clusters")

    medoid_idx = _select_medoids(X, micro_labels, centers, k_micro)
    valid = medoid_idx >= 0
    if not np.all(valid):
        # Drop empty clusters from representative set; renumber kept ones.
        keep = np.where(valid)[0]
        print(f"[micro1000] keeping {len(keep)} non-empty microclusters")
    else:
        keep = np.arange(k_micro)

    rep_pos = medoid_idx[keep]
    rep_cluster_ids = keep.astype(np.int32)
    S = X[rep_pos]  # (k_eff, dim)
    k_eff = int(S.shape[0])
    rep_ids = [ids[int(i)] for i in rep_pos]
    rep_sizes = sizes[keep]
    rep_ratios = rep_sizes / float(n)
    rep_macro = macro_labels[rep_pos]

    # --- 1) Representation coverage: d_i = 1 - max_s cos(x_i, s) ---
    print("[micro1000] computing coverage distances …")
    # Chunked matmul to limit peak memory.
    chunk = 4096
    max_sims = np.empty(n, dtype=np.float64)
    nearest_rep = np.empty(n, dtype=np.int32)
    St = S.T  # (dim, k_eff)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = X[start:end] @ St  # (chunk, k_eff)
        nearest_rep[start:end] = np.argmax(sims, axis=1).astype(np.int32)
        max_sims[start:end] = sims[np.arange(end - start), nearest_rep[start:end]]
    d = 1.0 - max_sims
    coverage = {
        "n_full": int(n),
        "n_representatives": int(k_eff),
        "mean_distance": float(np.mean(d)),
        "median_distance": float(np.median(d)),
        "p90_distance": _percentile(d, 90),
        "p95_distance": _percentile(d, 95),
        "max_distance": float(np.max(d)),
        "min_distance": float(np.min(d)),
        "embedding_sha256": emb_sha,
        "formula": "d_i = 1 - max_{s in S} cos(x_i, s)",
    }
    atomic_write_json(analysis_dir / "representation_coverage.json", coverage)
    atomic_write_parquet(
        stamp_run_id(
            pd.DataFrame(
                {
                    "image_id": ids,
                    "coverage_distance": d,
                    "nearest_rep_index": nearest_rep,
                    "nearest_rep_image_id": [rep_ids[int(j)] for j in nearest_rep],
                    "nearest_rep_cosine": max_sims,
                    "macro_k4": macro_labels.astype(int),
                    "microcluster_id": micro_labels.astype(int),
                }
            ),
            run_id,
        ),
        analysis_dir / "per_image_coverage.parquet",
    )
    print(
        f"[micro1000] coverage mean={coverage['mean_distance']:.4f} "
        f"median={coverage['median_distance']:.4f} "
        f"p90={coverage['p90_distance']:.4f} "
        f"p95={coverage['p95_distance']:.4f} "
        f"max={coverage['max_distance']:.4f}"
    )

    # --- 2) Redundancy among representatives ---
    print("[micro1000] computing representative redundancy …")
    sim_ss = S @ S.T  # (k_eff, k_eff)
    np.fill_diagonal(sim_ss, -np.inf)
    nn_sim = sim_ss.max(axis=1)
    nn_idx = sim_ss.argmax(axis=1)
    nn_dist = 1.0 - nn_sim  # cosine distance to nearest other rep
    high_pairs: dict[str, int] = {}
    # Count unique unordered pairs above threshold
    triu = np.triu(sim_ss, k=1)
    for thr in _HIGH_SIM_THRESHOLDS:
        high_pairs[f"pairs_cos_ge_{thr}"] = int(np.sum(triu >= thr))

    redundancy = {
        "n_representatives": int(k_eff),
        "mean_nn_cosine": float(np.mean(nn_sim)),
        "median_nn_cosine": float(np.median(nn_sim)),
        "mean_nn_distance": float(np.mean(nn_dist)),
        "median_nn_distance": float(np.median(nn_dist)),
        "min_nn_distance": float(np.min(nn_dist)),
        "max_nn_distance": float(np.max(nn_dist)),
        "high_similarity_pair_counts": high_pairs,
        "near_duplicate_note": (
            "Large pairs_cos_ge_0.99 / 0.995 suggests many near-duplicate representatives."
        ),
    }
    atomic_write_json(analysis_dir / "representative_redundancy.json", redundancy)

    # --- 3) Macro K=4 proportions ---
    rep_macro_counts = np.bincount(rep_macro.astype(int), minlength=k_macro)
    rep_macro_ratio = rep_macro_counts / float(k_eff)
    macro_cmp = pd.DataFrame(
        {
            "macro_cluster": list(range(k_macro)),
            "full_count": full_macro_counts.tolist(),
            "full_ratio": full_macro_ratio.tolist(),
            "rep_count": rep_macro_counts.tolist(),
            "rep_ratio": rep_macro_ratio.tolist(),
            "ratio_diff": (rep_macro_ratio - full_macro_ratio).tolist(),
        }
    )
    macro_cmp.to_csv(analysis_dir / "macro_k4_proportion_comparison.csv", index=False)
    atomic_write_json(
        analysis_dir / "macro_k4_proportion_comparison.json",
        {
            "method": "spherical_kmeans",
            "k": k_macro,
            "full_ratios": full_macro_ratio.tolist(),
            "rep_ratios": rep_macro_ratio.tolist(),
            "max_abs_ratio_diff": float(np.max(np.abs(rep_macro_ratio - full_macro_ratio))),
            "table": macro_cmp.to_dict(orient="records"),
        },
    )

    # --- 4) Representatives table (weights + nn + macro) ---
    rep_table = stamp_run_id(
        pd.DataFrame(
            {
                "rep_index": np.arange(k_eff, dtype=int),
                "microcluster_id": rep_cluster_ids.astype(int),
                "image_id": rep_ids,
                "image_path": [id_to_path.get(i, "") for i in rep_ids],
                "microcluster_size": rep_sizes.astype(int),
                "microcluster_ratio": rep_ratios.astype(float),
                "macro_k4": rep_macro.astype(int),
                "nn_rep_index": nn_idx.astype(int),
                "nn_rep_image_id": [rep_ids[int(j)] for j in nn_idx],
                "nn_cosine": nn_sim.astype(float),
                "nn_distance": nn_dist.astype(float),
            }
        ),
        run_id,
    )
    atomic_write_parquet(rep_table, analysis_dir / "representatives.parquet")
    rep_table.to_csv(analysis_dir / "representatives.csv", index=False)

    # --- Copy / link example images ---
    print(f"[micro1000] writing {k_eff} example images → {examples_dir}")
    # Clean previous copies (keep directory).
    for old in examples_dir.iterdir():
        if old.is_file():
            old.unlink()
    copied = 0
    missing = 0
    example_rows = []
    for row in rep_table.itertuples(index=False):
        src = Path(str(row.image_path))
        ext = src.suffix.lower() if src.suffix else ".jpg"
        dest_name = f"{int(row.rep_index):04d}__mc{int(row.microcluster_id):04d}__{row.image_id}{ext}"
        dest = examples_dir / dest_name
        if src.is_file():
            shutil.copy2(src, dest)
            copied += 1
            out_path = str(dest)
        else:
            missing += 1
            out_path = ""
        example_rows.append(
            {
                "rep_index": int(row.rep_index),
                "microcluster_id": int(row.microcluster_id),
                "image_id": row.image_id,
                "source_path": str(src),
                "example_path": out_path,
                "microcluster_size": int(row.microcluster_size),
                "microcluster_ratio": float(row.microcluster_ratio),
                "macro_k4": int(row.macro_k4),
            }
        )
    examples_index = stamp_run_id(pd.DataFrame(example_rows), run_id)
    examples_index.to_csv(examples_dir / "index.csv", index=False)
    atomic_write_parquet(examples_index, examples_dir / "index.parquet")
    atomic_write_json(
        examples_dir / "manifest.json",
        {
            "n_requested": int(k_micro),
            "n_representatives": int(k_eff),
            "n_copied": int(copied),
            "n_missing_source": int(missing),
            "embedding_sha256": emb_sha,
            "run_id": run_id,
        },
    )

    # --- 5) Visualization ---
    print("[micro1000] plotting PCA/UMAP …")
    P, U = _load_or_compute_xy(cfg, X, ids)
    rep_mask = np.zeros(n, dtype=bool)
    rep_mask[rep_pos] = True
    # Point sizes for all rows: only reps get non-zero size in plot helper.
    size_all = np.ones(n, dtype=np.float64)
    size_all[rep_pos] = rep_sizes.astype(np.float64)
    fig_paths = _plot_representatives(
        P=P,
        U=U,
        rep_mask=rep_mask,
        sizes=size_all,
        macro_labels=macro_labels,
        out_dir=figures_dir,
        dpi=dpi,
    )

    # Size histogram of microclusters
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(rep_sizes, bins=40, color="#1f4e79", edgecolor="white")
    ax.set_xlabel("microcluster_size")
    ax.set_ylabel("number of representatives")
    ax.set_title("Distribution of microcluster sizes (K=1000)")
    hist_path = figures_dir / "microcluster_size_hist.png"
    _save(fig, hist_path, dpi)
    fig_paths.append(str(hist_path))

    summary = {
        "run_id": run_id,
        "embedding_sha256": emb_sha,
        "n_full": int(n),
        "k_micro": int(k_micro),
        "k_micro_effective": int(k_eff),
        "k_macro": int(k_macro),
        "examples_dir": str(examples_dir),
        "analysis_dir": str(analysis_dir),
        "representation_coverage": coverage,
        "representative_redundancy": redundancy,
        "macro_k4_max_abs_ratio_diff": float(np.max(np.abs(rep_macro_ratio - full_macro_ratio))),
        "figures": fig_paths,
        "n_examples_copied": int(copied),
        "n_examples_missing": int(missing),
    }
    atomic_write_json(analysis_dir / "summary.json", summary)
    print(f"[micro1000] done examples→{examples_dir}")
    print(f"[micro1000] done analysis→{analysis_dir}")
    return summary
