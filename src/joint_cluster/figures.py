"""Figures and representative contact sheets for joint clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ..utils import ensure_dir
from .gt_features import FEATURE_NAMES

try:
    import umap as umap_lib
except Exception:  # noqa: BLE001
    umap_lib = None


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(cum: np.ndarray, n_keep: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, color="#1f4e79")
    ax.axhline(0.95, color="gray", linestyle="--", label="95%")
    ax.axvline(n_keep, color="#9a031e", linestyle=":", label=f"kept={n_keep}")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Image embedding PCA variance")
    ax.legend()
    _save(fig, out)


def plot_k_curves(metrics: pd.DataFrame, out_inertia: Path, out_sil: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for exp, color in (("E0", "#1f4e79"), ("E1", "#2a6f6f"), ("E2", "#9a031e")):
        sub = metrics[metrics["experiment"] == exp].sort_values("k")
        ax.plot(sub["k"], sub["inertia"], marker="o", label=exp, color=color)
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("K vs Inertia (E0 image / E1 GT / E2 joint)")
    ax.legend()
    _save(fig, out_inertia)

    fig, ax = plt.subplots(figsize=(7, 4))
    for exp, color in (("E0", "#1f4e79"), ("E1", "#2a6f6f"), ("E2", "#9a031e")):
        sub = metrics[metrics["experiment"] == exp].sort_values("k")
        ax.plot(sub["k"], sub["silhouette"], marker="o", label=exp, color=color)
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette")
    ax.set_title("K vs Silhouette (E0 / E1 / E2)")
    ax.legend()
    _save(fig, out_sil)


def plot_cluster_sizes(sizes: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(sizes))
    ax.bar(xs, sizes, color="#3d5a80")
    total = sizes.sum()
    for i, s in enumerate(sizes):
        ax.text(i, s, f"{s}\n({100 * s / total:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"C{i}" for i in xs])
    ax.set_ylabel("count")
    ax.set_title("Final joint cluster sizes")
    _save(fig, out)


def plot_joint_pca2d(joint: np.ndarray, labels: np.ndarray, out: Path, seed: int = 42) -> None:
    from sklearn.decomposition import PCA

    Z = PCA(n_components=2, random_state=seed).fit_transform(joint)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, s=6, cmap="tab10", linewidths=0, alpha=0.7)
    fig.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title("Joint features → PCA-2D (viz only)")
    _save(fig, out)


def plot_joint_umap(joint: np.ndarray, labels: np.ndarray, out: Path, cfg: dict[str, Any]) -> None:
    if umap_lib is None:
        print("[figures] umap unavailable; skip")
        return
    u = cfg.get("umap") or {}
    reducer = umap_lib.UMAP(
        n_neighbors=int(u.get("n_neighbors", 15)),
        min_dist=float(u.get("min_dist", 0.1)),
        metric=str(u.get("metric", "euclidean")),
        random_state=int(cfg.get("random_state", 42)),
    )
    n = joint.shape[0]
    if n > 30000:
        rng = np.random.default_rng(int(cfg.get("random_state", 42)))
        fit_idx = rng.choice(n, size=30000, replace=False)
        reducer.fit(joint[fit_idx])
        U = reducer.transform(joint)
    else:
        U = reducer.fit_transform(joint)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(U[:, 0], U[:, 1], c=labels, s=6, cmap="tab10", linewidths=0, alpha=0.7)
    fig.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title("Joint features → UMAP-2D (viz only)")
    _save(fig, out)


def plot_gt_heatmap(gt_scaled_fit: np.ndarray, labels: np.ndarray, out: Path) -> None:
    k = int(labels.max()) + 1
    mat = np.zeros((k, len(FEATURE_NAMES)))
    for c in range(k):
        m = labels == c
        mat[c] = np.median(gt_scaled_fit[m], axis=0)
    fig, ax = plt.subplots(figsize=(8, 0.6 * k + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(list(FEATURE_NAMES), rotation=30, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{i}" for i in range(k)])
    ax.set_title("Per-cluster median of scaled GT features")
    fig.colorbar(im, ax=ax)
    _save(fig, out)


def plot_gt_boxplots(gt_raw: np.ndarray, labels: np.ndarray, out: Path) -> None:
    k = int(labels.max()) + 1
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, j, name in zip(axes, range(5), FEATURE_NAMES):
        data = [gt_raw[labels == c, j] for c in range(k)]
        ax.boxplot(data, tick_labels=[f"C{c}" for c in range(k)], showfliers=False)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Raw GT features by final cluster")
    _save(fig, out)


def plot_image_vs_joint_crosstab(
    image_labels: np.ndarray,
    joint_labels: np.ndarray,
    out: Path,
) -> None:
    ct = pd.crosstab(image_labels, joint_labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ct.to_numpy(), cmap="Blues")
    ax.set_xlabel("joint cluster")
    ax.set_ylabel("image-only cluster")
    ax.set_title("E0 vs E2 crosstab (counts)")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_yticks(range(ct.shape[0]))
    fig.colorbar(im, ax=ax)
    _save(fig, out)


def _sheet(
    rows: list[dict[str, Any]],
    title: str,
    out: Path,
    ncols: int = 5,
) -> None:
    n = len(rows)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 3.0))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        r = rows[i]
        p = r.get("image_path")
        if p and Path(p).exists():
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((180, 180))
                ax.imshow(im)
            except Exception:
                ax.text(0.5, 0.5, "err", ha="center")
        gt = (
            f"t={r.get('ast_tree_count')} n={r.get('total_ast_node_count')} "
            f"d={r.get('max_ast_depth')}\n"
            f"plain={r.get('distinct_plain_token_count')} "
            f"struct={r.get('distinct_structure_token_count')}"
        )
        ax.set_xlabel(
            f"{r.get('page_id','')[:10]}\nC{r.get('cluster')} dist={r.get('dist',0):.2f}\n{gt}",
            fontsize=6,
        )
    fig.suptitle(title, fontsize=11)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_representatives(
    fit_df: pd.DataFrame,
    labels: np.ndarray,
    dists: np.ndarray,
    out_dir: Path,
    *,
    center_n: int = 20,
    outlier_n: int = 12,
) -> pd.DataFrame:
    rows = []
    k = int(labels.max()) + 1
    for c in range(k):
        members = np.where(labels == c)[0]
        order = np.argsort(dists[members])
        center_ids = members[order[:center_n]]
        far_ids = members[order[-outlier_n:][::-1]]
        for kind, ids in (("center", center_ids), ("outlier", far_ids)):
            sheet_rows = []
            for i in ids:
                r = fit_df.iloc[i]
                item = {
                    "cluster": c,
                    "kind": kind,
                    "page_id": r["page_id"],
                    "image_path": r["image_path"],
                    "dist": float(dists[i]),
                    "ast_tree_count": r["ast_tree_count"],
                    "total_ast_node_count": r["total_ast_node_count"],
                    "max_ast_depth": r["max_ast_depth"],
                    "distinct_plain_token_count": r["distinct_plain_token_count"],
                    "distinct_structure_token_count": r["distinct_structure_token_count"],
                }
                sheet_rows.append(item)
                rows.append(item)
            _sheet(
                sheet_rows,
                f"cluster_{c} {kind}",
                out_dir / f"cluster_{c}_{kind}.png",
            )
    return pd.DataFrame(rows)
