"""Visualization helpers for heatmaps and reports."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.io import read_image


def _setup_figure(title: str, n_samples: int | None = None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6, 7))
    if n_samples is not None:
        title = f"{title}\n(n={n_samples})"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (normalized, left→right)")
    ax.set_ylabel("y (normalized, top→bottom)")
    return fig, ax


def plot_heatmap(
    grid: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    n_samples: int | None = None,
    center: float | None = None,
) -> None:
    fig, ax = _setup_figure(title, n_samples)
    if center is not None:
        limit = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1e-9)
        im = ax.imshow(grid, origin="upper", cmap=cmap, vmin=-limit, vmax=limit)
    else:
        im = ax.imshow(grid, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_label("density" if center is None else "difference")
    if vmin is not None and vmax is not None:
        cbar.ax.set_ylabel(f"range [{vmin:.4f}, {vmax:.4f}]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(
    image_path: Path,
    grid: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str = "turbo",
    alpha: float = 0.45,
) -> None:
    img = read_image(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    heat = cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat_norm = heat / max(heat.max(), 1e-9)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img_rgb)
    im = ax.imshow(heat_norm, origin="upper", cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(coords: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="cluster")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_k_selection(k_metrics_path: Path, out_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(k_metrics_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["silhouette", "calinski_harabasz", "davies_bouldin"]):
        ax.plot(df["k"], df[col], marker="o")
        ax.set_xlabel("k")
        ax.set_title(col)
    fig.suptitle("Cluster count selection metrics")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_visualizations(cfg: AnalysisConfig) -> None:
    """Generate PNG outputs from cached npz results."""
    cmap = cfg.report.colormap
    out = cfg.output.output_dir

    agg_dir = out / "aggregate"
    for name, fname in [
        ("absolute_mean", "absolute_stats.npz"),
        ("relative_mean", "relative_stats.npz"),
    ]:
        fpath = agg_dir / fname
        if not fpath.exists():
            continue
        d = np.load(fpath)
        n = int(d["n_samples"]) if "n_samples" in d else None
        for stat in ["mean", "median", "std", "usage_probability", "cv", "p25", "p50", "p75"]:
            if stat in d:
                plot_heatmap(d[stat], agg_dir / f"{name}_{stat}.png", f"{name} {stat}", cmap=cmap, n_samples=n)

    groups_dir = out / "groups"
    if groups_dir.exists():
        for rel_npz in groups_dir.rglob("relative_stats.npz"):
            if "diff_" in str(rel_npz.parent.name):
                continue
            d = np.load(rel_npz)
            n = int(d["n_samples"])
            gname = rel_npz.parent.name
            plot_heatmap(d["mean"], rel_npz.parent / "mean.png", f"Group {gname} mean", cmap=cmap, n_samples=n)
            plot_heatmap(
                d["usage_probability"],
                rel_npz.parent / "usage_prob.png",
                f"Group {gname} usage",
                cmap=cmap,
                n_samples=n,
            )
        for cmp_npz in groups_dir.rglob("comparison.npz"):
            d = np.load(cmp_npz)
            plot_heatmap(
                d["difference"],
                cmp_npz.parent / "difference.png",
                f"Difference {cmp_npz.parent.name}",
                cmap=cfg.report.diff_colormap,
                n_samples=int(d["n_a"]) + int(d["n_b"]),
                center=0,
            )

    cl_dir = out / "clustering"
    if cl_dir.exists():
        for kcsv in cl_dir.rglob("k_selection_metrics.csv"):
            plot_k_selection(kcsv, kcsv.parent / "k_selection.png")
        for stats in cl_dir.rglob("stats.npz"):
            d = np.load(stats)
            if "mean_feature" not in d:
                continue
            n = int(d["n_samples"])
            parent = stats.parent
            plot_heatmap(
                d["mean_feature"],
                parent / "mean_center.png",
                f"{parent.name} center",
                cmap=cmap,
                n_samples=n,
            )
