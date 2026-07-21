"""Plotting helpers for analysis_v2 reports."""

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


def _save(fig, path_no_ext: Path, dpi: int = 200) -> None:
    ensure_dir(path_no_ext.parent)
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_colored(
    coords: np.ndarray,
    values: np.ndarray,
    out: Path,
    title: str,
    cbar_label: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=8, cmap="viridis", alpha=0.75, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(cbar_label)
    ax.set_title(title)
    _save(fig, out, dpi=dpi)


def plot_pca_panel(coords: np.ndarray, cum_var: list[float], out: Path, title: str, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(1, len(cum_var) + 1)
    axes[0].plot(x, cum_var, color="#1f4e79")
    for thr, ls in [(0.8, "--"), (0.9, "-."), (0.95, ":")]:
        axes[0].axhline(thr, color="gray", linestyle=ls, linewidth=1)
        idx = int(np.searchsorted(cum_var, thr)) + 1
        axes[0].axvline(idx, color="gray", linestyle=ls, linewidth=0.8, alpha=0.7)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(f"{title}: cumulative variance")
    if len(coords):
        axes[1].hexbin(coords[:, 0], coords[:, 1], gridsize=60, cmap="viridis", mincnt=1)
        axes[1].set_title(f"{title}: PCA-1×2")
    _save(fig, out, dpi=dpi)


def plot_umap_hex(emb: np.ndarray, out: Path, title: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(emb[:, 0], emb[:, 1], gridsize=70, cmap="magma", mincnt=1)
    ax.set_title(f"{title}: UMAP (n={len(emb)})")
    _save(fig, out, dpi=dpi)


def plot_knn_dist(kth: np.ndarray, out: Path, title: str, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(kth, bins=50, color="#2a6f6f", edgecolor="white")
    axes[0].set_title(f"{title}: 15-NN distance")
    axes[1].plot(np.sort(kth), color="#8b3a3a")
    axes[1].set_title(f"{title}: sorted 15-NN")
    _save(fig, out, dpi=dpi)


def plot_channel_overview(
    *,
    name: str,
    coords2: np.ndarray,
    umap_xy: np.ndarray,
    kth: np.ndarray,
    cum: list[float],
    out_dir: Path,
    dpi: int,
) -> None:
    plot_pca_panel(coords2, cum, out_dir / f"{name}_pca", name, dpi)
    plot_umap_hex(umap_xy, out_dir / f"{name}_umap", name, dpi)
    plot_knn_dist(kth, out_dir / f"{name}_knn15", name, dpi)


def _contact_sheet(
    image_ids: list[str],
    id_to_path: dict[str, str],
    nn_indices: np.ndarray,
    nn_distances: np.ndarray,
    anchors: list[int],
    out: Path,
    neighbors: int = 5,
) -> None:
    if not anchors:
        return
    ensure_dir(out.parent)
    rows = []
    for a in anchors:
        row_imgs = [image_ids[a]]
        row_dists = [0.0]
        for j in range(min(neighbors, nn_indices.shape[1])):
            row_imgs.append(image_ids[int(nn_indices[a, j])])
            row_dists.append(float(nn_distances[a, j]))
        rows.append((row_imgs, row_dists))

    thumb = 160
    fig, axes = plt.subplots(len(rows), neighbors + 1, figsize=((neighbors + 1) * 2.2, len(rows) * 2.4))
    if len(rows) == 1:
        axes = np.array([axes])
    for i, (ids, dists) in enumerate(rows):
        for j, (iid, dist) in enumerate(zip(ids, dists)):
            ax = axes[i, j]
            path = id_to_path.get(iid)
            ax.set_xticks([])
            ax.set_yticks([])
            if path and Path(path).exists():
                try:
                    im = Image.open(path).convert("RGB")
                    im.thumbnail((thumb, thumb))
                    ax.imshow(im)
                except Exception:
                    ax.text(0.5, 0.5, "err", ha="center")
            else:
                ax.text(0.5, 0.5, "missing", ha="center")
            ax.set_xlabel(f"{'ANCHOR' if j == 0 else f'd={dist:.3f}'}\n{iid[:10]}", fontsize=7)
    fig.suptitle(out.stem, fontsize=11)
    _save(fig, out, dpi=150)


def write_contact_sheets_for_channel(
    *,
    channel: str,
    ids: list[str],
    knn: dict[str, np.ndarray],
    anchors: dict[str, list[int]],
    out_dir: Path,
    cfg: dict[str, Any],
) -> None:
    out_dir = ensure_dir(out_dir)
    man_path = Path(cfg["paths"]["outputs_dir"]) / "manifest.parquet"
    id_to_path: dict[str, str] = {}
    if man_path.exists():
        man = pd.read_parquet(man_path)
        id_to_path = dict(zip(man["image_id"].astype(str), man["absolute_path"].astype(str)))
    neighbors = int((cfg.get("analysis") or {}).get("contact_sheet_neighbors", 5))
    for kind, idxs in anchors.items():
        _contact_sheet(
            ids,
            id_to_path,
            knn["nn_indices"],
            knn["nn_distances"],
            idxs,
            out_dir / f"{channel}_{kind}",
            neighbors=neighbors,
        )
