"""Plotting for independent feature channels (PNG + SVG)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from .config import get_seed, load_config
from .utils import ensure_dir

try:
    import umap
except Exception:  # noqa: BLE001
    umap = None


def _save(fig, path_no_ext: Path, dpi: int = 200) -> None:
    ensure_dir(path_no_ext.parent)
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path_no_ext}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_pca_panel(coords: np.ndarray, cum_var: list[float], out: Path, title: str, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(1, len(cum_var) + 1)
    axes[0].plot(x, cum_var, color="#1f4e79")
    for thr, ls in [(0.8, "--"), (0.9, "-."), (0.95, ":")]:
        axes[0].axhline(thr, color="gray", linestyle=ls, linewidth=1)
        # mark dim
        idx = int(np.searchsorted(cum_var, thr)) + 1
        axes[0].axvline(idx, color="gray", linestyle=ls, linewidth=0.8, alpha=0.7)
        axes[0].text(idx, thr, f" {int(thr*100)}%@d={idx}", fontsize=8, va="bottom")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].set_title(f"{title}: cumulative variance")
    axes[0].set_ylim(0, 1.02)

    if len(coords):
        axes[1].hexbin(coords[:, 0], coords[:, 1], gridsize=60, cmap="viridis", mincnt=1)
        axes[1].set_xlabel("PCA-1")
        axes[1].set_ylabel("PCA-2")
        axes[1].set_title(f"{title}: PCA-1 × PCA-2 density")
    _save(fig, out, dpi=dpi)


def plot_knn(kth: np.ndarray, out: Path, title: str, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(kth, bins=50, color="#2a6f6f", edgecolor="white")
    axes[0].set_title(f"{title}: 15-NN distance histogram")
    axes[0].set_xlabel("15th neighbor distance")
    sorted_d = np.sort(kth)
    axes[1].plot(sorted_d, color="#8b3a3a")
    axes[1].set_title(f"{title}: sorted 15-NN distance")
    axes[1].set_xlabel("rank")
    axes[1].set_ylabel("distance")
    _save(fig, out, dpi=dpi)


def plot_umap(X: np.ndarray, metric: str, seed: int, n_neighbors: int, min_dist: float, max_samples: int, out: Path, title: str, dpi: int) -> int:
    if umap is None or len(X) < 5:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "UMAP unavailable or too few samples", ha="center")
        _save(fig, out, dpi=dpi)
        return 0
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        n = len(X)
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, n - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    emb = reducer.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(emb[:, 0], emb[:, 1], gridsize=70, cmap="magma", mincnt=1)
    ax.set_title(f"{title}: UMAP (n={n}, metric={metric})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    _save(fig, out, dpi=dpi)
    return n


def plot_heatmap(mat: np.ndarray, labels: list[str], out: Path, title: str, dpi: int, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0 if vmin is None else None, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title(title)
    _save(fig, out, dpi=dpi)


def _select_anchors(kth: np.ndarray, coords: np.ndarray, n_anchors: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = len(kth)
    if n == 0:
        return []
    order = np.argsort(kth)
    picks = []
    # high density = small kth
    picks.append(int(order[0]))
    picks.append(int(order[len(order) // 2]))
    picks.append(int(order[int(0.9 * (len(order) - 1))]))
    picks.append(int(order[-1]))
    # PCA extremes
    if coords is not None and len(coords) == n:
        picks.append(int(np.argmax(coords[:, 0])))
        picks.append(int(np.argmin(coords[:, 0])))
        picks.append(int(np.argmax(coords[:, 1])))
        picks.append(int(np.argmin(coords[:, 1])))
    # unique
    uniq = []
    for p in picks:
        if p not in uniq:
            uniq.append(p)
    while len(uniq) < min(n_anchors, n):
        cand = int(rng.integers(0, n))
        if cand not in uniq:
            uniq.append(cand)
    return uniq[:n_anchors]


def contact_sheet(
    image_ids: list[str],
    id_to_path: dict[str, str],
    nn_indices: np.ndarray,
    nn_distances: np.ndarray,
    anchors: list[int],
    out: Path,
    neighbors: int = 5,
) -> None:
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
            label = f"{iid[:10]}\nd={dist:.3f}" if j else f"ANCHOR\n{iid[:10]}"
            ax.set_xlabel(label, fontsize=7)
    fig.suptitle(out.stem, fontsize=11)
    _save(fig, out, dpi=150)


def run_plots(cfg: dict[str, Any]) -> None:
    seed = get_seed(cfg)
    dpi = int(cfg["analysis"].get("figure_dpi", 200))
    out_dir = Path(cfg["paths"]["outputs_dir"])
    analysis = out_dir / "analysis"
    reports = Path(cfg["paths"]["reports_dir"])
    fig_dir = reports / "figures"
    cs_dir = reports / "contact_sheets"
    ensure_dir(fig_dir)
    ensure_dir(cs_dir)
    # also project
    proj_fig = Path(cfg["paths"].get("project_reports_dir", "reports")) / "figures"
    proj_cs = Path(cfg["paths"].get("project_reports_dir", "reports")) / "contact_sheets"
    ensure_dir(proj_fig)
    ensure_dir(proj_cs)

    man = pd.read_parquet(out_dir / "manifest.parquet")
    id_to_path = dict(zip(man["image_id"].astype(str), man["absolute_path"].astype(str)))

    channels = [
        ("visual", "cosine", analysis / "visual_pca_coords.npy", analysis / "visual_knn_kth.npy", analysis / "visual_index_aligned.parquet", analysis / "visual_nn_indices.npy", analysis / "visual_nn_distances.npy", out_dir / "visual_embeddings.f32.mmap"),
        ("layout", "euclidean", analysis / "layout_pca_coords.npy", analysis / "layout_knn_kth.npy", analysis / "layout_index_aligned.parquet", analysis / "layout_nn_indices.npy", analysis / "layout_nn_distances.npy", analysis / "layout_X_scaled.npy"),
        ("recognition", "euclidean", analysis / "recognition_pca_coords.npy", analysis / "recognition_knn_kth.npy", analysis / "recognition_index_aligned.parquet", analysis / "recognition_nn_indices.npy", analysis / "recognition_nn_distances.npy", analysis / "recognition_X_scaled.npy"),
    ]

    metrics = {}
    metrics_path = reports / "feature_metrics.json"
    if metrics_path.exists():
        import json

        metrics = json.loads(metrics_path.read_text())

    for name, metric, pca_p, kth_p, idx_p, nn_i_p, nn_d_p, Xp in channels:
        if not pca_p.exists() or not kth_p.exists() or not idx_p.exists():
            continue
        coords = np.load(pca_p)
        kth = np.load(kth_p)
        idx_df = pd.read_parquet(idx_p)
        cum = metrics.get("channels", {}).get(name, {}).get("pca", {}).get("cumulative_explained_variance", [])
        if not cum and coords is not None:
            cum = [0.5, 0.7, 0.85, 0.92, 0.96]
        plot_pca_panel(coords, cum, fig_dir / f"{name}_pca", name, dpi)
        plot_pca_panel(coords, cum, proj_fig / f"{name}_pca", name, dpi)
        plot_knn(kth, fig_dir / f"{name}_knn15", name, dpi)
        plot_knn(kth, proj_fig / f"{name}_knn15", name, dpi)

        # UMAP
        if Xp.exists():
            import json

            if Xp.suffix == ".npy":
                X = np.load(Xp)
            else:
                dim = json_dim(out_dir)
                n = len(idx_df)
                X = np.asarray(np.memmap(Xp, dtype=np.float32, mode="r", shape=(n, dim)))
            n_umap = plot_umap(
                X,
                metric=metric,
                seed=seed,
                n_neighbors=int(cfg["analysis"].get("umap_n_neighbors", 30)),
                min_dist=float(cfg["analysis"].get("umap_min_dist", 0.1)),
                max_samples=int(cfg["analysis"].get("umap_max_samples", 30000)),
                out=fig_dir / f"{name}_umap",
                title=name,
                dpi=dpi,
            )
            plot_umap(
                X,
                metric=metric,
                seed=seed,
                n_neighbors=int(cfg["analysis"].get("umap_n_neighbors", 30)),
                min_dist=float(cfg["analysis"].get("umap_min_dist", 0.1)),
                max_samples=int(cfg["analysis"].get("umap_max_samples", 30000)),
                out=proj_fig / f"{name}_umap",
                title=name,
                dpi=dpi,
            )
            (fig_dir / f"{name}_umap_n.txt").write_text(str(n_umap))

        # contact sheets
        if nn_i_p.exists() and nn_d_p.exists():
            nn_i = np.load(nn_i_p)
            nn_d = np.load(nn_d_p)
            anchors = _select_anchors(kth, coords, int(cfg["analysis"].get("contact_sheet_anchors", 8)), seed)
            ids = idx_df["image_id"].astype(str).tolist()
            contact_sheet(
                ids,
                id_to_path,
                nn_i,
                nn_d,
                anchors,
                cs_dir / f"{name}_contact",
                neighbors=int(cfg["analysis"].get("contact_sheet_neighbors", 5)),
            )
            contact_sheet(
                ids,
                id_to_path,
                nn_i,
                nn_d,
                anchors,
                proj_cs / f"{name}_contact",
                neighbors=int(cfg["analysis"].get("contact_sheet_neighbors", 5)),
            )

    # Layout special plots
    lay_corr_p = analysis / "layout_feature_spearman.parquet"
    if lay_corr_p.exists():
        corr = pd.read_parquet(lay_corr_p)
        plot_heatmap(corr.values, list(corr.columns), fig_dir / "layout_feature_spearman", "Layout feature Spearman", dpi)
        plot_heatmap(corr.values, list(corr.columns), proj_fig / "layout_feature_spearman", "Layout feature Spearman", dpi)

    lay_feat = out_dir / "layout_features.parquet"
    if lay_feat.exists():
        lay = pd.read_parquet(lay_feat)
        occ_cols = [f"occupancy_{i}_{j}" for i in range(4) for j in range(4)]
        if all(c in lay.columns for c in occ_cols):
            mean_occ = lay[occ_cols].mean().values.reshape(4, 4)
            std_occ = lay[occ_cols].std().values.reshape(4, 4)
            plot_heatmap(mean_occ, [str(i) for i in range(4)], fig_dir / "layout_occupancy_mean", "4×4 mean occupancy", dpi, vmin=0, vmax=1)
            plot_heatmap(std_occ, [str(i) for i in range(4)], fig_dir / "layout_occupancy_std", "4×4 occupancy std", dpi)
            plot_heatmap(mean_occ, [str(i) for i in range(4)], proj_fig / "layout_occupancy_mean", "4×4 mean occupancy", dpi, vmin=0, vmax=1)
            plot_heatmap(std_occ, [str(i) for i in range(4)], proj_fig / "layout_occupancy_std", "4×4 occupancy std", dpi)

    # Recognition PCA-target heatmap
    rec_sp = analysis / "recognition_pca_target_spearman.parquet"
    if rec_sp.exists():
        df = pd.read_parquet(rec_sp)
        piv = df.pivot(index="pca", columns="feature", values="spearman")
        plot_heatmap(piv.values.astype(float), list(piv.columns), fig_dir / "recognition_pca_target_spearman", "Recognition PCA vs stats Spearman", dpi)
        plot_heatmap(piv.values.astype(float), list(piv.columns), proj_fig / "recognition_pca_target_spearman", "Recognition PCA vs stats Spearman", dpi)

    # Cross channel
    cross_p = analysis / "cross_channel_spearman.npy"
    if cross_p.exists():
        mat = np.load(cross_p)
        labels = ["visual", "layout", "recognition"]
        plot_heatmap(mat, labels, fig_dir / "cross_channel_distance_spearman", "Cross-channel distance Spearman", dpi)
        plot_heatmap(mat, labels, proj_fig / "cross_channel_distance_spearman", "Cross-channel distance Spearman", dpi)


def json_dim(out_dir: Path) -> int:
    import json

    meta = out_dir / "visual_embeddings.f32.meta.json"
    if meta.exists():
        return int(json.loads(meta.read_text())["dim"])
    return 896


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_plots(cfg)
    print("[plot_features] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
