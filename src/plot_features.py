"""Plotting for independent feature channels (PNG only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .config import get_seed, load_config
from .utils import ensure_dir

try:
    import umap
except Exception as e:  # noqa: BLE001
    umap = None
    _UMAP_IMPORT_ERROR = e
else:
    _UMAP_IMPORT_ERROR = None


def _save(fig, path_no_ext: Path, dpi: int = 200) -> None:
    ensure_dir(path_no_ext.parent)
    fig.savefig(f"{path_no_ext}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pca_panel(coords: np.ndarray, cum_var: list[float], out: Path, title: str, dpi: int) -> None:
    if not cum_var:
        raise RuntimeError(f"{title}: missing cumulative explained variance for PCA panel")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(1, len(cum_var) + 1)
    axes[0].plot(x, cum_var, color="#1f4e79")
    for thr, ls in [(0.8, "--"), (0.9, "-."), (0.95, ":")]:
        axes[0].axhline(thr, color="gray", linestyle=ls, linewidth=1)
        idx = int(np.searchsorted(cum_var, thr)) + 1
        axes[0].axvline(idx, color="gray", linestyle=ls, linewidth=0.8, alpha=0.7)
        axes[0].text(idx, thr, f" {int(thr * 100)}%@d={idx}", fontsize=8, va="bottom")
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


def plot_pca_colored(
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
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title(title)
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


def plot_umap(
    X: np.ndarray,
    metric: str,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    max_samples: int,
    out: Path,
    title: str,
    dpi: int,
) -> tuple[int, dict[str, Any]]:
    if umap is None:
        raise RuntimeError(
            f"UMAP required for {title} but umap-learn is not installed: {_UMAP_IMPORT_ERROR}. "
            "Install with: uv sync --extra analysis"
        )
    if len(X) < 5:
        raise RuntimeError(f"UMAP requires at least 5 samples for {title}; got {len(X)}")
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        n = len(X)
    params = {
        "n_neighbors": int(min(n_neighbors, max(2, n - 1))),
        "min_dist": float(min_dist),
        "metric": metric,
        "random_state": int(seed),
        "n_samples": int(n),
    }
    reducer = umap.UMAP(
        n_neighbors=params["n_neighbors"],
        min_dist=params["min_dist"],
        metric=params["metric"],
        random_state=params["random_state"],
    )
    emb = reducer.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(emb[:, 0], emb[:, 1], gridsize=70, cmap="magma", mincnt=1)
    ax.set_title(f"{title}: UMAP (n={n}, metric={metric})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    _save(fig, out, dpi=dpi)
    return n, params


def plot_heatmap_annot(
    mat: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    out: Path,
    title: str,
    dpi: int,
    *,
    annot: bool = True,
    vmin=None,
    vmax=None,
) -> None:
    h = max(4.0, 0.35 * len(ylabels) + 2)
    w = max(5.0, 0.35 * len(xlabels) + 2)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(
        mat,
        annot=annot,
        fmt=".2f" if annot else "",
        xticklabels=xlabels,
        yticklabels=ylabels,
        cmap="coolwarm",
        center=0 if vmin is None else None,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_title(title)
    _save(fig, out, dpi=dpi)


def plot_layout_corr_clustered(corr: pd.DataFrame, out: Path, dpi: int) -> None:
    """Hierarchical-clustered Spearman heatmap without per-cell numbers."""
    mat = corr.values.astype(float)
    mat = np.nan_to_num(mat, nan=0.0)
    # distance = 1 - |corr|
    dist = 1.0 - np.abs(mat)
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, None)
    # symmetrize
    dist = (dist + dist.T) / 2
    try:
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
    except Exception:
        order = np.arange(len(corr.columns))
    labels = [corr.columns[i] for i in order]
    reordered = mat[np.ix_(order, order)]
    plot_heatmap_annot(
        reordered,
        labels,
        labels,
        out,
        "Layout feature Spearman (clustered, |ρ|)",
        dpi,
        annot=False,
        vmin=-1,
        vmax=1,
    )


def plot_feature_histograms(df: pd.DataFrame, cols: list[str], out: Path, title: str, dpi: int) -> None:
    use = [c for c in cols if c in df.columns][:12]
    if not use:
        return
    n = len(use)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.atleast_2d(axes)
    for i, c in enumerate(use):
        ax = axes[i // ncols, i % ncols]
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        ax.hist(vals, bins=40, color="#3d5a80", edgecolor="white")
        ax.set_title(c, fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(title)
    _save(fig, out, dpi=dpi)


def _select_anchors(kth: np.ndarray, coords: np.ndarray, n_anchors: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = len(kth)
    if n == 0:
        return []
    order = np.argsort(kth)
    picks = [
        int(order[0]),
        int(order[len(order) // 2]),
        int(order[int(0.9 * (len(order) - 1))]),
        int(order[-1]),
    ]
    if coords is not None and len(coords) == n:
        picks.extend(
            [
                int(np.argmax(coords[:, 0])),
                int(np.argmin(coords[:, 0])),
                int(np.argmax(coords[:, 1])),
                int(np.argmin(coords[:, 1])),
            ]
        )
    uniq: list[int] = []
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


def json_dim(out_dir: Path) -> int:
    meta = out_dir / "visual_embeddings.f32.meta.json"
    if meta.exists():
        return int(json.loads(meta.read_text())["dim"])
    return 896


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

    man = pd.read_parquet(out_dir / "manifest.parquet")
    id_to_path = dict(zip(man["image_id"].astype(str), man["absolute_path"].astype(str)))

    channels = [
        (
            "visual",
            "cosine",
            analysis / "visual_pca_coords.npy",
            analysis / "visual_knn_kth.npy",
            analysis / "visual_index_aligned.parquet",
            analysis / "visual_nn_indices.npy",
            analysis / "visual_nn_distances.npy",
            out_dir / "visual_embeddings.f32.mmap",
        ),
        (
            "layout",
            "euclidean",
            analysis / "layout_pca_coords.npy",
            analysis / "layout_knn_kth.npy",
            analysis / "layout_index_aligned.parquet",
            analysis / "layout_nn_indices.npy",
            analysis / "layout_nn_distances.npy",
            analysis / "layout_X_scaled.npy",
        ),
        (
            "recognition",
            "euclidean",
            analysis / "recognition_pca_coords.npy",
            analysis / "recognition_knn_kth.npy",
            analysis / "recognition_index_aligned.parquet",
            analysis / "recognition_nn_indices.npy",
            analysis / "recognition_nn_distances.npy",
            analysis / "recognition_X_scaled.npy",
        ),
    ]

    metrics: dict[str, Any] = {}
    metrics_path = reports / "feature_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    umap_meta: dict[str, Any] = {}

    for name, metric, pca_p, kth_p, idx_p, nn_i_p, nn_d_p, Xp in channels:
        if not pca_p.exists() or not kth_p.exists() or not idx_p.exists():
            continue
        coords = np.load(pca_p)
        kth = np.load(kth_p)
        idx_df = pd.read_parquet(idx_p)
        cum = metrics.get("channels", {}).get(name, {}).get("pca", {}).get("cumulative_explained_variance", [])
        if not cum:
            raise RuntimeError(f"Missing PCA cumulative variance for channel={name}; re-run analyze")
        plot_pca_panel(coords, cum, fig_dir / f"{name}_pca", name, dpi)
        plot_knn(kth, fig_dir / f"{name}_knn15", name, dpi)

        if not Xp.exists():
            raise RuntimeError(f"Missing feature matrix for UMAP: {Xp}")
        if Xp.suffix == ".npy":
            X = np.load(Xp)
        else:
            dim = json_dim(out_dir)
            n = len(idx_df)
            X = np.asarray(np.memmap(Xp, dtype=np.float32, mode="r", shape=(n, dim)))
        _, params = plot_umap(
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
        umap_meta[name] = params

        if nn_i_p.exists() and nn_d_p.exists():
            nn_i = np.load(nn_i_p)
            nn_d = np.load(nn_d_p)
            anchors = _select_anchors(kth, coords, int(cfg["analysis"].get("contact_sheet_anchors", 8)), seed)
            ids = idx_df["image_id"].astype(str).tolist()
            # nearest + outlier sheets
            near_a = anchors[: max(1, len(anchors) // 2)]
            far_a = anchors[max(1, len(anchors) // 2) :]
            contact_sheet(ids, id_to_path, nn_i, nn_d, near_a, cs_dir / f"{name}_contact", neighbors=int(cfg["analysis"].get("contact_sheet_neighbors", 5)))
            if far_a:
                contact_sheet(ids, id_to_path, nn_i, nn_d, far_a, cs_dir / f"{name}_outliers", neighbors=int(cfg["analysis"].get("contact_sheet_neighbors", 5)))

    # Persist UMAP params into metrics
    if umap_meta:
        metrics["umap"] = umap_meta
        (reports / "feature_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # Visual PCA colored by diagnostic labels (not fused into vector)
    vis_coords_p = analysis / "visual_coords.parquet"
    if vis_coords_p.exists():
        vcdf = pd.read_parquet(vis_coords_p)
        coords = vcdf[["pca1", "pca2"]].to_numpy()
        ids = vcdf["image_id"].astype(str)
        # join diagnostics
        diag = man[["image_id", "width", "height"]].copy() if {"width", "height"}.issubset(man.columns) else man[["image_id"]].copy()
        diag["image_id"] = diag["image_id"].astype(str)
        if {"width", "height"}.issubset(diag.columns):
            diag["aspect_ratio"] = pd.to_numeric(diag["width"], errors="coerce") / pd.to_numeric(diag["height"], errors="coerce").replace(0, np.nan)
        lay_p = out_dir / "layout_features.parquet"
        if lay_p.exists():
            lay = pd.read_parquet(lay_p)
            lay["image_id"] = lay["image_id"].astype(str)
            keep = [
                c
                for c in (
                    "content_area_ratio",
                    "blank_ratio_diag",
                    "formula_block_count",
                    "figure_block_count",
                    "block_count",
                )
                if c in lay.columns
            ]
            diag = diag.merge(lay[["image_id"] + keep], on="image_id", how="left")
        rec_p = out_dir / "recognition_features.parquet"
        if rec_p.exists():
            rec = pd.read_parquet(rec_p)
            rec["image_id"] = rec["image_id"].astype(str)
            if "output_character_count" in rec.columns:
                diag = diag.merge(rec[["image_id", "output_character_count"]], on="image_id", how="left")
        merged = vcdf[["image_id"]].assign(image_id=ids).merge(diag, on="image_id", how="left")
        color_fields = [
            ("aspect_ratio", "aspect_ratio"),
            ("content_area_ratio", "content_coverage"),
            ("blank_ratio_diag", "blank_ratio"),
            ("formula_block_count", "formula_blocks"),
            ("figure_block_count", "figure_blocks"),
            ("output_character_count", "ocr_chars"),
        ]
        for col, tag in color_fields:
            if col not in merged.columns:
                continue
            vals = pd.to_numeric(merged[col], errors="coerce").to_numpy()
            if np.isfinite(vals).sum() < 5:
                continue
            vals = np.nan_to_num(vals, nan=float(np.nanmedian(vals)))
            plot_pca_colored(coords, vals, fig_dir / f"visual_pca_by_{tag}", f"Visual PCA by {tag}", tag, dpi)

    # Layout corr + distributions
    lay_corr_p = analysis / "layout_feature_spearman.parquet"
    if lay_corr_p.exists():
        corr = pd.read_parquet(lay_corr_p)
        plot_layout_corr_clustered(corr, fig_dir / "layout_feature_spearman", dpi)

    lay_feat = out_dir / "layout_features.parquet"
    if lay_feat.exists():
        lay = pd.read_parquet(lay_feat)
        lay = lay[lay["layout_available"].astype(bool)] if "layout_available" in lay.columns else lay
        occ_cols = [f"occupancy_{i}_{j}" for i in range(4) for j in range(4)]
        if all(c in lay.columns for c in occ_cols):
            mean_occ = lay[occ_cols].mean().values.reshape(4, 4)
            std_occ = lay[occ_cols].std().values.reshape(4, 4)
            plot_heatmap_annot(mean_occ, [str(i) for i in range(4)], [str(i) for i in range(4)], fig_dir / "layout_occupancy_mean", "4×4 mean occupancy", dpi, annot=True, vmin=0, vmax=1)
            plot_heatmap_annot(std_occ, [str(i) for i in range(4)], [str(i) for i in range(4)], fig_dir / "layout_occupancy_std", "4×4 occupancy std", dpi, annot=True)
        hist_cols = [
            "content_area_ratio",
            "block_count",
            "formula_block_count",
            "figure_block_count",
            "mean_horizontal_gap",
            "two_column_score",
            "reading_order_inversion_count",
            "block_overlap_ratio",
        ]
        plot_feature_histograms(lay, hist_cols, fig_dir / "layout_feature_hist", "Layout feature distributions", dpi)

    # Recognition content PC×feature (rows=PC)
    rec_sp = analysis / "recognition_pca_target_spearman.parquet"
    if rec_sp.exists():
        df = pd.read_parquet(rec_sp)
        piv = df.pivot(index="pca", columns="feature", values="spearman")
        plot_heatmap_annot(
            piv.values.astype(float),
            list(piv.columns),
            list(piv.index),
            fig_dir / "recognition_pca_content_spearman",
            "Recognition PCA vs content metrics (Spearman)",
            dpi,
            annot=True,
        )

    # Quality metrics separate plot
    q_sp = analysis / "recognition_quality_spearman.parquet"
    if q_sp.exists():
        df = pd.read_parquet(q_sp)
        piv = df.pivot(index="pca", columns="feature", values="spearman")
        plot_heatmap_annot(
            piv.values.astype(float),
            list(piv.columns),
            list(piv.index),
            fig_dir / "recognition_pca_quality_spearman",
            "Recognition PCA vs QUALITY metrics (diagnostic only)",
            dpi,
            annot=True,
        )

    # OCR quality status distribution
    q_path = out_dir / "ocr_quality.parquet"
    if q_path.exists():
        qdf = pd.read_parquet(q_path)
        fig, ax = plt.subplots(figsize=(7, 4))
        vc = qdf["ocr_quality_status"].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color="#9a031e")
        ax.set_title("OCR quality status counts")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, fig_dir / "ocr_quality_status", dpi=dpi)

    cross_p = analysis / "cross_channel_spearman.npy"
    if cross_p.exists():
        mat = np.load(cross_p)
        labels = ["visual", "layout", "recognition"]
        plot_heatmap_annot(mat, labels, labels, fig_dir / "cross_channel_distance_spearman", "Cross-channel distance Spearman", dpi, annot=True)


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
