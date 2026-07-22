"""V2 figures."""

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

try:
    import umap as umap_lib
except Exception:  # noqa: BLE001
    umap_lib = None


def _save(fig, path: Path, dpi: int = 150) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_prevalence(prev: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(prev["feature"], prev["rate"], color="#3d5a80")
    ax.axhline(0.01, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(0.99, color="red", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("page rate")
    ax.set_title("Binary type feature prevalence")
    plt.xticks(rotation=45, ha="right")
    _save(fig, out)


def plot_cluster_sizes(sizes: np.ndarray, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(sizes))
    ax.bar(xs, sizes, color="#1f4e79")
    total = max(int(sizes.sum()), 1)
    for i, s in enumerate(sizes):
        ax.text(i, s, f"{int(s)}\n({100*s/total:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"C{i}" for i in xs])
    ax.set_title(title)
    _save(fig, out)


def plot_sizes_comparison(size_map: dict[str, np.ndarray], out: Path) -> None:
    keys = list(size_map.keys())
    k = len(next(iter(size_map.values())))
    x = np.arange(k)
    width = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, key in enumerate(keys):
        ax.bar(x + i * width, size_map[key], width=width, label=key)
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels([f"C{i}" for i in range(k)])
    ax.legend(fontsize=8)
    ax.set_title("K=4 cluster sizes comparison")
    _save(fig, out)


def plot_type_heatmap(
    binary_fit: np.ndarray,
    feature_names: list[str],
    labels: np.ndarray,
    out: Path,
    title: str = "Type prevalence by cluster (K=4)",
) -> None:
    k = int(labels.max()) + 1
    mat = np.zeros((k, len(feature_names)))
    for c in range(k):
        m = labels == c
        if m.any():
            mat[c] = binary_fit[m].mean(axis=0)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(feature_names)), 0.7 * k + 2))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{i}" for i in range(k)])
    ax.set_title(title)
    for i in range(k):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{100*mat[i,j]:.0f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)


def plot_depth_box(depth: np.ndarray, labels: np.ndarray, out: Path, ylabel: str = "max_ast_depth") -> None:
    k = int(labels.max()) + 1
    data = [depth[labels == c] for c in range(k)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=[f"C{c}" for c in range(k)], showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by cluster")
    _save(fig, out)


def plot_crosstab(row_lab: np.ndarray, col_lab: np.ndarray, out: Path, title: str) -> pd.DataFrame:
    ct = pd.crosstab(row_lab, col_lab, normalize="index")
    counts = pd.crosstab(row_lab, col_lab)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ct.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("joint cluster")
    ax.set_ylabel("visual cluster")
    ax.set_title(title)
    ax.set_xticks(range(ct.shape[1]))
    ax.set_yticks(range(ct.shape[0]))
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, str(int(counts.iloc[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)
    return counts


def plot_shared_visual_space(
    coords: np.ndarray,
    lab_a: np.ndarray,
    lab_b: np.ndarray,
    out: Path,
    title_a: str,
    title_b: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lab, title in ((axes[0], lab_a, title_a), (axes[1], lab_b, title_b)):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=lab, s=5, cmap="tab10", linewidths=0, alpha=0.7)
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.046)
    fig.suptitle("Same visual UMAP coordinates, different labels")
    _save(fig, out)


def fit_visual_umap(X_img: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    if umap_lib is None:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=int(cfg.get("random_state", 42))).fit_transform(X_img)
    u = cfg.get("umap") or {}
    reducer = umap_lib.UMAP(
        n_neighbors=int(u.get("n_neighbors", 15)),
        min_dist=float(u.get("min_dist", 0.1)),
        metric=str(u.get("metric", "euclidean")),
        random_state=int(cfg.get("random_state", 42)),
    )
    return reducer.fit_transform(X_img)


def plot_joint_umap(X: np.ndarray, labels: np.ndarray, out: Path, cfg: dict[str, Any]) -> None:
    coords = fit_visual_umap(X, cfg)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=5, cmap="tab10", linewidths=0, alpha=0.7)
    fig.colorbar(sc, ax=ax, fraction=0.046)
    ax.set_title("Joint features UMAP (viz only)")
    _save(fig, out)


def _active_types(row: pd.Series, features: list[str], limit: int = 4) -> str:
    hits = [f.replace("has_", "") for f in features if int(row.get(f, 0) or 0) == 1]
    return ",".join(hits[:limit]) if hits else "-"


def write_rep_sheets(
    page_df: pd.DataFrame,
    labels: np.ndarray,
    dists: np.ndarray,
    type_features: list[str],
    out_dir: Path,
    *,
    center_n: int = 20,
    outlier_n: int = 12,
    prefix: str = "",
) -> None:
    ensure_dir(out_dir)
    k = int(labels.max()) + 1
    for c in range(k):
        members = np.where(labels == c)[0]
        order = np.argsort(dists[members])
        for kind, ids in (
            ("center", members[order[:center_n]]),
            ("outlier", members[order[-outlier_n:][::-1]]),
        ):
            rows = []
            for i in ids:
                r = page_df.iloc[i]
                rows.append(
                    {
                        "page_id": r["page_id"],
                        "image_path": r["image_path"],
                        "cluster": c,
                        "dist": float(dists[i]),
                        "max_ast_depth": r.get("max_ast_depth"),
                        "types": _active_types(r, type_features),
                    }
                )
            _sheet(rows, f"{prefix}C{c} {kind}", out_dir / f"cluster_{c}_{kind}.png")


def _sheet(rows: list[dict[str, Any]], title: str, out: Path, ncols: int = 5) -> None:
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
        ax.set_xlabel(
            f"{str(r.get('page_id',''))[:10]}\nC{r.get('cluster')} d={r.get('dist',0):.2f}\n"
            f"depth={r.get('max_ast_depth')} {r.get('types')}",
            fontsize=6,
        )
    fig.suptitle(title, fontsize=11)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
