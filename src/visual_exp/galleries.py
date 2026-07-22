"""Cluster / boundary / outlier image galleries."""

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
from .io_util import load_aligned_embeddings


def _sheet(paths: list[str], title: str, out: Path, ncols: int = 4) -> None:
    n = len(paths)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes = np.atleast_2d(axes)
    thumb = 160
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        p = paths[i]
        if p and Path(p).exists():
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb, thumb))
                ax.imshow(im)
            except Exception:
                ax.text(0.5, 0.5, "err", ha="center")
        else:
            ax.text(0.5, 0.5, "missing", ha="center")
    fig.suptitle(title, fontsize=11)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_galleries(cfg: dict[str, Any]) -> None:
    gal = Path(cfg["paths"]["galleries_dir"])
    clus = Path(cfg["paths"]["clustering_dir"])
    n_show = int(cfg["analysis"].get("gallery_n", 16))
    seed = int(cfg.get("random_seed", 42))
    rng = np.random.default_rng(seed)

    X, idx, _ = load_aligned_embeddings(cfg)
    ids = idx["image_id"].astype(str).tolist()
    man = pd.read_parquet(Path(cfg["paths"]["metadata_dir"]) / "manifest.parquet")
    id_to_path = dict(zip(man["image_id"].astype(str), man["image_path"].astype(str)))

    assign = pd.read_parquet(clus / "cluster_assignments.parquet")
    labels = assign.set_index("image_id").loc[ids, "cluster"].to_numpy()
    k = int(labels.max()) + 1

    outliers = set()
    op = clus / "hdbscan_outliers.parquet"
    if op.exists():
        odf = pd.read_parquet(op)
        outliers = set(odf.loc[odf["is_outlier"].astype(bool), "image_id"].astype(str))

    for c in range(k):
        members = np.where(labels == c)[0]
        center = X[members].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)
        sims = X[members] @ center
        order = np.argsort(-sims)  # closest to center first
        center_ids = [ids[members[i]] for i in order[:n_show]]
        boundary_ids = [ids[members[i]] for i in order[-n_show:][::-1]]
        rand_ids = [ids[i] for i in rng.choice(members, size=min(n_show, len(members)), replace=False)]

        # confusion: members closest to other cluster centers
        other_centers = []
        for oc in range(k):
            if oc == c:
                continue
            om = np.where(labels == oc)[0]
            oc_c = X[om].mean(axis=0)
            oc_c = oc_c / (np.linalg.norm(oc_c) + 1e-12)
            other_centers.append(oc_c)
        conf_ids = []
        if other_centers:
            OC = np.stack(other_centers, axis=0)
            conf_score = (X[members] @ OC.T).max(axis=1)
            conf_order = np.argsort(-conf_score)
            conf_ids = [ids[members[i]] for i in conf_order[:n_show]]

        _sheet(
            [id_to_path.get(i, "") for i in center_ids],
            f"cluster_{c} centers",
            gal / "cluster_centers" / f"cluster_{c}.png",
        )
        _sheet(
            [id_to_path.get(i, "") for i in boundary_ids],
            f"cluster_{c} boundaries",
            gal / "cluster_boundaries" / f"cluster_{c}.png",
        )
        _sheet(
            [id_to_path.get(i, "") for i in rand_ids],
            f"cluster_{c} random",
            gal / "cluster_centers" / f"cluster_{c}_random.png",
        )
        if conf_ids:
            _sheet(
                [id_to_path.get(i, "") for i in conf_ids],
                f"cluster_{c} confused",
                gal / "cluster_boundaries" / f"cluster_{c}_confused.png",
            )

    if outliers:
        oids = list(outliers)[: max(n_show * 2, 32)]
        _sheet(
            [id_to_path.get(i, "") for i in oids],
            "HDBSCAN outliers",
            gal / "outliers" / "hdbscan_outliers.png",
        )

    print(f"[galleries] wrote sheets for k={k}")
