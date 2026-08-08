"""Plot K-selection curves (elbow distortion + cosine silhouette diagnostic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def plot_k_selection_curves(
    curve: list[dict[str, Any]],
    *,
    selected_k: int,
    out_path: Path,
    title: str | None = None,
    group_id: str | None = None,
    max_silhouette_k: int | None = None,
) -> Path:
    """
    Save a 2-panel figure:
      top: K vs spherical_kmeans_total_cosine_distortion (elbow selects K)
      bottom: K vs mean cosine silhouette (**diagnostic only**)
    """
    if not curve:
        raise ValueError("empty curve")
    ks = [int(c["k"]) for c in curve]
    distortions = [
        float(c.get("spherical_kmeans_total_cosine_distortion", c.get("distortion_total", c.get("distortion"))))
        for c in curve
    ]
    sils = [c.get("fused_silhouette") for c in curve]
    has_sil = all(s is not None for s in sils)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, constrained_layout=True)
    ax0, ax1 = axes

    ax0.plot(
        ks,
        distortions,
        "o-",
        color="#1f4e79",
        linewidth=1.8,
        markersize=4,
        label="spherical_kmeans_total_cosine_distortion (= N − J)",
    )
    if selected_k in ks:
        i = ks.index(selected_k)
        ax0.axvline(selected_k, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.85)
        ax0.scatter(
            [selected_k],
            [distortions[i]],
            color="#c0392b",
            s=60,
            zorder=5,
            label=f"elbow K={selected_k}",
        )
    ax0.set_ylabel("Spherical K-means total cosine distortion")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=8)
    ax0.set_title(
        title
        or "elbow method: max perpendicular distance to normalized endpoint chord"
    )

    if has_sil:
        sil_vals = [float(s) for s in sils]  # type: ignore[arg-type]
        ax1.plot(
            ks,
            sil_vals,
            "o-",
            color="#0e7c61",
            linewidth=1.8,
            markersize=4,
            label="mean cosine silhouette (diagnostic only)",
        )
        if selected_k in ks:
            ax1.axvline(selected_k, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.85)
            ax1.scatter(
                [selected_k],
                [sil_vals[ks.index(selected_k)]],
                color="#c0392b",
                s=60,
                zorder=5,
                label=f"elbow K={selected_k}",
            )
        best_sil_i = int(np.argmax(np.asarray(sil_vals)))
        sil_k = ks[best_sil_i] if max_silhouette_k is None else int(max_silhouette_k)
        if sil_k in ks and sil_k != selected_k:
            ax1.scatter(
                [sil_k],
                [sil_vals[ks.index(sil_k)]],
                color="#f39c12",
                s=60,
                zorder=5,
                marker="D",
                label=f"max silhouette K={sil_k} (diagnostic only)",
            )
        sil_n = next(
            (c.get("silhouette_sample_size") for c in curve if c.get("silhouette_sample_size")),
            None,
        )
        ylab = "Cosine silhouette (diagnostic only)"
        if sil_n is not None:
            ylab = f"Cosine silhouette (diagnostic; n={int(sil_n)} stratified)"
        ax1.set_ylabel(ylab)
        ax1.legend(loc="best", fontsize=8)
    else:
        ax1.text(
            0.5,
            0.5,
            "silhouette not available (diagnostic only)",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_ylabel("Cosine silhouette (diagnostic only)")

    ax1.set_xlabel("K")
    # show all integer ticks when range is moderate; otherwise every 2nd
    if len(ks) <= 55:
        ax1.set_xticks(ks if len(ks) <= 30 else ks[::2])
    ax1.grid(True, alpha=0.3)
    if group_id:
        fig.suptitle(str(group_id), fontsize=10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    sil_only = out_path.with_name("k_vs_cosine_silhouette.png")
    if has_sil:
        fig2, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
        sil_vals = [float(s) for s in sils]  # type: ignore[arg-type]
        ax.plot(ks, sil_vals, "o-", color="#0e7c61", linewidth=1.8, markersize=4)
        if selected_k in ks:
            ax.axvline(selected_k, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.85)
            ax.scatter(
                [selected_k],
                [sil_vals[ks.index(selected_k)]],
                color="#c0392b",
                s=60,
                zorder=5,
                label=f"elbow K={selected_k}",
            )
        best_sil_i = int(np.argmax(np.asarray(sil_vals)))
        sil_k = ks[best_sil_i]
        if sil_k != selected_k:
            ax.scatter(
                [sil_k],
                [sil_vals[best_sil_i]],
                color="#f39c12",
                s=60,
                zorder=5,
                marker="D",
                label=f"max silhouette K={sil_k} (diagnostic only)",
            )
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("K")
        sil_n = next(
            (c.get("silhouette_sample_size") for c in curve if c.get("silhouette_sample_size")),
            None,
        )
        if sil_n is not None:
            ax.set_ylabel(f"Mean cosine silhouette (n={int(sil_n)} stratified; diagnostic)")
        else:
            ax.set_ylabel("Mean cosine silhouette (all points; diagnostic only)")
        if len(ks) <= 55:
            ax.set_xticks(ks if len(ks) <= 30 else ks[::2])
        ax.grid(True, alpha=0.3)
        ax.set_title("Cosine silhouette vs K — diagnostic only (does not select K)")
        fig2.savefig(sil_only, dpi=140, bbox_inches="tight")
        plt.close(fig2)

    return out_path
