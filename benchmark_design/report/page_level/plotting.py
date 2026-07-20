"""Plotting for page-level aspect ratio and foreground density."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_design.config.page_level import DEFAULT_ASPECT_RATIO_BINS
from benchmark_design.page_level.models import ImageFeatureRow
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot

FIGURE_DPI = 300

FOREGROUND_DENSITY_BIN_EDGES: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, float("inf"))
FOREGROUND_DENSITY_BIN_LABELS: tuple[str, ...] = (
    "<2%",
    "2–4%",
    "4–6%",
    "6–8%",
    "8–10%",
    "≥10%",
)


def _foreground_density_bin_index(density: float) -> int:
    for index in range(len(FOREGROUND_DENSITY_BIN_LABELS)):
        lower = FOREGROUND_DENSITY_BIN_EDGES[index]
        upper = FOREGROUND_DENSITY_BIN_EDGES[index + 1]
        if density >= lower and density < upper:
            return index
    return len(FOREGROUND_DENSITY_BIN_LABELS) - 1


def _foreground_density_bin_counts(densities: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(FOREGROUND_DENSITY_BIN_LABELS), dtype=np.int64)
    for value in densities:
        counts[_foreground_density_bin_index(float(value))] += 1
    return counts

ASPECT_GROUP_LABELS: dict[str, str] = {
    "portrait": "Portrait\nR < 0.90",
    "near_square": "Near square\n0.90 ≤ R < 1.20",
    "landscape": "Landscape\nR ≥ 1.20",
    "all": "All layouts",
    "other": "Other",
}


@with_locked_pyplot
def plot_aspect_ratio_distribution(features: list[ImageFeatureRow], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _configure_matplotlib_fonts(plt)
    ratios = np.array([row.aspect_ratio for row in features], dtype=np.float64)
    groups = [row.aspect_ratio_group for row in features]
    order = [bin_.name for bin_ in DEFAULT_ASPECT_RATIO_BINS] + ["other"]
    counts = {name: groups.count(name) for name in order}
    labels = [ASPECT_GROUP_LABELS.get(name, name).replace("\n", " ") for name in order if counts[name]]
    values = [counts[name] for name in order if counts[name]]
    total = len(features)

    fig = plt.figure(figsize=(10, 4.5), dpi=FIGURE_DPI)
    gs = GridSpec(1, 2, width_ratios=[1.4, 1], wspace=0.25)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    bars = ax_bar.bar(range(len(values)), values, color="#4472C4", alpha=0.9)
    ax_bar.set_xticks(range(len(values)))
    ax_bar.set_xticklabels([label.split(" ")[0] for label in labels], rotation=20, ha="right")
    ax_bar.set_ylabel("Images")
    ax_bar.set_title("Layout groups (candidate bins)")
    for bar, count in zip(bars, values, strict=True):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({100 * count / total:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_hist.hist(ratios, bins=50, color="#70AD47", alpha=0.85, edgecolor="white")
    for edge in (0.90, 1.20):
        ax_hist.axvline(edge, color="#C00000", linestyle="--", linewidth=1)
    ax_hist.set_xlabel("Aspect ratio (W/H)")
    ax_hist.set_ylabel("Images")
    ax_hist.set_title("Full-range distribution")
    fig.suptitle(f"Aspect ratio ($n={total:,}$)", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _page_foreground_density_xlabel(*, gray_threshold: float | None = None) -> str:
    threshold_note = f"t_I = {gray_threshold:.0f}" if gray_threshold is not None else "t_I"
    return (
        "Foreground density interval "
        f"(I ≤ {threshold_note}; global pooled Otsu on dataset grayscale histogram)"
    )


@with_locked_pyplot
def plot_foreground_density_distribution(
    features: list[ImageFeatureRow],
    output_path: Path,
    *,
    gray_threshold: float | None = None,
) -> None:
    """Interval distribution of page foreground density."""
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    densities = np.array([row.foreground_density for row in features], dtype=np.float64)
    n = int(densities.size)
    if n == 0:
        raise ValueError("No foreground density values to plot")

    counts = _foreground_density_bin_counts(densities)
    ratios = counts / n

    x = np.arange(len(FOREGROUND_DENSITY_BIN_LABELS))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIGURE_DPI)

    bars = ax.bar(
        x,
        counts,
        color="#4472C4",
        alpha=0.85,
        edgecolor="white",
        width=0.72,
        label="Pages per bin",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(FOREGROUND_DENSITY_BIN_LABELS)
    ax.set_ylabel("Images")
    ax.set_ylim(0, max(int(counts.max()) * 1.15, 1))
    ax.set_title(f"Page foreground density — interval distribution ($n={n:,}$)")
    ax.set_xlabel(_page_foreground_density_xlabel(gray_threshold=gray_threshold))

    for bar, count, ratio in zip(bars, counts, ratios, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}\n({ratio * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_paper_figures(
    features: list[ImageFeatureRow],
    paper_dir: Path,
    *,
    gray_threshold: float | None = None,
) -> dict[str, Path]:
    paper_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    path = paper_dir / "aspect_ratio_distribution.png"
    plot_aspect_ratio_distribution(features, path)
    outputs["aspect_ratio_distribution"] = path

    path = paper_dir / "foreground_density_distribution.png"
    plot_foreground_density_distribution(features, path, gray_threshold=gray_threshold)
    outputs["foreground_density_distribution"] = path

    return outputs
