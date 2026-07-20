"""Plot block-level foreground density interval distributions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_design.block_level.block_foreground_density import BlockForegroundDensityRow
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.page_level.plotting import FIGURE_DPI
from benchmark_design.report.pyplot_lock import with_locked_pyplot

BLOCK_FOREGROUND_DENSITY_BIN_LABELS: tuple[str, ...] = (
    "<4%",
    "4–6%",
    "6–8%",
    "8–10%",
    "10–12%",
    "12–15%",
    "≥15%",
)


def _block_foreground_density_bin_index(density: float) -> int:
    if density < 0.04:
        return 0
    if density < 0.06:
        return 1
    if density < 0.08:
        return 2
    if density < 0.10:
        return 3
    if density < 0.12:
        return 4
    if density < 0.15:
        return 5
    return 6


def _block_foreground_density_bin_counts(densities: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(BLOCK_FOREGROUND_DENSITY_BIN_LABELS), dtype=np.int64)
    for value in densities:
        counts[_block_foreground_density_bin_index(float(value))] += 1
    return counts


@with_locked_pyplot
def _block_foreground_density_xlabel(*, gray_threshold: float | None = None) -> str:
    threshold_note = f"t_I = {gray_threshold:.0f}" if gray_threshold is not None else "t_I"
    return (
        "Foreground density interval "
        f"(Txtblock annotation mask; I ≤ {threshold_note}; "
        "global pooled Otsu on dataset grayscale histogram)"
    )


def plot_block_foreground_density_distribution(
    rows: list[BlockForegroundDensityRow],
    output_path: Path,
    *,
    title_suffix: str = "",
    gray_threshold: float | None = None,
) -> None:
    """Plot block foreground density bins (denominator = annotation mask pixels)."""
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    densities = np.array([row.foreground_density for row in rows], dtype=np.float64)
    n = int(densities.size)
    if n == 0:
        raise ValueError("No block foreground density values to plot")

    counts = _block_foreground_density_bin_counts(densities)
    ratios = counts / n

    x = np.arange(len(BLOCK_FOREGROUND_DENSITY_BIN_LABELS))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIGURE_DPI)

    bars = ax.bar(
        x,
        counts,
        color="#4472C4",
        alpha=0.85,
        edgecolor="white",
        width=0.72,
        label="Blocks per bin",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(BLOCK_FOREGROUND_DENSITY_BIN_LABELS)
    ax.set_ylabel("Blocks")
    ax.set_ylim(0, max(int(counts.max()) * 1.15, 1))
    title = f"Txtblock foreground density — interval distribution ($n={n:,}$)"
    if title_suffix:
        title = f"{title} [{title_suffix}]"
    ax.set_title(title)
    ax.set_xlabel(_block_foreground_density_xlabel(gray_threshold=gray_threshold))

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


def export_block_foreground_density_figure(
    rows: list[BlockForegroundDensityRow],
    output_dir: Path,
    *,
    title_suffix: str = "",
    gray_threshold: float | None = None,
) -> Path:
    output_path = output_dir / "block_foreground_density_distribution.png"
    plot_block_foreground_density_distribution(
        rows,
        output_path,
        title_suffix=title_suffix,
        gray_threshold=gray_threshold,
    )
    return output_path
