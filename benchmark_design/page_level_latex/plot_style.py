"""Shared style helpers for Chapter-6 page-level LaTeX figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot

FIGURE_DPI = 300

# Unified palette (same statistic -> same color across figures).
COLOR_PAGE_COVERAGE = "#4472C4"  # blue
COLOR_PAGE_MAX = "#ED7D31"  # orange
COLOR_PAGE_COUNT = "#70AD47"  # green
COLOR_RARE10 = "#4472C4"  # blue
COLOR_SIMILAR = "#4472C4"  # blue
HEATMAP_CMAP = "Blues"

FONT_TITLE = 12
FONT_LABEL = 10
FONT_TICK = 9
FONT_ANNOT = 8
FONT_BOX = 9


def format_count(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def format_ratio_pct(ratio: float) -> str:
    return f"{float(ratio) * 100:.2f}%"


def page_ratio(count: int, total_pages: int) -> float:
    return float(count) / float(total_pages) if total_pages else 0.0


def dual_label(count: int, ratio: float) -> str:
    return f"{format_count(count)}\n{format_ratio_pct(ratio)}"


def auto_bin_edges(values: np.ndarray, *, max_bins: int = 10) -> np.ndarray:
    """Build at most ``max_bins`` contiguous edges covering ``values``."""
    if values.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        return np.asarray([vmin - 0.5, vmax + 0.5], dtype=np.float64)

    # Prefer integer-aligned edges when values are near-integer counts.
    int_like = np.allclose(values, np.round(values))
    if int_like:
        lo = int(np.floor(vmin))
        hi = int(np.ceil(vmax))
        span = hi - lo + 1
        if span <= max_bins:
            return np.arange(lo, hi + 2, dtype=np.float64) - 0.5
        step = int(np.ceil(span / max_bins))
        edges = np.arange(lo, hi + step + 1, step, dtype=np.float64) - 0.5
        if edges[-1] < hi + 0.5:
            edges = np.append(edges, hi + 0.5)
        return edges

    edges = np.linspace(vmin, vmax, max_bins + 1, dtype=np.float64)
    edges[-1] = vmax + 1e-9
    return edges


def histogram_frame(
    values: np.ndarray,
    *,
    metric: str,
    total_pages: int,
    max_bins: int = 10,
) -> pd.DataFrame:
    edges = auto_bin_edges(values, max_bins=max_bins)
    counts, edges = np.histogram(values, bins=edges)
    rows = []
    for idx, count in enumerate(counts):
        start = float(edges[idx])
        end = float(edges[idx + 1])
        # Present integer-friendly closed intervals for near-integer data.
        if np.allclose(values, np.round(values)):
            bin_start = int(np.ceil(start + 1e-9))
            bin_end = int(np.floor(end - 1e-9))
            if bin_end < bin_start:
                bin_end = bin_start
        else:
            bin_start = start
            bin_end = end
        rows.append(
            {
                "metric": metric,
                "bin_start": bin_start,
                "bin_end": bin_end,
                "page_count": int(count),
                "page_ratio": page_ratio(int(count), total_pages),
            }
        )
    return pd.DataFrame(rows)


def apply_chapter6_style(plt) -> None:
    _configure_matplotlib_fonts(plt)
    plt.rcParams.update(
        {
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_TICK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def enable_horizontal_grid_only(ax) -> None:
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def annotate_bars_dual(
    ax,
    bars,
    counts: list[int] | np.ndarray,
    ratios: list[float] | np.ndarray,
    *,
    fontsize: int = FONT_ANNOT,
    horizontal: bool = False,
) -> None:
    """Annotate each bar with page_count and page_ratio (two lines)."""
    for bar, count, ratio in zip(bars, counts, ratios, strict=True):
        if int(count) <= 0 and float(ratio) <= 0:
            continue
        label = dual_label(int(count), float(ratio))
        if horizontal:
            width = bar.get_width()
            ax.annotate(
                label,
                xy=(width, bar.get_y() + bar.get_height() / 2.0),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=fontsize,
                clip_on=False,
            )
        else:
            height = bar.get_height()
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                clip_on=False,
            )
    if horizontal:
        xmax = ax.get_xlim()[1]
        ax.set_xlim(0, xmax * 1.28 if xmax > 0 else 1.0)
    else:
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1.0)


def annotate_hist_from_frame(ax, patches, frame: pd.DataFrame, *, fontsize: int = FONT_ANNOT) -> None:
    counts = frame["page_count"].tolist()
    ratios = frame["page_ratio"].tolist()
    n = len(counts)
    rotate = 90 if n > 12 else 0
    for patch, count, ratio in zip(patches, counts, ratios, strict=True):
        if int(count) <= 0:
            continue
        label = format_count(int(count)) if n > 12 else dual_label(int(count), float(ratio))
        height = patch.get_height()
        ax.annotate(
            label,
            xy=(patch.get_x() + patch.get_width() / 2.0, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=max(6, fontsize - (1 if n > 12 else 0)),
            rotation=rotate,
            clip_on=False,
        )
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * (1.28 if n > 12 else 1.18) if ymax > 0 else 1.0)


def stats_box_text(*, vmin: float, vmean: float, vmax: float) -> str:
    if float(vmin).is_integer() and float(vmax).is_integer():
        return f"min={int(vmin)}\nmean={vmean:.2f}\nmax={int(vmax)}"
    return f"min={vmin:.2f}\nmean={vmean:.2f}\nmax={vmax:.2f}"


def add_stats_box(ax, *, vmin: float, vmean: float, vmax: float) -> None:
    ax.text(
        0.98,
        0.98,
        stats_box_text(vmin=vmin, vmean=vmean, vmax=vmax),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_BOX,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#B0B0B0", "boxstyle": "round,pad=0.3"},
    )


def add_mean_vline(ax, mean_value: float, *, color: str = "#333333") -> None:
    ax.axvline(mean_value, color=color, linestyle="--", linewidth=1.2, alpha=0.85)


def save_figure_outputs(fig, output_stem: Path) -> dict[str, Path]:
    """Save PNG next to ``output_stem`` (without suffix)."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    return {"png": png_path}


def write_plot_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def assert_page_partition(counts: list[int] | np.ndarray, total_pages: int, *, label: str) -> None:
    total = int(np.sum(counts))
    if total != int(total_pages):
        raise ValueError(f"{label} page counts sum to {total}, expected {total_pages}")
