"""Plotting for page-level image analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


FOREGROUND_DENSITY_COLOR_BLUE = "#2F6DB3"
FOREGROUND_DENSITY_COLOR_BLUE_LIGHT = "#B7D0EA"
FOREGROUND_DENSITY_COLOR_BLUE_EDGE = "#7FA8D1"


def _foreground_density_hist_bins(densities: np.ndarray, *, x_max: float) -> np.ndarray:
    """Histogram edges on the 0–x_max density scale (fraction, not percent)."""
    capped = densities[densities <= x_max + 1e-12]
    if capped.size == 0:
        return np.linspace(0.0, x_max, 2)
    span = max(float(x_max), 1e-6)
    bin_count = int(np.clip(np.ceil(np.sqrt(capped.size) * 2.5), 24, 48))
    return np.linspace(0.0, span, bin_count + 1)


def _foreground_density_display_max(densities: np.ndarray) -> float:
    """Full-range display: do not truncate the long tail."""
    if densities.size == 0:
        return 0.02
    return float(np.max(densities))


def _foreground_density_style_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.5)
    ax.grid(True, axis="y", color="#D0D0D0", linestyle="-", linewidth=0.55, alpha=0.32)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color("#444444")


def _foreground_density_kde(values: np.ndarray, x_fit: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or float(np.std(values)) <= 1e-12:
        return np.zeros_like(x_fit)
    from scipy.stats import gaussian_kde

    return np.asarray(gaussian_kde(values, bw_method="scott")(x_fit), dtype=np.float64)


def _foreground_density_ylim_top(y_top: float) -> float:
    if y_top <= 0:
        return 22.0
    return max(22.0, y_top * 1.08)


def _feature_join_key(row: ImageFeatureRow) -> str:
    path = Path(row.relative_path)
    return path.stem


def _load_post_selection_by_join_key(kept_samples_csv: Path) -> dict[str, str]:
    import pandas as pd

    frame = pd.read_csv(kept_samples_csv)
    if "basename" not in frame.columns or "post_selection_source" not in frame.columns:
        raise ValueError(f"{kept_samples_csv} must contain basename and post_selection_source")
    return {
        Path(str(basename)).stem: str(source).strip()
        for basename, source in zip(frame["basename"], frame["post_selection_source"], strict=True)
        if str(source).strip()
    }


def _split_densities_by_hard(
    features: list[ImageFeatureRow],
    *,
    kept_samples_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_by_key = _load_post_selection_by_join_key(kept_samples_csv)
    all_densities: list[float] = []
    hard_densities: list[float] = []
    non_hard_densities: list[float] = []
    missing = 0
    for row in features:
        key = _feature_join_key(row)
        source = source_by_key.get(key)
        if source is None:
            missing += 1
            continue
        density = float(row.foreground_density)
        all_densities.append(density)
        if source == "hard":
            hard_densities.append(density)
        else:
            non_hard_densities.append(density)
    if missing:
        raise ValueError(
            f"{missing} feature rows missing from {kept_samples_csv} (join on basename stem)"
        )
    return (
        np.asarray(all_densities, dtype=np.float64),
        np.asarray(hard_densities, dtype=np.float64),
        np.asarray(non_hard_densities, dtype=np.float64),
    )


FOREGROUND_DENSITY_KDE_ALL_COLOR = "#B7D0EA"
FOREGROUND_DENSITY_KDE_ALL_LINEWIDTH = 0.9
FOREGROUND_DENSITY_KDE_HARD_COLOR = "#E89898"
FOREGROUND_DENSITY_KDE_NON_HARD_COLOR = "#A8D4A0"
FOREGROUND_DENSITY_KDE_FIT_POINTS = 512
FOREGROUND_DENSITY_PEAK_LABEL_Y_OFFSET = 8


def _annotate_kde_peak(
    ax,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    color: str,
) -> None:
    if y_fit.size == 0 or float(y_fit.max()) <= 0:
        return
    peak_idx = int(np.argmax(y_fit))
    peak_x = float(x_fit[peak_idx])
    peak_y = float(y_fit[peak_idx])
    ax.plot(peak_x, peak_y, "o", color=color, markersize=4.5, zorder=7)
    ax.annotate(
        f"{peak_x * 100.0:.1f}%",
        xy=(peak_x, peak_y),
        xytext=(0, FOREGROUND_DENSITY_PEAK_LABEL_Y_OFFSET),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=color,
        clip_on=False,
        zorder=7,
    )


@with_locked_pyplot
def plot_foreground_density_distribution(
    features: list[ImageFeatureRow],
    output_path: Path,
    *,
    gray_threshold: float | None = None,
    kept_samples_csv: Path | None = None,
) -> None:
    """Histogram bars overlaid with KDE curves of page foreground density (full range)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    _configure_matplotlib_fonts(plt)
    densities = np.asarray([row.foreground_density for row in features], dtype=np.float64)
    n = int(densities.size)
    if n == 0:
        raise ValueError("No foreground density values to plot")

    if kept_samples_csv is not None and kept_samples_csv.is_file():
        densities, hard_densities, non_hard_densities = _split_densities_by_hard(
            features,
            kept_samples_csv=kept_samples_csv,
        )
        n = int(densities.size)
    else:
        hard_densities = non_hard_densities = np.asarray([], dtype=np.float64)

    x_max = _foreground_density_display_max(densities)  # full range, no truncation

    x_fit = np.linspace(0.0, x_max, FOREGROUND_DENSITY_KDE_FIT_POINTS)
    y_all = _foreground_density_kde(densities, x_fit)
    has_hard_split = kept_samples_csv is not None and kept_samples_csv.is_file()
    if has_hard_split:
        n_hard = int(hard_densities.size)
        n_non_hard = int(non_hard_densities.size)
        kde_series: list[tuple[str, np.ndarray, str, str, float, int]] = [
            (
                f"All (n={n:,})",
                y_all,
                FOREGROUND_DENSITY_KDE_ALL_COLOR,
                "--",
                FOREGROUND_DENSITY_KDE_ALL_LINEWIDTH,
                5,
            ),
            (
                f"Hard (n={n_hard:,})",
                _foreground_density_kde(hard_densities, x_fit),
                FOREGROUND_DENSITY_KDE_HARD_COLOR,
                "-",
                2.6,
                7,
            ),
            (
                f"Non-hard (n={n_non_hard:,})",
                _foreground_density_kde(non_hard_densities, x_fit),
                FOREGROUND_DENSITY_KDE_NON_HARD_COLOR,
                "-",
                2.6,
                6,
            ),
        ]
    else:
        kde_series = [
            ("All", y_all, FOREGROUND_DENSITY_KDE_ALL_COLOR, "-", 2.6, 5),
        ]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=FIGURE_DPI)
    fig.subplots_adjust(top=0.85, bottom=0.12)
    fig.suptitle(
        f"Foreground-density distributions by sampling group (n={n:,})",
        fontsize=13,
        y=0.98,
    )

    # Histogram bars (density-normalized so bars and KDE share the y axis).
    hist_bins = _foreground_density_hist_bins(densities, x_max=x_max)
    hist_counts, hist_edges = np.histogram(densities, bins=hist_bins, density=True)
    hist_peak = float(np.max(hist_counts)) if hist_counts.size else 0.0
    ax.bar(
        hist_edges[:-1],
        hist_counts,
        width=np.diff(hist_edges),
        align="edge",
        color=FOREGROUND_DENSITY_COLOR_BLUE_LIGHT,
        edgecolor=FOREGROUND_DENSITY_COLOR_BLUE_EDGE,
        linewidth=0.6,
        alpha=0.6,
        zorder=2,
        label=f"Histogram (n={n:,})",
    )

    legend_handles: list = [
        Patch(
            facecolor=FOREGROUND_DENSITY_COLOR_BLUE_LIGHT,
            edgecolor=FOREGROUND_DENSITY_COLOR_BLUE_EDGE,
            label=f"Histogram (n={n:,})",
        )
    ]
    legend_handles += [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
        for label, _, color, linestyle, linewidth, _ in kde_series
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.4,
    )

    for label, y_fit, color, linestyle, linewidth, zorder in kde_series:
        if y_fit.size and float(y_fit.max()) > 0:
            ax.plot(
                x_fit,
                y_fit,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
            )

    y_top = hist_peak
    for _, y_fit, _, _, _, _ in kde_series:
        if y_fit.size:
            y_top = max(y_top, float(y_fit.max()))

    for _, y_fit, color, _, _, _ in kde_series:
        _annotate_kde_peak(ax, x_fit, y_fit, color)

    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, _foreground_density_ylim_top(y_top))
    ax.set_ylabel("Probability density", fontsize=10)
    ax.set_xlabel("Foreground density (%)", fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100.0:.0f}"))
    _foreground_density_style_axes(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_paper_figures(
    features: list[ImageFeatureRow],
    paper_dir: Path,
    *,
    gray_threshold: float | None = None,
    kept_samples_csv: Path | None = None,
) -> dict[str, Path]:
    paper_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if kept_samples_csv is None:
        candidate = paper_dir.resolve().parents[3] / "manifests" / "kept_samples.csv"
        if candidate.is_file():
            kept_samples_csv = candidate

    path = paper_dir / "foreground_density_distribution.png"
    plot_foreground_density_distribution(
        features,
        path,
        gray_threshold=gray_threshold,
        kept_samples_csv=kept_samples_csv,
    )
    outputs["foreground_density_distribution"] = path

    return outputs
