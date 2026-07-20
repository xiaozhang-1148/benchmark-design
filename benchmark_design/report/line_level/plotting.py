"""Plotting for line-level geometry analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_design.line_level.models import LineMetricsRow, PageMetricsRow
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot

FIGURE_DPI = 300

PLOT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("lines_per_image", "line_count", "Lines per image", "line_count"),
    ("line_width_px", "bbox_width_px", "Line width (horizontal, px)", "bbox_width_px"),
    ("aspect_ratio", "aspect_ratio", "Line aspect ratio (width/height)", "aspect_ratio"),
    ("orientation_angle", "orientation_deg", "Signed orientation α (deg)", "orientation_deg"),
    (
        "bbox_outside_ink_ratio",
        "bbox_outside_ink_ratio",
        "Ink ratio outside mask inside axis-aligned bbox",
        "bbox_outside_ink_ratio",
    ),
)


def _valid_values(rows: list[LineMetricsRow], metric: str) -> np.ndarray:
    values = []
    for row in rows:
        if not row.is_valid:
            continue
        value = getattr(row, metric, None)
        if value is None:
            continue
        values.append(float(value))
    return np.array(values, dtype=np.float64)


@with_locked_pyplot
def _save_hist(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    log_scale: bool = False,
    quantile_lines: tuple[float, float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    caption: str = "",
) -> None:
    import matplotlib.pyplot as plt

    if values.size == 0:
        return
    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=FIGURE_DPI)
    plot_values = values
    if xlim is not None:
        plot_values = values[(values >= xlim[0]) & (values <= xlim[1])]
    bins = 40 if plot_values.size > 40 else max(10, plot_values.size)
    if plot_values.size:
        ax.hist(plot_values, bins=bins, color="#4472C4", alpha=0.85, edgecolor="white", range=xlim)
    if quantile_lines is not None:
        p05, median, p95 = quantile_lines
        for value, label, color in (
            (p05, "P5", "#ED7D31"),
            (median, "median", "#00B050"),
            (p95, "P95", "#C00000"),
        ):
            if xlim is None or (xlim[0] <= value <= xlim[1]):
                ax.axvline(value, color=color, linestyle="--", linewidth=1.2, label=f"{label}={value:.1f}")
        ax.legend(fontsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if caption:
        fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


@with_locked_pyplot
def _save_line_height_px_plots(values: np.ndarray, plots_dir: Path, appendix_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    if values.size == 0:
        return outputs

    p05 = float(np.quantile(values, 0.05))
    median = float(np.median(values))
    p95 = float(np.quantile(values, 0.95))
    p99 = float(np.quantile(values, 0.99))
    above_p99 = int(np.sum(values > p99))
    quantile_lines = (p05, median, p95)

    main_path = plots_dir / "bbox_height_px_distribution.png"
    caption = f"{above_p99:,} lines above P99 ({p99:.1f} px); full range in appendix"
    _save_hist(
        values,
        title="Line height distribution (0–P99)",
        xlabel="Vertical height (pixels)",
        output_path=main_path,
        quantile_lines=quantile_lines,
        xlim=(0.0, p99),
        caption=caption,
    )
    outputs["bbox_height_px_distribution"] = main_path

    appendix_path = appendix_dir / "bbox_height_px_full_range.png"
    _save_hist(
        values,
        title="Line height distribution (full range)",
        xlabel="Vertical height (pixels)",
        output_path=appendix_path,
        quantile_lines=quantile_lines,
    )
    outputs["bbox_height_px_full_range"] = appendix_path
    return outputs


@with_locked_pyplot
def _save_image_resolution_hexbin(page_widths: np.ndarray, page_heights: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIGURE_DPI)
    hb = ax.hexbin(page_widths, page_heights, gridsize=35, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Pages")
    ax.set_title("Image resolution")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_line_level_plots(
    line_rows: list[LineMetricsRow],
    page_rows: list[PageMetricsRow],
    plots_dir: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    appendix_dir = plots_dir / "appendix"

    line_heights = _valid_values(line_rows, "bbox_height_px")
    outputs.update(_save_line_height_px_plots(line_heights, plots_dir, appendix_dir))

    for filename, metric, title, xlabel in PLOT_SPECS:
        if metric == "line_count":
            values = np.array([row.line_count for row in page_rows], dtype=np.float64)
        else:
            values = _valid_values(line_rows, metric)
        path = plots_dir / f"{filename}.png"
        log_scale = metric in {"aspect_ratio", "bbox_width_px"}
        _save_hist(values, title=title, xlabel=xlabel, output_path=path, log_scale=log_scale)
        outputs[filename] = path

    page_widths = np.array([row.width for row in page_rows], dtype=np.float64)
    page_heights = np.array([row.height for row in page_rows], dtype=np.float64)
    if page_widths.size and page_heights.size:
        resolution_path = plots_dir / "image_resolution.png"
        _save_image_resolution_hexbin(page_widths, page_heights, resolution_path)
        outputs["image_resolution"] = resolution_path

    return outputs


DATASET_COLORS: tuple[str, ...] = (
    "#4472C4",
    "#ED7D31",
    "#70AD47",
    "#C00000",
    "#7030A0",
    "#00B0F0",
    "#FFC000",
    "#00B050",
)

MATHWRITING_ALIASES = {"MathWritting", "MathWriting"}


def _dataset_plot_order(names: list[str]) -> list[str]:
    """Put ours first, then remaining datasets alphabetically."""
    others = sorted(name for name in names if name != "ours")
    if "ours" in names:
        return ["ours", *others]
    return others


def _is_mathwriting(name: str) -> bool:
    return name in MATHWRITING_ALIASES


def _shared_edges(values_list: list[np.ndarray], *, bins: int) -> np.ndarray:
    joined = np.concatenate([v for v in values_list if v.size]) if values_list else np.array([0.0, 1.0])
    lo = float(np.min(joined))
    hi = float(np.max(joined))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1)


def _shared_log_edges(values_list: list[np.ndarray], *, bins: int) -> np.ndarray:
    joined = np.concatenate([v[v > 0] for v in values_list if v.size]) if values_list else np.array([0.1, 10.0])
    lo = float(np.min(joined))
    hi = float(np.max(joined))
    if lo <= 0:
        lo = float(np.min(joined[joined > 0])) if np.any(joined > 0) else 1e-3
    if hi <= lo:
        hi = lo * 10.0
    return np.geomspace(lo, hi, bins + 1)


def _smooth_bin_share_curve(
    values: np.ndarray,
    bin_edges: np.ndarray,
    *,
    xscale: str,
    smooth_sigma: float = 2.0,
    oversample: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared-edge bin share (%) → smooth curve via Gaussian filter + cubic interp."""
    from scipy.interpolate import make_interp_spline
    from scipy.ndimage import gaussian_filter1d

    if values.size == 0:
        return np.array([]), np.array([])
    weights = np.full(values.shape, 100.0 / values.size, dtype=np.float64)
    counts, _ = np.histogram(values, bins=bin_edges, weights=weights)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if counts.size < 3:
        return centers, counts.astype(np.float64)

    smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=smooth_sigma, mode="nearest")
    smoothed = np.clip(smoothed, 0.0, None)

    if xscale == "log":
        valid = (centers > 0) & np.isfinite(centers)
        x_src = np.log(centers[valid])
        y_src = smoothed[valid]
        if x_src.size < 4:
            return centers[valid], y_src
        x_dense = np.linspace(float(x_src.min()), float(x_src.max()), max(oversample * len(x_src), 160))
        spline = make_interp_spline(x_src, y_src, k=min(3, x_src.size - 1))
        y_dense = np.clip(spline(x_dense), 0.0, None)
        return np.exp(x_dense), y_dense

    x_dense = np.linspace(float(centers.min()), float(centers.max()), max(oversample * len(centers), 160))
    spline = make_interp_spline(centers, smoothed, k=min(3, centers.size - 1))
    y_dense = np.clip(spline(x_dense), 0.0, None)
    return x_dense, y_dense


@with_locked_pyplot
def _save_dataset_bin_share_overlay(
    series_by_dataset: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    bin_edges: np.ndarray,
    xscale: str = "linear",
    main_xlim: tuple[float, float] | None = None,
    special_markers: dict[str, float] | None = None,
    skip_curve_datasets: set[str] | None = None,
) -> None:
    """Smooth within-dataset bin-share curves (%); shared edges; no fill, no inset."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not series_by_dataset:
        return
    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=FIGURE_DPI)
    skip = skip_curve_datasets or set()
    special_markers = special_markers or {}
    order = _dataset_plot_order(list(series_by_dataset.keys()))
    legend_handles: list = []
    legend_labels: list[str] = []

    for index, dataset in enumerate(order):
        values = series_by_dataset[dataset]
        if values.size == 0:
            continue
        color = DATASET_COLORS[index % len(DATASET_COLORS)]
        linewidth = 2.8 if dataset == "ours" else 1.6
        if dataset in skip:
            fixed = special_markers.get(dataset)
            fixed_note = f", fixed {fixed:g}" if fixed is not None else ""
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle="--", linewidth=linewidth)
            )
            legend_labels.append(f"{dataset} (n={values.size:,}{fixed_note})")
            continue
        x_curve, y_curve = _smooth_bin_share_curve(values, bin_edges, xscale=xscale)
        if x_curve.size == 0:
            continue
        (line,) = ax.plot(
            x_curve,
            y_curve,
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        legend_handles.append(line)
        legend_labels.append(f"{dataset} (n={values.size:,})")

    if main_xlim is not None:
        ax.set_xlim(*main_xlim)
    if xscale == "log":
        ax.set_xscale("log")
        ax.axvline(1.0, color="#888888", linestyle=":", linewidth=1.0, zorder=0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Samples per bin (%)")
    ax.set_ylim(bottom=0.0)
    ax.legend(legend_handles, legend_labels, fontsize=8, loc="best", frameon=True, fancybox=False, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_external_dataset_aspect_plots(
    rows: list,
    plots_dir: Path,
) -> dict[str, Path]:
    """Overlay AABB width / height / aspect-ratio shares across datasets."""
    from benchmark_design.line_level.dataset_aspect import DatasetLineGeometryRow

    typed_rows: list[DatasetLineGeometryRow] = list(rows)
    outputs: dict[str, Path] = {}
    if not typed_rows:
        return outputs

    by_dataset: dict[str, list[DatasetLineGeometryRow]] = {}
    for row in typed_rows:
        by_dataset.setdefault(row.dataset, []).append(row)

    height_series = {
        name: np.array([row.height_px for row in items], dtype=np.float64) for name, items in by_dataset.items()
    }
    width_series = {
        name: np.array([row.width_px for row in items], dtype=np.float64) for name, items in by_dataset.items()
    }
    aspect_series = {
        name: np.array([row.aspect_ratio for row in items], dtype=np.float64) for name, items in by_dataset.items()
    }

    # Height: MathWriting often fixed at 256 — legend-only, no spike / no annotation text.
    mathwriting_fixed: dict[str, float] = {}
    skip_height: set[str] = set()
    for name, values in height_series.items():
        if _is_mathwriting(name) and values.size and float(np.unique(values).size) == 1:
            mathwriting_fixed[name] = float(values[0])
            skip_height.add(name)

    non_mw_heights = [v for n, v in height_series.items() if n not in skip_height and v.size]
    if non_mw_heights:
        joined = np.concatenate(non_mw_heights)
        height_p99 = float(np.quantile(joined, 0.99))
        height_main_max = max(height_p99 * 1.02, 80.0)
        height_edges = _shared_edges(non_mw_heights, bins=50)
    else:
        height_main_max = 300.0
        height_edges = _shared_edges(list(height_series.values()), bins=50)

    height_path = plots_dir / "external_dataset_height_distribution.png"
    _save_dataset_bin_share_overlay(
        height_series,
        title="Line height by dataset",
        xlabel="Vertical height (px)",
        output_path=height_path,
        bin_edges=height_edges,
        main_xlim=(0.0, height_main_max),
        special_markers=mathwriting_fixed,
        skip_curve_datasets=skip_height,
    )
    outputs["external_dataset_height_distribution"] = height_path

    width_p99 = float(np.quantile(np.concatenate(list(width_series.values())), 0.99))
    width_main_max = max(width_p99 * 1.02, 100.0)
    width_edges = _shared_edges(list(width_series.values()), bins=50)
    width_path = plots_dir / "external_dataset_width_distribution.png"
    _save_dataset_bin_share_overlay(
        width_series,
        title="Line width by dataset",
        xlabel="Horizontal width (px)",
        output_path=width_path,
        bin_edges=width_edges,
        main_xlim=(0.0, width_main_max),
    )
    outputs["external_dataset_width_distribution"] = width_path

    aspect_edges = _shared_log_edges(list(aspect_series.values()), bins=50)
    aspect_path = plots_dir / "external_dataset_aspect_ratio_distribution.png"
    _save_dataset_bin_share_overlay(
        aspect_series,
        title="Aspect ratio width/height by dataset",
        xlabel="Aspect ratio (width / height)",
        output_path=aspect_path,
        bin_edges=aspect_edges,
        xscale="log",
        main_xlim=None,
    )
    outputs["external_dataset_aspect_ratio_distribution"] = aspect_path
    return outputs
