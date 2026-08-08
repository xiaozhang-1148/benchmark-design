"""Chapter-6 page-level LaTeX figures (English labels, PNG + plot-data CSVs)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.page_latex_metrics import PageLatexMetricsRow
from benchmark_design.page_level_latex.plot_data import (
    build_fig6_1_plot_data,
    build_fig6_4_plot_data,
    build_fig6_raw_count_curve_data,
)
from benchmark_design.page_level_latex.plot_style import (
    FONT_ANNOT,
    FONT_BOX,
    FONT_LABEL,
    FONT_TICK,
    apply_chapter6_style,
    format_count,
    format_ratio_pct,
    save_figure_outputs,
    write_plot_csv,
)
from benchmark_design.report.pyplot_lock import with_locked_pyplot

# Unified blue palette for Chapter-6 distribution panels.
COLOR_AST_BLUE = "#2F6DB3"  # KDE / discrete bars (high saturation)
COLOR_AST_BLUE_LIGHT = "#B7D0EA"  # empirical histogram fill
COLOR_AST_BLUE_EDGE = "#7FA8D1"
COLOR_AST_BAR = "#2F6DB3"
COLOR_MEAN_LINE = "#4A4A4A"
COLOR_KDE_ALL = "#B7D0EA"
COLOR_KDE_ALL_LINEWIDTH = 0.9
COLOR_KDE_HARD = "#E89898"
COLOR_KDE_NON_HARD = "#A8D4A0"
KDE_HARD_LINEWIDTH = 2.1
KDE_NON_HARD_LINEWIDTH = 2.1
KDE_LEGEND_LINEWIDTH = 2.0
FIG6_FIGSIZE = (14.4, 9.6)
FIG6_SUBPLOTS_ADJUST = dict(left=0.06, right=0.99, top=0.88, bottom=0.10, wspace=0.20, hspace=0.38)
KDE_FIT_POINTS = 512
KDE_PEAK_LABEL_Y_OFFSET = 8
SHARE_BAR_WIDTH = 0.22
SHARE_BAR_OFFSETS = (-0.26, 0.0, 0.26)
SHARE_WIDE_BAR_OFFSETS = (-0.34, 0.0, 0.34)
SHARE_WIDE_CATEGORY_GAP = 0.45
SHARE_CATEGORY_EDGE_PAD = 0.55
SHARE_LABEL_FONTSIZE = 5.5
SHARE_NON_HARD_LABEL_OFFSET = (-4, 3)
SHARE_HARD_LABEL_OFFSET = (0, 4)
SHARE_ALL_LABEL_OFFSET = (4, 3)
SHARE_LABEL_MIN_Y_FRAC = 0.015

SHARE_YLIM_PAD = 1.12

FONT_PANEL_SUPTITLE = 13
FONT_PANEL_SUBNOTE = 9
FONT_PANEL_AX_TITLE = 11


def _finalize_figure(fig, *, figure_stem: Path, csv_path: Path, plot_data) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    write_plot_csv(plot_data, csv_path)
    paths = save_figure_outputs(fig, figure_stem)
    plt.close(fig)
    paths["csv"] = csv_path
    return paths


def _hist_edges_for_display(values: np.ndarray, *, x_max: float, target_bins: int) -> np.ndarray:
    """Integer-friendly histogram edges on [0, x_max] with a controlled bin count."""
    x_max = float(max(x_max, 1.0))
    data = values[(values >= 0) & (values <= x_max + 1e-9)]
    if data.size == 0:
        return np.linspace(0.0, x_max, max(target_bins, 2) + 1)

    # Prefer automatic edges, then cap bin count for readability.
    edges = np.histogram_bin_edges(data, bins="auto", range=(0.0, x_max))
    n_bins = int(edges.size - 1)
    if n_bins > target_bins:
        # Keep roughly equal-width bins; snap to integers when span is integer-like.
        if np.allclose(data, np.round(data)):
            step = max(1, int(np.ceil(x_max / target_bins)))
            edges = np.arange(0.0, x_max + step + 1e-9, step, dtype=np.float64)
            if edges[-1] < x_max:
                edges = np.append(edges, x_max)
            else:
                edges[-1] = x_max
        else:
            edges = np.linspace(0.0, x_max, target_bins + 1)
    if edges[-1] < x_max:
        edges = np.append(edges[:-1], x_max)
    return edges.astype(np.float64)


def _style_distribution_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=0.8, length=3.5)
    ax.grid(True, axis="y", color="#D0D0D0", linestyle="--", linewidth=0.55, alpha=0.32)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color("#444444")


def _format_minmax_text(values: np.ndarray, *, multiline: bool = False) -> str:
    vmin = float(values.min())
    vmax = float(values.max())
    if float(vmin).is_integer() and float(vmax).is_integer():
        left, right = f"min={int(vmin)}", f"max={int(vmax)}"
    else:
        left, right = f"min={vmin:.2f}", f"max={vmax:.2f}"
    return f"{left}\n{right}" if multiline else f"{left}   {right}"


def _draw_minmax_box(ax, values: np.ndarray, *, multiline: bool = True) -> None:
    ax.text(
        0.98,
        0.98,
        _format_minmax_text(values, multiline=multiline),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_BOX,
        linespacing=1.25,
        zorder=10,
        clip_on=False,
        bbox={
            "facecolor": "white",
            "alpha": 0.95,
            "edgecolor": "#B0B0B0",
            "boxstyle": "round,pad=0.35",
            "linewidth": 0.8,
        },
    )


def _format_full_range_text(values: np.ndarray) -> str:
    vmin = float(values.min())
    vmax = float(values.max())
    if float(vmin).is_integer() and float(vmax).is_integer():
        return f"Full range: {int(vmin)}–{int(vmax)}"
    return f"Full range: {vmin:.2f}–{vmax:.2f}"


def _draw_plain_note(
    ax, text: str, *, x: float = 0.98, y: float = 0.98, ha: str = "right", fontsize: int = FONT_BOX
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=fontsize,
        color="#333333",
        zorder=10,
        clip_on=False,
    )


def _gaussian_kde_curve(values: np.ndarray, x_fit: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or float(np.std(values)) <= 1e-12:
        return np.zeros_like(x_fit)
    from scipy.stats import gaussian_kde

    return np.asarray(gaussian_kde(values, bw_method="scott")(x_fit), dtype=np.float64)


def _format_kde_peak_label(peak_x: float) -> str:
    rounded = round(peak_x)
    if abs(peak_x - rounded) < 0.05:
        return str(int(rounded))
    return f"{peak_x:.1f}"


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
        _format_kde_peak_label(peak_x),
        xy=(peak_x, peak_y),
        xytext=(0, KDE_PEAK_LABEL_Y_OFFSET),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=color,
        clip_on=False,
        zorder=7,
    )


def _share_label_text(
    share_pct: float,
    category: str,
    *,
    label_focus_range: tuple[int, int] | None,
) -> str | None:
    if share_pct <= 0.0:
        return None
    if share_pct >= 1.0:
        return f"{share_pct:.1f}%"
    if label_focus_range is None:
        return None
    category_value = int(category)
    if label_focus_range[0] <= category_value <= label_focus_range[1]:
        return f"{share_pct:.1f}%"
    return None


def _build_share_category_xs(
    categories: Sequence[str],
    *,
    wide_spacing_categories: set[str] | None = None,
) -> np.ndarray:
    wide_spacing_categories = wide_spacing_categories or set()
    xs: list[float] = []
    pos = 0.0
    for index, category in enumerate(categories):
        if index > 0 and str(category) in wide_spacing_categories:
            pos += SHARE_WIDE_CATEGORY_GAP
        xs.append(pos)
        pos += 1.0
    return np.asarray(xs, dtype=np.float64)


def _share_bar_offsets_for_category(
    category: str,
    *,
    wide_spacing_categories: set[str] | None,
) -> tuple[float, float, float]:
    if wide_spacing_categories and str(category) in wide_spacing_categories:
        return SHARE_WIDE_BAR_OFFSETS
    return SHARE_BAR_OFFSETS


def _annotate_share_label(ax, x: float, y: float, text: str, *, xytext: tuple[int, int]) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=SHARE_LABEL_FONTSIZE,
        color="#333333",
        zorder=6,
    )


def _draw_sampling_share_bars(
    ax,
    *,
    categories: list[str],
    hard_counts: np.ndarray,
    non_hard_counts: np.ndarray,
    n_all: int,
    n_hard: int,
    n_non_hard: int,
    xlabel: str,
    title: str = "",
    title_below: bool = False,
    label_focus_range: tuple[int, int] | None = None,
    wide_spacing_categories: set[str] | None = None,
) -> None:
    """Grouped non-hard / hard / all share bars."""
    categories = list(categories)
    hard_counts = np.asarray(hard_counts, dtype=np.float64)
    non_hard_counts = np.asarray(non_hard_counts, dtype=np.float64)
    all_counts = hard_counts + non_hard_counts
    xs = _build_share_category_xs(categories, wide_spacing_categories=wide_spacing_categories)

    hard_pct = np.where(n_hard > 0, hard_counts / float(n_hard) * 100.0, 0.0)
    non_hard_pct = np.where(n_non_hard > 0, non_hard_counts / float(n_non_hard) * 100.0, 0.0)
    all_pct = np.where(n_all > 0, all_counts / float(n_all) * 100.0, 0.0)

    ymax = float(
        max(
            float(np.max(hard_pct)) if hard_pct.size else 0.0,
            float(np.max(non_hard_pct)) if non_hard_pct.size else 0.0,
            float(np.max(all_pct)) if all_pct.size else 0.0,
        )
    )
    label_y_floor = ymax * SHARE_LABEL_MIN_Y_FRAC if ymax > 0 else 0.15

    for category_index, category in enumerate(categories):
        x_center = float(xs[category_index])
        offsets = _share_bar_offsets_for_category(
            str(category),
            wide_spacing_categories=wide_spacing_categories,
        )
        series = [
            (offsets[0], float(non_hard_pct[category_index]), COLOR_KDE_NON_HARD, SHARE_NON_HARD_LABEL_OFFSET),
            (offsets[1], float(hard_pct[category_index]), COLOR_KDE_HARD, SHARE_HARD_LABEL_OFFSET),
            (offsets[2], float(all_pct[category_index]), COLOR_KDE_ALL, SHARE_ALL_LABEL_OFFSET),
        ]
        for offset, pct, color, label_offset in series:
            if pct <= 0:
                continue
            bar_x = x_center + offset
            bar = ax.bar(
                bar_x,
                pct,
                width=SHARE_BAR_WIDTH,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            height = float(bar[0].get_height())
            label = _share_label_text(pct, str(category), label_focus_range=label_focus_range)
            if label is None:
                continue
            center = float(bar[0].get_x() + bar[0].get_width() / 2.0)
            label_y = max(height, label_y_floor)
            _annotate_share_label(ax, center, label_y, label, xytext=label_offset)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in categories])
    ax.set_xlim(float(xs[0]) - SHARE_CATEGORY_EDGE_PAD, float(xs[-1]) + SHARE_CATEGORY_EDGE_PAD)
    ax.set_ylim(0.0, ymax * SHARE_YLIM_PAD if ymax > 0 else 1.0)
    _apply_panel_caption(ax, title, below=title_below)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel("Percentage within group (%)", fontsize=FONT_LABEL)
    _style_distribution_axes(ax)


def _bar_label_text_color(hex_color: str) -> str:
    color = hex_color.lstrip("#")
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 160 else "#222222"


def _density_legend_handles(
    *,
    academic: bool = False,
    kde_split: bool = False,
    n_all: int = 0,
    n_hard: int = 0,
    n_non_hard: int = 0,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=COLOR_AST_BLUE_LIGHT,
            edgecolor=COLOR_AST_BLUE_EDGE,
            label="Histogram" if academic else "Empirical",
        ),
    ]
    if kde_split:
        return [
            Line2D(
                [0],
                [0],
                color=COLOR_KDE_ALL,
                linewidth=COLOR_KDE_ALL_LINEWIDTH,
                linestyle="--",
                label=f"All (n={n_all:,})",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_KDE_NON_HARD,
                linewidth=KDE_LEGEND_LINEWIDTH,
                label=f"Non-hard (n={n_non_hard:,})",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_KDE_HARD,
                linewidth=KDE_LEGEND_LINEWIDTH,
                label=f"Hard (n={n_hard:,})",
            ),
        ]
    if academic:
        handles.extend(
            [
                Line2D([0], [0], color=COLOR_AST_BLUE, linewidth=2.4, label="KDE"),
                Line2D([0], [0], color=COLOR_MEAN_LINE, linestyle="--", linewidth=1.25, label="Mean"),
            ]
        )
        return handles
    handles.extend(
        [
            Line2D([0], [0], color=COLOR_AST_BLUE, linewidth=2.4, label="KDE fit"),
            Line2D([0], [0], color=COLOR_MEAN_LINE, linestyle="--", linewidth=1.25, label="Mean"),
        ]
    )
    return handles


def _apply_panel_caption(ax, title: str, *, below: bool = False) -> None:
    if not title:
        return
    if below:
        ax.set_title("")
        ax.text(
            0.5,
            -0.34,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_PANEL_AX_TITLE,
            color="#222222",
            clip_on=False,
        )
    else:
        ax.set_title(title, fontsize=FONT_PANEL_AX_TITLE, pad=8)


def _draw_density_hist_kde(
    ax,
    *,
    values: np.ndarray,
    x_max: float,
    title: str,
    xlabel: str,
    truncation_note: str,
    target_bins: int = 28,
    show_legend: bool = True,
    show_minmax_box: bool | None = None,
    show_mean_label: bool = True,
    ylabel: str | None = "Density",
    mean_label_mode: str = "default",
    show_kde_peak: bool = True,
    kde_peak_round: bool = False,
    full_range_plain: bool = False,
    title_below: bool = False,
    academic_annotation_layout: bool = False,
    hard_values: np.ndarray | None = None,
    non_hard_values: np.ndarray | None = None,
) -> dict[str, float]:
    """Empirical density histogram + KDE on a shared Density axis (academic standard)."""
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        ax.set_title(title, fontsize=FONT_PANEL_AX_TITLE)
        return {"mean": 0.0, "hidden_ratio": 0.0}

    kde_split = (
        hard_values is not None
        and non_hard_values is not None
        and int(hard_values.size) + int(non_hard_values.size) > 0
    )

    display = values[values <= x_max + 1e-9]
    hidden_ratio = float((n - int(display.size)) / n) if n else 0.0
    edges = _hist_edges_for_display(values, x_max=x_max, target_bins=target_bins)
    widths = np.diff(edges)
    counts, _ = np.histogram(display, bins=edges)
    # Density = count / (N_full * bin_width); visible mass < 1 when truncated.
    dens = counts.astype(np.float64) / (float(n) * np.maximum(widths, 1e-12))

    if not kde_split:
        ax.bar(
            edges[:-1],
            dens,
            width=widths,
            align="edge",
            color=COLOR_AST_BLUE_LIGHT,
            edgecolor=COLOR_AST_BLUE_EDGE,
            linewidth=0.5,
            alpha=0.95,
            zorder=2,
            label="Empirical",
        )

    x_fit = np.linspace(0.0, float(x_max), KDE_FIT_POINTS if kde_split else 400)
    if kde_split:
        kde_series = [
            (COLOR_KDE_ALL, _gaussian_kde_curve(values, x_fit), "--", COLOR_KDE_ALL_LINEWIDTH, 3),
            (COLOR_KDE_HARD, _gaussian_kde_curve(hard_values, x_fit), "-", KDE_HARD_LINEWIDTH, 5),
            (COLOR_KDE_NON_HARD, _gaussian_kde_curve(non_hard_values, x_fit), "-", KDE_NON_HARD_LINEWIDTH, 4),
        ]
        y_fit = np.zeros_like(x_fit)
        for color, curve, linestyle, linewidth, zorder in kde_series:
            if curve.size and float(curve.max()) > 0:
                ax.plot(x_fit, curve, color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder)
            y_fit = np.maximum(y_fit, curve)
            _annotate_kde_peak(ax, x_fit, curve, color)
    else:
        y_fit = _gaussian_kde_curve(values, x_fit)
        ax.plot(x_fit, y_fit, color=COLOR_AST_BLUE, linewidth=2.6, zorder=4, label="KDE fit")

    # Unify y-limit from histogram + KDE before placing annotations.
    if kde_split:
        y_top = float(y_fit.max()) if y_fit.size else 0.0
        y_scale = 1.08
    else:
        y_top = float(max(float(dens.max()) if dens.size else 0.0, float(y_fit.max()) if y_fit.size else 0.0))
        y_scale = 1.28
    ax.set_ylim(0.0, y_top * y_scale if y_top > 0 else 1.0)
    y_lim_top = ax.get_ylim()[1]

    mean_val = float(values.mean())
    mean_visible = 0.0 <= mean_val <= x_max
    if mean_visible and not kde_split:
        ax.axvline(
            mean_val,
            color=COLOR_MEAN_LINE,
            linestyle="--",
            linewidth=1.25,
            alpha=0.95,
            zorder=5,
            label="Overall mean" if mean_label_mode == "overall" else "Mean",
        )
        if show_mean_label:
            if mean_label_mode == "overall":
                mean_text = f"Overall mean = {mean_val:.2f}"
            else:
                mean_text = f"Mean={mean_val:.2f}"
            if academic_annotation_layout:
                ax.annotate(
                    mean_text,
                    xy=(mean_val, 0.72),
                    xycoords=("data", "axes fraction"),
                    xytext=(7, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=8,
                    color=COLOR_MEAN_LINE,
                    clip_on=False,
                    zorder=6,
                )
            else:
                ax.annotate(
                    mean_text,
                    xy=(mean_val, y_lim_top * 0.72),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    rotation=0,
                    fontsize=7.5,
                    color=COLOR_MEAN_LINE,
                    clip_on=False,
                    zorder=6,
                )
    elif show_mean_label and mean_label_mode == "overall" and not kde_split:
        _draw_plain_note(ax, f"Overall mean = {mean_val:.2f}", y=0.72)

    if show_kde_peak and not kde_split and y_fit.size and float(y_fit.max()) > 0:
        peak_idx = int(np.argmax(y_fit))
        peak_x = float(x_fit[peak_idx])
        peak_y = float(y_fit[peak_idx])
        ax.plot(peak_x, peak_y, "o", color=COLOR_AST_BLUE, markersize=4.5, zorder=7)
        if kde_peak_round:
            peak_text = f"KDE peak ≈ {int(round(peak_x))}"
        else:
            peak_text = f"Mode={peak_x:.1f}"
        ax.annotate(
            peak_text,
            xy=(peak_x, peak_y),
            xytext=(8, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            rotation=0,
            fontsize=7.5,
            color=COLOR_AST_BLUE,
            clip_on=False,
            zorder=7,
        )

    if truncation_note and not kde_split:
        if academic_annotation_layout:
            ax.text(
                0.02,
                0.97,
                truncation_note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#555555",
                zorder=9,
            )
        else:
            ax.text(
                0.02,
                0.98,
                truncation_note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.0,
                color="#555555",
                zorder=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5},
            )

    ax.set_xlim(0.0, float(x_max))
    _apply_panel_caption(ax, title, below=title_below)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    _style_distribution_axes(ax)

    draw_minmax = show_legend if show_minmax_box is None else show_minmax_box
    if full_range_plain and not kde_split:
        if academic_annotation_layout:
            _draw_plain_note(ax, _format_full_range_text(values), x=0.98, y=0.97, fontsize=8)
        else:
            _draw_plain_note(ax, _format_full_range_text(values))
    elif show_legend:
        handles = _density_legend_handles()
        leg = ax.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#B0B0B0",
            fontsize=7.5,
            framealpha=0.95,
            handlelength=1.6,
            borderpad=0.45,
            labelspacing=0.35,
            title=_format_minmax_text(values, multiline=False),
            title_fontsize=FONT_BOX,
        )
        leg._legend_box.align = "left"
        leg.get_frame().set_linewidth(0.8)
    elif draw_minmax:
        _draw_minmax_box(ax, values, multiline=True)
    return {"mean": mean_val, "hidden_ratio": hidden_ratio}


def _draw_discrete_count_bars(
    ax,
    *,
    categories: list[str] | np.ndarray,
    counts: np.ndarray,
    ratios: np.ndarray,
    values: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str = "Pages",
    show_minmax_box: bool = True,
    show_count_labels: bool = True,
    title_below: bool = False,
    academic_bar_labels: bool = False,
    bar_colors: Sequence[str] | None = None,
    hard_counts: np.ndarray | None = None,
    non_hard_counts: np.ndarray | None = None,
    n_all: int | None = None,
    n_hard: int | None = None,
    n_non_hard: int | None = None,
    label_focus_range: tuple[int, int] | None = None,
    wide_spacing_categories: set[str] | None = None,
) -> None:
    """Discrete integer histogram: single-color or grouped hard/non-hard/all bars."""
    categories = list(categories)
    hard_counts_arr = None if hard_counts is None else np.asarray(hard_counts, dtype=np.float64)
    non_hard_counts_arr = None if non_hard_counts is None else np.asarray(non_hard_counts, dtype=np.float64)
    grouped = (
        hard_counts_arr is not None
        and non_hard_counts_arr is not None
        and hard_counts_arr.size == len(categories)
        and non_hard_counts_arr.size == len(categories)
    )
    if grouped:
        n_all_v = int(n_all if n_all is not None else int((hard_counts_arr + non_hard_counts_arr).sum()))
        n_hard_v = int(n_hard if n_hard is not None else int(hard_counts_arr.sum()))
        n_non_hard_v = int(n_non_hard if n_non_hard is not None else int(non_hard_counts_arr.sum()))
        _draw_sampling_share_bars(
            ax,
            categories=categories,
            hard_counts=hard_counts_arr,
            non_hard_counts=non_hard_counts_arr,
            n_all=n_all_v,
            n_hard=n_hard_v,
            n_non_hard=n_non_hard_v,
            xlabel=xlabel,
            title=title,
            title_below=title_below,
            label_focus_range=label_focus_range,
            wide_spacing_categories=wide_spacing_categories,
        )
        return

    counts = np.asarray(counts, dtype=np.float64)
    ratios = np.asarray(ratios, dtype=np.float64)
    xs = np.arange(len(categories), dtype=np.float64)
    if bar_colors is None:
        colors = [COLOR_AST_BAR] * len(categories)
    else:
        colors = list(bar_colors)
        if len(colors) < len(categories):
            colors.extend([COLOR_AST_BAR] * (len(categories) - len(colors)))
    bars = ax.bar(
        xs,
        counts,
        width=0.8,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    ymax = float(counts.max()) if counts.size else 1.0
    pct_gap = 0.14 * ymax
    for index, x in enumerate(xs):
            count = float(counts[index])
            ratio = float(ratios[index])
            c = int(count)
            if c <= 0:
                continue
            bar = bars[index]
            h = float(bar.get_height())
            bar_center = float(bar.get_x() + bar.get_width() / 2.0)
            bar_color = colors[index] if bar_colors is not None else COLOR_AST_BAR
            inside_color = _bar_label_text_color(bar_color)
            if show_count_labels:
                if academic_bar_labels:
                    min_inside_height = 0.12 * ymax
                    pct_text = format_ratio_pct(float(ratio))
                    if h >= min_inside_height:
                        ax.text(
                            bar_center,
                            h * 0.47,
                            format_count(c),
                            ha="center",
                            va="center",
                            color=inside_color,
                            fontsize=9,
                            zorder=5,
                        )
                        ax.annotate(
                            pct_text,
                            xy=(bar_center, h),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color="#333333",
                            zorder=5,
                        )
                    else:
                        ax.annotate(
                            format_count(c),
                            xy=(bar_center, h),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color="#222222",
                            zorder=5,
                        )
                        ax.annotate(
                            pct_text,
                            xy=(bar_center, h),
                            xytext=(0, 12),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color="#333333",
                            zorder=5,
                        )
                    continue
                if h > 0.18 * ymax:
                    ax.text(
                        x,
                        h * 0.48,
                        format_count(c),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=FONT_ANNOT,
                        zorder=5,
                    )
                    pct_y = h + pct_gap
                else:
                    ax.text(
                        x,
                        h + 0.025 * ymax,
                        format_count(c),
                        ha="center",
                        va="bottom",
                        color="#222222",
                        fontsize=FONT_ANNOT,
                        zorder=5,
                    )
                    pct_y = h + 0.025 * ymax + pct_gap
            else:
                pct_y = h + 0.04 * ymax
            ax.text(
                x,
                pct_y,
                format_ratio_pct(float(ratio)),
                ha="center",
                va="bottom",
                color="#333333",
                fontsize=max(6.5, FONT_ANNOT - 0.5),
                zorder=5,
            )

    if show_minmax_box:
        _draw_minmax_box(ax, values, multiline=True)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in categories])
    if academic_bar_labels:
        ax.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
    else:
        ax.set_ylim(0.0, ymax * 1.38 if ymax > 0 else 1.0)
    _apply_panel_caption(ax, title, below=title_below)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    _style_distribution_axes(ax)


def _truncation_note_absolute(*, threshold: float, values: np.ndarray, compact: bool = False) -> str:
    n = int(values.size)
    hidden = int(np.sum(values > threshold + 1e-9))
    pct = 100.0 * hidden / n if n else 0.0
    thr = int(round(threshold)) if abs(threshold - round(threshold)) < 1e-6 else threshold
    thr_txt = f"{thr:d}" if isinstance(thr, int) else f"{thr:g}"
    if compact:
        return f"Trunc. @{thr_txt}; {pct:.1f}% hidden (n={hidden})"
    return f"Truncated at {thr_txt}; {pct:.1f}% of outlier pages not shown (n_out={hidden})"


def _truncation_note_percentile(
    *, percentile: float, threshold: float, values: np.ndarray, compact: bool = False
) -> str:
    n = int(values.size)
    hidden = int(np.sum(values > threshold + 1e-9))
    pct = 100.0 * hidden / n if n else 0.0
    thr = int(round(threshold)) if abs(threshold - round(threshold)) < 1e-6 else float(np.floor(threshold))
    if not isinstance(thr, int):
        thr = int(thr)
    if compact:
        return f"Trunc. @P{percentile:g}={thr}; {pct:.1f}% hidden (n={hidden})"
    return (
        f"Truncated at {percentile:g}th percentile ({thr}); "
        f"{pct:.1f}% of pages not shown (n_out={hidden})"
    )


@with_locked_pyplot
def _save_fig6_ast_panel(
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
) -> dict[str, Path]:
    """Figure 6-1: AST trees / nodes / depth panel (density + discrete histogram)."""
    import matplotlib.pyplot as plt

    apply_chapter6_style(plt)
    n = len(page_rows)
    values_tree = np.array([row.ast_tree_count for row in page_rows], dtype=np.float64)
    values_node = np.array([row.total_ast_node_count for row in page_rows], dtype=np.float64)
    values_depth = np.array([row.max_ast_depth for row in page_rows], dtype=np.float64)
    frame_depth = build_fig6_1_plot_data(page_rows)
    frame_depth = frame_depth[frame_depth["metric"] == "max_ast_depth"].reset_index(drop=True)

    x_tree = 50.0
    x_node = float(np.ceil(np.percentile(values_node, 95)))

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0), constrained_layout=True)
    _draw_density_hist_kde(
        axes[0],
        values=values_tree,
        x_max=x_tree,
        title="(a) AST trees per page",
        xlabel="ast_tree_count",
        truncation_note=_truncation_note_absolute(threshold=x_tree, values=values_tree),
        target_bins=25,
        show_legend=True,
    )
    _draw_density_hist_kde(
        axes[1],
        values=values_node,
        x_max=x_node,
        title="(b) AST nodes per page",
        xlabel="total_ast_node_count",
        truncation_note=_truncation_note_percentile(percentile=95, threshold=x_node, values=values_node),
        target_bins=24,
        show_legend=True,
    )
    _draw_discrete_count_bars(
        axes[2],
        categories=frame_depth["bin_label"].tolist(),
        counts=frame_depth["page_count"].to_numpy(),
        ratios=frame_depth["page_ratio"].to_numpy(),
        values=values_depth,
        title="(c) Max AST depth per page",
        xlabel="max_ast_depth",
    )

    fig.suptitle(
        f"Figure 6-1 Distributions of AST scale metrics per page (n={n:,} pages)\n"
        "(a)(b) Light shade: empirical density histogram; dark line: KDE fit.  "
        "(c) Histogram of max AST depth.",
        fontsize=FONT_PANEL_SUPTITLE,
        linespacing=1.35,
    )
    paths = save_figure_outputs(fig, figure_stem)
    plt.close(fig)
    return paths


@with_locked_pyplot
def _save_fig6_4_9_10_panel(
    page_rows: Sequence[PageLatexMetricsRow],
    figure_stem: Path,
    csv_path: Path,
) -> dict[str, Path]:
    """Figure 6-4: total tokens / distinct tokens / structure-type count panel."""
    import matplotlib.pyplot as plt
    import pandas as pd

    apply_chapter6_style(plt)
    n = len(page_rows)
    values_total = np.array([row.total_token_count for row in page_rows], dtype=np.float64)
    values_distinct = np.array([row.distinct_token_count for row in page_rows], dtype=np.float64)
    frame_total = build_fig6_raw_count_curve_data(page_rows, field="total_token_count")
    frame_distinct = build_fig6_raw_count_curve_data(page_rows, field="distinct_token_count")
    structure = build_fig6_4_plot_data(page_rows)
    typ = structure[
        (structure["data_type"] == "structure_type_count") & (structure["category"] != "9")
    ]

    x_total = float(np.ceil(np.percentile(values_total, 99))) if values_total.size else 1.0
    x_distinct = float(np.ceil(np.percentile(values_distinct, 99))) if values_distinct.size else 1.0
    # For discrete structure-type count, build a page-count series aligned with categories.
    type_values = np.array([row.distinct_structure_type_count for row in page_rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.0), constrained_layout=True)
    _draw_density_hist_kde(
        axes[0],
        values=values_total,
        x_max=x_total,
        title="(a) Total tokens per page",
        xlabel="total_token_count",
        truncation_note=_truncation_note_percentile(percentile=99, threshold=x_total, values=values_total),
        target_bins=28,
        show_legend=True,
    )
    _draw_density_hist_kde(
        axes[1],
        values=values_distinct,
        x_max=x_distinct,
        title="(b) Distinct tokens per page",
        xlabel="distinct_token_count",
        truncation_note=_truncation_note_percentile(
            percentile=99, threshold=x_distinct, values=values_distinct
        ),
        target_bins=22,
        show_legend=True,
    )
    _draw_discrete_count_bars(
        axes[2],
        categories=typ["category"].tolist(),
        counts=typ["page_count"].to_numpy(),
        ratios=typ["page_ratio"].to_numpy(),
        values=type_values,
        title="(c) Distinct structure types per page",
        xlabel="Distinct structure type count",
        ylabel="Pages",
    )

    fig.suptitle(
        f"Figure 6-4 Distributions of token scale and structure-type count per page (n={n:,} pages)\n"
        "(a)(b) Light shade: empirical density histogram; dark line: KDE fit.  "
        "(c) Histogram of distinct structure-type count.",
        fontsize=FONT_PANEL_SUPTITLE,
        linespacing=1.35,
    )

    token_meta = pd.concat(
        [
            frame_total.assign(panel="total_token_count", display_x_max=x_total),
            frame_distinct.assign(panel="distinct_token_count", display_x_max=x_distinct),
        ],
        ignore_index=True,
    )
    plot_data = pd.concat(
        [token_meta, typ.assign(panel="structure_type_count")],
        ignore_index=True,
        sort=False,
    )
    return _finalize_figure(fig, figure_stem=figure_stem, csv_path=csv_path, plot_data=plot_data)


def export_page_latex_figures(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    figures_dir: Path,
) -> dict[str, Path]:
    """Export the two consolidated page-level panel figures (SVG/PNG + plot data).

    Outputs:
      - ``page_ast_scale``      (was fig6_1_3_page_ast_scale): AST trees/nodes/depth panel.
      - ``page_tokens_structure`` (was fig6_4_9_10_tokens_structure): token + structure panel.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir = figures_dir.parent / "plot_data"
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    def record(key: str, result: dict[str, Path]) -> None:
        outputs[f"{key}.png"] = result["png"]
        outputs[f"{key}.csv"] = result["csv"]

    panel = _save_fig6_ast_panel(page_rows, figures_dir / "page_ast_scale")
    outputs["page_ast_scale.png"] = panel["png"]
    record(
        "page_tokens_structure",
        _save_fig6_4_9_10_panel(
            page_rows,
            figures_dir / "page_tokens_structure",
            plot_data_dir / "page_tokens_structure_plot_data.csv",
        ),
    )

    return outputs
