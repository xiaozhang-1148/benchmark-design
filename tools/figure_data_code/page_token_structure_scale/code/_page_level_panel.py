"""Shared page-level distribution panel drawing helpers."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

COLOR_KDE_ALL = "#B7D0EA"
KDE_LINEWIDTH = 1.15
KDE_LINE_ALPHA = 1
COLOR_HARD = "#E89898"
COLOR_NON_HARD = "#A8D4A0"
KDE_MARKER_ALL = "o"
KDE_MARKER_NON_HARD = "s"
KDE_MARKER_HARD = "D"
KDE_MARKERSIZE = 4.5
KDE_FIT_POINTS = 512
KDE_PEAK_LABEL_Y_OFFSET = 8
KDE_FILL_ALPHA = 0.20
GRID_COLOR = "#8A8A8A"
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.65
GRID_ALPHA = 0.72
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
DEPTH_LABELS = ("0", "1", "2", "3", "4", "5")


def legend_handles(*, n: int, n_non_hard: int, n_hard: int) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLOR_KDE_ALL,
            linewidth=KDE_LINEWIDTH,
            linestyle="--",
            alpha=KDE_LINE_ALPHA,
            marker=KDE_MARKER_ALL,
            markersize=KDE_MARKERSIZE,
            label=f"All (n={n:,})",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_NON_HARD,
            linewidth=KDE_LINEWIDTH,
            linestyle="-",
            alpha=KDE_LINE_ALPHA,
            marker=KDE_MARKER_NON_HARD,
            markersize=KDE_MARKERSIZE,
            label=f"Non-hard (n={n_non_hard:,})",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_HARD,
            linewidth=KDE_LINEWIDTH,
            linestyle="-",
            alpha=KDE_LINE_ALPHA,
            marker=KDE_MARKER_HARD,
            markersize=KDE_MARKERSIZE,
            label=f"Hard (n={n_hard:,})",
        ),
    ]


def place_panel_captions(fig, axes, captions: list[str], *, gap: float = 0.012) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()) for ax in axes]
    y = min(b.y0 for b in bboxes) - gap
    for caption, bbox in zip(captions, bboxes, strict=True):
        fig.text(bbox.x0 + bbox.width / 2, y, caption, ha="center", va="top", fontsize=11, color="#222222")


def split_metrics_by_hard(frame: pd.DataFrame, column: str, kept_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept = pd.read_csv(kept_csv)
    source_map = {
        Path(b).stem: str(s).strip()
        for b, s in zip(kept["basename"], kept["post_selection_source"], strict=True)
    }
    frame = frame.copy()
    frame["post_selection_source"] = frame["image_id"].map(source_map)
    if frame["post_selection_source"].isna().any():
        missing = int(frame["post_selection_source"].isna().sum())
        raise ValueError(f"{missing} rows missing post_selection_source in {kept_csv}")
    values = frame[column].to_numpy(dtype=np.float64)
    hard = frame.loc[frame["post_selection_source"] == "hard", column].to_numpy(dtype=np.float64)
    non_hard = frame.loc[frame["post_selection_source"] != "hard", column].to_numpy(dtype=np.float64)
    return values, hard, non_hard


def split_density_by_hard(frame: pd.DataFrame, kept_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept = pd.read_csv(kept_csv)
    source_map = {
        Path(b).stem: str(s).strip()
        for b, s in zip(kept["basename"], kept["post_selection_source"], strict=True)
    }
    frame = frame.copy()
    frame["join_key"] = frame["relative_path"].map(lambda p: Path(str(p)).stem)
    frame["post_selection_source"] = frame["join_key"].map(source_map)
    if frame["post_selection_source"].isna().any():
        missing = int(frame["post_selection_source"].isna().sum())
        raise ValueError(f"{missing} rows missing post_selection_source in {kept_csv}")
    densities = frame["foreground_density"].to_numpy(dtype=np.float64)
    hard = frame.loc[frame["post_selection_source"] == "hard", "foreground_density"].to_numpy(dtype=np.float64)
    non_hard = frame.loc[frame["post_selection_source"] != "hard", "foreground_density"].to_numpy(dtype=np.float64)
    return densities, hard, non_hard


def count_by_category_and_hard(
    frame: pd.DataFrame,
    column: str,
    categories: list[str],
    kept_csv: Path,
) -> tuple[np.ndarray, np.ndarray]:
    kept = pd.read_csv(kept_csv)
    source_map = {
        Path(b).stem: str(s).strip()
        for b, s in zip(kept["basename"], kept["post_selection_source"], strict=True)
    }
    frame = frame.copy()
    frame["post_selection_source"] = frame["image_id"].map(source_map)
    if frame["post_selection_source"].isna().any():
        missing = int(frame["post_selection_source"].isna().sum())
        raise ValueError(f"{missing} rows missing post_selection_source in {kept_csv}")
    hard_counts: list[int] = []
    non_hard_counts: list[int] = []
    for category in categories:
        subset = frame[frame[column].astype(int) == int(category)]
        hard_counts.append(int((subset["post_selection_source"] == "hard").sum()))
        non_hard_counts.append(int((subset["post_selection_source"] != "hard").sum()))
    return (
        np.asarray(hard_counts, dtype=np.float64),
        np.asarray(non_hard_counts, dtype=np.float64),
    )


def bar_frame(series: pd.Series, categories: list[str], total: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    counts_map = Counter(series.astype(int))
    counts = np.array([counts_map.get(int(c), 0) for c in categories], dtype=np.float64)
    ratios = counts / float(total) if total else counts
    return categories, counts, ratios


def _kde(values: np.ndarray, x_fit: np.ndarray) -> np.ndarray:
    if values.size < 2 or float(np.std(values)) <= 1e-12:
        return np.zeros_like(x_fit)
    return np.asarray(gaussian_kde(values, bw_method="scott")(x_fit), dtype=np.float64)


def _style_axes(ax, *, axis_frame: str = "xy", grid: str = "y") -> None:
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.5)
    ax.set_axisbelow(True)
    if grid == "both":
        ax.grid(True, axis="both", color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    elif grid == "y":
        ax.grid(True, axis="y", color=GRID_COLOR, linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
        ax.grid(False, axis="x")
    else:
        ax.grid(False)
    visible_spines = {
        "closed": ("top", "right", "left", "bottom"),
        "xy": ("left", "bottom"),
        "x_only": ("bottom",),
    }.get(axis_frame, ("left", "bottom"))
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(spine in visible_spines)
        if spine in visible_spines:
            ax.spines[spine].set_linewidth(0.9)
            ax.spines[spine].set_color("#444444")


def _plot_kde_curve(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    *,
    linestyle: str,
    linewidth: float,
    zorder: int,
    fill_kde: bool,
) -> None:
    if not y.size or float(y.max()) <= 0:
        return
    if fill_kde:
        ax.fill_between(x, y, 0.0, color=color, alpha=KDE_FILL_ALPHA, linewidth=0.0, zorder=zorder - 1)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=KDE_LINE_ALPHA,
        zorder=zorder,
        solid_capstyle="round",
    )


def _density_count_xticks(*, head_tick: int, tail_tick: int) -> list[int]:
    if tail_tick <= head_tick:
        return [head_tick]
    span = tail_tick - head_tick
    step = 200 if span > 200 else 20
    ticks = [head_tick]
    next_tick = ((head_tick // step) + 1) * step
    if next_tick <= head_tick:
        next_tick += step
    ticks.extend(range(next_tick, tail_tick, step))
    if len(ticks) > 1 and tail_tick - ticks[-1] < step * 0.75:
        ticks.pop()
    if ticks[-1] != tail_tick:
        ticks.append(tail_tick)
    return ticks


def _density_percent_xticks(*, head_pct: float, tail_pct: float) -> list[float]:
    head = round(head_pct, 1)
    tail = round(tail_pct, 1)
    if tail <= head:
        return [head]
    span = tail - head
    step = 5.0 if span > 15 else 2.0 if span > 8 else 1.0
    ticks = [head]
    next_tick = math.ceil(head / step) * step
    if next_tick <= head:
        next_tick += step
    while next_tick < tail - step * 0.25:
        ticks.append(round(next_tick, 1))
        next_tick += step
    if len(ticks) > 1 and tail - ticks[-1] < step * 0.75:
        ticks.pop()
    if ticks[-1] != tail:
        ticks.append(tail)
    return ticks


def _annotate_kde_peak(
    ax,
    peak_x: float,
    y_fit: np.ndarray,
    x_fit: np.ndarray,
    color: str,
    *,
    label: str,
    marker: str,
    label_below: bool = False,
) -> None:
    if y_fit.size == 0 or float(y_fit.max()) <= 0:
        return
    peak_idx = int(np.argmax(y_fit))
    peak_y = float(y_fit[peak_idx])
    ax.plot(peak_x, peak_y, marker=marker, color=color, markersize=KDE_MARKERSIZE + 0.5, zorder=7)
    if label_below:
        label_offset = (1, -KDE_PEAK_LABEL_Y_OFFSET)
        label_va = "top"
    else:
        label_offset = (0, KDE_PEAK_LABEL_Y_OFFSET)
        label_va = "bottom"
    ax.annotate(
        label,
        xy=(peak_x, peak_y),
        xytext=label_offset,
        textcoords="offset points",
        ha="center",
        va=label_va,
        fontsize=7.5,
        color=color,
        clip_on=False,
        zorder=7,
    )


def _apply_xlim_from_ticks(ax, ticks: list[float]) -> None:
    if len(ticks) > 1:
        left_pad = (ticks[1] - ticks[0]) * 0.12
        right_pad = (ticks[-1] - ticks[-2]) * 0.12
        ax.set_xlim(ticks[0] - left_pad, ticks[-1] + right_pad)
    else:
        ax.set_xlim(ticks[0], ticks[0])


def _share_label_text(share_pct: float, category: str, *, label_focus_range: tuple[int, int] | None) -> str | None:
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
    categories: list[str],
    *,
    wide_spacing_categories: set[str] | None = None,
    wide_category_gap: float = SHARE_WIDE_CATEGORY_GAP,
) -> np.ndarray:
    wide_spacing_categories = wide_spacing_categories or set()
    xs: list[float] = []
    pos = 0.0
    for index, category in enumerate(categories):
        if index > 0 and str(category) in wide_spacing_categories:
            pos += wide_category_gap
        xs.append(pos)
        pos += 1.0
    return np.asarray(xs, dtype=np.float64)


def draw_density_count(
    ax,
    values: np.ndarray,
    xlabel: str,
    *,
    head_tick: int,
    tail_tick: int,
    hard_values: np.ndarray,
    non_hard_values: np.ndarray,
    round_peak: bool = True,
    fill_kde: bool = False,
    axis_frame: str = "xy",
    grid: str = "y",
) -> None:
    values = np.asarray(values, dtype=np.float64)
    x_min = float(head_tick)
    x_max = float(tail_tick)
    x_fit = np.linspace(x_min, x_max, KDE_FIT_POINTS)
    kde_series = [
        (COLOR_KDE_ALL, _kde(values, x_fit), "--", KDE_LINEWIDTH, 3, KDE_MARKER_ALL),
        (COLOR_HARD, _kde(hard_values, x_fit), "-", KDE_LINEWIDTH, 5, KDE_MARKER_HARD),
        (COLOR_NON_HARD, _kde(non_hard_values, x_fit), "-", KDE_LINEWIDTH, 4, KDE_MARKER_NON_HARD),
    ]
    y_fit = np.zeros_like(x_fit)
    for color, curve, linestyle, linewidth, zorder, marker in kde_series:
        _plot_kde_curve(
            ax,
            x_fit,
            curve,
            color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            fill_kde=fill_kde,
        )
        y_fit = np.maximum(y_fit, curve)
        peak_x = float(x_fit[int(np.argmax(curve))]) if curve.size else x_min
        label = str(int(round(peak_x))) if round_peak else f"{peak_x:.1f}"
        _annotate_kde_peak(ax, peak_x, curve, x_fit, color, label=label, marker=marker)
    y_top = float(y_fit.max()) if y_fit.size else 0.0
    ax.set_ylim(0.0, y_top * 1.08 if y_top > 0 else 1.0)
    ticks = _density_count_xticks(head_tick=head_tick, tail_tick=tail_tick)
    ax.set_xticks(ticks)
    _apply_xlim_from_ticks(ax, [float(t) for t in ticks])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    _style_axes(ax, axis_frame=axis_frame, grid=grid)


def draw_density_percent(
    ax,
    densities: np.ndarray,
    xlabel: str,
    *,
    hard_values: np.ndarray,
    non_hard_values: np.ndarray,
    fill_kde: bool = False,
    axis_frame: str = "xy",
    grid: str = "y",
) -> None:
    densities = np.asarray(densities, dtype=np.float64)
    x_min = float(densities.min())
    x_max = float(densities.max())
    head_pct = round(x_min * 100.0, 1)
    tail_pct = round(x_max * 100.0, 1)
    x_fit = np.linspace(x_min, x_max, KDE_FIT_POINTS)
    kde_series = [
        (COLOR_KDE_ALL, _kde(densities, x_fit), "--", KDE_LINEWIDTH, 3, KDE_MARKER_ALL),
        (COLOR_HARD, _kde(hard_values, x_fit), "-", KDE_LINEWIDTH, 5, KDE_MARKER_HARD),
        (COLOR_NON_HARD, _kde(non_hard_values, x_fit), "-", KDE_LINEWIDTH, 4, KDE_MARKER_NON_HARD),
    ]
    y_top = 0.0
    for color, curve, linestyle, linewidth, zorder, marker in kde_series:
        x_plot = x_fit * 100.0
        _plot_kde_curve(
            ax,
            x_plot,
            curve,
            color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
            fill_kde=fill_kde,
        )
        y_top = max(y_top, float(curve.max()) if curve.size else 0.0)
        peak_x = float(x_fit[int(np.argmax(curve))]) * 100.0 if curve.size else head_pct
        _annotate_kde_peak(
            ax,
            peak_x,
            curve,
            x_fit,
            color,
            label=f"{peak_x:.1f}%",
            marker=marker,
            label_below=color == COLOR_NON_HARD,
        )
    if y_top <= 0:
        y_top = 1.0
    ax.set_ylim(0.0, min(22.0, max(21.0, y_top * 1.05)))
    ticks = _density_percent_xticks(head_pct=head_pct, tail_pct=tail_pct)
    ax.set_xticks(ticks)
    _apply_xlim_from_ticks(ax, ticks)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    _style_axes(ax, axis_frame=axis_frame, grid=grid)


def draw_share_bars(
    ax,
    categories: list[str],
    hard_counts: np.ndarray,
    non_hard_counts: np.ndarray,
    xlabel: str,
    *,
    n_all: int,
    n_hard: int,
    n_non_hard: int,
    label_focus_range: tuple[int, int] | None = None,
    wide_spacing_categories: set[str] | None = None,
    bar_width: float = SHARE_BAR_WIDTH,
    category_edge_pad: float = SHARE_CATEGORY_EDGE_PAD,
    bar_offsets: tuple[float, float, float] = SHARE_BAR_OFFSETS,
    wide_bar_offsets: tuple[float, float, float] = SHARE_WIDE_BAR_OFFSETS,
    wide_category_gap: float = SHARE_WIDE_CATEGORY_GAP,
    axis_frame: str = "xy",
    grid: str = "y",
) -> None:
    hard_counts = np.asarray(hard_counts, dtype=np.float64)
    non_hard_counts = np.asarray(non_hard_counts, dtype=np.float64)
    all_counts = hard_counts + non_hard_counts
    xs = _build_share_category_xs(
        categories,
        wide_spacing_categories=wide_spacing_categories,
        wide_category_gap=wide_category_gap,
    )

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
        if wide_spacing_categories and str(category) in wide_spacing_categories:
            offsets = wide_bar_offsets
        else:
            offsets = bar_offsets
        series = [
            (offsets[0], float(non_hard_pct[category_index]), COLOR_NON_HARD, SHARE_NON_HARD_LABEL_OFFSET),
            (offsets[1], float(hard_pct[category_index]), COLOR_HARD, SHARE_HARD_LABEL_OFFSET),
            (offsets[2], float(all_pct[category_index]), COLOR_KDE_ALL, SHARE_ALL_LABEL_OFFSET),
        ]
        for offset, pct, color, label_offset in series:
            if pct <= 0:
                continue
            bar = ax.bar(
                x_center + offset,
                pct,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            label = _share_label_text(pct, str(category), label_focus_range=label_focus_range)
            if label is None:
                continue
            center = float(bar[0].get_x() + bar[0].get_width() / 2.0)
            label_y = max(float(bar[0].get_height()), label_y_floor)
            ax.annotate(
                label,
                xy=(center, label_y),
                xytext=label_offset,
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=SHARE_LABEL_FONTSIZE,
                color="#333333",
                zorder=6,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(categories)
    ax.set_xlim(float(xs[0]) - category_edge_pad, float(xs[-1]) + category_edge_pad)
    ax.set_ylim(0.0, ymax * SHARE_YLIM_PAD if ymax > 0 else 1.0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Percentage within group (%)", fontsize=10)
    _style_axes(ax, axis_frame=axis_frame, grid=grid)
