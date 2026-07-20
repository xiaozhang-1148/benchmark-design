"""Chapter 4 distribution figures for line-level analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_design.line_level.models import LineMetricsRow, TargetPairRow
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.line_level.chapter_tables import (
    INK_STATE_POSITIVE_INK,
    INK_STATE_ZERO_INK,
    classify_bbox_outside_ink_state,
    ink_natural_state_rows,
)
from benchmark_design.report.pyplot_lock import with_locked_pyplot

FIGURE_DPI = 300
# Display cutoff for heavy right skew: keep bulk, drop sparse tail.
TAIL_QUANTILE = 0.95

BBOX_OUTSIDE_INK_RATIO_BIN_EDGES: tuple[float, ...] = (
    0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    float("inf"),
)
BBOX_OUTSIDE_INK_RATIO_BIN_LABELS: tuple[str, ...] = (
    "0%–1%",
    "1%–2%",
    "2%–3%",
    "3%–4%",
    "4%–5%",
    "≥5%",
)


def _bbox_outside_ink_ratio_bin_index(ratio: float) -> int:
    for index in range(len(BBOX_OUTSIDE_INK_RATIO_BIN_LABELS)):
        lower = BBOX_OUTSIDE_INK_RATIO_BIN_EDGES[index]
        upper = BBOX_OUTSIDE_INK_RATIO_BIN_EDGES[index + 1]
        if ratio >= lower and ratio < upper:
            return index
    return len(BBOX_OUTSIDE_INK_RATIO_BIN_LABELS) - 1


def _bbox_outside_ink_ratio_bin_counts(values: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(BBOX_OUTSIDE_INK_RATIO_BIN_LABELS), dtype=np.int64)
    for value in values:
        counts[_bbox_outside_ink_ratio_bin_index(float(value))] += 1
    return counts


def _positive_values(values: np.ndarray) -> np.ndarray:
    return values[values > 0]


def _truncate_tail(values: np.ndarray, *, quantile: float = TAIL_QUANTILE) -> tuple[np.ndarray, float, int]:
    """Keep values in (0, q]; returns filtered values, xmax, dropped_count."""
    if values.size == 0:
        return values, 0.0, 0
    xmax = float(np.quantile(values, quantile))
    if xmax <= 0:
        xmax = float(values.max())
    kept = values[values <= xmax]
    dropped = int(values.size - kept.size)
    return kept, xmax, dropped


def _annotate_hist_bars(ax, counts: np.ndarray, bin_edges: np.ndarray, *, total: int) -> None:
    """Annotate each non-empty bar with count and share of the plotted set."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    max_count = float(counts.max()) if counts.size else 1.0
    for center, count in zip(centers, counts, strict=True):
        if count <= 0:
            continue
        ratio = count / total if total else 0.0
        ax.text(
            center,
            count + max_count * 0.015,
            f"{int(count):,}\n{ratio * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
            linespacing=0.95,
        )
    ax.set_ylim(0.0, max_count * 1.35)


def _save_positive_truncated_hist(
    values: np.ndarray,
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str,
    bins: int = 25,
) -> Path | None:
    import matplotlib.pyplot as plt

    positive = _positive_values(np.asarray(values, dtype=np.float64))
    if positive.size == 0:
        return None
    kept, xmax, dropped = _truncate_tail(positive)
    if kept.size == 0:
        return None

    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=FIGURE_DPI)
    counts, bin_edges, _patches = ax.hist(
        kept,
        bins=min(bins, max(8, kept.size // 20)),
        range=(float(kept.min()), xmax),
        color=color,
        alpha=0.9,
        edgecolor="white",
    )
    # Drop an empty first bin that would include the exclusive 0 boundary only.
    # Histogram range starts at 0 for scale; values themselves are already > 0.
    _annotate_hist_bars(ax, counts.astype(np.float64), bin_edges, total=int(kept.size))
    ax.set_xlim(float(kept.min()), xmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    note = f"n = {kept.size:,} (values > 0, ≤ P{int(TAIL_QUANTILE * 100)}; dropped tail {dropped:,})"
    ax.text(0.98, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


@with_locked_pyplot
def export_orientation_abs_distribution(
    line_rows: list[LineMetricsRow],
    plots_dir: Path,
) -> Path | None:
    """Signed long-side angle α; 1° bins; symmetric display truncated at |α| P99.

    Percentages use the full orientation-valid set as denominator (including the
    dropped long tail beyond |P99|).
    """
    import matplotlib.pyplot as plt

    values = np.array(
        [
            float(row.orientation_deg)
            for row in line_rows
            if row.is_valid and row.orientation_direction_valid
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return None

    total_all = int(values.size)
    abs_values = np.abs(values)
    p99 = float(np.quantile(abs_values, 0.99))
    kept = values[abs_values <= p99]
    dropped = total_all - int(kept.size)
    if kept.size == 0:
        return None

    xmax = float(np.ceil(p99))
    if xmax < 1.0:
        xmax = 1.0
    bin_edges = np.arange(-xmax, xmax + 1.0, 1.0)

    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=FIGURE_DPI)
    counts, edges, _patches = ax.hist(
        kept,
        bins=bin_edges,
        color="#4472C4",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
    )
    _annotate_hist_bars(ax, counts.astype(np.float64), edges, total=total_all)
    ax.set_xlim(-xmax, xmax)
    ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    ax.grid(False)
    ax.set_title("Orientation from horizontal (valid lines)")
    ax.set_xlabel("orientation_deg (signed α, long side vs +x, degrees)")
    ax.set_ylabel("Count")
    ax.text(
        0.98,
        0.97,
        (
            f"shown n = {kept.size:,} (|α| ≤ P99 = {p99:.2f}°; dropped {dropped:,}); "
            f"bin = 1°; % of all {total_all:,}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "orientation_abs_distribution.png"
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


@with_locked_pyplot
def export_positive_ioa_distribution(
    pair_rows: list[TargetPairRow],
    plots_dir: Path,
) -> Path | None:
    """Positive IoA only; fixed 0.05 bins; full range including long tail."""
    import matplotlib.pyplot as plt

    values = np.array(
        [float(row.ioa) for row in pair_rows if row.ioa_positive and float(row.ioa) > 0.0],
        dtype=np.float64,
    )
    if values.size == 0:
        return None

    bin_width = 0.05
    xmax = max(bin_width, float(np.ceil(float(values.max()) / bin_width) * bin_width))
    bin_edges = np.arange(0.0, xmax + bin_width * 0.5, bin_width)

    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=FIGURE_DPI)
    counts, _edges, _patches = ax.hist(
        values,
        bins=bin_edges,
        color="#C55A11",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlim(0.0, float(bin_edges[-1]))
    ax.set_ylim(0.0, float(counts.max()) * 1.12 if counts.size else 1.0)
    ax.set_title("Positive IoA distribution (same-page unique pairs)")
    ax.set_xlabel("IoA = intersection / min(area_a, area_b)")
    ax.set_ylabel("Count")
    ax.text(
        0.98,
        0.97,
        f"n = {values.size:,}; bin width = {bin_width:g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "positive_ioa_distribution.png"
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


@with_locked_pyplot
def export_bbox_outside_ink_natural_states_figure(
    line_rows: list[LineMetricsRow],
    plots_dir: Path,
) -> Path | None:
    """Binary natural-state counts: zero-ink (incl. empty region) vs positive ink."""
    import matplotlib.pyplot as plt

    states = ink_natural_state_rows(line_rows)
    if not states:
        return None
    counts = [int(row["count"]) for row in states]
    colors = {
        INK_STATE_ZERO_INK: "#4472C4",
        INK_STATE_POSITIVE_INK: "#C00000",
    }
    bar_colors = [colors[str(row["state"])] for row in states]
    short_labels = ["区域内墨=0", "区域内墨>0"]

    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=FIGURE_DPI)
    ax.bar(range(len(counts)), counts, color=bar_colors, edgecolor="white")
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Neighboring context pixels: natural states")
    total = sum(counts) or 1
    for index, count in enumerate(counts):
        ax.text(
            index,
            count,
            f"{count:,}\n{count / total * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(counts) * 1.18)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / "bbox_outside_ink_natural_states.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


@with_locked_pyplot
def export_bbox_outside_ink_ratio_distribution(
    line_rows: list[LineMetricsRow],
    plots_dir: Path,
) -> Path | None:
    """Positive-ink interference ratio using fixed 1% interval bins up to ≥5%."""
    import matplotlib.pyplot as plt

    values = np.array(
        [
            float(row.bbox_outside_ink_ratio)
            for row in line_rows
            if row.is_valid
            and classify_bbox_outside_ink_state(row) == INK_STATE_POSITIVE_INK
            and row.bbox_outside_ink_ratio is not None
            and float(row.bbox_outside_ink_ratio) > 0.0
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return None

    total_all = int(values.size)
    counts = _bbox_outside_ink_ratio_bin_counts(values)
    ratios = counts / total_all

    _configure_matplotlib_fonts(plt)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=FIGURE_DPI)
    x = np.arange(len(BBOX_OUTSIDE_INK_RATIO_BIN_LABELS))
    bars = ax.bar(
        x,
        counts,
        color="#4472C4",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
        width=0.72,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(BBOX_OUTSIDE_INK_RATIO_BIN_LABELS)
    ax.set_ylim(0, max(int(counts.max()) * 1.18, 1))
    ax.grid(False)
    ax.set_title(f"Neighboring interference ratio ($n={total_all:,}$, positive-ink only)")
    ax.set_xlabel(r"Interference ratio interval ($D_{interference}$ in bbox \ mask)")
    ax.set_ylabel("Count")
    for bar, count, ratio in zip(bars, counts, ratios, strict=True):
        if count <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}\n{ratio * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "bbox_outside_ink_ratio_distribution.png"
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def export_chapter_distribution_figures(
    line_rows: list[LineMetricsRow],
    pair_rows: list[TargetPairRow],
    plots_dir: Path,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    orient = export_orientation_abs_distribution(line_rows, plots_dir)
    if orient is not None:
        outputs["orientation_abs_distribution"] = orient
    ratio = export_bbox_outside_ink_ratio_distribution(line_rows, plots_dir)
    if ratio is not None:
        outputs["bbox_outside_ink_ratio_distribution"] = ratio
    return outputs
