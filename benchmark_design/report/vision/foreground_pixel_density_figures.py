"""Summary figures for foreground pixel density exports."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot
from benchmark_design.report.vision.foreground_pixel_density_stats import (
    DensityBandSummary,
    DensityDistributionStats,
    compute_block_density_bands,
    compute_block_density_stats,
    compute_page_density_bands,
    compute_page_density_stats,
    density_to_pct,
)
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageFlowStructureResult
from benchmark_design.vision.foreground_load.models import PageForegroundLoadResult
from benchmark_design.vision.masks import UnifiedPageMaskBundle

BAND_COLOR_MIDDLE = "#3498DB"
BAND_COLOR_SPARSE = "#F39C12"
BAND_COLOR_DENSE = "#C0392B"
PAGE_COLOR = "#2E86C1"
BLOCK_COLOR = "#E67E22"


def _require_matplotlib():
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    return plt


def _band_colors(n_bands: int) -> list[str]:
    if n_bands == 0:
        return []
    colors = [BAND_COLOR_MIDDLE] * n_bands
    colors[0] = BAND_COLOR_SPARSE
    colors[-1] = BAND_COLOR_DENSE
    return colors


@with_locked_pyplot
def _draw_density_bands(
    band_summary: DensityBandSummary,
    output_path: Path,
    *,
    density_symbol: str,
    y_label: str,
    title: str,
    caption: str,
) -> bool:
    if band_summary.total == 0:
        return False
    plt = _require_matplotlib()

    range_labels = [band.range_label for band in band_summary.bands]
    ratios_pct = [band.ratio * 100.0 for band in band_summary.bands]
    counts = [band.count for band in band_summary.bands]
    colors = _band_colors(len(range_labels))

    fig, axis = plt.subplots(figsize=(10, 5.5))
    x_positions = range(len(range_labels))
    bars = axis.bar(
        x_positions,
        ratios_pct,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        width=0.72,
    )
    for bar, count, ratio_pct in zip(bars, counts, ratios_pct, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.6,
            f"{count:,} / {ratio_pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(range_labels, rotation=0)
    axis.set_xlabel(f"{density_symbol} density range")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.set_ylim(0, max(ratios_pct) * 1.22 + 2.0)
    axis.grid(True, axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.01,
        caption,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#566573",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _draw_page_density_bands(band_summary: DensityBandSummary, output_path: Path) -> bool:
    return _draw_density_bands(
        band_summary,
        output_path,
        density_symbol=r"$D_{page}$",
        y_label="Percentage of pages",
        title="Page-level Foreground Pixel Density Distribution",
        caption=(
            f"n = {band_summary.total:,} pages; "
            "Foreground pixel density over annotated page regions (R_eff)"
        ),
    )


def _draw_block_density_bands(band_summary: DensityBandSummary, output_path: Path) -> bool:
    return _draw_density_bands(
        band_summary,
        output_path,
        density_symbol=r"$D_{block}$",
        y_label="Percentage of txtBlocks",
        title="Block-level Foreground Pixel Density Distribution",
        caption=(
            f"n = {band_summary.total:,} txtBlocks; "
            "Foreground pixel density with shared dataset-level tau_D"
        ),
    )


@with_locked_pyplot
def _draw_density_level_comparison(
    page_stats: DensityDistributionStats,
    block_stats: DensityDistributionStats,
    output_path: Path,
) -> bool:
    if page_stats.n == 0 or block_stats.n == 0:
        return False
    plt = _require_matplotlib()

    metric_labels = ("Mean", "Median", "Area-weighted")
    page_values = (
        density_to_pct(page_stats.mean),
        density_to_pct(page_stats.median),
        density_to_pct(page_stats.area_weighted_density or 0.0),
    )
    block_values = (
        density_to_pct(block_stats.mean),
        density_to_pct(block_stats.median),
        density_to_pct(block_stats.area_weighted_density or 0.0),
    )

    x_positions = range(len(metric_labels))
    bar_width = 0.34
    fig, axis = plt.subplots(figsize=(8, 5))
    page_bars = axis.bar(
        [x - bar_width / 2 for x in x_positions],
        page_values,
        width=bar_width,
        label=r"$D_{page}$",
        color=PAGE_COLOR,
        edgecolor="white",
    )
    block_bars = axis.bar(
        [x + bar_width / 2 for x in x_positions],
        block_values,
        width=bar_width,
        label=r"$D_{block}$",
        color=BLOCK_COLOR,
        edgecolor="white",
    )

    for bars in (page_bars, block_bars):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.15,
                f"{bar.get_height():.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(metric_labels)
    axis.set_ylabel("Foreground pixel density (%)")
    axis.set_title("Page-level and block-level foreground density comparison")
    axis.set_ylim(0, max(*page_values, *block_values) * 1.18 + 0.5)
    axis.legend(loc="upper right", framealpha=0.9)
    axis.grid(True, axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def export_foreground_pixel_density_figures(
    results: list[PageForegroundLoadResult],
    pages: list[PageAnnotation],
    flow_results: list[PageFlowStructureResult],
    *,
    input_dir: Path,
    figures_root: Path,
    page_masks: dict[str, UnifiedPageMaskBundle] | None = None,
    include_level_comparison: bool = True,
) -> dict[str, bool]:
    del pages, flow_results, input_dir, page_masks

    page_stats = compute_page_density_stats(results)
    block_stats = compute_block_density_stats(results)
    page_band_summary = compute_page_density_bands(results)
    block_band_summary = compute_block_density_bands(results)

    counts: dict[str, bool] = {}
    counts["d_page_density_bands.png"] = _draw_page_density_bands(
        page_band_summary,
        figures_root / "d_page_density_bands.png",
    )
    counts["d_block_density_bands.png"] = _draw_block_density_bands(
        block_band_summary,
        figures_root / "d_block_density_bands.png",
    )
    if include_level_comparison:
        counts["density_level_comparison.png"] = _draw_density_level_comparison(
            page_stats,
            block_stats,
            figures_root / "density_level_comparison.png",
        )
    return counts
