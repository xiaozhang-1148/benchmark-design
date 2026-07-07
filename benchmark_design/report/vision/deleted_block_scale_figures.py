"""Continuous distribution figures for Deleted-Block Scale metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot
from benchmark_design.report.vision.deleted_block_scale_stats import (
    HIGH_R_DEL_EXAMPLE_COUNT,
    compute_deleted_instance_bands,
    compute_r_del_bands,
    top_r_del_results,
)
from benchmark_design.vision.deleted_block_scale.masks import build_answer_deleted_masks
from benchmark_design.vision.deleted_block_scale.models import PageDeletedBlockScaleResult
from benchmark_design.vision.flow_structure.models import PageAnnotation
from benchmark_design.vision.masks import UnifiedPageMaskBundle

X_LABEL = "Deleted-area ratio R_del"
Y_LABEL_PAGES = "Number of affected pages"
HISTOGRAM_TITLE = "R_del Distribution among Affected Pages"
INSTANCE_HISTOGRAM_TITLE = "Deleted block count per affected page"
R_DEL_BAR_COLOR = "#4472C4"
INSTANCE_BAR_COLOR = "#4472C4"


def _require_matplotlib():
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    return plt


def _safe_filename(page_id: str) -> str:
    return page_id.replace("/", "_").replace("\\", "_")


def _affected_r_del_values(results: list[PageDeletedBlockScaleResult]) -> list[float]:
    return [
        result.r_del
        for result in results
        if result.has_deleted_text_block and result.r_del is not None
    ]


@with_locked_pyplot
def _draw_histogram(values: list[float], output_path: Path) -> bool:
    if not values:
        return False
    plt = _require_matplotlib()
    band_summary = compute_r_del_bands(values)

    range_labels = [band.range_label for band in band_summary.bands]
    counts = [band.count for band in band_summary.bands]
    ratios_pct = [band.ratio * 100.0 for band in band_summary.bands]

    fig, axis = plt.subplots(figsize=(9, 5.5))
    x_positions = range(len(range_labels))
    bars = axis.bar(
        x_positions,
        counts,
        color=R_DEL_BAR_COLOR,
        edgecolor="white",
        linewidth=0.8,
        width=0.68,
    )
    ymax = max(counts) if counts else 1
    for bar, count, ratio_pct in zip(bars, counts, ratios_pct, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{count:,}\n{ratio_pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    mean_value = float(np.mean(values))
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(range_labels)
    axis.set_xlabel(X_LABEL)
    axis.set_ylabel(Y_LABEL_PAGES)
    axis.set_title(HISTOGRAM_TITLE)
    axis.set_ylim(0, ymax * 1.18)
    axis.grid(True, axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.text(
        0.5,
        0.01,
        f"n = {band_summary.total_pages:,} affected pages; mean R_del = {mean_value:.3f}",
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


@with_locked_pyplot
def _draw_instance_histogram(
    results: list[PageDeletedBlockScaleResult],
    output_path: Path,
) -> bool:
    bands = compute_deleted_instance_bands(results)
    if not bands or not any(band.page_count for band in bands):
        return False
    plt = _require_matplotlib()

    labels = [str(band.instance_count) for band in bands]
    counts = [band.page_count for band in bands]
    ratios_pct = [band.ratio * 100.0 for band in bands]

    fig, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(
        labels,
        counts,
        color=INSTANCE_BAR_COLOR,
        edgecolor="white",
        linewidth=0.8,
        width=0.68,
    )
    ymax = max(counts) if counts else 1
    for bar, count, ratio_pct in zip(bars, counts, ratios_pct, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{count:,}\n{ratio_pct:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.set_xlabel("deleted_text_block instances per page")
    axis.set_ylabel(Y_LABEL_PAGES)
    axis.set_title(INSTANCE_HISTOGRAM_TITLE)
    axis.set_ylim(0, ymax * 1.18)
    axis.grid(True, axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


@with_locked_pyplot
def _draw_overlay(
    result: PageDeletedBlockScaleResult,
    page: PageAnnotation | None,
    *,
    input_dir: Path,
    output_path: Path,
    unified_masks: UnifiedPageMaskBundle | None = None,
) -> bool:
    plt = _require_matplotlib()
    image_path = input_dir / result.image_name
    if not image_path.is_file():
        image_path = Path(result.review_image_path)
    if not image_path.is_file():
        return False

    fig, axis = plt.subplots(figsize=(10, 14))
    image = plt.imread(str(image_path))
    axis.imshow(image)

    if page is not None:
        if unified_masks is not None:
            valid_union = unified_masks.valid_union
            deleted_union = unified_masks.deleted_union
        else:
            masks = build_answer_deleted_masks(
                page.blocks,
                image_width=page.image_width,
                image_height=page.image_height,
            )
            valid_union = masks.valid_union
            deleted_union = masks.deleted_union
        valid_overlay = np.zeros((*valid_union.shape, 4), dtype=float)
        valid_overlay[valid_union] = (0.2, 0.4, 1.0, 0.35)
        deleted_overlay = np.zeros((*deleted_union.shape, 4), dtype=float)
        deleted_overlay[deleted_union] = (1.0, 0.2, 0.2, 0.35)
        axis.imshow(valid_overlay)
        axis.imshow(deleted_overlay)

    title = (
        f"Deleted-area ratio R_del={result.r_del if result.r_del is not None else 'NA'}\n"
        f"valid_area={result.valid_area} deleted_area={result.deleted_area} "
        f"answer_related_area={result.answer_related_area}"
    )
    if result.review_reason:
        title += f"\nreview: {result.review_reason}"
    axis.set_title(title, fontsize=9)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def export_deleted_block_scale_figures(
    results: list[PageDeletedBlockScaleResult],
    pages: list[PageAnnotation],
    *,
    input_dir: Path,
    figures_root: Path,
    show_progress: bool = False,
    page_masks: dict[str, UnifiedPageMaskBundle] | None = None,
) -> dict[str, int | bool]:
    del show_progress
    page_map = {page.page_id: page for page in pages}
    values = _affected_r_del_values(results)
    counts: dict[str, int | bool] = {}

    counts["r_del_histogram.png"] = int(
        _draw_histogram(values, figures_root / "r_del_histogram.png")
    )
    counts["deleted_instance_histogram.png"] = int(
        _draw_instance_histogram(results, figures_root / "deleted_instance_histogram.png")
    )

    examples_dir = figures_root / "high_r_del_examples"
    written = 0
    for result in top_r_del_results(results, top_k=HIGH_R_DEL_EXAMPLE_COUNT):
        if _draw_overlay(
            result,
            page_map.get(result.page_id),
            input_dir=input_dir,
            output_path=examples_dir / f"{_safe_filename(result.page_id)}.png",
            unified_masks=page_masks.get(result.page_id) if page_masks else None,
        ):
            written += 1
    counts["high_r_del_examples"] = written
    return counts
