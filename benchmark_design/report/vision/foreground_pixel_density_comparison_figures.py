"""Comparison figures for dataset-threshold vs raw Otsu foreground density."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot
from benchmark_design.vision.flow_structure.models import PageAnnotation
from benchmark_design.vision.foreground_load.compute import load_grayscale_image
from benchmark_design.vision.foreground_load.models import (
    BlockForegroundLoadResult,
    GlobalForegroundLoadConfig,
    PageForegroundLoadResult,
)
from benchmark_design.vision.foreground_load.normalization import compute_darkness_from_gray
from benchmark_design.vision.foreground_load.otsu import histogram_uint8, otsu_from_histogram
from benchmark_design.vision.masks import build_unified_page_masks

HIGH_DENSITY_THRESHOLD = 0.15
COMPARISON_PANEL_TITLES: tuple[str, ...] = (
    "Original",
    "Raw Otsu foreground",
    "Dataset-threshold foreground",
    "Difference overlay",
    "Darkness S",
)


def _require_matplotlib():
    import matplotlib.pyplot as plt

    _configure_matplotlib_fonts(plt)
    return plt


def _safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def build_difference_overlay(raw_fg: np.ndarray, norm_fg: np.ndarray) -> np.ndarray:
    """black=both, red=raw only, blue=dataset-threshold only."""
    overlay = np.ones((*raw_fg.shape, 3), dtype=np.float32)
    both = raw_fg & norm_fg
    raw_only = raw_fg & ~norm_fg
    norm_only = norm_fg & ~raw_fg
    overlay[both] = (0.0, 0.0, 0.0)
    overlay[raw_only] = (1.0, 0.0, 0.0)
    overlay[norm_only] = (0.0, 0.0, 1.0)
    return overlay


def _load_page_arrays(
    page: PageAnnotation,
    *,
    input_dir: Path,
    global_config: GlobalForegroundLoadConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    image_path = input_dir / page.image_name
    if not image_path.is_file():
        return None
    gray = load_grayscale_image(image_path)
    if gray.shape != (page.image_height, page.image_width):
        try:
            import cv2
        except ImportError:
            return None
        gray = cv2.resize(
            gray,
            (page.image_width, page.image_height),
            interpolation=cv2.INTER_AREA,
        )

    darkness = compute_darkness_from_gray(
        gray,
        q_low=global_config.q_low,
        q_high=global_config.q_high,
    )
    masks = build_unified_page_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    ).foreground_masks()
    return gray, darkness, masks.effective_union, masks.txt_block_masks


@with_locked_pyplot
def _draw_comparison_sheet(
    *,
    rgb_image: np.ndarray,
    region_mask: np.ndarray,
    gray: np.ndarray,
    darkness: np.ndarray,
    global_config: GlobalForegroundLoadConfig,
    output_path: Path,
    title: str,
) -> bool:
    if not region_mask.any():
        return False

    plt = _require_matplotlib()
    page_otsu = otsu_from_histogram(histogram_uint8(gray))
    raw_fg = (gray <= page_otsu) & region_mask
    norm_fg = (darkness >= global_config.tau_D) & region_mask
    diff_overlay = build_difference_overlay(raw_fg, norm_fg)

    darkness_display = darkness.copy()
    darkness_display[~region_mask] = np.nan

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    panels = (
        (rgb_image, None),
        (raw_fg.astype(np.float32), "gray"),
        (norm_fg.astype(np.float32), "gray"),
        (diff_overlay, None),
        (darkness_display, "inferno"),
    )
    for axis, panel_title, (data, cmap) in zip(
        axes,
        COMPARISON_PANEL_TITLES,
        panels,
        strict=True,
    ):
        if cmap is None:
            axis.imshow(data)
        else:
            axis.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(panel_title, fontsize=9)
        axis.axis("off")

    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _crop_with_margin(
    array: np.ndarray,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    margin: int = 8,
) -> np.ndarray:
    height, width = array.shape[:2]
    left = max(0, x1 - margin)
    top = max(0, y1 - margin)
    right = min(width, x2 + margin)
    bottom = min(height, y2 + margin)
    return array[top:bottom, left:right]


def export_foreground_pixel_density_comparison_figures(
    results: list[PageForegroundLoadResult],
    pages: list[PageAnnotation],
    *,
    input_dir: Path,
    figures_root: Path,
    global_config: GlobalForegroundLoadConfig,
    density_threshold: float = HIGH_DENSITY_THRESHOLD,
) -> dict[str, int]:
    page_map = {page.page_id: page for page in pages}
    output_dir = figures_root / "foreground_pixel_density" / "high_density_comparisons"
    page_written = 0
    block_written = 0

    for result in results:
        if result.D_page_eff is None or result.D_page_eff <= density_threshold:
            continue
        page = page_map.get(result.page_id)
        if page is None:
            continue
        arrays = _load_page_arrays(page, input_dir=input_dir, global_config=global_config)
        if arrays is None:
            continue
        gray, darkness, R_eff, _ = arrays
        rgb = np.stack([gray, gray, gray], axis=-1)
        page_path = output_dir / f"{_safe_filename(result.page_id)}.png"
        if _draw_comparison_sheet(
            rgb_image=rgb,
            region_mask=R_eff,
            gray=gray,
            darkness=darkness,
            global_config=global_config,
            output_path=page_path,
            title=(
                f"{result.page_id} D_page={result.D_page_eff:.3f} "
                f"tau_D={global_config.tau_D:.4f}"
            ),
        ):
            page_written += 1

    for result in results:
        page = page_map.get(result.page_id)
        if page is None:
            continue
        arrays = _load_page_arrays(page, input_dir=input_dir, global_config=global_config)
        if arrays is None:
            continue
        gray, darkness, _, txt_masks = arrays
        rgb = np.stack([gray, gray, gray], axis=-1)
        for block in result.block_results:
            if block.D_block_i is None or block.D_block_i <= density_threshold:
                continue
            block_mask = txt_masks.get(block.block_id)
            if block_mask is None or not block_mask.any():
                continue
            x1 = max(0, int(block.bbox_x1))
            y1 = max(0, int(block.bbox_y1))
            x2 = min(page.image_width, int(np.ceil(block.bbox_x2)))
            y2 = min(page.image_height, int(np.ceil(block.bbox_y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            block_path = output_dir / f"{_safe_filename(result.page_id)}_{_safe_filename(block.block_id)}.png"
            if _draw_comparison_sheet(
                rgb_image=_crop_with_margin(rgb, x1=x1, y1=y1, x2=x2, y2=y2),
                region_mask=_crop_with_margin(block_mask, x1=x1, y1=y1, x2=x2, y2=y2),
                gray=_crop_with_margin(gray, x1=x1, y1=y1, x2=x2, y2=y2),
                darkness=_crop_with_margin(darkness, x1=x1, y1=y1, x2=x2, y2=y2),
                global_config=global_config,
                output_path=block_path,
                title=(
                    f"{block.block_id} D_block={block.D_block_i:.3f} "
                    f"tau_D={global_config.tau_D:.4f}"
                ),
            ):
                block_written += 1

    return {
        "high_density_page_comparisons": page_written,
        "high_density_block_comparisons": block_written,
    }
