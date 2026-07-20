"""Block-level foreground density over block annotation polygons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from benchmark_design.block_level.flow_structure.geometry import polygon_bbox
from benchmark_design.block_level.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.io.image import load_grayscale_image
from benchmark_design.page_level.foreground import extract_block_foreground_mask_from_gray
from benchmark_design.page_level.gray_cache import PageGrayCache
from benchmark_design.page_level.models import CalibrationResult
from benchmark_design.progress import parallel_map, resolve_workers


@dataclass(frozen=True, slots=True)
class BlockForegroundDensityRow:
    page_id: str
    block_id: str
    block_type: str
    block_order: int
    foreground_density: float
    annotation_pixel_count: int
    foreground_pixel_count: int


def _polygon_bbox_slice(
    polygon: tuple[tuple[float, float], ...],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if len(polygon) < 3 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = polygon_bbox(polygon)
    left = max(int(np.floor(x1)), 0)
    top = max(int(np.floor(y1)), 0)
    right = min(int(np.ceil(x2)) + 1, width)
    bottom = min(int(np.ceil(y2)) + 1, height)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _rasterize_polygon_bbox_mask(
    polygon: tuple[tuple[float, float], ...],
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> np.ndarray:
    roi_h = bottom - top
    roi_w = right - left
    yy, xx = np.mgrid[top:bottom, left:right]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = MplPath(polygon).contains_points(points).reshape(roi_h, roi_w)
    return inside


def rasterize_polygon_mask(
    polygon: tuple[tuple[float, float], ...],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    bbox = _polygon_bbox_slice(polygon, width=width, height=height)
    if bbox is None:
        return np.zeros((height, width), dtype=bool)
    left, top, right, bottom = bbox
    inside = _rasterize_polygon_bbox_mask(
        polygon,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = inside
    return mask


def compute_block_foreground_density_from_masks(
    block: PageBlockAnnotation,
    *,
    image_width: int,
    image_height: int,
    foreground_mask: np.ndarray,
) -> BlockForegroundDensityRow | None:
    polygon = block.polygon
    if len(polygon) < 3:
        return None
    if foreground_mask.shape != (image_height, image_width):
        return None
    annotation_mask = rasterize_polygon_mask(
        polygon,
        width=image_width,
        height=image_height,
    )
    annotation_pixel_count = int(annotation_mask.sum())
    if annotation_pixel_count == 0:
        return None
    foreground_pixel_count = int((foreground_mask & annotation_mask).sum())
    density = foreground_pixel_count / annotation_pixel_count
    return BlockForegroundDensityRow(
        page_id=block.page_id,
        block_id=block.block_id,
        block_type=block.block_type,
        block_order=block.block_order,
        foreground_density=density,
        annotation_pixel_count=annotation_pixel_count,
        foreground_pixel_count=foreground_pixel_count,
    )


def _page_image_cache_id(page: PageAnnotation) -> str:
    return Path(page.image_name).stem


def _load_page_gray(
    page: PageAnnotation,
    *,
    input_dir: Path,
    gray_cache: PageGrayCache | None,
) -> np.ndarray | None:
    image_cache_id = _page_image_cache_id(page)
    if gray_cache is not None:
        cached = gray_cache.try_load(image_cache_id)
        if cached is not None:
            return cached
    image_path = input_dir / page.image_name
    if not image_path.is_file():
        return None
    return load_grayscale_image(image_path)


def _is_txtblock(block: PageBlockAnnotation) -> bool:
    return block.block_type.lower() in {"txtblock", "txt_block"}


def _compute_page_block_densities(args: tuple) -> list[BlockForegroundDensityRow]:
    page, input_dir, calibration, gray_cache = args
    if not page.blocks:
        return []
    gray = _load_page_gray(page, input_dir=input_dir, gray_cache=gray_cache)
    if gray is None:
        return []
    if gray.shape[0] != page.image_height or gray.shape[1] != page.image_width:
        return []
    foreground_mask = extract_block_foreground_mask_from_gray(gray, calibration)
    rows: list[BlockForegroundDensityRow] = []
    for block in page.blocks:
        if not _is_txtblock(block):
            continue
        row = compute_block_foreground_density_from_masks(
            block,
            image_width=page.image_width,
            image_height=page.image_height,
            foreground_mask=foreground_mask,
        )
        if row is not None:
            rows.append(row)
    return rows


def _default_block_density_workers(workers: int | None) -> int:
    return resolve_workers(workers)


def compute_block_foreground_densities(
    pages: list[PageAnnotation],
    *,
    input_dir: Path,
    calibration: CalibrationResult,
    show_progress: bool = False,
    workers: int | None = None,
    chunk_size: int = 8,
    gray_cache_root: Path | None = None,
) -> list[BlockForegroundDensityRow]:
    """Compute per-block density with page-independent batch parallelism."""
    gray_cache = PageGrayCache(gray_cache_root) if gray_cache_root is not None else None
    page_tasks = [
        (page, input_dir, calibration, gray_cache) for page in pages if page.blocks
    ]
    nested = parallel_map(
        _compute_page_block_densities,
        page_tasks,
        description="Computing block foreground density",
        show_progress=show_progress,
        workers=_default_block_density_workers(workers),
        chunk_size=chunk_size,
        executor="thread",
    )
    return [row for page_rows in nested for row in page_rows]
