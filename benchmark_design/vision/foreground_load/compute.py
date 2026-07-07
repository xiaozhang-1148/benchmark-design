"""Per-page foreground load computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark_design.vision.flow_structure.geometry import polygon_bbox
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.vision.flow_structure.thresholds import is_txt_block
from benchmark_design.vision.foreground_load.models import (
    BlockForegroundLoadResult,
    GlobalForegroundLoadConfig,
    PageForegroundLoadResult,
)
from benchmark_design.vision.foreground_load.normalization import (
    DARKNESS_BINS,
    compute_darkness_from_gray,
    darkness_histogram_in_mask,
)
from benchmark_design.vision.foreground_load.otsu import (
    foreground_count,
    foreground_count_darkness,
    histogram_uint8,
    ink_mass,
    otsu_from_histogram,
)
from benchmark_design.vision.foreground_load.raster import polygon_out_of_bounds
from benchmark_design.vision.foreground_load.thresholds import (
    D_REVIEW_HIGH,
    D_REVIEW_LOW,
    MIN_MASK_PIXELS,
    count_blocks_by_type,
)
from benchmark_design.vision.masks import UnifiedPageMaskBundle, build_unified_page_masks
from benchmark_design.vision.processing import _resolve_image_path


@dataclass(frozen=True, slots=True)
class PageDarknessHistogram:
    page_id: str
    histogram: np.ndarray


def load_grayscale_image(image_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            return gray.astype(np.uint8, copy=False)

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for foreground load metrics. Install with: pip install Pillow"
        ) from exc
    with Image.open(image_path) as image:
        return np.array(image.convert("L"), dtype=np.uint8)


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _density_review_reasons(density: float | None) -> list[str]:
    if density is None:
        return []
    reasons: list[str] = []
    if density <= D_REVIEW_LOW:
        reasons.append("density_saturated_low")
    elif density >= D_REVIEW_HIGH:
        reasons.append("density_saturated_high")
    return reasons


def _block_bbox(block: PageBlockAnnotation) -> tuple[float, float, float, float]:
    if len(block.polygon) < 3:
        return 0.0, 0.0, 0.0, 0.0
    return polygon_bbox(block.polygon)


def _density_at_threshold(darkness: np.ndarray, mask: np.ndarray, threshold: float) -> float | None:
    area = int(mask.sum())
    if area == 0:
        return None
    foreground = foreground_count_darkness(darkness, mask, threshold)
    return foreground / area


def _empty_block_result(
    block: PageBlockAnnotation,
    *,
    reasons: list[str],
    global_config: GlobalForegroundLoadConfig | None = None,
    block_mask_area: int = 0,
    F_i: int = 0,
    D_block_i: float | None = None,
    mean_darkness: float | None = None,
    raw_otsu_density: float | None = None,
    block_otsu: float | None = None,
    D_block_tau_minus: float | None = None,
    D_block_tau_plus: float | None = None,
) -> BlockForegroundLoadResult:
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _block_bbox(block)
    return BlockForegroundLoadResult(
        page_id=block.page_id,
        block_id=block.block_id,
        block_order=block.block_order,
        block_type=block.block_type,
        bbox_x1=bbox_x1,
        bbox_y1=bbox_y1,
        bbox_x2=bbox_x2,
        bbox_y2=bbox_y2,
        D_block_i=D_block_i,
        mean_darkness=mean_darkness,
        raw_otsu_density=raw_otsu_density,
        D_block_tau_minus=D_block_tau_minus,
        D_block_tau_plus=D_block_tau_plus,
        foreground_load_level="",
        relative_load_tertile="",
        foreground_load_tags="",
        block_otsu_threshold=block_otsu,
        threshold_dataset=global_config.tau_D if global_config is not None else None,
        block_mask_area=block_mask_area,
        F_i=F_i,
        needs_manual_review=bool(reasons),
        review_reason=";".join(reasons),
    )


def _compute_block_result(
    block: PageBlockAnnotation,
    gray: np.ndarray,
    block_mask: np.ndarray,
    darkness: np.ndarray,
    *,
    page_out_of_bounds: bool,
    global_config: GlobalForegroundLoadConfig,
) -> BlockForegroundLoadResult:
    reasons: list[str] = []
    if page_out_of_bounds:
        _append_reason(reasons, "mask_out_of_bounds")
    if polygon_out_of_bounds(
        block.polygon,
        image_width=gray.shape[1],
        image_height=gray.shape[0],
    ):
        _append_reason(reasons, "mask_out_of_bounds")

    block_mask_area = int(block_mask.sum())
    if block_mask_area == 0:
        _append_reason(reasons, "empty_effective_mask")
        return _empty_block_result(block, reasons=reasons, global_config=global_config)

    pixels = gray[block_mask]
    if pixels.size < MIN_MASK_PIXELS:
        _append_reason(reasons, "insufficient_mask_pixels")
        return _empty_block_result(
            block,
            reasons=reasons,
            global_config=global_config,
            block_mask_area=block_mask_area,
        )

    block_otsu = otsu_from_histogram(histogram_uint8(pixels))
    raw_F_i = foreground_count(pixels, block_otsu)
    raw_otsu_density = raw_F_i / block_mask_area

    tau = global_config.tau_D
    delta = global_config.sensitivity_delta
    F_i = foreground_count_darkness(darkness, block_mask, tau)
    D_block_i = F_i / block_mask_area
    mean_darkness = ink_mass(darkness, block_mask)
    D_block_tau_minus = _density_at_threshold(darkness, block_mask, tau - delta)
    D_block_tau_plus = _density_at_threshold(darkness, block_mask, tau + delta)
    reasons.extend(_density_review_reasons(D_block_i))
    return _empty_block_result(
        block,
        reasons=reasons,
        global_config=global_config,
        block_mask_area=block_mask_area,
        F_i=F_i,
        D_block_i=D_block_i,
        mean_darkness=mean_darkness,
        raw_otsu_density=raw_otsu_density,
        block_otsu=block_otsu,
        D_block_tau_minus=D_block_tau_minus,
        D_block_tau_plus=D_block_tau_plus,
    )


def _page_block_counts(page: PageAnnotation) -> tuple[int, int, int, int]:
    return count_blocks_by_type(page.blocks)


def _resize_gray_to_page(gray: np.ndarray, page: PageAnnotation) -> np.ndarray:
    if gray.shape == (page.image_height, page.image_width):
        return gray
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        return cv2.resize(
            gray,
            (page.image_width, page.image_height),
            interpolation=cv2.INTER_AREA,
        )
    from PIL import Image

    return np.array(
        Image.fromarray(gray).resize((page.image_width, page.image_height)),
        dtype=np.uint8,
    )


def _resolve_masks(
    page: PageAnnotation,
    unified_masks: UnifiedPageMaskBundle | None,
) -> UnifiedPageMaskBundle:
    if unified_masks is not None:
        return unified_masks
    return build_unified_page_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    )


def _compute_darkness_for_page(gray: np.ndarray, global_config: GlobalForegroundLoadConfig) -> np.ndarray:
    return compute_darkness_from_gray(gray, q_low=global_config.q_low, q_high=global_config.q_high)


def collect_page_darkness_histogram(
    page: PageAnnotation,
    *,
    input_dir: Path | None = None,
    unified_masks: UnifiedPageMaskBundle | None = None,
    gray: np.ndarray | None = None,
    bins: int = DARKNESS_BINS,
    global_config: GlobalForegroundLoadConfig | None = None,
) -> PageDarknessHistogram | None:
    image_dir = input_dir or Path(page.source_file).parent
    image_path = _resolve_image_path(page.image_name, image_dir)
    if image_path is None:
        image_path = image_dir / page.image_name

    if gray is None:
        if not image_path.is_file():
            return None
        try:
            gray = _resize_gray_to_page(load_grayscale_image(image_path), page)
        except OSError:
            return None
    elif gray.shape != (page.image_height, page.image_width):
        gray = _resize_gray_to_page(gray, page)

    masks = _resolve_masks(page, unified_masks).foreground_masks()
    R_eff = masks.effective_union
    if not R_eff.any():
        return PageDarknessHistogram(page_id=page.page_id, histogram=np.zeros(bins, dtype=np.int64))

    config = global_config or GlobalForegroundLoadConfig(
        tau_D=0.5,
        threshold_method="pooled_otsu",
    )
    darkness = _compute_darkness_for_page(gray, config)
    hist = darkness_histogram_in_mask(darkness, R_eff, bins=bins)
    return PageDarknessHistogram(page_id=page.page_id, histogram=hist)


def compute_page_foreground_load(
    page: PageAnnotation,
    *,
    input_dir: Path | None = None,
    unified_masks: UnifiedPageMaskBundle | None = None,
    gray: np.ndarray | None = None,
    global_config: GlobalForegroundLoadConfig | None = None,
) -> PageForegroundLoadResult:
    image_dir = input_dir or Path(page.source_file).parent
    review_path = str(image_dir / page.image_name)
    image_path = _resolve_image_path(page.image_name, image_dir)
    if image_path is None:
        image_path = image_dir / page.image_name
    page_area = page.image_width * page.image_height
    num_txt, num_figure, num_chart, num_deleted = _page_block_counts(page)

    if global_config is None:
        raise ValueError("global_config is required for foreground load computation")

    if gray is None:
        if not image_path.is_file():
            return PageForegroundLoadResult(
                page_id=page.page_id,
                image_name=page.image_name,
                image_width=page.image_width,
                image_height=page.image_height,
                page_area=page_area,
                effective_region_area_ratio=None,
                num_txtBlock=num_txt,
                num_figure=num_figure,
                num_chart=num_chart,
                num_deleted_text_block=num_deleted,
                D_page_eff=None,
                mean_darkness=None,
                raw_otsu_density=None,
                D_page_tau_minus=None,
                D_page_tau_plus=None,
                foreground_load_level="",
                relative_load_tertile="",
                foreground_load_tags="",
                page_otsu_threshold=None,
                threshold_dataset=global_config.tau_D,
                R_eff_area=0,
                F_eff=0,
                num_effective_blocks=0,
                needs_manual_review=True,
                review_reason="missing_image",
                review_image_path=review_path,
                block_results=(),
            )
        try:
            gray = _resize_gray_to_page(load_grayscale_image(image_path), page)
        except OSError:
            return PageForegroundLoadResult(
                page_id=page.page_id,
                image_name=page.image_name,
                image_width=page.image_width,
                image_height=page.image_height,
                page_area=page_area,
                effective_region_area_ratio=None,
                num_txtBlock=num_txt,
                num_figure=num_figure,
                num_chart=num_chart,
                num_deleted_text_block=num_deleted,
                D_page_eff=None,
                mean_darkness=None,
                raw_otsu_density=None,
                D_page_tau_minus=None,
                D_page_tau_plus=None,
                foreground_load_level="",
                relative_load_tertile="",
                foreground_load_tags="",
                page_otsu_threshold=None,
                threshold_dataset=global_config.tau_D,
                R_eff_area=0,
                F_eff=0,
                num_effective_blocks=0,
                needs_manual_review=True,
                review_reason="missing_image",
                review_image_path=review_path,
                block_results=(),
            )
    elif gray.shape != (page.image_height, page.image_width):
        gray = _resize_gray_to_page(gray, page)

    page_otsu = otsu_from_histogram(histogram_uint8(gray))
    masks = _resolve_masks(page, unified_masks).foreground_masks()
    R_eff = masks.effective_union
    R_eff_area = int(R_eff.sum())
    effective_region_area_ratio = (R_eff_area / page_area) if page_area > 0 else None
    reasons: list[str] = []
    if masks.out_of_bounds:
        _append_reason(reasons, "mask_out_of_bounds")

    darkness = _compute_darkness_for_page(gray, global_config)
    D_page_eff: float | None = None
    raw_otsu_density: float | None = None
    mean_darkness: float | None = None
    D_page_tau_minus: float | None = None
    D_page_tau_plus: float | None = None
    F_eff = 0
    tau = global_config.tau_D
    delta = global_config.sensitivity_delta
    if R_eff_area == 0:
        _append_reason(reasons, "empty_effective_mask")
    else:
        F_eff = foreground_count_darkness(darkness, R_eff, tau)
        D_page_eff = F_eff / R_eff_area
        mean_darkness = ink_mass(darkness, R_eff)
        D_page_tau_minus = _density_at_threshold(darkness, R_eff, tau - delta)
        D_page_tau_plus = _density_at_threshold(darkness, R_eff, tau + delta)
        raw_foreground = gray <= page_otsu
        raw_F_eff = int(np.count_nonzero(raw_foreground & R_eff))
        raw_otsu_density = raw_F_eff / R_eff_area
        reasons.extend(_density_review_reasons(D_page_eff))

    block_by_id = {block.block_id: block for block in page.blocks if is_txt_block(block.block_type)}
    block_results = tuple(
        _compute_block_result(
            block_by_id[block_id],
            gray,
            block_mask,
            darkness,
            page_out_of_bounds=masks.out_of_bounds,
            global_config=global_config,
        )
        for block_id, block_mask in masks.txt_block_masks.items()
        if block_id in block_by_id
    )

    return PageForegroundLoadResult(
        page_id=page.page_id,
        image_name=page.image_name,
        image_width=page.image_width,
        image_height=page.image_height,
        page_area=page_area,
        effective_region_area_ratio=effective_region_area_ratio,
        num_txtBlock=num_txt,
        num_figure=num_figure,
        num_chart=num_chart,
        num_deleted_text_block=num_deleted,
        D_page_eff=D_page_eff,
        mean_darkness=mean_darkness,
        raw_otsu_density=raw_otsu_density,
        D_page_tau_minus=D_page_tau_minus,
        D_page_tau_plus=D_page_tau_plus,
        foreground_load_level="",
        relative_load_tertile="",
        foreground_load_tags="",
        page_otsu_threshold=page_otsu,
        threshold_dataset=global_config.tau_D,
        R_eff_area=R_eff_area,
        F_eff=F_eff,
        num_effective_blocks=masks.num_effective_blocks,
        needs_manual_review=bool(reasons),
        review_reason=";".join(reasons),
        review_image_path=review_path,
        block_results=block_results,
    )
