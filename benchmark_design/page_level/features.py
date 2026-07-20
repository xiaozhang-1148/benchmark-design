"""Per-image aspect ratio grouping and foreground density features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark_design.page_level.foreground import extract_foreground_mask, extract_foreground_mask_from_gray
from benchmark_design.page_level.gray_cache import PageGrayCache
from benchmark_design.page_level.models import (
    AspectRatioBin,
    CalibrationResult,
    ImageFeatureRow,
    ImageInventoryRow,
    ImageRecord,
    PageLevelConfig,
)
from benchmark_design.progress import parallel_map


@dataclass(frozen=True, slots=True)
class PageImageMetrics:
    features: ImageFeatureRow


def assign_aspect_ratio_group(aspect_ratio: float, bins: tuple[AspectRatioBin, ...]) -> str:
    if not bins:
        return "all"
    for item in bins:
        if item.min_ratio <= aspect_ratio < item.max_ratio:
            return item.name
    return "other"


def compute_image_features_from_arrays(
    mask,
    inventory: ImageInventoryRow,
    *,
    aspect_ratio_bins: tuple[AspectRatioBin, ...] = (),
) -> ImageFeatureRow:
    foreground_density = float(mask.sum() / mask.size) if mask.size else 0.0
    return ImageFeatureRow(
        image_id=inventory.image_id,
        relative_path=inventory.relative_path,
        width=inventory.width,
        height=inventory.height,
        aspect_ratio=inventory.aspect_ratio,
        file_format=inventory.file_format,
        stored_color_mode=inventory.stored_color_mode,
        effective_color_type=inventory.effective_color_type,
        bits_per_channel=inventory.bits_per_channel,
        foreground_density=foreground_density,
        aspect_ratio_group=assign_aspect_ratio_group(inventory.aspect_ratio, aspect_ratio_bins),
    )


def compute_image_features_from_gray(
    gray: np.ndarray,
    inventory: ImageInventoryRow,
    calibration: CalibrationResult,
    *,
    aspect_ratio_bins: tuple[AspectRatioBin, ...] = (),
) -> ImageFeatureRow:
    _gray, _normalized, mask = extract_foreground_mask_from_gray(gray, calibration)
    return compute_image_features_from_arrays(
        mask,
        inventory,
        aspect_ratio_bins=aspect_ratio_bins,
    )


def compute_image_features(
    record: ImageRecord,
    inventory: ImageInventoryRow,
    calibration: CalibrationResult,
    *,
    aspect_ratio_bins: tuple[AspectRatioBin, ...] = (),
) -> ImageFeatureRow:
    _gray, _normalized, mask = extract_foreground_mask(record, calibration)
    return compute_image_features_from_arrays(
        mask,
        inventory,
        aspect_ratio_bins=aspect_ratio_bins,
    )


def compute_page_image_metrics(
    record: ImageRecord,
    inventory: ImageInventoryRow,
    calibration: CalibrationResult,
    config: PageLevelConfig,
    *,
    aspect_ratio_bins: tuple[AspectRatioBin, ...] = (),
    gray_cache: PageGrayCache | None = None,
) -> PageImageMetrics:
    del config  # retained for call-site compatibility
    if gray_cache is not None:
        gray = gray_cache.load(record.image_id)
        features = compute_image_features_from_gray(
            gray,
            inventory,
            calibration,
            aspect_ratio_bins=aspect_ratio_bins,
        )
    else:
        features = compute_image_features(
            record,
            inventory,
            calibration,
            aspect_ratio_bins=aspect_ratio_bins,
        )
    return PageImageMetrics(features=features)


def compute_all_page_image_metrics(
    records: list[ImageRecord],
    inventory_rows: list[ImageInventoryRow],
    calibration: CalibrationResult,
    config: PageLevelConfig,
    *,
    show_progress: bool = False,
    workers: int | None = None,
    gray_cache: PageGrayCache | None = None,
) -> list[PageImageMetrics]:
    inventory_by_id = {row.image_id: row for row in inventory_rows}
    aspect_ratio_bins = config.aspect_ratio_bins if config.aspect_ratio_groups_enabled else ()

    def _worker(record: ImageRecord) -> PageImageMetrics:
        inventory = inventory_by_id[record.image_id]
        return compute_page_image_metrics(
            record,
            inventory,
            calibration,
            config,
            aspect_ratio_bins=aspect_ratio_bins,
            gray_cache=gray_cache,
        )

    return parallel_map(
        _worker,
        records,
        description="Computing page-level image features",
        show_progress=show_progress,
        workers=workers,
    )


def compute_all_image_features(
    records: list[ImageRecord],
    inventory_rows: list[ImageInventoryRow],
    calibration: CalibrationResult,
    *,
    aspect_ratio_bins: tuple[AspectRatioBin, ...] = (),
    show_progress: bool = False,
    workers: int | None = None,
) -> list[ImageFeatureRow]:
    inventory_by_id = {row.image_id: row for row in inventory_rows}

    def _worker(record: ImageRecord) -> ImageFeatureRow:
        inventory = inventory_by_id[record.image_id]
        return compute_image_features(
            record,
            inventory,
            calibration,
            aspect_ratio_bins=aspect_ratio_bins,
        )

    return parallel_map(
        _worker,
        records,
        description="Computing page-level image features",
        show_progress=show_progress,
        workers=workers,
    )
