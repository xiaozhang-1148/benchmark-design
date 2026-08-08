"""Per-image aspect ratio grouping and foreground density features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark_design.page_level.foreground import extract_foreground_mask, extract_foreground_mask_from_gray
from benchmark_design.page_level.gray_cache import PageGrayCache
from benchmark_design.page_level.models import (
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


def compute_image_features_from_arrays(
    mask,
    inventory: ImageInventoryRow,
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
    )


def compute_image_features_from_gray(
    gray: np.ndarray,
    inventory: ImageInventoryRow,
    calibration: CalibrationResult,
) -> ImageFeatureRow:
    _gray, _normalized, mask = extract_foreground_mask_from_gray(gray, calibration)
    return compute_image_features_from_arrays(mask, inventory)


def compute_image_features(
    record: ImageRecord,
    inventory: ImageInventoryRow,
    calibration: CalibrationResult,
) -> ImageFeatureRow:
    _gray, _normalized, mask = extract_foreground_mask(record, calibration)
    return compute_image_features_from_arrays(mask, inventory)


def compute_page_image_metrics(
    record: ImageRecord,
    inventory: ImageInventoryRow,
    calibration: CalibrationResult,
    config: PageLevelConfig,
    *,
    gray_cache: PageGrayCache | None = None,
) -> PageImageMetrics:
    del config  # retained for call-site compatibility
    if gray_cache is not None:
        gray = gray_cache.load(record.image_id)
        features = compute_image_features_from_gray(gray, inventory, calibration)
    else:
        features = compute_image_features(record, inventory, calibration)
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

    def _worker(record: ImageRecord) -> PageImageMetrics:
        inventory = inventory_by_id[record.image_id]
        return compute_page_image_metrics(
            record,
            inventory,
            calibration,
            config,
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
    show_progress: bool = False,
    workers: int | None = None,
) -> list[ImageFeatureRow]:
    inventory_by_id = {row.image_id: row for row in inventory_rows}

    def _worker(record: ImageRecord) -> ImageFeatureRow:
        inventory = inventory_by_id[record.image_id]
        return compute_image_features(record, inventory, calibration)

    return parallel_map(
        _worker,
        records,
        description="Computing page-level image features",
        show_progress=show_progress,
        workers=workers,
    )
