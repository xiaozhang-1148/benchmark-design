"""Dataset-level foreground calibration shared across page/block/line."""

from __future__ import annotations

from benchmark_design.foreground.models import ForegroundThresholdConfig
from benchmark_design.foreground.threshold import (
    load_foreground_threshold_config,
    save_foreground_threshold_config,
)
from benchmark_design.page_level.models import CalibrationResult


def calibration_to_threshold_config(
    calibration: CalibrationResult,
    *,
    dataset_version: str = "",
) -> ForegroundThresholdConfig:
    return ForegroundThresholdConfig(
        dataset_version=dataset_version,
        dark_reference=calibration.dark_reference,
        light_reference=calibration.light_reference,
        dark_percentile=calibration.dark_percentile,
        light_percentile=calibration.light_percentile,
        gray_threshold=calibration.gray_threshold,
        tau_d=calibration.tau_d,
        image_count=calibration.image_count,
        gray_histogram=calibration.gray_histogram,
    )
