"""Load YAML configuration for line-level analysis."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from benchmark_design.config.hmer import DEFAULT_BENCHMARK_INPUT
from benchmark_design.config.line_level import DEFAULT_LINE_LEVEL_OUTPUT
from benchmark_design.line_level.models import LineLevelConfig
from benchmark_design.page_level.models import CalibrationResult

DEFAULT_EXTERNAL_DATASET_ROOT = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/dataset"
)
DEFAULT_CALIBRATION_PATH = Path("page_level/calibration/calibration.json")


def default_line_level_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(32, max(1, cpu_count * 2))


def _optional_path(value: object | None) -> Path | None:
    if value is None or value == "" or value is False:
        return None
    return Path(str(value))


def load_line_level_config(
    config_path: Path | None = None,
    *,
    input_root: Path | None = None,
    output_root: Path | None = None,
    workers: int | None = None,
    show_progress: bool = True,
    calibration_path: Path | None = None,
    calibration: CalibrationResult | None = None,
) -> LineLevelConfig:
    payload: dict = {}
    if config_path is not None and config_path.is_file():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    resolved_input = input_root or Path(payload.get("input_root", DEFAULT_BENCHMARK_INPUT))
    resolved_output = output_root or Path(payload.get("output_root", DEFAULT_LINE_LEVEL_OUTPUT))
    max_inflight = payload.get("max_inflight_pages")
    calibration_raw = payload.get("calibration_path", DEFAULT_CALIBRATION_PATH if payload else None)
    resolved_calibration = calibration_path or _optional_path(calibration_raw)
    # External datasets only when explicitly configured (avoid scanning huge roots in unit tests).
    if "external_dataset_root" in payload:
        external_root = _optional_path(payload.get("external_dataset_root"))
    else:
        external_root = None
    external_enabled = bool(payload.get("external_dataset_aspect_enabled", "external_dataset_root" in payload))

    return LineLevelConfig(
        input_root=resolved_input,
        output_root=resolved_output,
        workers=workers if workers is not None else payload.get("workers"),
        random_seed=int(payload.get("random_seed", 42)),
        image_extensions=tuple(payload.get("image_extensions", [".jpg", ".jpeg", ".png", ".tif", ".tiff"])),
        ignore_labels=tuple(payload.get("ignore_labels", [])),
        angle_thresholds=tuple(float(v) for v in payload.get("angle_thresholds", [2, 5, 10])),
        orientation_min_aspect_ratio=float(payload.get("orientation_min_aspect_ratio", 2.0)),
        extreme_sample_count=int(payload.get("extreme_sample_count", 20)),
        height_similarity_threshold=float(payload.get("height_similarity_threshold", 0.7)),
        vertical_overlap_ratio_threshold=float(payload.get("vertical_overlap_ratio_threshold", 0.7)),
        horizontal_gap_px_threshold=float(payload.get("horizontal_gap_px_threshold", 50.0)),
        max_inflight_pages=None if max_inflight is None else int(max_inflight),
        show_progress=show_progress,
        calibration_path=resolved_calibration,
        calibration=calibration,
        bbox_outside_ink_enabled=bool(payload.get("bbox_outside_ink_enabled", True)),
        external_dataset_root=external_root,
        external_dataset_aspect_enabled=external_enabled,
    )
