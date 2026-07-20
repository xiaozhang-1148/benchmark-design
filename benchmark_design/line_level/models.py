"""Data models for line-level geometry analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_design.page_level.models import CalibrationResult


@dataclass(frozen=True, slots=True)
class LineLevelConfig:
    input_root: Path
    output_root: Path
    workers: int | None = None
    random_seed: int = 42
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")
    ignore_labels: tuple[str, ...] = ()
    angle_thresholds: tuple[float, ...] = (2.0, 5.0, 10.0)
    orientation_min_aspect_ratio: float = 2.0
    extreme_sample_count: int = 20
    height_similarity_threshold: float = 0.7
    vertical_overlap_ratio_threshold: float = 0.7
    horizontal_gap_px_threshold: float = 50.0
    max_inflight_pages: int | None = None
    show_progress: bool = True
    calibration_path: Path | None = None
    calibration: CalibrationResult | None = None
    bbox_outside_ink_enabled: bool = True
    external_dataset_root: Path | None = None
    external_dataset_aspect_enabled: bool = True


@dataclass(frozen=True, slots=True)
class LineAnnotation:
    image_id: str
    image_name: str
    line_id: str
    block_order: int
    line_order: int
    block_type: str
    polygon: tuple[tuple[float, float], ...]
    ocr: str
    source_file: str
    is_ignore: bool = False


@dataclass(frozen=True, slots=True)
class PageTask:
    image_id: str
    image_name: str
    json_path: Path
    image_path: Path | None
    width: int
    height: int
    dpi: float | None
    lines: tuple[LineAnnotation, ...]


@dataclass(frozen=True, slots=True)
class InvalidAnnotationRow:
    image_id: str
    line_id: str
    block_type: str
    reason: str
    polygon_point_count: int


@dataclass(frozen=True, slots=True)
class ProcessingErrorRow:
    image_id: str
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class LineMetricsRow:
    image_id: str
    line_id: str
    bbox_width_px: float
    bbox_height_px: float
    aspect_ratio: float
    orientation_deg: float
    orientation_direction_valid: bool
    is_ignore: bool
    is_valid: bool
    invalid_reason: str
    page_orientation: str = ""
    block_type: str = ""
    bbox_outside_ink_ratio: float | None = None
    bbox_outside_pixel_count: int = 0
    bbox_outside_ink_count: int = 0
    bbox_pixel_count: int = 0
    mask_area: int = 0
    has_interference: bool = False
    # Internal OBB fields for orientation audit only (not used in 4.1.1 size stats).
    obb_long_side_px: float = 0.0
    obb_short_side_px: float = 0.0


@dataclass(frozen=True, slots=True)
class TargetPairRow:
    image_id: str
    line_id_a: str
    line_id_b: str
    intersection_area: float
    ioa: float
    horizontal_gap_px: float
    height_similarity: float
    vertical_overlap_px: float
    vertical_overlap_ratio: float
    ioa_positive: bool
    horizontal_adjacent: bool


@dataclass(frozen=True, slots=True)
class PageMetricsRow:
    image_id: str
    width: int
    height: int
    total_pixels: int
    dpi: float | None
    line_count: int
    valid_line_count: int
    ignore_line_count: int
    ioa_positive_pair_count: int
    horizontal_adjacent_pair_count: int
    page_orientation: str
    median_bbox_height_px: float
    p05_bbox_height_px: float
    p95_bbox_height_px: float
    processing_time_ms: float
    status: str
    error_message: str = ""


@dataclass(frozen=True, slots=True)
class PageProcessResult:
    page_metrics: PageMetricsRow
    line_metrics: tuple[LineMetricsRow, ...]
    invalid_rows: tuple[InvalidAnnotationRow, ...]
    pair_rows: tuple[TargetPairRow, ...] = ()
    error: ProcessingErrorRow | None = None


@dataclass(frozen=True, slots=True)
class LineLevelAnalysisResult:
    config: LineLevelConfig
    page_metrics: tuple[PageMetricsRow, ...]
    line_metrics: tuple[LineMetricsRow, ...]
    invalid_rows: tuple[InvalidAnnotationRow, ...]
    processing_errors: tuple[ProcessingErrorRow, ...]
    pair_rows: tuple[TargetPairRow, ...] = ()
    processing_time_ms: float = 0.0
    discovered_page_count: int = 0
