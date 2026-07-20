"""Data models for page-level image analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AspectRatioBin:
    name: str
    min_ratio: float
    max_ratio: float


@dataclass(frozen=True, slots=True)
class PageLevelConfig:
    input_root: Path
    output_root: Path
    random_seed: int = 42
    dark_percentile: float = 1.0
    light_percentile: float = 99.5
    threshold_method: str = "global_pooled_otsu"
    aspect_ratio_groups_enabled: bool = True
    aspect_ratio_bins: tuple[AspectRatioBin, ...] = ()
    workers: int | None = None
    show_progress: bool = True


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    relative_path: str
    absolute_path: Path


@dataclass(frozen=True, slots=True)
class ImageInventoryRow:
    image_id: str
    relative_path: str
    width: int
    height: int
    aspect_ratio: float
    file_format: str
    stored_color_mode: str
    channel_count: int
    dtype: str
    bits_per_channel: int
    rgb_channels_identical: bool
    alpha_nonopaque_ratio: float
    effective_color_type: str


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    dark_reference: float
    light_reference: float
    gray_threshold: float
    tau_d: float
    dark_percentile: float
    light_percentile: float
    threshold_method: str
    image_count: int
    gray_histogram: tuple[int, ...] = field(default_factory=tuple)
    average_histogram: tuple[int, ...] = field(default_factory=tuple)
    normalized_average_histogram: tuple[int, ...] = field(default_factory=tuple)

    @property
    def global_threshold(self) -> float:
        """Backward-compatible alias for pooled Otsu gray threshold t_I."""
        return self.gray_threshold

    @property
    def foreground_valley_threshold(self) -> float:
        """Backward-compatible alias for darkness threshold tau_D."""
        return self.tau_d

    @property
    def darkness_histogram(self) -> tuple[int, ...]:
        """Backward-compatible alias; pooled histogram is on raw gray levels."""
        return self.gray_histogram


@dataclass(frozen=True, slots=True)
class ImageFeatureRow:
    image_id: str
    relative_path: str
    width: int
    height: int
    aspect_ratio: float
    file_format: str
    stored_color_mode: str
    effective_color_type: str
    bits_per_channel: int
    foreground_density: float
    aspect_ratio_group: str = ""


@dataclass(frozen=True, slots=True)
class PageLevelAnalysisResult:
    config: PageLevelConfig
    image_records: tuple[ImageRecord, ...]
    inventory: tuple[ImageInventoryRow, ...]
    calibration: CalibrationResult
    features: tuple[ImageFeatureRow, ...]
