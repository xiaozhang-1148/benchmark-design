"""Data models for effective-region foreground load."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ForegroundLoadLevel = Literal["low", "medium", "high", ""]
RelativeLoadTertile = Literal["lower", "middle", "upper", ""]
RegionType = Literal["page", "block"]


@dataclass(frozen=True, slots=True)
class GlobalForegroundLoadConfig:
    tau_D: float
    threshold_method: str
    q_low: float = 1.0
    q_high: float = 99.0
    darkness_bins: int = 256
    sensitivity_delta: float = 0.03
    calibration_histogram: tuple[int, ...] = field(default_factory=tuple)

    @property
    def T_global(self) -> float:
        return self.tau_D


@dataclass(frozen=True, slots=True)
class BlockForegroundLoadResult:
    page_id: str
    block_id: str
    block_order: int
    block_type: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    D_block_i: float | None
    mean_darkness: float | None
    raw_otsu_density: float | None
    D_block_tau_minus: float | None
    D_block_tau_plus: float | None
    foreground_load_level: ForegroundLoadLevel
    relative_load_tertile: RelativeLoadTertile
    foreground_load_tags: str
    block_otsu_threshold: float | None
    threshold_dataset: float | None
    block_mask_area: int
    F_i: int
    needs_manual_review: bool
    review_reason: str

    @property
    def image_id(self) -> str:
        return self.page_id

    @property
    def region_id(self) -> str:
        return self.block_id

    @property
    def region_type(self) -> RegionType:
        return "block"

    @property
    def mask_area(self) -> int:
        return self.block_mask_area

    @property
    def foreground_pixels(self) -> int:
        return self.F_i

    @property
    def foreground_density(self) -> float | None:
        return self.D_block_i

    @property
    def raw_otsu_threshold(self) -> float | None:
        return self.block_otsu_threshold

    @property
    def T_global(self) -> float | None:
        return self.threshold_dataset

    @property
    def ink_mass(self) -> float | None:
        return self.mean_darkness


@dataclass(frozen=True, slots=True)
class PageForegroundLoadResult:
    page_id: str
    image_name: str
    image_width: int
    image_height: int
    page_area: int
    effective_region_area_ratio: float | None
    num_txtBlock: int
    num_figure: int
    num_chart: int
    num_deleted_text_block: int
    D_page_eff: float | None
    mean_darkness: float | None
    raw_otsu_density: float | None
    D_page_tau_minus: float | None
    D_page_tau_plus: float | None
    foreground_load_level: ForegroundLoadLevel
    relative_load_tertile: RelativeLoadTertile
    foreground_load_tags: str
    page_otsu_threshold: float | None
    threshold_dataset: float | None
    R_eff_area: int
    F_eff: int
    num_effective_blocks: int
    needs_manual_review: bool
    review_reason: str
    review_image_path: str
    block_results: tuple[BlockForegroundLoadResult, ...] = field(default_factory=tuple)

    @property
    def image_id(self) -> str:
        return self.page_id

    @property
    def region_id(self) -> str:
        return self.page_id

    @property
    def region_type(self) -> RegionType:
        return "page"

    @property
    def mask_area(self) -> int:
        return self.R_eff_area

    @property
    def foreground_pixels(self) -> int:
        return self.F_eff

    @property
    def foreground_density(self) -> float | None:
        return self.D_page_eff

    @property
    def raw_otsu_threshold(self) -> float | None:
        return self.page_otsu_threshold

    @property
    def T_global(self) -> float | None:
        return self.threshold_dataset

    @property
    def ink_mass(self) -> float | None:
        return self.mean_darkness


@dataclass(frozen=True, slots=True)
class ForegroundLoadThresholds:
    absolute_low_medium: float
    absolute_medium_high: float
    absolute_very_high: float
    page_p33: float
    page_p66: float
    block_p33: float
    block_p66: float
    review_low: float
    review_high: float
    tau_D: float | None = None
    threshold_method: str | None = None
    q_low: float | None = None
    q_high: float | None = None

    @property
    def T_global(self) -> float | None:
        return self.tau_D
