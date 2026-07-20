"""Shared foreground threshold configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ForegroundThresholdConfig:
    """Dataset-level foreground threshold metadata shared by page/block/line."""

    dataset_version: str
    dark_reference: float
    light_reference: float
    dark_percentile: float
    light_percentile: float
    gray_threshold: float
    tau_d: float
    threshold_method: str = "global_pooled_otsu"
    foreground_rule: str = "gray <= gray_threshold"
    quantization: str = "uint8"
    histogram_weighting: str = "pooled_pixels"
    morphology: str = "none"
    image_count: int = 0
    gray_histogram: tuple[int, ...] = field(default_factory=tuple)

    @property
    def darkness_threshold(self) -> float:
        """Equivalent darkness cutoff: S >= tau_d."""
        return self.tau_d

    @property
    def g_tilde_cutoff(self) -> float:
        return 1.0 - self.tau_d
