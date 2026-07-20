"""Unified foreground normalization, thresholding, and mask generation."""

from benchmark_design.foreground.mask import foreground_mask_from_darkness, foreground_mask_from_gray
from benchmark_design.foreground.models import ForegroundThresholdConfig
from benchmark_design.foreground.normalize import compute_darkness, compute_g_tilde
from benchmark_design.foreground.threshold import (
    load_foreground_threshold_config,
    save_foreground_threshold_config,
)

__all__ = [
    "ForegroundThresholdConfig",
    "compute_darkness",
    "compute_g_tilde",
    "foreground_mask_from_darkness",
    "foreground_mask_from_gray",
    "load_foreground_threshold_config",
    "save_foreground_threshold_config",
]
