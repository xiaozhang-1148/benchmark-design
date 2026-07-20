"""Binary foreground masks from shared grayscale threshold."""

from __future__ import annotations

import numpy as np

from benchmark_design.foreground.models import ForegroundThresholdConfig
from benchmark_design.foreground.normalize import compute_darkness


def foreground_mask_from_gray_threshold(gray: np.ndarray, gray_threshold: float) -> np.ndarray:
    """Return F where foreground pixels satisfy I <= t_I."""
    return gray.astype(np.uint8) <= gray_threshold


def foreground_mask_from_darkness(darkness: np.ndarray, tau_d: float) -> np.ndarray:
    """Return F where foreground pixels satisfy S >= tau_D."""
    return darkness >= tau_d


def foreground_mask_from_gray(
    gray: np.ndarray,
    config: ForegroundThresholdConfig,
) -> np.ndarray:
    """Build page-level foreground mask using frozen gray threshold."""
    return foreground_mask_from_gray_threshold(gray, config.gray_threshold)


def foreground_mask_from_gray_calibration(
    gray: np.ndarray,
    *,
    gray_threshold: float,
    dark_reference: float,
    light_reference: float,
    tau_d: float | None = None,
) -> np.ndarray:
    """Build foreground mask; prefers direct gray rule over darkness conversion."""
    del dark_reference, light_reference, tau_d
    return foreground_mask_from_gray_threshold(gray, gray_threshold)
