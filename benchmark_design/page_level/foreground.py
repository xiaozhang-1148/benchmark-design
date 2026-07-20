"""Full-page foreground mask generation using shared gray threshold."""

from __future__ import annotations

import numpy as np

from benchmark_design.foreground.mask import foreground_mask_from_gray_threshold
from benchmark_design.foreground.normalize import compute_g_tilde
from benchmark_design.io.image import load_grayscale_image
from benchmark_design.page_level.models import CalibrationResult, ImageRecord


def extract_foreground_mask(
    record: ImageRecord,
    calibration: CalibrationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = load_grayscale_image(record.absolute_path)
    return extract_foreground_mask_from_gray(gray, calibration)


def extract_foreground_mask_from_gray(
    gray: np.ndarray,
    calibration: CalibrationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_tilde = compute_g_tilde(
        gray,
        dark_reference=calibration.dark_reference,
        light_reference=calibration.light_reference,
    )
    mask = foreground_mask_from_gray_threshold(gray, calibration.gray_threshold)
    return gray, g_tilde, mask


def extract_block_foreground_mask_from_gray(
    gray: np.ndarray,
    calibration: CalibrationResult,
) -> np.ndarray:
    """Block-level foreground mask using the same page-level gray threshold."""
    return foreground_mask_from_gray_threshold(gray, calibration.gray_threshold)
