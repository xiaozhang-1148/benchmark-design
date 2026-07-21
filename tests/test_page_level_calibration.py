"""Tests for page-level equal-image-weighted calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmark_design.page_level.calibration import (
    average_equal_image_histograms,
    compute_calibration,
    density_gray_histogram,
    normalize_gray_to_uint8,
    percentile_from_density_histogram,
)
from benchmark_design.page_level.models import ImageRecord, PageLevelConfig


def _write_gray_image(path: Path, value: int, *, size: tuple[int, int] = (8, 8)) -> None:
    Image.fromarray(np.full(size, value, dtype=np.uint8), mode="L").save(path)


def test_density_gray_histogram_normalizes_by_area() -> None:
    gray = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    hist = density_gray_histogram(gray)
    assert hist.sum() == pytest.approx(1.0)
    assert hist[0] == pytest.approx(0.5)
    assert hist[255] == pytest.approx(0.5)


def test_average_equal_image_histograms_uses_equal_image_weight() -> None:
    dark = density_gray_histogram(np.zeros((4, 4), dtype=np.uint8))
    bright = density_gray_histogram(np.full((2, 2), 255, dtype=np.uint8))
    average = average_equal_image_histograms([dark, bright])
    assert average[0] == pytest.approx(0.5)
    assert average[255] == pytest.approx(0.5)


def test_percentile_from_density_histogram() -> None:
    hist = np.zeros(256, dtype=np.float64)
    hist[10] = 0.01
    hist[200] = 0.99
    assert percentile_from_density_histogram(hist, 1.0) == pytest.approx(10.0)
    assert percentile_from_density_histogram(hist, 99.5) == pytest.approx(200.0)


def test_normalize_gray_to_uint8_clips_to_range() -> None:
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    normalized = normalize_gray_to_uint8(gray, dark_reference=0.0, light_reference=255.0)
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_compute_calibration_on_two_images(tmp_path: Path) -> None:
    dark_path = tmp_path / "dark.png"
    bright_path = tmp_path / "bright.png"
    _write_gray_image(dark_path, 20)
    _write_gray_image(bright_path, 230)
    records = [
        ImageRecord("dark", "dark.png", dark_path),
        ImageRecord("bright", "bright.png", bright_path),
    ]
    config = PageLevelConfig(
        input_root=tmp_path,
        output_root=tmp_path / "out",
        show_progress=False,
        workers=1,
    )
    calibration = compute_calibration(records, config)
    assert calibration.image_count == 2
    assert calibration.dark_reference < calibration.light_reference
    assert 0.0 <= calibration.gray_threshold <= 255.0
    assert 0.0 <= calibration.tau_d <= 1.0
    assert len(calibration.gray_histogram) == 256
    assert calibration.threshold_method == "global_pooled_otsu"


def test_compute_calibration_empty_histogram_fallback() -> None:
    calibration_hist = average_equal_image_histograms([])
    assert calibration_hist.sum() == pytest.approx(0.0)
    assert percentile_from_density_histogram(calibration_hist, 50.0) == pytest.approx(0.0)
