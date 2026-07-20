"""Tests for pooled grayscale Otsu threshold calibration."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark_design.foreground.threshold import (
    accumulate_grayscale_histogram,
    global_pooled_otsu_gray_threshold,
    gray_threshold_to_tau_d,
)


def test_global_pooled_otsu_on_bimodal_gray_histogram() -> None:
    histogram = np.zeros(256, dtype=np.int64)
    histogram[240:256] = 1_000_000
    histogram[0:40] = 50_000
    t_i = global_pooled_otsu_gray_threshold(histogram)
    assert 0.0 <= t_i <= 255.0


def test_gray_threshold_to_tau_d_conversion() -> None:
    tau_d = gray_threshold_to_tau_d(128.0, dark_reference=0.0, light_reference=255.0)
    assert tau_d == pytest.approx(1.0 - 128.0 / 255.0)


def test_accumulate_grayscale_histogram_uses_bincount() -> None:
    gray = np.array([[0, 10], [10, 255]], dtype=np.uint8)
    histogram = accumulate_grayscale_histogram(gray)
    assert histogram[0] == 1
    assert histogram[10] == 2
    assert histogram[255] == 1
