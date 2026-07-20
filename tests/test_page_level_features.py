"""Tests for page-level foreground density features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.page_level.features import (
    compute_image_features,
    compute_image_features_from_gray,
    compute_page_image_metrics,
)
from benchmark_design.page_level.foreground import extract_foreground_mask
from benchmark_design.page_level.gray_cache import PageGrayCache
from benchmark_design.page_level.models import CalibrationResult, ImageInventoryRow, ImageRecord, PageLevelConfig
from benchmark_design.io.image import load_grayscale_image


def _inventory_row(record: ImageRecord, *, width: int, height: int) -> ImageInventoryRow:
    return ImageInventoryRow(
        image_id=record.image_id,
        relative_path=record.relative_path,
        width=width,
        height=height,
        aspect_ratio=width / height,
        file_format="png",
        stored_color_mode="L",
        channel_count=1,
        dtype="uint8",
        bits_per_channel=8,
        rgb_channels_identical=True,
        alpha_nonopaque_ratio=0.0,
        effective_color_type="grayscale_content;opaque",
    )


def test_extract_foreground_mask_uses_full_page(tmp_path: Path) -> None:
    array = np.full((10, 12), 240, dtype=np.uint8)
    array[2:5, 3:7] = 30
    image_path = tmp_path / "page.png"
    Image.fromarray(array, mode="L").save(image_path)
    record = ImageRecord("page", "page.png", image_path)
    calibration = CalibrationResult(
        dark_reference=0.0,
        light_reference=255.0,
        gray_threshold=128.0,
        tau_d=128.0 / 255.0,
        dark_percentile=1.0,
        light_percentile=99.5,
        threshold_method="test",
        image_count=1,
    )
    gray, normalized, mask = extract_foreground_mask(record, calibration)
    assert gray.shape == (10, 12)
    assert normalized.shape == (10, 12)
    assert normalized.dtype == np.float32
    assert mask.shape == (10, 12)
    assert mask.sum() > 0
    assert mask.sum() < mask.size


def test_compute_image_features_density_and_aspect_ratio(tmp_path: Path) -> None:
    array = np.full((8, 10), 220, dtype=np.uint8)
    array[1:4, 1:4] = 40
    image_path = tmp_path / "density.png"
    Image.fromarray(array, mode="L").save(image_path)
    record = ImageRecord("density", "density.png", image_path)
    inventory = _inventory_row(record, width=10, height=8)
    calibration = CalibrationResult(
        dark_reference=0.0,
        light_reference=255.0,
        gray_threshold=128.0,
        tau_d=128.0 / 255.0,
        dark_percentile=1.0,
        light_percentile=99.5,
        threshold_method="test",
        image_count=1,
    )
    features = compute_image_features(record, inventory, calibration)
    assert 0.0 < features.foreground_density < 1.0
    assert features.aspect_ratio == 1.25


def test_gray_cache_produces_same_features_as_reload(tmp_path: Path) -> None:
    array = np.full((8, 10), 220, dtype=np.uint8)
    array[1:4, 1:4] = 40
    image_path = tmp_path / "density.png"
    Image.fromarray(array, mode="L").save(image_path)
    record = ImageRecord("density", "density.png", image_path)
    inventory = _inventory_row(record, width=10, height=8)
    calibration = CalibrationResult(
        dark_reference=0.0,
        light_reference=255.0,
        gray_threshold=128.0,
        tau_d=128.0 / 255.0,
        dark_percentile=1.0,
        light_percentile=99.5,
        threshold_method="test",
        image_count=1,
    )
    direct = compute_image_features(record, inventory, calibration)
    gray = load_grayscale_image(image_path)
    cache = PageGrayCache(tmp_path / "gray_cache")
    cache.store(record.image_id, gray)
    cached = compute_page_image_metrics(
        record,
        inventory,
        calibration,
        PageLevelConfig(input_root=tmp_path, output_root=tmp_path / "out", show_progress=False),
        gray_cache=cache,
    ).features
    from_gray = compute_image_features_from_gray(gray, inventory, calibration)
    assert cached.foreground_density == direct.foreground_density
    assert from_gray.foreground_density == direct.foreground_density
    cache.cleanup()
