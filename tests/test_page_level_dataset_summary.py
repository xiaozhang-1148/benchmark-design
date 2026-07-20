"""Tests for dataset summary highlights."""

from __future__ import annotations

from benchmark_design.page_level.models import ImageFeatureRow
from benchmark_design.page_level.statistics import build_dataset_summary, compute_dataset_highlights


def _row(**kwargs: object) -> ImageFeatureRow:
    defaults = {
        "image_id": "img",
        "relative_path": "img.png",
        "width": 100,
        "height": 200,
        "aspect_ratio": 0.5,
        "file_format": "png",
        "stored_color_mode": "L",
        "effective_color_type": "grayscale_content;opaque",
        "bits_per_channel": 8,
        "foreground_density": 0.04,
        "aspect_ratio_group": "portrait",
    }
    defaults.update(kwargs)
    return ImageFeatureRow(**defaults)  # type: ignore[arg-type]


def test_compute_dataset_highlights() -> None:
    features = [
        _row(foreground_density=0.02),
        _row(foreground_density=0.09),
        _row(foreground_density=0.04),
    ]
    highlights = compute_dataset_highlights(features)
    assert highlights["density_below_0_03_ratio"] == 1 / 3
    assert highlights["density_above_0_08_ratio"] == 1 / 3
    assert highlights["aspect_groups"]["portrait"]["count"] == 3


def test_build_dataset_summary_includes_highlights() -> None:
    features = [_row(), _row(image_id="b", aspect_ratio_group="landscape", aspect_ratio=1.5)]
    summary = build_dataset_summary(features, [], calibration={"global_threshold": 128.0})
    assert "density_below_0_03_ratio" in summary
    assert "aspect_groups" in summary
    assert "background_brightness_equal_255_ratio" not in summary
