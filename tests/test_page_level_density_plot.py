"""Tests for the page-level foreground density distribution figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.page_level.models import ImageFeatureRow
from benchmark_design.report.page_level.plotting import (
    _foreground_density_bin_counts,
    _page_foreground_density_xlabel,
    plot_foreground_density_distribution,
)


def _row(density: float, image_id: str = "img") -> ImageFeatureRow:
    return ImageFeatureRow(
        image_id=image_id,
        relative_path=f"{image_id}.png",
        width=100,
        height=200,
        aspect_ratio=0.5,
        file_format="png",
        stored_color_mode="L",
        effective_color_type="grayscale_content;opaque",
        bits_per_channel=8,
        foreground_density=density,
        aspect_ratio_group="portrait",
    )


def test_page_foreground_density_xlabel_uses_pooled_otsu_threshold() -> None:
    label = _page_foreground_density_xlabel(gray_threshold=157.0)
    assert "pooled Otsu" in label
    assert "t_I = 157" in label
    assert "valley" not in label.lower()


def test_plot_foreground_density_distribution_writes_png(tmp_path: Path) -> None:
    features = [
        _row(0.01, "a"),
        _row(0.03, "b"),
        _row(0.05, "c"),
        _row(0.07, "d"),
        _row(0.09, "e"),
        _row(0.11, "f"),
    ]
    counts = _foreground_density_bin_counts(np.array([row.foreground_density for row in features]))
    assert counts.tolist() == [1, 1, 1, 1, 1, 1]

    output_path = tmp_path / "foreground_density_distribution.png"
    plot_foreground_density_distribution(features, output_path)
    assert output_path.is_file()
    with Image.open(output_path) as image:
        assert image.size[0] > 100
        assert image.size[1] > 100
