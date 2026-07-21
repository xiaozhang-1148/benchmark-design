"""Tests for block-level foreground density over block annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image

from benchmark_design.block_level.block_foreground_density import (
    compute_block_foreground_density_from_masks,
    compute_block_foreground_densities,
    rasterize_polygon_mask,
)
from benchmark_design.block_level.flow_structure.geometry import polygon_bbox
from benchmark_design.block_level.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.page_level.foreground import extract_block_foreground_mask_from_gray
from benchmark_design.page_level.gray_cache import PageGrayCache
from benchmark_design.page_level.models import CalibrationResult


def test_rasterize_polygon_mask_counts_pixels_in_rectangle() -> None:
    polygon = ((1.0, 1.0), (4.0, 1.0), (4.0, 3.0), (1.0, 3.0))
    mask = rasterize_polygon_mask(polygon, width=5, height=5)
    assert int(mask.sum()) >= 6
    assert not mask[0, 0]
    assert mask[2:4, 1:4].all()


def test_rasterize_polygon_mask_matches_matplotlib_bbox_reference() -> None:
    polygon = ((2.2, 1.1), (7.8, 1.4), (7.5, 5.6), (2.0, 5.3))
    width, height = 10, 8
    optimized = rasterize_polygon_mask(polygon, width=width, height=height)

    x1, y1, x2, y2 = polygon_bbox(polygon)
    left = max(int(np.floor(x1)), 0)
    top = max(int(np.floor(y1)), 0)
    right = min(int(np.ceil(x2)) + 1, width)
    bottom = min(int(np.ceil(y2)) + 1, height)
    yy, xx = np.mgrid[top:bottom, left:right]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = MplPath(polygon).contains_points(points).reshape(bottom - top, right - left)
    reference = np.zeros((height, width), dtype=bool)
    reference[top:bottom, left:right] = inside

    assert optimized.shape == reference.shape
    assert np.array_equal(optimized, reference)


def test_compute_block_foreground_density_uses_annotation_pixels_as_denominator(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "page.png"
    gray = np.full((10, 10), 220, dtype=np.uint8)
    gray[2:6, 2:6] = 40
    Image.fromarray(gray, mode="L").save(image_path)

    calibration = CalibrationResult(
        dark_reference=40.0,
        light_reference=220.0,
        gray_threshold=130.0,
        tau_d=130.0 / 255.0,
        dark_percentile=5.0,
        light_percentile=95.0,
        threshold_method="test",
        image_count=1,
    )
    block = PageBlockAnnotation(
        page_id="page.jpg",
        block_id="b1",
        block_type="Txtblock",
        block_order=0,
        polygon=((2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5)),
    )
    foreground_mask = extract_block_foreground_mask_from_gray(gray, calibration)
    row = compute_block_foreground_density_from_masks(
        block,
        image_width=10,
        image_height=10,
        foreground_mask=foreground_mask,
    )
    assert row is not None
    assert row.annotation_pixel_count > 0
    assert row.foreground_pixel_count <= row.annotation_pixel_count
    assert row.foreground_density == row.foreground_pixel_count / row.annotation_pixel_count
    assert row.foreground_density == 1.0
    assert row.foreground_density > (16 / 100)


def test_compute_block_foreground_densities_uses_gray_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    gray = np.full((10, 10), 220, dtype=np.uint8)
    gray[2:6, 2:6] = 40
    Image.fromarray(gray, mode="L").save(image_path)

    cache_root = tmp_path / "gray_cache"
    cache = PageGrayCache(cache_root)
    cache.store("page", gray)

    calibration = CalibrationResult(
        dark_reference=40.0,
        light_reference=220.0,
        gray_threshold=130.0,
        tau_d=130.0 / 255.0,
        dark_percentile=5.0,
        light_percentile=95.0,
        threshold_method="test",
        image_count=1,
    )
    page = PageAnnotation(
        page_id="page.json",
        image_name="page.png",
        source_file="page.json",
        image_width=10,
        image_height=10,
        blocks=(
            PageBlockAnnotation(
                page_id="page.json",
                block_id="b1",
                block_type="Txtblock",
                block_order=0,
                polygon=((2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5)),
            ),
        ),
    )

    rows = compute_block_foreground_densities(
        [page],
        input_dir=tmp_path,
        calibration=calibration,
        gray_cache_root=cache_root,
        show_progress=False,
        workers=1,
    )
    assert len(rows) == 1
    assert rows[0].foreground_density == 1.0


def test_compute_block_foreground_densities_skips_non_txtblocks(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    gray = np.full((10, 10), 220, dtype=np.uint8)
    gray[2:6, 2:6] = 40
    Image.fromarray(gray, mode="L").save(image_path)

    calibration = CalibrationResult(
        dark_reference=40.0,
        light_reference=220.0,
        gray_threshold=130.0,
        tau_d=130.0 / 255.0,
        dark_percentile=5.0,
        light_percentile=95.0,
        threshold_method="test",
        image_count=1,
    )
    page = PageAnnotation(
        page_id="page.json",
        image_name="page.png",
        source_file="page.json",
        image_width=10,
        image_height=10,
        blocks=(
            PageBlockAnnotation(
                page_id="page.json",
                block_id="b1",
                block_type="Txtblock",
                block_order=0,
                polygon=((2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5)),
            ),
            PageBlockAnnotation(
                page_id="page.json",
                block_id="b2",
                block_type="figure",
                block_order=1,
                polygon=((0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)),
            ),
        ),
    )

    rows = compute_block_foreground_densities(
        [page],
        input_dir=tmp_path,
        calibration=calibration,
        show_progress=False,
        workers=1,
    )
    assert len(rows) == 1
    assert rows[0].block_type == "Txtblock"
