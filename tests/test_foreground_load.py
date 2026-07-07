"""Foreground load metric tests."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from benchmark_design.report.vision.foreground_pixel_density_comparison_figures import (
    HIGH_DENSITY_THRESHOLD,
    build_difference_overlay,
    export_foreground_pixel_density_comparison_figures,
)
from benchmark_design.report.vision.foreground_pixel_density_export import (
    BLOCK_METRICS_COLUMNS,
    OVERALL_COLUMNS,
    PAGE_METRICS_COLUMNS,
    REGION_METRICS_COLUMNS,
    write_foreground_pixel_density_block_metrics_csv,
    write_foreground_pixel_density_diagnostics_json,
    write_foreground_pixel_density_overall_csv,
    write_foreground_pixel_density_page_metrics_csv,
    write_foreground_pixel_density_region_metrics_csv,
)
from benchmark_design.report.vision.foreground_pixel_density_figures import (
    export_foreground_pixel_density_figures,
)
from benchmark_design.report.vision.foreground_pixel_density_summary import (
    write_foreground_pixel_density_summary_md,
)
from benchmark_design.report.vision.foreground_pixel_density_stats import (
    compute_block_density_bands,
    compute_block_density_stats,
    compute_page_density_stats,
)
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.vision.foreground_load.classification import (
    EXTREME_DENSITY_TAG,
    diagnostic_tags,
    foreground_load_level,
    relative_load_tertile,
)
from benchmark_design.vision.foreground_load.models import PageForegroundLoadResult
from benchmark_design.vision.foreground_load.normalization import (
    compute_darkness_from_gray,
    darkness_histogram_in_mask,
    robust_normalize_gray,
)
from benchmark_design.vision.foreground_load.otsu import otsu_from_darkness_histogram, otsu_threshold
from benchmark_design.vision.foreground_load.threshold_estimation import (
    bimodal_valley_threshold,
    estimate_dataset_threshold,
)
from benchmark_design.vision.foreground_load.pipeline import (
    assign_foreground_density_diagnostics,
    compute_foreground_load_from_pages,
    compute_foreground_load_thresholds,
    estimate_global_foreground_config,
)
from benchmark_design.vision.foreground_load.raster import rasterize_polygon
from benchmark_design.vision.page_metrics import compute_vision_benchmark_results
from benchmark_design.vision.processing_options import VisionProcessingOptions


def _page(
    page_id: str,
    blocks: list[tuple[str, list[list[float]]]],
    *,
    width: int = 200,
    height: int = 200,
) -> PageAnnotation:
    page_blocks = tuple(
        PageBlockAnnotation(
            page_id=page_id,
            block_id=f"{page_id}:block_{index}",
            block_type=block_type,
            block_order=index,
            polygon=tuple((float(x), float(y)) for x, y in polygon),
        )
        for index, (block_type, polygon) in enumerate(blocks)
    )
    return PageAnnotation(
        page_id=page_id,
        image_name=f"{page_id}.jpg",
        source_file=f"/tmp/{page_id}.json",
        image_width=width,
        image_height=height,
        blocks=page_blocks,
    )


def _page_result(
    page_id: str,
    *,
    density: float | None,
    review_reason: str = "",
) -> PageForegroundLoadResult:
    return PageForegroundLoadResult(
        page_id=page_id,
        image_name=f"{page_id}.jpg",
        image_width=100,
        image_height=100,
        page_area=10000,
        effective_region_area_ratio=0.5 if density is not None else None,
        num_txtBlock=1,
        num_figure=0,
        num_chart=0,
        num_deleted_text_block=0,
        D_page_eff=density,
        mean_darkness=0.05 if density is not None else None,
        raw_otsu_density=0.04 if density is not None else None,
        D_page_tau_minus=(density - 0.01) if density is not None else None,
        D_page_tau_plus=(density + 0.01) if density is not None else None,
        foreground_load_level="",
        relative_load_tertile="",
        foreground_load_tags="",
        page_otsu_threshold=100.0,
        threshold_dataset=0.2,
        R_eff_area=100,
        F_eff=int(density * 100) if density is not None else 0,
        num_effective_blocks=1,
        needs_manual_review=bool(review_reason),
        review_reason=review_reason,
        review_image_path="",
        block_results=(),
    )


def _save_binary_page(path: Path, *, width: int, height: int, left_dark: bool = True) -> None:
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    if left_dark:
        draw.rectangle((0, 0, width // 2, height), fill=0)
    else:
        draw.rectangle((0, 0, width, height), fill=0)
    image.save(path)


def _compute_pages(pages: list[PageAnnotation], input_dir: Path):
    processing = VisionProcessingOptions(show_progress=False, workers=1)
    return compute_foreground_load_from_pages(pages, input_dir=input_dir, processing=processing)


def test_robust_normalize_gray_clips_to_unit_interval() -> None:
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    normalized = robust_normalize_gray(gray, q_low=0.0, q_high=100.0)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_darkness_increases_for_darker_pixels() -> None:
    gray = np.full((16, 16), 240, dtype=np.uint8)
    gray[6:10, 6:10] = 20
    darkness = compute_darkness_from_gray(gray, q_low=0.0, q_high=100.0)
    assert float(darkness[8, 8]) > float(darkness[0, 0])


def test_otsu_on_bimodal_values() -> None:
    values = np.array([0] * 100 + [200] * 100, dtype=np.uint8)
    threshold = otsu_threshold(values)
    assert 0 <= threshold <= 255


def test_otsu_from_darkness_histogram_unit_interval() -> None:
    hist = np.zeros(256, dtype=np.int64)
    hist[20:80] = 100
    hist[180:240] = 100
    threshold = otsu_from_darkness_histogram(hist)
    assert 0.0 < threshold < 1.0


def test_build_difference_overlay_colors() -> None:
    raw_fg = np.array([[True, True, False]], dtype=bool)
    norm_fg = np.array([[True, False, True]], dtype=bool)
    overlay = build_difference_overlay(raw_fg, norm_fg)
    assert overlay.shape == (1, 3, 3)
    assert tuple(overlay[0, 0]) == (0.0, 0.0, 0.0)
    assert tuple(overlay[0, 1]) == (1.0, 0.0, 0.0)
    assert tuple(overlay[0, 2]) == (0.0, 0.0, 1.0)


def test_rasterize_polygon_uses_pixel_area_not_bbox() -> None:
    triangle = ((10.0, 10.0), (90.0, 10.0), (50.0, 90.0))
    mask = rasterize_polygon(triangle, image_width=100, image_height=100)
    assert mask.sum() < 80 * 80


def test_diagnostic_level_absolute_thresholds() -> None:
    assert foreground_load_level(0.05) == "low"
    assert foreground_load_level(0.08) == "medium"
    assert foreground_load_level(0.15) == "high"
    assert foreground_load_level(None) == ""


def test_diagnostic_tags_extreme_candidate() -> None:
    tags = diagnostic_tags(0.20, "")
    assert EXTREME_DENSITY_TAG in tags.split(";")
    assert foreground_load_level(0.20) == "high"


def test_diagnostic_tags_maps_saturation() -> None:
    tags = diagnostic_tags(0.01, "density_saturated_low")
    assert "saturated_low" in tags.split(";")
    tags = diagnostic_tags(0.99, "density_saturated_high")
    assert "saturated_high" in tags.split(";")


def test_relative_load_tertile_assignment() -> None:
    assert relative_load_tertile(0.1, p33=0.2, p66=0.4) == "lower"
    assert relative_load_tertile(0.3, p33=0.2, p66=0.4) == "middle"
    assert relative_load_tertile(0.5, p33=0.2, p66=0.4) == "upper"


def test_page_foreground_load_density_from_mask(tmp_path: Path) -> None:
    page = _page(
        "p1",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("figure", [[0, 0], [0, 0], [0, 0]]),
        ],
        width=200,
        height=200,
    )
    image_path = tmp_path / page.image_name
    _save_binary_page(image_path, width=200, height=200, left_dark=True)
    results, _, global_config = _compute_pages([page], tmp_path)
    result = results[0]
    assert result.R_eff_area > 0
    assert result.D_page_eff is not None
    assert 0.0 <= result.D_page_eff <= 1.0
    assert result.threshold_dataset == global_config.tau_D
    assert result.mean_darkness is not None
    assert result.raw_otsu_density is not None
    assert result.page_otsu_threshold is not None
    assert result.F_eff <= result.R_eff_area


def test_shared_tau_D_across_pages(tmp_path: Path) -> None:
    page_a = _page("pa", [("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]])])
    page_b = _page("pb", [("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]])])
    _save_binary_page(tmp_path / page_a.image_name, width=200, height=200, left_dark=True)
    _save_binary_page(tmp_path / page_b.image_name, width=200, height=200, left_dark=False)
    results, _, global_config = _compute_pages([page_a, page_b], tmp_path)
    assert all(result.threshold_dataset == global_config.tau_D for result in results)


def test_block_raw_otsu_can_differ_while_sharing_tau_D(tmp_path: Path) -> None:
    page = _page(
        "p2",
        [
            ("Txtblock", [[20, 20], [80, 20], [80, 180], [20, 180]]),
            ("Txtblock", [[120, 20], [180, 20], [180, 180], [120, 180]]),
        ],
        width=200,
        height=200,
    )
    image = Image.new("L", (200, 200), color=255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 80, 180), fill=0)
    draw.rectangle((120, 20, 180, 180), fill=128)
    image.save(tmp_path / page.image_name)
    results, _, global_config = _compute_pages([page], tmp_path)
    result = results[0]
    assert len(result.block_results) == 2
    assert all(block.D_block_i is not None for block in result.block_results)
    assert all(block.threshold_dataset == global_config.tau_D for block in result.block_results)
    thresholds = {block.block_otsu_threshold for block in result.block_results}
    assert len(thresholds) >= 1


def test_missing_image_marks_review(tmp_path: Path) -> None:
    page = _page("p3", [("Txtblock", [[10, 10], [90, 10], [90, 90], [10, 90]])])
    results, _, _ = _compute_pages([page], tmp_path)
    result = results[0]
    assert result.needs_manual_review is True
    assert "missing_image" in result.review_reason
    assert result.D_page_eff is None


def test_empty_effective_mask_marks_review(tmp_path: Path) -> None:
    page = _page("p4", [("unknown", [[10, 10], [90, 10], [90, 90], [10, 90]])])
    image_path = tmp_path / page.image_name
    _save_binary_page(image_path, width=200, height=200)
    results, _, _ = _compute_pages([page], tmp_path)
    result = results[0]
    assert result.R_eff_area == 0
    assert result.needs_manual_review is True
    assert "empty_effective_mask" in result.review_reason


def test_out_of_bounds_polygon_marks_review(tmp_path: Path) -> None:
    page = _page("p5", [("Txtblock", [[10, 10], [250, 10], [250, 90], [10, 90]])], width=200, height=200)
    image_path = tmp_path / page.image_name
    _save_binary_page(image_path, width=200, height=200)
    results, _, _ = _compute_pages([page], tmp_path)
    assert "mask_out_of_bounds" in results[0].review_reason


def test_density_saturation_review(tmp_path: Path) -> None:
    page = _page("p6", [("Txtblock", [[10, 10], [190, 10], [190, 190], [10, 190]])], width=200, height=200)
    image_path = tmp_path / page.image_name
    Image.new("L", (200, 200), color=0).save(image_path)
    results, _, _ = _compute_pages([page], tmp_path)
    result = results[0]
    assert result.D_page_eff is not None
    assert result.D_page_eff >= 0.98
    assert "density_saturated_high" in result.review_reason


def test_assign_foreground_density_diagnostics_terciles() -> None:
    results = [
        _page_result(f"p{i}", density=density)
        for i, density in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ]
    thresholds = compute_foreground_load_thresholds(results)
    labeled = assign_foreground_density_diagnostics(results, thresholds)
    tertiles = {result.page_id: result.relative_load_tertile for result in labeled}
    assert tertiles["p0"] == "lower"
    assert tertiles["p8"] == "upper"
    assert all(result.foreground_load_level for result in labeled)


def test_foreground_pixel_density_csv_columns(tmp_path: Path) -> None:
    page = _page("p_csv", [("Txtblock", [[10, 10], [90, 10], [90, 90], [10, 90]])])
    image_path = tmp_path / page.image_name
    _save_binary_page(image_path, width=200, height=200)
    results, thresholds, global_config = _compute_pages([page], tmp_path)
    labeled = assign_foreground_density_diagnostics(
        results,
        compute_foreground_load_thresholds(results, global_config=global_config),
    )[0]
    page_csv = tmp_path / "page.csv"
    block_csv = tmp_path / "block.csv"
    region_csv = tmp_path / "region.csv"
    overall_csv = tmp_path / "overall.csv"
    diagnostics_json = tmp_path / "diagnostics.json"
    write_foreground_pixel_density_page_metrics_csv([labeled], page_csv)
    write_foreground_pixel_density_block_metrics_csv(list(labeled.block_results), block_csv)
    write_foreground_pixel_density_region_metrics_csv([labeled], region_csv)
    write_foreground_pixel_density_overall_csv([labeled], overall_csv)
    write_foreground_pixel_density_diagnostics_json(
        [labeled],
        thresholds,
        diagnostics_json,
        global_config=global_config,
    )
    with page_csv.open(encoding="utf-8", newline="") as handle:
        page_header = next(csv.reader(handle))
    with block_csv.open(encoding="utf-8", newline="") as handle:
        block_header = next(csv.reader(handle))
    with region_csv.open(encoding="utf-8", newline="") as handle:
        region_header = next(csv.reader(handle))
    with overall_csv.open(encoding="utf-8", newline="") as handle:
        overall_header = next(csv.reader(handle))
    assert page_header == list(PAGE_METRICS_COLUMNS)
    assert block_header == list(BLOCK_METRICS_COLUMNS)
    assert region_header == list(REGION_METRICS_COLUMNS)
    assert overall_header == list(OVERALL_COLUMNS)
    assert "D_page" in page_header
    assert "mean_darkness" in page_header
    assert "raw_otsu_density" in page_header
    assert "threshold_dataset" in region_header
    payload = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    assert "methodology" in payload
    assert payload["methodology"]["tau_D"] == global_config.tau_D
    assert "sensitivity_analysis" in payload


def test_foreground_pixel_density_summary_smoke(tmp_path: Path) -> None:
    results = [
        _page_result("p1", density=0.05, review_reason="density_saturated_low"),
        _page_result("p2", density=0.20, review_reason=""),
    ]
    thresholds = compute_foreground_load_thresholds(results)
    labeled = assign_foreground_density_diagnostics(results, thresholds)
    page_stats = compute_page_density_stats(labeled)
    block_stats = compute_block_density_stats(labeled)
    summary_path = tmp_path / "summary.md"
    write_foreground_pixel_density_summary_md(
        labeled,
        summary_path,
        page_stats=page_stats,
        block_stats=block_stats,
        flow_stats=[],
        thresholds=thresholds,
    )
    text = summary_path.read_text(encoding="utf-8")
    assert "# Foreground Pixel Density Summary" in text
    assert "foreground pixel density" in text
    assert "tau_D" in text
    assert "mean darkness" in text
    assert "high_density_comparisons" in text


def test_foreground_pixel_density_figure_smoke(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return

    page = _page("p_fig", [("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]])], width=200, height=200)
    image_path = tmp_path / page.image_name
    _save_binary_page(image_path, width=200, height=200)
    results, thresholds, global_config = _compute_pages([page], tmp_path)
    labeled = assign_foreground_density_diagnostics(
        results,
        compute_foreground_load_thresholds(results, global_config=global_config),
    )[0]
    counts = export_foreground_pixel_density_figures(
        [labeled],
        [page],
        [],
        input_dir=tmp_path,
        figures_root=tmp_path / "figures",
    )
    assert counts.get("d_page_density_bands.png") is True
    assert counts.get("d_block_density_bands.png") is True
    assert counts.get("density_level_comparison.png") is True


def test_high_density_comparison_figure_smoke(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return

    page = _page("p_high", [("Txtblock", [[10, 10], [190, 10], [190, 190], [10, 190]])], width=200, height=200)
    Image.new("L", (200, 200), color=0).save(tmp_path / page.image_name)
    results, _, global_config = _compute_pages([page], tmp_path)
    result = results[0]
    assert result.D_page_eff is not None
    assert result.D_page_eff > HIGH_DENSITY_THRESHOLD
    counts = export_foreground_pixel_density_comparison_figures(
        results,
        [page],
        input_dir=tmp_path,
        figures_root=tmp_path / "figures",
        global_config=global_config,
    )
    assert counts["high_density_page_comparisons"] >= 1


def test_estimate_global_from_darkness_histogram() -> None:
    hist = darkness_histogram_in_mask(
        np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16),
        np.ones((16, 16), dtype=bool),
    )
    config = estimate_global_foreground_config([hist])
    assert 0.0 < config.tau_D < 1.0
    assert config.threshold_method


def test_bimodal_valley_prefers_valley_between_peaks() -> None:
    hist = np.zeros(256, dtype=np.int64)
    hist[20:60] = 1000
    hist[180:220] = 800
    valley = bimodal_valley_threshold(hist)
    assert valley is not None
    assert 0.2 < valley < 0.8


def test_estimate_dataset_threshold_fallback_to_otsu() -> None:
    hist = np.zeros(256, dtype=np.int64)
    hist[200:240] = 100
    tau, method = estimate_dataset_threshold(hist)
    assert 0.0 < tau < 1.0
    assert method.value in {"bimodal_valley", "gmm_intersection", "pooled_otsu"}


def test_fixture_page_loads(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "flow_structure_page.json"
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(fixture, input_dir / "sample.jpg.json")
    image = Image.new("RGB", (1000, 1000), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 80, 900, 800), fill=(20, 20, 20))
    image.save(input_dir / "sample.jpg")
    from benchmark_design.vision.flow_structure.page_loader import _load_page_annotation

    page = _load_page_annotation(input_dir / "sample.jpg.json", input_dir=input_dir, dataset="ours")
    results, _, _ = _compute_pages([page], input_dir)
    result = results[0]
    assert result.num_effective_blocks >= 2
    assert result.D_page_eff is not None


def test_vision_benchmark_results_includes_global_config(tmp_path: Path) -> None:
    page = _page("p_vis", [("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]])])
    Image.new("RGB", (200, 200), color=(255, 255, 255)).save(tmp_path / page.image_name)
    vision = compute_vision_benchmark_results(
        tmp_path,
        processing=VisionProcessingOptions(show_progress=False, workers=1),
        pages=[page],
    )
    assert vision.global_foreground_config.tau_D > 0
    assert vision.foreground_load[0].threshold_dataset == vision.global_foreground_config.tau_D
