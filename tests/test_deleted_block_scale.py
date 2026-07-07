"""Deleted-Block Scale metric tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from benchmark_design.report.vision.deleted_block_scale_export import (
    BLOCK_GEOMETRY_COLUMNS,
    PAGE_METRICS_COLUMNS,
    write_deleted_block_scale_block_geometry_csv,
    write_deleted_block_scale_diagnostics_json,
    write_deleted_block_scale_page_metrics_csv,
)
from benchmark_design.report.vision.deleted_block_scale_figures import export_deleted_block_scale_figures
from benchmark_design.report.vision.deleted_block_scale_stats import (
    compute_deleted_block_scale_summary_stats,
)
from benchmark_design.report.vision.deleted_block_scale_summary import write_deleted_block_scale_summary_md
from benchmark_design.vision.deleted_block_scale.compute import compute_page_deleted_block_scale
from benchmark_design.vision.deleted_block_scale.masks import build_answer_deleted_masks
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.vision.foreground_load.raster import rasterize_polygon


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


def _save_blank_image(path: Path, *, width: int, height: int) -> None:
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(path)


def test_no_deleted_block_r_del_zero(tmp_path: Path) -> None:
    page = _page(
        "p_none",
        [("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]])],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    assert result.deleted_area == 0
    assert result.r_del == 0.0
    assert result.has_deleted_text_block is False


def test_low_r_del_page(tmp_path: Path) -> None:
    page = _page(
        "p_small",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("deleted_text_block", [[20, 20], [40, 20], [40, 40], [20, 40]]),
        ],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    assert result.deleted_area > 0
    assert result.r_del is not None
    assert 0 < result.r_del <= 0.10
    assert result.has_deleted_text_block is True


def test_high_r_del_page(tmp_path: Path) -> None:
    page = _page(
        "p_large",
        [
            ("Txtblock", [[20, 20], [80, 20], [80, 180], [20, 180]]),
            ("deleted_text_block", [[100, 20], [180, 20], [180, 180], [100, 180]]),
        ],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    assert result.r_del is not None
    assert result.r_del > 0.10
    assert result.has_deleted_text_block is True


def test_cross_class_overlap_uses_union_denominator() -> None:
    overlap = [[50, 50], [150, 50], [150, 150], [50, 150]]
    page = _page(
        "p_overlap",
        [
            ("Txtblock", overlap),
            ("deleted_text_block", overlap),
        ],
    )
    masks = build_answer_deleted_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    )
    valid_area = int(masks.valid_union.sum())
    deleted_area = int(masks.deleted_union.sum())
    answer_related_area = int(masks.answer_related_union.sum())
    assert valid_area == deleted_area
    assert answer_related_area == valid_area
    assert answer_related_area < valid_area + deleted_area


def test_same_class_overlap_not_double_counted() -> None:
    page = _page(
        "p_same_overlap",
        [
            ("Txtblock", [[20, 20], [120, 20], [120, 120], [20, 120]]),
            ("Txtblock", [[80, 80], [180, 80], [180, 180], [80, 180]]),
        ],
    )
    masks = build_answer_deleted_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    )
    block_areas = sum(
        int(
            rasterize_polygon(
                block.polygon,
                image_width=page.image_width,
                image_height=page.image_height,
            ).sum()
        )
        for block in page.blocks
    )
    valid_area = int(masks.valid_union.sum())
    assert valid_area < block_areas


def test_figure_only_page_has_valid_area(tmp_path: Path) -> None:
    page = _page("p_empty", [("figure", [[10, 10], [90, 10], [90, 90], [10, 90]])])
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    assert result.valid_area > 0
    assert result.answer_related_area == result.valid_area
    assert result.deleted_area == 0
    assert result.r_del == 0.0
    assert result.has_deleted_text_block is False


def test_chart_and_figure_increase_valid_union() -> None:
    page = _page(
        "p_valid_mix",
        [
            ("Txtblock", [[20, 20], [80, 20], [80, 80], [20, 80]]),
            ("figure", [[100, 20], [180, 20], [180, 80], [100, 80]]),
            ("chart", [[20, 100], [80, 100], [80, 180], [20, 180]]),
        ],
    )
    masks = build_answer_deleted_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    )
    valid_area = int(masks.valid_union.sum())
    txt_only = _page("p_txt", [("Txtblock", [[20, 20], [80, 20], [80, 80], [20, 80]])])
    txt_masks = build_answer_deleted_masks(
        txt_only.blocks,
        image_width=txt_only.image_width,
        image_height=txt_only.image_height,
    )
    assert valid_area > int(txt_masks.valid_union.sum())


def test_missing_image_marks_review(tmp_path: Path) -> None:
    page = _page("p_missing", [("Txtblock", [[10, 10], [90, 10], [90, 90], [10, 90]])])
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    assert "missing_image" in result.review_reason
    assert result.needs_manual_review is True


def test_deleted_text_block_present_but_zero_area(tmp_path: Path) -> None:
    page = _page(
        "p_zero_deleted",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("deleted_text_block", [[10, 10], [20, 10]]),
        ],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    assert "invalid_polygon" in result.review_reason or "deleted_text_block_present_but_zero_area" in result.review_reason


def test_deleted_block_scale_csv_columns(tmp_path: Path) -> None:
    page = _page(
        "p_csv",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("deleted_text_block", [[20, 20], [40, 20], [40, 40], [20, 40]]),
        ],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    page_csv = tmp_path / "page.csv"
    block_csv = tmp_path / "block.csv"
    diagnostics_json = tmp_path / "diagnostics.json"
    write_deleted_block_scale_page_metrics_csv([result], page_csv)
    write_deleted_block_scale_block_geometry_csv(list(result.block_records), block_csv)
    write_deleted_block_scale_diagnostics_json(
        compute_deleted_block_scale_summary_stats([result]),
        diagnostics_json,
    )
    with page_csv.open(encoding="utf-8", newline="") as handle:
        page_header = next(csv.reader(handle))
    with block_csv.open(encoding="utf-8", newline="") as handle:
        block_header = next(csv.reader(handle))
    assert page_header == list(PAGE_METRICS_COLUMNS)
    assert block_header == list(BLOCK_GEOMETRY_COLUMNS)
    assert "r_del" in page_header
    assert "deletion_scale" not in page_header
    assert "tau" not in page_header
    payload = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    assert "dataset_level_deleted_area_ratio" in payload
    assert payload["area_definitions"]["A_valid"] == "Txtblock ∪ chart ∪ figure"


def test_deleted_block_scale_summary_smoke(tmp_path: Path) -> None:
    page = _page(
        "p_sum",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("deleted_text_block", [[100, 20], [180, 20], [180, 180], [100, 180]]),
        ],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    stats = compute_deleted_block_scale_summary_stats([result])
    summary_path = tmp_path / "summary.md"
    write_deleted_block_scale_summary_md([result], summary_path, stats=stats)
    text = summary_path.read_text(encoding="utf-8")
    assert "### 2.3.2 Overview" in text
    assert "Dataset-level deleted area ratio" in text
    assert "Pages with R_del >= 0.2" in text
    assert "Threshold tau" not in text
    assert "deletion_scale" not in text
    assert "### Distribution figures" in text
    assert "r_del_histogram.png" in text


def test_r_del_in_valid_range(tmp_path: Path) -> None:
    page = _page(
        "p_range",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("deleted_text_block", [[20, 20], [60, 20], [60, 60], [20, 60]]),
        ],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    assert result.r_del is not None
    assert 0.0 <= result.r_del <= 1.0


def test_deleted_block_scale_continuous_figures(tmp_path: Path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return

    page = _page(
        "p_fig",
        [
            ("Txtblock", [[20, 20], [80, 20], [80, 180], [20, 180]]),
            ("deleted_text_block", [[100, 20], [180, 20], [180, 180], [100, 180]]),
        ],
    )
    image_path = tmp_path / page.image_name
    _save_blank_image(image_path, width=200, height=200)
    result = compute_page_deleted_block_scale(page, input_dir=tmp_path)
    counts = export_deleted_block_scale_figures(
        [result],
        [page],
        input_dir=tmp_path,
        figures_root=tmp_path / "figures" / "deleted_block_scale",
    )
    assert counts.get("r_del_histogram.png")
    assert counts.get("deleted_instance_histogram.png")
    assert (tmp_path / "figures" / "deleted_block_scale" / "r_del_histogram.png").is_file()
    assert (tmp_path / "figures" / "deleted_block_scale" / "deleted_instance_histogram.png").is_file()
    assert (tmp_path / "figures" / "deleted_block_scale" / "high_r_del_examples" / "p_fig.png").is_file()
