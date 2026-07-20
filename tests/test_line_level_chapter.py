"""Tests for chapter 4 tables and distribution figures."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.line_level.layout import horizontal_adjacent_scope_rows
from benchmark_design.line_level.models import LineLevelConfig, LineMetricsRow, TargetPairRow
from benchmark_design.report.line_level.chapter_figures import (
    BBOX_OUTSIDE_INK_RATIO_BIN_LABELS,
    _bbox_outside_ink_ratio_bin_counts,
    _bbox_outside_ink_ratio_bin_index,
    export_chapter_distribution_figures,
)
from benchmark_design.report.line_level.chapter_tables import (
    INK_STATE_POSITIVE_INK,
    INK_STATE_ZERO_INK,
    classify_bbox_outside_ink_state,
    write_chapter_tables,
)


def _line(**kwargs: object) -> LineMetricsRow:
    defaults = {
        "image_id": "img",
        "line_id": "0:0",
        "bbox_width_px": 100.0,
        "bbox_height_px": 20.0,
        "aspect_ratio": 5.0,
        "orientation_deg": 3.0,
        "orientation_direction_valid": True,
        "is_ignore": False,
        "is_valid": True,
        "invalid_reason": "",
        "page_orientation": "portrait",
        "block_type": "Txtblock",
        "bbox_outside_ink_ratio": 0.0,
        "bbox_outside_pixel_count": 0,
        "bbox_outside_ink_count": 0,
        "bbox_pixel_count": 100,
    }
    defaults.update(kwargs)
    return LineMetricsRow(**defaults)  # type: ignore[arg-type]


def test_ink_state_classifier() -> None:
    # Empty (bbox \ mask) counts as zero-ink, not a separate "no region" class.
    assert classify_bbox_outside_ink_state(_line()) == INK_STATE_ZERO_INK
    assert (
        classify_bbox_outside_ink_state(_line(bbox_outside_pixel_count=10, bbox_outside_ink_count=0))
        == INK_STATE_ZERO_INK
    )
    assert (
        classify_bbox_outside_ink_state(
            _line(bbox_outside_pixel_count=10, bbox_outside_ink_count=3, bbox_outside_ink_ratio=0.3)
        )
        == INK_STATE_POSITIVE_INK
    )


def test_horizontal_adjacent_scope_rows() -> None:
    pairs = [
        TargetPairRow(
            image_id="img",
            line_id_a="0:0",
            line_id_b="0:1",
            intersection_area=5.0,
            ioa=0.25,
            horizontal_gap_px=0.0,
            height_similarity=1.0,
            vertical_overlap_px=10.0,
            vertical_overlap_ratio=1.0,
            ioa_positive=True,
            horizontal_adjacent=False,
        ),
        TargetPairRow(
            image_id="img2",
            line_id_a="1:0",
            line_id_b="1:1",
            intersection_area=0.0,
            ioa=0.0,
            horizontal_gap_px=12.0,
            height_similarity=0.9,
            vertical_overlap_px=18.0,
            vertical_overlap_ratio=0.9,
            ioa_positive=False,
            horizontal_adjacent=True,
        ),
    ]
    rows = horizontal_adjacent_scope_rows(pairs, valid_line_count=10, page_count=5)
    assert rows[0]["item"] == "水平相邻 line（去重）"
    assert rows[0]["count"] == 2
    assert rows[0]["ratio"] == 0.2
    assert rows[1]["item"] == "涉及页面"
    assert rows[1]["count"] == 1
    assert rows[1]["ratio"] == 0.2
    assert rows[2]["item"] == "水平相邻无序 pair"
    assert rows[2]["count"] == 1
    assert rows[2]["ratio"] == 0.1


def test_bbox_outside_ink_ratio_bins_use_one_percent_intervals() -> None:
    assert list(BBOX_OUTSIDE_INK_RATIO_BIN_LABELS) == [
        "0%–1%",
        "1%–2%",
        "2%–3%",
        "3%–4%",
        "4%–5%",
        "≥5%",
    ]
    assert _bbox_outside_ink_ratio_bin_index(0.005) == 0
    assert _bbox_outside_ink_ratio_bin_index(0.01) == 1
    assert _bbox_outside_ink_ratio_bin_index(0.049) == 4
    assert _bbox_outside_ink_ratio_bin_index(0.05) == 5
    assert _bbox_outside_ink_ratio_bin_index(0.20) == 5

    import numpy as np

    counts = _bbox_outside_ink_ratio_bin_counts(np.array([0.005, 0.015, 0.025, 0.06]))
    assert counts.tolist() == [1, 1, 1, 0, 0, 1]


def test_write_chapter_tables_and_figures(tmp_path: Path) -> None:
    config = LineLevelConfig(input_root=tmp_path, output_root=tmp_path)
    lines = [
        _line(line_id="0:0", orientation_direction_valid=True),
        _line(line_id="0:1", orientation_direction_valid=False, aspect_ratio=1.1),
        _line(
            line_id="0:2",
            bbox_outside_pixel_count=20,
            bbox_outside_ink_count=0,
            bbox_outside_ink_ratio=0.0,
        ),
        _line(
            line_id="0:3",
            bbox_outside_pixel_count=50,
            bbox_outside_ink_count=10,
            bbox_outside_ink_ratio=0.2,
        ),
    ]
    pairs = [
        TargetPairRow(
            image_id="img",
            line_id_a="0:0",
            line_id_b="0:1",
            intersection_area=5.0,
            ioa=0.25,
            horizontal_gap_px=0.0,
            height_similarity=1.0,
            vertical_overlap_px=10.0,
            vertical_overlap_ratio=1.0,
            ioa_positive=True,
            horizontal_adjacent=False,
        ),
        TargetPairRow(
            image_id="img",
            line_id_a="0:2",
            line_id_b="0:3",
            intersection_area=0.0,
            ioa=0.0,
            horizontal_gap_px=12.0,
            height_similarity=0.9,
            vertical_overlap_px=18.0,
            vertical_overlap_ratio=0.9,
            ioa_positive=False,
            horizontal_adjacent=True,
        ),
    ]
    table_paths = write_chapter_tables(lines, pairs, config, tmp_path)
    assert (tmp_path / "tables" / "orientation_validity.csv").is_file()
    assert (tmp_path / "tables" / "spatial_relations.csv").is_file()
    assert (tmp_path / "tables" / "horizontal_adjacent_scope.csv").is_file()
    assert (tmp_path / "tables" / "bbox_outside_ink_natural_states.csv").is_file()
    assert "bbox_outside_ink_calibration_threshold" in table_paths

    plots = export_chapter_distribution_figures(lines, pairs, tmp_path / "plots")
    assert "orientation_abs_distribution" in plots
    assert "bbox_outside_ink_ratio_distribution" in plots
    assert "positive_ioa_distribution" not in plots
    assert "bbox_outside_ink_natural_states" not in plots
    assert plots["orientation_abs_distribution"].is_file()
    assert plots["bbox_outside_ink_ratio_distribution"].is_file()
