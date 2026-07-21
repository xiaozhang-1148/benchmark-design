"""Tests for same-page target-pair IoA and horizontal adjacency."""

from __future__ import annotations

from shapely.geometry import Polygon

from benchmark_design.line_level.layout import (
    compute_target_pairs,
    evaluate_target_pair,
    summarize_target_pairs,
)


def test_unordered_pair_ids_are_canonical() -> None:
    a = Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])
    b = Polygon([(60, 0), (100, 0), (100, 20), (60, 20)])
    forward = evaluate_target_pair(
        image_id="img",
        line_id_a="0:1",
        shape_a=a,
        line_id_b="0:0",
        shape_b=b,
    )
    reverse = evaluate_target_pair(
        image_id="img",
        line_id_a="0:0",
        shape_a=b,
        line_id_b="0:1",
        shape_b=a,
    )
    assert forward.line_id_a == "0:0"
    assert forward.line_id_b == "0:1"
    assert reverse.line_id_a == "0:0"
    assert reverse.line_id_b == "0:1"
    assert forward.horizontal_gap_px == reverse.horizontal_gap_px == 20.0


def test_ioa_positive_uses_polygon_intersection_not_empty_touch() -> None:
    a = Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])
    # Only boundary touch along x=40: area intersection must be 0.
    b = Polygon([(40, 0), (80, 0), (80, 20), (40, 20)])
    row = evaluate_target_pair(
        image_id="img",
        line_id_a="0:0",
        shape_a=a,
        line_id_b="0:1",
        shape_b=b,
    )
    assert row.intersection_area == 0.0
    assert row.ioa == 0.0
    assert row.ioa_positive is False

    overlapping = Polygon([(20, 0), (60, 0), (60, 20), (20, 20)])
    positive = evaluate_target_pair(
        image_id="img",
        line_id_a="0:0",
        shape_a=a,
        line_id_b="0:2",
        shape_b=overlapping,
    )
    assert positive.intersection_area > 0.0
    assert positive.ioa > 0.0
    assert positive.ioa_positive is True
    assert positive.horizontal_adjacent is False


def test_example_height_similar_but_vertical_overlap_insufficient() -> None:
    # A:[100,140] h=40 ; B:[120,170] h=50 ; H_overlap=20 ; S_h=0.8 ; R_v=0.5
    a = Polygon([(0, 100), (30, 100), (30, 140), (0, 140)])
    b = Polygon([(50, 120), (80, 120), (80, 170), (50, 170)])
    row = evaluate_target_pair(
        image_id="img",
        line_id_a="0:0",
        shape_a=a,
        line_id_b="0:1",
        shape_b=b,
    )
    assert abs(row.height_similarity - 0.8) < 1e-9
    assert abs(row.vertical_overlap_px - 20.0) < 1e-9
    assert abs(row.vertical_overlap_ratio - 0.5) < 1e-9
    assert row.ioa_positive is False
    assert row.horizontal_gap_px == 20.0
    assert row.horizontal_adjacent is False


def test_horizontal_adjacent_requires_all_thresholds() -> None:
    a = Polygon([(0, 100), (30, 100), (30, 140), (0, 140)])
    b = Polygon([(50, 105), (80, 105), (80, 145), (50, 145)])
    row = evaluate_target_pair(
        image_id="img",
        line_id_a="0:0",
        shape_a=a,
        line_id_b="0:1",
        shape_b=b,
    )
    assert row.ioa_positive is False
    assert row.horizontal_gap_px == 20.0
    assert abs(row.height_similarity - 1.0) < 1e-9
    assert abs(row.vertical_overlap_ratio - 35.0 / 40.0) < 1e-9
    assert row.horizontal_adjacent is True


def test_horizontal_overlap_blocks_adjacent_even_if_ioa_zero() -> None:
    # Vertically stacked: no polygon IoA needed; x ranges fully overlap.
    a = Polygon([(0, 0), (40, 0), (40, 20), (0, 20)])
    b = Polygon([(5, 40), (35, 40), (35, 60), (5, 60)])
    row = evaluate_target_pair(
        image_id="img",
        line_id_a="0:0",
        shape_a=a,
        line_id_b="0:1",
        shape_b=b,
    )
    assert row.ioa_positive is False
    assert row.horizontal_adjacent is False


def test_compute_target_pairs_enumerates_unique_unordered_once() -> None:
    shapes = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        Polygon([(40, 0), (50, 0), (50, 10), (40, 10)]),
    ]
    pairs = compute_target_pairs(
        image_id="img",
        line_ids=["0:0", "0:1", "0:2"],
        shapes=shapes,
    )
    assert len(pairs) == 3
    keys = {(row.line_id_a, row.line_id_b) for row in pairs}
    assert keys == {("0:0", "0:1"), ("0:0", "0:2"), ("0:1", "0:2")}
    summary = summarize_target_pairs(pairs)
    assert summary["pair_count"] == 3
    assert summary["ioa_positive_pair_count"] == 0
    assert summary["horizontal_adjacent_pair_count"] == 3


def test_compute_target_pairs_skips_vertically_disjoint_bands() -> None:
    # Same x band, stacked vertically with a gap → not nearby candidates.
    shapes = [
        Polygon([(0, 0), (40, 0), (40, 20), (0, 20)]),
        Polygon([(0, 80), (40, 80), (40, 100), (0, 100)]),
        Polygon([(0, 160), (40, 160), (40, 180), (0, 180)]),
    ]
    pairs = compute_target_pairs(
        image_id="img",
        line_ids=["0:0", "0:1", "0:2"],
        shapes=shapes,
    )
    assert pairs == []


def test_compute_target_pairs_skips_far_horizontal_even_if_y_overlaps() -> None:
    a = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
    b = Polygon([(200, 0), (210, 0), (210, 20), (200, 20)])  # gap=190 > 50
    pairs = compute_target_pairs(
        image_id="img",
        line_ids=["0:0", "0:1"],
        shapes=[a, b],
    )
    assert pairs == []


def test_gap_above_50_px_is_not_adjacent() -> None:
    a = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
    b = Polygon([(70, 0), (80, 0), (80, 20), (70, 20)])
    row = evaluate_target_pair(
        image_id="img",
        line_id_a="0:0",
        shape_a=a,
        line_id_b="0:1",
        shape_b=b,
    )
    assert row.horizontal_gap_px == 60.0
    assert row.horizontal_adjacent is False
