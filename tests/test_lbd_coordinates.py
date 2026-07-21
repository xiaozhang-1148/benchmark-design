"""Tests for L/B/D coordinate binning."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.lbd_coordinates import (
    STRUCTURAL_DIFFICULTY_TIERS,
    assign_b_bin,
    assign_d_bin,
    assign_expression_lbd_coordinate,
    assign_l_bin,
    classify_lbd,
    compute_lbd_coordinate_metrics,
    difficulty_region,
    iter_all_position_ids,
    lbd_structure_types_present,
    validate_lbd_coordinates,
)
from benchmark_design.report.lbd_coordinate_table import (
    write_expression_lbd_coordinate_counts_csv,
    write_expression_structural_difficulty_counts_csv,
)

EXPECTED_POSITION_STRUCTURAL_DIFFICULTY: dict[str, str] = {
    "L0B0D0": "L1",
    "L0B0D1": "L2",
    "L0B0D2": "L3",
    "L0B1D0": "L2",
    "L0B1D1": "L2",
    "L0B1D2": "L3",
    "L0B2D0": "L2",
    "L0B2D1": "L3",
    "L0B2D2": "L3",
    "L1B0D0": "L2",
    "L1B0D1": "L2",
    "L1B0D2": "L3",
    "L1B1D0": "L2",
    "L1B1D1": "L3",
    "L1B1D2": "L3",
    "L1B2D0": "L3",
    "L1B2D1": "L3",
    "L1B2D2": "L4",
    "L2B0D0": "L3",
    "L2B0D1": "L3",
    "L2B0D2": "L4",
    "L2B1D0": "L3",
    "L2B1D1": "L3",
    "L2B1D2": "L4",
    "L2B2D0": "L3",
    "L2B2D1": "L4",
    "L2B2D2": "L4",
}


def _feature(
    *,
    expression_id: str,
    tokens: tuple[str, ...],
    ast_depth: int,
    normalized_latex: str = "x",
) -> ExpressionFeatures:
    return ExpressionFeatures(
        expression_id=expression_id,
        dataset="ours",
        source_file="sample.json",
        line_id="0:0",
        normalized_latex=normalized_latex,
        token_sequence=tokens,
        token_length=len(tokens),
        length_bin="short",
        is_duplicate=False,
        duplicate_group_id=0,
        duplicate_count=1,
        token_type_counts={},
        has_rare_1=False,
        has_rare_5=False,
        has_rare_10=False,
        structure_types=(),
        structure_type_count=0,
        structure_max_depths={},
        ast_depth=ast_depth,
        mean_token_nested_level=0.0,
        parse_status="ok",
    )


def test_lbd_bin_boundaries() -> None:
    assert assign_l_bin(20) == "L0"
    assert assign_l_bin(21) == "L1"
    assert assign_l_bin(40) == "L1"
    assert assign_l_bin(41) == "L2"
    assert assign_b_bin(0) == "B0"
    assert assign_b_bin(1) == "B0"
    assert assign_b_bin(2) == "B1"
    assert assign_b_bin(3) == "B2"
    assert assign_d_bin(0) == "D0"
    assert assign_d_bin(1) == "D0"
    assert assign_d_bin(2) == "D1"
    assert assign_d_bin(3) == "D2"


def test_lbd_structure_types_use_fixed_table() -> None:
    tokens = ("x", "^", "{", "2", "}", "_", "{", "3", "}", r"\frac", "{", "a", "}", "{", "b", "}")
    assert lbd_structure_types_present(tokens) == ("frac", "sub", "sup")


def test_classify_lbd_all_27_positions() -> None:
    for position_id, l_bin, b_bin, d_bin in iter_all_position_ids():
        expected = EXPECTED_POSITION_STRUCTURAL_DIFFICULTY[position_id]
        assert classify_lbd(l_bin, b_bin, d_bin) == expected
        assert difficulty_region(l_bin=l_bin, b_bin=b_bin, d_bin=d_bin) == expected


def test_classify_lbd_extreme_high_requires_nonzero_l_and_d() -> None:
    assert classify_lbd("L2", "B2", "D0") == "L3"
    assert classify_lbd("L0", "B2", "D2") == "L3"


def test_compute_lbd_metrics_has_27_positions_and_zero_rows() -> None:
    features = [
        _feature(expression_id="plain", tokens=("x",), ast_depth=0),
        _feature(
            expression_id="frac",
            tokens=(r"\frac", "{", "a", "}", "{", "b", "}"),
            ast_depth=2,
        ),
    ]
    metrics = compute_lbd_coordinate_metrics(features)
    assert len(metrics.position_counts) == 27
    assert len(iter_all_position_ids()) == 27
    assert sum(row.count for row in metrics.position_counts) == 2
    assert len(metrics.structural_difficulty_counts) == 4
    zero_rows = [row for row in metrics.position_counts if row.count == 0]
    assert len(zero_rows) == 25
    assert validate_lbd_coordinates(metrics) == []


def test_lbd_csv_export_format(tmp_path: Path) -> None:
    features = [_feature(expression_id="plain", tokens=("x",), ast_depth=0)]
    write_expression_lbd_coordinate_counts_csv(features, tmp_path / "expression_lbd_coordinate_counts.csv")
    with (tmp_path / "expression_lbd_coordinate_counts.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 27
    l0b0d0 = next(row for row in rows if row["position_id"] == "L0B0D0")
    assert l0b0d0["count"] == "1"
    assert l0b0d0["ratio"] == "100.00%"
    assert l0b0d0["structural_difficulty"] == "L1"


def test_structural_difficulty_csv_export_format(tmp_path: Path) -> None:
    features = [
        _feature(expression_id="plain", tokens=("x",), ast_depth=0),
        _feature(expression_id="frac", tokens=(r"\frac", "{", "a", "}", "{", "b", "}"), ast_depth=2),
    ]
    write_expression_structural_difficulty_counts_csv(
        features,
        tmp_path / "expression_structural_difficulty_counts.csv",
    )
    with (tmp_path / "expression_structural_difficulty_counts.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert [row["structural_difficulty"] for row in rows] == list(STRUCTURAL_DIFFICULTY_TIERS)
    l1 = next(row for row in rows if row["structural_difficulty"] == "L1")
    assert l1["count"] == "1"


def test_lbd_quality_checks_fail_on_invalid_depth_without_structure() -> None:
    metrics = compute_lbd_coordinate_metrics(
        [_feature(expression_id="bad", tokens=("x",), ast_depth=2)]
    )
    violations = validate_lbd_coordinates(metrics)
    assert any("structure_type_count=0" in item for item in violations)


def test_lbd_figure_export_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from benchmark_design.report.export_figures import write_expression_lbd_coordinate_distribution

    features = [
        _feature(expression_id="plain", tokens=("x",), ast_depth=0),
        _feature(expression_id="frac", tokens=(r"\frac", "{", "a", "}", "{", "b", "}"), ast_depth=1),
    ]
    output_path = tmp_path / "expression_lbd_coordinate_distribution.png"
    write_expression_lbd_coordinate_distribution(features, output_path)
    assert output_path.exists()
