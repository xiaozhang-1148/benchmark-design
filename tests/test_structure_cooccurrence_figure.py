"""Tests for structure count vs AST depth heatmap."""

from __future__ import annotations

from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.report.export_figures import (
    _ast_depth_column_index,
    _structure_count_row_index,
    _structure_count_vs_ast_depth_matrix,
)


def _feature(*, structure_type_count: int, ast_depth: int) -> ExpressionFeatures:
    return ExpressionFeatures(
        expression_id=f"e-{structure_type_count}-{ast_depth}",
        dataset="ours",
        source_file="sample.json",
        line_id="0:0",
        normalized_latex="x",
        token_sequence=("x",),
        token_length=1,
        length_bin="short",
        is_duplicate=False,
        duplicate_group_id=0,
        duplicate_count=1,
        token_type_counts={},
        has_rare_1=False,
        has_rare_5=False,
        has_rare_10=False,
        structure_types=(),
        structure_type_count=structure_type_count,
        structure_max_depths={},
        ast_depth=ast_depth,
        mean_token_nested_level=0.0,
        parse_status="ok",
    )


def test_structure_count_row_index_excludes_zero_and_buckets_ge_four() -> None:
    assert _structure_count_row_index(0) is None
    assert _structure_count_row_index(1) == 0
    assert _structure_count_row_index(3) == 2
    assert _structure_count_row_index(4) == 3
    assert _structure_count_row_index(6) == 3


def test_ast_depth_column_index_excludes_zero_and_caps_at_five() -> None:
    assert _ast_depth_column_index(0) is None
    assert _ast_depth_column_index(1) == 0
    assert _ast_depth_column_index(5) == 4
    assert _ast_depth_column_index(8) == 4


def test_structure_count_vs_ast_depth_matrix() -> None:
    features = [
        _feature(structure_type_count=0, ast_depth=0),
        _feature(structure_type_count=1, ast_depth=1),
        _feature(structure_type_count=2, ast_depth=2),
        _feature(structure_type_count=3, ast_depth=3),
        _feature(structure_type_count=4, ast_depth=4),
        _feature(structure_type_count=5, ast_depth=7),
    ]
    matrix = _structure_count_vs_ast_depth_matrix(features)
    assert matrix.shape == (4, 5)
    assert matrix.sum() == 5
    assert matrix[0, 0] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 1
    assert matrix[3, 3] == 1
    assert matrix[3, 4] == 1
