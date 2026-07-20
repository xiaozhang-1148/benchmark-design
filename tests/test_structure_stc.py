"""Tests for Nested Traversal Cost (NTC) and Co-occurrence Breadth Count (CBC)."""

from __future__ import annotations

from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.structure_stc import (
    StructurePath,
    compute_ntc_cbc,
    format_chain_segments,
    format_structure_path,
    is_stc_export_cohort,
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


def _cases_block(*body: str) -> list[str]:
    tokens: list[str] = [r"\begin", "{", "cases", "}"]
    tokens.extend(body)
    tokens.extend([r"\end", "{", "cases", "}"])
    return tokens


def _nested_sup_chain(depth: int) -> list[str]:
    tokens: list[str] = []
    for _ in range(depth):
        tokens.extend(["^", "{"])
    tokens.append("x")
    tokens.extend(["}"] * depth)
    return tokens


def _frac_body(*body: str) -> list[str]:
    tokens: list[str] = [r"\frac", "{"]
    tokens.extend(body)
    tokens.extend(["}", "{", "y", "}"])
    return tokens


def test_ntc_nested_sup_with_parallel_singles() -> None:
    tokens: list[str] = []
    tokens.extend(_nested_sup_chain(5))
    tokens.extend([r"\frac", "{", "a", "}", "{", "b", "}"])
    tokens.extend([r"\sqrt", "{", "x", "}"])
    tokens.extend(["_", "{", "y", "}"])

    result = compute_ntc_cbc(tokens)
    assert result.ntc == 5
    assert result.cbc == 3
    assert result.max_recursive_len == 5
    assert len(result.nested_paths) == 1
    assert result.nested_paths[0].segments == ("Sup", "Sup*")
    assert result.nested_paths[0].cost == 5
    assert {node.stc_type for node in result.single_nodes} == {"Frac", "Sqrt", "Sub"}


def test_ntc_seven_example() -> None:
    env_deep = _cases_block(
        r"\frac",
        "{",
        "^",
        "{",
        r"\sqrt",
        "{",
        "x",
        "}",
        "}",
        "}",
        "{",
        "y",
        "}",
        r"\\",
        "z",
    )
    env_shallow = _cases_block(
        r"\frac",
        "{",
        "^",
        "{",
        "a",
        "}",
        "}",
        "{",
        "b",
        "}",
        r"\\",
        "c",
    )
    tokens = env_deep + env_shallow

    result = compute_ntc_cbc(tokens)
    assert result.ntc == 7
    assert result.cbc == 0

    paths_by_segments = {path.segments: path for path in result.nested_paths}
    assert paths_by_segments[("Env", "Frac", "Sup", "Sqrt")].cost == 4
    assert paths_by_segments[("Env", "Frac", "Sup")].cost == 3


def test_ntc_path_merge_frequency() -> None:
    shallow_env = _cases_block(
        r"\frac",
        "{",
        "^",
        "{",
        "a",
        "}",
        "}",
        "{",
        "b",
        "}",
        r"\\",
        "c",
    )
    tokens = shallow_env + shallow_env

    result = compute_ntc_cbc(tokens)
    assert result.ntc == 6
    assert result.cbc == 0
    assert len(result.nested_paths) == 1
    path = result.nested_paths[0]
    assert path.segments == ("Env", "Frac", "Sup")
    assert path.cost == 3
    assert path.frequency == 2


def test_ntc_drops_subchains() -> None:
    tokens = _frac_body(
        r"\sqrt",
        "{",
        r"\frac",
        "{",
        "^",
        "{",
        "a",
        "}",
        "}",
        "{",
        "b",
        "}",
        "}",
    )
    tokens.extend(_frac_body("c"))
    tokens.extend(_frac_body("d"))
    tokens.extend(_frac_body("e"))
    tokens.extend(["_", "{", "z", "}"])
    tokens.extend(["^", "{", "w", "}"])

    result = compute_ntc_cbc(tokens)
    assert len(result.nested_paths) == 1
    assert result.nested_paths[0].segments == ("Frac", "Sqrt", "Frac", "Sup")
    assert result.nested_paths[0].cost == 4
    assert result.ntc == 4
    assert result.cbc == 5


def test_ntc_same_type_repeat_cost() -> None:
    tokens = _frac_body(
        r"\frac",
        "{",
        r"\frac",
        "{",
        "^",
        "{",
        "a",
        "}",
        "}",
        "{",
        "b",
        "}",
        "}",
        "{",
        "c",
        "}",
    )
    tokens.extend(_frac_body("_", "{", "s", "}"))
    tokens.extend(_frac_body("x"))

    result = compute_ntc_cbc(tokens)
    paths_by_segments = {path.segments: path for path in result.nested_paths}
    assert paths_by_segments[("Frac", "Frac*", "Sup")].cost == 5
    assert paths_by_segments[("Frac", "Sub")].cost == 2
    assert result.ntc == 7
    assert result.cbc == 1
    assert result.single_nodes[0].stc_type == "Frac"
    assert result.single_nodes[0].frequency == 1


def test_is_stc_export_cohort_boundaries() -> None:
    assert is_stc_export_cohort(_feature(structure_type_count=3, ast_depth=4))
    assert is_stc_export_cohort(_feature(structure_type_count=4, ast_depth=5))
    assert not is_stc_export_cohort(_feature(structure_type_count=3, ast_depth=3))
    assert not is_stc_export_cohort(_feature(structure_type_count=2, ast_depth=4))


def test_format_structure_path() -> None:
    path = StructurePath(
        segments=("Env", "Frac", "Sup"),
        raw_repeat_lens=(),
        raw_length=3,
        cost=3,
        frequency=2,
    )
    assert format_chain_segments(path.segments) == "Env -> Frac -> Sup"
    assert format_structure_path(path) == "Env -> Frac -> Sup ×2"
