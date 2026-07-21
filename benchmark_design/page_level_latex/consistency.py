"""Chapter-5 consistency checks and page aggregation invariants."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.latex_protocol import (
    STRUCTURE_TYPE_ORDER,
    ast_depth_field_key,
    length_bin_for_token_count,
)
from benchmark_design.page_level_latex.page_latex_metrics import PageLatexMetricsRow

# Expected Chapter-5 corpus statistics for the current benchmark release.
CHAPTER5_EXPECTED: dict[str, float | int] = {
    "expression_count": 152_012,
    "token_count": 3_550_614,
    "vocab_size": 1_004,
    "parse_ok_ratio": 1.0,
    "length_1_10": 34_891,
    "length_11_20": 47_976,
    "length_21_40": 49_066,
    "length_41_80": 17_699,
    "length_gt80": 2_380,
    "ast_depth_0": 53_052,
    "ast_depth_1": 71_972,
    "ast_depth_2": 23_834,
    "ast_depth_3": 3_049,
    "ast_depth_4": 102,
    "ast_depth_5": 3,
    "page_count": 9_911,
}


@dataclass(frozen=True, slots=True)
class ConsistencyCheckResult:
    passed: bool
    chapter5_frame: pd.DataFrame
    invariant_errors: tuple[str, ...]


def _valid_rows(rows: Sequence[ExpressionLatexMetricsRow]) -> list[ExpressionLatexMetricsRow]:
    return [row for row in rows if row.valid_for_latex]


def compute_chapter5_observed(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    *,
    vocab_size: int,
) -> dict[str, float | int]:
    valid = _valid_rows(expression_rows)
    length_counts = Counter()
    for row in valid:
        _, key = length_bin_for_token_count(row.token_count)
        length_counts[key] += 1
    depth_counts = Counter(ast_depth_field_key(row.ast_depth) for row in valid)
    parse_ok = sum(1 for row in valid if row.parse_ok)
    return {
        "expression_count": len(valid),
        "token_count": sum(row.token_count for row in valid),
        "vocab_size": vocab_size,
        "parse_ok_ratio": (parse_ok / len(valid)) if valid else 0.0,
        "length_1_10": length_counts.get("length_1_10", 0),
        "length_11_20": length_counts.get("length_11_20", 0),
        "length_21_40": length_counts.get("length_21_40", 0),
        "length_41_80": length_counts.get("length_41_80", 0),
        "length_gt80": length_counts.get("length_gt80", 0),
        "ast_depth_0": depth_counts.get("ast_depth_0", 0),
        "ast_depth_1": depth_counts.get("ast_depth_1", 0),
        "ast_depth_2": depth_counts.get("ast_depth_2", 0),
        "ast_depth_3": depth_counts.get("ast_depth_3", 0),
        "ast_depth_4": depth_counts.get("ast_depth_4", 0),
        "ast_depth_5": depth_counts.get("ast_depth_5", 0),
        "page_count": len({row.image_id for row in valid}),
    }


def build_chapter5_consistency_frame(observed: dict[str, float | int]) -> pd.DataFrame:
    rows = []
    for metric, expected in CHAPTER5_EXPECTED.items():
        actual = observed.get(metric, np.nan)
        if isinstance(expected, float):
            match = abs(float(actual) - float(expected)) < 1e-9
        else:
            match = int(actual) == int(expected)
        rows.append(
            {
                "metric": metric,
                "chapter5_expected": expected,
                "observed": actual,
                "match": bool(match),
            }
        )
    return pd.DataFrame(rows)


def check_page_invariants(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
) -> list[str]:
    errors: list[str] = []
    valid = _valid_rows(expression_rows)
    if len(page_rows) != CHAPTER5_EXPECTED["page_count"]:
        errors.append(
            f"page_count={len(page_rows)} != expected {CHAPTER5_EXPECTED['page_count']}"
        )

    if sum(row.ast_tree_count for row in page_rows) != len(valid):
        errors.append("sum(page.ast_tree_count) != valid_expression_count")
    if sum(row.total_ast_node_count for row in page_rows) != sum(row.ast_node_count for row in valid):
        errors.append("sum(page.total_ast_node_count) != sum(expression.ast_node_count)")

    for page in page_rows:
        if not (0 <= page.distinct_structure_type_count <= 6):
            errors.append(f"{page.image_id}: distinct_structure_type_count out of range")
            break
        if page.ast_tree_count == 0:
            if page.max_ast_depth != 0 or page.total_ast_node_count != 0:
                errors.append(f"{page.image_id}: empty page has non-zero AST metrics")
                break
        else:
            page_exprs = [row for row in valid if row.image_id == page.image_id]
            if page.max_ast_depth != max(row.ast_depth for row in page_exprs):
                errors.append(f"{page.image_id}: max_ast_depth inconsistent with expressions")
                break

    page_count = len(page_rows)
    for name in STRUCTURE_TYPE_ORDER:
        covered = sum(1 for page in page_rows if getattr(page, f"{name}_expression_count") > 0)
        if covered > page_count:
            errors.append(f"{name} page coverage exceeds page_count")

    if len({row.image_id for row in page_rows}) != len(page_rows):
        errors.append("duplicate image_id in page_metrics")

    depth_hist = Counter(page.max_ast_depth for page in page_rows)
    if sum(depth_hist.values()) != page_count:
        errors.append("max AST depth pages do not sum to page_count")

    structure_type_sum = sum(
        Counter(page.distinct_structure_type_count for page in page_rows).get(v, 0)
        for v in range(0, 7)
    )
    if structure_type_sum != page_count:
        errors.append("structure type count pages do not sum to page_count")

    joint_sum = sum(
        1
        for page in page_rows
        if 0 <= page.distinct_structure_type_count <= 6 and 0 <= page.max_ast_depth <= 5
    )
    if joint_sum != page_count and all(page.max_ast_depth <= 5 for page in page_rows):
        errors.append("structure-depth joint cells do not sum to page_count")

    return errors


def run_consistency_checks(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    *,
    vocab_size: int,
) -> ConsistencyCheckResult:
    observed = compute_chapter5_observed(expression_rows, vocab_size=vocab_size)
    chapter5_frame = build_chapter5_consistency_frame(observed)
    invariant_errors = tuple(check_page_invariants(expression_rows, page_rows))
    chapter5_ok = bool(chapter5_frame["match"].all())
    return ConsistencyCheckResult(
        passed=chapter5_ok and not invariant_errors,
        chapter5_frame=chapter5_frame,
        invariant_errors=invariant_errors,
    )
