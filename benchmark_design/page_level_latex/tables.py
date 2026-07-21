"""Paper tables and rare-10 detail exports for page-level LaTeX analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.latex_protocol import (
    STRUCTURE_TYPE_ORDER,
    TAXONOMY_CATEGORY_TO_FIELD,
    TOKEN_CATEGORY_ORDER,
    RARE10_THRESHOLD,
    rare10_token_set,
)
from benchmark_design.page_level_latex.page_latex_metrics import PageLatexMetricsRow


def _quantile_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": np.nan, "median": np.nan, "max": np.nan}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def write_protocol_audit(audit: dict[str, int | float], output_path: Path) -> Path:
    frame = pd.DataFrame([{"metric": key, "value": value} for key, value in audit.items()])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_ast_page_summary(page_rows: Sequence[PageLatexMetricsRow], output_path: Path) -> Path:
    trees = np.array([row.ast_tree_count for row in page_rows], dtype=np.float64)
    nodes = np.array([row.total_ast_node_count for row in page_rows], dtype=np.float64)
    depths = np.array([row.max_ast_depth for row in page_rows], dtype=np.float64)
    tree_stats = _quantile_stats(trees)
    node_stats = _quantile_stats(nodes)
    depth_stats = _quantile_stats(depths)
    frame = pd.DataFrame(
        [
            {
                "metric": "ast_tree_count",
                "total": int(trees.sum()) if trees.size else 0,
                "mean": tree_stats["mean"],
                "median": tree_stats["median"],
                "max": tree_stats["max"],
            },
            {
                "metric": "total_ast_node_count",
                "total": int(nodes.sum()) if nodes.size else 0,
                "mean": node_stats["mean"],
                "median": node_stats["median"],
                "max": node_stats["max"],
            },
            {
                "metric": "max_ast_depth",
                "total": "",
                "mean": depth_stats["mean"],
                "median": depth_stats["median"],
                "max": depth_stats["max"],
            },
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_ast_depth_coverage(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    output_path: Path,
) -> Path:
    valid = [row for row in expression_rows if row.valid_for_latex]
    total_expr = len(valid)
    total_pages = len(page_rows)
    expr_counts = Counter(row.ast_depth for row in valid if row.ast_depth <= 5)
    max_depth_page_counts = Counter(page.max_ast_depth for page in page_rows)
    depths_by_page: dict[str, set[int]] = defaultdict(set)
    for row in valid:
        depths_by_page[row.image_id].add(row.ast_depth)
    records = []
    for depth in range(0, 6):
        expr_count = expr_counts.get(depth, 0)
        page_cover = sum(1 for page in page_rows if depth in depths_by_page.get(page.image_id, set()))
        records.append(
            {
                "ast_depth": depth,
                "expression_count": expr_count,
                "expression_ratio": expr_count / total_expr if total_expr else 0.0,
                "page_count": page_cover,
                "page_ratio": page_cover / total_pages if total_pages else 0.0,
                "pages_with_max_depth": max_depth_page_counts.get(depth, 0),
            }
        )
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_structure_coverage(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    output_path: Path,
) -> Path:
    valid = [row for row in expression_rows if row.valid_for_latex]
    total_expr = len(valid)
    total_pages = len(page_rows)
    records = []
    for name in STRUCTURE_TYPE_ORDER:
        expr_count = sum(1 for row in valid if getattr(row, f"has_{name}"))
        page_cover = sum(1 for page in page_rows if getattr(page, f"{name}_expression_count") > 0)
        records.append(
            {
                "structure_type": name,
                "expression_count": expr_count,
                "expression_ratio": expr_count / total_expr if total_expr else 0.0,
                "page_count": page_cover,
                "page_ratio": page_cover / total_pages if total_pages else 0.0,
            }
        )
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_structure_type_count(page_rows: Sequence[PageLatexMetricsRow], output_path: Path) -> Path:
    total_pages = len(page_rows)
    counts = Counter(page.distinct_structure_type_count for page in page_rows)
    frame = pd.DataFrame(
        [
            {
                "distinct_structure_type_count": value,
                "page_count": counts.get(value, 0),
                "page_ratio": counts.get(value, 0) / total_pages if total_pages else 0.0,
            }
            for value in range(0, 7)
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_structure_combinations(page_rows: Sequence[PageLatexMetricsRow], output_path: Path) -> Path:
    total_pages = len(page_rows)
    counts = Counter(page.structure_combination for page in page_rows)
    frame = pd.DataFrame(
        [
            {
                "structure_combination": key or "(none)",
                "page_count": count,
                "page_ratio": count / total_pages if total_pages else 0.0,
            }
            for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_token_category_coverage(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    output_path: Path,
) -> Path:
    valid = [row for row in expression_rows if row.valid_for_latex]
    total_tokens = sum(row.token_count for row in valid)
    total_pages = len(page_rows)
    records = []
    for category in TOKEN_CATEGORY_ORDER:
        field = TAXONOMY_CATEGORY_TO_FIELD[category]
        token_total = sum(row.token_category_counts.get(field, 0) for row in valid)
        page_cover = sum(1 for page in page_rows if page.token_category_counts.get(field, 0) > 0)
        records.append(
            {
                "token_category": category.value,
                "token_occurrence_count": token_total,
                "token_count": token_total,
                "token_ratio": token_total / total_tokens if total_tokens else 0.0,
                "page_count": page_cover,
                "page_ratio": page_cover / total_pages if total_pages else 0.0,
            }
        )
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_rare10_summary(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    token_counter: Counter[str],
    output_path: Path,
) -> Path:
    rare_tokens = rare10_token_set(token_counter, threshold=RARE10_THRESHOLD)
    valid = [row for row in expression_rows if row.valid_for_latex]
    total_pages = len(page_rows)
    frame = pd.DataFrame(
        [
            {
                "rare_vocab_count": len(rare_tokens),
                "token_occurrence_count": sum(row.rare10_token_occurrence_count for row in valid),
                "expression_count": sum(1 for row in valid if row.has_rare10),
                "page_count": sum(1 for page in page_rows if page.has_rare10),
                "page_ratio": (
                    sum(1 for page in page_rows if page.has_rare10) / total_pages if total_pages else 0.0
                ),
            }
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_rare10_token_detail(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    token_counter: Counter[str],
    output_path: Path,
    *,
    total_pages: int | None = None,
) -> Path:
    rare_tokens = rare10_token_set(token_counter, threshold=RARE10_THRESHOLD)
    expr_cover: dict[str, set[str]] = defaultdict(set)
    page_cover: dict[str, set[str]] = defaultdict(set)
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        row_key = f"{row.image_id}:{row.line_id}"
        for token in row.tokens:
            if token in rare_tokens:
                expr_cover[token].add(row_key)
                page_cover[token].add(row.image_id)
    if total_pages is None:
        total_pages = len({row.image_id for row in expression_rows if row.valid_for_latex})
    frame = pd.DataFrame(
        [
            {
                "token": token,
                "corpus_frequency": token_counter[token],
                "expression_count": len(expr_cover[token]),
                "page_count": len(page_cover[token]),
                "page_ratio": len(page_cover[token]) / total_pages if total_pages else 0.0,
            }
            for token in sorted(rare_tokens, key=lambda item: (token_counter[item], item))
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_structure_depth_joint_distribution(
    page_rows: Sequence[PageLatexMetricsRow],
    output_path: Path,
) -> Path:
    from benchmark_design.page_level_latex.plot_data import build_fig6_5_joint_exact

    frame = build_fig6_5_joint_exact(page_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_distinct_token_distribution(page_rows: Sequence[PageLatexMetricsRow], output_path: Path) -> Path:
    from benchmark_design.page_level_latex.plot_data import build_fig6_6_distinct_token_plot_data

    frame = build_fig6_6_distinct_token_plot_data(page_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_distinct_token_summary(page_rows: Sequence[PageLatexMetricsRow], output_path: Path) -> Path:
    values = np.array([page.distinct_token_count for page in page_rows], dtype=np.float64)
    frame = pd.DataFrame(
        [
            {"metric": "min", "value": float(values.min()) if values.size else np.nan},
            {"metric": "mean", "value": float(values.mean()) if values.size else np.nan},
            {"metric": "max", "value": float(values.max()) if values.size else np.nan},
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_rare10_occurrence_distribution(page_rows: Sequence[PageLatexMetricsRow], output_path: Path) -> Path:
    """Legacy Rare-10 (freq ≤ 10) page occurrence distribution from page metrics."""
    from collections import Counter

    total = len(page_rows)
    rare_pages = [page for page in page_rows if page.has_rare10]
    rare_total = len(rare_pages)
    counts = Counter(page.rare10_token_occurrence_count for page in rare_pages)
    frame = pd.DataFrame(
        [
            {
                "rare10_occurrence_count": occurrence,
                "page_count": count,
                "page_ratio_among_rare10_pages": count / rare_total if rare_total else 0.0,
                "page_ratio_among_all_pages": count / total if total else 0.0,
            }
            for occurrence, count in sorted(counts.items())
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def write_rare8_occurrence_distribution(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    rare_tokens: set[str],
    output_path: Path,
) -> Path:
    from benchmark_design.page_level_latex.plot_data import build_fig6_7_plot_data

    frame = build_fig6_7_plot_data(expression_rows, page_rows, rare_tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
