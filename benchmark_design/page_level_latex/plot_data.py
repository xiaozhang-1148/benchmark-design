"""Plot-data builders for Chapter-6 page-level LaTeX figures (English CSV columns)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.latex_protocol import (
    STRUCTURE_TYPE_ORDER,
    TAXONOMY_CATEGORY_TO_FIELD,
    TOKEN_CATEGORY_ORDER,
    TokenCategory,
)
from benchmark_design.page_level_latex.page_latex_metrics import PageLatexMetricsRow
from benchmark_design.page_level_latex.plot_style import page_ratio


def total_pages(page_rows: Sequence[PageLatexMetricsRow]) -> int:
    return len(page_rows)


# Fixed intervals for Figure 6-1 (inclusive closed ranges; last bin is open-ended ">X").
FIG6_1_BIN_SPECS: dict[str, tuple[tuple[int | None, int | None, str], ...]] = {
    "ast_tree_count": (
        (1, 10, "1–10"),
        (11, 20, "11–20"),
        (21, 30, "21–30"),
        (31, 40, "31–40"),
        (41, 50, "41–50"),
        (51, None, ">50"),
    ),
    "total_ast_node_count": (
        (1, 200, "1–200"),
        (201, 400, "201–400"),
        (401, 600, "401–600"),
        (601, 800, "601–800"),
        (801, 1000, "801–1,000"),
        (1001, None, ">1,000"),
    ),
    "max_ast_depth": (
        (0, 0, "0"),
        (1, 1, "1"),
        (2, 2, "2"),
        (3, 3, "3"),
        (4, 4, "4"),
        (5, 5, "5"),
        (6, None, ">5"),
    ),
}


def _assign_fixed_bin(value: float, specs: tuple[tuple[int | None, int | None, str], ...]) -> str | None:
    v = int(value)
    for start, end, label in specs:
        if end is None:
            if start is not None and v >= int(start):
                return label
        elif start is not None and int(start) <= v <= int(end):
            return label
    return None


def fixed_bin_frame(
    values: np.ndarray,
    *,
    metric: str,
    specs: tuple[tuple[int | None, int | None, str], ...],
    total_pages: int,
) -> pd.DataFrame:
    counts = Counter(_assign_fixed_bin(v, specs) for v in values)
    rows = []
    for start, end, label in specs:
        count = int(counts.get(label, 0))
        rows.append(
            {
                "metric": metric,
                "bin_label": label,
                "bin_start": "" if start is None else int(start),
                "bin_end": "" if end is None else int(end),
                "page_count": count,
                "page_ratio": page_ratio(count, total_pages),
            }
        )
    return pd.DataFrame(rows)


def build_fig6_1_plot_data(page_rows: Sequence[PageLatexMetricsRow]) -> pd.DataFrame:
    total = total_pages(page_rows)
    frames = []
    for metric, values in (
        ("ast_tree_count", np.array([row.ast_tree_count for row in page_rows], dtype=np.float64)),
        ("total_ast_node_count", np.array([row.total_ast_node_count for row in page_rows], dtype=np.float64)),
        ("max_ast_depth", np.array([row.max_ast_depth for row in page_rows], dtype=np.float64)),
    ):
        frames.append(
            fixed_bin_frame(
                values,
                metric=metric,
                specs=FIG6_1_BIN_SPECS[metric],
                total_pages=total,
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_fig6_3_plot_data(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
) -> pd.DataFrame:
    from collections import defaultdict

    total = total_pages(page_rows)
    valid = [row for row in expression_rows if row.valid_for_latex]
    depths_by_page: dict[str, set[int]] = defaultdict(set)
    for row in valid:
        depths_by_page[row.image_id].add(row.ast_depth)
    max_depth_counts = Counter(page.max_ast_depth for page in page_rows)
    records = []
    for depth in range(0, 6):
        coverage = sum(1 for page in page_rows if depth in depths_by_page.get(page.image_id, set()))
        max_count = max_depth_counts.get(depth, 0)
        records.append(
            {
                "ast_depth": depth,
                "coverage_page_count": coverage,
                "coverage_page_ratio": page_ratio(coverage, total),
                "max_depth_page_count": max_count,
                "max_depth_page_ratio": page_ratio(max_count, total),
            }
        )
    return pd.DataFrame(records)


def build_fig6_4_plot_data(page_rows: Sequence[PageLatexMetricsRow]) -> pd.DataFrame:
    total = total_pages(page_rows)
    records = []
    for name in STRUCTURE_TYPE_ORDER:
        count = sum(1 for page in page_rows if getattr(page, f"{name}_expression_count") > 0)
        records.append(
            {
                "data_type": "structure_coverage",
                "category": name,
                "page_count": count,
                "page_ratio": page_ratio(count, total),
            }
        )
    type_counts = Counter(page.distinct_structure_type_count for page in page_rows)
    for value in range(0, 7):
        count = type_counts.get(value, 0)
        records.append(
            {
                "data_type": "structure_type_count",
                "category": str(value),
                "page_count": count,
                "page_ratio": page_ratio(count, total),
            }
        )
    return pd.DataFrame(records)


def _structure_group(count: int) -> str:
    if count >= 4:
        return "at_least_4"
    return str(count)


def build_fig6_5_joint_exact(page_rows: Sequence[PageLatexMetricsRow]) -> pd.DataFrame:
    total = total_pages(page_rows)
    records = []
    for structure_count in range(0, 7):
        for depth in range(0, 6):
            count = sum(
                1
                for page in page_rows
                if page.distinct_structure_type_count == structure_count
                and page.max_ast_depth == depth
            )
            records.append(
                {
                    "structure_type_count": structure_count,
                    "max_ast_depth": depth,
                    "page_count": count,
                    "page_ratio": page_ratio(count, total),
                }
            )
    return pd.DataFrame(records)


def build_fig6_5_joint_grouped(page_rows: Sequence[PageLatexMetricsRow]) -> pd.DataFrame:
    total = total_pages(page_rows)
    records = []
    for group in ("0", "1", "2", "3", "at_least_4"):
        for depth in range(0, 6):
            count = sum(
                1
                for page in page_rows
                if _structure_group(page.distinct_structure_type_count) == group
                and page.max_ast_depth == depth
            )
            records.append(
                {
                    "structure_type_count_group": group,
                    "max_ast_depth": depth,
                    "page_count": count,
                    "page_ratio": page_ratio(count, total),
                }
            )
    return pd.DataFrame(records)


def build_fig6_6_distinct_token_plot_data(page_rows: Sequence[PageLatexMetricsRow]) -> pd.DataFrame:
    total = total_pages(page_rows)
    values = np.array([page.distinct_token_count for page in page_rows], dtype=np.float64)
    specs = (
        (1, 20, "1–20"),
        (21, 40, "21–40"),
        (41, 60, "41–60"),
        (61, 80, "61–80"),
        (81, 100, "81–100"),
        (101, 120, "101–120"),
        (121, None, ">120"),
    )
    return fixed_bin_frame(values, metric="distinct_token_count", specs=specs, total_pages=total)


def build_fig6_6_category_plot_data(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
) -> pd.DataFrame:
    total = total_pages(page_rows)
    valid = [row for row in expression_rows if row.valid_for_latex]
    records = []
    for category in TOKEN_CATEGORY_ORDER:
        if category == TokenCategory.OTHER:
            continue
        field = TAXONOMY_CATEGORY_TO_FIELD[category]
        token_total = sum(row.token_category_counts.get(field, 0) for row in valid)
        page_count = sum(1 for page in page_rows if page.token_category_counts.get(field, 0) > 0)
        records.append(
            {
                "token_category": category.value,
                "token_occurrence_count": token_total,
                "page_count": page_count,
                "page_ratio": page_ratio(page_count, total),
            }
        )
    return pd.DataFrame(records)


def build_fig6_7_plot_data(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    page_rows: Sequence[PageLatexMetricsRow],
    rare_tokens: set[str],
) -> pd.DataFrame:
    """Occurrence distribution of rare-tail tokens on pages that contain them."""
    total = total_pages(page_rows)
    page_occ: Counter[str] = Counter()
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        page_occ[row.image_id] += sum(1 for tok in row.tokens if tok in rare_tokens)
    rare_pages = [(image_id, count) for image_id, count in page_occ.items() if count > 0]
    # Include pages with zero expression coverage? only those with occurrences.
    rare_total = len(rare_pages)
    counts = Counter(count for _, count in rare_pages)
    records = []
    for occurrence in sorted(counts):
        count = counts[occurrence]
        records.append(
            {
                "rare8_occurrence_count": occurrence,
                "page_count": count,
                "page_ratio_among_rare8_pages": page_ratio(count, rare_total),
                "page_ratio_among_all_pages": page_ratio(count, total),
            }
        )
    return pd.DataFrame(records)


def build_fig6_8_plot_data(group_summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for rec in group_summary.itertuples(index=False):
        display = getattr(rec, "group_display", rec.group_name)
        records.append(
            {
                "group_name": rec.group_name,
                "group_display": display,
                "cooccurrence_event_count": int(rec.cooccurrence_event_count),
                "cooccurrence_page_count": int(rec.cooccurrence_page_count),
                "cooccurrence_page_ratio": float(rec.cooccurrence_page_ratio),
            }
        )
    return pd.DataFrame(records)
