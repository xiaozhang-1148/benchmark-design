"""Aggregate expression LaTeX metrics to page-level rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.latex_protocol import (
    STRUCTURE_TYPE_ORDER,
    TAXONOMY_FIELD_KEYS,
    empty_token_category_counts,
)


@dataclass(frozen=True, slots=True)
class PageLatexMetricsRow:
    image_id: str
    ast_tree_count: int
    total_ast_node_count: int
    max_ast_depth: int
    frac_expression_count: int
    sup_expression_count: int
    sub_expression_count: int
    sqrt_expression_count: int
    env_expression_count: int
    bigop_expression_count: int
    accent_expression_count: int
    stackrel_expression_count: int
    textcircled_expression_count: int
    distinct_structure_type_count: int
    structure_combination: str
    total_token_count: int
    distinct_token_count: int
    rare10_token_occurrence_count: int
    rare10_expression_count: int
    has_rare10: bool
    token_category_counts: dict[str, int] = field(default_factory=dict)


def _aggregate_page(image_id: str, rows: list[ExpressionLatexMetricsRow]) -> PageLatexMetricsRow:
    structure_counts = {name: 0 for name in STRUCTURE_TYPE_ORDER}
    category_counts = empty_token_category_counts()
    page_token_counter: Counter[str] = Counter()
    page_structure_present = {name: False for name in STRUCTURE_TYPE_ORDER}
    rare10_token_occurrences = 0
    rare10_expression_count = 0
    total_token_count = 0
    max_ast_depth = 0
    total_ast_node_count = 0

    for row in rows:
        total_token_count += row.token_count
        total_ast_node_count += row.ast_node_count
        max_ast_depth = max(max_ast_depth, row.ast_depth)
        for name in STRUCTURE_TYPE_ORDER:
            if getattr(row, f"has_{name}"):
                structure_counts[name] += 1
                page_structure_present[name] = True
        for key, value in row.token_category_counts.items():
            category_counts[key] = category_counts.get(key, 0) + int(value)
        page_token_counter.update(row.tokens)
        rare10_token_occurrences += row.rare10_token_occurrence_count
        if row.has_rare10:
            rare10_expression_count += 1

    present_types = tuple(name for name in STRUCTURE_TYPE_ORDER if page_structure_present[name])
    return PageLatexMetricsRow(
        image_id=image_id,
        ast_tree_count=len(rows),
        total_ast_node_count=total_ast_node_count,
        max_ast_depth=max_ast_depth,
        frac_expression_count=structure_counts["frac"],
        sup_expression_count=structure_counts["sup"],
        sub_expression_count=structure_counts["sub"],
        sqrt_expression_count=structure_counts["sqrt"],
        env_expression_count=structure_counts["env"],
        bigop_expression_count=structure_counts["bigop"],
        accent_expression_count=structure_counts["accent"],
        stackrel_expression_count=structure_counts["stackrel"],
        textcircled_expression_count=structure_counts["textcircled"],
        distinct_structure_type_count=len(present_types),
        structure_combination="+".join(present_types),
        total_token_count=total_token_count,
        distinct_token_count=len(page_token_counter),
        rare10_token_occurrence_count=rare10_token_occurrences,
        rare10_expression_count=rare10_expression_count,
        has_rare10=rare10_expression_count > 0,
        token_category_counts=category_counts,
    )


def aggregate_page_latex_metrics(
    expression_rows: Sequence[ExpressionLatexMetricsRow],
    *,
    all_image_ids: Sequence[str] | None = None,
) -> list[PageLatexMetricsRow]:
    by_page: dict[str, list[ExpressionLatexMetricsRow]] = defaultdict(list)
    for row in expression_rows:
        if not row.valid_for_latex:
            continue
        by_page[row.image_id].append(row)

    image_ids = list(all_image_ids) if all_image_ids is not None else list(by_page.keys())
    ordered_ids = sorted(set(image_ids) | set(by_page.keys()))
    pages = [_aggregate_page(image_id, by_page.get(image_id, [])) for image_id in ordered_ids]
    return pages


def page_metrics_to_frame(rows: Sequence[PageLatexMetricsRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        payload = asdict(row)
        category_counts = payload.pop("token_category_counts")
        payload.update(category_counts)
        records.append(payload)
    frame = pd.DataFrame.from_records(records)
    preferred = [
        "image_id",
        "ast_tree_count",
        "total_ast_node_count",
        "max_ast_depth",
        *[f"{name}_expression_count" for name in STRUCTURE_TYPE_ORDER],
        "distinct_structure_type_count",
        "structure_combination",
        "total_token_count",
        "distinct_token_count",
        "rare10_token_occurrence_count",
        "rare10_expression_count",
        "has_rare10",
    ]
    columns = [col for col in preferred if col in frame.columns]
    columns.extend(col for col in frame.columns if col not in columns)
    return frame.loc[:, columns]


def read_page_latex_metrics_csv(path: Path) -> list[PageLatexMetricsRow]:
    frame = pd.read_csv(path)
    rows: list[PageLatexMetricsRow] = []
    for record in frame.to_dict(orient="records"):
        category_counts = {key: int(record.get(key, 0) or 0) for key in TAXONOMY_FIELD_KEYS}
        rows.append(
            PageLatexMetricsRow(
                image_id=str(record["image_id"]),
                ast_tree_count=int(record.get("ast_tree_count", record.get("expression_count", 0)) or 0),
                total_ast_node_count=int(
                    record.get("total_ast_node_count", record.get("total_token_count", 0)) or 0
                ),
                max_ast_depth=int(
                    record.get("max_ast_depth", record.get("max_expression_ast_depth", 0)) or 0
                ),
                frac_expression_count=int(record.get("frac_expression_count", 0) or 0),
                sup_expression_count=int(record.get("sup_expression_count", 0) or 0),
                sub_expression_count=int(record.get("sub_expression_count", 0) or 0),
                sqrt_expression_count=int(record.get("sqrt_expression_count", 0) or 0),
                env_expression_count=int(record.get("env_expression_count", 0) or 0),
                bigop_expression_count=int(
                    record.get("bigop_expression_count", record.get("sum_expression_count", 0)) or 0
                ),
                accent_expression_count=int(record.get("accent_expression_count", 0) or 0),
                stackrel_expression_count=int(record.get("stackrel_expression_count", 0) or 0),
                textcircled_expression_count=int(record.get("textcircled_expression_count", 0) or 0),
                distinct_structure_type_count=int(record.get("distinct_structure_type_count", 0) or 0),
                structure_combination=str(record.get("structure_combination", "") or ""),
                total_token_count=int(record.get("total_token_count", 0) or 0),
                distinct_token_count=int(record.get("distinct_token_count", 0) or 0),
                rare10_token_occurrence_count=int(record.get("rare10_token_occurrence_count", 0) or 0),
                rare10_expression_count=int(record.get("rare10_expression_count", 0) or 0),
                has_rare10=bool(record.get("has_rare10", False)),
                token_category_counts=category_counts,
            )
        )
    return rows
