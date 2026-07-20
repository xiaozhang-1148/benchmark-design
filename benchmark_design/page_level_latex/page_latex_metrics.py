"""Aggregate expression LaTeX metrics to page-level rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import pandas as pd

from benchmark_design.page_level_latex.expression_latex_metrics import ExpressionLatexMetricsRow
from benchmark_design.page_level_latex.latex_protocol import (
    AST_DEPTH_FIELD_KEYS,
    LENGTH_BIN_FIELD_KEYS,
    STRUCTURE_TYPE_ORDER,
    TAXONOMY_FIELD_KEYS,
    ast_depth_field_key,
    empty_token_category_counts,
)


@dataclass(frozen=True, slots=True)
class PageLatexMetricsRow:
    image_id: str
    expression_count: int
    total_token_count: int
    max_expression_token_count: int
    length_1_10_count: int
    length_11_20_count: int
    length_21_40_count: int
    length_41_80_count: int
    length_gt80_count: int
    ast_depth_0_count: int
    ast_depth_1_count: int
    ast_depth_2_count: int
    ast_depth_3_count: int
    ast_depth_4_count: int
    ast_depth_5_count: int
    ast_depth_gt5_count: int
    max_expression_ast_depth: int
    frac_expression_count: int
    sup_expression_count: int
    sub_expression_count: int
    sqrt_expression_count: int
    sum_expression_count: int
    env_expression_count: int
    distinct_structure_type_count: int
    structure_combination: str
    distinct_token_count: int
    rare10_token_occurrence_count: int
    rare10_expression_count: int
    has_rare10: bool
    token_category_counts: dict[str, int] = field(default_factory=dict)


def _aggregate_page(image_id: str, rows: list[ExpressionLatexMetricsRow]) -> PageLatexMetricsRow:
    length_counts = {key: 0 for key in LENGTH_BIN_FIELD_KEYS}
    depth_counts = {key: 0 for key in AST_DEPTH_FIELD_KEYS}
    structure_counts = {name: 0 for name in STRUCTURE_TYPE_ORDER}
    category_counts = empty_token_category_counts()
    page_token_counter: Counter[str] = Counter()
    page_structure_present = {name: False for name in STRUCTURE_TYPE_ORDER}
    rare10_token_occurrences = 0
    rare10_expression_count = 0
    max_token_count = 0
    max_ast_depth = 0

    for row in rows:
        length_counts[row.length_bin_key] = length_counts.get(row.length_bin_key, 0) + 1
        depth_key = ast_depth_field_key(row.ast_depth)
        depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
        max_token_count = max(max_token_count, row.token_count)
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
        expression_count=len(rows),
        total_token_count=sum(row.token_count for row in rows),
        max_expression_token_count=max_token_count,
        length_1_10_count=length_counts["length_1_10"],
        length_11_20_count=length_counts["length_11_20"],
        length_21_40_count=length_counts["length_21_40"],
        length_41_80_count=length_counts["length_41_80"],
        length_gt80_count=length_counts["length_gt80"],
        ast_depth_0_count=depth_counts["ast_depth_0"],
        ast_depth_1_count=depth_counts["ast_depth_1"],
        ast_depth_2_count=depth_counts["ast_depth_2"],
        ast_depth_3_count=depth_counts["ast_depth_3"],
        ast_depth_4_count=depth_counts["ast_depth_4"],
        ast_depth_5_count=depth_counts["ast_depth_5"],
        ast_depth_gt5_count=depth_counts["ast_depth_gt5"],
        max_expression_ast_depth=max_ast_depth,
        frac_expression_count=structure_counts["frac"],
        sup_expression_count=structure_counts["sup"],
        sub_expression_count=structure_counts["sub"],
        sqrt_expression_count=structure_counts["sqrt"],
        sum_expression_count=structure_counts["sum"],
        env_expression_count=structure_counts["env"],
        distinct_structure_type_count=len(present_types),
        structure_combination="+".join(present_types),
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
    # Keep order deterministic and include pages with zero valid expressions.
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
        "expression_count",
        "total_token_count",
        "max_expression_token_count",
        *[f"{key}_count" for key in LENGTH_BIN_FIELD_KEYS],
        *[f"{key}_count" for key in AST_DEPTH_FIELD_KEYS],
        "max_expression_ast_depth",
        *[f"{name}_expression_count" for name in STRUCTURE_TYPE_ORDER],
        "distinct_structure_type_count",
        "structure_combination",
        *TAXONOMY_FIELD_KEYS,
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
                expression_count=int(record.get("expression_count", 0) or 0),
                total_token_count=int(record.get("total_token_count", 0) or 0),
                max_expression_token_count=int(record.get("max_expression_token_count", 0) or 0),
                length_1_10_count=int(record.get("length_1_10_count", 0) or 0),
                length_11_20_count=int(record.get("length_11_20_count", 0) or 0),
                length_21_40_count=int(record.get("length_21_40_count", 0) or 0),
                length_41_80_count=int(record.get("length_41_80_count", 0) or 0),
                length_gt80_count=int(record.get("length_gt80_count", 0) or 0),
                ast_depth_0_count=int(record.get("ast_depth_0_count", 0) or 0),
                ast_depth_1_count=int(record.get("ast_depth_1_count", 0) or 0),
                ast_depth_2_count=int(record.get("ast_depth_2_count", 0) or 0),
                ast_depth_3_count=int(record.get("ast_depth_3_count", 0) or 0),
                ast_depth_4_count=int(record.get("ast_depth_4_count", 0) or 0),
                ast_depth_5_count=int(record.get("ast_depth_5_count", 0) or 0),
                ast_depth_gt5_count=int(record.get("ast_depth_gt5_count", 0) or 0),
                max_expression_ast_depth=int(record.get("max_expression_ast_depth", 0) or 0),
                frac_expression_count=int(record.get("frac_expression_count", 0) or 0),
                sup_expression_count=int(record.get("sup_expression_count", 0) or 0),
                sub_expression_count=int(record.get("sub_expression_count", 0) or 0),
                sqrt_expression_count=int(record.get("sqrt_expression_count", 0) or 0),
                sum_expression_count=int(record.get("sum_expression_count", 0) or 0),
                env_expression_count=int(record.get("env_expression_count", 0) or 0),
                distinct_structure_type_count=int(record.get("distinct_structure_type_count", 0) or 0),
                structure_combination=str(record.get("structure_combination", "") or ""),
                distinct_token_count=int(record.get("distinct_token_count", 0) or 0),
                rare10_token_occurrence_count=int(record.get("rare10_token_occurrence_count", 0) or 0),
                rare10_expression_count=int(record.get("rare10_expression_count", 0) or 0),
                has_rare10=bool(record.get("has_rare10", False)),
                token_category_counts=category_counts,
            )
        )
    return rows
