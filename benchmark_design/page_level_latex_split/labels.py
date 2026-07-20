"""Convert continuous page features into multilabel stratification tags."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from benchmark_design.page_level_latex_split.config import BinSpec, SplitConfig

BINARY_LABEL_COLUMNS = (
    "has_frac",
    "has_sup",
    "has_sub",
    "has_sqrt",
    "has_sum",
    "has_env",
    "has_rare8",
    "has_digit_letter_pair",
    "has_circle_like_pair",
    "has_latin_greek_pair",
    "has_greek_variant_pair",
    "has_operator_variable_pair",
    "has_relation_stroke_pair",
)

EXPR_DEPTH_COLUMNS = tuple(f"has_expr_depth_{depth}" for depth in range(6))


def structure_count_label(count: int) -> str:
    return f"struc_cnt_{int(count)}"


def joint_structure_depth_label(structure_count: int, max_ast_depth: int) -> str:
    depth = min(int(max_ast_depth), 5)
    return f"joint_sc{int(structure_count)}_d{depth}"


@dataclass(frozen=True, slots=True)
class PageLabels:
    page_id: str
    labels: frozenset[str]
    expr_bin: str
    page_token_bin: str
    maxlen_bin: str
    depth_bin: str


def assign_bin(value: float, bins: tuple[BinSpec, ...]) -> str:
    # Pages with zero expressions: map into first open bin when value == 0.
    if value <= 0 and bins:
        # Prefer an explicit zero-capable first bin; otherwise use first label.
        first = bins[0]
        if first.min <= 0:
            return first.label
        return first.label
    for spec in bins:
        upper_ok = True if spec.max is None else value < spec.max
        if value >= spec.min and upper_ok:
            return spec.label
    return bins[-1].label


def assign_depth_label(depth: int, mapping: dict[str, tuple[int, ...]]) -> str:
    for label, values in mapping.items():
        if int(depth) in values:
            return label
    # Fallback: deepest bucket
    return list(mapping.keys())[-1]


def build_page_labels(features: pd.DataFrame, config: SplitConfig) -> list[PageLabels]:
    rows: list[PageLabels] = []
    for rec in features.itertuples(index=False):
        expr_bin = assign_bin(float(rec.expression_count), config.expression_count_bins)
        token_bin = assign_bin(float(rec.page_token_count), config.page_token_bins)
        maxlen_bin = assign_bin(float(rec.max_expression_token_count), config.max_expression_token_bins)
        depth_bin = assign_depth_label(int(rec.max_ast_depth), config.ast_depth_labels)
        structure_count = int(rec.structure_type_count)
        max_ast_depth = int(rec.max_ast_depth)
        labels = {
            expr_bin,
            token_bin,
            maxlen_bin,
            depth_bin,
            structure_count_label(structure_count),
            joint_structure_depth_label(structure_count, max_ast_depth),
        }
        for col in BINARY_LABEL_COLUMNS:
            if int(getattr(rec, col)) == 1:
                labels.add(col)
        for col in EXPR_DEPTH_COLUMNS:
            if hasattr(rec, col) and int(getattr(rec, col)) == 1:
                labels.add(col)
        rows.append(
            PageLabels(
                page_id=str(rec.page_id),
                labels=frozenset(labels),
                expr_bin=expr_bin,
                page_token_bin=token_bin,
                maxlen_bin=maxlen_bin,
                depth_bin=depth_bin,
            )
        )
    return rows


def labels_to_frame(page_labels: list[PageLabels], features: pd.DataFrame) -> pd.DataFrame:
    label_map = {row.page_id: row for row in page_labels}
    records = []
    for rec in features.itertuples(index=False):
        page_id = str(rec.page_id)
        lab = label_map[page_id]
        record = {
            "page_id": page_id,
            "expression_count": int(rec.expression_count),
            "page_token_count": int(rec.page_token_count),
            "max_expression_token_count": int(rec.max_expression_token_count),
            "max_ast_depth": int(rec.max_ast_depth),
            "structure_type_count": int(rec.structure_type_count),
            "expr_bin": lab.expr_bin,
            "page_token_bin": lab.page_token_bin,
            "maxlen_bin": lab.maxlen_bin,
            "depth_bin": lab.depth_bin,
            "labels": ";".join(sorted(lab.labels)),
            **{col: int(getattr(rec, col)) for col in BINARY_LABEL_COLUMNS},
        }
        for col in EXPR_DEPTH_COLUMNS:
            if hasattr(rec, col):
                record[col] = int(getattr(rec, col))
        records.append(record)
    return pd.DataFrame.from_records(records)
