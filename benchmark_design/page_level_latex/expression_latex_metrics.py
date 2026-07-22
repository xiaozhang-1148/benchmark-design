"""Per-expression LaTeX metrics (Chapter-6 intermediate table)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from benchmark_design.page_level_latex.latex_protocol import (
    TAXONOMY_FIELD_KEYS,
    ParsedExpression,
    accumulate_token_counter,
    empty_token_category_counts,
    is_valid_for_latex,
    parse_expression,
    rare10_occurrence_count,
    rare10_token_set,
)
from benchmark_design.page_level_latex.loader import RawExpressionRow, load_raw_expressions
from benchmark_design.progress import parallel_map


@dataclass(frozen=True, slots=True)
class ExpressionLatexMetricsRow:
    image_id: str
    block_id: str
    line_id: str
    global_line_index: int
    raw_ocr_text: str
    normalized_latex: str
    valid_for_latex: bool
    exclusion_reason: str
    token_count: int
    ast_node_count: int
    ast_depth: int
    parse_ok: bool
    parse_error_count: int
    parse_status: str
    has_frac: bool
    has_sup: bool
    has_sub: bool
    has_sqrt: bool
    has_env: bool
    has_bigop: bool
    has_accent: bool
    has_stackrel: bool
    has_textcircled: bool
    structure_type_count: int
    structure_combination: str
    contains_delete: bool
    unknown_token_count: int
    rare10_token_occurrence_count: int
    has_rare10: bool
    token_category_counts: dict[str, int] = field(default_factory=dict)
    tokens: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpressionPass1Result:
    rows: tuple[ExpressionLatexMetricsRow, ...]
    token_counter: Counter[str]
    audit: dict[str, int]


def _row_from_parsed(
    raw: RawExpressionRow,
    parsed: ParsedExpression,
    *,
    rare_tokens: set[str] | None = None,
) -> ExpressionLatexMetricsRow:
    valid, reason = is_valid_for_latex(raw.raw_ocr_text, parsed.normalized_latex, parsed.tokens)
    rare_count = 0
    has_rare = False
    if valid and rare_tokens is not None:
        rare_count = rare10_occurrence_count(parsed.tokens, rare_tokens)
        has_rare = rare_count > 0
    return ExpressionLatexMetricsRow(
        image_id=raw.image_id,
        block_id=raw.block_id,
        line_id=raw.line_id,
        global_line_index=raw.global_line_index,
        raw_ocr_text=raw.raw_ocr_text,
        normalized_latex=parsed.normalized_latex,
        valid_for_latex=valid,
        exclusion_reason="" if valid else reason,
        token_count=parsed.token_count if valid else 0,
        ast_node_count=parsed.ast_node_count if valid else 0,
        ast_depth=parsed.ast_depth if valid else 0,
        parse_ok=parsed.parse_ok if valid else False,
        parse_error_count=parsed.parse_error_count if valid else 0,
        parse_status=parsed.parse_status if valid else "",
        has_frac=parsed.structure.has_frac if valid else False,
        has_sup=parsed.structure.has_sup if valid else False,
        has_sub=parsed.structure.has_sub if valid else False,
        has_sqrt=parsed.structure.has_sqrt if valid else False,
        has_env=parsed.structure.has_env if valid else False,
        has_bigop=parsed.structure.has_bigop if valid else False,
        has_accent=parsed.structure.has_accent if valid else False,
        has_stackrel=parsed.structure.has_stackrel if valid else False,
        has_textcircled=parsed.structure.has_textcircled if valid else False,
        structure_type_count=parsed.structure.structure_type_count if valid else 0,
        structure_combination=parsed.structure_combination if valid else "",
        contains_delete=parsed.contains_delete,
        unknown_token_count=parsed.unknown_token_count if valid else 0,
        rare10_token_occurrence_count=rare_count,
        has_rare10=has_rare,
        token_category_counts=parsed.token_category_counts if valid else empty_token_category_counts(),
        tokens=parsed.tokens if valid else (),
    )


def _parse_raw_row(raw: RawExpressionRow) -> ExpressionLatexMetricsRow:
    return _row_from_parsed(raw, parse_expression(raw.raw_ocr_text), rare_tokens=None)


def _compute_pass1_audit(raw_rows: list[RawExpressionRow], rows: list[ExpressionLatexMetricsRow]) -> dict[str, int]:
    page_ids = {row.image_id for row in raw_rows}
    line_id_dupes = 0
    order_dupes = 0
    seen_line: dict[str, set[str]] = defaultdict(set)
    seen_order: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for raw in raw_rows:
        if raw.line_id in seen_line[raw.image_id]:
            line_id_dupes += 1
        else:
            seen_line[raw.image_id].add(raw.line_id)
        key = (raw.block_order, raw.line_order)
        if key in seen_order[raw.image_id]:
            order_dupes += 1
        else:
            seen_order[raw.image_id].add(key)

    valid_rows = [row for row in rows if row.valid_for_latex]
    exclusion_counts = Counter(row.exclusion_reason for row in rows if not row.valid_for_latex)
    return {
        "raw_page_count": len(page_ids),
        "raw_expression_count": len(raw_rows),
        "valid_expression_count": len(valid_rows),
        "excluded_expression_count": len(rows) - len(valid_rows),
        "empty_annotation_count": exclusion_counts.get("empty_annotation", 0),
        "empty_after_normalization_count": exclusion_counts.get("empty_after_normalization", 0),
        "empty_after_tokenization_count": exclusion_counts.get("empty_after_tokenization", 0),
        "contains_delete_expression_count": sum(1 for row in rows if row.contains_delete),
        "ast_parse_error_expression_count": sum(1 for row in valid_rows if not row.parse_ok),
        "unknown_token_count": sum(row.unknown_token_count for row in valid_rows),
        "duplicate_line_id_count": line_id_dupes,
        "duplicate_page_order_count": order_dupes,
        "total_token_count": sum(row.token_count for row in valid_rows),
        "vocab_size": 0,  # filled after counter
    }


def build_expression_metrics_pass1(
    input_dir: Path,
    *,
    show_progress: bool = True,
    workers: int | None = None,
) -> ExpressionPass1Result:
    raw_rows = load_raw_expressions(input_dir, show_progress=show_progress, workers=workers)
    if not raw_rows:
        return ExpressionPass1Result(rows=(), token_counter=Counter(), audit={})

    if workers is not None and workers <= 1:
        rows = [_parse_raw_row(raw) for raw in raw_rows]
    else:
        rows = parallel_map(
            _parse_raw_row,
            raw_rows,
            description="Parsing expressions (pass 1)",
            show_progress=show_progress,
            workers=workers,
        )

    valid_token_sequences = [row.tokens for row in rows if row.valid_for_latex]
    token_counter = accumulate_token_counter(valid_token_sequences)
    audit = _compute_pass1_audit(raw_rows, rows)
    audit["vocab_size"] = len(token_counter)
    return ExpressionPass1Result(rows=tuple(rows), token_counter=token_counter, audit=audit)


def apply_rare10_pass2(
    pass1: ExpressionPass1Result,
) -> tuple[ExpressionLatexMetricsRow, ...]:
    rare_tokens = rare10_token_set(pass1.token_counter)
    updated: list[ExpressionLatexMetricsRow] = []
    for row in pass1.rows:
        if not row.valid_for_latex:
            updated.append(row)
            continue
        rare_count = rare10_occurrence_count(row.tokens, rare_tokens)
        updated.append(
            ExpressionLatexMetricsRow(
                image_id=row.image_id,
                block_id=row.block_id,
                line_id=row.line_id,
                global_line_index=row.global_line_index,
                raw_ocr_text=row.raw_ocr_text,
                normalized_latex=row.normalized_latex,
                valid_for_latex=row.valid_for_latex,
                exclusion_reason=row.exclusion_reason,
                token_count=row.token_count,
                ast_node_count=row.ast_node_count,
                ast_depth=row.ast_depth,
                parse_ok=row.parse_ok,
                parse_error_count=row.parse_error_count,
                parse_status=row.parse_status,
                has_frac=row.has_frac,
                has_sup=row.has_sup,
                has_sub=row.has_sub,
                has_sqrt=row.has_sqrt,
                has_env=row.has_env,
                has_bigop=row.has_bigop,
                has_accent=row.has_accent,
                has_stackrel=row.has_stackrel,
                has_textcircled=row.has_textcircled,
                structure_type_count=row.structure_type_count,
                structure_combination=row.structure_combination,
                contains_delete=row.contains_delete,
                unknown_token_count=row.unknown_token_count,
                rare10_token_occurrence_count=rare_count,
                has_rare10=rare_count > 0,
                token_category_counts=row.token_category_counts,
                tokens=row.tokens,
            )
        )
    return tuple(updated)


def expression_metrics_to_frame(rows: Sequence[ExpressionLatexMetricsRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        payload = asdict(row)
        tokens = payload.pop("tokens")
        category_counts = payload.pop("token_category_counts")
        payload.update(category_counts)
        payload["token_sequence"] = " ".join(tokens)
        records.append(payload)
    frame = pd.DataFrame.from_records(records)
    preferred = [
        "image_id",
        "block_id",
        "line_id",
        "global_line_index",
        "raw_ocr_text",
        "normalized_latex",
        "valid_for_latex",
        "exclusion_reason",
        "token_count",
        "ast_node_count",
        "ast_depth",
        "parse_ok",
        "parse_error_count",
        "parse_status",
        "has_frac",
        "has_sup",
        "has_sub",
        "has_sqrt",
        "has_env",
        "has_bigop",
        "has_accent",
        "has_stackrel",
        "has_textcircled",
        "structure_type_count",
        "structure_combination",
        *TAXONOMY_FIELD_KEYS,
        "contains_delete",
        "unknown_token_count",
        "rare10_token_occurrence_count",
        "has_rare10",
        "token_sequence",
    ]
    columns = [col for col in preferred if col in frame.columns]
    columns.extend(col for col in frame.columns if col not in columns)
    return frame.loc[:, columns]


def write_expression_latex_metrics(rows: Sequence[ExpressionLatexMetricsRow], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expression_metrics_to_frame(rows).to_csv(output_path, index=False)
    return output_path


def read_expression_latex_metrics_csv(path: Path) -> tuple[ExpressionLatexMetricsRow, ...]:
    frame = pd.read_csv(path)
    rows: list[ExpressionLatexMetricsRow] = []
    for record in frame.to_dict(orient="records"):
        category_counts = {key: int(record.get(key, 0) or 0) for key in TAXONOMY_FIELD_KEYS}
        token_sequence = str(record.get("token_sequence", "") or "")
        tokens = tuple(token_sequence.split()) if token_sequence else ()
        rows.append(
            ExpressionLatexMetricsRow(
                image_id=str(record["image_id"]),
                block_id=str(record["block_id"]),
                line_id=str(record["line_id"]),
                global_line_index=int(record["global_line_index"]),
                raw_ocr_text=str(record.get("raw_ocr_text", "") or ""),
                normalized_latex=str(record.get("normalized_latex", "") or ""),
                valid_for_latex=bool(record.get("valid_for_latex", False)),
                exclusion_reason=str(record.get("exclusion_reason", "") or ""),
                token_count=int(record.get("token_count", 0) or 0),
                ast_node_count=int(record.get("ast_node_count", record.get("token_count", 0)) or 0),
                ast_depth=int(record.get("ast_depth", 0) or 0),
                parse_ok=bool(record.get("parse_ok", False)),
                parse_error_count=int(record.get("parse_error_count", 0) or 0),
                parse_status=str(record.get("parse_status", "") or ""),
                has_frac=bool(record.get("has_frac", False)),
                has_sup=bool(record.get("has_sup", False)),
                has_sub=bool(record.get("has_sub", False)),
                has_sqrt=bool(record.get("has_sqrt", False)),
                has_env=bool(record.get("has_env", False)),
                has_bigop=bool(record.get("has_bigop", record.get("has_sum", False))),
                has_accent=bool(record.get("has_accent", False)),
                has_stackrel=bool(record.get("has_stackrel", False)),
                has_textcircled=bool(record.get("has_textcircled", False)),
                structure_type_count=int(record.get("structure_type_count", 0) or 0),
                structure_combination=str(record.get("structure_combination", "") or ""),
                contains_delete=bool(record.get("contains_delete", False)),
                unknown_token_count=int(record.get("unknown_token_count", 0) or 0),
                rare10_token_occurrence_count=int(record.get("rare10_token_occurrence_count", 0) or 0),
                has_rare10=bool(record.get("has_rare10", False)),
                token_category_counts=category_counts,
                tokens=tokens,
            )
        )
    return tuple(rows)
