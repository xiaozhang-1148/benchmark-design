"""Write consolidated OCR benchmark tables (1–10) to Markdown and CSV."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from benchmark_design.ocr.consolidated import OcrConsolidatedMetrics
from benchmark_design.report.output_layout import relative_input_path, relative_output_path

CSV_COLUMNS: tuple[str, ...] = (
    "table_id",
    "table_name",
    "metric",
    "definition",
    "value",
    "count",
    "share",
    "expr_ratio",
    "occ_ratio",
    "max_depth",
    "trigger_tokens",
)

TABLE_TITLES: tuple[tuple[int, str, str], ...] = (
    (1, "OCR Data Scale", "OCR 数据规模"),
    (2, "Expression Length Distribution", "表达式长度分布"),
    (3, "Fixed Length Bins", "固定长度分桶"),
    (4, "Token Taxonomy Composition", "Token 分类组成"),
    (5, "Token Long-Tail Statistics", "Token 长尾分布"),
    (6, "Structure Type Distribution", "结构类型分布"),
    (7, "Structure Combination Complexity", "结构组合复杂度"),
    (8, "Expression Content Type", "表达式内容类型"),
    (9, "Confusable Token Group Statistics", "易混 Token 组统计"),
    (10, "Expression-level Structural Difficulty", "表达式结构难度"),
)


@dataclass(frozen=True, slots=True)
class ConsolidatedCsvRow:
    table_id: int
    table_name: str
    metric: str
    definition: str = ""
    value: str = ""
    count: str = ""
    share: str = ""
    expr_ratio: str = ""
    occ_ratio: str = ""
    max_depth: str = ""
    trigger_tokens: str = ""

    def as_list(self) -> list[str]:
        return [
            str(self.table_id),
            self.table_name,
            self.metric,
            self.definition,
            self.value,
            self.count,
            self.share,
            self.expr_ratio,
            self.occ_ratio,
            self.max_depth,
            self.trigger_tokens,
        ]


def _fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _fmt_int(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _table_name(table_id: int) -> str:
    for tid, english, chinese in TABLE_TITLES:
        if tid == table_id:
            return f"{english} / {chinese}"
    msg = f"unknown table id: {table_id}"
    raise ValueError(msg)


_LENGTH_EXPORT_SKIP: frozenset[str] = frozenset({"p95", "p99"})


def _length_export_rows(metrics: OcrConsolidatedMetrics) -> list[tuple[str, float | int]]:
    return [(metric, value) for metric, value in metrics.length.as_rows() if metric not in _LENGTH_EXPORT_SKIP]


def build_consolidated_csv_rows(metrics: OcrConsolidatedMetrics) -> list[ConsolidatedCsvRow]:
    rows: list[ConsolidatedCsvRow] = []
    name = _table_name(1)

    for metric, value in metrics.scale.as_rows():
        formatted = _fmt_float(value) if metric == "duplicate rate" else _fmt_int(value)
        rows.append(ConsolidatedCsvRow(table_id=1, table_name=name, metric=metric, value=formatted))

    name = _table_name(2)
    for metric, value in _length_export_rows(metrics):
        if metric in {"p50", "p90", "max"}:
            formatted = _fmt_int(value) if metric == "max" else _fmt_float(value)
        else:
            formatted = _fmt_float(value)
        rows.append(ConsolidatedCsvRow(table_id=2, table_name=name, metric=metric, value=formatted))

    name = _table_name(3)
    for label, count, share in metrics.bins.as_rows():
        rows.append(
            ConsolidatedCsvRow(
                table_id=3,
                table_name=name,
                metric=label,
                count=str(count),
                share=_fmt_float(share),
            )
        )

    name = _table_name(4)
    for category, count, share in metrics.taxonomy.as_rows():
        rows.append(
            ConsolidatedCsvRow(
                table_id=4,
                table_name=name,
                metric=category,
                count=str(count),
                share=_fmt_float(share),
            )
        )
    rows.append(
        ConsolidatedCsvRow(
            table_id=4,
            table_name=name,
            metric="other / unknown token ratio",
            value=_fmt_float(metrics.taxonomy.other_unknown_ratio),
        )
    )

    name = _table_name(5)
    for metric, value in metrics.longtail.summary_rows():
        formatted = _fmt_float(value) if isinstance(value, float) else _fmt_int(value)
        rows.append(ConsolidatedCsvRow(table_id=5, table_name=name, metric=metric, value=formatted))

    name = _table_name(6)
    for (
        structure_type,
        trigger_tokens,
        expression_ratio,
        occurrence_ratio,
        max_depth,
        expression_count,
        occurrence_count,
    ) in metrics.structure.as_rows():
        rows.append(
            ConsolidatedCsvRow(
                table_id=6,
                table_name=name,
                metric=structure_type,
                trigger_tokens=trigger_tokens,
                expr_ratio=_fmt_float(expression_ratio),
                occ_ratio=_fmt_float(occurrence_ratio),
                max_depth=str(max_depth),
                count=str(expression_count),
                value=str(occurrence_count),
            )
        )

    name = _table_name(7)
    for metric, definition, value in metrics.complexity.as_rows():
        if isinstance(value, float) and metric != "Max structure type count":
            formatted = _fmt_float(value)
        else:
            formatted = _fmt_int(value)
        rows.append(
            ConsolidatedCsvRow(
                table_id=7,
                table_name=name,
                metric=metric,
                definition=definition,
                value=formatted,
            )
        )

    name = _table_name(8)
    for label, count, share in metrics.content.as_rows():
        rows.append(
            ConsolidatedCsvRow(
                table_id=8,
                table_name=name,
                metric=label,
                count=str(count),
                share=_fmt_float(share),
            )
        )

    name = _table_name(9)
    for group_metrics in metrics.confusable.primary_groups:
        group_label = group_metrics.group.name
        rows.append(
            ConsolidatedCsvRow(
                table_id=9,
                table_name=name,
                metric=f"{group_label} :: token count",
                count=str(group_metrics.token_count),
                share=_fmt_float(group_metrics.token_ratio),
            )
        )
        rows.append(
            ConsolidatedCsvRow(
                table_id=9,
                table_name=name,
                metric=f"{group_label} :: expression count",
                count=str(group_metrics.expression_count),
                expr_ratio=_fmt_float(group_metrics.expression_ratio),
            )
        )
        rows.append(
            ConsolidatedCsvRow(
                table_id=9,
                table_name=name,
                metric=f"{group_label} :: co-occurrence expression count",
                count=str(group_metrics.co_occurrence_expression_count),
            )
        )
        rows.append(
            ConsolidatedCsvRow(
                table_id=9,
                table_name=name,
                metric=f"{group_label} :: dominant tokens",
                value=", ".join(group_metrics.dominant_tokens),
            )
        )
        rows.append(
            ConsolidatedCsvRow(
                table_id=9,
                table_name=name,
                metric=f"{group_label} :: rare-side tokens",
                value=", ".join(group_metrics.rare_side_tokens),
            )
        )
        for token_count in group_metrics.token_counts:
            rows.append(
                ConsolidatedCsvRow(
                    table_id=9,
                    table_name=name,
                    metric=f"{group_label} :: token `{token_count.token}`",
                    count=str(token_count.count),
                    share=_fmt_float(token_count.share_of_group),
                    occ_ratio=_fmt_float(token_count.share_of_corpus),
                )
            )

    return rows


def write_consolidated_csv(rows: list[ConsolidatedCsvRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(row.as_list())


def _append_structural_difficulty_table(lines: list[str], features: Sequence) -> None:
    from benchmark_design.ocr.lbd_coordinates import (
        EXPRESSION_STRUCTURAL_DIFFICULTY_LABEL,
        compute_lbd_coordinate_metrics,
    )

    metrics = compute_lbd_coordinate_metrics(features)
    lines.extend(
        [
            "",
            "## Table 10. Expression-level Structural Difficulty / 表达式结构难度",
            "",
            f"**{EXPRESSION_STRUCTURAL_DIFFICULTY_LABEL}** — structural difficulty of individual "
            "expressions from L/B/D coordinates, not full-page image recognition difficulty.",
            "",
            "**L (token length):** L0 ≤ 20, L1 21–40, L2 > 40.  ",
            "**B (structure breadth):** B0 0–1 types, B1 2 types, B2 ≥ 3 types.  ",
            "**D (AST nesting depth):** D0 0–1, D1 2, D2 ≥ 3.",
            "",
            "Classification order: L1 (L0B0D0); L4 (≥2 of L2/B2/D2 with L≠L0 and D≠D0); "
            "L2 (score=1, or score=2 with L≠L2 and D≠D2); L3 (all remaining).  ",
            "27-cell counts: `tables/expression_lbd_coordinate_counts.csv`.  ",
            "Example crops: `figures/lbd_coordinate_examples/<tier>/`.",
            "",
            "| Structural difficulty | Count | Share |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in metrics.structural_difficulty_counts:
        lines.append(f"| {row.structural_difficulty} | {row.count:,} | {_fmt_float(row.ratio)} |")


def write_consolidated_markdown(
    metrics: OcrConsolidatedMetrics,
    output_path: Path,
    *,
    features: Sequence | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OCR Benchmark Statistics Summary",
        "",
        f"- Input: `{relative_input_path(metrics.input_dir)}`",
        f"- JSON files: {metrics.json_file_count:,}",
        f"- Expressions: {metrics.scale.expression_count:,}",
        f"- Total tokens: {metrics.scale.total_token_count:,}",
        "",
        "Consolidated report for tables 1–10 (`LATEX_DICT` greedy tokenization).",
        "",
        "## Table 1. OCR Data Scale / OCR 数据规模",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric, value in metrics.scale.as_rows():
        formatted = _fmt_float(value) if metric == "duplicate rate" else _fmt_int(value)
        lines.append(f"| {metric} | {formatted} |")

    lines.extend(
        [
            "",
            "## Table 2. Expression Length Distribution / 表达式长度分布",
            "",
            "Per-expression token count after `LATEX_DICT` greedy tokenization.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for metric, value in _length_export_rows(metrics):
        if metric in {"p50", "p90", "max"}:
            formatted = _fmt_int(value) if metric == "max" else _fmt_float(value)
        else:
            formatted = _fmt_float(value)
        lines.append(f"| {metric} | {formatted} |")

    lines.extend(
        [
            "",
            "## Table 3. Fixed Length Bins / 固定长度分桶",
            "",
            "| Length bin | Count | Share |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, count, share in metrics.bins.as_rows():
        lines.append(f"| {label} | {count:,} | {_fmt_float(share)} |")

    lines.extend(
        [
            "",
            "## Table 4. Token Taxonomy Composition / Token 分类组成",
            "",
            "Mutually exclusive token categories over corpus tokens.",
            "",
            "| Token type | Count | Share |",
            "| --- | ---: | ---: |",
        ]
    )
    for category, count, share in metrics.taxonomy.as_rows():
        lines.append(f"| {category} | {count:,} | {_fmt_float(share)} |")
    lines.append("")
    lines.append(
        f"**other / unknown token ratio:** {_fmt_float(metrics.taxonomy.other_unknown_ratio)}"
    )

    lines.extend(
        [
            "",
            "## Table 5. Token Long-Tail Statistics / Token 长尾分布",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for metric, value in metrics.longtail.summary_rows():
        formatted = _fmt_float(value) if isinstance(value, float) else _fmt_int(value)
        lines.append(f"| {metric} | {formatted} |")

    lines.extend(
        [
            "",
            "## Table 6. Structure Type Distribution / 结构类型分布",
            "",
            "Expr. Ratio = expressions containing the structure / all expressions.",
            "Occ. Ratio = structure trigger occurrences / all structural token occurrences.",
            "",
            "| Structure Type | Trigger tokens | Expr. Ratio | Occ. Ratio | Max Depth |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for (
        structure_type,
        trigger_tokens,
        expression_ratio,
        occurrence_ratio,
        max_depth,
        _expression_count,
        _occurrence_count,
    ) in metrics.structure.as_rows():
        lines.append(
            f"| {structure_type} | {trigger_tokens} | "
            f"{_fmt_float(expression_ratio)} | {_fmt_float(occurrence_ratio)} | {max_depth} |"
        )

    lines.extend(
        [
            "",
            "## Table 7. Structure Combination Complexity / 结构组合复杂度",
            "",
            "Distinct table-6 structure types co-occurring within one expression.",
            "",
            "| Metric | Definition | Value |",
            "| --- | --- | ---: |",
        ]
    )
    for metric, definition, value in metrics.complexity.as_rows():
        if isinstance(value, float) and metric != "Max structure type count":
            formatted = _fmt_float(value)
        else:
            formatted = _fmt_int(value)
        lines.append(f"| {metric} | {definition} | {formatted} |")

    lines.extend(
        [
            "",
            "## Table 8. Expression Content Type / 表达式内容类型",
            "",
            "Per-expression classification: **pure latex_command** — no CJK tokens; "
            "**pure CJK** — every token is CJK; **mixed** — both CJK and non-CJK tokens.",
            "",
            "| Content type | Count | Share |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, count, share in metrics.content.as_rows():
        lines.append(f"| {label} | {count:,} | {_fmt_float(share)} |")

    lines.extend(
        [
            "",
            "## Table 9. Confusable Token Group Statistics / 易混 Token 组统计",
            "",
            "Primary groups for the main narrative. Token ratio = group token count / all tokens; "
            "Expr. ratio = expressions with any group token / all expressions. "
            "Full token-by-token counts: `tables/appendix/confusable_token_counts.csv`. "
            "Example crops for `4` and `\\varphi`: `figures/confusable_token_examples/greek-variant/`.",
            "",
            "| Group | Representative tokens | Token count | Token ratio | Expr. count | Expr. ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group_metrics in metrics.confusable.primary_groups:
        group_name, representatives, token_count, token_ratio, expr_count, expr_ratio = (
            group_metrics.main_table_row()
        )
        lines.append(
            f"| {group_name} | `{representatives}` | {token_count:,} | {_fmt_float(token_ratio)} | "
            f"{expr_count:,} | {_fmt_float(expr_ratio)} |"
        )

    if features is not None:
        _append_structural_difficulty_table(lines, features)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_consolidated_metadata(
    metrics: OcrConsolidatedMetrics,
    *,
    output_path: Path,
    markdown_path: Path,
    output_root: Path | None = None,
) -> None:
    root = output_root or output_path.parent
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": relative_input_path(metrics.input_dir),
        "json_file_count": metrics.json_file_count,
        "expression_count": metrics.scale.expression_count,
        "tables": [
            {"table_id": table_id, "table_name_en": en, "table_name_zh": zh}
            for table_id, en, zh in TABLE_TITLES
        ],
        "outputs": {
            "markdown": relative_output_path(markdown_path, root),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_consolidated_report(
    metrics: OcrConsolidatedMetrics,
    output_dir: Path,
    *,
    output_root: Path | None = None,
    metadata_dir: Path | None = None,
    features: Sequence | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_root or output_dir
    metadata_dir = metadata_dir or (root / "docs" / "metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = root / "ocr_benchmark_summary.md"
    metadata_path = metadata_dir / "ocr_benchmark_metadata.json"

    write_consolidated_markdown(metrics, markdown_path, features=features)
    write_consolidated_metadata(
        metrics,
        output_path=metadata_path,
        markdown_path=markdown_path,
        output_root=root,
    )

    return {
        "markdown": markdown_path,
        "metadata": metadata_path,
    }
