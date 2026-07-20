"""Generate the root ``summary.md`` benchmark report."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from benchmark_design.ocr.ast_statistics import OcrAstStatisticsMetrics
from benchmark_design.ocr.consolidated import OcrConsolidatedMetrics
from benchmark_design.ocr.duplicates import duplicate_stats_from_features
from benchmark_design.ocr.expression_content import (
    ExpressionContentKind,
    compute_ocr_expression_content_from_token_sequences,
)
from benchmark_design.ocr.expression_features import ExpressionFeatures, parse_success_rate
from benchmark_design.ocr.processing import EnrichedCorpus
from benchmark_design.ocr.token_taxonomy import TokenCategory

STRUCTURE_TYPE_SUMMARY_ORDER: tuple[str, ...] = (
    "分式",
    "上标",
    "下标",
    "根式",
    "Env.",
    "求和",
    "极限",
    "积分",
)

STRUCTURE_TYPE_SUMMARY_LABELS: dict[str, str] = {
    "分式": "Fraction",
    "上标": "Superscript",
    "下标": "Subscript",
    "根式": "Radical",
    "求和": "Summation",
    "积分": "Integral",
    "Env.": "Env.",
    "极限": "Limit",
}

CONTENT_KIND_SUMMARY_LABELS: dict[ExpressionContentKind, str] = {
    ExpressionContentKind.LATEX_COMMAND: "Pure latex_command",
    ExpressionContentKind.CJK: "Pure CJK",
    ExpressionContentKind.MIXED: "Mixed",
}

TAXONOMY_SUMMARY_LABELS: dict[TokenCategory, str] = {
    TokenCategory.LATIN_VARIABLE: "Latin variable tokens",
    TokenCategory.DIGIT: "Digit tokens",
    TokenCategory.SPECIAL_SYMBOL: "Special symbol tokens",
    TokenCategory.OPERATOR: "Operator tokens",
    TokenCategory.GROUPING: "Grouping tokens",
    TokenCategory.STRUCTURAL: "Structural tokens",
    TokenCategory.CJK: "CJK tokens",
    TokenCategory.PUNCTUATION: "Punctuation tokens",
    TokenCategory.LAYOUT_ALIGNMENT: "Layout / alignment tokens",
    TokenCategory.OTHER: "Unclassified tokens",
}


def _fmt_int(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{int(value):,}"


def _fmt_pct(share: float) -> str:
    pct = share * 100.0
    if pct >= 0.01:
        return f"{pct:.2f}%"
    return f"{pct:.4f}%"


def _fmt_length_metric(value: float | int) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def _fmt_float(value: float, *, decimals: int = 6) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{decimals}f}"


def _ast_depth_rows(features: list[ExpressionFeatures]) -> list[tuple[int, int, float]]:
    counter = Counter(feature.ast_depth for feature in features)
    total = len(features)
    max_depth = max(counter) if counter else 0
    upper = max(max_depth, 5)
    return [
        (depth, counter.get(depth, 0), counter.get(depth, 0) / total if total else 0.0)
        for depth in range(upper + 1)
    ]


def build_benchmark_summary_markdown(
    enriched: EnrichedCorpus,
    metrics: OcrConsolidatedMetrics,
    ast_metrics: OcrAstStatisticsMetrics,
) -> str:
    features = list(enriched.features)
    scale = metrics.scale
    length = metrics.length
    parse_ok = parse_success_rate(features)
    duplicate_stats = duplicate_stats_from_features(features)
    unique_count = scale.unique_normalized_latex_count
    redundant_count = duplicate_stats.redundant_expression_count
    duplicate_rate = scale.duplicate_rate

    lines = [
        "# OCR Benchmark Summary",
        "",
        f"Dataset: {enriched.dataset}  ",
        f"Expressions: {_fmt_int(scale.expression_count)}  ",
        f"JSON files: {_fmt_int(enriched.json_file_count)}  ",
        "Tokenizer: LATEX_DICT greedy longest-match tokenizer  ",
        "AST metric: PosFormer max nested level  ",
        "",
        "## 1. Dataset Scale",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| JSON files | {_fmt_int(enriched.json_file_count)} |",
        f"| Expression count | {_fmt_int(scale.expression_count)} |",
        f"| Total token count | {_fmt_int(scale.total_token_count)} |",
        f"| Vocabulary size | {_fmt_int(scale.vocabulary_size)} |",
        f"| Unique normalized LaTeX count | {_fmt_int(unique_count)} |",
        f"| Parse success rate | {parse_ok * 100:.3f}% |",
        "",
        "## 2. Expression Length",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean length | {_fmt_length_metric(length.mean_length)} |",
        f"| Std | {_fmt_length_metric(length.std)} |",
        f"| CV | {_fmt_length_metric(length.cv)} |",
        f"| P50 | {_fmt_length_metric(length.p50)} |",
        f"| P90 | {_fmt_length_metric(length.p90)} |",
        f"| Max | {_fmt_length_metric(length.max_length)} |",
        "",
        "| Length bin | Count | Share |",
        "| --- | ---: | ---: |",
    ]

    for label, count, share in metrics.bins.as_rows():
        lines.append(f"| {label} | {_fmt_int(count)} | {_fmt_pct(share)} |")

    lines.extend(
        [
            "",
            "## 3. Duplicate Summary",
            "",
            "Duplicate definition: two expression records are duplicates iff their full "
            "`normalized_latex` strings are exactly equal (whole-expression match only).",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Expression count | {_fmt_int(scale.expression_count)} |",
            f"| Unique normalized LaTeX count | {_fmt_int(unique_count)} |",
            f"| Redundant expression count | {_fmt_int(redundant_count)} |",
            f"| Redundancy / duplicate rate | {_fmt_pct(duplicate_rate)} |",
            f"| Expressions belonging to duplicated groups | {_fmt_int(duplicate_stats.duplicated_group_expression_count)} |",
            f"| Duplicated-group expression ratio | {_fmt_pct(duplicate_stats.duplicated_group_expression_ratio)} |",
            f"| Max duplicate group size | {_fmt_int(duplicate_stats.max_duplicate_group_size)} |",
            "",
            "## 4. Token Taxonomy",
            "",
            "| Token type | Count | Share |",
            "| --- | ---: | ---: |",
        ]
    )

    taxonomy_rows = sorted(
        metrics.taxonomy.categories,
        key=lambda item: item.count,
        reverse=True,
    )
    for item in taxonomy_rows:
        label = TAXONOMY_SUMMARY_LABELS[item.category]
        lines.append(f"| {label} | {_fmt_int(item.count)} | {_fmt_pct(item.share)} |")

    lines.extend(
        [
            "",
            "## 5. Token Long-Tail",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Vocabulary size | {_fmt_int(metrics.longtail.vocabulary_size)} |",
            f"| Gini | {_fmt_float(metrics.longtail.gini)} |",
        ]
    )
    for k, coverage in metrics.longtail.top_k_coverage:
        lines.append(f"| Top-{k} coverage | {_fmt_pct(coverage)} |")

    lines.extend(
        [
            "",
            "## 6. Structure Complexity",
            "",
            "| Structure type | Trigger tokens | Expr. ratio | Occ. ratio | Max depth |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    structure_by_type = {row.structure_type: row for row in metrics.structure.rows}
    for structure_type in STRUCTURE_TYPE_SUMMARY_ORDER:
        row = structure_by_type[structure_type]
        label = STRUCTURE_TYPE_SUMMARY_LABELS[structure_type]
        triggers = f"`{row.trigger_tokens}`"
        lines.append(
            f"| {label} | {triggers} | {_fmt_pct(row.expression_ratio)} | "
            f"{_fmt_pct(row.occurrence_ratio)} | {row.max_depth} |"
        )

    lines.extend(
        [
            "",
            "## 7. AST Depth",
            "",
            "| AST depth | Count | Share |",
            "| ---: | ---: | ---: |",
        ]
    )
    for depth, count, share in _ast_depth_rows(features):
        lines.append(f"| {depth} | {_fmt_int(count)} | {_fmt_pct(share)} |")

    lines.extend(
        [
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Mean max nested level | {_fmt_float(ast_metrics.mean_max_nested_level)} |",
            f"| P50 max nested level | {_fmt_length_metric(ast_metrics.p50_max_nested_level)} |",
            f"| P90 max nested level | {_fmt_length_metric(ast_metrics.p90_max_nested_level)} |",
            f"| Max nested level | {ast_metrics.max_max_nested_level} |",
            f"| Complex expression ratio, depth > 2 | {_fmt_pct(ast_metrics.complex_expression_ratio)} |",
            "",
            "## 8. Expression Content Type",
            "",
            "Per-expression classification after `LATEX_DICT` greedy tokenization: "
            "**pure latex_command** — no CJK tokens; **pure CJK** — every token is CJK; "
            "**mixed** — both CJK and non-CJK tokens.",
            "",
            "| Content type | Count | Share |",
            "| --- | ---: | ---: |",
        ]
    )

    content_metrics = compute_ocr_expression_content_from_token_sequences(
        feature.token_sequence for feature in features
    )
    for item in content_metrics.kinds:
        label = CONTENT_KIND_SUMMARY_LABELS[item.kind]
        lines.append(f"| {label} | {_fmt_int(item.count)} | {_fmt_pct(item.share)} |")

    lines.extend(
        [
            "",
            "## 9. Confusable Token Groups",
            "",
            "Potentially confusable token groups after `LATEX_DICT` greedy tokenization. "
            "See `tables/confusable_token_group_summary.csv`. "
            "Example crops for `4` and `\\varphi` (20 samples, OCR length > 3): "
            "`figures/confusable_token_examples/greek-variant/`.",
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
            f"| {group_name} | `{representatives}` | {_fmt_int(token_count)} | {_fmt_pct(token_ratio)} | "
            f"{_fmt_int(expr_count)} | {_fmt_pct(expr_ratio)} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_benchmark_summary_md(
    output_path: Path,
    enriched: EnrichedCorpus,
    metrics: OcrConsolidatedMetrics,
    ast_metrics: OcrAstStatisticsMetrics,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_benchmark_summary_markdown(enriched, metrics, ast_metrics),
        encoding="utf-8",
    )
