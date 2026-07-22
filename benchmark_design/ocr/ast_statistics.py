"""AST depth statistics for OCR LaTeX expressions (structure-forest based)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.length_distribution import percentile
from benchmark_design.ocr.structure_forest import compute_ast_forest_metrics
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

AST_DEPTH_COMPLEXITY_THRESHOLD = 2


@dataclass(frozen=True, slots=True)
class NestedLevelBin:
    label: str
    min_level: int
    max_level: int | None
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class OcrAstStatisticsMetrics:
    expression_count: int
    mean_max_nested_level: float
    p50_max_nested_level: float
    p90_max_nested_level: float
    max_max_nested_level: int
    mean_token_nested_level: float
    nested_level_0_ratio: float
    nested_level_1_ratio: float
    nested_level_2_ratio: float
    nested_level_ge_3_ratio: float
    complex_expression_ratio: float
    bins: tuple[NestedLevelBin, ...]

    def as_summary_rows(self) -> list[tuple[str, str, float | int]]:
        return [
            (
                "Mean max nested level",
                "结构森林 AST 深度：每个表达式结构树最大深度之平均",
                self.mean_max_nested_level,
            ),
            (
                "P50 max nested level",
                "典型表达式的结构森林 AST 最大深度",
                self.p50_max_nested_level,
            ),
            (
                "P90 max nested level",
                "高复杂度表达式的结构森林 AST 最大深度",
                self.p90_max_nested_level,
            ),
            (
                "Max max nested level",
                "单个表达式的结构森林 AST 最大深度上限",
                self.max_max_nested_level,
            ),
            (
                "Mean token nested level",
                "结构森林中全部结构节点深度的平均值",
                self.mean_token_nested_level,
            ),
            (
                "Nested level = 0 ratio",
                "无结构嵌套（max nested level = 0）的表达式比例",
                self.nested_level_0_ratio,
            ),
            (
                "Nested level = 1 ratio",
                "max nested level = 1 的表达式比例",
                self.nested_level_1_ratio,
            ),
            (
                "Nested level = 2 ratio",
                "max nested level = 2 的表达式比例",
                self.nested_level_2_ratio,
            ),
            (
                "Nested level >= 3 ratio",
                "max nested level >= 3 的表达式比例",
                self.nested_level_ge_3_ratio,
            ),
            (
                "Complex expression ratio (>2)",
                "结构森林定义的复杂表达式比例（AST 深度 > 2）",
                self.complex_expression_ratio,
            ),
        ]


def compute_max_nested_levels(expressions: Iterable[ExpressionRecord]) -> list[int]:
    vocab = build_latex_vocab()
    levels: list[int] = []
    for record in expressions:
        tokens = tokenize_greedy(record.ocr, vocab)
        levels.append(compute_ast_forest_metrics(tokens).ast_depth)
    return levels


def compute_mean_token_nested_levels(expressions: Iterable[ExpressionRecord]) -> list[float]:
    vocab = build_latex_vocab()
    means: list[float] = []
    for record in expressions:
        tokens = tokenize_greedy(record.ocr, vocab)
        means.append(compute_ast_forest_metrics(tokens).mean_ast_node_depth)
    return means


def _build_bins(max_levels: Sequence[int]) -> tuple[NestedLevelBin, ...]:
    expression_count = len(max_levels)
    specs = (
        ("0", 0, 0),
        ("1", 1, 1),
        ("2", 2, 2),
        (">=3", 3, None),
    )
    bins: list[NestedLevelBin] = []
    for label, min_level, max_level in specs:
        if max_level is None:
            count = sum(level >= min_level for level in max_levels)
        else:
            count = sum(min_level <= level <= max_level for level in max_levels)
        share = count / expression_count if expression_count else 0.0
        bins.append(
            NestedLevelBin(
                label=label,
                min_level=min_level,
                max_level=max_level,
                count=count,
                share=share,
            )
        )
    return tuple(bins)


def compute_ocr_ast_statistics_from_levels(
    max_levels: Sequence[int],
    mean_token_levels: Sequence[float],
) -> OcrAstStatisticsMetrics:
    expression_count = len(max_levels)
    if expression_count == 0:
        empty_bins = _build_bins(())
        return OcrAstStatisticsMetrics(
            expression_count=0,
            mean_max_nested_level=0.0,
            p50_max_nested_level=0.0,
            p90_max_nested_level=0.0,
            max_max_nested_level=0,
            mean_token_nested_level=0.0,
            nested_level_0_ratio=0.0,
            nested_level_1_ratio=0.0,
            nested_level_2_ratio=0.0,
            nested_level_ge_3_ratio=0.0,
            complex_expression_ratio=0.0,
            bins=empty_bins,
        )

    bins = _build_bins(max_levels)

    def ratio_exact(level: int) -> float:
        return sum(value == level for value in max_levels) / expression_count

    return OcrAstStatisticsMetrics(
        expression_count=expression_count,
        mean_max_nested_level=sum(max_levels) / expression_count,
        p50_max_nested_level=percentile(max_levels, 50),
        p90_max_nested_level=percentile(max_levels, 90),
        max_max_nested_level=max(max_levels),
        mean_token_nested_level=sum(mean_token_levels) / expression_count,
        nested_level_0_ratio=ratio_exact(0),
        nested_level_1_ratio=ratio_exact(1),
        nested_level_2_ratio=ratio_exact(2),
        nested_level_ge_3_ratio=sum(level >= 3 for level in max_levels) / expression_count,
        complex_expression_ratio=sum(
            level > AST_DEPTH_COMPLEXITY_THRESHOLD for level in max_levels
        )
        / expression_count,
        bins=bins,
    )


def _nested_levels_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
) -> tuple[list[int], list[float]]:
    max_levels: list[int] = []
    mean_token_levels: list[float] = []
    for tokens in token_sequences:
        token_list = list(tokens)
        forest = compute_ast_forest_metrics(token_list)
        max_levels.append(forest.ast_depth)
        mean_token_levels.append(forest.mean_ast_node_depth)
    return max_levels, mean_token_levels


def compute_ocr_ast_statistics_from_features(
    features: Sequence[ExpressionFeatures],
) -> OcrAstStatisticsMetrics:
    max_levels = [feature.ast_depth for feature in features]
    mean_token_levels = [feature.mean_token_nested_level for feature in features]
    return compute_ocr_ast_statistics_from_levels(max_levels, mean_token_levels)


def compute_ocr_ast_statistics_from_token_sequences(
    token_sequences: Sequence[Sequence[str]],
) -> OcrAstStatisticsMetrics:
    max_levels, mean_token_levels = _nested_levels_from_token_sequences(token_sequences)
    return compute_ocr_ast_statistics_from_levels(max_levels, mean_token_levels)


def compute_ocr_ast_statistics_from_expressions(
    expressions: Iterable[ExpressionRecord],
) -> OcrAstStatisticsMetrics:
    vocab = build_latex_vocab()
    max_levels: list[int] = []
    mean_token_levels: list[float] = []

    for record in expressions:
        tokens = tokenize_greedy(record.ocr, vocab)
        forest = compute_ast_forest_metrics(tokens)
        max_levels.append(forest.ast_depth)
        mean_token_levels.append(forest.mean_ast_node_depth)

    return compute_ocr_ast_statistics_from_levels(max_levels, mean_token_levels)


def compute_ocr_ast_statistics(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrAstStatisticsMetrics:
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_ast_statistics_from_token_sequences(corpus.token_sequences)
