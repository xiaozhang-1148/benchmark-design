"""Consolidated OCR benchmark metrics (tables 1–9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.confusable_tokens import (
    OcrConfusableTokenMetrics,
    compute_ocr_confusable_token_metrics_from_token_sequences,
)
from benchmark_design.ocr.expression_content import (
    OcrExpressionContentMetrics,
    compute_ocr_expression_content_from_token_sequences,
)
from benchmark_design.ocr.length_bins import OcrLengthBinMetrics, compute_ocr_length_bins_from_lengths
from benchmark_design.ocr.length_distribution import (
    OcrLengthDistributionMetrics,
    compute_expression_token_lengths_from_tokenized,
    compute_ocr_length_distribution_from_lengths,
)
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.scale import OcrScaleMetrics, compute_ocr_scale_from_tokenized
from benchmark_design.ocr.structure_complexity import (
    OcrStructureComplexityMetrics,
    compute_ocr_structure_complexity_from_counts,
    compute_structure_type_counts_from_features,
    compute_structure_type_counts_from_token_sequences,
)
from benchmark_design.ocr.structure_distribution import (
    OcrStructureDistributionMetrics,
    compute_ocr_structure_distribution_from_features,
    compute_ocr_structure_distribution_from_token_sequences,
)
from benchmark_design.ocr.token_longtail import (
    OcrTokenLongtailMetrics,
    compute_ocr_token_longtail_from_token_sequences,
)
from benchmark_design.ocr.token_taxonomy import (
    OcrTokenTaxonomyMetrics,
    compute_ocr_token_taxonomy_from_token_sequences,
)

if TYPE_CHECKING:
    from benchmark_design.ocr.processing import EnrichedCorpus, TokenizedCorpus


@dataclass(frozen=True, slots=True)
class OcrConsolidatedMetrics:
    input_dir: Path
    json_file_count: int
    scale: OcrScaleMetrics
    length: OcrLengthDistributionMetrics
    bins: OcrLengthBinMetrics
    taxonomy: OcrTokenTaxonomyMetrics
    longtail: OcrTokenLongtailMetrics
    structure: OcrStructureDistributionMetrics
    complexity: OcrStructureComplexityMetrics
    content: OcrExpressionContentMetrics
    confusable: OcrConfusableTokenMetrics


def compute_ocr_consolidated_metrics_from_corpus(
    corpus: TokenizedCorpus | EnrichedCorpus,
) -> OcrConsolidatedMetrics:
    """Compute tables 1–9 from a pre-built tokenized corpus."""
    lengths = compute_expression_token_lengths_from_tokenized(corpus.token_sequences)
    if hasattr(corpus, "features") and corpus.features:
        features = list(corpus.features)
        structure_type_counts = compute_structure_type_counts_from_features(features)
        structure = compute_ocr_structure_distribution_from_features(features)
    else:
        structure_type_counts = compute_structure_type_counts_from_token_sequences(corpus.token_sequences)
        structure = compute_ocr_structure_distribution_from_token_sequences(corpus.token_sequences)

    return OcrConsolidatedMetrics(
        input_dir=corpus.input_dir,
        json_file_count=corpus.json_file_count,
        scale=compute_ocr_scale_from_tokenized(
            corpus.expressions,
            corpus.token_sequences,
            json_file_count=corpus.json_file_count,
        ),
        length=compute_ocr_length_distribution_from_lengths(lengths),
        bins=compute_ocr_length_bins_from_lengths(lengths),
        taxonomy=compute_ocr_token_taxonomy_from_token_sequences(corpus.token_sequences),
        longtail=compute_ocr_token_longtail_from_token_sequences(corpus.token_sequences),
        structure=structure,
        complexity=compute_ocr_structure_complexity_from_counts(structure_type_counts),
        content=compute_ocr_expression_content_from_token_sequences(corpus.token_sequences),
        confusable=compute_ocr_confusable_token_metrics_from_token_sequences(corpus.token_sequences),
    )


def compute_ocr_consolidated_metrics(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrConsolidatedMetrics:
    """Compute tables 1–9 with one parallel load + tokenize pass."""
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_consolidated_metrics_from_corpus(corpus)


def compute_ocr_consolidated_metrics_from_expressions(
    expressions: list[ExpressionRecord],
    *,
    input_dir: Path,
    json_file_count: int = 0,
) -> OcrConsolidatedMetrics:
    from benchmark_design.ocr.processing import TokenizedCorpus, tokenize_expressions_parallel

    token_sequences = tokenize_expressions_parallel(expressions, ProcessingOptions())
    corpus = TokenizedCorpus(
        input_dir=input_dir,
        json_file_count=json_file_count,
        expressions=tuple(expressions),
        token_sequences=token_sequences,
    )
    return compute_ocr_consolidated_metrics_from_corpus(corpus)
