"""Cross-benchmark profile metrics and aggregation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from benchmark_design.config import CROSS_BENCHMARK_REPORT_GROUPS, CROSS_BENCHMARK_REPORT_ORDER, CROSS_BENCHMARK_SETS
from benchmark_design.ocr.duplicates import duplicate_stats_from_features
from benchmark_design.ocr.expression_features import (
    ExpressionFeatures,
    corpus_token_counter,
    parse_success_rate,
)
from benchmark_design.ocr.length_bin_specs import DEFAULT_LENGTH_BINS, assign_length_bin
from benchmark_design.ocr.length_distribution import percentile
from benchmark_design.ocr.processing import EnrichedCorpus, ProcessingOptions, build_enriched_corpus_cached
from benchmark_design.ocr.structure_complexity import compute_ocr_structure_complexity_from_counts
from benchmark_design.ocr.structure_distribution import (
    MATRIX_STRUCTURE_TYPE,
    compute_ocr_structure_distribution_from_token_sequences,
)
from benchmark_design.ocr.token_longtail import (
    gini_coefficient,
    rare_expression_ratio_from_token_sequences,
    rare_vocab_ratio,
    top_k_coverage,
)
from benchmark_design.ocr.token_taxonomy import (
    TokenCategory,
    classify_token,
    compute_ocr_token_taxonomy_from_token_sequences,
)
from benchmark_design.ocr.lbd_coordinates import (
    STRUCTURAL_DIFFICULTY_TIERS,
    assign_lbd_from_feature,
    compute_structural_difficulty_counts,
)


@dataclass(frozen=True, slots=True)
class CrossBenchmarkRow:
    dataset: str
    expression_count: int
    vocabulary_size: int
    total_token_count: int
    duplicate_rate: float
    mean_length: float
    p50_length: float
    p90_length: float
    p95_length: float
    p99_length: float
    max_length: int
    gini: float
    top_10_coverage: float
    unknown_token_ratio: float
    rare_token_occurrence_ratio: float
    mean_structure_type_count: float
    mean_ast_depth: float
    parse_success_rate: float
    vocab_coverage: float


@dataclass(frozen=True, slots=True)
class LengthBinStat:
    label: str
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class CrossBenchmarkProfile:
    display_name: str
    expression_count: int
    unique_expression_count: int
    duplicate_rate: float
    vocabulary_size: int
    parse_success_rate: float
    mean_length: float
    p50_length: float
    p90_length: float
    p95_length: float
    p99_length: float
    max_length: int
    length_bins: tuple[LengthBinStat, ...]
    gini: float
    top_10_coverage: float
    top_50_coverage: float
    top_100_coverage: float
    rare_1_vocab_ratio: float
    rare_5_vocab_ratio: float
    rare_10_expression_ratio: float
    structural_expression_ratio: float
    mean_structure_type_count: float
    multi_structure_ratio_ge_2: float
    multi_structure_ratio_ge_3: float
    matrix_structure_ratio: float
    mean_ast_depth: float
    p50_ast_depth: float
    p90_ast_depth: float
    max_ast_depth: int
    ast_ge_3_ratio: float
    count_gt_40_tokens: int
    count_gt_80_tokens: int
    count_any_structure: int
    count_multi_struct_ge_3: int
    structure_type_count_bins: tuple[int, ...]
    count_matrix_structure: int
    count_ast_ge_3: int
    ast_depth_counts: tuple[int, ...]
    count_gt_40_and_ast_ge_2: int
    count_gt_80_and_ast_ge_3: int
    total_token_count: int
    taxonomy_token_counts: tuple[int, ...]
    english_token_ratio: float
    digit_token_ratio: float
    greek_token_ratio: float
    special_symbol_token_ratio: float
    operator_token_ratio: float
    grouping_token_ratio: float
    structural_token_ratio: float
    cjk_token_ratio: float
    other_unknown_token_ratio: float
    notes: str
    structural_difficulty_counts: tuple[int, ...]


AST_DEPTH_COUNT_LABELS: tuple[str, ...] = ("0", "1", "2", "3", "4", "≥5")
STRUCTURE_ANY_COLUMN = "Any Structure ≥1"
STRUCTURE_TYPE_COUNT_LABELS: tuple[str, ...] = (
    "1 Structure Type",
    "2 Structure Types",
    "3 Structure Types",
    "≥4 Structure Types",
)

CROSS_BENCHMARK_TAXONOMY_CATEGORIES: tuple[TokenCategory, ...] = (
    TokenCategory.ENGLISH,
    TokenCategory.DIGIT,
    TokenCategory.GREEK,
    TokenCategory.SPECIAL_SYMBOL,
    TokenCategory.OPERATOR,
    TokenCategory.GROUPING,
    TokenCategory.STRUCTURAL,
    TokenCategory.CJK,
    TokenCategory.OTHER,
)


def _structure_type_count_bins(counts: Sequence[int]) -> tuple[int, ...]:
    counter = Counter(counts)
    return (
        counter.get(1, 0),
        counter.get(2, 0),
        counter.get(3, 0),
        sum(count for structure_count, count in counter.items() if structure_count >= 4),
    )


def _structural_difficulty_count_bins(features: list[ExpressionFeatures]) -> tuple[int, ...]:
    coordinates = [assign_lbd_from_feature(feature) for feature in features]
    tier_counts = compute_structural_difficulty_counts(coordinates)
    return tuple(row.count for row in tier_counts)


def _ast_depth_count_bins(features: list[ExpressionFeatures]) -> tuple[int, ...]:
    counter = Counter(feature.ast_depth for feature in features)
    return (
        counter.get(0, 0),
        counter.get(1, 0),
        counter.get(2, 0),
        counter.get(3, 0),
        counter.get(4, 0),
        sum(count for depth, count in counter.items() if depth >= 5),
    )


def _taxonomy_category_stats(
    token_sequences: list[tuple[str, ...]],
) -> tuple[int, tuple[int, ...], tuple[float, ...]]:
    metrics = compute_ocr_token_taxonomy_from_token_sequences(token_sequences)
    by_category = {item.category: item for item in metrics.categories}
    counts = tuple(by_category[category].count for category in CROSS_BENCHMARK_TAXONOMY_CATEGORIES)
    shares = tuple(by_category[category].share for category in CROSS_BENCHMARK_TAXONOMY_CATEGORIES)
    return metrics.total_token_count, counts, shares


def _duplicate_rate(features: list[ExpressionFeatures]) -> float:
    return duplicate_stats_from_features(features).duplicate_rate


def _unknown_token_ratio(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    unknown = sum(count for token, count in counter.items() if classify_token(token) is TokenCategory.OTHER)
    return unknown / total


def _rare_token_occurrence_ratio(counter: Counter[str], *, threshold: int = 10) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    rare = sum(count for token, count in counter.items() if count <= threshold)
    return rare / total


def _vocab_coverage(counter: Counter[str]) -> float:
    if not counter:
        return 0.0
    known = sum(count for token, count in counter.items() if classify_token(token) is not TokenCategory.OTHER)
    return known / sum(counter.values())


def _length_bin_stats(features: list[ExpressionFeatures]) -> tuple[LengthBinStat, ...]:
    counts = {spec.label: 0 for spec in DEFAULT_LENGTH_BINS}
    for feature in features:
        counts[assign_length_bin(feature.token_length)] += 1
    total = len(features)
    return tuple(
        LengthBinStat(label=spec.label, count=counts[spec.label], share=counts[spec.label] / total if total else 0.0)
        for spec in DEFAULT_LENGTH_BINS
    )


def load_merged_features(
    source_datasets: Sequence[str],
    *,
    processing: ProcessingOptions,
    corpus_cache: dict[str, EnrichedCorpus] | None = None,
) -> tuple[list[ExpressionFeatures], list[tuple[str, ...]]]:
    features: list[ExpressionFeatures] = []
    token_sequences: list[tuple[str, ...]] = []
    for dataset_name in source_datasets:
        if dataset_name not in CROSS_BENCHMARK_SETS:
            msg = f"Unknown cross-benchmark dataset: {dataset_name}"
            raise ValueError(msg)
        if corpus_cache is not None and dataset_name in corpus_cache:
            enriched = corpus_cache[dataset_name]
        else:
            enriched = build_enriched_corpus_cached(
                dataset_name,
                CROSS_BENCHMARK_SETS[dataset_name],
                processing,
            )
            if corpus_cache is not None:
                corpus_cache[dataset_name] = enriched
        features.extend(enriched.features)
        token_sequences.extend(enriched.token_sequences)
    return features, token_sequences


def _load_all_cross_benchmark_corpora(
    dataset_names: Sequence[str],
    *,
    processing: ProcessingOptions,
    corpus_cache: dict[str, EnrichedCorpus] | None = None,
) -> dict[str, EnrichedCorpus]:
    cache = corpus_cache if corpus_cache is not None else {}
    for dataset_name in dataset_names:
        if dataset_name not in CROSS_BENCHMARK_SETS:
            msg = f"Unknown cross-benchmark dataset: {dataset_name}"
            raise ValueError(msg)
        if dataset_name not in cache:
            cache[dataset_name] = build_enriched_corpus_cached(
                dataset_name,
                CROSS_BENCHMARK_SETS[dataset_name],
                processing,
            )
    return cache


def compute_cross_benchmark_results(
    *,
    processing: ProcessingOptions | None = None,
    report_groups: dict[str, tuple[str, ...]] | None = None,
    report_order: tuple[str, ...] | None = None,
    notes_by_name: dict[str, str] | None = None,
    dataset_names: list[str] | None = None,
    corpus_cache: dict[str, EnrichedCorpus] | None = None,
) -> tuple[list[CrossBenchmarkProfile], list[CrossBenchmarkRow]]:
    """Build profiles and per-dataset rows with a single corpus load per dataset."""
    from benchmark_design.config import CROSS_BENCHMARK_NOTES

    processing = processing or ProcessingOptions()
    groups = report_groups or CROSS_BENCHMARK_REPORT_GROUPS
    order = report_order or CROSS_BENCHMARK_REPORT_ORDER
    notes = notes_by_name or CROSS_BENCHMARK_NOTES
    raw_names = dataset_names or sorted({name for sources in groups.values() for name in sources})

    all_names = sorted(set(raw_names) | {name for sources in groups.values() for name in sources})
    cache = _load_all_cross_benchmark_corpora(all_names, processing=processing, corpus_cache=corpus_cache)

    rows = [
        summarize_features(dataset_name, list(cache[dataset_name].features))
        for dataset_name in raw_names
    ]
    profiles: list[CrossBenchmarkProfile] = []
    for display_name in order:
        source_datasets = groups[display_name]
        features: list[ExpressionFeatures] = []
        token_sequences: list[tuple[str, ...]] = []
        for dataset_name in source_datasets:
            enriched = cache[dataset_name]
            features.extend(enriched.features)
            token_sequences.extend(enriched.token_sequences)
        profiles.append(
            build_cross_benchmark_profile(
                display_name,
                features,
                token_sequences,
                notes=notes.get(display_name, ""),
            )
        )
    return profiles, rows


def summarize_features(dataset: str, features: list[ExpressionFeatures]) -> CrossBenchmarkRow:
    counter = corpus_token_counter(features)
    lengths = [feature.token_length for feature in features]
    expression_count = len(features)
    return CrossBenchmarkRow(
        dataset=dataset,
        expression_count=expression_count,
        vocabulary_size=len(counter),
        total_token_count=sum(counter.values()),
        duplicate_rate=_duplicate_rate(features),
        mean_length=sum(lengths) / expression_count if expression_count else 0.0,
        p50_length=percentile(lengths, 50),
        p90_length=percentile(lengths, 90),
        p95_length=percentile(lengths, 95),
        p99_length=percentile(lengths, 99),
        max_length=max(lengths) if lengths else 0,
        gini=gini_coefficient(list(counter.values())),
        top_10_coverage=top_k_coverage(counter, 10),
        unknown_token_ratio=_unknown_token_ratio(counter),
        rare_token_occurrence_ratio=_rare_token_occurrence_ratio(counter),
        mean_structure_type_count=(
            sum(feature.structure_type_count for feature in features) / expression_count if expression_count else 0.0
        ),
        mean_ast_depth=sum(feature.ast_depth for feature in features) / expression_count if expression_count else 0.0,
        parse_success_rate=parse_success_rate(features),
        vocab_coverage=_vocab_coverage(counter),
    )


def build_cross_benchmark_profile(
    display_name: str,
    features: list[ExpressionFeatures],
    token_sequences: list[tuple[str, ...]],
    *,
    notes: str,
) -> CrossBenchmarkProfile:
    counter = corpus_token_counter(features)
    lengths = [feature.token_length for feature in features]
    ast_depths = [feature.ast_depth for feature in features]
    structure_counts = [feature.structure_type_count for feature in features]
    expression_count = len(features)

    structure_complexity = compute_ocr_structure_complexity_from_counts(structure_counts)
    structure_distribution = compute_ocr_structure_distribution_from_token_sequences(token_sequences)
    matrix_row = next(
        row for row in structure_distribution.rows if row.structure_type == MATRIX_STRUCTURE_TYPE
    )

    duplicate_stats = duplicate_stats_from_features(features)
    total_token_count, taxonomy_counts, taxonomy_shares = _taxonomy_category_stats(token_sequences)
    return CrossBenchmarkProfile(
        display_name=display_name,
        expression_count=expression_count,
        unique_expression_count=duplicate_stats.unique_normalized_latex_count,
        duplicate_rate=duplicate_stats.duplicate_rate,
        vocabulary_size=len(counter),
        parse_success_rate=parse_success_rate(features),
        mean_length=sum(lengths) / expression_count if expression_count else 0.0,
        p50_length=percentile(lengths, 50),
        p90_length=percentile(lengths, 90),
        p95_length=percentile(lengths, 95),
        p99_length=percentile(lengths, 99),
        max_length=max(lengths) if lengths else 0,
        length_bins=_length_bin_stats(features),
        gini=gini_coefficient(list(counter.values())),
        top_10_coverage=top_k_coverage(counter, 10),
        top_50_coverage=top_k_coverage(counter, 50),
        top_100_coverage=top_k_coverage(counter, 100),
        rare_1_vocab_ratio=rare_vocab_ratio(counter, 1),
        rare_5_vocab_ratio=rare_vocab_ratio(counter, 5),
        rare_10_expression_ratio=rare_expression_ratio_from_token_sequences(token_sequences, counter, 10),
        structural_expression_ratio=structure_complexity.structural_expression_ratio,
        mean_structure_type_count=structure_complexity.mean_structure_type_count,
        multi_structure_ratio_ge_2=structure_complexity.multi_structure_ratio_ge_2,
        multi_structure_ratio_ge_3=structure_complexity.multi_structure_ratio_ge_3,
        matrix_structure_ratio=matrix_row.expression_ratio,
        mean_ast_depth=sum(ast_depths) / expression_count if expression_count else 0.0,
        p50_ast_depth=percentile(ast_depths, 50),
        p90_ast_depth=percentile(ast_depths, 90),
        max_ast_depth=max(ast_depths) if ast_depths else 0,
        ast_ge_3_ratio=sum(depth >= 3 for depth in ast_depths) / expression_count if expression_count else 0.0,
        count_gt_40_tokens=sum(length > 40 for length in lengths),
        count_gt_80_tokens=sum(length > 80 for length in lengths),
        count_any_structure=sum(count >= 1 for count in structure_counts),
        count_multi_struct_ge_3=sum(count >= 3 for count in structure_counts),
        structure_type_count_bins=_structure_type_count_bins(structure_counts),
        count_matrix_structure=sum(
            MATRIX_STRUCTURE_TYPE in feature.structure_types for feature in features
        ),
        count_ast_ge_3=sum(depth >= 3 for depth in ast_depths),
        ast_depth_counts=_ast_depth_count_bins(features),
        count_gt_40_and_ast_ge_2=sum(
            feature.token_length > 40 and feature.ast_depth >= 2 for feature in features
        ),
        count_gt_80_and_ast_ge_3=sum(
            feature.token_length > 80 and feature.ast_depth >= 3 for feature in features
        ),
        total_token_count=total_token_count,
        taxonomy_token_counts=taxonomy_counts,
        english_token_ratio=taxonomy_shares[0],
        digit_token_ratio=taxonomy_shares[1],
        greek_token_ratio=taxonomy_shares[2],
        special_symbol_token_ratio=taxonomy_shares[3],
        operator_token_ratio=taxonomy_shares[4],
        grouping_token_ratio=taxonomy_shares[5],
        structural_token_ratio=taxonomy_shares[6],
        cjk_token_ratio=taxonomy_shares[7],
        other_unknown_token_ratio=taxonomy_shares[8],
        notes=notes,
        structural_difficulty_counts=_structural_difficulty_count_bins(features),
    )


def compute_cross_benchmark_profiles(
    *,
    processing: ProcessingOptions | None = None,
    report_groups: dict[str, tuple[str, ...]] | None = None,
    report_order: tuple[str, ...] | None = None,
    notes_by_name: dict[str, str] | None = None,
    corpus_cache: dict[str, EnrichedCorpus] | None = None,
) -> list[CrossBenchmarkProfile]:
    profiles, _rows = compute_cross_benchmark_results(
        processing=processing,
        report_groups=report_groups,
        report_order=report_order,
        notes_by_name=notes_by_name,
        corpus_cache=corpus_cache,
    )
    return profiles


def compute_cross_benchmark_rows(
    dataset_names: list[str],
    *,
    processing: ProcessingOptions | None = None,
    corpus_cache: dict[str, EnrichedCorpus] | None = None,
) -> list[CrossBenchmarkRow]:
    _profiles, rows = compute_cross_benchmark_results(
        processing=processing,
        dataset_names=dataset_names,
        corpus_cache=corpus_cache,
    )
    return rows


def length_bin_rows_for_features(dataset: str, features: list[ExpressionFeatures]) -> list[tuple[str, str, int, float]]:
    counts = {spec.label: 0 for spec in DEFAULT_LENGTH_BINS}
    for feature in features:
        counts[assign_length_bin(feature.token_length)] += 1
    total = len(features)
    return [
        (dataset, spec.label, counts[spec.label], counts[spec.label] / total if total else 0.0)
        for spec in DEFAULT_LENGTH_BINS
    ]


def length_bin_rows_for_profile(profile: CrossBenchmarkProfile) -> list[tuple[str, str, int, float]]:
    return [
        (profile.display_name, item.label, item.count, item.share)
        for item in profile.length_bins
    ]
