"""Parallel corpus loading, tokenization, and feature extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from benchmark_design.io.benchmark_loader import ExpressionRecord, iter_benchmark_json_paths, load_expressions
from benchmark_design.io.dataset_loaders import load_dataset
from benchmark_design.ocr.expression_features import (
    ExpressionFeatures,
    build_corpus_feature_context,
    build_expression_features,
    extract_single_features,
)
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy
from benchmark_design.progress import parallel_map


@dataclass(frozen=True, slots=True)
class TokenizedCorpus:
    input_dir: Path
    dataset: str
    json_file_count: int
    expressions: tuple[ExpressionRecord, ...]
    token_sequences: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class EnrichedCorpus:
    input_dir: Path
    dataset: str
    json_file_count: int
    expressions: tuple[ExpressionRecord, ...]
    token_sequences: tuple[tuple[str, ...], ...]
    features: tuple[ExpressionFeatures, ...]


@lru_cache(maxsize=1)
def _cached_vocab() -> frozenset[str]:
    return frozenset(build_latex_vocab())


def _tokenize_chunk(records: tuple[ExpressionRecord, ...]) -> list[tuple[str, ...]]:
    vocab = _cached_vocab()
    return [tuple(tokenize_greedy(record.ocr, vocab)) for record in records]


def _chunk_records(
    records: Sequence[ExpressionRecord],
    *,
    workers: int,
) -> list[tuple[ExpressionRecord, ...]]:
    if not records:
        return []
    chunk_count = max(workers * 4, 1)
    chunk_size = max(1, (len(records) + chunk_count - 1) // chunk_count)
    return [tuple(records[index : index + chunk_size]) for index in range(0, len(records), chunk_size)]


def tokenize_expressions_parallel(
    expressions: Sequence[ExpressionRecord],
    options: ProcessingOptions,
) -> tuple[tuple[str, ...], ...]:
    if not expressions:
        return ()

    chunks = _chunk_records(expressions, workers=options.worker_count)
    if len(chunks) == 1 and options.worker_count <= 1:
        return tuple(_tokenize_chunk(chunks[0]))

    tokenized_chunks = parallel_map(
        _tokenize_chunk,
        chunks,
        description="Tokenizing expressions",
        show_progress=options.show_progress,
        workers=options.worker_count,
    )
    flattened: list[tuple[str, ...]] = []
    for chunk_tokens in tokenized_chunks:
        flattened.extend(chunk_tokens)
    return tuple(flattened)


def _feature_chunk(
    args: tuple[
        tuple[ExpressionRecord, ...],
        tuple[tuple[str, ...], ...],
        tuple[int, ...],
        tuple[int, ...],
        dict[int, set[str]],
    ],
) -> list[ExpressionFeatures]:
    records, tokens, group_ids, group_sizes, rare_sets = args
    return [
        extract_single_features(
            record,
            token_seq,
            duplicate_group_id=group_id,
            duplicate_count=group_size,
            rare_sets=rare_sets,
        )
        for record, token_seq, group_id, group_size in zip(
            records, tokens, group_ids, group_sizes, strict=True
        )
    ]


def extract_features_parallel(
    expressions: Sequence[ExpressionRecord],
    token_sequences: Sequence[tuple[str, ...]],
    options: ProcessingOptions,
) -> tuple[ExpressionFeatures, ...]:
    """Extract features with corpus-global duplicate / rare-token labels.

    Worker sharding must not recompute those labels per chunk, otherwise
    ``is_duplicate`` / ``has_rare_*`` depend on worker count.
    """
    if not expressions:
        return ()

    expression_list = list(expressions)
    token_list = list(token_sequences)
    if len(expression_list) != len(token_list):
        msg = (
            f"expression/token length mismatch: {len(expression_list)} vs {len(token_list)}"
        )
        raise ValueError(msg)

    if options.worker_count <= 1:
        return tuple(build_expression_features(expression_list, token_list))

    duplicate_index, rare_sets = build_corpus_feature_context(expression_list, token_list)
    record_chunks = _chunk_records(expression_list, workers=options.worker_count)
    token_chunks = _chunk_records(token_list, workers=options.worker_count)
    paired: list[
        tuple[
            tuple[ExpressionRecord, ...],
            tuple[tuple[str, ...], ...],
            tuple[int, ...],
            tuple[int, ...],
            dict[int, set[str]],
        ]
    ] = []
    offset = 0
    for records, tokens in zip(record_chunks, token_chunks, strict=True):
        end = offset + len(records)
        paired.append(
            (
                records,
                tokens,
                tuple(duplicate_index.group_id_by_index[offset:end]),
                tuple(duplicate_index.group_size_by_index[offset:end]),
                rare_sets,
            )
        )
        offset = end

    feature_chunks = parallel_map(
        _feature_chunk,
        paired,
        description="Extracting expression features",
        show_progress=options.show_progress,
        workers=options.worker_count,
    )
    flattened: list[ExpressionFeatures] = []
    for chunk in feature_chunks:
        flattened.extend(chunk)
    return tuple(flattened)


def build_tokenized_corpus(input_dir: Path, options: ProcessingOptions | None = None) -> TokenizedCorpus:
    return build_tokenized_corpus_for_dataset("ours", input_dir, options)


def build_tokenized_corpus_for_dataset(
    dataset_name: str,
    input_dir: Path,
    options: ProcessingOptions | None = None,
) -> TokenizedCorpus:
    processing = options or ProcessingOptions()
    if dataset_name == "ours":
        json_paths = iter_benchmark_json_paths(input_dir)
        json_file_count = len(json_paths)
    else:
        json_paths = []
        json_file_count = 0

    expressions = load_dataset(
        dataset_name,
        input_dir,
        show_progress=processing.show_progress,
        workers=processing.worker_count,
    )
    token_sequences = tokenize_expressions_parallel(expressions, processing)
    return TokenizedCorpus(
        input_dir=input_dir,
        dataset=dataset_name,
        json_file_count=json_file_count,
        expressions=tuple(expressions),
        token_sequences=token_sequences,
    )


def build_enriched_corpus(
    dataset_name: str,
    input_dir: Path,
    options: ProcessingOptions | None = None,
) -> EnrichedCorpus:
    processing = options or ProcessingOptions()
    tokenized = build_tokenized_corpus_for_dataset(dataset_name, input_dir, processing)
    features = extract_features_parallel(tokenized.expressions, tokenized.token_sequences, processing)
    return EnrichedCorpus(
        input_dir=input_dir,
        dataset=dataset_name,
        json_file_count=tokenized.json_file_count,
        expressions=tokenized.expressions,
        token_sequences=tokenized.token_sequences,
        features=features,
    )


_CORPUS_CACHE: dict[tuple[str, str], EnrichedCorpus] = {}


def clear_enriched_corpus_cache() -> None:
    """Clear the in-process enriched corpus cache (mainly for tests)."""
    _CORPUS_CACHE.clear()


def build_enriched_corpus_cached(
    dataset_name: str,
    input_dir: Path,
    options: ProcessingOptions | None = None,
    *,
    prebuilt: EnrichedCorpus | None = None,
) -> EnrichedCorpus:
    """Return a cached enriched corpus, optionally seeding with a pre-built instance."""
    resolved = str(Path(input_dir).resolve())
    cache_key = (dataset_name, resolved)
    if prebuilt is not None:
        _CORPUS_CACHE[cache_key] = prebuilt
        return prebuilt
    if cache_key in _CORPUS_CACHE:
        return _CORPUS_CACHE[cache_key]
    enriched = build_enriched_corpus(dataset_name, input_dir, options)
    _CORPUS_CACHE[cache_key] = enriched
    return enriched
