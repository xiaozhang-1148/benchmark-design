"""Regression tests for corpus-global feature labels under parallel extraction."""

from __future__ import annotations

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.processing import ProcessingOptions, extract_features_parallel
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy


def _synthetic_corpus() -> tuple[tuple[ExpressionRecord, ...], tuple[tuple[str, ...], ...]]:
    vocab = frozenset(build_latex_vocab())
    records: list[ExpressionRecord] = []
    tokens: list[tuple[str, ...]] = []
    for index in range(8):
        ocr = "x+y" if index % 2 == 0 else f"a_{index}"
        records.append(
            ExpressionRecord(
                image_name=f"img_{index}",
                block_order=0,
                line_order=0,
                block_type="",
                ocr=ocr,
                dataset="synthetic",
                expression_id=f"synthetic:{index}",
                line_id="0",
            )
        )
        tokens.append(tuple(tokenize_greedy(ocr, vocab)))
    return tuple(records), tuple(tokens)


def test_extract_features_parallel_matches_single_worker() -> None:
    expressions, token_sequences = _synthetic_corpus()
    baseline = extract_features_parallel(
        expressions,
        token_sequences,
        ProcessingOptions(workers=1, show_progress=False),
    )
    for workers in (2, 4, 8):
        parallel = extract_features_parallel(
            expressions,
            token_sequences,
            ProcessingOptions(workers=workers, show_progress=False),
        )
        assert len(parallel) == len(baseline)
        for left, right in zip(baseline, parallel, strict=True):
            assert left.is_duplicate == right.is_duplicate
            assert left.duplicate_count == right.duplicate_count
            assert left.duplicate_group_id == right.duplicate_group_id
            assert left.has_rare_1 == right.has_rare_1
            assert left.has_rare_5 == right.has_rare_5
            assert left.has_rare_10 == right.has_rare_10
