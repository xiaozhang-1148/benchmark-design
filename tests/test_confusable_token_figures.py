"""Tests for confusable token example figure selection."""

from __future__ import annotations

from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.confusable_tokens import PRIMARY_CONFUSABLE_GROUPS
from benchmark_design.ocr.expression_features import extract_single_features
from benchmark_design.ocr.tokenizer import tokenize_greedy
from benchmark_design.report.confusable_token_figures import select_confusable_line_examples


def _record(ocr: str, *, expression_id: str) -> ExpressionRecord:
    return ExpressionRecord(
        image_name="sample.jpg",
        block_order=0,
        line_order=0,
        block_type="Txtblock",
        ocr=ocr,
        expression_id=expression_id,
        line_polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0)),
    )


def _feature(record: ExpressionRecord) -> tuple:
    tokens = tuple(tokenize_greedy(record.ocr))
    return extract_single_features(
        record,
        tokens,
        duplicate_group_id=0,
        duplicate_count=1,
        rare_sets={1: set(), 5: set(), 10: set()},
    )


def test_select_confusable_line_examples_prefers_shorter_lines() -> None:
    short = _record("2 z", expression_id="short")
    long = _record("2 z + x + y + a + b + c + d", expression_id="long")
    features = [_feature(short), _feature(long)]
    expressions = [short, long]

    examples = select_confusable_line_examples(features, expressions)
    digit_letter = next(example for example in examples if example.group.name == "digit-letter")

    assert digit_letter.feature.expression_id == "short"
    assert digit_letter.token_type_label == "2 / z"


def test_select_confusable_line_examples_one_per_primary_group() -> None:
    record = _record(r"x \times 2", expression_id="mixed")
    features = [_feature(record)]
    expressions = [record]

    examples = select_confusable_line_examples(features, expressions)
    assert len(examples) <= len(PRIMARY_CONFUSABLE_GROUPS)
