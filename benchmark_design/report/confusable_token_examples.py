"""Export confusable-token example expressions for manual review."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark_design.ocr.confusable_tokens import (
    CONFUSABLE_EXAMPLE_COUNT_PER_TOKEN,
    CONFUSABLE_EXAMPLE_MIN_OCR_CHARS,
    ocr_non_whitespace_char_count,
    select_confusable_token_examples,
)
from benchmark_design.ocr.expression_features import ExpressionFeatures

GREEK_VARIANT_EXAMPLE_TOKENS: tuple[str, ...] = ("4", r"\varphi")


def write_confusable_token_examples_csv(
    features: list[ExpressionFeatures],
    output_path: Path,
    *,
    tokens: tuple[str, ...] = GREEK_VARIANT_EXAMPLE_TOKENS,
    min_ocr_chars: int = CONFUSABLE_EXAMPLE_MIN_OCR_CHARS,
    per_token: int = CONFUSABLE_EXAMPLE_COUNT_PER_TOKEN,
) -> int:
    rows = select_confusable_token_examples(
        features,
        tokens=tokens,
        min_ocr_chars=min_ocr_chars,
        per_token=per_token,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "group",
                "matched_token",
                "expression_id",
                "normalized_latex",
                "ocr_char_count",
                "token_length",
            ]
        )
        for rank, (token, feature) in enumerate(rows, start=1):
            writer.writerow(
                [
                    rank,
                    "greek-variant",
                    token,
                    feature.expression_id,
                    feature.normalized_latex,
                    ocr_non_whitespace_char_count(feature.normalized_latex),
                    feature.token_length,
                ]
            )
    return len(rows)
