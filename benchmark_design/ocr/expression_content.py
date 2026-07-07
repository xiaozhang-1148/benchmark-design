"""Expression-level content classification: pure LaTeX/math, pure CJK, or mixed."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from benchmark_design.ocr.token_taxonomy import TokenCategory, classify_token


class ExpressionContentKind(StrEnum):
    LATEX_COMMAND = "pure latex_command"
    CJK = "pure CJK"
    MIXED = "mixed"


EXPRESSION_CONTENT_KIND_ORDER: tuple[ExpressionContentKind, ...] = (
    ExpressionContentKind.LATEX_COMMAND,
    ExpressionContentKind.CJK,
    ExpressionContentKind.MIXED,
)


def classify_expression_content(tokens: Sequence[str]) -> ExpressionContentKind:
    """Classify one tokenized expression by CJK vs non-CJK token presence."""
    has_cjk = False
    has_non_cjk = False
    for token in tokens:
        if classify_token(token) is TokenCategory.CJK:
            has_cjk = True
        else:
            has_non_cjk = True
        if has_cjk and has_non_cjk:
            return ExpressionContentKind.MIXED
    if has_cjk:
        return ExpressionContentKind.CJK
    return ExpressionContentKind.LATEX_COMMAND


@dataclass(frozen=True, slots=True)
class ExpressionContentKindCount:
    kind: ExpressionContentKind
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class OcrExpressionContentMetrics:
    expression_count: int
    kinds: tuple[ExpressionContentKindCount, ...]

    def as_rows(self) -> list[tuple[str, int, float]]:
        return [(item.kind.value, item.count, item.share) for item in self.kinds]


def compute_ocr_expression_content_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
) -> OcrExpressionContentMetrics:
    counter: Counter[ExpressionContentKind] = Counter()
    for tokens in token_sequences:
        counter[classify_expression_content(tokens)] += 1

    total = sum(counter.values())
    kinds = tuple(
        ExpressionContentKindCount(
            kind=kind,
            count=counter[kind],
            share=counter[kind] / total if total else 0.0,
        )
        for kind in EXPRESSION_CONTENT_KIND_ORDER
    )
    return OcrExpressionContentMetrics(expression_count=total, kinds=kinds)
