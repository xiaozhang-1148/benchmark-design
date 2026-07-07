"""Mutually exclusive token taxonomy for OCR corpus composition."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from benchmark_design import commands as cmd
from benchmark_design.io.benchmark_loader import ExpressionRecord
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.tokenizer import tokenize_greedy


class TokenCategory(StrEnum):
    LATIN_VARIABLE = "latin variable tokens"
    DIGIT = "digit tokens"
    SPECIAL_SYMBOL = "special symbol tokens"
    OPERATOR = "operator tokens"
    GROUPING = "grouping tokens"
    STRUCTURAL = "structural tokens"
    CJK = "CJK tokens"
    PUNCTUATION = "punctuation tokens"
    LAYOUT_ALIGNMENT = "layout / alignment tokens"
    OTHER = "other / unknown tokens"


PUNCTUATION_TOKENS: frozenset[str] = frozenset({",", ":", ".", ";", "!", "?", "'", "、", "。"})
LAYOUT_ALIGNMENT_TOKENS: frozenset[str] = frozenset({r"\\", "&", "\\"})

TOKEN_CATEGORY_ORDER: tuple[TokenCategory, ...] = (
    TokenCategory.LATIN_VARIABLE,
    TokenCategory.DIGIT,
    TokenCategory.SPECIAL_SYMBOL,
    TokenCategory.OPERATOR,
    TokenCategory.GROUPING,
    TokenCategory.STRUCTURAL,
    TokenCategory.CJK,
    TokenCategory.PUNCTUATION,
    TokenCategory.LAYOUT_ALIGNMENT,
    TokenCategory.OTHER,
)


def _is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def build_taxonomy_sets() -> dict[TokenCategory, frozenset[str]]:
    """Build mutually exclusive lookup sets derived from ``commands.py`` groups."""
    grouping = frozenset(
        {"(", ")", "{", "}", "[", "]"}
        | {
            r"\left",
            r"\right",
            r"\(",
            r"\)",
            r"\[",
            r"\]",
            r"\{",
            r"\}",
            r"\langle",
            r"\rangle",
            r"\|",
            "|",
            r"\lvert",
            r"\rvert",
            r"\lceil",
            r"\rceil",
            r"\lbrace",
            r"\rbrace",
            r"\lfloor",
            r"\rfloor",
            "''",
        }
    )

    structural = frozenset(
        {"^", "_"}
        | cmd._FRACTION_BINOMIAL
        | cmd._ENVIRONMENTS
        | cmd._ACCENTS_STYLE
        | {
            r"\sqrt",
            r"\sum",
            r"\int",
            r"\iint",
            r"\iiint",
            r"\iiiint",
            r"\oint",
            r"\lim",
            r"\limsup",
            r"\liminf",
            r"\limits",
            r"\binom",
        }
    )

    operators = frozenset(
        {"+", "-", "=", ">", "<", "*", "/"}
        | cmd._BINARY_RELATIONS
        | {r"\pm", r"\mp", r"\gt", r"\lt"}
    )

    structural_ops = frozenset(
        {
            r"\sqrt",
            r"\sum",
            r"\int",
            r"\iint",
            r"\iiint",
            r"\iiiint",
            r"\oint",
            r"\lim",
            r"\limsup",
            r"\liminf",
            r"\limits",
        }
    )

    special = frozenset(
        cmd._GREEK_LOWER
        | cmd._GREEK_UPPER
        | cmd._GREEK_VARIANTS
        | cmd._ARROWS
        | cmd._MISC_SYMBOLS
        | (cmd._OPERATORS - structural_ops)
        | (cmd._SPACING_PUNCT - {r"\\", r"\,", r"\quad", r"\qquad", r"---"})
        | cmd._CUSTOM_SYMBOLS
        | cmd._INCORRECT_CMD_VARIANTS
        | cmd._INCORRECT_DELETE_VARIANTS
    )

    structural = frozenset(set(structural) - set(grouping))
    operators = frozenset(set(operators) - set(grouping) - set(structural))
    special = frozenset(set(special) - set(grouping) - set(structural) - set(operators))

    return {
        TokenCategory.GROUPING: grouping,
        TokenCategory.STRUCTURAL: structural,
        TokenCategory.OPERATOR: operators,
        TokenCategory.SPECIAL_SYMBOL: special,
        TokenCategory.PUNCTUATION: PUNCTUATION_TOKENS,
        TokenCategory.LAYOUT_ALIGNMENT: LAYOUT_ALIGNMENT_TOKENS,
    }


_TAXONOMY_LOOKUP: dict[str, TokenCategory] | None = None


def _taxonomy_lookup() -> Mapping[str, TokenCategory]:
    global _TAXONOMY_LOOKUP
    if _TAXONOMY_LOOKUP is None:
        lookup: dict[str, TokenCategory] = {}
        for category, tokens in build_taxonomy_sets().items():
            for token in tokens:
                lookup[token] = category
        _TAXONOMY_LOOKUP = lookup
    return _TAXONOMY_LOOKUP


def classify_token(token: str) -> TokenCategory:
    """Assign *token* to exactly one taxonomy category."""
    if len(token) == 1:
        char = token
        if _is_cjk_char(char):
            return TokenCategory.CJK
        if char.isdigit():
            return TokenCategory.DIGIT
        if char.isalpha() and char.isascii():
            return TokenCategory.LATIN_VARIABLE

    lookup = _taxonomy_lookup()
    if token in lookup:
        return lookup[token]
    return TokenCategory.OTHER


@dataclass(frozen=True, slots=True)
class TokenCategoryCount:
    category: TokenCategory
    count: int
    share: float


@dataclass(frozen=True, slots=True)
class OcrTokenTaxonomyMetrics:
    total_token_count: int
    categories: tuple[TokenCategoryCount, ...]

    @property
    def other_unknown_ratio(self) -> float:
        for item in self.categories:
            if item.category is TokenCategory.OTHER:
                return item.share
        return 0.0

    def as_rows(self) -> list[tuple[str, int, float]]:
        return [(item.category.value, item.count, item.share) for item in self.categories]


def compute_ocr_token_taxonomy_from_token_sequences(
    token_sequences: Iterable[Sequence[str]],
) -> OcrTokenTaxonomyMetrics:
    counter: Counter[TokenCategory] = Counter()
    for tokens in token_sequences:
        for token in tokens:
            counter[classify_token(token)] += 1

    total = sum(counter.values())
    categories = tuple(
        TokenCategoryCount(
            category=category,
            count=counter[category],
            share=counter[category] / total if total else 0.0,
        )
        for category in TOKEN_CATEGORY_ORDER
    )
    return OcrTokenTaxonomyMetrics(total_token_count=total, categories=categories)


def compute_ocr_token_taxonomy_from_tokens(tokens: Iterable[str]) -> OcrTokenTaxonomyMetrics:
    counter: Counter[TokenCategory] = Counter()
    for token in tokens:
        counter[classify_token(token)] += 1

    total = sum(counter.values())
    categories = tuple(
        TokenCategoryCount(
            category=category,
            count=counter[category],
            share=counter[category] / total if total else 0.0,
        )
        for category in TOKEN_CATEGORY_ORDER
    )
    return OcrTokenTaxonomyMetrics(total_token_count=total, categories=categories)


def iter_corpus_tokens(expressions: Iterable[ExpressionRecord]) -> Iterable[str]:
    for record in expressions:
        yield from tokenize_greedy(record.ocr)


def compute_ocr_token_taxonomy_from_expressions(
    expressions: Iterable[ExpressionRecord],
) -> OcrTokenTaxonomyMetrics:
    return compute_ocr_token_taxonomy_from_tokens(iter_corpus_tokens(expressions))


def compute_ocr_token_taxonomy(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> OcrTokenTaxonomyMetrics:
    from benchmark_design.ocr.processing import build_tokenized_corpus

    corpus = build_tokenized_corpus(input_dir, processing)
    return compute_ocr_token_taxonomy_from_token_sequences(corpus.token_sequences)
