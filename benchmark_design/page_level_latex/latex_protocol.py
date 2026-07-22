"""Shared Chapter-5 / Chapter-6 LaTeX statistics protocol (HMER-compatible).

This module is the single entry point for:
- normalized LaTeX rules
- greedy tokenization
- \\delete retention (as special symbol tokens)
- token taxonomy
- unified structure-forest AST parsing
- nine structure types (six core + three extended)
- length-bin assignment
- Rare-10 corpus-frequency helpers

All call sites must go through these helpers so Chapter 5 and Chapter 6 agree on
AST depth, AST node count, and structure flags for the same expression.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from benchmark_design.ocr.duplicates import normalize_expression_latex
from benchmark_design.ocr.length_bin_specs import (
    DEFAULT_LENGTH_BINS,
    LengthBinSpec,
    assign_length_bin,
)
from benchmark_design.ocr.parse_validate import validate_parse_status
from benchmark_design.ocr.structure_forest import (
    AST_STRUCTURE_SPECS,
    STRUCTURE_TYPE_ORDER,
    compute_ast_forest_metrics,
)
from benchmark_design.ocr.token_taxonomy import (
    TOKEN_CATEGORY_ORDER,
    TokenCategory,
    classify_token,
)
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

DELETE_TOKENS: frozenset[str] = frozenset({r"\delete", r"\ddelete", r"\deleted"})

LENGTH_BIN_FIELD_KEYS: tuple[str, ...] = (
    "length_1_10",
    "length_11_20",
    "length_21_40",
    "length_41_80",
    "length_gt80",
)

LENGTH_BIN_LABEL_TO_KEY: dict[str, str] = {
    "1-10 tokens": "length_1_10",
    "11-20 tokens": "length_11_20",
    "21-40 tokens": "length_21_40",
    "41-80 tokens": "length_41_80",
    "> 80 tokens": "length_gt80",
}

LENGTH_BIN_KEY_TO_DISPLAY: dict[str, str] = {
    "length_1_10": "1–10",
    "length_11_20": "11–20",
    "length_21_40": "21–40",
    "length_41_80": "41–80",
    "length_gt80": ">80",
}

AST_DEPTH_FIELD_KEYS: tuple[str, ...] = (
    "ast_depth_0",
    "ast_depth_1",
    "ast_depth_2",
    "ast_depth_3",
    "ast_depth_4",
    "ast_depth_5",
    "ast_depth_gt5",
)

RARE10_THRESHOLD = 10

TAXONOMY_FIELD_KEYS: tuple[str, ...] = tuple(
    category.value.replace(" ", "_").replace("/", "_").replace("__", "_")
    for category in TOKEN_CATEGORY_ORDER
)

TAXONOMY_CATEGORY_TO_FIELD: dict[TokenCategory, str] = {
    category: field
    for category, field in zip(TOKEN_CATEGORY_ORDER, TAXONOMY_FIELD_KEYS, strict=True)
}


@dataclass(frozen=True, slots=True)
class StructureFlags:
    has_frac: bool
    has_sup: bool
    has_sub: bool
    has_sqrt: bool
    has_env: bool
    has_bigop: bool
    has_accent: bool
    has_stackrel: bool
    has_textcircled: bool

    @property
    def structure_type_count(self) -> int:
        return sum(
            (
                self.has_frac,
                self.has_sup,
                self.has_sub,
                self.has_sqrt,
                self.has_env,
                self.has_bigop,
                self.has_accent,
                self.has_stackrel,
                self.has_textcircled,
            )
        )

    def present_types(self) -> tuple[str, ...]:
        flags = (
            ("frac", self.has_frac),
            ("sup", self.has_sup),
            ("sub", self.has_sub),
            ("sqrt", self.has_sqrt),
            ("env", self.has_env),
            ("bigop", self.has_bigop),
            ("accent", self.has_accent),
            ("stackrel", self.has_stackrel),
            ("textcircled", self.has_textcircled),
        )
        return tuple(name for name, present in flags if present)


@dataclass(frozen=True, slots=True)
class ParsedExpression:
    raw_ocr_text: str
    normalized_latex: str
    tokens: tuple[str, ...]
    token_count: int
    ast_node_count: int
    ast_depth: int
    parse_status: str
    parse_ok: bool
    parse_error_count: int
    structure: StructureFlags
    structure_combination: str
    token_category_counts: dict[str, int]
    contains_delete: bool
    unknown_token_count: int


@lru_cache(maxsize=1)
def latex_vocab() -> frozenset[str]:
    return frozenset(build_latex_vocab())


def normalize_latex(raw: str) -> str:
    """Chapter-5 normalization: strip leading/trailing whitespace only."""
    return normalize_expression_latex(raw)


def handle_delete_content(normalized_latex: str) -> str:
    """Retain ``\\delete`` markers; Chapter-5 treats them as special symbol tokens."""
    return normalized_latex


def tokenize_latex(normalized_latex: str) -> tuple[str, ...]:
    return tuple(tokenize_greedy(normalized_latex, latex_vocab()))


def length_bin_for_token_count(token_count: int) -> tuple[str, str]:
    label = assign_length_bin(token_count, DEFAULT_LENGTH_BINS)
    return label, LENGTH_BIN_LABEL_TO_KEY[label]


def detect_structures(tokens: Sequence[str]) -> StructureFlags:
    metrics = compute_ast_forest_metrics(list(tokens))
    flags = metrics.structure_flags
    return StructureFlags(
        has_frac=flags["frac"],
        has_sup=flags["sup"],
        has_sub=flags["sub"],
        has_sqrt=flags["sqrt"],
        has_env=flags["env"],
        has_bigop=flags["bigop"],
        has_accent=flags["accent"],
        has_stackrel=flags["stackrel"],
        has_textcircled=flags["textcircled"],
    )


def structure_combination(flags: StructureFlags) -> str:
    present = flags.present_types()
    return "+".join(present)


def token_category_counts(tokens: Sequence[str]) -> dict[str, int]:
    counts = {field: 0 for field in TAXONOMY_FIELD_KEYS}
    for token in tokens:
        category = classify_token(token)
        counts[TAXONOMY_CATEGORY_TO_FIELD[category]] += 1
    return counts


def empty_token_category_counts() -> dict[str, int]:
    return {field: 0 for field in TAXONOMY_FIELD_KEYS}


def contains_delete_marker(raw_or_normalized: str, tokens: Sequence[str] | None = None) -> bool:
    if r"\delete" in raw_or_normalized or r"\ddelete" in raw_or_normalized or r"\deleted" in raw_or_normalized:
        return True
    if tokens is None:
        return False
    return any(token in DELETE_TOKENS for token in tokens)


def ast_depth_field_key(depth: int) -> str:
    if depth <= 5:
        return f"ast_depth_{depth}"
    return "ast_depth_gt5"


def parse_expression(raw_ocr_text: str) -> ParsedExpression:
    """Full Chapter-5 protocol pass for one OCR string."""
    normalized = handle_delete_content(normalize_latex(raw_ocr_text))
    tokens = tokenize_latex(normalized) if normalized else ()
    forest = compute_ast_forest_metrics(list(tokens)) if tokens else None
    parse_status = validate_parse_status(list(tokens)) if tokens else "ok"
    parse_ok = parse_status == "ok"
    structure = detect_structures(tokens) if tokens else StructureFlags(
        False, False, False, False, False, False, False, False, False
    )
    category_counts = token_category_counts(tokens) if tokens else empty_token_category_counts()
    unknown = category_counts.get(TAXONOMY_CATEGORY_TO_FIELD[TokenCategory.OTHER], 0)
    return ParsedExpression(
        raw_ocr_text=raw_ocr_text,
        normalized_latex=normalized,
        tokens=tokens,
        token_count=len(tokens),
        ast_node_count=int(forest.ast_node_count) if forest is not None else 0,
        ast_depth=int(forest.ast_depth) if forest is not None else 0,
        parse_status=parse_status,
        parse_ok=parse_ok,
        parse_error_count=0 if parse_ok else 1,
        structure=structure,
        structure_combination=structure_combination(structure),
        token_category_counts=category_counts,
        contains_delete=contains_delete_marker(normalized, tokens),
        unknown_token_count=int(unknown),
    )


def rare10_token_set(token_counter: Counter[str], *, threshold: int = RARE10_THRESHOLD) -> set[str]:
    return {token for token, count in token_counter.items() if count <= threshold}


def rare_tail_token_set(token_counter: Counter[str], *, fraction: float = 0.08) -> set[str]:
    """Return the rarest ``fraction`` of vocabulary tokens by corpus frequency."""
    if not token_counter:
        return set()
    items = sorted(token_counter.items(), key=lambda item: (item[1], item[0]))
    n = max(1, int(round(len(items) * float(fraction))))
    n = min(n, len(items))
    return {token for token, _ in items[:n]}


def rare10_occurrence_count(tokens: Sequence[str], rare_tokens: set[str]) -> int:
    return sum(1 for token in tokens if token in rare_tokens)


def is_valid_for_latex(raw_ocr_text: str, normalized_latex: str, tokens: Sequence[str]) -> tuple[bool, str]:
    """Text validity independent of polygon/geometry validity."""
    if not raw_ocr_text or not raw_ocr_text.strip():
        return False, "empty_annotation"
    if not normalized_latex:
        return False, "empty_after_normalization"
    if not tokens:
        return False, "empty_after_tokenization"
    return True, ""


def accumulate_token_counter(token_sequences: Iterable[Sequence[str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for tokens in token_sequences:
        counter.update(tokens)
    return counter


__all__ = [
    "AST_DEPTH_FIELD_KEYS",
    "AST_STRUCTURE_SPECS",
    "DEFAULT_LENGTH_BINS",
    "DELETE_TOKENS",
    "LENGTH_BIN_FIELD_KEYS",
    "LENGTH_BIN_KEY_TO_DISPLAY",
    "LENGTH_BIN_LABEL_TO_KEY",
    "ParsedExpression",
    "RARE10_THRESHOLD",
    "STRUCTURE_TYPE_ORDER",
    "StructureFlags",
    "TAXONOMY_CATEGORY_TO_FIELD",
    "TAXONOMY_FIELD_KEYS",
    "TOKEN_CATEGORY_ORDER",
    "TokenCategory",
    "accumulate_token_counter",
    "ast_depth_field_key",
    "classify_token",
    "contains_delete_marker",
    "detect_structures",
    "empty_token_category_counts",
    "handle_delete_content",
    "is_valid_for_latex",
    "latex_vocab",
    "length_bin_for_token_count",
    "normalize_latex",
    "parse_expression",
    "rare10_occurrence_count",
    "rare10_token_set",
    "structure_combination",
    "token_category_counts",
    "tokenize_latex",
]
