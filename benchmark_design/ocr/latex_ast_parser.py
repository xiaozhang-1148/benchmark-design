"""Structural LaTeX AST validation for Parse OK (atom/group arguments)."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_design.ocr.position_forest import FRACTION_TRIGGERS
from benchmark_design.ocr.token_taxonomy import TokenCategory, classify_token

SUPERSCRIPT_TRIGGERS: frozenset[str] = frozenset({"^"})
SUBSCRIPT_TRIGGERS: frozenset[str] = frozenset({"_"})
SCRIPT_TRIGGERS: frozenset[str] = SUPERSCRIPT_TRIGGERS | SUBSCRIPT_TRIGGERS
SQRT_TRIGGERS: frozenset[str] = frozenset({r"\sqrt"})
LIMIT_OPERATOR_TRIGGERS: frozenset[str] = frozenset(
    {
        r"\sum",
        r"\int",
        r"\iint",
        r"\iiint",
        r"\iiiint",
        r"\oint",
        r"\lim",
        r"\limsup",
        r"\liminf",
    }
)
STRUCTURE_PRIMARY_TRIGGERS: frozenset[str] = (
    FRACTION_TRIGGERS | SQRT_TRIGGERS | LIMIT_OPERATOR_TRIGGERS
)


class ParseError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(slots=True)
class _TokenStream:
    tokens: list[str]
    index: int = 0

    def peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def consume(self) -> str:
        token = self.peek()
        if token is None:
            raise ParseError("incomplete_substructure")
        self.index += 1
        return token

    def at_end(self) -> bool:
        return self.index >= len(self.tokens)


def dictionary_ok(tokens: list[str]) -> tuple[bool, str | None]:
    """Return whether every token is covered by the frozen HMER vocabulary.

    Coverage follows the unified tokenizer + taxonomy: ``LATEX_DICT`` entries plus
    single-character atoms (Latin, digits, CJK), structural/grouping tokens, and
    explicit punctuation. Residue classified as ``other / unknown tokens`` fails.
    """
    for token in tokens:
        if classify_token(token) is TokenCategory.OTHER:
            return False, token
    return True, None


def _parse_group(stream: _TokenStream) -> None:
    if stream.consume() != "{":
        raise ParseError("unbalanced_braces")
    if stream.peek() == "}":
        stream.consume()
        return
    _parse_math_list(stream)
    if stream.peek() != "}":
        raise ParseError("unbalanced_braces")
    stream.consume()


def _parse_bracket_content(stream: _TokenStream) -> None:
    """Parse ``[...]`` optional sqrt root until the closing ``]``."""
    while stream.peek() not in ("]", None):
        if stream.peek() in SCRIPT_TRIGGERS:
            _parse_script_suffixes(stream)
        else:
            _parse_atom_with_scripts(stream)


def _parse_optional_sqrt_root(stream: _TokenStream) -> None:
    if stream.peek() != "[":
        return
    stream.consume()
    _parse_bracket_content(stream)
    if stream.peek() != "]":
        raise ParseError("unbalanced_braces")
    stream.consume()


def _parse_script_argument(stream: _TokenStream) -> None:
    if stream.at_end():
        raise ParseError("incomplete_substructure")
    if stream.peek() == "{":
        _parse_group(stream)
        return
    if stream.peek() in SCRIPT_TRIGGERS:
        _parse_script_suffixes(stream)
        return
    _parse_primary(stream)


def _parse_script_suffixes(stream: _TokenStream) -> None:
    while stream.peek() in SCRIPT_TRIGGERS:
        stream.consume()
        _parse_script_argument(stream)


def _parse_primary(stream: _TokenStream) -> None:
    token = stream.peek()
    if token is None:
        raise ParseError("incomplete_substructure")
    if token == "{":
        _parse_group(stream)
        return
    if token in FRACTION_TRIGGERS:
        stream.consume()
        _parse_script_argument(stream)
        _parse_script_argument(stream)
        return
    if token in SQRT_TRIGGERS:
        stream.consume()
        _parse_optional_sqrt_root(stream)
        _parse_script_argument(stream)
        return
    if token in LIMIT_OPERATOR_TRIGGERS:
        stream.consume()
        _parse_script_suffixes(stream)
        return
    if token in SCRIPT_TRIGGERS:
        raise ParseError("incomplete_substructure")
    stream.consume()


def _parse_atom_with_scripts(stream: _TokenStream) -> None:
    _parse_primary(stream)
    _parse_script_suffixes(stream)


def _parse_math_list(stream: _TokenStream) -> None:
    while not stream.at_end() and stream.peek() != "}":
        if stream.peek() in SCRIPT_TRIGGERS:
            _parse_script_suffixes(stream)
        else:
            _parse_atom_with_scripts(stream)


def validate_structural_ast(tokens: list[str]) -> str | None:
    """Return an error code when structural AST construction fails, else ``None``.

    Structural failures are limited to:
    - ``unbalanced_braces`` — unmatched or incorrectly nested ``{`` / ``}``
    - ``incomplete_substructure`` — a structure operator (``^``, ``_``, ``\\frac``,
      ``\\sqrt``, limits, …) is missing its atom/group argument
    """
    if not tokens:
        return None
    stream = _TokenStream(list(tokens))
    try:
        _parse_math_list(stream)
    except ParseError as error:
        return error.code
    if not stream.at_end():
        return "incomplete_substructure"
    return None
