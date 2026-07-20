"""Two-layer Parse OK validation: dictionary membership and structural AST."""

from __future__ import annotations

from benchmark_design.ocr.latex_ast_parser import (
    dictionary_ok,
    validate_structural_ast,
)


def brace_balance_depth(tokens: list[str]) -> int:
    depth = 0
    for token in tokens:
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
    return depth


def has_incomplete_substructure(tokens: list[str]) -> bool:
    """Legacy helper retained for tests; prefer ``validate_structural_ast``."""
    return validate_structural_ast(tokens) == "incomplete_substructure"


def validate_parse_status(tokens: list[str]) -> str:
    """Return parse status: ``ok`` only when dictionary and structural AST checks pass."""
    if not tokens:
        return "ok"

    in_dictionary, _missing = dictionary_ok(tokens)
    if not in_dictionary:
        return "unknown_token"

    structural_error = validate_structural_ast(tokens)
    if structural_error is not None:
        return structural_error
    return "ok"
