"""Validate LaTeX token sequences for brace balance and substructure closure."""

from __future__ import annotations

from benchmark_design.ocr.position_forest import (
    FRACTION_TRIGGERS,
    SINGLE_ARG_TRIGGERS,
    _find_fraction_arg_ends,
    _find_group_end,
    _find_single_arg_substructure_end,
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
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in SINGLE_ARG_TRIGGERS:
            sub_end = _find_single_arg_substructure_end(tokens, index)
            if index + 1 >= len(tokens):
                return True
            if tokens[index + 1] == "{" and sub_end < index + 1:
                return True
            index = sub_end + 1
        elif token in FRACTION_TRIGGERS:
            first_end, second_end = _find_fraction_arg_ends(tokens, index)
            if first_end <= index or second_end <= first_end:
                return True
            index = second_end + 1
        else:
            index += 1
    return False


def validate_parse_status(tokens: list[str]) -> str:
    if brace_balance_depth(tokens) != 0:
        return "unbalanced_braces"
    if has_incomplete_substructure(tokens):
        return "incomplete_substructure"
    return "ok"
