"""PosFormer position-forest coding for LaTeX token sequences.

Reference: Guan et al., *PosFormer* (arXiv:2407.07764), Algorithm 1 and
Appendix A.1. Each token receives an identifier string over ``{M, L, R}``;
its nested level is ``len(identifier) - 1``. Expression structural complexity
is the maximum nested level among its tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

FRACTION_TRIGGERS: frozenset[str] = frozenset({r"\frac", r"\dfrac", r"\tfrac", r"\cfrac"})
SUPERSCRIPT_TRIGGERS: frozenset[str] = frozenset({"^"})
SUBSCRIPT_RADICAL_TRIGGERS: frozenset[str] = frozenset({"_", r"\sqrt"})
SINGLE_ARG_TRIGGERS: frozenset[str] = SUPERSCRIPT_TRIGGERS | SUBSCRIPT_RADICAL_TRIGGERS
STRUCTURE_TRIGGERS: frozenset[str] = SINGLE_ARG_TRIGGERS | FRACTION_TRIGGERS


@dataclass(frozen=True, slots=True)
class PositionForestEncoding:
    tokens: tuple[str, ...]
    identifiers: tuple[str, ...]
    nested_levels: tuple[int, ...]
    max_nested_level: int


def _find_group_end(tokens: list[str], open_idx: int) -> int:
    if open_idx >= len(tokens) or tokens[open_idx] != "{":
        return open_idx

    depth = 0
    for index, token in enumerate(tokens[open_idx:], start=open_idx):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(tokens) - 1


def _find_single_arg_substructure_end(tokens: list[str], trigger_idx: int) -> int:
    next_idx = trigger_idx + 1
    if next_idx >= len(tokens):
        return trigger_idx
    if tokens[next_idx] == "{":
        return _find_group_end(tokens, next_idx)
    return next_idx


def _find_fraction_arg_ends(tokens: list[str], trigger_idx: int) -> tuple[int, int]:
    index = trigger_idx + 1
    while index < len(tokens) and tokens[index] != "{":
        index += 1
    if index >= len(tokens):
        return trigger_idx, trigger_idx

    first_end = _find_group_end(tokens, index)
    index = first_end + 1
    while index < len(tokens) and tokens[index] != "{":
        index += 1
    if index >= len(tokens):
        return first_end, first_end
    second_end = _find_group_end(tokens, index)
    return first_end, second_end


def _append_identifier(
    identifiers: list[str],
    start: int,
    end: int,
    suffix: str,
) -> None:
    for index in range(start, end + 1):
        identifiers[index] += suffix


def _substruct_to_identifier(
    identifiers: list[str],
    start: int,
    end: int | tuple[int, int],
    trigger: str,
) -> None:
    if trigger in SUPERSCRIPT_TRIGGERS:
        last = end if isinstance(end, int) else end[1]
        _append_identifier(identifiers, start, last, "L")
        return

    if trigger in SUBSCRIPT_RADICAL_TRIGGERS:
        last = end if isinstance(end, int) else end[1]
        _append_identifier(identifiers, start, last, "R")
        return

    if trigger in FRACTION_TRIGGERS:
        first_end, second_end = end
        _append_identifier(identifiers, start, first_end, "L")
        _append_identifier(identifiers, first_end + 1, second_end, "R")


def _sequence_to_substruct(
    tokens: list[str],
    identifiers: list[str],
    start: int,
    end: int,
) -> None:
    index = start
    while index < end:
        token = tokens[index]
        if token in SINGLE_ARG_TRIGGERS:
            sub_end = _find_single_arg_substructure_end(tokens, index)
            _substruct_to_identifier(identifiers, index, sub_end, token)
            _sequence_to_substruct(tokens, identifiers, index + 1, sub_end)
            index = sub_end + 1
        elif token in FRACTION_TRIGGERS:
            first_end, second_end = _find_fraction_arg_ends(tokens, index)
            _substruct_to_identifier(identifiers, index, (first_end, second_end), token)
            _sequence_to_substruct(tokens, identifiers, index + 1, second_end)
            index = second_end + 1
        else:
            index += 1


def encode_position_forest_tokens(tokens: list[str]) -> PositionForestEncoding:
    """Apply PosFormer Algorithm 1 to a pre-tokenized LaTeX expression."""
    if not tokens:
        return PositionForestEncoding((), (), (), 0)

    identifiers = ["M"] * len(tokens)
    _sequence_to_substruct(tokens, identifiers, 0, len(tokens))
    nested_levels = tuple(len(identifier) - 1 for identifier in identifiers)
    max_nested_level = max(nested_levels) if nested_levels else 0
    return PositionForestEncoding(
        tokens=tuple(tokens),
        identifiers=tuple(identifiers),
        nested_levels=nested_levels,
        max_nested_level=max_nested_level,
    )


def max_nested_level(tokens: list[str]) -> int:
    """Return PosFormer structural complexity (max nested level) for *tokens*."""
    return encode_position_forest_tokens(tokens).max_nested_level
