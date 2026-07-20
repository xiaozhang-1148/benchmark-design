"""Detect valid matrix-style LaTeX environments in token sequences."""

from __future__ import annotations

from dataclasses import dataclass

MATRIX_ENVIRONMENT_NAMES: frozenset[str] = frozenset(
    {
        "cases",
        "pmatrix",
        "bmatrix",
        "Bmatrix",
        "vmatrix",
        "Vmatrix",
        "matrix",
        "array",
        "rcases",
    }
)

MATRIX_STRUCTURE_TRIGGER_TOKENS = (
    r"\begin{env} \\ \end{env}, env ∈ {cases, pmatrix, bmatrix, Bmatrix, "
    r"vmatrix, Vmatrix, matrix, array, rcases}"
)


@dataclass(frozen=True, slots=True)
class MatrixEnvironmentStats:
    count: int
    max_depth: int


def _parse_begin_env(tokens: list[str], index: int) -> tuple[str, int] | None:
    if index + 3 >= len(tokens):
        return None
    if tokens[index] != r"\begin":
        return None
    if tokens[index + 1] != "{":
        return None
    env = tokens[index + 2]
    if tokens[index + 3] != "}":
        return None
    if env not in MATRIX_ENVIRONMENT_NAMES:
        return None
    return env, index + 4


def _parse_end_env(tokens: list[str], index: int) -> str | None:
    if index + 3 >= len(tokens):
        return None
    if tokens[index] != r"\end":
        return None
    if tokens[index + 1] != "{":
        return None
    return tokens[index + 2]


def _after_end_env(index: int) -> int:
    return index + 4


def find_valid_matrix_environment_span(tokens: list[str], start: int) -> tuple[int, int, int] | None:
    """Return ``(body_start, end_begin, index_after_block)`` for a valid matrix env at *start*."""
    begin = _parse_begin_env(tokens, start)
    if begin is None:
        return None
    env, body_start = begin

    stack: list[tuple[str, bool]] = [(env, False)]
    index = body_start
    while index < len(tokens):
        nested_begin = _parse_begin_env(tokens, index)
        if nested_begin is not None:
            stack.append((nested_begin[0], False))
            index = nested_begin[1]
            continue

        if tokens[index] == r"\\":
            stack = [(environment, True) for environment, _has_row_break in stack]
            index += 1
            continue

        end_env = _parse_end_env(tokens, index)
        if end_env is not None and stack and stack[-1][0] == end_env:
            if len(stack) == 1:
                _, has_row_break = stack.pop()
                after = _after_end_env(index)
                if has_row_break:
                    return body_start, index, after
                return None
            stack.pop()
            index = _after_end_env(index)
            continue

        index += 1
    return None


def matrix_environment_stats(tokens: list[str]) -> MatrixEnvironmentStats:
    """Count valid matrix environments and their maximum nesting depth."""
    count = 0
    max_depth = 0
    stack: list[tuple[str, bool, int]] = []
    index = 0
    while index < len(tokens):
        begin = _parse_begin_env(tokens, index)
        if begin is not None:
            environment, next_index = begin
            stack.append((environment, False, len(stack) + 1))
            index = next_index
            continue

        end_env = _parse_end_env(tokens, index)
        if end_env is not None and stack and stack[-1][0] == end_env:
            environment, has_row_break, depth = stack.pop()
            index = _after_end_env(index)
            if has_row_break:
                count += 1
                max_depth = max(max_depth, depth)
            continue

        if tokens[index] == r"\\":
            stack = [(environment, True, depth) for environment, _has_row_break, depth in stack]
            index += 1
            continue

        index += 1
    return MatrixEnvironmentStats(count=count, max_depth=max_depth)


def expression_has_matrix_environment(tokens: list[str]) -> bool:
    return matrix_environment_stats(tokens).count > 0
