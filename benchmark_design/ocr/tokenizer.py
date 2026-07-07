"""Greedy dictionary-based tokenization using the LaTeX command catalogue."""

from __future__ import annotations

from functools import lru_cache

from benchmark_design.commands import LATEX_DICT

_MAX_MATCH_LEN = 32


def build_latex_vocab() -> set[str]:
    """Return the LaTeX command vocabulary used for greedy matching."""
    return set(LATEX_DICT)


@lru_cache(maxsize=1)
def _cached_vocab() -> frozenset[str]:
    return frozenset(LATEX_DICT)


def tokenize_greedy(text: str, vocab: set[str] | frozenset[str] | None = None) -> list[str]:
    """Tokenize *text* with longest-match-first over *vocab*, falling back to single chars."""
    if not text:
        return []

    token_vocab = _cached_vocab() if vocab is None else frozenset(vocab)
    pieces: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        if text[index].isspace():
            index += 1
            continue

        matched: str | None = None
        end = min(length, index + _MAX_MATCH_LEN)
        for j in range(end, index, -1):
            candidate = text[index:j]
            if candidate in token_vocab:
                matched = candidate
                break

        if matched is None:
            pieces.append(text[index])
            index += 1
        else:
            pieces.append(matched)
            index += len(matched)

    return pieces


def tokenize_greedy_with_spans(
    text: str,
    vocab: set[str] | frozenset[str] | None = None,
) -> list[tuple[str, int, int]]:
    """Tokenize *text* and return ``(token, start, end)`` spans over the original string."""
    if not text:
        return []

    token_vocab = _cached_vocab() if vocab is None else frozenset(vocab)
    pieces: list[tuple[str, int, int]] = []
    index = 0
    length = len(text)

    while index < length:
        if text[index].isspace():
            index += 1
            continue

        matched: str | None = None
        end = min(length, index + _MAX_MATCH_LEN)
        for j in range(end, index, -1):
            candidate = text[index:j]
            if candidate in token_vocab:
                matched = candidate
                break

        if matched is None:
            pieces.append((text[index], index, index + 1))
            index += 1
        else:
            pieces.append((matched, index, index + len(matched)))
            index += len(matched)

    return pieces
