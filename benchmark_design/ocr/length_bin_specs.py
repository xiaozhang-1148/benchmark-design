"""Fixed token-length bin definitions (no processing dependencies)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LengthBinSpec:
    label: str
    min_tokens: int
    max_tokens: int | None


DEFAULT_LENGTH_BINS: tuple[LengthBinSpec, ...] = (
    LengthBinSpec("1-10 tokens", 1, 10),
    LengthBinSpec("11-20 tokens", 11, 20),
    LengthBinSpec("21-40 tokens", 21, 40),
    LengthBinSpec("41-80 tokens", 41, 80),
    LengthBinSpec("> 80 tokens", 81, None),
)


def assign_length_bin(length: int, bins: Sequence[LengthBinSpec] = DEFAULT_LENGTH_BINS) -> str:
    for spec in bins:
        if spec.max_tokens is None:
            if length >= spec.min_tokens:
                return spec.label
        elif spec.min_tokens <= length <= spec.max_tokens:
            return spec.label
    msg = f"length {length} does not match any fixed bin"
    raise ValueError(msg)
