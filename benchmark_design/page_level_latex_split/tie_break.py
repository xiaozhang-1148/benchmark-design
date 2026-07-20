"""Deterministic tie-breaking keys for stratified split assignment."""

from __future__ import annotations

import hashlib

from benchmark_design.page_level_latex_split.config import SplitConfig


def seeded_hash_tie_key(seed: int, page_id: str) -> int:
    """Stable pseudo-random order from (seed, page_id); independent of ID lexicographic order."""
    digest = hashlib.sha256(f"{seed}:{page_id}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def page_tie_key(config: SplitConfig, seed: int, page_id: str) -> tuple[int | str, ...]:
    mode = config.tie_break
    if mode == "seeded_hash":
        return (seeded_hash_tie_key(seed, page_id), page_id)
    if mode == "page_id_asc":
        return (page_id,)
    raise ValueError(f"unsupported tie_break mode: {mode!r}")


def sort_key_for_page(config: SplitConfig, seed: int, page_id: str) -> tuple[int | str, ...]:
    return page_tie_key(config, seed, page_id)
