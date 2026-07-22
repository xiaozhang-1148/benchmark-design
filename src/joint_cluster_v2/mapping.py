"""Load mapping YAML and resolve token → unique category."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

STRUCTURE_FEATURES = (
    "has_fraction",
    "has_root",
    "has_superscript",
    "has_subscript",
    "has_matrix",
    "has_cases",
    "has_over_under",
    "has_accent_vector",
)

CONTENT_FEATURES = (
    "has_relation",
    "has_trigonometric",
    "has_log_exponential",
    "has_calculus",
    "has_probability_statistics",
    "has_geometry",
    "has_vector_linear_algebra",
    "has_proof_language",
)

BINARY_FEATURES = STRUCTURE_FEATURES + CONTENT_FEATURES


def load_mapping(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_token_lookup(mapping: dict[str, Any]) -> dict[str, str]:
    """Each token → exactly one feature id. Structure before content; overrides last."""
    lookup: dict[str, str] = {}
    exclude = set(mapping.get("exclude_tokens") or [])

    def _add(feature: str, tokens: list[str]) -> None:
        for t in tokens or []:
            if t in exclude:
                continue
            if t not in lookup:
                lookup[t] = feature

    for feat in STRUCTURE_FEATURES:
        block = (mapping.get("structure") or {}).get(feat) or {}
        _add(feat, list(block.get("tokens") or []))
    for feat in CONTENT_FEATURES:
        block = (mapping.get("content") or {}).get(feat) or {}
        _add(feat, list(block.get("tokens") or []))

    for tok, feat in (mapping.get("token_priority_overrides") or {}).items():
        if tok not in exclude:
            lookup[str(tok)] = str(feat)
    return lookup


def build_node_type_lookup(mapping: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for section in ("structure", "content"):
        for feat, block in (mapping.get(section) or {}).items():
            for nt in (block or {}).get("node_types") or []:
                if nt not in lookup:
                    lookup[str(nt)] = str(feat)
    return lookup


def build_phrase_lists(mapping: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for feat in CONTENT_FEATURES:
        block = (mapping.get("content") or {}).get(feat) or {}
        phrases = [p for p in (block.get("phrases") or []) if p]
        if phrases:
            out[feat] = phrases
    return out
