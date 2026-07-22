"""V2 GT type-identity features from GT text + structure forest."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark_design.ocr.parse_validate import validate_parse_status
from benchmark_design.ocr.structure_stc import StructureNode, build_structure_forest
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy

from .mapping import (
    BINARY_FEATURES,
    CONTENT_FEATURES,
    STRUCTURE_FEATURES,
    build_node_type_lookup,
    build_phrase_lists,
    build_token_lookup,
    load_mapping,
)

_VOCAB = None


def _vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = build_latex_vocab()
    return _VOCAB


def _walk_node_types(node: StructureNode, out: set[str]) -> None:
    out.add(node.stc_type)
    for c in node.children:
        _walk_node_types(c, out)


def extract_page_type_flags(
    gt_text: str,
    *,
    token_lookup: dict[str, str],
    node_lookup: dict[str, str],
    phrases: dict[str, list[str]],
    exclude_tokens: set[str],
) -> dict[str, Any]:
    flags = {f: 0 for f in BINARY_FEATURES}
    unmapped: Counter[str] = Counter()
    mapped_tokens: Counter[str] = Counter()
    all_tokens: list[str] = []

    lines = [ln.strip() for ln in (gt_text or "").split("\n") if ln.strip()]
    node_types_seen: set[str] = set()

    for line in lines:
        toks = tokenize_greedy(line, _vocab())
        status = validate_parse_status(toks)
        if status == "ok":
            for rt in build_structure_forest(toks):
                _walk_node_types(rt, node_types_seen)
        for t in toks:
            if t in exclude_tokens:
                continue
            all_tokens.append(t)
            feat = token_lookup.get(t)
            if feat is None:
                unmapped[t] += 1
            else:
                flags[feat] = 1
                mapped_tokens[t] += 1

    for nt in node_types_seen:
        feat = node_lookup.get(nt)
        if feat is not None:
            flags[feat] = 1

    text_join = "\n".join(lines)
    for feat, plist in phrases.items():
        for p in plist:
            if p and p in text_join:
                flags[feat] = 1
                break

    return {
        **flags,
        "n_tokens": len(all_tokens),
        "n_unmapped_token_instances": int(sum(unmapped.values())),
        "n_distinct_unmapped": len(unmapped),
        "top_unmapped": dict(unmapped.most_common(20)),
        "type_vector_all_zero": int(sum(flags[f] for f in BINARY_FEATURES) == 0),
    }


def compute_v2_text_table(
    page_df: pd.DataFrame,
    mapping_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = load_mapping(mapping_path)
    token_lookup = build_token_lookup(mapping)
    node_lookup = build_node_type_lookup(mapping)
    phrases = build_phrase_lists(mapping)
    exclude = set(mapping.get("exclude_tokens") or [])

    rows = []
    unmapped_global: Counter[str] = Counter()
    for r in page_df.itertuples(index=False):
        gt = getattr(r, "gt_text", "") or ""
        feat = extract_page_type_flags(
            gt,
            token_lookup=token_lookup,
            node_lookup=node_lookup,
            phrases=phrases,
            exclude_tokens=exclude,
        )
        for t, c in (feat.pop("top_unmapped") or {}).items():
            unmapped_global[t] += c
        rows.append(
            {
                "page_id": r.page_id,
                "max_ast_depth": r.max_ast_depth,
                "total_ast_node_count": r.total_ast_node_count,
                **{f: feat[f] for f in BINARY_FEATURES},
                "n_tokens": feat["n_tokens"],
                "n_unmapped_token_instances": feat["n_unmapped_token_instances"],
                "type_vector_all_zero": feat["type_vector_all_zero"],
            }
        )

    df = pd.DataFrame(rows)
    n = len(df)
    prevalence = []
    for f in BINARY_FEATURES:
        rate = float(df[f].mean()) if n else 0.0
        prevalence.append({"feature": f, "n_pages": int(df[f].sum()), "rate": rate, "group": (
            "structure" if f in STRUCTURE_FEATURES else "content"
        )})
    meta = {
        "n_pages": n,
        "n_all_zero_type_pages": int(df["type_vector_all_zero"].sum()),
        "token_lookup_size": len(token_lookup),
        "top_unmapped_tokens": unmapped_global.most_common(50),
        "prevalence": prevalence,
    }
    return df, meta


def filter_binary_features(
    prevalence: list[dict[str, Any]],
    *,
    min_rate: float = 0.01,
    max_rate: float = 0.99,
) -> tuple[list[str], pd.DataFrame]:
    kept, log_rows = [], []
    for row in prevalence:
        f, rate = row["feature"], float(row["rate"])
        if rate < min_rate:
            action = "drop_low"
        elif rate > max_rate:
            action = "drop_high"
        else:
            action = "keep"
            kept.append(f)
        log_rows.append({**row, "action": action, "min_rate": min_rate, "max_rate": max_rate})
    return kept, pd.DataFrame(log_rows)
