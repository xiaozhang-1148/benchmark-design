"""Frozen v1: five explicit GT features from structure forest + token taxonomy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from benchmark_design.ocr.parse_validate import validate_parse_status
from benchmark_design.ocr.structure_stc import StructureNode, build_structure_forest
from benchmark_design.ocr.token_taxonomy import TokenCategory, classify_token
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy
from benchmark_design.page_level_latex.loader import RawExpressionRow, load_raw_expressions

# --- Frozen counting rules (do not change mid-experiment) -----------------
# ast_tree_count: number of top-level StructureNode roots from build_structure_forest
# total_ast_node_count: every StructureNode counted once (roots + internal + leaves);
#   atoms (numbers/vars) are NOT nodes; braces are NOT nodes; no virtual page root
# max_ast_depth: node-level depth; leaf root = 1; no forest = 0; parse fail = null
# distinct_plain_token_count: distinct tokens whose taxonomy is not STRUCTURAL / LAYOUT
# distinct_structure_token_count: distinct STRUCTURAL LaTeX tokens excluding { } $
# ----------------------------------------------------------------------

FEATURE_NAMES = (
    "ast_tree_count",
    "total_ast_node_count",
    "max_ast_depth",
    "distinct_plain_token_count",
    "distinct_structure_token_count",
)

_VOCAB = None


def _vocab():
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = build_latex_vocab()
    return _VOCAB


def count_structure_nodes(node: StructureNode) -> int:
    return 1 + sum(count_structure_nodes(c) for c in node.children)


def structure_depth(node: StructureNode) -> int:
    """Node-level depth: leaf = 1, parent = 1 + max(child)."""
    if not node.children:
        return 1
    return 1 + max(structure_depth(c) for c in node.children)


@dataclass(frozen=True, slots=True)
class PageGtFeatures:
    page_id: str
    gt_text: str
    ast_tree_count: int
    total_ast_node_count: int
    max_ast_depth: int | None
    distinct_plain_token_count: int
    distinct_structure_token_count: int
    ast_parse_status: str  # ok | fail | empty
    n_lines: int
    n_lines_ok: int
    n_lines_fail: int
    structure_token_types: tuple[str, ...]
    plain_token_sample: tuple[str, ...]


def _is_plain(token: str, *, exclude_layout: bool) -> bool:
    cat = classify_token(token)
    if cat is TokenCategory.STRUCTURAL:
        return False
    if exclude_layout and cat is TokenCategory.LAYOUT_ALIGNMENT:
        return False
    return True


def _is_structure(token: str, exclude: set[str]) -> bool:
    if token in exclude:
        return False
    return classify_token(token) is TokenCategory.STRUCTURAL


def features_from_line_texts(
    page_id: str,
    line_texts: Iterable[str],
    *,
    exclude_structure_tokens: Iterable[str] = ("{", "}", "$"),
    exclude_layout_from_plain: bool = True,
) -> PageGtFeatures:
    exclude = set(exclude_structure_tokens)
    vocab = _vocab()

    trees = 0
    nodes = 0
    depths: list[int] = []
    plain: set[str] = set()
    struct: set[str] = set()
    n_lines = 0
    n_ok = 0
    n_fail = 0
    gt_parts: list[str] = []

    for raw in line_texts:
        text = (raw or "").strip()
        if not text:
            continue
        n_lines += 1
        gt_parts.append(text)
        tokens = tokenize_greedy(text, vocab)
        status = validate_parse_status(tokens)
        if status != "ok":
            n_fail += 1
            continue
        n_ok += 1
        roots = build_structure_forest(tokens)
        trees += len(roots)
        for rt in roots:
            nodes += count_structure_nodes(rt)
            depths.append(structure_depth(rt))
        for t in tokens:
            if t in exclude:
                continue
            if _is_structure(t, exclude):
                struct.add(t)
            elif _is_plain(t, exclude_layout=exclude_layout_from_plain):
                plain.add(t)

    if n_lines == 0:
        parse_status = "empty"
        return PageGtFeatures(
            page_id=page_id,
            gt_text="",
            ast_tree_count=0,
            total_ast_node_count=0,
            max_ast_depth=0,
            distinct_plain_token_count=0,
            distinct_structure_token_count=0,
            ast_parse_status=parse_status,
            n_lines=0,
            n_lines_ok=0,
            n_lines_fail=0,
            structure_token_types=(),
            plain_token_sample=(),
        )

    if n_fail > 0:
        # Parse failure: do not coerce to "no structure"; clustering will exclude.
        return PageGtFeatures(
            page_id=page_id,
            gt_text="\n".join(gt_parts),
            ast_tree_count=0,
            total_ast_node_count=0,
            max_ast_depth=None,
            distinct_plain_token_count=len(plain),
            distinct_structure_token_count=len(struct),
            ast_parse_status="fail",
            n_lines=n_lines,
            n_lines_ok=n_ok,
            n_lines_fail=n_fail,
            structure_token_types=tuple(sorted(struct)),
            plain_token_sample=tuple(sorted(plain)[:32]),
        )

    return PageGtFeatures(
        page_id=page_id,
        gt_text="\n".join(gt_parts),
        ast_tree_count=trees,
        total_ast_node_count=nodes,
        max_ast_depth=max(depths) if depths else 0,
        distinct_plain_token_count=len(plain),
        distinct_structure_token_count=len(struct),
        ast_parse_status="ok",
        n_lines=n_lines,
        n_lines_ok=n_ok,
        n_lines_fail=n_fail,
        structure_token_types=tuple(sorted(struct)),
        plain_token_sample=tuple(sorted(plain)[:32]),
    )


def compute_all_page_gt(
    images_dir: str | Path,
    *,
    exclude_structure_tokens: Iterable[str] = ("{", "}", "$"),
    exclude_layout_from_plain: bool = True,
    workers: int | None = None,
) -> pd.DataFrame:
    rows = load_raw_expressions(Path(images_dir), show_progress=True, workers=workers)
    by_page: dict[str, list[RawExpressionRow]] = defaultdict(list)
    for r in rows:
        by_page[r.image_id].append(r)

    out_rows: list[dict[str, Any]] = []
    for page_id in sorted(by_page.keys()):
        lines = [r.raw_ocr_text for r in by_page[page_id]]
        feat = features_from_line_texts(
            page_id,
            lines,
            exclude_structure_tokens=exclude_structure_tokens,
            exclude_layout_from_plain=exclude_layout_from_plain,
        )
        out_rows.append(
            {
                "page_id": feat.page_id,
                "gt_text": feat.gt_text,
                "ast_tree_count": feat.ast_tree_count,
                "total_ast_node_count": feat.total_ast_node_count,
                "max_ast_depth": feat.max_ast_depth,
                "distinct_plain_token_count": feat.distinct_plain_token_count,
                "distinct_structure_token_count": feat.distinct_structure_token_count,
                "ast_parse_status": feat.ast_parse_status,
                "n_lines": feat.n_lines,
                "n_lines_ok": feat.n_lines_ok,
                "n_lines_fail": feat.n_lines_fail,
            }
        )
    return pd.DataFrame(out_rows)
