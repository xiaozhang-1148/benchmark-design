"""V2 markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..utils import atomic_write_text, ensure_dir
from .mapping import BINARY_FEATURES, CONTENT_FEATURES, STRUCTURE_FEATURES


FEATURE_SPEC_V2 = """# Feature specification V2

## Reused from V1 (unchanged)

- DeepSeek-OCR2 1280-d L2 embeddings
- Same PCA model → 81-d (~95.05% variance)
- Same `page_id` ↔ `image_path` ↔ embedding alignment
- AST parse status / `max_ast_depth` / `total_ast_node_count` from V1 page table

## Deleted from V1 main text vector

- `ast_tree_count` (collinear with node count, ρ≈0.95)
- `total_ast_node_count` (main only; kept in nodes ablation)
- `distinct_plain_token_count`
- `distinct_structure_token_count`

## V2-main text vector (17 raw dims before filter)

1. `max_ast_depth` (continuous; StandardScaler)
2–9. structure binaries: """ + ", ".join(STRUCTURE_FEATURES) + """
10–17. content binaries: """ + ", ".join(CONTENT_FEATURES) + """

Binaries are **presence** (1 if any matching token/node/phrase on the page).

## Mapping source of truth

See `token_category_mapping.yaml`:

- each LaTeX **token** maps to at most one category (structure before content; overrides apply)
- Chinese / multi-char **phrases** matched on raw GT substrings
- excluded: `{ } $ &` and layout-only spacing tokens
- structure also OR-ed from `StructureNode.stc_type` when listed

## Prevalence filter

Keep binary features with page rate in `[1%, 99%]`. Dropped features stay in raw tables / quality report.

## Preprocess

- StandardScaler **only** on continuous columns
- binaries stay 0/1
- concatenate → text-block mean-L2 scale → 1
- F1: text weight 1.0; F2: text weight 0.5 after block scale

## Experiments (main K=4)

| ID | Description |
|----|-------------|
| E0 | visual image block only |
| E1 | V1 joint features (5-d GT path) |
| E2 | V2-main + F1 equal |
| E3 | V2-main + F2 text-aux |
| E4 | V2-nodes-ablation (+ `total_ast_node_count`) + F1 |
"""


def write_docs(config_dir: Path, cfg: dict[str, Any], mapping_src: Path) -> None:
    ensure_dir(config_dir)
    atomic_write_text(config_dir / "feature_spec_v2.md", FEATURE_SPEC_V2)
    atomic_write_text(
        config_dir / "experiment_config_v2.yaml",
        yaml.safe_dump({k: v for k, v in cfg.items() if not str(k).startswith("_")}, allow_unicode=True, sort_keys=False),
    )
    # copy mapping
    text = Path(mapping_src).read_text(encoding="utf-8")
    atomic_write_text(config_dir / "token_category_mapping.yaml", text)


def write_quality_report(
    reports_dir: Path,
    *,
    meta: dict[str, Any],
    filter_log: pd.DataFrame,
    corr_pc1: pd.Series | None,
) -> Path:
    lines = [
        "# Data quality report V2",
        "",
        f"- pages: {meta.get('n_pages')}",
        f"- all-zero type vectors: {meta.get('n_all_zero_type_pages')}",
        f"- token lookup size: {meta.get('token_lookup_size')}",
        "",
        "## Prevalence / filter",
        filter_log.to_markdown(index=False),
        "",
        "## Top unmapped tokens (instances)",
        "\n".join(f"- `{t}`: {c}" for t, c in (meta.get("top_unmapped_tokens") or [])[:40]),
        "",
    ]
    if corr_pc1 is not None:
        lines += ["## Spearman |ρ| with visual PC1 (fit set)", corr_pc1.to_markdown(), ""]
    path = Path(reports_dir) / "data_quality_report_v2.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def write_cluster_cards_v2(
    cards_dir: Path,
    page_df: pd.DataFrame,
    labels: np.ndarray,
    type_features: list[str],
    experiment_id: str,
) -> None:
    ensure_dir(cards_dir)
    k = int(labels.max()) + 1
    n = len(labels)
    for c in range(k):
        m = labels == c
        sub = page_df.iloc[np.where(m)[0]]
        rates = {f: float(sub[f].mean()) if f in sub.columns else 0.0 for f in type_features}
        top = sorted(rates.items(), key=lambda x: -x[1])[:8]
        depth = sub["max_ast_depth"].describe().to_markdown() if "max_ast_depth" in sub.columns else ""
        lines = [
            f"# Cluster {c} ({experiment_id})",
            "",
            f"- size: {int(m.sum())} ({100*m.sum()/n:.2f}%)",
            "",
            "## Top type rates",
            *[f"- `{f}`: {100*r:.1f}%" for f, r in top],
            "",
            "## max_ast_depth",
            depth,
            "",
            "## Interpretation template",
            "- Visual commonality: (fill after reviewing representatives)",
            "- Dominant structures / content types: see rates above",
            "- Distinction vs other clusters: (fill)",
            "",
        ]
        atomic_write_text(Path(cards_dir) / f"cluster_{c}.md", "\n".join(lines))


def write_experiment_report_v2(
    reports_dir: Path,
    *,
    cfg: dict[str, Any],
    metrics: pd.DataFrame,
    kept_features: list[str],
    sizes: dict[str, list[int]],
    notes: dict[str, Any],
) -> Path:
    lines = [
        "# Experiment report V2",
        "",
        "## Goal",
        "Replace V1 count-based GT with type-identity binaries; fix K=4; compare F1/F2 and nodes ablation.",
        "",
        "## Kept binary features after prevalence filter",
        ", ".join(f"`{f}`" for f in kept_features) or "(none)",
        "",
        "## K=4 metrics (do not compare Inertia across feature spaces)",
        metrics[metrics["k"] == 4].to_markdown(index=False),
        "",
        "## Cluster sizes @ K=4",
        *[f"- **{k}**: {v}" for k, v in sizes.items()],
        "",
        "## Questions",
        "1. Did type features avoid low/high volume dichotomy? See type heatmaps vs V1.",
        "2. Do K=4 clusters show distinct structure/content profiles?",
        "3. F1 vs F2: which preserves visual structure better? Compare crosstabs + shared UMAP.",
        "4. Nodes ablation (E4): does adding `total_ast_node_count` re-collapse?",
        "5. Per-cluster cards + representatives for visual + GT consistency.",
        "",
        "## Auto notes",
        yaml.safe_dump(notes, allow_unicode=True, sort_keys=False),
        "",
    ]
    path = Path(reports_dir) / "experiment_report_v2.md"
    atomic_write_text(path, "\n".join(lines))
    return path
