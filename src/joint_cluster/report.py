"""Markdown reports and feature_spec for joint clustering v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..utils import atomic_write_text, ensure_dir
from .gt_features import FEATURE_NAMES


FEATURE_SPEC_MD = """# Feature specification (frozen v1)

## Image embedding

- Method: DeepSeek-OCR2 global 256 projected tokens → mean → L2
- Dimension: 1280
- Alignment key: `page_id` = image filename stem; join via embedding run manifest `image_path`

## Five GT features

Built with:

- tokenizer: `benchmark_design.ocr.tokenizer.tokenize_greedy` + `build_latex_vocab`
- parse status: `benchmark_design.ocr.parse_validate.validate_parse_status`
- structure forest: `benchmark_design.ocr.structure_stc.build_structure_forest`
- taxonomy: `benchmark_design.ocr.token_taxonomy.classify_token`

### 1. `ast_tree_count`

Number of **top-level structure-forest roots** returned by `build_structure_forest` over all parse-OK lines on the page.

- Not expression count
- Flat `1+1=2` → 0 roots
- Atoms (digits/vars) are not trees

### 2. `total_ast_node_count`

Sum of `StructureNode` counts over all roots (root + internal + leaf structure nodes).

- Roots **are** counted
- Leaves **are** counted
- Numbers / variables are **not** nodes
- No virtual page root
- Braces `{` `}` are **not** nodes

### 3. `max_ast_depth`

Maximum node-level depth among all structure trees on the page:

- leaf node depth = **1**
- parent depth = `1 + max(child depths)`
- legal page with no structure trees → **0**
- any line parse failure on the page → **null** (page excluded from clustering fit)

### 4. `distinct_plain_token_count`

Number of distinct tokens whose taxonomy is **not** `STRUCTURAL` and **not** `LAYOUT_ALIGNMENT`.

Excluded from both plain and structure counts: `{`, `}`, `$`.

### 5. `distinct_structure_token_count`

Number of distinct tokens classified as `STRUCTURAL`, excluding `{`, `}`, `$`.

This is **LaTeX structure tokens**, not `StructureNode.stc_type` counts.
Do not rename to `distinct_ast_node_type_count` in v1.

## Parse failure policy

- Legal empty / no-structure page: AST features = 0, status `ok` or `empty`
- Parse failure: `ast_parse_status=fail`, `max_ast_depth=null`, **excluded from K-Means fit**
- Failures listed in `data/parse_failures.parquet`

## Preprocessing

- Image: PCA retain ≥95% variance on L2 embeddings; **no** per-PC StandardScaler; then image-block L2-mean scale → 1
- GT: StandardScaler on raw 5-d; then text-block L2-mean scale → 1
- Joint: concatenate `[image_block, text_block]`
- No log1p / ratios / density / entropy / manual clipping
"""


def write_feature_spec(config_dir: Path) -> Path:
    path = Path(config_dir) / "feature_spec.md"
    atomic_write_text(path, FEATURE_SPEC_MD)
    return path


def write_experiment_config_copy(cfg: dict[str, Any], config_dir: Path) -> Path:
    slim = {k: v for k, v in cfg.items() if not str(k).startswith("_") and k != "paths"}
    slim["paths"] = {
        "images_dir": cfg["paths"]["images_dir"],
        "embedding_run_dir": cfg["paths"]["embedding_run_dir"],
        "embedding_name": cfg["paths"]["embedding_name"],
        "output_root": cfg["paths"]["output_root"],
    }
    path = Path(config_dir) / "experiment_config.yaml"
    atomic_write_text(path, yaml.safe_dump(slim, allow_unicode=True, sort_keys=False))
    return path


def write_data_quality_report(
    qa: dict[str, Any],
    fit_df: pd.DataFrame,
    reports_dir: Path,
) -> Path:
    corr = fit_df[list(FEATURE_NAMES)].corr(method="spearman")
    lines = [
        "# Data quality report",
        "",
        "## Counts",
        *[f"- `{k}`: {v}" for k, v in qa.items()],
        "",
        "## GT feature summary (cluster-fit pages)",
        fit_df[list(FEATURE_NAMES)].describe().to_markdown(),
        "",
        "## Spearman correlation (fit set)",
        corr.to_markdown(),
        "",
        "Correlation is diagnostic only; no feature is auto-dropped.",
        "",
    ]
    path = Path(reports_dir) / "data_quality_report.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def write_cluster_cards(
    fit_df: pd.DataFrame,
    labels: np.ndarray,
    cards_dir: Path,
) -> None:
    ensure_dir(cards_dir)
    k = int(labels.max()) + 1
    n = len(labels)
    for c in range(k):
        m = labels == c
        sub = fit_df.iloc[np.where(m)[0]]
        stats = sub[list(FEATURE_NAMES)].describe().to_markdown()
        md = f"""# Cluster {c}

- size: {int(m.sum())} ({100 * m.sum() / n:.2f}%)

## GT stats (raw)

{stats}

## Notes

- See `representatives/cluster_{c}_center.png` and `cluster_{c}_outlier.png`
- Visual commonality / GT commonality: fill after human review
"""
        atomic_write_text(Path(cards_dir) / f"cluster_{c}.md", md)


def write_experiment_report(
    *,
    cfg: dict[str, Any],
    qa: dict[str, Any],
    bundle: dict[str, Any],
    metrics: pd.DataFrame,
    final_k: int,
    final_reason: str,
    sizes: np.ndarray,
    reports_dir: Path,
) -> Path:
    lines = [
        "# Joint clustering experiment report (v1)",
        "",
        "## 1. Goal",
        "Concatenate PCA image embedding with five explicit GT features and run K-Means (E2).",
        "E0 (image) and E1 (GT) are controls only.",
        "",
        "## 2. Data",
        f"- images: `{cfg['paths']['images_dir']}`",
        f"- embedding run: `{cfg['paths']['embedding_run_dir']}`",
        f"- embedding sha256: `{qa.get('embedding_sha256')}`",
        f"- cluster-fit N: `{qa.get('n_cluster_fit')}`",
        "",
        "## 3. GT features",
        "See `config/feature_spec.md`.",
        "",
        "## 4. Image processing",
        f"- PCA components kept: **{bundle['pca_n_components']}**",
        f"- cumulative variance: **{bundle['pca_explained_variance_ratio_sum']:.6f}**",
        f"- image_block_scale: `{bundle['image_block_scale']}`",
        f"- text_block_scale: `{bundle['text_block_scale']}`",
        "",
        "## 5. Joint feature",
        f"- dim = {bundle['pca_n_components']} + 5 = {bundle['pca_n_components'] + 5}",
        "",
        "## 6. K selection",
        f"- final K: **{final_k}**",
        f"- reason: {final_reason}",
        "",
        "## 7. E0 / E1 / E2 metrics",
        metrics.to_markdown(index=False),
        "",
        "## 8. Final cluster sizes",
        ", ".join(f"C{i}={int(s)}" for i, s in enumerate(sizes)),
        "",
        "## 9–13. Figures",
        "See `figures/` and `representatives/`.",
        "",
        "## 14. Conclusions",
        "- Fill after reviewing GT heatmap, boxplots, and contact sheets.",
        "",
        "## 15. Limits",
        "- No TF-IDF / SVD text; no learned fusion; structure tokens are LaTeX commands not STC types.",
        "- UMAP/PCA-2D are visualization only.",
        "",
    ]
    path = Path(reports_dir) / "experiment_report.md"
    atomic_write_text(path, "\n".join(lines))
    return path
