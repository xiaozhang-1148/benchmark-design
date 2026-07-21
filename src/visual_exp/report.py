"""HTML report for visual embedding analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import atomic_write_text


def build_html_report(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["paths"]["experiment_root"])
    report_dir = Path(cfg["paths"]["report_dir"])
    diag = Path(cfg["paths"]["diagnostics_dir"])
    clus = Path(cfg["paths"]["clustering_dir"])

    verify = _read_json(diag / "verify_report.json")
    diagnostics = _read_json(diag / "diagnostics_summary.json")
    clustering = _read_json(clus / "clustering_summary.json")
    ksel = ""
    if (clus / "k_selection.csv").exists():
        ksel = pd.read_csv(clus / "k_selection.csv").to_html(index=False)

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except Exception:
            return str(p)

    figs = [
        ("Similarity", diag / "similarity_distribution.png"),
        ("Norms", diag / "norm_distribution.png"),
        ("PCA variance", diag / "pca_variance.png"),
        ("Technical confounders", diag / "technical_confounders.png"),
        ("PCA scatter", Path(cfg["paths"]["projections_dir"]) / "pca_scatter.png"),
        ("UMAP scatter", Path(cfg["paths"]["projections_dir"]) / "umap_scatter.png"),
    ]
    img_tags = ""
    for title, p in figs:
        if p.exists():
            img_tags += f"<h3>{title}</h3><img src='../{rel(p)}' style='max-width:900px'/><br/>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Visual Embedding Analysis</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; max-width: 1100px; color: #222; }}
code {{ background: #f4f4f4; padding: 0.1em 0.3em; }}
table {{ border-collapse: collapse; font-size: 0.9rem; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
</style></head><body>
<h1>DeepSeek-OCR2 mean-pooled projected-token embedding</h1>
<p>Pure visual pipeline: no OCR text, no GT, no layout. Clustering on L2 embeddings (not UMAP 2D).</p>
<h2>Verify</h2>
<pre>{json.dumps(verify, indent=2, ensure_ascii=False)}</pre>
<h2>Diagnostics</h2>
<pre>{json.dumps(diagnostics, indent=2, ensure_ascii=False)}</pre>
<h2>Clustering</h2>
<pre>{json.dumps(clustering, indent=2, ensure_ascii=False)}</pre>
<h2>K selection</h2>
{ksel}
<h2>Figures</h2>
{img_tags}
<p>Galleries: <code>galleries/cluster_centers</code>, <code>cluster_boundaries</code>, <code>outliers</code>, <code>nearest_neighbors</code>.</p>
</body></html>
"""
    out = report_dir / "visual_embedding_analysis.html"
    atomic_write_text(out, html)
    print(f"[report] -> {out}")
    return out


def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}
