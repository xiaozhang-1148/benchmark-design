"""Write feature_analysis_v2.md (visual embedding only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_config
from ..utils import atomic_write_text
from .paths import analysis_v2_dir, reports_v2_dir


def build_report_v2(cfg: dict[str, Any]) -> str:
    v2 = analysis_v2_dir(cfg)
    reports = reports_v2_dir(cfg)
    out_dir = Path(cfg["paths"]["outputs_dir"])

    gates = {}
    gp = reports / "quality_gates_v2.json"
    if gp.exists():
        gates = json.loads(gp.read_text())

    qsum = {}
    qs = out_dir / "embedding_quality_summary.json"
    if qs.exists():
        qsum = json.loads(qs.read_text())

    metrics = {}
    mp = v2 / "feature_metrics_v2.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())

    vg = (
        pd.read_parquet(v2 / "visual_token_group_metrics.parquet")
        if (v2 / "visual_token_group_metrics.parquet").exists()
        else None
    )

    lines = [
        "# Feature Analysis v2 (visual embedding only)",
        "",
        "Pipeline no longer runs OCR text generation, grounding layout boxes, or OCR quality labels.",
        "Cluster interpretation of math content should use GT labels, not model-generated text.",
        "",
        "## Embedding quality",
        f"- summary: `{json.dumps(qsum, ensure_ascii=False)}`",
        "",
        "## Quality gates",
        f"- alert_count: **{gates.get('alert_count', 'n/a')}**",
        f"- alerts: `{json.dumps(gates.get('alerts', []), ensure_ascii=False)}`",
        "",
        "```json",
        json.dumps(gates.get("checks", {}), indent=2, ensure_ascii=False)[:4000],
        "```",
        "",
        "## Visual PCA / UMAP",
        "```json",
        json.dumps(metrics.get("channels", {}), indent=2, ensure_ascii=False)[:5000],
        "```",
        "",
        "## Visual token groups",
    ]
    if vg is not None:
        lines.append(vg.to_markdown(index=False))
    lines += [
        "",
        "## Notes",
        "- Clustering input: L2-normalized visual embeddings (or PCA retaining 95% variance).",
        "- Do not cluster on 2D PCA / 2D UMAP.",
        "- Contact sheets: ordinary / high_density / low_density / pca_extremes / umap_islands.",
        "",
    ]
    md = "\n".join(lines)
    atomic_write_text(reports / "feature_analysis_v2.md", md)
    print(f"[report_v2] -> {reports / 'feature_analysis_v2.md'}")
    return md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    build_report_v2(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
