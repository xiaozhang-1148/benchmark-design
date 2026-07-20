"""Copy representative sample overlays into dedicated output folder."""

from __future__ import annotations

import shutil
from pathlib import Path

from heatmap_analysis.config import AnalysisConfig


def export_representative_samples(cfg: AnalysisConfig) -> None:
    out = cfg.output.output_dir / "representative_samples"
    out.mkdir(parents=True, exist_ok=True)
    cl_root = cfg.output.output_dir / "clustering"
    hm_dir = cfg.output.output_dir / "per_image_heatmaps"
    if not cl_root.exists():
        return
    for csv_path in cl_root.rglob("representative_samples.csv"):
        cluster_name = csv_path.parent.name
        tmpl = csv_path.parent.parent.name
        import pandas as pd

        df = pd.read_csv(csv_path)
        col = df.columns[0]
        for iid in df[col].astype(str):
            src = hm_dir / f"{iid}_overlay.png"
            if src.exists():
                dst = out / f"{tmpl}_{cluster_name}_{iid}.png"
                shutil.copy2(src, dst)
