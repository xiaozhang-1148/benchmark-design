"""Quality acceptance gates for visual-only analysis_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from ..utils import atomic_write_json
from .paths import analysis_v2_dir, reports_v2_dir


def run_quality_gates_v2(cfg: dict[str, Any]) -> dict[str, Any]:
    v2 = analysis_v2_dir(cfg)
    reports = reports_v2_dir(cfg)
    out_dir = Path(cfg["paths"]["outputs_dir"])
    alerts: list[str] = []
    checks: dict[str, Any] = {}

    q_path = out_dir / "embedding_quality.parquet"
    if q_path.exists():
        q = pd.read_parquet(q_path)
        checks["embedding_quality_counts"] = {
            "n": int(len(q)),
            "image_readable": int(q["image_readable"].sum()),
            "embedding_finite": int(q["embedding_finite"].sum()),
            "embedding_norm_valid": int(q["embedding_norm_valid"].sum()),
            "embedding_dimension_valid": int(q["embedding_dimension_valid"].sum()),
            "embedding_usable": int(q["embedding_usable"].sum()),
        }
        n = max(int(len(q)), 1)
        unusable_rate = 1.0 - checks["embedding_quality_counts"]["embedding_usable"] / n
        checks["embedding_unusable_rate"] = unusable_rate
        if unusable_rate > 0.01:
            alerts.append(f"embedding_unusable_rate={unusable_rate:.4f} > 0.01")
        if int(q["embedding_finite"].sum()) < int(q["image_readable"].sum()):
            alerts.append("some readable images have non-finite embeddings")
    else:
        alerts.append("missing embedding_quality.parquet")

    # Visual matrix sanity
    mmap = out_dir / "visual_embeddings.f32.mmap"
    meta_p = out_dir / "visual_embeddings.f32.meta.json"
    if mmap.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        dim = int(meta.get("dim", 896))
        n_rows = int(meta.get("n_rows", 0))
        X = np.memmap(mmap, dtype=np.float32, mode="r", shape=(n_rows, dim))
        sample_n = min(n_rows, 2048)
        Xs = np.asarray(X[:sample_n])
        nan_inf = int((~np.isfinite(Xs)).sum())
        checks["visual_sample_nan_inf_count"] = nan_inf
        checks["visual_dim"] = dim
        checks["visual_n_rows"] = n_rows
        if nan_inf > 0:
            alerts.append(f"visual embedding nan_inf_count={nan_inf} in first {sample_n}")

    metrics_p = v2 / "feature_metrics_v2.json"
    if metrics_p.exists():
        metrics = json.loads(metrics_p.read_text())
        vis = (metrics.get("channels") or {}).get("visual") or {}
        pc1 = vis.get("pc1_variance")
        checks["visual_pc1_variance"] = pc1
        if pc1 is not None and pc1 > 0.95:
            alerts.append(f"pca_pc1_variance[visual]={pc1:.4f} > 0.95")
        umap_n = vis.get("umap_n")
        n = vis.get("n")
        if umap_n is not None and n is not None and umap_n != n:
            alerts.append(f"UMAP n mismatch: umap={umap_n} expected={n}")

    vg = v2 / "visual_token_group_metrics.parquet"
    if vg.exists():
        m = pd.read_parquet(vg)
        s = int(m["n"].sum())
        checks["visual_token_group_sum"] = s
        vis_idx = out_dir / "visual_index.parquet"
        if vis_idx.exists():
            # After usable filter, group sum should match analyzed n, not always full index
            n_vis = int(len(pd.read_parquet(vis_idx)))
            checks["visual_index_n"] = n_vis
            if s != n_vis and metrics_p.exists():
                analyzed_n = ((json.loads(metrics_p.read_text()).get("channels") or {}).get("visual") or {}).get("n")
                if analyzed_n is not None and s != int(analyzed_n):
                    alerts.append(f"visual token groups sum={s} != analyzed n={analyzed_n}")

    # Ensure OCR/layout artifacts are not required
    checks["mode"] = "visual_only"
    checks["ocr_layout_required"] = False

    gates = {
        "checks": checks,
        "alerts": alerts,
        "alert_count": len(alerts),
        "note": "Alerts require human review; not hard failures. OCR/layout channels removed.",
    }
    atomic_write_json(reports / "quality_gates_v2.json", gates)
    atomic_write_json(v2 / "quality_gates_v2.json", gates)
    print(f"[quality_gates_v2] alerts={len(alerts)} {alerts[:5]}")
    return gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    run_quality_gates_v2(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
