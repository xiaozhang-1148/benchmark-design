"""Dataset-level heatmap aggregation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.gpu import is_gpu_available
from heatmap_analysis.utils import ensure_dir, save_json

logger = logging.getLogger("heatmap_analysis.aggregation")


def load_heatmap_arrays(cfg: AnalysisConfig) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load all per-image heatmaps from cache."""
    cache_dir = cfg.cache_dir / "per_image"
    ids: list[str] = []
    abs_list: list[np.ndarray] = []
    rel_list: list[np.ndarray] = []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        data = np.load(npz_path)
        ids.append(npz_path.stem)
        abs_list.append(data["d_abs"])
        rel_list.append(data["d_rel_smooth"] if "d_rel_smooth" in data else data["d_rel"])
    if not ids:
        raise FileNotFoundError(f"no cached heatmaps in {cache_dir}")
    return ids, np.stack(abs_list), np.stack(rel_list)


def aggregate_stack(
    stack: np.ndarray,
    active_threshold: float = 0.001,
    *,
    use_gpu: bool = False,
) -> dict[str, np.ndarray]:
    """Compute dataset-level statistics from (N, H, W) stack."""
    if use_gpu and stack.shape[0] >= 64:
        from heatmap_analysis.gpu import get_xp, to_numpy

        xp = get_xp(True)
        if xp is not np:
            s = xp.asarray(stack, dtype=xp.float64)
            mean = to_numpy(xp.mean(s, axis=0))
            median = to_numpy(xp.median(s, axis=0))
            std = to_numpy(xp.std(s, axis=0))
            cv = std / np.maximum(mean, 1e-12)
            usage_prob = to_numpy(xp.mean(s > active_threshold, axis=0))
            p25 = to_numpy(xp.percentile(s, 25, axis=0))
            p50 = to_numpy(xp.percentile(s, 50, axis=0))
            p75 = to_numpy(xp.percentile(s, 75, axis=0))
            return {
                "mean": mean,
                "median": median,
                "std": std,
                "cv": cv,
                "usage_probability": usage_prob,
                "p25": p25,
                "p50": p50,
                "p75": p75,
            }

    mean = np.mean(stack, axis=0)
    median = np.median(stack, axis=0)
    std = np.std(stack, axis=0)
    cv = std / np.maximum(mean, 1e-12)
    usage_prob = np.mean(stack > active_threshold, axis=0)
    p25 = np.percentile(stack, 25, axis=0)
    p50 = np.percentile(stack, 50, axis=0)
    p75 = np.percentile(stack, 75, axis=0)
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "cv": cv,
        "usage_probability": usage_prob,
        "p25": p25,
        "p50": p50,
        "p75": p75,
    }


def run_aggregation(cfg: AnalysisConfig) -> dict:
    """Aggregate all cached heatmaps and save results."""
    ids, abs_stack, rel_stack = load_heatmap_arrays(cfg)
    n = len(ids)

    use_gpu = cfg.gpu.enabled and is_gpu_available()
    abs_stats = aggregate_stack(abs_stack, cfg.heatmap.active_cell_threshold, use_gpu=use_gpu)
    rel_stats = aggregate_stack(rel_stack, cfg.heatmap.active_cell_threshold, use_gpu=use_gpu)

    out_dir = ensure_dir(cfg.output.output_dir / "aggregate")
    np.savez_compressed(out_dir / "absolute_stats.npz", n_samples=n, **abs_stats)
    np.savez_compressed(out_dir / "relative_stats.npz", n_samples=n, **rel_stats)

    metrics_path = cfg.output.output_dir / "tables" / "per_image_metrics.csv"
    summary: dict = {"n_samples": n, "image_ids": ids}
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        summary["mean_ink_coverage"] = float(df["ink_coverage"].mean())
        summary["mean_spatial_entropy"] = float(df["spatial_entropy"].mean())
        summary["mean_centroid_x"] = float(df["centroid_x"].mean())
        summary["mean_centroid_y"] = float(df["centroid_y"].mean())
        summary["blank_count"] = int(df["is_blank"].sum())

    save_json(out_dir / "summary.json", summary)
    logger.info("Aggregated %d heatmaps", n)
    return {"n_samples": n, "abs_stats": abs_stats, "rel_stats": rel_stats, "summary": summary}
