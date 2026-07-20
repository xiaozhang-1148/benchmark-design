"""Group comparison and difference heatmaps."""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from heatmap_analysis.aggregation import aggregate_stack
from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.utils import ensure_dir, save_json

logger = logging.getLogger("heatmap_analysis.comparison")


def _load_metrics_and_heatmaps(cfg: AnalysisConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    metrics = pd.read_csv(cfg.output.output_dir / "tables" / "per_image_metrics.csv")
    cache_dir = cfg.cache_dir / "per_image"
    abs_map: dict[str, np.ndarray] = {}
    rel_map: dict[str, np.ndarray] = {}
    for npz in cache_dir.glob("*.npz"):
        d = np.load(npz)
        abs_map[npz.stem] = d["d_abs"]
        rel_map[npz.stem] = d["d_rel_smooth"] if "d_rel_smooth" in d else d["d_rel"]
    return metrics, abs_map, rel_map


def compare_groups(
    cfg: AnalysisConfig,
    group_by: str | list[str] | None = None,
) -> dict:
    """Generate group-level heatmaps and pairwise difference maps."""
    fields = group_by if group_by is not None else cfg.report.group_by
    if isinstance(fields, str):
        fields = [fields]
    if not fields:
        logger.warning("No group_by field specified")
        return {}

    metrics, abs_map, rel_map = _load_metrics_and_heatmaps(cfg)
    results: dict = {}

    for field in fields:
        if field not in metrics.columns:
            logger.warning("Group field %s not in metadata", field)
            continue
        out_base = ensure_dir(cfg.output.output_dir / "groups" / field)
        groups = metrics.groupby(field, dropna=False)
        group_stats: dict = {}

        for name, gdf in groups:
            gname = str(name) if name is not None and str(name) != "nan" else "NA"
            ids = [i for i in gdf["image_id"].astype(str) if i in rel_map]
            if not ids:
                continue
            rel_stack = np.stack([rel_map[i] for i in ids])
            abs_stack = np.stack([abs_map[i] for i in ids])
            rel_agg = aggregate_stack(rel_stack, cfg.heatmap.active_cell_threshold)
            abs_agg = aggregate_stack(abs_stack, cfg.heatmap.active_cell_threshold)

            gdir = ensure_dir(out_base / gname)
            np.savez_compressed(gdir / "relative_stats.npz", n_samples=len(ids), **rel_agg)
            np.savez_compressed(gdir / "absolute_stats.npz", n_samples=len(ids), **abs_agg)

            group_stats[gname] = {
                "n_samples": len(ids),
                "mean_ink_coverage": float(gdf["ink_coverage"].mean()),
                "mean_centroid_x": float(gdf["centroid_x"].mean()),
                "mean_centroid_y": float(gdf["centroid_y"].mean()),
                "mean_spatial_entropy": float(gdf["spatial_entropy"].mean()),
                "mean_hotspot_concentration": float(gdf["hotspot_concentration"].mean()),
            }

        # Pairwise differences
        group_names = list(group_stats.keys())
        for ga, gb in combinations(group_names, 2):
            ids_a = metrics[metrics[field].astype(str) == ga]["image_id"].astype(str).tolist()
            ids_b = metrics[metrics[field].astype(str) == gb]["image_id"].astype(str).tolist()
            ids_a = [i for i in ids_a if i in rel_map]
            ids_b = [i for i in ids_b if i in rel_map]
            if len(ids_a) < 2 or len(ids_b) < 2:
                continue
            mean_a = np.mean(np.stack([rel_map[i] for i in ids_a]), axis=0)
            mean_b = np.mean(np.stack([rel_map[i] for i in ids_b]), axis=0)
            diff = mean_a - mean_b

            # Effect size (Cohen's d per cell) and bootstrap CI
            stack_a = np.stack([rel_map[i] for i in ids_a])
            stack_b = np.stack([rel_map[i] for i in ids_b])
            pooled_std = np.sqrt((np.var(stack_a, axis=0) + np.var(stack_b, axis=0)) / 2)
            effect_size = diff / np.maximum(pooled_std, 1e-12)

            # Welch t-test per cell with BH-FDR correction
            t_stat, p_vals = stats.ttest_ind(stack_a, stack_b, axis=0, equal_var=False)
            p_flat = p_vals.ravel()
            n_tests = p_flat.size
            order = np.argsort(p_flat)
            ranked = p_flat[order]
            q = np.empty_like(ranked)
            prev = 1.0
            for i in range(n_tests - 1, -1, -1):
                val = ranked[i] * n_tests / (i + 1)
                prev = min(prev, val)
                q[i] = prev
            q_vals = np.empty_like(q)
            q_vals[order] = q
            sig_mask = (q_vals.reshape(p_vals.shape) < 0.05).astype(np.float32)

            pair_dir = ensure_dir(out_base / f"diff_{ga}_vs_{gb}")
            np.savez_compressed(
                pair_dir / "comparison.npz",
                difference=diff,
                effect_size=effect_size,
                p_values=p_vals,
                q_values=q_vals.reshape(p_vals.shape),
                significant_mask=sig_mask,
                n_a=len(ids_a),
                n_b=len(ids_b),
            )

        save_json(out_base / "group_summary.json", group_stats)
        results[field] = group_stats

    return results
