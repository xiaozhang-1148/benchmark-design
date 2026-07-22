"""Joint clustering V2 pipeline."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from ..utils import atomic_write_json, ensure_dir
from ..visual_exp.io_util import atomic_write_npy, atomic_write_parquet
from .cluster_v2 import distances_to_centers, fit_kmeans
from .figures_v2 import (
    fit_visual_umap,
    plot_cluster_sizes,
    plot_crosstab,
    plot_depth_box,
    plot_joint_umap,
    plot_prevalence,
    plot_shared_visual_space,
    plot_sizes_comparison,
    plot_type_heatmap,
    write_rep_sheets,
)
from .gt_features_v2 import compute_v2_text_table, filter_binary_features
from .mapping import BINARY_FEATURES
from .preprocess_v2 import build_text_block, joint_from_blocks, load_v1_image_block
from .report_v2 import (
    write_cluster_cards_v2,
    write_docs,
    write_experiment_report_v2,
    write_quality_report,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = deepcopy(cfg)
    root = Path(cfg["paths"]["output_root"])
    ensure_dir(root)
    cfg["paths"]["output_root"] = str(root)
    for name in (
        "config",
        "data",
        "models",
        "metrics",
        "figures",
        "representatives",
        "cluster_cards",
        "reports",
    ):
        cfg["paths"][f"{name}_dir"] = str(ensure_dir(root / name))
    return cfg


def _resolve_mapping_path(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["paths"]["mapping_yaml"])
    if not p.is_absolute():
        # relative to benchmark-design root
        p = Path(__file__).resolve().parents[2] / p
    return p


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    v1 = Path(cfg["paths"]["v1_root"])
    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    reps_dir = Path(cfg["paths"]["representatives_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    config_dir = Path(cfg["paths"]["config_dir"])
    seed = int(cfg.get("random_state", 42))
    k_main = int(cfg.get("k_main", 4))
    n_init = int(cfg["clustering"]["n_init"])
    init = str(cfg["clustering"].get("init", "k-means++"))
    mapping_path = _resolve_mapping_path(cfg)

    write_docs(config_dir, cfg, mapping_path)

    page_v1 = pd.read_parquet(v1 / "data" / "page_features.parquet")
    fit_mask = page_v1["cluster_fit"].astype(bool).to_numpy()
    page_fit = page_v1.loc[fit_mask].reset_index(drop=True)
    assert len(page_fit) == int(fit_mask.sum())

    print("[v2] building type features from V1 gt_text …")
    type_df, meta = compute_v2_text_table(page_fit, mapping_path)
    # attach binaries onto page_fit
    page_fit = page_fit.merge(type_df.drop(columns=["max_ast_depth", "total_ast_node_count"], errors="ignore"), on="page_id", how="left")

    prev_df = pd.DataFrame(meta["prevalence"])
    prev_df.to_csv(metrics_dir / "type_prevalence.csv", index=False)
    plot_prevalence(prev_df, figures_dir / "type_prevalence_global.png")

    kept, filter_log = filter_binary_features(
        meta["prevalence"],
        min_rate=float(cfg["prevalence"]["min_rate"]),
        max_rate=float(cfg["prevalence"]["max_rate"]),
    )
    filter_log.to_csv(metrics_dir / "feature_filter_log.csv", index=False)
    print(f"[v2] kept binaries ({len(kept)}): {kept}")

    # raw 17-d matrix (depth + all binaries before filter)
    text_raw = np.column_stack(
        [
            page_fit["max_ast_depth"].to_numpy(dtype=np.float64),
            page_fit[list(BINARY_FEATURES)].to_numpy(dtype=np.float64),
        ]
    )
    atomic_write_npy(data_dir / "text_features_v2_raw.npy", text_raw.astype(np.float32))

    cont_main = page_fit[["max_ast_depth"]].to_numpy(dtype=np.float64)
    bin_kept = page_fit[kept].to_numpy(dtype=np.float64) if kept else np.zeros((len(page_fit), 0))
    text_main, scaler_main, text_scale_main = build_text_block(cont_main, bin_kept)
    atomic_write_npy(data_dir / "text_features_v2_filtered.npy", text_main.astype(np.float32))
    joblib.dump(scaler_main, models_dir / "gt_continuous_scaler_main.pkl")
    joblib.dump({"text_block_scale": text_scale_main, "kept_features": kept}, models_dir / "text_block_meta_main.pkl")

    cont_nodes = page_fit[["max_ast_depth", "total_ast_node_count"]].to_numpy(dtype=np.float64)
    text_nodes, scaler_nodes, text_scale_nodes = build_text_block(cont_nodes, bin_kept)
    joblib.dump(scaler_nodes, models_dir / "gt_continuous_scaler_nodes.pkl")
    joblib.dump({"text_block_scale": text_scale_nodes, "kept_features": kept}, models_dir / "text_block_meta_nodes.pkl")

    X_img, img_meta = load_v1_image_block(v1)
    assert X_img.shape[0] == len(page_fit)
    # copy reference to models
    joblib.dump(joblib.load(v1 / "models" / "image_pca.pkl"), models_dir / "image_pca_reused.pkl")

    w1 = float(cfg["fusion"]["f1_text_weight"])
    w2 = float(cfg["fusion"]["f2_text_weight"])
    joint_equal = joint_from_blocks(X_img, text_main, w1)
    joint_aux = joint_from_blocks(X_img, text_main, w2)
    joint_nodes = joint_from_blocks(X_img, text_nodes, w1)
    atomic_write_npy(data_dir / "joint_features_equal.npy", joint_equal)
    atomic_write_npy(data_dir / "joint_features_text_aux.npy", joint_aux)
    atomic_write_npy(data_dir / "joint_features_nodes_ablation.npy", joint_nodes)

    X_v1_joint = np.load(v1 / "data" / "joint_features.npy")
    assert X_v1_joint.shape[0] == len(page_fit)

    experiments = {
        "E0_visual": X_img,
        "E1_v1_joint": X_v1_joint,
        "E2_v2_equal": joint_equal,
        "E3_v2_text_aux": joint_aux,
        "E4_v2_nodes": joint_nodes,
    }

    metric_rows = []
    labels_k4: dict[str, np.ndarray] = {}
    models_k4 = {}
    sizes_k4: dict[str, list[int]] = {}
    dists_k4: dict[str, np.ndarray] = {}

    # main K=4
    for exp_id, X in experiments.items():
        km, lab, meta_km = fit_kmeans(X, k_main, seed=seed, n_init=n_init, init=init)
        meta_km = {"experiment_id": exp_id, **meta_km}
        metric_rows.append(meta_km)
        labels_k4[exp_id] = lab
        models_k4[exp_id] = km
        sizes_k4[exp_id] = meta_km["cluster_sizes"]
        dists_k4[exp_id] = distances_to_centers(X, lab, km.cluster_centers_)
        joblib.dump(km, models_dir / f"kmeans_{exp_id}_k{k_main}.pkl")
        plot_cluster_sizes(
            np.asarray(meta_km["cluster_sizes"]),
            f"{exp_id} K={k_main}",
            figures_dir / f"{exp_id.lower()}_k4_cluster_sizes.png",
        )
        print(f"[v2] {exp_id} K={k_main} sil={meta_km['silhouette']:.4f} sizes={meta_km['cluster_sizes']}")

    # aux K scan on E2 only (and maybe E0) for report
    for exp_id in ("E0_visual", "E2_v2_equal"):
        X = experiments[exp_id]
        for k in range(int(cfg["clustering"]["k_scan_min"]), int(cfg["clustering"]["k_scan_max"]) + 1):
            if k == k_main:
                continue
            _, _, meta_km = fit_kmeans(X, k, seed=seed, n_init=n_init, init=init)
            metric_rows.append({"experiment_id": exp_id, **meta_km, "aux_scan": True})

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(metrics_dir / "clustering_metrics_v2.csv", index=False)

    # figures: type heatmap for E2/E3
    bin_mat = page_fit[kept].to_numpy(dtype=np.float64) if kept else np.zeros((len(page_fit), 0))
    for exp_id in ("E2_v2_equal", "E3_v2_text_aux", "E4_v2_nodes"):
        if kept:
            plot_type_heatmap(
                bin_mat,
                kept,
                labels_k4[exp_id],
                figures_dir / f"cluster_type_heatmap_k4_{exp_id}.png",
                title=f"Type rates by cluster ({exp_id})",
            )
    if kept:
        plot_type_heatmap(
            bin_mat,
            kept,
            labels_k4["E2_v2_equal"],
            figures_dir / "cluster_type_heatmap_k4.png",
            title="Type rates by cluster (E2 F1)",
        )

    plot_depth_box(
        page_fit["max_ast_depth"].to_numpy(dtype=np.float64),
        labels_k4["E2_v2_equal"],
        figures_dir / "max_depth_by_cluster_k4.png",
    )
    plot_depth_box(
        page_fit["total_ast_node_count"].to_numpy(dtype=np.float64),
        labels_k4["E4_v2_nodes"],
        figures_dir / "nodes_by_cluster_k4_e4.png",
        ylabel="total_ast_node_count",
    )

    plot_sizes_comparison(
        {k: np.asarray(v) for k, v in sizes_k4.items() if k in ("E0_visual", "E1_v1_joint", "E2_v2_equal", "E3_v2_text_aux")},
        figures_dir / "cluster_sizes_comparison_k4.png",
    )

    ct_e2 = plot_crosstab(
        labels_k4["E0_visual"],
        labels_k4["E2_v2_equal"],
        figures_dir / "visual_vs_joint_crosstab_k4_e2.png",
        "E0 visual vs E2 equal (row-normalized; cell=count)",
    )
    ct_e3 = plot_crosstab(
        labels_k4["E0_visual"],
        labels_k4["E3_v2_text_aux"],
        figures_dir / "visual_vs_joint_crosstab_k4_e3.png",
        "E0 visual vs E3 text-aux (row-normalized; cell=count)",
    )
    ct_e2.to_csv(metrics_dir / "visual_vs_joint_k4_e2.csv")
    ct_e3.to_csv(metrics_dir / "visual_vs_joint_k4_e3.csv")
    # combined alias
    ct_e2.to_csv(metrics_dir / "visual_vs_joint_k4.csv")

    print("[v2] shared visual UMAP …")
    vis_coords = fit_visual_umap(X_img, cfg)
    plot_shared_visual_space(
        vis_coords,
        labels_k4["E0_visual"],
        labels_k4["E2_v2_equal"],
        figures_dir / "visual_space_labels_comparison_k4.png",
        "E0 visual labels",
        "E2 joint labels",
    )
    plot_shared_visual_space(
        vis_coords,
        labels_k4["E0_visual"],
        labels_k4["E3_v2_text_aux"],
        figures_dir / "visual_space_labels_comparison_k4_e3.png",
        "E0 visual labels",
        "E3 joint labels",
    )
    plot_joint_umap(joint_equal, labels_k4["E2_v2_equal"], figures_dir / "joint_umap_k4.png", cfg)

    gal = cfg.get("galleries") or {}
    # merge type cols for sheets
    for exp_id, sub in (
        ("e0_visual", "E0_visual"),
        ("e2_equal", "E2_v2_equal"),
        ("e3_text_aux", "E3_v2_text_aux"),
    ):
        write_rep_sheets(
            page_fit,
            labels_k4[sub],
            dists_k4[sub],
            kept,
            reps_dir / exp_id,
            center_n=int(gal.get("center_n", 20)),
            outlier_n=int(gal.get("outlier_n", 12)),
            prefix=f"{sub} ",
        )

    write_cluster_cards_v2(
        Path(cfg["paths"]["cluster_cards_dir"]),
        page_fit,
        labels_k4["E2_v2_equal"],
        kept,
        "E2_v2_equal",
    )

    # page features table
    out_pages = page_fit.copy()
    for exp_id, lab in labels_k4.items():
        out_pages[f"cluster_{exp_id}_k4"] = lab
        out_pages[f"dist_{exp_id}_k4"] = dists_k4[exp_id]
    out_pages["final_cluster"] = labels_k4["E2_v2_equal"]
    out_pages["final_experiment"] = "E2_v2_equal"
    atomic_write_parquet(out_pages, data_dir / "page_features_v2.parquet")
    out_pages.drop(columns=["gt_text"], errors="ignore").to_csv(
        data_dir / "page_features_v2_summary.csv", index=False
    )

    # PC1 correlations for quality report
    pc1 = X_img[:, 0]
    corr_rows = {}
    for f in ["max_ast_depth", "total_ast_node_count", *kept]:
        if f in page_fit.columns:
            rho, _ = spearmanr(page_fit[f].to_numpy(dtype=float), pc1)
            corr_rows[f] = float(rho) if np.isfinite(rho) else None
    corr_s = pd.Series(corr_rows, name="spearman_with_visual_dim0")

    write_quality_report(reports_dir, meta=meta, filter_log=filter_log, corr_pc1=corr_s)

    # heuristic notes
    def _collapse_score(lab: np.ndarray, bin_mat: np.ndarray) -> float:
        if bin_mat.size == 0:
            return float("nan")
        k = int(lab.max()) + 1
        means = np.stack([bin_mat[lab == c].mean(axis=0) for c in range(k)], axis=0)
        # mean abs difference between cluster mean-rate vectors
        diffs = []
        for i in range(k):
            for j in range(i + 1, k):
                diffs.append(np.mean(np.abs(means[i] - means[j])))
        return float(np.mean(diffs)) if diffs else 0.0

    notes = {
        "image_pca_reused": img_meta,
        "kept_binary_features": kept,
        "n_all_zero_type_pages": meta.get("n_all_zero_type_pages"),
        "type_separation_e2": _collapse_score(labels_k4["E2_v2_equal"], bin_mat),
        "type_separation_e4": _collapse_score(labels_k4["E4_v2_nodes"], bin_mat),
        "e0_sizes": sizes_k4["E0_visual"],
        "e2_sizes": sizes_k4["E2_v2_equal"],
        "e3_sizes": sizes_k4["E3_v2_text_aux"],
        "e4_sizes": sizes_k4["E4_v2_nodes"],
    }
    write_experiment_report_v2(
        reports_dir,
        cfg=cfg,
        metrics=metrics,
        kept_features=kept,
        sizes=sizes_k4,
        notes=notes,
    )

    summary = {
        "output_root": cfg["paths"]["output_root"],
        "k_main": k_main,
        "kept_features": kept,
        "sizes_k4": sizes_k4,
        "n_fit": int(len(page_fit)),
    }
    atomic_write_json(Path(cfg["paths"]["output_root"]) / "summary.json", summary)
    print(f"[v2] done → {cfg['paths']['output_root']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Joint clustering V2")
    parser.add_argument(
        "--config",
        default="configs/joint_cluster/experiment_config_v2.yaml",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
