"""Joint clustering pipeline: embedding + 5 GT → concat → K-Means."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import atomic_write_json
from ..visual_exp.io_util import atomic_write_npy, atomic_write_parquet
from .align import build_aligned_table
from .cluster import (
    attach_labels,
    run_kmeans_sweep,
    save_kmeans_models,
    select_final_k,
)
from .config import load_config
from .figures import (
    plot_cluster_sizes,
    plot_gt_boxplots,
    plot_gt_heatmap,
    plot_image_vs_joint_crosstab,
    plot_joint_pca2d,
    plot_joint_umap,
    plot_k_curves,
    plot_pca_variance,
    write_representatives,
)
from .gt_features import compute_all_page_gt
from .preprocess import fit_transform_features, gt_matrix, save_models
from .report import (
    write_cluster_cards,
    write_data_quality_report,
    write_experiment_config_copy,
    write_experiment_report,
    write_feature_spec,
)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    reps_dir = Path(cfg["paths"]["representatives_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    config_dir = Path(cfg["paths"]["config_dir"])

    write_feature_spec(config_dir)
    write_experiment_config_copy(cfg, config_dir)

    print("[joint] computing GT features …")
    gt_df = compute_all_page_gt(
        cfg["paths"]["images_dir"],
        exclude_structure_tokens=(cfg.get("gt") or {}).get(
            "exclude_structure_tokens", ["{", "}", "$"]
        ),
        exclude_layout_from_plain=bool((cfg.get("gt") or {}).get("exclude_layout_from_plain", True)),
    )
    atomic_write_parquet(gt_df, data_dir / "gt_features_raw.parquet")
    fail = gt_df[gt_df["ast_parse_status"] == "fail"]
    atomic_write_parquet(fail, data_dir / "parse_failures.parquet")
    print(f"[joint] GT pages={len(gt_df)} fail={len(fail)}")

    print("[joint] aligning embeddings …")
    page_df, X_fit, qa = build_aligned_table(cfg, gt_df=gt_df)
    fit_df = page_df.loc[page_df["cluster_fit"]].reset_index(drop=True)
    assert len(fit_df) == X_fit.shape[0]

    # save original embeddings for fit set
    atomic_write_npy(data_dir / "image_embedding_original.npy", X_fit)
    gt_raw = gt_matrix(fit_df)
    atomic_write_npy(data_dir / "gt_features_raw.npy", gt_raw.astype(np.float32))
    atomic_write_json(data_dir / "qa_summary.json", qa)

    print("[joint] PCA + scalers …")
    bundle = fit_transform_features(X_fit, gt_raw, cfg)
    save_models(bundle, models_dir)
    atomic_write_npy(data_dir / "image_embedding_pca.npy", bundle["image_embedding_pca"])
    atomic_write_npy(data_dir / "gt_features_scaled.npy", bundle["gt_features_scaled"])
    atomic_write_npy(data_dir / "joint_features.npy", bundle["joint_features"])
    plot_pca_variance(
        bundle["pca_cumulative_full"],
        bundle["pca_n_components"],
        figures_dir / "image_pca_variance.png",
    )

    feat_blocks = {
        "image": bundle["image_features"],
        "text": bundle["text_features"],
        "joint": bundle["joint_features"],
    }
    print("[joint] K-Means sweeps E0/E1/E2 …")
    metrics, models, labels = run_kmeans_sweep(feat_blocks, cfg)
    metrics.to_csv(metrics_dir / "clustering_metrics.csv", index=False)
    plot_k_curves(metrics, figures_dir / "inertia_by_k.png", figures_dir / "silhouette_by_k.png")

    final_k, reason = select_final_k(metrics, cfg, n=len(fit_df))
    print(f"[joint] final_k={final_k} ({reason})")
    save_kmeans_models(models, models_dir, final_k)

    joint_km = models["joint"][final_k]
    joint_lab = labels["joint"][final_k]
    centers = joint_km.cluster_centers_
    dists = np.linalg.norm(bundle["joint_features"] - centers[joint_lab], axis=1)
    sizes = np.bincount(joint_lab, minlength=final_k)

    page_df = attach_labels(
        page_df,
        page_df["cluster_fit"].to_numpy(),
        labels,
        final_k,
        joint_km,
        bundle["joint_features"],
    )
    # keep only useful columns orderly
    atomic_write_parquet(page_df, data_dir / "page_features.parquet")
    page_df.drop(columns=["gt_text"], errors="ignore").to_csv(
        data_dir / "page_features_summary.csv", index=False
    )

    pd.DataFrame(
        {
            "cluster_id": np.arange(final_k),
            "cluster_size": sizes,
            "cluster_frac": sizes / sizes.sum(),
        }
    ).to_csv(metrics_dir / "cluster_sizes.csv", index=False)

    center_rows = []
    for c in range(final_k):
        center_rows.append(
            {
                "cluster_id": c,
                "cluster_size": int(sizes[c]),
                **{f"joint_dim_{j}": float(centers[c, j]) for j in range(min(8, centers.shape[1]))},
            }
        )
    pd.DataFrame(center_rows).to_csv(metrics_dir / "cluster_centers.csv", index=False)
    atomic_write_npy(metrics_dir / "joint_cluster_centers.npy", centers.astype(np.float32))

    plot_cluster_sizes(sizes, figures_dir / "final_cluster_sizes.png")
    plot_joint_pca2d(
        bundle["joint_features"],
        joint_lab,
        figures_dir / "joint_pca_2d.png",
        seed=int(cfg.get("random_state", 42)),
    )
    plot_joint_umap(bundle["joint_features"], joint_lab, figures_dir / "joint_umap_2d.png", cfg)
    plot_gt_heatmap(bundle["gt_features_scaled"], joint_lab, figures_dir / "gt_cluster_heatmap.png")
    plot_gt_boxplots(gt_raw, joint_lab, figures_dir / "gt_cluster_boxplots.png")
    # crosstab at same K for image vs joint
    plot_image_vs_joint_crosstab(
        labels["image"][final_k],
        joint_lab,
        figures_dir / "image_vs_joint_crosstab.png",
    )

    gal = cfg.get("galleries") or {}
    reps = write_representatives(
        fit_df,
        joint_lab,
        dists,
        reps_dir,
        center_n=int(gal.get("center_n", 20)),
        outlier_n=int(gal.get("outlier_n", 12)),
    )
    reps.to_csv(metrics_dir / "representatives.csv", index=False)

    write_data_quality_report(qa, fit_df, reports_dir)
    write_cluster_cards(fit_df, joint_lab, Path(cfg["paths"]["cluster_cards_dir"]))
    write_experiment_report(
        cfg=cfg,
        qa=qa,
        bundle=bundle,
        metrics=metrics,
        final_k=final_k,
        final_reason=reason,
        sizes=sizes,
        reports_dir=reports_dir,
    )

    summary = {
        "output_root": cfg["paths"]["output_root"],
        "final_k": final_k,
        "final_k_reason": reason,
        "pca_n_components": bundle["pca_n_components"],
        "pca_variance": bundle["pca_explained_variance_ratio_sum"],
        "n_cluster_fit": int(qa["n_cluster_fit"]),
        "cluster_sizes": sizes.tolist(),
    }
    atomic_write_json(Path(cfg["paths"]["output_root"]) / "summary.json", summary)
    print(f"[joint] done → {cfg['paths']['output_root']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Joint image+GT K-Means clustering v1")
    parser.add_argument(
        "--config",
        default="configs/joint_cluster/experiment_config.yaml",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
