"""Full (non-sampled) per-cluster cosine silhouette for B2 → B2_2.

Mirrors B2_1 cluster_silhouette_* outputs, but silhouette_samples / silhouette_score
run on all points (no stratified subsample).
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from src.utils import atomic_write_json, ensure_dir
from src.visual_exp.ablation_a0_a3 import _spherical_kmeans
from src.visual_exp.io_util import atomic_write_npy


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b2-dir",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp/exp/B2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp/exp/B2_2"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--variant", type=str, default="B2_2")
    args = parser.parse_args()

    b2 = args.b2_dir
    out = ensure_dir(args.output_dir)
    metrics_src = b2 / "spherical_kmeans_metrics_k2_10.csv"
    feat_src = b2 / "features_l2.npy"
    if not metrics_src.is_file() or not feat_src.is_file():
        raise FileNotFoundError(f"missing B2 inputs under {b2}")

    metrics = pd.read_csv(metrics_src)
    k_list = sorted(int(k) for k in metrics["k"].unique())

    X = _l2(np.load(feat_src))
    n, d = X.shape
    print(f"[B2_2] features={feat_src} shape={X.shape} ks={k_list}", flush=True)

    feat_dst = out / "features_l2.npy"
    if not feat_dst.is_file():
        shutil.copy2(feat_src, feat_dst)

    shutil.copy2(metrics_src, out / "spherical_kmeans_metrics_k2_10.csv")

    atomic_write_json(
        out / "variant_spec.json",
        {
            "variant": args.variant,
            "description": "B2 full (non-sampled) per-cluster cosine silhouette",
            "baseline_of": "B2",
            "source_features": str(feat_src),
            "source_metrics": str(metrics_src),
            "feature_shape": [int(n), int(d)],
            "silhouette": "full_all_points",
            "metric": "cosine",
            "n_seeds": int(args.n_seeds),
            "seed_base": int(args.seed),
            "kmeans_max_iter": 1000,
        },
    )

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for k in k_list:
        t_k = time.time()
        print(f"[B2_2] k={k}: fitting {args.n_seeds} seeds + full silhouette …", flush=True)
        best_i = -1
        best_sil = float("-inf")
        best_labels: np.ndarray | None = None
        best_s_i: np.ndarray | None = None
        for s in range(args.n_seeds):
            lab = _spherical_kmeans(X, k, args.seed + s, n_init=1)
            # full (no sample): samples once; mean == silhouette_score
            s_i = silhouette_samples(X, lab, metric="cosine")
            sil = float(np.mean(s_i))
            print(f"  seed={args.seed + s} global_sil={sil:.6f}", flush=True)
            if sil > best_sil:
                best_sil = sil
                best_i = s
                best_labels = lab
                best_s_i = s_i

        assert best_labels is not None and best_s_i is not None
        labels = best_labels
        s_i = best_s_i
        ref_global = float(best_sil)
        global_sil = ref_global
        atomic_write_npy(out / f"labels_spherical_k{k}.npy", labels)
        sizes = np.bincount(labels, minlength=k)

        cluster_sils: list[float] = []
        for c in range(k):
            mask = labels == c
            n_c = int(mask.sum())
            c_mean = float(np.mean(s_i[mask])) if n_c else float("nan")
            c_std = float(np.std(s_i[mask])) if n_c else float("nan")
            cluster_sils.append(c_mean)
            long_rows.append(
                {
                    "variant": args.variant,
                    "method": "spherical_kmeans",
                    "k": int(k),
                    "cluster": int(c),
                    "n_cluster": int(sizes[c]),
                    "n_in_silhouette": int(n_c),
                    "cluster_silhouette": c_mean,
                    "cluster_silhouette_std": c_std,
                    "metric": "cosine",
                    "ref_global_silhouette": global_sil,
                    "n_total": int(n),
                    "seed_base": int(args.seed),
                    "n_seeds_for_ref": int(args.n_seeds),
                    "ref_seed": int(args.seed + best_i),
                    "ref_seed_global_silhouette": ref_global,
                }
            )

        summary_rows.append(
            {
                "k": int(k),
                "ref_global_silhouette": global_sil,
                "mean_cluster_silhouette": float(np.nanmean(cluster_sils)),
                "min_cluster_silhouette": float(np.nanmin(cluster_sils)),
                "max_cluster_silhouette": float(np.nanmax(cluster_sils)),
                "n_clusters": int(k),
                "n_total": int(n),
            }
        )
        wide = {
            "variant": args.variant,
            "method": "spherical_kmeans",
            "k": int(k),
            "ref_global_silhouette": global_sil,
            "n_total": int(n),
        }
        for c, v in enumerate(cluster_sils):
            wide[f"cluster_{c}_silhouette"] = v
        wide_rows.append(wide)

        # also write per-k csv (same spirit as exp_1000 / B2_1 large-k files)
        per_k = pd.DataFrame([r for r in long_rows if r["k"] == k])
        per_k.to_csv(out / f"cluster_silhouette_k{k}.csv", index=False)

        print(
            f"[B2_2] k={k} global={global_sil:.4f} "
            f"cluster=[{min(cluster_sils):.3f},{max(cluster_sils):.3f}] "
            f"{time.time() - t_k:.1f}s",
            flush=True,
        )

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out / "cluster_silhouette_by_k.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "cluster_silhouette_summary.csv", index=False)
    pd.DataFrame(wide_rows).to_csv(out / "cluster_silhouette_by_k_wide.csv", index=False)

    atomic_write_json(
        out / "cluster_silhouette_overview.json",
        {
            "source_metrics": str(metrics_src),
            "features": str(feat_dst),
            "metric": "cosine",
            "note": (
                "Per-cluster silhouette = mean of silhouette_samples over ALL points "
                "(no subsample). Ref labels = best full cosine silhouette among n_seeds runs."
            ),
            "k_list": k_list,
            "n_samples": int(n),
            "outputs": [
                str(out / "cluster_silhouette_by_k.csv"),
                str(out / "cluster_silhouette_by_k_wide.csv"),
                str(out / "cluster_silhouette_summary.csv"),
            ],
        },
    )

    print(f"[B2_2] done → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
