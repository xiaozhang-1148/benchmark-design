"""Sweep spherical KMeans for k=2..1000 on B2 PCA64 features; per-cluster cosine silhouette.

Supports resume: skips k with existing cluster_silhouette/k{k}.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from src.utils import ensure_dir


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def fit_spherical_kmeans(
    X: np.ndarray, k: int, *, seed: int, n_init: int, max_iter: int
) -> tuple[np.ndarray, float]:
    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
        algorithm="lloyd",
        max_iter=max_iter,
    )
    labels = km.fit_predict(X).astype(np.int32)
    return labels, float(km.inertia_)


def stratified_indices(labels: np.ndarray, k: int, sample_n: int, rng: np.random.Generator) -> np.ndarray:
    n = labels.shape[0]
    sizes = np.bincount(labels, minlength=k)
    sample_n = min(sample_n, n)
    quotas = np.zeros(k, dtype=int)
    for c in range(k):
        if sizes[c] <= 0:
            continue
        quotas[c] = min(sizes[c], max(3, int(round(sample_n * sizes[c] / n))))
    if quotas.sum() == 0:
        return rng.choice(n, size=min(sample_n, n), replace=False)
    if quotas.sum() > sample_n:
        quotas = np.maximum(0, (quotas * sample_n // max(int(quotas.sum()), 1)).astype(int))
    # fill remainder toward larger clusters
    guard = 0
    while quotas.sum() < sample_n and guard < n:
        c = int(np.argmax(sizes - quotas))
        if quotas[c] < sizes[c]:
            quotas[c] += 1
        else:
            break
        guard += 1
    parts: list[np.ndarray] = []
    for c in range(k):
        q = int(quotas[c])
        if q <= 0:
            continue
        members = np.where(labels == c)[0]
        parts.append(rng.choice(members, size=min(q, len(members)), replace=False))
    return np.concatenate(parts) if parts else rng.choice(n, size=sample_n, replace=False)


def per_cluster_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    sample_n: int,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed + k)
    sizes = np.bincount(labels, minlength=k)
    ii = stratified_indices(labels, k, sample_n, rng)
    labs = labels[ii]
    if len(set(int(x) for x in labs.tolist())) < 2:
        global_sil = float("nan")
        s_i = np.full(len(ii), np.nan, dtype=np.float64)
    else:
        global_sil = float(silhouette_score(X[ii], labs, metric="cosine"))
        s_i = silhouette_samples(X[ii], labs, metric="cosine")

    rows: list[dict[str, Any]] = []
    for c in range(k):
        mask = labs == c
        n_in = int(mask.sum())
        rows.append(
            {
                "k": int(k),
                "cluster": int(c),
                "n_cluster": int(sizes[c]),
                "n_in_silhouette_sample": n_in,
                "cluster_silhouette": float(np.mean(s_i[mask])) if n_in else float("nan"),
                "cluster_silhouette_std": float(np.std(s_i[mask])) if n_in else float("nan"),
                "metric": "cosine",
                "global_silhouette_on_sample": global_sil,
                "sample_n_total": int(len(ii)),
            }
        )
    return pd.DataFrame(rows), global_sil


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    write_header = not path.is_file()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "cluster_exp/exp/B2_1/features_l2.npy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp/exp_1000"
        ),
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--silhouette-sample", type=int, default=10000)
    parser.add_argument("--save-labels", action="store_true", default=True)
    parser.add_argument("--no-save-labels", action="store_true")
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    sil_dir = ensure_dir(out / "cluster_silhouette")
    lab_dir = ensure_dir(out / "labels")
    metrics_path = out / "metrics_by_k.csv"
    save_labels = bool(args.save_labels) and not bool(args.no_save_labels)

    X = _l2(np.load(args.features))
    n, d = X.shape
    print(f"[exp_1000] features={args.features} shape={X.shape}", flush=True)

    overview = {
        "features": str(args.features),
        "n_samples": int(n),
        "dim": int(d),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "seed": int(args.seed),
        "n_init": int(args.n_init),
        "max_iter": int(args.max_iter),
        "silhouette_sample": int(args.silhouette_sample),
        "metric": "cosine",
        "method": "spherical_kmeans",
        "resume": True,
    }
    (out / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    # copy features once for self-contained run
    feat_dst = out / "features_l2.npy"
    if not feat_dst.is_file():
        print("[exp_1000] copying features …", flush=True)
        np.save(feat_dst, X)

    done = {int(p.stem[1:]) for p in sil_dir.glob("k*.csv") if p.stem[1:].isdigit()}
    print(f"[exp_1000] already done: {len(done)} ks", flush=True)

    t0 = time.time()
    for k in range(args.k_min, args.k_max + 1):
        if k >= n:
            print(f"[exp_1000] stop: k={k} >= n={n}", flush=True)
            break
        if k in done:
            continue
        t_k = time.time()
        labels, inertia = fit_spherical_kmeans(
            X, k, seed=args.seed, n_init=args.n_init, max_iter=args.max_iter
        )
        sizes = np.bincount(labels, minlength=k)
        per_df, global_sil = per_cluster_silhouette(
            X,
            labels,
            k,
            sample_n=min(args.silhouette_sample, n),
            seed=args.seed,
        )
        per_path = sil_dir / f"k{k}.csv"
        per_df.to_csv(per_path, index=False)

        if save_labels:
            np.save(lab_dir / f"k{k}.npy", labels)

        row = {
            "k": int(k),
            "method": "spherical_kmeans",
            "n_samples": int(n),
            "dim": int(d),
            "inertia": float(inertia),
            "global_cosine_silhouette": float(global_sil),
            "mean_cluster_silhouette": float(per_df["cluster_silhouette"].mean()),
            "min_cluster_silhouette": float(per_df["cluster_silhouette"].min()),
            "max_cluster_silhouette": float(per_df["cluster_silhouette"].max()),
            "min_cluster_size": int(sizes[sizes > 0].min()) if sizes.any() else 0,
            "max_cluster_size": int(sizes.max()) if sizes.any() else 0,
            "mean_cluster_size": float(sizes.mean()),
            "n_singleton": int(np.sum(sizes == 1)),
            "seed": int(args.seed),
            "n_init": int(args.n_init),
            "max_iter": int(args.max_iter),
            "elapsed_sec": float(time.time() - t_k),
        }
        append_metrics_row(metrics_path, row)
        done.add(k)
        print(
            f"[exp_1000] k={k} sil={global_sil:.4f} "
            f"cluster_sil=[{row['min_cluster_silhouette']:.3f},{row['max_cluster_silhouette']:.3f}] "
            f"size=[{row['min_cluster_size']},{row['max_cluster_size']}] "
            f"{row['elapsed_sec']:.1f}s  progress={len(done)}/{args.k_max - args.k_min + 1}",
            flush=True,
        )

    # consolidate per-cluster silhouettes into one parquet (may be large)
    print("[exp_1000] consolidating cluster silhouette tables …", flush=True)
    parts = []
    for k in range(args.k_min, args.k_max + 1):
        p = sil_dir / f"k{k}.csv"
        if p.is_file():
            parts.append(pd.read_csv(p))
    if parts:
        all_per = pd.concat(parts, ignore_index=True)
        all_per.to_csv(out / "cluster_silhouette_all.csv", index=False)
        try:
            all_per.to_parquet(out / "cluster_silhouette_all.parquet", index=False)
        except Exception as e:  # noqa: BLE001
            print(f"[exp_1000] parquet skip: {e}", flush=True)

    print(f"[exp_1000] done in {(time.time() - t0) / 60:.1f} min → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
