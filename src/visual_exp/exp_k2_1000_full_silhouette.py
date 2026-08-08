"""exp_1000 → exp_1000_1: 10-seed spherical KMeans + full cosine silhouette.

For each k: fit 10 seeds in a thread pool, keep the seed with best full global
silhouette, write per-cluster silhouette on ALL points. Supports resume.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from src.utils import ensure_dir


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def _fit_and_silhouette(
    X: np.ndarray, k: int, seed: int, max_iter: int
) -> tuple[int, np.ndarray, np.ndarray, float, float]:
    km = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=1,
        algorithm="lloyd",
        max_iter=max_iter,
    )
    labels = km.fit_predict(X).astype(np.int32)
    s_i = silhouette_samples(X, labels, metric="cosine")
    global_sil = float(np.mean(s_i))
    return int(seed), labels, s_i, global_sil, float(km.inertia_)


def rows_from_samples(
    labels: np.ndarray, s_i: np.ndarray, k: int, global_sil: float, *, seed_base: int, n_seeds: int, ref_seed: int
) -> pd.DataFrame:
    sizes = np.bincount(labels, minlength=k)
    n = int(labels.shape[0])
    rows: list[dict[str, Any]] = []
    for c in range(k):
        mask = labels == c
        n_in = int(mask.sum())
        rows.append(
            {
                "k": int(k),
                "cluster": int(c),
                "n_cluster": int(sizes[c]),
                "n_in_silhouette": n_in,
                "cluster_silhouette": float(np.mean(s_i[mask])) if n_in else float("nan"),
                "cluster_silhouette_std": float(np.std(s_i[mask])) if n_in else float("nan"),
                "metric": "cosine",
                "global_silhouette": float(global_sil),
                "n_total": n,
                "seed_base": int(seed_base),
                "n_seeds": int(n_seeds),
                "ref_seed": int(ref_seed),
            }
        )
    return pd.DataFrame(rows)


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    write_header = not path.is_file()
    frame.to_csv(path, mode="a", header=write_header, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp/exp_1000"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp/exp_1000_1"
        ),
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10, help="thread pool size for seeds")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="clear prior cluster_silhouette/metrics/labels under output-dir before run",
    )
    args = parser.parse_args()

    src = args.source_dir
    out = ensure_dir(args.output_dir)
    sil_dir = ensure_dir(out / "cluster_silhouette")
    lab_dir = ensure_dir(out / "labels")
    metrics_path = out / "metrics_by_k.csv"

    feat_src = src / "features_l2.npy"
    if not feat_src.is_file():
        raise FileNotFoundError(feat_src)

    if args.reset:
        print("[exp_1000_1] reset: clearing prior silhouette/metrics/labels …", flush=True)
        for p in sil_dir.glob("k*.csv"):
            p.unlink()
        for p in lab_dir.glob("k*.npy"):
            p.unlink()
        if metrics_path.is_file():
            metrics_path.unlink()
        for name in (
            "cluster_silhouette_all.csv",
            "cluster_silhouette_all.parquet",
        ):
            p = out / name
            if p.is_file():
                p.unlink()

    X = _l2(np.load(feat_src))
    n, d = X.shape
    print(
        f"[exp_1000_1] features={feat_src} shape={X.shape} "
        f"n_seeds={args.n_seeds} workers={args.workers}",
        flush=True,
    )

    feat_dst = out / "features_l2.npy"
    if not feat_dst.is_file():
        print("[exp_1000_1] copying features …", flush=True)
        shutil.copy2(feat_src, feat_dst)

    overview = {
        "source_dir": str(src),
        "features": str(feat_dst),
        "n_samples": int(n),
        "dim": int(d),
        "k_min": int(args.k_min),
        "k_max": int(args.k_max),
        "seed_base": int(args.seed),
        "n_seeds": int(args.n_seeds),
        "max_iter": int(args.max_iter),
        "workers": int(args.workers),
        "metric": "cosine",
        "method": "spherical_kmeans",
        "silhouette": "full_all_points",
        "selection": "best_of_n_seeds_by_global_silhouette",
        "resume": True,
        "note": (
            "For each k, fit n_seeds KMeans (n_init=1 each) in a thread pool; "
            "keep the seed with highest full cosine silhouette; "
            "per-cluster silhouette = mean of silhouette_samples over ALL points."
        ),
    }
    (out / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    done = {int(p.stem[1:]) for p in sil_dir.glob("k*.csv") if p.stem[1:].isdigit()}
    total = args.k_max - args.k_min + 1
    print(f"[exp_1000_1] already done: {len(done)} / {total}", flush=True)

    workers = max(1, min(int(args.workers), int(args.n_seeds)))
    t0 = time.time()
    for k in range(args.k_min, args.k_max + 1):
        if k >= n:
            print(f"[exp_1000_1] stop: k={k} >= n={n}", flush=True)
            break
        if k in done:
            continue

        t_k = time.time()
        seeds = [args.seed + s for s in range(args.n_seeds)]
        results: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(_fit_and_silhouette, X, k, seed, args.max_iter): seed for seed in seeds
            }
            for fut in as_completed(futs):
                seed, labels, s_i, sil, inertia = fut.result()
                results.append((seed, labels, s_i, sil, inertia))
                print(f"  k={k} seed={seed} sil={sil:.6f}", flush=True)

        # best by global silhouette (not average)
        best = max(results, key=lambda r: r[3])
        ref_seed, labels, s_i, global_sil, inertia = best
        sils = np.array([r[3] for r in results], dtype=np.float64)
        sizes = np.bincount(labels, minlength=k)

        per_df = rows_from_samples(
            labels,
            s_i,
            k,
            global_sil,
            seed_base=args.seed,
            n_seeds=args.n_seeds,
            ref_seed=ref_seed,
        )
        per_df.to_csv(sil_dir / f"k{k}.csv", index=False)
        np.save(lab_dir / f"k{k}.npy", labels)

        row = {
            "k": int(k),
            "method": "spherical_kmeans",
            "n_samples": int(n),
            "dim": int(d),
            "inertia": float(inertia),
            "global_cosine_silhouette": float(global_sil),
            "global_cosine_silhouette_mean_seeds": float(np.mean(sils)),
            "global_cosine_silhouette_std_seeds": float(np.std(sils)),
            "mean_cluster_silhouette": float(per_df["cluster_silhouette"].mean()),
            "min_cluster_silhouette": float(per_df["cluster_silhouette"].min()),
            "max_cluster_silhouette": float(per_df["cluster_silhouette"].max()),
            "min_cluster_size": int(sizes[sizes > 0].min()) if sizes.any() else 0,
            "max_cluster_size": int(sizes.max()) if sizes.any() else 0,
            "mean_cluster_size": float(sizes.mean()),
            "n_singleton": int(np.sum(sizes == 1)),
            "seed_base": int(args.seed),
            "ref_seed": int(ref_seed),
            "n_seeds": int(args.n_seeds),
            "max_iter": int(args.max_iter),
            "workers": int(workers),
            "elapsed_sec": float(time.time() - t_k),
            "silhouette_mode": "full_best_of_seeds",
        }
        append_metrics_row(metrics_path, row)
        done.add(k)
        print(
            f"[exp_1000_1] k={k} best_sil={global_sil:.4f} "
            f"mean_seeds={row['global_cosine_silhouette_mean_seeds']:.4f}±"
            f"{row['global_cosine_silhouette_std_seeds']:.4f} "
            f"ref_seed={ref_seed} "
            f"cluster_sil=[{row['min_cluster_silhouette']:.3f},{row['max_cluster_silhouette']:.3f}] "
            f"{row['elapsed_sec']:.1f}s  progress={len(done)}/{total}",
            flush=True,
        )

    print("[exp_1000_1] consolidating cluster silhouette tables …", flush=True)
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
            print(f"[exp_1000_1] parquet skip: {e}", flush=True)

    print(f"[exp_1000_1] done in {(time.time() - t0) / 60:.1f} min → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
