"""Leiden clustering on full 1280-D L2 embeddings; sweep resolution 0.1 → 1.0."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from ..utils import atomic_write_json, ensure_dir
from .io_util import atomic_write_parquet, file_sha256

try:
    import igraph as ig
    import leidenalg
except Exception as e:  # noqa: BLE001
    ig = None
    leidenalg = None
    _LEIDEN_ERR = e
else:
    _LEIDEN_ERR = None


def _silhouette_cosine(
    X: np.ndarray, labels: np.ndarray, rng: np.random.Generator, sample: int = 5000
) -> float:
    uniq = set(int(x) for x in labels.tolist())
    if len(uniq) < 2:
        return float("nan")
    n = X.shape[0]
    if sample < n:
        ii = rng.choice(n, size=sample, replace=False)
        lab = labels[ii]
        if len(set(int(x) for x in lab.tolist())) < 2:
            return float("nan")
        return float(silhouette_score(X[ii], lab, metric="cosine"))
    return float(silhouette_score(X, labels, metric="cosine"))


def _size_stats(labels: np.ndarray) -> dict[str, Any]:
    sizes = np.bincount(labels.astype(int))
    sizes = sizes[sizes > 0]
    return {
        "n_clusters": int(sizes.size),
        "min_cluster_size": int(sizes.min()) if sizes.size else 0,
        "max_cluster_size": int(sizes.max()) if sizes.size else 0,
        "mean_cluster_size": float(sizes.mean()) if sizes.size else 0.0,
    }


def build_knn_graph(
    X: np.ndarray, *, knn_k: int = 15, n_jobs: int = 8
) -> "ig.Graph":
    """Undirected weighted kNN graph on L2 unit vectors (euclidean ≡ cosine rank)."""
    if ig is None:
        raise RuntimeError(f"igraph/leidenalg unavailable: {_LEIDEN_ERR}")
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, n), metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    dists, inds = nn.kneighbors(X)

    edge_w: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j_pos in range(1, inds.shape[1]):
            j = int(inds[i, j_pos])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            w = float(1.0 / (1e-6 + dists[i, j_pos]))
            prev = edge_w.get((a, b))
            edge_w[(a, b)] = w if prev is None else max(prev, w)

    g = ig.Graph(n=n, edges=list(edge_w.keys()), directed=False)
    g.es["weight"] = list(edge_w.values())
    return g


def run_leiden_once(g: "ig.Graph", *, resolution: float, seed: int) -> tuple[np.ndarray, float]:
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    labels = np.asarray(partition.membership, dtype=np.int32)
    return labels, float(partition.modularity)


def evaluate_leiden_resolutions(
    X: np.ndarray,
    *,
    out_dir: Path,
    resolutions: list[float],
    knn_k: int = 15,
    n_seeds: int = 10,
    n_boot: int = 5,
    seed: int = 42,
    ids: list[str] | None = None,
    save_assignments: bool = True,
    metrics_csv_name: str | None = None,
) -> pd.DataFrame:
    """Sweep Leiden resolution; write metrics CSV (+ optional assignments)."""
    if leidenalg is None or ig is None:
        raise RuntimeError(f"igraph/leidenalg unavailable: {_LEIDEN_ERR}")

    X = np.asarray(X, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    assert X.ndim == 2 and X.shape[1] == 1280, f"expected (*,1280), got {X.shape}"

    out_dir = ensure_dir(out_dir)
    print(f"[leiden] building kNN graph knn={knn_k} on {X.shape} …")
    g = build_knn_graph(X, knn_k=knn_k)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for res in resolutions:
        print(f"[leiden] knn={knn_k} resolution={res:.2f}")
        labels_runs: list[np.ndarray] = []
        sils: list[float] = []
        mods: list[float] = []
        for s in range(n_seeds):
            lab, mod = run_leiden_once(g, resolution=res, seed=seed + s)
            labels_runs.append(lab)
            mods.append(mod)
            sils.append(_silhouette_cosine(X, lab, rng))

        aris = [
            float(adjusted_rand_score(labels_runs[i], labels_runs[j]))
            for i in range(len(labels_runs))
            for j in range(i + 1, len(labels_runs))
        ]
        nmis = [
            float(normalized_mutual_info_score(labels_runs[i], labels_runs[j]))
            for i in range(len(labels_runs))
            for j in range(i + 1, len(labels_runs))
        ]

        ref = labels_runs[int(np.nanargmax(sils))]
        boot_aris = [
            float(adjusted_rand_score(ref, run_leiden_once(g, resolution=res, seed=seed + 1000 + b)[0]))
            for b in range(n_boot)
        ]
        boot_nmis = [
            float(
                normalized_mutual_info_score(
                    ref, run_leiden_once(g, resolution=res, seed=seed + 2000 + b)[0]
                )
            )
            for b in range(n_boot)
        ]

        ari_seed = float(np.mean(aris)) if aris else 1.0
        nmi_seed = float(np.mean(nmis)) if nmis else 1.0
        ari_boot = float(np.mean(boot_aris)) if boot_aris else float("nan")
        nmi_boot = float(np.mean(boot_nmis)) if boot_nmis else float("nan")
        stability = float(np.nanmean([ari_seed, ari_boot]))
        stats = _size_stats(ref)

        row = {
            "method": "leiden",
            "resolution": float(res),
            "knn_k": int(knn_k),
            "cosine_silhouette": float(np.nanmean(sils)),
            "cosine_silhouette_std": float(np.nanstd(sils)),
            "modularity": float(np.mean(mods)),
            "modularity_std": float(np.std(mods)),
            "ARI": ari_seed,
            "ARI_std": float(np.std(aris)) if aris else 0.0,
            "NMI": nmi_seed,
            "NMI_std": float(np.std(nmis)) if nmis else 0.0,
            "ARI_bootstrap": ari_boot,
            "NMI_bootstrap": nmi_boot,
            "clustering_stability": stability,
            "n_seeds": int(n_seeds),
            "n_bootstrap": int(n_boot),
            "n_samples": int(X.shape[0]),
            "dim": int(X.shape[1]),
            **stats,
        }
        rows.append(row)

        if save_assignments:
            res_dir = ensure_dir(out_dir / f"res_{res:.1f}")
            np.save(res_dir / "labels.npy", ref)
            if ids is not None:
                assign = pd.DataFrame(
                    {
                        "image_id": ids,
                        "cluster": ref.astype(int),
                        "method": "leiden",
                        "resolution": float(res),
                        "knn_k": int(knn_k),
                    }
                )
                atomic_write_parquet(assign, res_dir / "assignments.parquet")
                assign.to_csv(res_dir / "assignments.csv", index=False)
            atomic_write_json(res_dir / "summary.json", row)

    frame = pd.DataFrame(rows)
    csv_name = metrics_csv_name or "leiden_metrics.csv"
    csv_path = out_dir / csv_name
    frame.to_csv(csv_path, index=False)
    atomic_write_json(
        out_dir / "overview.json",
        {
            "method": "leiden",
            "dim": int(X.shape[1]),
            "knn_k": knn_k,
            "resolutions": [float(r) for r in resolutions],
            "metrics_csv": str(csv_path),
            "n_samples": int(X.shape[0]),
        },
    )
    print(f"[leiden] wrote {csv_path} rows={len(frame)}")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leiden on 1280-D embeddings")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/embeddings/deepseek_ocr2_mean_l2.npy"
        ),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/metadata/embedding_index.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp/leiden"),
    )
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument(
        "--knn-list",
        type=int,
        nargs="+",
        default=None,
        help="If set, sweep these knn values (each under output-dir/knn_{k}/).",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--res-min", type=float, default=0.1)
    parser.add_argument("--res-max", type=float, default=1.0)
    parser.add_argument("--res-step", type=float, default=0.1)
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=None,
        help="Explicit resolution list (overrides --res-min/max/step).",
    )
    args = parser.parse_args(argv)

    if leidenalg is None or ig is None:
        raise SystemExit(f"igraph/leidenalg required: {_LEIDEN_ERR}")

    X = np.load(args.embeddings)
    print(f"[leiden] load {args.embeddings} shape={X.shape} sha={file_sha256(args.embeddings)[:12]}")
    ids: list[str] | None = None
    if args.index.is_file():
        idx = pd.read_parquet(args.index)
        ok = idx[idx["status"] == "ok"] if "status" in idx.columns else idx
        ids = ok["image_id"].astype(str).tolist()
        if len(ids) != X.shape[0]:
            print(f"[leiden] warn: index {len(ids)} != emb {X.shape[0]}; skip id alignment")
            ids = None

    if args.resolutions is not None:
        resolutions = [float(r) for r in args.resolutions]
    else:
        resolutions = [
            round(float(r), 10)
            for r in np.arange(args.res_min, args.res_max + 1e-9, args.res_step)
        ]

    knn_list = list(args.knn_list) if args.knn_list else [int(args.knn_k)]
    parts: list[pd.DataFrame] = []
    for knn_k in knn_list:
        if args.knn_list:
            out_dir = args.output_dir / f"knn_{knn_k}"
            csv_name = f"leiden_metrics_knn{knn_k}_res{'_'.join(f'{r:g}' for r in resolutions)}.csv"
        else:
            out_dir = args.output_dir
            csv_name = "leiden_metrics_resolution_0.1_1.0.csv"
        print(f"[leiden] === knn={knn_k} → {out_dir} ===")
        frame = evaluate_leiden_resolutions(
            X,
            out_dir=out_dir,
            resolutions=resolutions,
            knn_k=int(knn_k),
            n_seeds=args.n_seeds,
            n_boot=args.n_boot,
            seed=args.seed,
            ids=ids,
            save_assignments=True,
            metrics_csv_name=csv_name,
        )
        parts.append(frame)

    if args.knn_list and parts:
        combined = pd.concat(parts, ignore_index=True)
        comb_path = args.output_dir / "leiden_metrics_knn_sweep.csv"
        combined.to_csv(comb_path, index=False)
        print(f"[leiden] combined → {comb_path} rows={len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
