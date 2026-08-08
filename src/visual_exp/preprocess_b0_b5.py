"""Preprocessing ablations B0–B5 on A1 1280-D embeddings + Spherical KMeans.

Pipeline: A1 (mean-pooled 1280) → variant transform → L2 → Spherical KMeans (K=2..10).

B0  A1 + L2
B1  A1 + center + L2
B2  A1 + PCA 64 + L2
B3  A1 + PCA 128 + L2
B4  A1 + PCA 256 + L2
B5  A1 + drop first 1/2/3 PCs + L2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..utils import atomic_write_json, ensure_dir
from .ablation_a0_a3 import evaluate_spherical_kmeans_csv
from .io_util import atomic_write_npy, file_sha256


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def prep_b0(X: np.ndarray, *, seed: int = 42) -> np.ndarray:
    return _l2(X)


def prep_b1(X: np.ndarray, *, seed: int = 42) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return _l2(X - X.mean(axis=0, keepdims=True))


def prep_pca(X: np.ndarray, *, n_components: int, seed: int = 42) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    pca = PCA(n_components=n_components, random_state=seed, svd_solver="randomized")
    Z = pca.fit_transform(X)
    return _l2(Z)


def prep_drop_top_pcs(X: np.ndarray, *, n_drop: int, seed: int = 42) -> np.ndarray:
    """Subtract projection onto the first n_drop PCs, then L2 (still 1280-D)."""
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    pca = PCA(n_components=n_drop, random_state=seed, svd_solver="full")
    Z = pca.fit_transform(Xc)
    X_res = Xc - Z @ pca.components_
    return _l2(X_res)

VARIANT_PREP: dict[str, tuple[str, Callable[..., np.ndarray], dict[str, Any]]] = {
    "B0": ("A1 + L2", prep_b0, {}),
    "B1": ("A1 + center + L2", prep_b1, {}),
    "B2": ("A1 + PCA 64 + L2", prep_pca, {"n_components": 64}),
    "B3": ("A1 + PCA 128 + L2", prep_pca, {"n_components": 128}),
    "B4": ("A1 + PCA 256 + L2", prep_pca, {"n_components": 256}),
    "B5_pc1": ("A1 + drop first 1 PC + L2", prep_drop_top_pcs, {"n_drop": 1}),
    "B5_pc2": ("A1 + drop first 2 PCs + L2", prep_drop_top_pcs, {"n_drop": 2}),
    "B5_pc3": ("A1 + drop first 3 PCs + L2", prep_drop_top_pcs, {"n_drop": 3}),
}


def run_b_ablation(
    *,
    embeddings: Path,
    output_root: Path,
    variants: tuple[str, ...] | None = None,
    k_min: int = 2,
    k_max: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    output_root = ensure_dir(output_root)
    X0 = np.load(embeddings)
    print(f"[B-exp] load {embeddings} shape={X0.shape} sha={file_sha256(embeddings)[:12]}")
    assert X0.ndim == 2 and X0.shape[1] == 1280, f"expected (*,1280), got {X0.shape}"

    selected = variants or tuple(VARIANT_PREP.keys())
    parts: list[pd.DataFrame] = []

    for vid in selected:
        if vid not in VARIANT_PREP:
            raise KeyError(f"unknown variant {vid}; choose from {list(VARIANT_PREP)}")
        desc, fn, kwargs = VARIANT_PREP[vid]
        out_dir = ensure_dir(output_root / vid)
        # B5_* live under B5/ as well for a clean folder tree.
        if vid.startswith("B5_"):
            out_dir = ensure_dir(output_root / "B5" / vid)

        print(f"[{vid}] {desc}")
        Z = fn(X0, seed=seed, **kwargs)
        emb_path = out_dir / "features_l2.npy"
        atomic_write_npy(emb_path, Z)
        atomic_write_json(
            out_dir / "variant_spec.json",
            {
                "variant": vid,
                "description": desc,
                "source_embeddings": str(embeddings),
                "source_sha12": file_sha256(embeddings)[:12],
                "feature_shape": list(Z.shape),
                "kwargs": kwargs,
            },
        )
        csv_path = out_dir / "spherical_kmeans_metrics_k2_10.csv"
        frame = evaluate_spherical_kmeans_csv(
            Z,
            out_csv=csv_path,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
            variant=vid,
        )
        print(f"[{vid}] wrote {csv_path} dim={Z.shape[1]} rows={len(frame)}")
        parts.append(frame)

    combined = pd.concat(parts, ignore_index=True)
    comb_path = output_root / "B0_B5_spherical_kmeans_metrics_k2_10.csv"
    combined.to_csv(comb_path, index=False)
    atomic_write_json(
        output_root / "B0_B5_overview.json",
        {
            "source_embeddings": str(embeddings),
            "variants": list(selected),
            "k_min": k_min,
            "k_max": k_max,
            "metrics_csv": str(comb_path),
            "n_samples": int(X0.shape[0]),
            "source_dim": int(X0.shape[1]),
        },
    )
    print(f"[B-exp] combined → {comb_path} rows={len(combined)}")
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B0–B5 preprocess + Spherical KMeans on A1 embeddings")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp/"
            "A1/embeddings/deepseek_ocr2_mean_l2.npy"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=f"Subset of {list(VARIANT_PREP)}",
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    emb = args.embeddings
    if not emb.is_file():
        emb = Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/embeddings/deepseek_ocr2_mean_l2.npy"
        )
        print(f"[B-exp] fallback embeddings → {emb}")

    run_b_ablation(
        embeddings=emb,
        output_root=args.output_root,
        variants=tuple(args.variants) if args.variants else None,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
