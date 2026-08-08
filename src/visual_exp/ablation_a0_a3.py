"""Ablation A0–A3 + spherical KMeans metrics (K=2..10) on existing 1280-D embeddings.

By default embeddings are loaded from all_benchmark_v1 shards (no model re-extract).
True A2/A3 feature recipes need a separate extract pass (--force-extract).

Clustering uses full 1280-D L2 embeddings; spherical KMeans; CSV metrics only.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from ..utils import atomic_write_json, ensure_dir
from .ablation_extractor import VARIANT_SPECS
from .io_util import atomic_write_npy, atomic_write_parquet, file_sha256


def _spherical_kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 10) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, algorithm="lloyd", max_iter=1000)
    return km.fit_predict(X).astype(np.int32)


def _silhouette_cosine(X: np.ndarray, labels: np.ndarray, rng: np.random.Generator, sample: int = 5000) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    n = X.shape[0]
    if sample < n:
        ii = rng.choice(n, size=sample, replace=False)
        return float(silhouette_score(X[ii], labels[ii], metric="cosine"))
    return float(silhouette_score(X, labels, metric="cosine"))


def evaluate_spherical_kmeans_csv(
    X: np.ndarray,
    *,
    out_csv: Path,
    k_min: int = 2,
    k_max: int = 10,
    n_seeds: int = 10,
    n_boot: int = 5,
    seed: int = 42,
    variant: str = "A0",
) -> pd.DataFrame:
    """Write one CSV with ARI / NMI / cosine silhouette / stability for K=k_min..k_max."""
    X = np.asarray(X, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for k in range(k_min, k_max + 1):
        print(f"[{variant}] spherical_kmeans k={k}")
        labels_runs: list[np.ndarray] = []
        sils: list[float] = []
        for s in range(n_seeds):
            lab = _spherical_kmeans(X, k, seed + s, n_init=1)
            labels_runs.append(lab)
            sils.append(_silhouette_cosine(X, lab, rng))

        aris: list[float] = []
        nmis: list[float] = []
        for i in range(len(labels_runs)):
            for j in range(i + 1, len(labels_runs)):
                aris.append(float(adjusted_rand_score(labels_runs[i], labels_runs[j])))
                nmis.append(float(normalized_mutual_info_score(labels_runs[i], labels_runs[j])))

        ref = labels_runs[int(np.nanargmax(sils))]
        boot_aris = [
            float(adjusted_rand_score(ref, _spherical_kmeans(X, k, seed + 1000 + b, n_init=1)))
            for b in range(n_boot)
        ]
        boot_nmis = [
            float(
                normalized_mutual_info_score(
                    ref, _spherical_kmeans(X, k, seed + 2000 + b, n_init=1)
                )
            )
            for b in range(n_boot)
        ]

        ari_seed = float(np.mean(aris)) if aris else 1.0
        nmi_seed = float(np.mean(nmis)) if nmis else 1.0
        ari_boot = float(np.mean(boot_aris)) if boot_aris else float("nan")
        nmi_boot = float(np.mean(boot_nmis)) if boot_nmis else float("nan")
        stability = float(np.nanmean([ari_seed, ari_boot]))

        rows.append(
            {
                "variant": variant,
                "method": "spherical_kmeans",
                "k": int(k),
                "cosine_silhouette": float(np.nanmean(sils)),
                "cosine_silhouette_std": float(np.nanstd(sils)),
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
            }
        )

    frame = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    frame.to_csv(out_csv, index=False)
    return frame


def load_embeddings_from_shards(shard_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Merge worker shards under all_benchmark_v1 into one (N, 1280) matrix + index."""
    shard_dirs = sorted(
        [p for p in shard_root.iterdir() if p.is_dir() and p.name.startswith("w")],
        key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else p.name,
    )
    if not shard_dirs:
        raise FileNotFoundError(f"no worker shards under {shard_root}")

    vecs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for shard in shard_dirs:
        emb_path = shard / "embeddings.npy"
        idx_path = shard / "index.parquet"
        if not emb_path.is_file() or not idx_path.is_file():
            raise FileNotFoundError(f"incomplete shard: {shard}")
        X = np.load(emb_path)
        idx = pd.read_parquet(idx_path)
        if len(idx) != X.shape[0]:
            raise RuntimeError(f"{shard.name}: index {len(idx)} != emb {X.shape[0]}")
        vecs.append(np.asarray(X, dtype=np.float32))
        for r in idx.to_dict("records"):
            rows.append(r)
        print(f"[shards] {shard.name}: {X.shape}")

    X_all = np.concatenate(vecs, axis=0)
    frame = pd.DataFrame(rows).reset_index(drop=True)
    frame["embedding_row"] = np.arange(len(frame), dtype=np.int64)
    if X_all.ndim != 2 or X_all.shape[1] != 1280:
        raise RuntimeError(f"expected (*,1280) from shards, got {X_all.shape}")
    print(f"[shards] merged {X_all.shape} from {len(shard_dirs)} workers @ {shard_root}")
    return X_all, frame


def materialize_baseline_from_shards(
    shard_root: Path,
    *,
    dest_dir: Path | None = None,
    fallback_emb: Path | None = None,
    fallback_idx: Path | None = None,
) -> tuple[Path, Path, np.ndarray]:
    """Load merged (N,1280) from shards, or reuse already-merged npy if counts match."""
    emb_parent = shard_root.parent if shard_root.is_dir() else None
    merged_emb = (emb_parent / "deepseek_ocr2_mean_l2.npy") if emb_parent else None
    merged_idx = (
        (emb_parent.parent / "metadata" / "embedding_index.parquet") if emb_parent else None
    )

    # Fast path: already-merged npy with matching shard row count.
    if (
        shard_root.is_dir()
        and merged_emb is not None
        and merged_emb.is_file()
        and fallback_idx
        and Path(fallback_idx).is_file()
    ):
        n_shard = 0
        for p in shard_root.glob("w*/embeddings.npy"):
            n_shard += int(np.load(p, mmap_mode="r").shape[0])
        X_merged = np.load(merged_emb)
        if X_merged.shape == (n_shard, 1280):
            print(f"[shards] reuse merged {X_merged.shape} @ {merged_emb}")
            X = np.asarray(X_merged, dtype=np.float32)
            emb_path = merged_emb
            idx_path = Path(fallback_idx) if Path(fallback_idx).is_file() else (merged_idx or Path())
            if dest_dir is not None:
                emb_dir = ensure_dir(dest_dir / "embeddings")
                meta_dir = ensure_dir(dest_dir / "metadata")
                emb_path = emb_dir / "deepseek_ocr2_mean_l2.npy"
                idx_dst = meta_dir / "embedding_index.parquet"
                atomic_write_npy(emb_path, X)
                if idx_path.is_file():
                    shutil.copy2(idx_path, idx_dst)
                    idx_path = idx_dst
            return emb_path, idx_path, X

    if shard_root.is_dir() and any(shard_root.glob("w*/embeddings.npy")):
        X, idx = load_embeddings_from_shards(shard_root)
        emb_path = merged_emb or Path("deepseek_ocr2_mean_l2.npy")
        idx_path = merged_idx or Path("embedding_index.parquet")
        if dest_dir is not None:
            emb_dir = ensure_dir(dest_dir / "embeddings")
            meta_dir = ensure_dir(dest_dir / "metadata")
            emb_path = emb_dir / "deepseek_ocr2_mean_l2.npy"
            idx_path = meta_dir / "embedding_index.parquet"
            atomic_write_npy(emb_path, X)
            atomic_write_parquet(idx, idx_path)
        elif not emb_path.is_file() or np.load(emb_path, mmap_mode="r").shape != X.shape:
            atomic_write_npy(emb_path, X)
            ensure_dir(idx_path.parent)
            atomic_write_parquet(idx, idx_path)
        return emb_path, idx_path, X

    if fallback_emb is None or not fallback_emb.is_file():
        raise FileNotFoundError(f"no shards at {shard_root} and no fallback emb")
    X = np.load(fallback_emb)
    if dest_dir is not None:
        emb_dir = ensure_dir(dest_dir / "embeddings")
        meta_dir = ensure_dir(dest_dir / "metadata")
        emb_path = emb_dir / "deepseek_ocr2_mean_l2.npy"
        idx_path = meta_dir / "embedding_index.parquet"
        shutil.copy2(fallback_emb, emb_path)
        if fallback_idx and fallback_idx.is_file():
            shutil.copy2(fallback_idx, idx_path)
        return emb_path, idx_path, X
    return fallback_emb, fallback_idx or Path(), X


def prepare_variant_embeddings(
    *,
    variant: str,
    output_root: Path,
    shard_root: Path,
    baseline_emb: Path,
    baseline_idx: Path,
    shared_X: np.ndarray,
) -> tuple[Path, np.ndarray]:
    """Mirror shard-merged embeddings under variant/; never re-runs the model."""
    spec = VARIANT_SPECS[variant]
    out_dir = ensure_dir(output_root / variant)
    atomic_write_json(
        out_dir / "variant_spec.json",
        {
            "variant": variant,
            **spec,
            "embedding_source": str(shard_root),
            "note": "Embeddings merged from existing all_benchmark_v1 shards (no model re-extract).",
        },
    )
    emb_dir = ensure_dir(out_dir / "embeddings")
    emb_path = emb_dir / "deepseek_ocr2_mean_l2.npy"
    atomic_write_npy(emb_path, shared_X)
    print(f"[{variant}] shards embedding shape={shared_X.shape} -> {emb_path}")
    return emb_path, shared_X


def run_ablation(
    *,
    variants: tuple[str, ...] = ("A0", "A1", "A2", "A3"),
    output_root: Path,
    shard_root: Path,
    baseline_emb: Path,
    baseline_idx: Path,
    k_min: int = 2,
    k_max: int = 10,
    seed: int = 42,
) -> None:
    """Cluster-only: load 1280-D from shards, write metrics CSV.

    Existing shards are global-only mean pool (A0). A1 matches that recipe under
    current extract settings. A2/A3 need different features — skipped unless the
    same baseline is explicitly accepted via including them (then metrics == A0).
    """
    output_root = ensure_dir(output_root)
    print(f"[ablation] loading embeddings from shards: {shard_root}")
    _, shared_idx_path, shared_X = materialize_baseline_from_shards(
        shard_root,
        dest_dir=None,
        fallback_emb=baseline_emb,
        fallback_idx=baseline_idx,
    )
    if not shared_idx_path or not Path(shared_idx_path).is_file():
        shared_idx_path = baseline_idx

    # Shards are global-only; true A2/A3 recipes are unavailable without re-extract.
    runnable = []
    skipped = []
    for variant in variants:
        needs_new = variant in ("A2", "A3")
        if needs_new:
            skipped.append(variant)
            out_dir = ensure_dir(output_root / variant)
            atomic_write_json(
                out_dir / "variant_spec.json",
                {
                    "variant": variant,
                    **VARIANT_SPECS[variant],
                    "status": "skipped",
                    "reason": (
                        "all_benchmark_v1 shards are global-only (n_local=0); "
                        f"{variant} requires a different extract recipe. "
                        "No model re-extract per user request."
                    ),
                },
            )
            print(f"[{variant}] SKIP — needs re-extract (not in shards)")
            continue
        runnable.append(variant)

    for variant in runnable:
        out_dir = ensure_dir(output_root / variant)
        emb_path, X = prepare_variant_embeddings(
            variant=variant,
            output_root=output_root,
            shard_root=shard_root,
            baseline_emb=baseline_emb,
            baseline_idx=Path(shared_idx_path),
            shared_X=shared_X,
        )
        idx_dst = out_dir / "metadata" / "embedding_index.parquet"
        if Path(shared_idx_path).is_file():
            ensure_dir(idx_dst.parent)
            shutil.copy2(shared_idx_path, idx_dst)

        assert X.ndim == 2 and X.shape[1] == 1280, f"{variant}: expected (*,1280), got {X.shape}"
        csv_path = out_dir / "spherical_kmeans_metrics_k2_10.csv"
        frame = evaluate_spherical_kmeans_csv(
            X,
            out_csv=csv_path,
            k_min=k_min,
            k_max=k_max,
            seed=seed,
            variant=variant,
        )
        print(f"[{variant}] wrote {csv_path} rows={len(frame)} sha={file_sha256(emb_path)[:12]}")

    parts = []
    for variant in runnable:
        p = output_root / variant / "spherical_kmeans_metrics_k2_10.csv"
        if p.is_file():
            parts.append(pd.read_csv(p))
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(output_root / "A0_A3_spherical_kmeans_metrics_k2_10.csv", index=False)
        print(f"[ablation] combined → {output_root / 'A0_A3_spherical_kmeans_metrics_k2_10.csv'}")
    if skipped:
        print(f"[ablation] skipped (need re-extract): {', '.join(skipped)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A0–A3 spherical KMeans metrics from existing all_benchmark_v1 shards"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/cluster_exp"),
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/embeddings/shards/all_benchmark_v1"
        ),
    )
    parser.add_argument(
        "--baseline-emb",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/embeddings/deepseek_ocr2_mean_l2.npy"
        ),
    )
    parser.add_argument(
        "--baseline-index",
        type=Path,
        default=Path(
            "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/"
            "ALL-data/ALL_embedding/runs/all_benchmark_v1/metadata/embedding_index.parquet"
        ),
    )
    parser.add_argument("--variants", nargs="+", default=["A0", "A1", "A2", "A3"])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run_ablation(
        variants=tuple(args.variants),
        output_root=args.output_root,
        shard_root=args.shard_root,
        baseline_emb=args.baseline_emb,
        baseline_idx=args.baseline_index,
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
