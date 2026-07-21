"""Three-channel PCA / UMAP / kNN + visual token-group analysis (v2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from ..config import get_seed, load_config
from ..feature_store import EmbeddingStore, atomic_replace_parquet
from ..utils import atomic_write_json, ensure_dir
from .paths import analysis_v2_dir, reports_v2_dir, transformers_dir

try:
    import umap as umap_lib
except Exception as e:  # noqa: BLE001
    umap_lib = None
    _UMAP_ERR = e
else:
    _UMAP_ERR = None


def _l2_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def fit_pca_full(X: np.ndarray, n_components: int, seed: int) -> dict[str, Any]:
    n_samples, n_dim = X.shape
    k = int(min(n_components, n_samples - 1, n_dim))
    if k < 2:
        raise RuntimeError(f"PCA needs >=2 components; got k={k}, shape={X.shape}")
    pca = PCA(n_components=k, random_state=seed, svd_solver="randomized" if n_dim > 100 else "auto")
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    def dims_for(thr: float) -> int:
        return int(min(int(np.searchsorted(cum, thr)) + 1, len(cum)))

    d95 = dims_for(0.95)
    return {
        "pca": pca,
        "Z": Z.astype(np.float32),
        "Z95": Z[:, :d95].astype(np.float32),
        "coords2": Z[:, :2].astype(np.float32),
        "evr": evr,
        "cum": cum,
        "dims_95": d95,
        "pc1_var": float(evr[0]) if len(evr) else None,
    }


def knn_exclude_self(X: np.ndarray, metric: str, k: int = 15) -> dict[str, np.ndarray]:
    n = X.shape[0]
    k_eff = min(k + 1, n)
    if n < 2:
        raise RuntimeError("kNN needs >=2 samples")
    m = "cosine" if metric == "cosine" else "euclidean"
    nn = NearestNeighbors(n_neighbors=k_eff, metric=m, algorithm="auto")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X)
    return {
        "nn_indices": idxs[:, 1:].astype(np.int32),
        "nn_distances": dists[:, 1:].astype(np.float32),
        "kth_neighbor_distance": dists[:, min(k, dists.shape[1] - 1)].astype(np.float32),
    }


def run_umap(X: np.ndarray, *, metric: str, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    if umap_lib is None:
        raise RuntimeError(f"umap-learn required: {_UMAP_ERR}")
    n = len(X)
    if n < 5:
        raise RuntimeError(f"UMAP needs >=5 samples, got {n}")
    reducer = umap_lib.UMAP(
        n_neighbors=min(n_neighbors, max(2, n - 1)),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X).astype(np.float32)


def select_contact_anchors(
    *,
    kth: np.ndarray,
    pca2: np.ndarray,
    umap2: np.ndarray | None,
    seed: int,
) -> dict[str, list[int]]:
    """Diverse anchors: ordinary / high-density / low-density / PCA extremes / UMAP islands."""
    rng = np.random.default_rng(seed)
    n = len(kth)
    order = np.argsort(kth)  # small kth = dense

    def uniq_take(cands: list[int], m: int) -> list[int]:
        out: list[int] = []
        for c in cands:
            c = int(c)
            if 0 <= c < n and c not in out:
                out.append(c)
            if len(out) >= m:
                break
        return out

    # ordinary: spread across density ranks
    ordinary = uniq_take(
        [order[int(i)] for i in np.linspace(n // 5, 4 * n // 5, 8)],
        4,
    )
    high_density = uniq_take(order[: max(20, n // 20)].tolist(), 4)
    low_density = uniq_take(order[-max(20, n // 20) :][::-1].tolist(), 4)

    pca_dirs = uniq_take(
        [
            int(np.argmax(pca2[:, 0])),
            int(np.argmin(pca2[:, 0])),
            int(np.argmax(pca2[:, 1])),
            int(np.argmin(pca2[:, 1])),
        ],
        4,
    )

    umap_isl: list[int] = []
    if umap2 is not None and len(umap2) == n:
        # 4 quadrant medoids in UMAP space
        mx, my = np.median(umap2[:, 0]), np.median(umap2[:, 1])
        quads = [
            (umap2[:, 0] >= mx) & (umap2[:, 1] >= my),
            (umap2[:, 0] < mx) & (umap2[:, 1] >= my),
            (umap2[:, 0] < mx) & (umap2[:, 1] < my),
            (umap2[:, 0] >= mx) & (umap2[:, 1] < my),
        ]
        for mask in quads:
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            center = umap2[idxs].mean(axis=0)
            d = np.linalg.norm(umap2[idxs] - center, axis=1)
            umap_isl.append(int(idxs[int(np.argmin(d))]))
        umap_isl = uniq_take(umap_isl, 4)

    # fill gaps randomly
    for bucket in (ordinary, high_density, low_density, pca_dirs, umap_isl):
        while len(bucket) < 4 and len(bucket) < n:
            c = int(rng.integers(0, n))
            if c not in bucket:
                bucket.append(c)

    return {
        "ordinary": ordinary[:4],
        "high_density": high_density[:4],
        "low_density": low_density[:4],
        "pca_extremes": pca_dirs[:4],
        "umap_islands": umap_isl[:4],
    }


def _analyze_channel(
    *,
    name: str,
    X_for_pca: np.ndarray,
    knn_metric: str,
    umap_metric: str,
    ids: list[str],
    seed: int,
    cfg: dict[str, Any],
    v2: Path,
    tf: Path,
    save_pca_name: str | None,
    X_for_knn: np.ndarray | None = None,
) -> dict[str, Any]:
    acfg = cfg.get("analysis") or {}
    n_comp = int(acfg.get("pca_n_components", 64))
    k = int(acfg.get("knn_k", 15))
    n_neighbors = int(acfg.get("umap_n_neighbors", 30))
    min_dist = float(acfg.get("umap_min_dist", 0.1))

    pca_info = fit_pca_full(X_for_pca, n_comp, seed)
    if save_pca_name:
        joblib.dump(pca_info["pca"], tf / f"{save_pca_name}_pca.joblib")

    umap_xy = run_umap(
        pca_info["Z95"],
        metric=umap_metric,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    knn_X = X_for_knn if X_for_knn is not None else pca_info["Z95"]
    knn = knn_exclude_self(knn_X, metric=knn_metric, k=k)

    coords = pd.DataFrame(
        {
            "image_id": ids,
            "pca1": pca_info["coords2"][:, 0],
            "pca2": pca_info["coords2"][:, 1],
            "umap1": umap_xy[:, 0],
            "umap2": umap_xy[:, 1],
            "kth_neighbor_distance": knn["kth_neighbor_distance"],
        }
    )
    atomic_replace_parquet(coords, v2 / f"{name}_coords_v2.parquet")
    np.save(v2 / f"{name}_pca_coords.npy", pca_info["coords2"])
    np.save(v2 / f"{name}_pca95.npy", pca_info["Z95"])
    np.save(v2 / f"{name}_umap_coords.npy", umap_xy)
    np.save(v2 / f"{name}_nn_indices.npy", knn["nn_indices"])
    np.save(v2 / f"{name}_nn_distances.npy", knn["nn_distances"])
    np.save(v2 / f"{name}_knn_kth.npy", knn["kth_neighbor_distance"])

    anchors = select_contact_anchors(
        kth=knn["kth_neighbor_distance"],
        pca2=pca_info["coords2"],
        umap2=umap_xy,
        seed=seed,
    )
    atomic_write_json(v2 / f"{name}_contact_anchors.json", anchors)

    return {
        "name": name,
        "n": int(len(ids)),
        "dims_95": pca_info["dims_95"],
        "pc1_var": pca_info["pc1_var"],
        "cum": pca_info["cum"].tolist(),
        "evr": pca_info["evr"].tolist(),
        "umap_n": int(len(umap_xy)),
        "anchors": anchors,
        "coords2": pca_info["coords2"],
        "umap_xy": umap_xy,
        "knn": knn,
        "ids": ids,
    }


def analyze_visual_token_groups(
    *,
    X_l2: np.ndarray,
    vis_idx: pd.DataFrame,
    global_pca2: np.ndarray,
    global_umap: np.ndarray,
    seed: int,
    cfg: dict[str, Any],
    reports: Path,
    v2: Path,
) -> pd.DataFrame:
    from .plotting import plot_scatter_colored, write_contact_sheets_for_channel

    acfg = cfg.get("analysis") or {}
    token = pd.to_numeric(vis_idx["token_count"], errors="coerce").fillna(-1).astype(int)
    ids = vis_idx["image_id"].astype(str).tolist()

    # Global colored plots
    fig_dir = ensure_dir(reports / "figures")
    plot_scatter_colored(
        global_pca2,
        token.to_numpy(),
        fig_dir / "visual_pca_by_token_count",
        "Visual PCA by token_count",
        "token_count",
        int(acfg.get("figure_dpi", 200)),
    )
    plot_scatter_colored(
        global_umap,
        token.to_numpy(),
        fig_dir / "visual_umap_by_token_count",
        "Visual UMAP by token_count",
        "token_count",
        int(acfg.get("figure_dpi", 200)),
    )

    groups = sorted(t for t in token.unique().tolist() if t > 0)
    rows = []
    centers = {}
    for g in groups:
        mask = (token == g).to_numpy()
        Xi = X_l2[mask]
        n = int(mask.sum())
        center = Xi.mean(axis=0)
        centers[g] = center
        # mean cosine distance within group
        if n >= 2:
            sims = Xi @ center
            # center not unit; normalize
            c_n = center / (np.linalg.norm(center) + 1e-12)
            sims = Xi @ c_n
            mean_cos_dist = float(np.mean(1.0 - sims))
        else:
            mean_cos_dist = 0.0
        pc_mean = global_pca2[mask].mean(axis=0) if n else np.zeros(2)
        rows.append(
            {
                "token_count": int(g),
                "n": n,
                "mean_cosine_distance_to_group_center": mean_cos_dist,
                "global_pc1_mean": float(pc_mean[0]),
                "global_pc2_mean": float(pc_mean[1]),
            }
        )

    # pairwise inter-group mean cosine distance between centers
    inter = {}
    for i, gi in enumerate(groups):
        for gj in groups[i + 1 :]:
            ci = centers[gi] / (np.linalg.norm(centers[gi]) + 1e-12)
            cj = centers[gj] / (np.linalg.norm(centers[gj]) + 1e-12)
            inter[f"{gi}_vs_{gj}"] = float(1.0 - float(ci @ cj))

    # NMI between token_count and a coarse visual clustering proxy (kmeans on PCA95 subsample)
    nmi = None
    try:
        from sklearn.cluster import MiniBatchKMeans

        kmeans = MiniBatchKMeans(n_clusters=min(8, max(2, len(groups))), random_state=seed, n_init=3)
        # use global pca2 as cheap proxy labels for NMI with token groups
        labels = kmeans.fit_predict(global_pca2)
        nmi = float(normalized_mutual_info_score(token.to_numpy(), labels))
    except Exception as e:  # noqa: BLE001
        nmi = None
        inter["nmi_error"] = f"{type(e).__name__}: {e}"

    metrics = pd.DataFrame(rows)
    metrics_path = v2 / "visual_token_group_metrics.parquet"
    atomic_replace_parquet(metrics, metrics_path)
    atomic_write_json(
        v2 / "visual_token_group_summary.json",
        {
            "group_counts": {str(int(r.token_count)): int(r.n) for r in metrics.itertuples()},
            "sum_n": int(metrics["n"].sum()),
            "inter_group_mean_cosine_distance": inter,
            "token_vs_kmeans_pca2_nmi": nmi,
        },
    )

    # Per-group independent PCA/UMAP/kNN
    vg_root = ensure_dir(reports / "visual_groups")
    for g in groups:
        mask = (token == g).to_numpy()
        if mask.sum() < 10:
            continue
        gdir = ensure_dir(vg_root / f"token_{int(g)}")
        Xi = X_l2[mask]
        id_g = [ids[i] for i, m in enumerate(mask) if m]
        pca_info = fit_pca_full(Xi, int(acfg.get("pca_n_components", 64)), seed)
        umap_xy = run_umap(
            pca_info["Z95"],
            metric="cosine",
            seed=seed,
            n_neighbors=int(acfg.get("umap_n_neighbors", 30)),
            min_dist=float(acfg.get("umap_min_dist", 0.1)),
        )
        knn = knn_exclude_self(Xi, metric="cosine", k=int(acfg.get("knn_k", 15)))
        np.save(gdir / "pca95.npy", pca_info["Z95"])
        np.save(gdir / "umap.npy", umap_xy)
        np.save(gdir / "nn_indices.npy", knn["nn_indices"])
        np.save(gdir / "nn_distances.npy", knn["nn_distances"])
        np.save(gdir / "knn_kth.npy", knn["kth_neighbor_distance"])
        coords = pd.DataFrame(
            {
                "image_id": id_g,
                "pca1": pca_info["coords2"][:, 0],
                "pca2": pca_info["coords2"][:, 1],
                "umap1": umap_xy[:, 0],
                "umap2": umap_xy[:, 1],
                "kth_neighbor_distance": knn["kth_neighbor_distance"],
            }
        )
        coords.to_parquet(gdir / "coords.parquet", index=False)
        anchors = select_contact_anchors(
            kth=knn["kth_neighbor_distance"],
            pca2=pca_info["coords2"],
            umap2=umap_xy,
            seed=seed,
        )
        atomic_write_json(gdir / "contact_anchors.json", anchors)
        # figures
        from .plotting import plot_pca_panel, plot_knn_dist, plot_umap_hex

        plot_pca_panel(pca_info["coords2"], pca_info["cum"].tolist(), gdir / "pca", f"token_{g}", int(acfg.get("figure_dpi", 200)))
        plot_umap_hex(umap_xy, gdir / "umap", f"token_{g}", int(acfg.get("figure_dpi", 200)))
        plot_knn_dist(knn["kth_neighbor_distance"], gdir / "knn15", f"token_{g}", int(acfg.get("figure_dpi", 200)))
        write_contact_sheets_for_channel(
            channel=f"token_{g}",
            ids=id_g,
            knn=knn,
            anchors=anchors,
            out_dir=gdir / "contact_sheets",
            cfg=cfg,
        )

    return metrics


def run_analyze_v2(cfg: dict[str, Any]) -> dict[str, Any]:
    """Visual-channel PCA / UMAP / kNN only (no layout / recognition)."""
    seed = get_seed(cfg)
    out_dir = Path(cfg["paths"]["outputs_dir"])
    v2 = analysis_v2_dir(cfg)
    tf = transformers_dir(cfg)
    reports = reports_v2_dir(cfg)
    acfg = cfg.get("analysis") or {}

    results: dict[str, Any] = {}

    dim = 896
    meta_p = out_dir / "visual_embeddings.f32.meta.json"
    if meta_p.exists():
        dim = int(json.loads(meta_p.read_text()).get("dim", 896))
    store = EmbeddingStore(
        mmap_path=out_dir / "visual_embeddings.f32.mmap",
        index_path=out_dir / "visual_index.parquet",
        dim=dim,
    )
    Xv, vis_idx = store.load_matrix()

    # Prefer embedding_usable filter when quality table exists
    q_path = out_dir / "embedding_quality.parquet"
    if q_path.exists():
        q = pd.read_parquet(q_path)
        usable = set(q.loc[q["embedding_usable"].astype(bool), "image_id"].astype(str))
        mask = vis_idx["image_id"].astype(str).isin(usable)
        if int(mask.sum()) >= 5:
            Xv = Xv[mask.to_numpy()]
            vis_idx = vis_idx.loc[mask].reset_index(drop=True)
            print(f"[analyze_v2] filtered to embedding_usable n={len(vis_idx)}")

    X_l2 = _l2_rows(Xv)
    vis_ids = vis_idx["image_id"].astype(str).tolist()
    vis_res = _analyze_channel(
        name="visual",
        X_for_pca=X_l2,
        X_for_knn=X_l2,
        knn_metric="cosine",
        umap_metric="cosine",
        ids=vis_ids,
        seed=seed,
        cfg=cfg,
        v2=v2,
        tf=tf,
        save_pca_name="visual",
    )
    results["visual"] = {k: vis_res[k] for k in ("n", "dims_95", "pc1_var", "umap_n")}

    analyze_visual_token_groups(
        X_l2=X_l2,
        vis_idx=vis_idx,
        global_pca2=vis_res["coords2"],
        global_umap=vis_res["umap_xy"],
        seed=seed,
        cfg=cfg,
        reports=reports,
        v2=v2,
    )

    metrics = {
        "channels": {
            "visual": {
                "n": vis_res["n"],
                "dims_95": vis_res["dims_95"],
                "pc1_variance": vis_res["pc1_var"],
                "pca": {
                    "cumulative_explained_variance": vis_res["cum"],
                    "explained_variance_ratio": vis_res["evr"],
                },
                "umap_n": vis_res["umap_n"],
            }
        },
        "mode": "visual_only",
    }
    atomic_write_json(v2 / "feature_metrics_v2.json", metrics)
    atomic_write_json(reports / "feature_metrics_v2.json", metrics)

    from .plotting import write_contact_sheets_for_channel, plot_channel_overview

    write_contact_sheets_for_channel(
        channel="visual",
        ids=vis_res["ids"],
        knn=vis_res["knn"],
        anchors=vis_res["anchors"],
        out_dir=reports / "contact_sheets",
        cfg=cfg,
    )
    plot_channel_overview(
        name="visual",
        coords2=vis_res["coords2"],
        umap_xy=vis_res["umap_xy"],
        kth=vis_res["knn"]["kth_neighbor_distance"],
        cum=vis_res["cum"],
        out_dir=reports / "figures",
        dpi=int(acfg.get("figure_dpi", 200)),
    )

    print(f"[analyze_v2] visual={results['visual']}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    run_analyze_v2(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
