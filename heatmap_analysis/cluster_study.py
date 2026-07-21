"""64×64 dual-feature clustering study (abs_smooth + rel_smooth × KMeans/GMM/HDBSCAN)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from heatmap_analysis.clustering import _fit_kmeans_at_k
from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.gpu import is_gpu_available
from heatmap_analysis.gpu_clustering import pca_reduce
from heatmap_analysis.heatmap import flatten_heatmap
from heatmap_analysis.utils import ensure_dir, save_json
from heatmap_analysis.visualization import plot_heatmap, plot_overlay

logger = logging.getLogger("heatmap_analysis.cluster_study")

@dataclass(frozen=True)
class FeatureTrack:
    name: str
    npz_key: str
    label: str


TRACKS = (
    FeatureTrack("abs_smooth", "d_abs_smooth", "墨迹密度 (d_abs_smooth)"),
    FeatureTrack("rel_smooth", "d_rel_smooth", "空间分布 (d_rel_smooth)"),
)


def load_dataset(
    cfg: AnalysisConfig,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Load image ids, abs/rel smoothed grids, and path map from cache."""
    metrics_path = cfg.output.output_dir / "tables" / "per_image_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        ids = metrics["image_id"].astype(str).tolist()
        path_map = dict(zip(metrics["image_id"].astype(str), metrics["image_path"].astype(str)))
    else:
        cache_dir = cfg.cache_dir / "per_image"
        ids = sorted(p.stem for p in cache_dir.glob("*.npz"))
        from heatmap_analysis.io import build_metadata_index, records_from_metadata

        recs = records_from_metadata(cfg, build_metadata_index(cfg))
        path_map = {r.image_id: str(r.image_path) for r in recs}

    abs_map: dict[str, np.ndarray] = {}
    rel_map: dict[str, np.ndarray] = {}
    valid_ids: list[str] = []
    for iid in ids:
        npz = cfg.cache_dir / "per_image" / f"{iid}.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        if "d_abs_smooth" not in d or "d_rel_smooth" not in d:
            logger.warning("Skip %s: missing smoothed grids", iid)
            continue
        abs_map[iid] = d["d_abs_smooth"]
        rel_map[iid] = d["d_rel_smooth"]
        valid_ids.append(iid)
    return valid_ids, abs_map, rel_map, path_map


def _feature_matrix(grid_map: dict[str, np.ndarray], ids: list[str]) -> np.ndarray:
    return np.stack([flatten_heatmap(grid_map[i]) for i in ids])


def fit_pca(cfg: AnalysisConfig, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return (X_scaled for KMeans/GMM, X_pca unscaled for HDBSCAN, cumvar, n_comp).

    Unscaled PCA keeps variance ordering so the first 2 components remain informative.
    Scaling after PCA equalizes axes and makes a second PCA→2D nearly isotropic (~2/d variance).
    """
    use_gpu = cfg.gpu.enabled and cfg.gpu.clustering and is_gpu_available()
    X_pca, cumvar, n_comp = pca_reduce(
        X,
        cfg.clustering.pca_variance,
        cfg.clustering.pca_n_components,
        cfg.clustering.random_seed,
        use_gpu=use_gpu,
    )
    return StandardScaler().fit_transform(X_pca), X_pca, cumvar, n_comp


def _distances_to_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - centroids[None, :, :]
    return np.sum(diff**2, axis=2)


def _closest_and_boundary(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    cluster_id: int,
    n_close: int,
    n_bound: int,
    id_list: list[str],
) -> tuple[list[str], list[str]]:
    mask = labels == cluster_id
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return [], []
    sub = X[idxs]
    dists = _distances_to_centroids(sub, centroids)
    d_assigned = dists[np.arange(len(idxs)), labels[idxs]]
    closest = [id_list[idxs[i]] for i in np.argsort(d_assigned)[:n_close]]
    if centroids.shape[0] < 2:
        return closest, closest[:n_bound]
    sorted_d = np.sort(dists, axis=1)
    boundary_score = sorted_d[:, 1] / np.maximum(sorted_d[:, 0], 1e-12)
    boundary = [id_list[idxs[i]] for i in np.argsort(-boundary_score)[:n_bound]]
    return closest, boundary


def _export_samples(
    sample_ids: list[str],
    out_dir: Path,
    path_map: dict[str, str],
    grid_map: dict[str, np.ndarray],
    cfg: AnalysisConfig,
    tag: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, iid in enumerate(sample_ids):
        src = Path(path_map.get(iid, ""))
        row: dict = {"rank": rank, "image_id": iid, "tag": tag}
        if src.exists():
            dst = out_dir / f"{rank:02d}_{iid}.jpg"
            shutil.copy2(src, dst)
            row["copy"] = str(dst)
            plot_overlay(src, grid_map[iid], out_dir / f"{rank:02d}_{iid}_overlay.png", f"{tag}: {iid}", cfg.report.colormap)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)


def _plot_cluster_centers(
    root: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    center_name: str = "mean_center.png",
) -> None:
    for cdir in sorted(root.glob("cluster_*")) + sorted(root.glob("component_*")):
        stats = cdir / "stats.npz"
        if not stats.exists():
            continue
        d = np.load(stats)
        n = int(d["n_samples"])
        plot_heatmap(d["mean_feature"], cdir / center_name, f"{cdir.name} center", cmap=cmap, vmin=vmin, vmax=vmax, n_samples=n)


def run_kmeans(
    cfg: AnalysisConfig,
    track: FeatureTrack,
    track_root: Path,
    ids: list[str],
    grid_map: dict[str, np.ndarray],
    X_pca: np.ndarray,
    path_map: dict[str, str],
    cumvar: float,
) -> pd.DataFrame:
    out = ensure_dir(track_root / "kmeans")
    use_gpu = cfg.gpu.enabled and cfg.gpu.clustering and is_gpu_available()
    k_values = cfg.clustering.k_values
    n_close = cfg.clustering.closest_samples
    n_bound = cfg.clustering.boundary_samples
    cmap = cfg.report.colormap

    metric_rows: list[dict] = []
    all_centers: list[np.ndarray] = []

    for k in k_values:
        if k < 2 or k >= len(ids):
            continue
        labels, centroids = _fit_kmeans_at_k(X_pca, k, cfg.clustering.random_seed, use_gpu)
        sil = float(silhouette_score(X_pca, labels)) if len(set(labels)) > 1 else float("nan")
        metric_rows.append(
            {
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": float(calinski_harabasz_score(X_pca, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_pca, labels)),
            }
        )

        kdir = ensure_dir(out / f"k_{k:02d}")
        pd.DataFrame({"image_id": ids, "cluster": labels}).to_csv(kdir / "cluster_labels.csv", index=False)

        for cid in range(k):
            cids = [ids[i] for i, lb in enumerate(labels) if lb == cid]
            if not cids:
                continue
            mean_feat = np.mean(np.stack([grid_map[i] for i in cids]), axis=0)
            all_centers.append(mean_feat)
            cdir = ensure_dir(kdir / f"cluster_{cid:02d}")
            np.savez_compressed(
                cdir / "stats.npz",
                mean_feature=mean_feat,
                n_samples=len(cids),
                centroid_pca=centroids[cid],
            )
            (cdir / "n_samples.txt").write_text(str(len(cids)), encoding="utf-8")
            closest, boundary = _closest_and_boundary(X_pca, labels, centroids, cid, n_close, n_bound, ids)
            _export_samples(closest, cdir / "closest_to_center", path_map, grid_map, cfg, "closest")
            _export_samples(boundary, cdir / "boundary_nearby", path_map, grid_map, cfg, "boundary")

        save_json(kdir / "summary.json", {"k": k, "n_samples": len(ids), "pca_variance": cumvar})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "k_selection_metrics.csv", index=False)

    vmax = float(np.max(all_centers)) if all_centers else 1.0
    save_json(out / "unified_colormap.json", {"vmin": 0.0, "vmax": vmax, "cmap": cmap})
    for kdir in sorted(out.glob("k_*")):
        _plot_cluster_centers(kdir, cmap, 0.0, vmax)

    save_json(out / "method.json", {"method": "KMeans", "feature": track.name, "k_values": k_values})
    logger.info("[%s] K-means done (%d k values)", track.name, len(metric_rows))
    return metrics_df


def run_gmm(
    cfg: AnalysisConfig,
    track: FeatureTrack,
    track_root: Path,
    ids: list[str],
    grid_map: dict[str, np.ndarray],
    X_pca: np.ndarray,
    path_map: dict[str, str],
    cumvar: float,
) -> pd.DataFrame:
    out = ensure_dir(track_root / "gmm")
    n_close = cfg.clustering.closest_samples
    cmap = cfg.report.colormap
    comp_range = cfg.clustering.gmm_components

    sel_rows: list[dict] = []
    models: dict[int, GaussianMixture] = {}
    for n in comp_range:
        if n < 2 or n >= len(ids):
            continue
        gmm = GaussianMixture(n_components=n, random_state=cfg.clustering.random_seed, n_init=5, max_iter=300)
        gmm.fit(X_pca)
        models[n] = gmm
        sel_rows.append(
            {"n_components": n, "aic": float(gmm.aic(X_pca)), "bic": float(gmm.bic(X_pca)), "log_likelihood": float(gmm.score(X_pca))}
        )
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out / "model_selection_aic_bic.csv", index=False)

    best_n = int(sel.loc[sel["bic"].idxmin(), "n_components"]) if not sel.empty else comp_range[0]
    run_ns = sorted(set(comp_range) | {best_n})
    all_centers: list[np.ndarray] = []

    for n in run_ns:
        gmm = models.get(n) or GaussianMixture(n_components=n, random_state=cfg.clustering.random_seed, n_init=5).fit(X_pca)
        probs = gmm.predict_proba(X_pca)
        labels = np.argmax(probs, axis=1)
        ndir = ensure_dir(out / f"n_{n:02d}")

        prob_df = pd.DataFrame(probs, columns=[f"prob_c{i}" for i in range(n)])
        prob_df.insert(0, "image_id", ids)
        prob_df["assigned_cluster"] = labels
        prob_df["max_prob"] = probs.max(axis=1)
        prob_df["entropy"] = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        prob_df.to_csv(ndir / "posterior_probabilities.csv", index=False)

        transitional = prob_df.nsmallest(30, "max_prob")["image_id"].tolist()
        _export_samples(transitional, ndir / "transitional_samples", path_map, grid_map, cfg, "transitional")

        for cid in range(n):
            cids = [ids[i] for i, lb in enumerate(labels) if lb == cid]
            if not cids:
                continue
            mean_feat = np.mean(np.stack([grid_map[i] for i in cids]), axis=0)
            all_centers.append(mean_feat)
            cdir = ensure_dir(ndir / f"component_{cid:02d}")
            np.savez_compressed(cdir / "stats.npz", mean_feature=mean_feat, n_samples=len(cids))
            (cdir / "n_samples.txt").write_text(str(len(cids)), encoding="utf-8")
            order = np.argsort(-probs[:, cid])
            repr_ids = [ids[i] for i in order[:n_close]]
            _export_samples(repr_ids, cdir / "representative", path_map, grid_map, cfg, "representative")

        save_json(ndir / "summary.json", {"n_components": n, "bic_best": n == best_n, "pca_variance": cumvar})

    vmax = float(np.max(all_centers)) if all_centers else 1.0
    for ndir in sorted(out.glob("n_*")):
        _plot_cluster_centers(ndir, cmap, 0.0, vmax)

    save_json(out / "method.json", {"method": "GMM", "feature": track.name, "bic_best_n": best_n})
    logger.info("[%s] GMM done (best BIC n=%d)", track.name, best_n)
    return sel


def _project_hdbscan_2d(
    X_pca_unscaled: np.ndarray,
    cumvar: float,
    n_comp: int,
) -> tuple[np.ndarray, float]:
    """Take the leading 2 PCA axes (variance-ordered), not a second PCA on scaled coords."""
    d = X_pca_unscaled.shape[1]
    n_take = min(2, d)
    X_2d = np.ascontiguousarray(X_pca_unscaled[:, :n_take])
    # Approximate share of original retained PCA variance in the first 2 components.
    # pca_reduce keeps components ordered by eigenvalue; with equalized scaling this would be ~2/d.
    # On unscaled PCA scores, leading axes dominate — estimate via column variance share.
    col_var = np.var(X_pca_unscaled, axis=0)
    total = float(np.sum(col_var))
    var_2d = float(np.sum(col_var[:n_take]) / total) if total > 0 else 0.0
    var_2d_of_original = var_2d * float(cumvar) if n_comp > 0 else var_2d
    return X_2d, var_2d_of_original


def run_hdbscan(
    cfg: AnalysisConfig,
    track: FeatureTrack,
    track_root: Path,
    ids: list[str],
    grid_map: dict[str, np.ndarray],
    X_pca_unscaled: np.ndarray,
    path_map: dict[str, str],
    cumvar: float,
    n_comp: int,
) -> dict:
    try:
        import hdbscan
    except ImportError as e:
        logger.error("hdbscan not installed: %s", e)
        return {}

    out = ensure_dir(track_root / "hdbscan")
    n_close = cfg.clustering.closest_samples
    cmap = cfg.report.colormap

    # Use top-2 unscaled PCA components (keeps real variance); avoid StandardScaler→PCA2D.
    X_2d, var_2d = _project_hdbscan_2d(X_pca_unscaled, cumvar, n_comp)
    logger.info(
        "[%s] HDBSCAN input: %d samples × %d-D PCA → top-2 (≈%.1f%% of original feature var)",
        track.name,
        len(ids),
        X_pca_unscaled.shape[1],
        100.0 * var_2d,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(30, len(ids) // 100),
        min_samples=10,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X_2d)

    n_clusters = len({u for u in labels if u >= 0})
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / max(len(ids), 1)

    label_df = pd.DataFrame(
        {
            "image_id": ids,
            "cluster": labels,
            "emb_x": X_2d[:, 0],
            "emb_y": X_2d[:, 1] if X_2d.shape[1] > 1 else 0.0,
        }
    )
    if clusterer.probabilities_ is not None:
        label_df["membership_strength"] = clusterer.probabilities_
    label_df.to_csv(out / "cluster_labels.csv", index=False)
    np.save(out / "embedding_2d.npy", X_2d.astype(np.float32))

    noise_ids = [ids[i] for i, lb in enumerate(labels) if lb == -1]
    _export_samples(noise_ids[:n_close], out / "noise_samples", path_map, grid_map, cfg, "noise")

    all_centers: list[np.ndarray] = []
    for cid in sorted({u for u in labels if u >= 0}):
        cids = [ids[i] for i, lb in enumerate(labels) if lb == cid]
        mean_feat = np.mean(np.stack([grid_map[i] for i in cids]), axis=0)
        all_centers.append(mean_feat)
        cdir = ensure_dir(out / f"cluster_{cid:02d}")
        np.savez_compressed(cdir / "stats.npz", mean_feature=mean_feat, n_samples=len(cids))
        (cdir / "n_samples.txt").write_text(str(len(cids)), encoding="utf-8")
        sub = X_2d[[i for i, lb in enumerate(labels) if lb == cid]]
        dists = np.linalg.norm(sub - sub.mean(axis=0), axis=1)
        repr_ids = [cids[i] for i in np.argsort(dists)[:n_close]]
        _export_samples(repr_ids, cdir / "representative", path_map, grid_map, cfg, "representative")

    vmax = float(np.max(all_centers)) if all_centers else 1.0
    _plot_cluster_centers(out, cmap, 0.0, vmax)

    summary = {
        "method": "HDBSCAN",
        "feature": track.name,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "n_samples": len(ids),
        "pca_variance": cumvar,
        "hdbscan_dims": int(X_2d.shape[1]),
        "embedding_2d_variance": var_2d,
        "prior_pca_dims": int(X_pca_unscaled.shape[1]),
        "embedding": "top2_unscaled_pca",
    }
    save_json(out / "summary.json", summary)
    logger.info("[%s] HDBSCAN: %d clusters, %d noise (%.1f%%)", track.name, n_clusters, n_noise, 100 * noise_ratio)
    return summary


def evaluate_track(
    track: FeatureTrack,
    track_root: Path,
    kmeans_metrics: pd.DataFrame,
    gmm_sel: pd.DataFrame,
    hdb_summary: dict,
) -> dict:
    """Compare algorithms to assess discrete vs continuous structure."""
    eval_out: dict = {"feature": track.name, "label": track.label}

    if not kmeans_metrics.empty:
        best_k_row = kmeans_metrics.loc[kmeans_metrics["silhouette"].idxmax()]
        eval_out["kmeans"] = {
            "best_k_by_silhouette": int(best_k_row["k"]),
            "best_silhouette": float(best_k_row["silhouette"]),
            "silhouette_at_k": kmeans_metrics.set_index("k")["silhouette"].to_dict(),
            "low_silhouette_warning": float(best_k_row["silhouette"]) < 0.05,
        }

    if not gmm_sel.empty:
        best_n = int(gmm_sel.loc[gmm_sel["bic"].idxmin(), "n_components"])
        eval_out["gmm"] = {
            "bic_best_n": best_n,
            "aic_bic": gmm_sel.to_dict(orient="records"),
        }

    if hdb_summary:
        eval_out["hdbscan"] = hdb_summary
        noise_ratio = hdb_summary.get("noise_ratio", 0.0)
        km_low = eval_out.get("kmeans", {}).get("low_silhouette_warning", False)
        eval_out["interpretation"] = (
            "数据更可能呈连续变化（HDBSCAN 大量噪声 + K-means 轮廓系数低）"
            if noise_ratio > 0.3 and km_low
            else "可能存在若干稳定类别，需结合类中心热力图判断"
            if hdb_summary.get("n_clusters", 0) >= 2 and noise_ratio < 0.3
            else "结构不清晰，建议查看 GMM 过渡样本与各类中心"
        )

    save_json(track_root / "evaluation.json", eval_out)
    return eval_out


def write_study_report(cfg: AnalysisConfig, root: Path, evaluations: list[dict]) -> None:
    gs = cfg.heatmap.grid_size
    lines = [
        "# 64×64 双特征轨聚类研究",
        "",
        "## 处理流程",
        "1. 读取灰度图 → [可选]模板对齐 → 全图提取笔迹 mask",
        f"2. 生成 {gs}×{gs} `d_abs` → `d_abs_smooth`（墨迹密度）/ `d_rel_smooth`（空间分布）",
        f"3. 分别展开为 {gs*gs} 维 → PCA 保留 {cfg.clustering.pca_variance:.0%} 方差",
        "4. K-means / GMM 用标准化后的 PCA 坐标；HDBSCAN 用未标准化的前 2 个 PCA 分量",
        "5. 评价聚类结果 → 输出类中心热力图与代表原图",
        "",
        "## 特征轨",
        "| 目录 | 特征 | 含义 |",
        "|------|------|------|",
        f"| abs_smooth/ | d_abs_smooth | 保留墨迹密度 |",
        f"| rel_smooth/ | d_rel_smooth | 只保留空间分布 |",
        "",
        "## K-means",
        f"- k = {cfg.clustering.k_values}",
        "- 强制划分所有样本；需结合轮廓系数判断，不能只看类别数",
        "",
        "## GMM",
        "- 输出 AIC/BIC、后验概率、过渡样本",
        "",
        "## HDBSCAN",
        "- 使用方差有序的前 2 个 PCA 分量（避免对已标准化 PCA 再降维导致信息丢失）",
        "- 自动发现类别数；大量噪声提示连续分布",
        "",
        "## 评价摘要",
    ]
    for ev in evaluations:
        lines.append(f"### {ev.get('label', ev.get('feature'))}")
        interp = ev.get("interpretation", "")
        if interp:
            lines.append(f"- {interp}")
        km = ev.get("kmeans", {})
        if km:
            lines.append(f"- K-means 最佳轮廓系数: k={km.get('best_k_by_silhouette')} sil={km.get('best_silhouette', 0):.4f}")
        hdb = ev.get("hdbscan", {})
        if hdb:
            lines.append(
                f"- HDBSCAN: {hdb.get('n_clusters', 0)} 类, "
                f"{hdb.get('n_noise', 0)} 噪声 ({100*hdb.get('noise_ratio', 0):.1f}%)"
            )
        lines.append("")

    (root / "cluster_study_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_cluster_study(
    cfg: AnalysisConfig,
    *,
    skip_extract: bool = False,
    cfg_path: Path | None = None,
    limit: int | None = None,
    methods: str | None = None,
) -> None:
    """Run full dual-track clustering study."""
    root = ensure_dir(cfg.output.output_dir / "clustering")
    cache_count = len(list((cfg.cache_dir / "per_image").glob("*.npz")))

    if not skip_extract or cache_count < 10:
        from heatmap_analysis.pipeline import extract_all

        logger.info("Extracting %d×%d heatmaps...", cfg.heatmap.grid_size, cfg.heatmap.grid_size)
        extract_all(cfg, limit=limit, cfg_path=cfg_path)

    ids, abs_map, rel_map, path_map = load_dataset(cfg)
    if len(ids) < cfg.clustering.min_samples_for_clustering:
        raise ValueError(f"Too few samples for clustering: {len(ids)}")

    logger.info("Loaded %d samples, grid %d×%d", len(ids), cfg.heatmap.grid_size, cfg.heatmap.grid_size)

    X_abs = _feature_matrix(abs_map, ids)
    X_rel = _feature_matrix(rel_map, ids)

    run_set = (methods or "KGM").upper()
    evaluations: list[dict] = []

    for track, grid_map, X_raw in [
        (TRACKS[0], abs_map, X_abs),
        (TRACKS[1], rel_map, X_rel),
    ]:
        track_root = ensure_dir(root / track.name)
        X_pca, X_pca_unscaled, cumvar, n_comp = fit_pca(cfg, X_raw)
        save_json(track_root / "pca.json", {"cumvar": cumvar, "n_components": n_comp, "n_samples": len(ids)})

        km_df = pd.DataFrame()
        gmm_df = pd.DataFrame()
        hdb_sum: dict = {}

        if "K" in run_set:
            km_df = run_kmeans(cfg, track, track_root, ids, grid_map, X_pca, path_map, cumvar)
        if "G" in run_set:
            gmm_df = run_gmm(cfg, track, track_root, ids, grid_map, X_pca, path_map, cumvar)
        if "M" in run_set:
            hdb_sum = run_hdbscan(
                cfg, track, track_root, ids, grid_map, X_pca_unscaled, path_map, cumvar, n_comp
            )

        evaluations.append(evaluate_track(track, track_root, km_df, gmm_df, hdb_sum))

    write_study_report(cfg, root, evaluations)
    logger.info("Cluster study finished: %s", root)
