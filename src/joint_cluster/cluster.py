"""E0/E1/E2 K-Means over K=2..8 and final model selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


EXPERIMENTS = {
    "image": "E0",
    "text": "E1",
    "joint": "E2",
}


def run_kmeans_sweep(
    features: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[int, KMeans]], dict[str, dict[int, np.ndarray]]]:
    seed = int(cfg.get("random_state", 42))
    k_min = int(cfg["clustering"]["k_min"])
    k_max = int(cfg["clustering"]["k_max"])
    n_init = int(cfg["clustering"]["n_init"])
    init = str(cfg["clustering"].get("init", "k-means++"))

    metric_rows: list[dict[str, Any]] = []
    models: dict[str, dict[int, KMeans]] = {name: {} for name in EXPERIMENTS}
    labels: dict[str, dict[int, np.ndarray]] = {name: {} for name in EXPERIMENTS}

    for feat_key, exp_id in EXPERIMENTS.items():
        X = features[feat_key]
        n = X.shape[0]
        for k in range(k_min, k_max + 1):
            km = KMeans(
                n_clusters=k,
                init=init,
                n_init=n_init,
                random_state=seed,
                algorithm="lloyd",
            )
            lab = km.fit_predict(X)
            sizes = np.bincount(lab, minlength=k)
            # silhouette on subsample if large
            sample_n = min(5000, n)
            if sample_n < n:
                rng = np.random.default_rng(seed)
                ii = rng.choice(n, size=sample_n, replace=False)
                sil = float(silhouette_score(X[ii], lab[ii], metric="euclidean"))
            else:
                sil = float(silhouette_score(X, lab, metric="euclidean"))
            row = {
                "experiment": exp_id,
                "feature_block": feat_key,
                "k": k,
                "n_samples": n,
                "inertia": float(km.inertia_),
                "silhouette": sil,
                "n_iter": int(km.n_iter_),
                "min_cluster_size": int(sizes.min()),
                "max_cluster_size": int(sizes.max()),
                "cluster_sizes": sizes.tolist(),
            }
            metric_rows.append(row)
            models[feat_key][k] = km
            labels[feat_key][k] = lab.astype(int)
            print(
                f"[kmeans] {exp_id} K={k} sil={sil:.4f} inertia={km.inertia_:.1f} "
                f"sizes={sizes.tolist()}"
            )

    return pd.DataFrame(metric_rows), models, labels


def select_final_k(metrics: pd.DataFrame, cfg: dict[str, Any], n: int) -> tuple[int, str]:
    forced = (cfg.get("clustering") or {}).get("final_k")
    if forced is not None:
        return int(forced), f"forced by config final_k={forced}"

    floor = max(
        int(cfg["clustering"].get("min_cluster_floor", 5)),
        int(round(float(cfg["clustering"].get("min_cluster_frac", 0.005)) * n)),
    )
    joint = metrics[metrics["feature_block"] == "joint"].copy()
    eligible = joint[joint["min_cluster_size"] >= floor]
    if eligible.empty:
        eligible = joint
        note = f"no K met min_cluster_size>={floor}; fallback to all K"
    else:
        note = f"among K with min_cluster_size>={floor}"
    best = eligible.loc[eligible["silhouette"].idxmax()]
    reason = (
        f"{note}; chose K={int(best['k'])} by max joint silhouette "
        f"({best['silhouette']:.4f}); also inspect inertia elbow / galleries / GT heatmap"
    )
    return int(best["k"]), reason


def attach_labels(
    page_df: pd.DataFrame,
    fit_mask: np.ndarray,
    labels: dict[str, dict[int, np.ndarray]],
    final_k: int,
    joint_model: KMeans,
    joint_X: np.ndarray,
) -> pd.DataFrame:
    out = page_df.copy()
    fit_mask = np.asarray(fit_mask, dtype=bool)
    for feat_key, prefix in (("image", "cluster_image"), ("text", "cluster_text"), ("joint", "cluster_joint")):
        for k, lab in labels[feat_key].items():
            col = f"{prefix}_k{k}"
            out[col] = pd.NA
            out.loc[fit_mask, col] = lab

    out["final_cluster"] = pd.NA
    out["distance_to_final_center"] = np.nan
    final_lab = labels["joint"][final_k]
    centers = joint_model.cluster_centers_
    dists = np.linalg.norm(joint_X - centers[final_lab], axis=1)
    out.loc[fit_mask, "final_cluster"] = final_lab
    out.loc[fit_mask, "distance_to_final_center"] = dists
    return out


def save_kmeans_models(
    models: dict[str, dict[int, KMeans]],
    models_dir: Path,
    final_k: int,
) -> None:
    models_dir = Path(models_dir)
    for k, km in models["joint"].items():
        joblib.dump(km, models_dir / f"kmeans_k{k}.pkl")
    joblib.dump(models["joint"][final_k], models_dir / "final_kmeans.pkl")
    for feat in ("image", "text"):
        for k, km in models[feat].items():
            joblib.dump(km, models_dir / f"kmeans_{feat}_k{k}.pkl")
