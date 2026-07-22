"""PCA / StandardScaler / block-scale / joint feature construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .gt_features import FEATURE_NAMES


def fit_transform_features(
    X_img: np.ndarray,
    gt_raw: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    X_img: [N, 1280] L2-normalized embeddings (fit set only)
    gt_raw: [N, 5] raw GT (non-negative; max_ast_depth never null here)
    """
    seed = int(cfg.get("random_state", 42))
    var = float(cfg["image"]["pca_variance"])

    pca = PCA(n_components=var, svd_solver="full", random_state=seed)
    X_pca = pca.fit_transform(X_img).astype(np.float64)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(X_pca.shape[1])
    # block scale: mean L2 energy -> 1
    img_energy = np.linalg.norm(X_pca, axis=1)
    image_block_scale = float(img_energy.mean())
    if image_block_scale < 1e-12:
        raise RuntimeError("image_block_scale ~ 0")
    X_img_block = (X_pca / image_block_scale).astype(np.float64)

    scaler = StandardScaler()
    gt_scaled = scaler.fit_transform(gt_raw.astype(np.float64))
    txt_energy = np.linalg.norm(gt_scaled, axis=1)
    text_block_scale = float(txt_energy.mean())
    if text_block_scale < 1e-12:
        raise RuntimeError("text_block_scale ~ 0")
    X_txt_block = (gt_scaled / text_block_scale).astype(np.float64)

    joint = np.concatenate([X_img_block, X_txt_block], axis=1)

    return {
        "pca": pca,
        "scaler": scaler,
        "image_block_scale": image_block_scale,
        "text_block_scale": text_block_scale,
        "image_embedding_pca": X_pca.astype(np.float32),
        "image_features": X_img_block.astype(np.float32),
        "gt_features_scaled": gt_scaled.astype(np.float32),
        "text_features": X_txt_block.astype(np.float32),
        "joint_features": joint.astype(np.float32),
        "pca_n_components": n_keep,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
        "pca_cumulative_full": cum.astype(np.float64),
    }


def gt_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float64)


def save_models(bundle: dict[str, Any], models_dir: Path) -> None:
    models_dir = Path(models_dir)
    joblib.dump(bundle["pca"], models_dir / "image_pca.pkl")
    joblib.dump(bundle["scaler"], models_dir / "gt_scaler.pkl")
    meta = {
        "image_block_scale": bundle["image_block_scale"],
        "text_block_scale": bundle["text_block_scale"],
        "pca_n_components": bundle["pca_n_components"],
        "pca_explained_variance_ratio_sum": bundle["pca_explained_variance_ratio_sum"],
    }
    joblib.dump(meta, models_dir / "block_scales.pkl")
    (models_dir / "block_scales.txt").write_text(
        "\n".join(f"{k}={v}" for k, v in meta.items()) + "\n", encoding="utf-8"
    )
