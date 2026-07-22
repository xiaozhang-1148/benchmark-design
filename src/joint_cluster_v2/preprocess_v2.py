"""V2 preprocess: scale continuous only; keep binaries 0/1; block scale; F1/F2 weights."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def build_text_block(
    continuous: np.ndarray,
    binary: np.ndarray,
    *,
    fit_scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler, float]:
    """
    continuous: [N, C] raw continuous (depth, optional nodes)
    binary: [N, B] 0/1 (already filtered)
    Returns text_block scaled so mean L2 ≈ 1 (before fusion weight).
    """
    if continuous.ndim == 1:
        continuous = continuous.reshape(-1, 1)
    if binary.size == 0:
        binary = np.zeros((continuous.shape[0], 0), dtype=np.float64)

    scaler = fit_scaler or StandardScaler()
    if fit_scaler is None:
        cont_s = scaler.fit_transform(continuous.astype(np.float64))
    else:
        cont_s = scaler.transform(continuous.astype(np.float64))

    raw = np.concatenate([cont_s, binary.astype(np.float64)], axis=1)
    energy = np.linalg.norm(raw, axis=1)
    block_scale = float(energy.mean())
    if block_scale < 1e-12:
        raise RuntimeError("text_block_scale ~ 0")
    text = (raw / block_scale).astype(np.float64)
    return text, scaler, block_scale


def joint_from_blocks(
    image_block: np.ndarray,
    text_block: np.ndarray,
    text_weight: float,
) -> np.ndarray:
    return np.concatenate(
        [image_block.astype(np.float64), (text_weight * text_block).astype(np.float64)],
        axis=1,
    ).astype(np.float32)


def load_v1_image_block(v1_root) -> tuple[np.ndarray, dict[str, Any]]:
    from pathlib import Path

    v1 = Path(v1_root)
    X_pca = np.load(v1 / "data" / "image_embedding_pca.npy").astype(np.float64)
    scales = joblib.load(v1 / "models" / "block_scales.pkl")
    image_block_scale = float(scales["image_block_scale"])
    X_img = (X_pca / image_block_scale).astype(np.float32)
    meta = {
        "pca_n_components": int(X_pca.shape[1]),
        "image_block_scale": image_block_scale,
        "pca_variance": float(scales.get("pca_explained_variance_ratio_sum", 0.9505)),
    }
    return X_img, meta
