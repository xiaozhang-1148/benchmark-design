"""Shared utilities for heatmap analysis."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from typing import Any

import numpy as np


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with console and optional file handler."""
    logger = logging.getLogger("heatmap_analysis")
    logger.handlers.clear()
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Return MD5 hex digest of file contents."""
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def normalized_grid_coords(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-center coordinates in [0,1] for a grid_size x grid_size grid."""
    edges = np.linspace(0.0, 1.0, grid_size + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    return xx, yy


def gaussian_smooth_grid(grid: np.ndarray, sigma: float, *, use_gpu: bool = False, xp: Any | None = None) -> np.ndarray:
    """Apply Gaussian smoothing on a normalized grid (sigma in cell units)."""
    if sigma <= 0:
        return grid.copy()
    if use_gpu:
        from heatmap_analysis.gpu import get_xp, to_numpy

        xp = xp or get_xp(True)
        if xp is not np:
            try:
                from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter

                smoothed = gpu_gaussian_filter(xp.asarray(grid, dtype=xp.float64), sigma=sigma, mode="nearest")
                return to_numpy(smoothed)
            except Exception:
                pass
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(grid.astype(np.float64), sigma=sigma, mode="nearest")


def standardize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardize columns; return scaled X, mean, std."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std
