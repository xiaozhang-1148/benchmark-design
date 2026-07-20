"""Pooled grayscale histogram and global Otsu threshold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmark_design.foreground.models import ForegroundThresholdConfig
from benchmark_design.io.image import otsu_from_histogram

GRAY_BIN_COUNT = 256


def accumulate_grayscale_histogram(
    gray: np.ndarray,
    histogram: np.ndarray | None = None,
) -> np.ndarray:
    """Add one page's raw uint8 grayscale values into a pooled histogram."""
    if histogram is None:
        histogram = np.zeros(GRAY_BIN_COUNT, dtype=np.int64)
    flat = np.asarray(gray, dtype=np.uint8).ravel()
    if flat.size == 0:
        return histogram
    histogram += np.bincount(flat, minlength=GRAY_BIN_COUNT).astype(np.int64, copy=False)
    return histogram


def global_pooled_otsu_gray_threshold(histogram: np.ndarray) -> float:
    """Return t_I from a pooled uint8 grayscale histogram using Otsu."""
    counts = np.asarray(histogram, dtype=np.int64).ravel()
    if counts.size != GRAY_BIN_COUNT:
        raise ValueError(f"grayscale histogram must have {GRAY_BIN_COUNT} bins")
    if counts.sum() <= 0:
        return 128.0
    return otsu_from_histogram(counts)


def gray_threshold_to_tau_d(
    gray_threshold: float,
    *,
    dark_reference: float,
    light_reference: float,
) -> float:
    """Convert gray threshold t_I to equivalent darkness threshold tau_D."""
    denom = light_reference - dark_reference
    if denom <= 0:
        denom = 1.0
    g_tilde = float(np.clip((gray_threshold - dark_reference) / denom, 0.0, 1.0))
    return 1.0 - g_tilde


def save_foreground_threshold_config(
    config: ForegroundThresholdConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_version": config.dataset_version,
        "dark_reference": config.dark_reference,
        "light_reference": config.light_reference,
        "dark_percentile": config.dark_percentile,
        "light_percentile": config.light_percentile,
        "threshold_method": config.threshold_method,
        "num_pages": config.image_count,
        "image_count": config.image_count,
        "gray_threshold": config.gray_threshold,
        "darkness_threshold": config.tau_d,
        "tau_D": config.tau_d,
        "foreground_rule": config.foreground_rule,
        "quantization": config.quantization,
        "histogram_weighting": config.histogram_weighting,
        "morphology": config.morphology,
        "gray_histogram": list(config.gray_histogram),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_foreground_threshold_config(path: Path) -> ForegroundThresholdConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gray_hist = payload.get("gray_histogram")
    gray_threshold = float(
        payload.get(
            "gray_threshold",
            payload.get("global_threshold", payload.get("t_I", 128.0)),
        )
    )
    tau_d = float(
        payload.get(
            "darkness_threshold",
            payload.get(
                "tau_D",
                payload.get(
                    "tau_d",
                    gray_threshold_to_tau_d(
                        gray_threshold,
                        dark_reference=float(payload["dark_reference"]),
                        light_reference=float(payload["light_reference"]),
                    ),
                ),
            ),
        )
    )
    return ForegroundThresholdConfig(
        dataset_version=str(payload.get("dataset_version", "")),
        dark_reference=float(payload["dark_reference"]),
        light_reference=float(payload["light_reference"]),
        dark_percentile=float(payload.get("dark_percentile", 1.0)),
        light_percentile=float(payload.get("light_percentile", 99.5)),
        gray_threshold=gray_threshold,
        tau_d=tau_d,
        threshold_method=str(payload.get("threshold_method", "global_pooled_otsu")),
        foreground_rule=str(payload.get("foreground_rule", "gray <= gray_threshold")),
        quantization=str(payload.get("quantization", "uint8")),
        histogram_weighting=str(payload.get("histogram_weighting", "pooled_pixels")),
        morphology=str(payload.get("morphology", "none")),
        image_count=int(payload.get("num_pages", payload.get("image_count", 0))),
        gray_histogram=tuple(int(v) for v in gray_hist) if gray_hist else (),
    )
