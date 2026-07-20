"""Shared page-level intensity normalization (float [0, 1])."""

from __future__ import annotations

import numpy as np


def compute_g_tilde(
    gray: np.ndarray,
    *,
    dark_reference: float,
    light_reference: float,
) -> np.ndarray:
    """Return normalized brightness G~ in [0, 1] (1 = white background)."""
    denom = light_reference - dark_reference
    if denom <= 0:
        denom = 1.0
    normalized = (gray.astype(np.float32) - dark_reference) / denom
    return np.clip(normalized, 0.0, 1.0)


def compute_darkness(
    gray: np.ndarray,
    *,
    dark_reference: float,
    light_reference: float,
) -> np.ndarray:
    """Return darkness S = 1 - G~ (0 = white background, 1 = black foreground)."""
    return 1.0 - compute_g_tilde(
        gray,
        dark_reference=dark_reference,
        light_reference=light_reference,
    )
