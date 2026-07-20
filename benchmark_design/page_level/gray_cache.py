"""Temporary per-page grayscale cache to avoid duplicate image decodes."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PageGrayCache:
    """Store grayscale arrays between calibration and feature passes."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def store(self, image_id: str, gray: np.ndarray) -> None:
        np.save(self.root / f"{image_id}.npy", np.asarray(gray, dtype=np.uint8))

    def load(self, image_id: str) -> np.ndarray:
        cached = self.try_load(image_id)
        if cached is None:
            path = self.root / f"{image_id}.npy"
            raise FileNotFoundError(f"missing cached grayscale for image_id={image_id}: {path}")
        return cached

    def try_load(self, image_id: str) -> np.ndarray | None:
        path = self.root / f"{image_id}.npy"
        if not path.is_file():
            return None
        return np.load(path)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
