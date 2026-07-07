"""Vision / image-side benchmark configuration."""

from __future__ import annotations

from pathlib import Path

# Placeholder paths — update when the vision benchmark input layout is finalized.
DEFAULT_VISION_INPUT = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark"
)
DEFAULT_VISION_OUTPUT_ROOT = Path("vision")

# Image extensions accepted when scanning sample directories.
VISION_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
