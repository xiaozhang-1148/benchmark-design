"""Block-level image-side benchmark configuration."""

from __future__ import annotations

from pathlib import Path

DEFAULT_BLOCK_LEVEL_INPUT = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark"
)
DEFAULT_BLOCK_LEVEL_OUTPUT_ROOT = Path("block_level")

BLOCK_LEVEL_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)

# Backward-compatible aliases (deprecated).
DEFAULT_VISION_INPUT = DEFAULT_BLOCK_LEVEL_INPUT
DEFAULT_VISION_OUTPUT_ROOT = DEFAULT_BLOCK_LEVEL_OUTPUT_ROOT
VISION_IMAGE_EXTENSIONS = BLOCK_LEVEL_IMAGE_EXTENSIONS
