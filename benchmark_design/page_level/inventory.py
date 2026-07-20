"""Discover benchmark images and extract inventory metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark_design.io.benchmark_loader import iter_benchmark_json_paths
from benchmark_design.page_level.calibration import density_gray_histogram
from benchmark_design.page_level.models import ImageInventoryRow, ImageRecord
from benchmark_design.block_level.processing import _resolve_image_path
from benchmark_design.progress import parallel_map


def _discover_image_record(json_path: Path, *, input_dir: Path) -> ImageRecord | None:
    with json_path.open(encoding="utf-8") as handle:
        page = json.load(handle)
    image_name = str(page.get("image_name", json_path.name.removesuffix(".json")))
    image_path = _resolve_image_path(image_name, input_dir)
    if image_path is None or not image_path.is_file():
        return None
    return ImageRecord(
        image_id=image_path.stem,
        relative_path=image_path.relative_to(input_dir).as_posix(),
        absolute_path=image_path,
    )


def discover_images_from_benchmark(
    input_dir: Path,
    *,
    dataset: str = "ours",
    show_progress: bool = False,
    workers: int | None = None,
) -> list[ImageRecord]:
    del dataset  # ours-only benchmark layout; kept for API compatibility.
    json_paths = iter_benchmark_json_paths(input_dir)
    if not json_paths:
        return []

    if workers is not None and workers <= 1:
        records = [
            record
            for path in json_paths
            if (record := _discover_image_record(path, input_dir=input_dir)) is not None
        ]
    else:
        discovered = parallel_map(
            lambda path: _discover_image_record(path, input_dir=input_dir),
            json_paths,
            description="Discovering benchmark images",
            show_progress=show_progress,
            workers=workers,
        )
        records = [record for record in discovered if record is not None]

    records.sort(key=lambda item: item.image_id)
    return records


def _effective_color_type(
    *,
    rgb_channels_identical: bool,
    alpha_nonopaque_ratio: float,
) -> str:
    tags: list[str] = []
    if rgb_channels_identical:
        tags.append("grayscale_content")
    else:
        tags.append("color_content")
    if alpha_nonopaque_ratio > 0.0:
        tags.append("transparency_used")
    else:
        tags.append("opaque")
    return ";".join(tags)


def inspect_image_inventory_and_histogram(
    record: ImageRecord,
) -> tuple[ImageInventoryRow, np.ndarray]:
    with Image.open(record.absolute_path) as image:
        stored_color_mode = image.mode
        rgb = image.convert("RGBA")
        array = np.asarray(rgb)
        height, width = array.shape[:2]
        channels = array.shape[2] if array.ndim == 3 else 1
        rgb_channels = array[..., :3]
        rgb_channels_identical = bool(
            np.all(rgb_channels[..., 0] == rgb_channels[..., 1])
            and np.all(rgb_channels[..., 1] == rgb_channels[..., 2])
        )
        alpha = array[..., 3] if channels == 4 else None
        alpha_nonopaque_ratio = 0.0
        if alpha is not None and alpha.size > 0:
            alpha_nonopaque_ratio = float(np.mean(alpha < 255))
        gray = np.array(image.convert("L"), dtype=np.uint8)

    inventory = ImageInventoryRow(
        image_id=record.image_id,
        relative_path=record.relative_path,
        width=int(width),
        height=int(height),
        aspect_ratio=float(width / height) if height else 0.0,
        file_format=record.absolute_path.suffix.lstrip(".").lower(),
        stored_color_mode=stored_color_mode,
        channel_count=int(channels),
        dtype=str(array.dtype),
        bits_per_channel=8,
        rgb_channels_identical=rgb_channels_identical,
        alpha_nonopaque_ratio=alpha_nonopaque_ratio,
        effective_color_type=_effective_color_type(
            rgb_channels_identical=rgb_channels_identical,
            alpha_nonopaque_ratio=alpha_nonopaque_ratio,
        ),
    )
    return inventory, density_gray_histogram(gray)


def build_image_inventory_and_histograms(
    records: list[ImageRecord],
    *,
    show_progress: bool = False,
    workers: int | None = None,
) -> tuple[list[ImageInventoryRow], list[np.ndarray]]:
    paired = parallel_map(
        inspect_image_inventory_and_histogram,
        records,
        description="Building image inventory and calibration histograms",
        show_progress=show_progress,
        workers=workers,
    )
    inventory = [item[0] for item in paired]
    histograms = [item[1] for item in paired]
    return inventory, histograms
