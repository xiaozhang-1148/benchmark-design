"""Unified vision benchmark dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_design.vision.flow_structure.models import PageAnnotation
from benchmark_design.vision.flow_structure.page_loader import load_page_annotations
from benchmark_design.vision.processing import _read_dimensions, _resolve_image_path
from benchmark_design.vision.processing_options import VisionProcessingOptions
from benchmark_design.vision.sample_record import ImageSampleRecord


@dataclass(frozen=True, slots=True)
class VisionBenchmarkDataset:
    pages: tuple[PageAnnotation, ...]
    samples: tuple[ImageSampleRecord, ...]


_VISION_DATASET_CACHE: dict[tuple[str, str, bool], VisionBenchmarkDataset] = {}


def clear_vision_benchmark_dataset_cache() -> None:
    """Clear the in-process vision dataset cache (mainly for tests)."""
    _VISION_DATASET_CACHE.clear()


def _page_to_sample(
    page: PageAnnotation,
    *,
    input_dir: Path,
    dataset: str,
    read_image_dimensions: bool,
) -> ImageSampleRecord:
    image_path = _resolve_image_path(page.image_name, input_dir)
    if image_path is None:
        image_path = input_dir / page.image_name
    width_px: int | None = page.image_width
    height_px: int | None = page.image_height
    if (
        read_image_dimensions
        and image_path.is_file()
        and (width_px is None or height_px is None or width_px <= 0 or height_px <= 0)
    ):
        read_w, read_h = _read_dimensions(image_path)
        if read_w is not None:
            width_px = read_w
        if read_h is not None:
            height_px = read_h
    return ImageSampleRecord(
        sample_id=page.image_name,
        image_path=image_path,
        dataset=dataset,
        source_file=page.source_file,
        width_px=width_px,
        height_px=height_px,
        page_id=page.page_id,
        expression_id=page.page_id,
    )


def load_vision_benchmark_dataset(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    dataset: str = "ours",
) -> VisionBenchmarkDataset:
    processing = processing or VisionProcessingOptions()
    pages = tuple(
        load_page_annotations(input_dir, dataset=dataset, processing=processing)
    )
    samples = tuple(
        _page_to_sample(
            page,
            input_dir=input_dir,
            dataset=dataset,
            read_image_dimensions=processing.read_image_dimensions,
        )
        for page in pages
    )
    return VisionBenchmarkDataset(pages=pages, samples=samples)


def load_vision_benchmark_dataset_cached(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    dataset: str = "ours",
    prebuilt: VisionBenchmarkDataset | None = None,
) -> VisionBenchmarkDataset:
    """Return a cached vision dataset, optionally seeding with a pre-built instance."""
    processing = processing or VisionProcessingOptions()
    resolved = str(Path(input_dir).resolve())
    cache_key = (dataset, resolved, processing.read_image_dimensions)
    if prebuilt is not None:
        _VISION_DATASET_CACHE[cache_key] = prebuilt
        return prebuilt
    if cache_key in _VISION_DATASET_CACHE:
        return _VISION_DATASET_CACHE[cache_key]
    loaded = load_vision_benchmark_dataset(input_dir, processing=processing, dataset=dataset)
    _VISION_DATASET_CACHE[cache_key] = loaded
    return loaded
