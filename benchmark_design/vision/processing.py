"""Load and enumerate vision benchmark image samples."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.config.vision import VISION_IMAGE_EXTENSIONS
from benchmark_design.io.benchmark_loader import load_expressions
from benchmark_design.vision.processing_options import VisionProcessingOptions
from benchmark_design.vision.sample_record import ImageSampleRecord


def _resolve_image_path(image_name: str, input_dir: Path) -> Path | None:
    stem = Path(image_name).stem
    for ext in VISION_IMAGE_EXTENSIONS:
        candidate = input_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
        candidate = input_dir / image_name
        if candidate.is_file():
            return candidate
    return None


def _read_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
    except ImportError:
        return None, None
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def load_image_samples_from_benchmark_json(
    input_dir: Path,
    *,
    dataset: str = "ours",
    processing: VisionProcessingOptions | None = None,
) -> list[ImageSampleRecord]:
    """Build vision records from the same JSON page export used by the HMER pipeline."""
    processing = processing or VisionProcessingOptions()
    expressions = load_expressions(
        input_dir,
        dataset=dataset,
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
    seen: set[str] = set()
    records: list[ImageSampleRecord] = []
    for expression in expressions:
        if expression.image_name in seen:
            continue
        seen.add(expression.image_name)
        image_path = _resolve_image_path(expression.image_name, input_dir)
        if image_path is None:
            continue
        width_px, height_px = (None, None)
        if processing.read_image_dimensions:
            width_px, height_px = _read_dimensions(image_path)
        records.append(
            ImageSampleRecord(
                sample_id=expression.image_name,
                image_path=image_path,
                dataset=dataset,
                source_file=expression.source_file,
                width_px=width_px,
                height_px=height_px,
                page_id=expression.source_file,
                expression_id=expression.expression_id or expression.image_name,
            )
        )
    return records
