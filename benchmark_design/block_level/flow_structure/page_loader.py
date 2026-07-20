"""Load page-level mask annotations from benchmark JSON exports."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_design.io.benchmark_loader import iter_benchmark_json_paths
from benchmark_design.progress import parallel_map
from benchmark_design.block_level.flow_structure.geometry import polygon_bbox
from benchmark_design.block_level.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.block_level.processing_options import VisionProcessingOptions


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    if not image_path.is_file():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(image_path) as image:
        return image.size


def _infer_page_size(
    blocks: tuple[PageBlockAnnotation, ...],
    *,
    input_dir: Path,
    image_name: str,
    read_image_dimensions: bool = True,
) -> tuple[int, int]:
    if read_image_dimensions:
        image_path = input_dir / image_name
        size = _read_image_size(image_path)
        if size is not None:
            return size

    max_x = 0.0
    max_y = 0.0
    for block in blocks:
        if not block.polygon:
            continue
        _, _, x2, y2 = polygon_bbox(block.polygon)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    width = max(int(max_x), 1)
    height = max(int(max_y), 1)
    return width, height


def _load_page_annotation(
    json_path: Path,
    *,
    input_dir: Path,
    dataset: str,
    read_image_dimensions: bool = True,
) -> PageAnnotation:
    with json_path.open(encoding="utf-8") as handle:
        page = json.load(handle)
    image_name = str(page.get("image_name", json_path.name.removesuffix(".json")))
    page_id = json_path.stem
    blocks: list[PageBlockAnnotation] = []
    for block in page.get("blocks", []):
        polygon_raw = block.get("polygon") or []
        polygon = tuple((float(point[0]), float(point[1])) for point in polygon_raw if len(point) >= 2)
        block_order = int(block.get("order", 0))
        blocks.append(
            PageBlockAnnotation(
                page_id=page_id,
                block_id=f"{page_id}:block_{block_order}",
                block_type=str(block.get("type", "")),
                block_order=block_order,
                polygon=polygon,
            )
        )
    block_tuple = tuple(blocks)
    image_width, image_height = _infer_page_size(
        block_tuple,
        input_dir=input_dir,
        image_name=image_name,
        read_image_dimensions=read_image_dimensions,
    )
    return PageAnnotation(
        page_id=page_id,
        image_name=image_name,
        source_file=str(json_path.resolve()),
        image_width=image_width,
        image_height=image_height,
        blocks=block_tuple,
    )


def load_page_annotations(
    input_dir: Path,
    *,
    dataset: str = "ours",
    processing: VisionProcessingOptions | None = None,
) -> list[PageAnnotation]:
    processing = processing or VisionProcessingOptions()
    json_paths = iter_benchmark_json_paths(input_dir)
    if not json_paths:
        return []

    loader = lambda path: _load_page_annotation(
        path,
        input_dir=input_dir,
        dataset=dataset,
        read_image_dimensions=processing.read_image_dimensions,
    )
    if processing.workers is not None and processing.workers <= 1:
        return [loader(path) for path in json_paths]

    return parallel_map(
        loader,
        json_paths,
        description="Loading page annotations",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
