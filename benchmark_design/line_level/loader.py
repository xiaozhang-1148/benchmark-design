"""Discover benchmark pages and raw line annotations."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_design.io.benchmark_loader import iter_benchmark_json_paths
from benchmark_design.io.polygon import parse_polygon_points
from benchmark_design.line_level.models import LineAnnotation, LineLevelConfig, PageTask
from benchmark_design.ocr.line_crop import resolve_page_image_path
from benchmark_design.progress import parallel_map


def _load_page_lines(
    json_path: Path,
    *,
    ignore_labels: tuple[str, ...],
) -> tuple[str, tuple[LineAnnotation, ...]]:
    with json_path.open(encoding="utf-8") as handle:
        page = json.load(handle)
    image_name = str(page.get("image_name", json_path.name.removesuffix(".json")))
    image_id = Path(image_name).stem
    source_file = str(json_path.resolve())
    lines: list[LineAnnotation] = []
    for block in page.get("blocks", []):
        block_order = int(block.get("order", 0))
        block_type = str(block.get("type", ""))
        is_ignore = block_type in ignore_labels
        for line in block.get("lines", []):
            line_order = int(line.get("order", 0))
            line_id = f"{block_order}:{line_order}"
            polygon = parse_polygon_points(line.get("polygon"))
            ocr = str(line.get("ocr", ""))
            lines.append(
                LineAnnotation(
                    image_id=image_id,
                    image_name=image_name,
                    line_id=line_id,
                    block_order=block_order,
                    line_order=line_order,
                    block_type=block_type,
                    polygon=polygon,
                    ocr=ocr,
                    source_file=source_file,
                    is_ignore=is_ignore,
                )
            )
    return image_name, tuple(lines)


def _load_page_task(
    json_path: Path,
    *,
    input_dir: Path,
    ignore_labels: tuple[str, ...],
) -> PageTask:
    image_name, lines = _load_page_lines(json_path, ignore_labels=ignore_labels)
    image_id = Path(image_name).stem
    image_path = resolve_page_image_path(image_name, input_dir)
    return PageTask(
        image_id=image_id,
        image_name=image_name,
        json_path=json_path,
        image_path=image_path,
        width=0,
        height=0,
        dpi=None,
        lines=lines,
    )


def discover_pages_from_benchmark(config: LineLevelConfig) -> list[PageTask]:
    input_dir = config.input_root
    json_paths = iter_benchmark_json_paths(input_dir)
    if not json_paths:
        return []

    loader = lambda path: _load_page_task(
        path,
        input_dir=input_dir,
        ignore_labels=config.ignore_labels,
    )
    if config.workers is not None and config.workers <= 1:
        pages = [loader(path) for path in json_paths]
    else:
        pages = parallel_map(
            loader,
            json_paths,
            description="Loading line-level page annotations",
            show_progress=config.show_progress,
            workers=config.workers,
        )
    pages.sort(key=lambda item: item.image_id)
    return pages
