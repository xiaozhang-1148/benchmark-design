"""Load OCR expressions from per-image benchmark JSON files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.progress import parallel_map_flatten


from benchmark_design.io.polygon import parse_polygon_points


@dataclass(frozen=True, slots=True)
class ExpressionRecord:
    image_name: str
    block_order: int
    line_order: int
    block_type: str
    ocr: str
    dataset: str = "ours"
    source_file: str = ""
    expression_id: str = ""
    line_id: str = ""
    line_polygon: tuple[tuple[float, float], ...] = ()


def iter_benchmark_json_paths(input_dir: Path) -> list[Path]:
    """Return sorted paths to ``*.jpg.json`` annotation files."""
    if not input_dir.is_dir():
        msg = f"Benchmark input directory not found: {input_dir}"
        raise FileNotFoundError(msg)
    return sorted(input_dir.glob("*.jpg.json"))


def _load_json_expressions(json_path: Path, *, dataset: str = "ours") -> list[ExpressionRecord]:
    with json_path.open(encoding="utf-8") as handle:
        page = json.load(handle)
    image_name = str(page.get("image_name", json_path.name.removesuffix(".json")))
    source_file = str(json_path.resolve())
    records: list[ExpressionRecord] = []
    for block in page.get("blocks", []):
        block_order = int(block.get("order", 0))
        block_type = str(block.get("type", ""))
        for line in block.get("lines", []):
            ocr = str(line.get("ocr", "")).strip()
            if not ocr:
                continue
            line_order = int(line.get("order", 0))
            line_id = f"{block_order}:{line_order}"
            line_polygon = parse_polygon_points(line.get("polygon"))
            records.append(
                ExpressionRecord(
                    image_name=image_name,
                    block_order=block_order,
                    line_order=line_order,
                    block_type=block_type,
                    ocr=ocr,
                    dataset=dataset,
                    source_file=source_file,
                    expression_id=f"{dataset}:{json_path.stem}:{line_id}",
                    line_id=line_id,
                    line_polygon=line_polygon,
                )
            )
    return records


def load_expressions(
    input_dir: Path,
    *,
    dataset: str = "ours",
    json_paths: list[Path] | None = None,
    show_progress: bool = False,
    workers: int | None = None,
) -> list[ExpressionRecord]:
    """Load all OCR expressions, optionally using parallel JSON parsing."""
    paths = json_paths if json_paths is not None else iter_benchmark_json_paths(input_dir)
    if not paths:
        return []

    loader = lambda path: _load_json_expressions(path, dataset=dataset)

    if workers is not None and workers <= 1:
        if show_progress:
            return parallel_map_flatten(
                loader,
                paths,
                description="Loading benchmark JSON",
                show_progress=True,
                workers=1,
            )
        records: list[ExpressionRecord] = []
        for json_path in paths:
            records.extend(loader(json_path))
        return records

    return parallel_map_flatten(
        loader,
        paths,
        description="Loading benchmark JSON",
        show_progress=show_progress,
        workers=workers,
    )


def iter_expressions(input_dir: Path) -> Iterator[ExpressionRecord]:
    """Yield one record per line with non-empty OCR text."""
    for record in load_expressions(input_dir, show_progress=False, workers=1):
        yield record
