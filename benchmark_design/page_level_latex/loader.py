"""Load page expressions for page-level LaTeX metrics.

Uses the same benchmark JSON source as Chapter-5 HMER. Unlike
``load_expressions``, this loader keeps empty OCR lines for audit statistics.
Geometry / polygon validity is never used as a filter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.progress import parallel_map_flatten


@dataclass(frozen=True, slots=True)
class RawExpressionRow:
    image_id: str
    image_name: str
    block_id: str
    line_id: str
    block_order: int
    line_order: int
    block_type: str
    raw_ocr_text: str
    source_file: str
    global_line_index: int = -1


def iter_benchmark_json_paths(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Benchmark input directory not found: {input_dir}")
    paths = sorted(input_dir.glob("*.jpg.json"))
    if not paths:
        paths = sorted(set(input_dir.glob("*.json")) - set(input_dir.glob("*.jpg.json")))
    return paths


def _load_page_rows(json_path: Path) -> list[RawExpressionRow]:
    with json_path.open(encoding="utf-8") as handle:
        page = json.load(handle)
    image_name = str(page.get("image_name", json_path.name.removesuffix(".json")))
    image_id = Path(image_name).stem
    source_file = str(json_path.resolve())
    rows: list[RawExpressionRow] = []
    for block in page.get("blocks", []):
        block_order = int(block.get("order", 0))
        block_type = str(block.get("type", ""))
        block_id = str(block_order)
        for line in block.get("lines", []):
            line_order = int(line.get("order", 0))
            line_id = f"{block_order}:{line_order}"
            raw_ocr = str(line.get("ocr", ""))
            rows.append(
                RawExpressionRow(
                    image_id=image_id,
                    image_name=image_name,
                    block_id=block_id,
                    line_id=line_id,
                    block_order=block_order,
                    line_order=line_order,
                    block_type=block_type,
                    raw_ocr_text=raw_ocr,
                    source_file=source_file,
                )
            )
    return rows


def load_raw_expressions(
    input_dir: Path,
    *,
    show_progress: bool = True,
    workers: int | None = None,
) -> list[RawExpressionRow]:
    paths = iter_benchmark_json_paths(input_dir)
    if not paths:
        return []
    if workers is not None and workers <= 1:
        rows: list[RawExpressionRow] = []
        for path in paths:
            rows.extend(_load_page_rows(path))
    else:
        rows = parallel_map_flatten(
            _load_page_rows,
            paths,
            description="Loading page LaTeX annotations",
            show_progress=show_progress,
            workers=workers,
        )
    rows.sort(key=lambda row: (row.image_id, row.block_order, row.line_order))
    return [
        RawExpressionRow(
            image_id=row.image_id,
            image_name=row.image_name,
            block_id=row.block_id,
            line_id=row.line_id,
            block_order=row.block_order,
            line_order=row.line_order,
            block_type=row.block_type,
            raw_ocr_text=row.raw_ocr_text,
            source_file=row.source_file,
            global_line_index=index,
        )
        for index, row in enumerate(rows)
    ]
