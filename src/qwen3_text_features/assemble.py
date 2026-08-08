"""Assemble whole-page text from page-level JSON OCR lines (order-preserving, unmodified)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class AssembleError(ValueError):
    """Raised when OCR fields are missing/null/non-string (sample must fail)."""


@dataclass
class OrderedLine:
    block_order: int
    line_order: int
    ocr: str
    empty_ocr: bool = False
    internal_newline: bool = False


@dataclass
class AssembledPage:
    sample_id: str
    json_path: str
    image_name: str
    page_text: str
    ordered_lines: list[OrderedLine]
    block_count: int
    line_count: int
    empty_ocr_count: int
    internal_newline_count: int
    character_count: int
    block_types: list[str] = field(default_factory=list)

    def to_index_dict(self, *, row_index: int, token_count: int | None = None, status: str = "success") -> dict[str, Any]:
        return {
            "row_index": row_index,
            "sample_id": self.sample_id,
            "json_path": self.json_path,
            "image_name": self.image_name,
            "block_count": self.block_count,
            "line_count": self.line_count,
            "empty_ocr_count": self.empty_ocr_count,
            "internal_newline_count": self.internal_newline_count,
            "character_count": self.character_count,
            "token_count": token_count,
            "status": status,
        }

    def to_assembled_dict(self, *, row_index: int) -> dict[str, Any]:
        return {
            "row_index": row_index,
            "sample_id": self.sample_id,
            "json_path": self.json_path,
            "page_text": self.page_text,
            "ordered_lines": [asdict(x) for x in self.ordered_lines],
        }


def _as_int_order(value: Any, *, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssembleError(f"{ctx}: order must be int, got {type(value).__name__}={value!r}")
    return int(value)


def assemble_page_text(
    data: dict[str, Any],
    *,
    json_path: str,
    sample_id: str | None = None,
) -> AssembledPage:
    """
    Sort by block.order then line.order; join OCR strings with a single ``\\n``.

    OCR strings are used exactly as returned by ``json.loads`` — no cleanup,
    no unicode_escape, no stripping.
    """
    if not isinstance(data, dict):
        raise AssembleError(f"top-level JSON must be object, got {type(data).__name__}")
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        raise AssembleError("missing or non-list 'blocks'")

    image_name = data.get("image_name")
    if image_name is not None and not isinstance(image_name, str):
        raise AssembleError(f"image_name must be str|null, got {type(image_name).__name__}")
    image_name = image_name or Path(json_path).name.removesuffix(".json")

    # Sort blocks by order (stable secondary key = original index for ties)
    indexed_blocks = []
    for bi, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise AssembleError(f"blocks[{bi}] is not an object")
        bo = _as_int_order(block.get("order"), ctx=f"blocks[{bi}].order")
        indexed_blocks.append((bo, bi, block))
    indexed_blocks.sort(key=lambda t: (t[0], t[1]))

    ordered: list[OrderedLine] = []
    block_types: list[str] = []
    empty_ocr_count = 0
    internal_newline_count = 0

    for bo, bi, block in indexed_blocks:
        btype = block.get("type")
        block_types.append(str(btype) if btype is not None else "")
        lines = block.get("lines")
        if not isinstance(lines, list):
            raise AssembleError(f"block order={bo}: missing or non-list 'lines'")
        indexed_lines = []
        for li, line in enumerate(lines):
            if not isinstance(line, dict):
                raise AssembleError(f"block order={bo} lines[{li}] is not an object")
            lo = _as_int_order(line.get("order"), ctx=f"block order={bo} line[{li}].order")
            indexed_lines.append((lo, li, line))
        indexed_lines.sort(key=lambda t: (t[0], t[1]))

        for lo, li, line in indexed_lines:
            if "ocr" not in line:
                raise AssembleError(
                    f"sample={sample_id or json_path} block_order={bo} line_order={lo}: missing ocr"
                )
            ocr = line["ocr"]
            if ocr is None:
                raise AssembleError(
                    f"sample={sample_id or json_path} block_order={bo} line_order={lo}: ocr is null"
                )
            if not isinstance(ocr, str):
                raise AssembleError(
                    f"sample={sample_id or json_path} block_order={bo} line_order={lo}: "
                    f"ocr is {type(ocr).__name__}, expected str"
                )
            empty = ocr == ""
            has_nl = "\n" in ocr
            if empty:
                empty_ocr_count += 1
            if has_nl:
                internal_newline_count += 1
            ordered.append(
                OrderedLine(
                    block_order=bo,
                    line_order=lo,
                    ocr=ocr,
                    empty_ocr=empty,
                    internal_newline=has_nl,
                )
            )

    page_text = "\n".join(ol.ocr for ol in ordered)
    sid = sample_id or image_name
    return AssembledPage(
        sample_id=sid,
        json_path=str(Path(json_path).resolve()) if json_path else json_path,
        image_name=image_name,
        page_text=page_text,
        ordered_lines=ordered,
        block_count=len(indexed_blocks),
        line_count=len(ordered),
        empty_ocr_count=empty_ocr_count,
        internal_newline_count=internal_newline_count,
        character_count=len(page_text),
        block_types=block_types,
    )


def assemble_from_json_path(json_path: str | Path, *, sample_id: str | None = None) -> AssembledPage:
    path = Path(json_path)
    # utf-8 JSON parse only — no unicode_escape afterwards
    data = json.loads(path.read_text(encoding="utf-8"))
    return assemble_page_text(data, json_path=str(path.resolve()), sample_id=sample_id)
