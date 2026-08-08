#!/usr/bin/env python3
"""Convert public-release benchmark format to per-page benchmark JSON for benchmark_design.

Source (formatter output, single file per split):
  .../datasets/benchmark/annotations.json   # list of page dicts
  .../datasets/benchmark/images/*.jpg       # page images
  .../datasets/benchmark/manifest.json

Target (benchmark_design input convention, one JSON per page):
  <out>/<image_name>.json  -> {"image_name": ..., "blocks": [
      {"order": int, "type": str, "polygon": [[x, y], ...], "lines": [
          {"order": int, "ocr": str, "polygon": [[x, y], ...]}, ... ]}, ... ]}
  <out>/<image_name>.jpg   -> copied image

Mapping:
  region_type          -> block.type          (Txtblock / deleted_text_block / figure / chart)
  region_polygon       -> block.polygon
  region.reading_order -> block.order
  line.formula         -> line.ocr
  line.reading_order   -> line.order
  line.polygon         -> line.polygon
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Ordering key: block reading order (stable), then original index.
def _order_value(value, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _block_sort_key(page_index: int, region_index: int, region: dict) -> tuple[int, int]:
    return (_order_value(region.get("reading_order"), region_index), region_index)


def convert(src_dir: Path, out_dir: Path) -> None:
    ann_path = src_dir / "annotations.json"
    images_dir = src_dir / "images"
    if not ann_path.is_file():
        raise FileNotFoundError(f"annotations.json not found: {ann_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pages = json.loads(ann_path.read_text(encoding="utf-8"))
    if not isinstance(pages, list):
        raise ValueError(f"annotations.json expected a list of pages, got {type(pages).__name__}")

    converted = 0
    copied = 0
    for page_index, page in enumerate(pages):
        image_name = str(page.get("image_name", "")).strip()
        if not image_name:
            raise ValueError(f"page[{page_index}] missing image_name")
        stem = image_name[:-4] if image_name.lower().endswith(".jpg") else image_name

        # Copy page image (idempotent: only copy if the target is missing or differs).
        src_img = images_dir / image_name
        dst_img = out_dir / image_name
        if src_img.is_file():
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                copied += 1
        else:
            print(f"WARN: missing source image {src_img.name}")

        blocks: list[dict] = []
        regions = page.get("regions", [])
        ordered = sorted(
            ((i, r) for i, r in enumerate(regions)),
            key=lambda pair: _block_sort_key(page_index, pair[0], pair[1]),
        )
        for region_index, region in ordered:
            lines: list[dict] = []
            for line_index, line in enumerate(region.get("lines", [])):
                ocr = str(line.get("formula", "")).strip()
                if not ocr:
                    continue
                lines.append(
                    {
                        "order": _order_value(line.get("reading_order"), line_index),
                        "ocr": ocr,
                        "polygon": line.get("polygon") or [],
                    }
                )
            blocks.append(
                {
                    "order": _order_value(region.get("reading_order"), region_index),
                    "type": str(region.get("region_type", "")),
                    "polygon": region.get("region_polygon") or [],
                    "lines": lines,
                }
            )

        out_json = out_dir / f"{stem}.jpg.json"
        out_json.write_text(
            json.dumps({"image_name": image_name, "blocks": blocks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        converted += 1

    print(f"converted pages: {converted}")
    print(f"images copied: {copied}")
    print(f"output: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="public-release benchmark formatter dir (has annotations.json + images/)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output raw_dataset dir (per-page *.jpg.json + *.jpg)")
    args = parser.parse_args()
    convert(args.src, args.out)


if __name__ == "__main__":
    main()
