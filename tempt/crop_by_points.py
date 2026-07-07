#!/usr/bin/env python3
"""Crop image regions defined by polygon points in annotation JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip())
    return cleaned or "region"


def _load_points(annotation: dict) -> list[tuple[float, float]]:
    contour = annotation.get("contour") or {}
    raw_points = contour.get("points") or []
    points: list[tuple[float, float]] = []
    for point in raw_points:
        if isinstance(point, dict):
            points.append((float(point["x"]), float(point["y"])))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def _polygon_bbox(points: list[tuple[float, float]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(int(min(xs)), 0)
    y1 = max(int(min(ys)), 0)
    x2 = max(int(max(xs)) + 1, x1 + 1)
    y2 = max(int(max(ys)) + 1, y1 + 1)
    return x1, y1, x2, y2


def crop_polygon(image: Image.Image, points: list[tuple[float, float]]) -> Image.Image:
    if len(points) < 3:
        raise ValueError("polygon must contain at least 3 points")

    x1, y1, x2, y2 = _polygon_bbox(points)
    x1 = min(x1, image.width - 1)
    y1 = min(y1, image.height - 1)
    x2 = min(x2, image.width)
    y2 = min(y2, image.height)

    cropped = image.crop((x1, y1, x2, y2))
    mask = Image.new("L", (x2 - x1, y2 - y1), 0)
    shifted = [(x - x1, y - y1) for x, y in points]
    ImageDraw.Draw(mask).polygon(shifted, fill=255)

    rgba = cropped.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def crop_image_by_annotations(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    *,
    background: str = "white",
) -> list[Path]:
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    image = Image.open(image_path).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for index, annotation in enumerate(payload.get("annotations", [])):
        points = _load_points(annotation)
        if len(points) < 3:
            continue

        order = str(annotation.get("order") or index)
        track_id = str(annotation.get("trackId") or f"idx_{index}")
        class_alias = str((annotation.get("class") or {}).get("alias") or "block")

        cropped = crop_polygon(image, points)
        if background == "white":
            background_image = Image.new("RGBA", cropped.size, (255, 255, 255, 255))
            cropped = Image.alpha_composite(background_image, cropped).convert("RGB")
        elif background == "transparent":
            pass
        else:
            raise ValueError(f"unsupported background: {background}")

        output_path = output_dir / f"{image_path.stem}_{index:03d}_{_safe_name(order)}_{_safe_name(class_alias)}_{_safe_name(track_id)}.png"
        cropped.save(output_path)
        saved.append(output_path)

    return saved


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Crop image regions by polygon points in JSON.")
    parser.add_argument(
        "--image",
        type=Path,
        default=script_dir / "mb7r6ah2iimsvlne2s.jpg",
        help="Input image path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=script_dir / "mb7r6ah2iimsvlne2s.json",
        help="Annotation JSON path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "crops",
        help="Directory for cropped images",
    )
    parser.add_argument(
        "--background",
        choices=("white", "transparent"),
        default="white",
        help="Background outside polygon (default: white)",
    )
    args = parser.parse_args()

    saved = crop_image_by_annotations(
        args.image,
        args.json,
        args.output_dir,
        background=args.background,
    )
    print(f"Saved {len(saved)} cropped images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
