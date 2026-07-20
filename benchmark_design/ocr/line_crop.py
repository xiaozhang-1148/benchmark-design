"""Crop benchmark line regions from page images using polygon annotations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from benchmark_design.config.block_level import VISION_IMAGE_EXTENSIONS
from benchmark_design.io.polygon import parse_polygon_points


def resolve_page_image_path(image_name: str, input_dir: Path) -> Path | None:
    stem = Path(image_name).stem
    for ext in VISION_IMAGE_EXTENSIONS:
        candidate = input_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    candidate = input_dir / image_name
    if candidate.is_file():
        return candidate
    return None


def polygon_bbox(points: Sequence[tuple[float, float]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(int(min(xs)), 0)
    y1 = max(int(min(ys)), 0)
    x2 = max(int(max(xs)) + 1, x1 + 1)
    y2 = max(int(max(ys)) + 1, y1 + 1)
    return x1, y1, x2, y2


def crop_line_polygon(
    image_path: Path,
    points: Sequence[tuple[float, float]],
    *,
    margin_px: int = 4,
    background: str = "white",
):
    """Return an RGB PIL image cropped to the line polygon with optional margin."""
    from PIL import Image, ImageDraw

    if len(points) < 3:
        raise ValueError("polygon must contain at least 3 points")

    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = polygon_bbox(points)
    x1 = max(x1 - margin_px, 0)
    y1 = max(y1 - margin_px, 0)
    x2 = min(x2 + margin_px, image.width)
    y2 = min(y2 + margin_px, image.height)

    cropped = image.crop((x1, y1, x2, y2))
    mask = Image.new("L", (x2 - x1, y2 - y1), 0)
    shifted = [(x - x1, y - y1) for x, y in points]
    ImageDraw.Draw(mask).polygon(shifted, fill=255)

    rgba = cropped.convert("RGBA")
    rgba.putalpha(mask)
    if background == "white":
        background_image = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgba = Image.alpha_composite(background_image, rgba).convert("RGB")
    elif background != "transparent":
        msg = f"unsupported background: {background}"
        raise ValueError(msg)
    return rgba
