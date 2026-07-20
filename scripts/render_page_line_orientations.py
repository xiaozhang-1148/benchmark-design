"""Render per-line orientation visualizations for one benchmark page."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from benchmark_design.line_level.geometry import (
    _contour_points_for_orientation,
    _long_side_alpha_deg,
    _long_side_signed_deg,
    axis_aligned_bbox,
    geometry_from_line,
)
from benchmark_design.line_level.loader import _load_page_lines
from benchmark_design.line_level.validation import validate_line_polygon


def _obb_axis_from_contour(contour: np.ndarray) -> tuple[list[tuple[float, float]], tuple[float, float], tuple[float, float]]:
    rect = cv2.minAreaRect(contour.astype(np.float32))
    box = [tuple(map(float, point)) for point in cv2.boxPoints(rect)]
    (cx, cy), (rw, rh), angle = rect
    if rw >= rh:
        length = float(rw)
        axis_angle = math.radians(float(angle))
    else:
        length = float(rh)
        axis_angle = math.radians(float(angle) + 90.0)
    half = length * 0.5
    p1 = (float(cx) - math.cos(axis_angle) * half, float(cy) - math.sin(axis_angle) * half)
    p2 = (float(cx) + math.cos(axis_angle) * half, float(cy) + math.sin(axis_angle) * half)
    return box, p1, p2


def _crop_around(image: Image.Image, points: list[tuple[float, float]], *, pad: int = 40) -> tuple[Image.Image, tuple[int, int]]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = max(0, int(min(xs) - pad))
    top = max(0, int(min(ys) - pad))
    right = min(image.width, int(max(xs) + pad))
    bottom = min(image.height, int(max(ys) + pad))
    return image.crop((left, top, right, bottom)), (left, top)


def _aabb_box(shape: Polygon) -> list[tuple[float, float]]:
    x_min, y_min, x_max, y_max = axis_aligned_bbox(shape)
    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]


def _shift(points: list[tuple[float, float]], origin: tuple[int, int]) -> list[tuple[float, float]]:
    ox, oy = origin
    return [(x - ox, y - oy) for x, y in points]


def _draw_line_overlays(
    draw: ImageDraw.ImageDraw,
    *,
    origin: tuple[int, int],
    polygon_pts: list[tuple[float, float]],
    aabb_pts: list[tuple[float, float]],
    obb_pts: list[tuple[float, float]],
    axis_pts: tuple[tuple[float, float], tuple[float, float]],
    line_width: int,
) -> None:
    draw.polygon(_shift(aabb_pts, origin), outline=(0, 100, 220), width=line_width)
    draw.polygon(_shift(obb_pts, origin), outline=(255, 140, 0), width=line_width)
    draw.polygon(_shift(polygon_pts, origin), outline=(220, 0, 0), width=line_width)
    draw.line(_shift([axis_pts[0], axis_pts[1]], origin), fill=(0, 160, 0), width=line_width)


def _draw_overview_legend(draw: ImageDraw.ImageDraw, *, x: int, y: int) -> None:
    items = [
        ((220, 0, 0), "GT polygon"),
        ((0, 100, 220), "AABB (axis-aligned)"),
        ((255, 140, 0), "OBB (min rotated rect)"),
        ((0, 160, 0), "OBB long axis"),
    ]
    box_w, row_h = 250, 18
    draw.rectangle((x, y, x + box_w, y + row_h * len(items) + 8), fill=(255, 255, 255, 220), outline=(80, 80, 80))
    for index, (color, label) in enumerate(items):
        row_y = y + 6 + index * row_h
        draw.line([(x + 8, row_y + 8), (x + 28, row_y + 8)], fill=color, width=3)
        draw.text((x + 34, row_y), label, fill=(0, 0, 0))


def render_page(
    image_path: Path,
    json_path: Path,
    output_dir: Path,
    *,
    pad: int = 40,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    image_name, lines = _load_page_lines(json_path, ignore_labels=())
    page = Image.open(image_path).convert("RGB")
    page_w, page_h = page.size

    rows: list[dict[str, object]] = []
    overview = page.copy()
    overview_draw = ImageDraw.Draw(overview)

    for line in lines:
        ok, reason, shape = validate_line_polygon(line, image_width=page_w, image_height=page_h)
        row: dict[str, object] = {
            "image_id": line.image_id,
            "line_id": line.line_id,
            "block_type": line.block_type,
            "is_ignore": line.is_ignore,
            "valid": ok and not line.is_ignore,
            "invalid_reason": reason if not ok else "",
            "polygon_point_count": len(line.polygon),
            "orientation_deg": "",
            "alpha_deg": "",
            "obb_long_side_px": "",
            "obb_short_side_px": "",
            "bbox_width_px": "",
            "bbox_height_px": "",
            "aspect_ratio": "",
            "crop_path": "",
        }
        if not ok or shape is None or line.is_ignore:
            rows.append(row)
            continue

        geom = geometry_from_line(line, shape, page_width=page_w, page_height=page_h)
        contour = _contour_points_for_orientation(shape, page_width=page_w, page_height=page_h)
        rect = cv2.minAreaRect(contour.astype(np.float32))
        (rw, rh), angle = rect[1], float(rect[2])
        alpha = _long_side_alpha_deg(rw, rh, angle)
        signed = float(geom["orientation_deg"])
        obb_box, p1, p2 = _obb_axis_from_contour(contour)
        aabb_box = _aabb_box(shape)

        row.update(
            {
                "orientation_deg": f"{signed:.4f}",
                "alpha_deg": f"{alpha:.4f}",
                "obb_long_side_px": f"{geom['obb_long_side_px']:.2f}",
                "obb_short_side_px": f"{geom['obb_short_side_px']:.2f}",
                "bbox_width_px": f"{geom['bbox_width_px']:.2f}",
                "bbox_height_px": f"{geom['bbox_height_px']:.2f}",
                "aspect_ratio": f"{geom['aspect_ratio']:.4f}",
            }
        )

        polygon_pts = list(line.polygon)
        crop, origin = _crop_around(
            page,
            polygon_pts + aabb_box + obb_box + [p1, p2],
            pad=pad,
        )
        draw = ImageDraw.Draw(crop)
        _draw_line_overlays(
            draw,
            origin=origin,
            polygon_pts=polygon_pts,
            aabb_pts=aabb_box,
            obb_pts=obb_box,
            axis_pts=(p1, p2),
            line_width=3,
        )
        label = f"{line.image_id} {line.line_id} α={signed:.2f}° |α|={alpha:.2f}°"
        draw.rectangle((0, 0, crop.width, 44), fill=(255, 255, 220))
        draw.text((4, 4), label[:140], fill=(0, 0, 0))
        _draw_overview_legend(draw, x=4, y=20)

        safe_id = line.line_id.replace(":", "_")
        crop_path = crops_dir / f"{safe_id}.png"
        crop.save(crop_path)
        row["crop_path"] = crop_path.relative_to(output_dir).as_posix()
        rows.append(row)

        _draw_line_overlays(
            overview_draw,
            origin=(0, 0),
            polygon_pts=polygon_pts,
            aabb_pts=aabb_box,
            obb_pts=obb_box,
            axis_pts=(p1, p2),
            line_width=2,
        )
        overview_draw.text(
            (polygon_pts[0][0], max(0, polygon_pts[0][1] - 14)),
            f"{line.line_id}:{signed:.1f}°",
            fill=(0, 100, 0),
        )

    _draw_overview_legend(overview_draw, x=12, y=12)
    overview_path = output_dir / "overview_all_lines.png"
    overview.save(overview_path)

    csv_path = output_dir / "line_orientations.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "image_path": str(image_path),
        "json_path": str(json_path),
        "page_size": [page_w, page_h],
        "line_count": len(lines),
        "valid_line_count": sum(1 for row in rows if row["valid"]),
        "overview": overview_path.name,
        "csv": csv_path.name,
        "crops_dir": crops_dir.name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad", type=int, default=40)
    args = parser.parse_args()
    out = render_page(args.image, args.json, args.output, pad=args.pad)
    print(f"Wrote outputs to {out}")


if __name__ == "__main__":
    main()
