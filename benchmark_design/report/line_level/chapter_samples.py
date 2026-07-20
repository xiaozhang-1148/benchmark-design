"""Real-page sample overlays for chapter 4 figures."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from benchmark_design.line_level.bbox_ink import (
    compute_bbox_outside_ink,
    load_calibration_result,
    load_normalized_ink_mask,
)
from benchmark_design.line_level.models import LineLevelConfig, LineMetricsRow, TargetPairRow
from benchmark_design.report.line_level.chapter_tables import (
    INK_STATE_POSITIVE_INK,
    classify_bbox_outside_ink_state,
)


def _resolve_image_path(input_dir: Path, image_id: str, extensions: tuple[str, ...]) -> Path | None:
    for ext in extensions:
        candidate = input_dir / f"{image_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _load_polygons_for_ids(
    config: LineLevelConfig,
    needed: set[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[float, float], ...]]:
    """Load polygons only for requested (image_id, line_id) pairs."""
    if not needed:
        return {}
    from benchmark_design.line_level.loader import _load_page_lines

    needed_images = sorted({image_id for image_id, _line_id in needed})
    lookup: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {}
    for image_id in needed_images:
        json_path = None
        for ext in config.image_extensions:
            candidate = config.input_root / f"{image_id}{ext}.json"
            if candidate.is_file():
                json_path = candidate
                break
        if json_path is None:
            # fallback naming: image_id.jpg.json already covered; also bare .json
            bare = config.input_root / f"{image_id}.json"
            if bare.is_file():
                json_path = bare
        if json_path is None:
            continue
        _image_name, lines = _load_page_lines(json_path, ignore_labels=config.ignore_labels)
        for line in lines:
            key = (line.image_id, line.line_id)
            if key in needed:
                lookup[key] = line.polygon
    return lookup


def _select_orientation_rows(line_rows: list[LineMetricsRow], count: int) -> list[LineMetricsRow]:
    candidates = [row for row in line_rows if row.is_valid and row.orientation_direction_valid]
    candidates.sort(key=lambda row: abs(row.orientation_deg), reverse=True)
    picks: list[LineMetricsRow] = []
    high = candidates[: max(1, count // 3)]
    mid_start = len(candidates) // 3
    mid = candidates[mid_start : mid_start + max(1, count // 3)]
    low = [row for row in reversed(candidates) if abs(row.orientation_deg) < 2.0][: max(1, count // 3)]
    for group in (high, mid, low):
        for row in group:
            if row not in picks:
                picks.append(row)
            if len(picks) >= count:
                return picks
    return picks


def _select_pair_groups(
    pair_rows: list[TargetPairRow],
    *,
    count_each: int,
) -> dict[str, list[TargetPairRow]]:
    return {
        "ioa_positive": sorted(
            [row for row in pair_rows if row.ioa_positive],
            key=lambda row: row.ioa,
            reverse=True,
        )[:count_each],
        "horizontal_adjacent": sorted(
            [row for row in pair_rows if row.horizontal_adjacent],
            key=lambda row: row.horizontal_gap_px,
        )[:count_each],
    }


def _select_ink_rows(line_rows: list[LineMetricsRow], count: int) -> list[LineMetricsRow]:
    positives = [
        row
        for row in line_rows
        if row.is_valid
        and classify_bbox_outside_ink_state(row) == INK_STATE_POSITIVE_INK
        and row.bbox_outside_ink_ratio is not None
    ]
    positives.sort(key=lambda row: float(row.bbox_outside_ink_ratio or 0.0), reverse=True)
    return positives[:count]


def _min_area_box_and_axis(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[list[tuple[float, float]], tuple[float, float], tuple[float, float], float]:
    """Return OBB corners and the long-side axis from the min-area rect."""
    coords = np.asarray(polygon, dtype=np.float32)
    rect = cv2.minAreaRect(coords)
    box = [tuple(map(float, point)) for point in cv2.boxPoints(rect)]
    (cx, cy), (rw, rh), angle = rect
    # Long-side direction matches geometry._long_side_alpha_deg.
    if rw < rh:
        length = float(rh)
        axis_angle = math.radians(float(angle) + 90.0)
    else:
        length = float(rw)
        axis_angle = math.radians(float(angle))
    dx = math.cos(axis_angle) * length * 0.5
    dy = math.sin(axis_angle) * length * 0.5
    p1 = (float(cx) - dx, float(cy) - dy)
    p2 = (float(cx) + dx, float(cy) + dy)
    return box, p1, p2, float(cx)


def _crop_around(
    image: Image.Image,
    points: list[tuple[float, float]],
    *,
    pad: int = 40,
) -> tuple[Image.Image, tuple[int, int]]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = max(0, int(min(xs) - pad))
    top = max(0, int(min(ys) - pad))
    right = min(image.width, int(max(xs) + pad))
    bottom = min(image.height, int(max(ys) + pad))
    return image.crop((left, top, right, bottom)), (left, top)


def _shift(points: list[tuple[float, float]], origin: tuple[int, int]) -> list[tuple[float, float]]:
    ox, oy = origin
    return [(x - ox, y - oy) for x, y in points]


def export_orientation_axis_samples(
    line_rows: list[LineMetricsRow],
    polygon_lookup: dict[tuple[str, str], tuple[tuple[float, float], ...]],
    config: LineLevelConfig,
    samples_dir: Path,
    *,
    count: int = 6,
) -> dict[str, Path]:
    picks = [
        row
        for row in _select_orientation_rows(line_rows, count)
        if (row.image_id, row.line_id) in polygon_lookup
    ]

    out_dir = samples_dir / "orientation_axis"
    outputs: dict[str, Path] = {}
    for index, row in enumerate(picks, start=1):
        polygon = polygon_lookup[(row.image_id, row.line_id)]
        image_path = _resolve_image_path(config.input_root, row.image_id, config.image_extensions)
        if image_path is None or len(polygon) < 3:
            continue
        image = Image.open(image_path).convert("RGB")
        box, p1, p2, _cx = _min_area_box_and_axis(polygon)
        crop, origin = _crop_around(image, list(polygon) + box + [p1, p2])
        draw = ImageDraw.Draw(crop)
        draw.polygon(_shift(list(polygon), origin), outline=(220, 0, 0), width=3)
        draw.line(_shift([p1, p2], origin), fill=(0, 160, 0), width=3)
        label = f"{row.image_id} {row.line_id} orientation_deg={row.orientation_deg:.2f}"
        draw.rectangle((0, 0, crop.width, 24), fill=(255, 255, 220))
        draw.text((4, 4), label[:110], fill=(0, 0, 0))
        path = out_dir / f"{index:02d}_{row.line_id.replace(':', '_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path)
        outputs[f"orientation_axis/{path.name}"] = path
    return outputs


def export_spatial_relation_samples(
    pair_rows: list[TargetPairRow],
    polygon_lookup: dict[tuple[str, str], tuple[tuple[float, float], ...]],
    config: LineLevelConfig,
    samples_dir: Path,
    *,
    count_each: int = 4,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    groups = _select_pair_groups(pair_rows, count_each=count_each)
    for group_name, rows in groups.items():
        out_dir = samples_dir / "spatial_relations" / group_name
        for index, pair in enumerate(rows, start=1):
            poly_a = polygon_lookup.get((pair.image_id, pair.line_id_a))
            poly_b = polygon_lookup.get((pair.image_id, pair.line_id_b))
            image_path = _resolve_image_path(config.input_root, pair.image_id, config.image_extensions)
            if poly_a is None or poly_b is None or image_path is None:
                continue
            image = Image.open(image_path).convert("RGB")
            crop, origin = _crop_around(image, list(poly_a) + list(poly_b), pad=50)
            draw = ImageDraw.Draw(crop)
            draw.polygon(_shift(list(poly_a), origin), outline=(220, 0, 0), width=3)
            draw.polygon(_shift(list(poly_b), origin), outline=(0, 100, 220), width=3)
            if group_name == "ioa_positive":
                label = f"{pair.image_id} IoA={pair.ioa:.4f} {pair.line_id_a}/{pair.line_id_b}"
            else:
                label = (
                    f"{pair.image_id} adj Gx={pair.horizontal_gap_px:.1f} "
                    f"Sh={pair.height_similarity:.2f} Rv={pair.vertical_overlap_ratio:.2f}"
                )
            draw.rectangle((0, 0, crop.width, 24), fill=(255, 255, 220))
            draw.text((4, 4), label[:120], fill=(0, 0, 0))
            path = out_dir / f"{index:02d}_{pair.line_id_a.replace(':', '_')}_{pair.line_id_b.replace(':', '_')}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(path)
            outputs[f"spatial_relations/{group_name}/{path.name}"] = path
    return outputs


def export_bbox_outside_ink_samples(
    line_rows: list[LineMetricsRow],
    polygon_lookup: dict[tuple[str, str], tuple[tuple[float, float], ...]],
    config: LineLevelConfig,
    samples_dir: Path,
    *,
    count: int = 6,
) -> dict[str, Path]:
    calibration = config.calibration
    if calibration is None:
        if config.calibration_path is None or not config.calibration_path.is_file():
            return {}
        calibration = load_calibration_result(config.calibration_path)
    picks = [
        row for row in _select_ink_rows(line_rows, count) if (row.image_id, row.line_id) in polygon_lookup
    ]
    out_dir = samples_dir / "bbox_outside_ink"
    outputs: dict[str, Path] = {}
    for index, row in enumerate(picks, start=1):
        polygon = polygon_lookup[(row.image_id, row.line_id)]
        image_path = _resolve_image_path(config.input_root, row.image_id, config.image_extensions)
        if image_path is None or len(polygon) < 3:
            continue
        try:
            ink_mask = load_normalized_ink_mask(image_path, calibration)
        except OSError:
            continue
        shape = Polygon(polygon)
        stats = compute_bbox_outside_ink(ink_mask, shape)
        height, width = ink_mask.shape[:2]
        from benchmark_design.line_level.bbox_ink import _bbox_mask_from_line_mask, _raster_polygon

        line_mask = _raster_polygon(shape, height, width)
        bbox_mask = _bbox_mask_from_line_mask(line_mask)
        outside_ink = ink_mask & bbox_mask & ~line_mask

        image = Image.open(image_path).convert("RGB")
        overlay = np.asarray(image).copy()
        overlay[outside_ink] = (220, 40, 40)
        mask_pixels = line_mask
        overlay[mask_pixels] = (
            (0.7 * overlay[mask_pixels] + np.array([40, 160, 40])).clip(0, 255)
        ).astype(np.uint8)
        image = Image.fromarray(overlay)
        minx, miny, maxx, maxy = shape.bounds
        box = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        crop, origin = _crop_around(image, list(polygon) + box, pad=40)
        draw = ImageDraw.Draw(crop)
        draw.polygon(_shift(list(polygon), origin), outline=(0, 180, 0), width=2)
        draw.rectangle(
            [
                minx - origin[0],
                miny - origin[1],
                maxx - origin[0],
                maxy - origin[1],
            ],
            outline=(0, 100, 220),
            width=2,
        )
        label = (
            f"{row.image_id} {row.line_id} ratio={stats.bbox_outside_ink_ratio:.4f} "
            f"thr={calibration.global_threshold:g}"
        )
        draw.rectangle((0, 0, crop.width, 24), fill=(255, 255, 220))
        draw.text((4, 4), label[:120], fill=(0, 0, 0))
        path = out_dir / f"{index:02d}_{row.line_id.replace(':', '_')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path)
        outputs[f"bbox_outside_ink/{path.name}"] = path
    return outputs


def export_chapter_sample_figures(
    line_rows: list[LineMetricsRow],
    pair_rows: list[TargetPairRow],
    config: LineLevelConfig,
    samples_dir: Path,
) -> dict[str, Path]:
    orient_picks = _select_orientation_rows(line_rows, 6)
    pair_groups = _select_pair_groups(pair_rows, count_each=4)
    ink_picks = _select_ink_rows(line_rows, 6)
    needed: set[tuple[str, str]] = {(row.image_id, row.line_id) for row in orient_picks}
    needed.update((row.image_id, row.line_id) for row in ink_picks)
    for rows in pair_groups.values():
        for pair in rows:
            needed.add((pair.image_id, pair.line_id_a))
            needed.add((pair.image_id, pair.line_id_b))
    polygon_lookup = _load_polygons_for_ids(config, needed)
    outputs: dict[str, Path] = {}
    outputs.update(export_orientation_axis_samples(line_rows, polygon_lookup, config, samples_dir))
    outputs.update(export_spatial_relation_samples(pair_rows, polygon_lookup, config, samples_dir))
    outputs.update(export_bbox_outside_ink_samples(line_rows, polygon_lookup, config, samples_dir))
    return outputs
