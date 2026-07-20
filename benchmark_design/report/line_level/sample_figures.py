"""Sample overlay visualizations for line-level geometry analysis."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from benchmark_design.line_level.loader import discover_pages_from_benchmark
from benchmark_design.line_level.models import LineLevelConfig, LineMetricsRow
from benchmark_design.line_level.sample_selection import LineSampleSelection


def _load_page_lines_by_id(config: LineLevelConfig) -> dict[tuple[str, str], tuple[tuple[float, float], ...]]:
    lookup: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {}
    for page in discover_pages_from_benchmark(config):
        for line in page.lines:
            lookup[(line.image_id, line.line_id)] = line.polygon
    return lookup


def _draw_overlay(
    image_path: Path,
    polygon: tuple[tuple[float, float], ...],
    *,
    label: str,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if len(polygon) >= 3:
        draw.polygon(polygon, outline=(220, 0, 0), width=3)
    draw.rectangle((0, 0, image.width, 28), fill=(255, 255, 220))
    draw.text((6, 6), label[:120], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def export_line_level_samples(
    selections: list[LineSampleSelection],
    line_rows: list[LineMetricsRow],
    config: LineLevelConfig,
    samples_dir: Path,
    input_dir: Path,
) -> dict[str, Path]:
    _ = line_rows
    polygon_lookup = _load_page_lines_by_id(config)
    outputs: dict[str, Path] = {}

    categories = {
        "smallest_lines": ("bbox_height_px", True),
        "extreme_aspect_ratio": ("aspect_ratio", False),
    }

    for category, (metric, pick_low) in categories.items():
        subset = [s for s in selections if s.metric == metric and ("low" in s.rank) == pick_low]
        category_dir = samples_dir / category
        for selection in subset[: config.extreme_sample_count]:
            polygon = polygon_lookup.get((selection.image_id, selection.line_id), ())
            if len(polygon) < 3:
                continue
            image_path = None
            for ext in config.image_extensions:
                candidate = input_dir / f"{selection.image_id}{ext}"
                if candidate.is_file():
                    image_path = candidate
                    break
            if image_path is None:
                continue
            label = f"{selection.image_id} {selection.line_id} {metric}={selection.metric_value:.4g}"
            out = category_dir / f"{selection.rank}_{selection.line_id.replace(':', '_')}.png"
            _draw_overlay(
                image_path,
                polygon,
                label=label,
                output_path=out,
            )
            outputs[f"{category}/{out.name}"] = out

    return outputs
