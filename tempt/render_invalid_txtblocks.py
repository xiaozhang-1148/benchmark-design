"""Render annotation overlays for Txtblocks with empty block polygons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from benchmark_design.report.export_figures import _configure_matplotlib_fonts

INPUT_DIR = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark")
OUTPUT_DIR = Path("/home/baoquan/ocr-process/benchmark-design/tempt/block")

INVALID_BLOCKS: list[tuple[str, int]] = [
    ("mamatwrik22198rwdlg.jpg", 0),
    ("mamatwrik22198rwdlg.jpg", 1),
    ("mamazs85c60l4l8ndvn.jpg", 0),
    ("mambdy7izqgcxesn8pm.jpg", 0),
    ("mambjn816wy7rbounc4.jpg", 1),
]


def _polygon_points(raw) -> list[tuple[float, float]]:
    if not raw:
        return []
    return [(float(x), float(y)) for x, y in raw]


def _block_center(block: dict) -> tuple[float, float]:
    polygon = _polygon_points(block.get("polygon"))
    if len(polygon) >= 3:
        xs, ys = zip(*polygon)
        return sum(xs) / len(xs), sum(ys) / len(ys)
    line_centers: list[tuple[float, float]] = []
    for line in block.get("lines") or []:
        pts = _polygon_points(line.get("polygon"))
        if len(pts) >= 3:
            xs, ys = zip(*pts)
            line_centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    if line_centers:
        xs, ys = zip(*line_centers)
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.0, 0.0


def _draw_block_polygon(axis, block: dict, *, facecolor: str, edgecolor: str, alpha: float, linewidth: float, linestyle: str) -> None:
    points = _polygon_points(block.get("polygon"))
    if len(points) < 3:
        return
    patch = MplPolygon(
        points,
        closed=True,
        fill=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    axis.add_patch(patch)


def _draw_line_polygons(axis, block: dict, *, facecolor: str, edgecolor: str, alpha: float, linewidth: float, linestyle: str) -> None:
    for line in block.get("lines") or []:
        points = _polygon_points(line.get("polygon"))
        if len(points) < 3:
            continue
        patch = MplPolygon(
            points,
            closed=True,
            fill=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        axis.add_patch(patch)


def _ocr_preview(block: dict, limit: int = 4) -> str:
    lines = [str(line.get("ocr") or "").strip() for line in block.get("lines") or []]
    lines = [line for line in lines if line]
    if not lines:
        return "(no OCR text)"
    preview = lines[:limit]
    suffix = f"\n... (+{len(lines) - limit} more lines)" if len(lines) > limit else ""
    return "\n".join(preview) + suffix


def render_invalid_block(page_id: str, block_order: int) -> Path:
    _configure_matplotlib_fonts(plt)
    json_path = INPUT_DIR / f"{page_id}.json"
    image_path = INPUT_DIR / page_id
    page = json.loads(json_path.read_text(encoding="utf-8"))
    target = next(block for block in page["blocks"] if int(block.get("order", -1)) == block_order)

    fig, axis = plt.subplots(figsize=(12, 16))
    axis.imshow(plt.imread(str(image_path)))

    for block in page["blocks"]:
        order = int(block.get("order", -1))
        block_type = str(block.get("type") or "")
        if order == block_order:
            continue
        if block_type == "Txtblock":
            _draw_block_polygon(
                axis,
                block,
                facecolor="#3498DB",
                edgecolor="#2C3E50",
                alpha=0.18,
                linewidth=1.2,
                linestyle="-",
            )
        elif len(_polygon_points(block.get("polygon"))) >= 3:
            _draw_block_polygon(
                axis,
                block,
                facecolor="#95A5A6",
                edgecolor="#7F8C8D",
                alpha=0.12,
                linewidth=1.0,
                linestyle="-",
            )

    _draw_line_polygons(
        axis,
        target,
        facecolor="#E74C3C",
        edgecolor="#C0392B",
        alpha=0.45,
        linewidth=2.0,
        linestyle="--",
    )

    center_x, center_y = _block_center(target)
    axis.text(
        center_x,
        center_y,
        f"INVALID\nblock_{block_order}",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
        bbox={"facecolor": "#C0392B", "alpha": 0.92, "pad": 3, "edgecolor": "white", "linewidth": 1.2},
    )

    line_count = len(target.get("lines") or [])
    block_poly_count = len(_polygon_points(target.get("polygon")))
    title = (
        f"{page_id}  |  Txtblock block_{block_order}\n"
        f"block polygon vertices: {block_poly_count}  |  expression lines: {line_count}\n"
        "Excluded from D_block: empty block polygon (line polygons shown in red dashed)"
    )
    axis.set_title(title, fontsize=10, loc="left")
    axis.text(
        0.01,
        0.01,
        _ocr_preview(target),
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.72, "pad": 4},
    )
    axis.axis("off")

    stem = page_id.replace(".jpg", "")
    output_path = OUTPUT_DIR / f"{stem}_block_{block_order}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    written: list[Path] = []
    for page_id, block_order in INVALID_BLOCKS:
        written.append(render_invalid_block(page_id, block_order))
    print(f"Wrote {len(written)} figures to {OUTPUT_DIR}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
