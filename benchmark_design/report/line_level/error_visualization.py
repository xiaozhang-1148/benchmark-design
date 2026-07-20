"""Visualize line polygon validation failures for manual inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from benchmark_design.io.image import load_grayscale_image
from benchmark_design.line_level.loader import discover_pages_from_benchmark
from benchmark_design.line_level.models import InvalidAnnotationRow, LineAnnotation, LineLevelConfig, PageTask
from benchmark_design.line_level.validation import validate_line_polygon
from benchmark_design.progress import parallel_map
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.report.pyplot_lock import with_locked_pyplot

FIGURE_DPI = 150
ZOOM_PADDING_PX = 40

REASON_LABELS: dict[str, str] = {
    "insufficient_points": "点数不足（少于3个点）",
    "non_finite_coordinates": "坐标非有限值",
    "zero_area": "面积为0",
    "out_of_bounds": "超出图像边界",
    "duplicate_line_id": "重复line_id",
}

REASON_COLORS: dict[str, str] = {
    "insufficient_points": "#ED7D31",
    "non_finite_coordinates": "#7030A0",
    "zero_area": "#7F7F7F",
    "out_of_bounds": "#C00000",
    "duplicate_line_id": "#00B050",
}


@dataclass(frozen=True, slots=True)
class LineValidationErrorCase:
    reason: str
    page: PageTask
    line: LineAnnotation
    related_lines: tuple[LineAnnotation, ...] = ()
    polygon_point_count: int = 0
    detail: str = ""


def _safe_filename(image_id: str, line_id: str) -> str:
    return f"{image_id}__{line_id.replace(':', '-')}.png"


def _ocr_preview(ocr: str, *, max_len: int = 48) -> str:
    text = ocr.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def collect_line_validation_errors(pages: list[PageTask]) -> list[LineValidationErrorCase]:
    cases: list[LineValidationErrorCase] = []
    for page in pages:
        if page.image_path is None or not page.image_path.is_file():
            continue
        try:
            gray = load_grayscale_image(page.image_path)
        except OSError:
            continue
        page_height, page_width = gray.shape[:2]

        first_by_id: dict[str, LineAnnotation] = {}
        for line in page.lines:
            if line.line_id in first_by_id:
                cases.append(
                    LineValidationErrorCase(
                        reason="duplicate_line_id",
                        page=page,
                        line=line,
                        related_lines=(first_by_id[line.line_id],),
                        polygon_point_count=len(line.polygon),
                        detail=f"与先出现的 line_id={line.line_id} 重复",
                    )
                )
                continue
            first_by_id[line.line_id] = line

            ok, reason, _shape = validate_line_polygon(
                line,
                image_width=page_width,
                image_height=page_height,
            )
            if ok:
                continue
            cases.append(
                LineValidationErrorCase(
                    reason=reason or "unknown",
                    page=page,
                    line=line,
                    polygon_point_count=len(line.polygon),
                )
            )
    return cases


def _polygon_xy(polygon: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    if not polygon:
        return np.array([]), np.array([])
    xs = np.array([point[0] for point in polygon], dtype=np.float64)
    ys = np.array([point[1] for point in polygon], dtype=np.float64)
    return xs, ys


def _closed_polygon_xy(polygon: tuple[tuple[float, float], ...]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = _polygon_xy(polygon)
    if xs.size == 0:
        return xs, ys
    return np.append(xs, xs[0]), np.append(ys, ys[0])


def _zoom_bounds(
    polygons: tuple[tuple[tuple[float, float], ...], ...],
    *,
    image_width: int,
    image_height: int,
    padding: int = ZOOM_PADDING_PX,
) -> tuple[int, int, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    for polygon in polygons:
        for x, y in polygon:
            xs.append(x)
            ys.append(y)
    if not xs:
        return 0, 0, image_width, image_height
    x0 = max(0, int(np.floor(min(xs) - padding)))
    y0 = max(0, int(np.floor(min(ys) - padding)))
    x1 = min(image_width, int(np.ceil(max(xs) + padding)))
    y1 = min(image_height, int(np.ceil(max(ys) + padding)))
    if x1 <= x0:
        x1 = min(image_width, x0 + 1)
    if y1 <= y0:
        y1 = min(image_height, y0 + 1)
    return x0, y0, x1, y1


def _draw_polygon(
    ax,
    polygon: tuple[tuple[float, float], ...],
    *,
    edge_color: str,
    face_color: str | None = None,
    label: str | None = None,
    linewidth: float = 2.0,
    show_vertices: bool = False,
) -> None:
    xs, ys = _closed_polygon_xy(polygon)
    if xs.size == 0:
        return
    ax.plot(xs, ys, color=edge_color, linewidth=linewidth, label=label)
    if face_color is not None:
        ax.fill(xs, ys, color=face_color, alpha=0.18)
    if show_vertices:
        vx, vy = _polygon_xy(polygon)
        ax.scatter(vx, vy, s=28, c=edge_color, zorder=4)
        for index, (x, y) in enumerate(zip(vx, vy, strict=True)):
            ax.text(x, y, str(index), color="yellow", fontsize=7, ha="center", va="center")


@with_locked_pyplot
def _save_error_figure(case: LineValidationErrorCase, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _configure_matplotlib_fonts(plt)
    page = case.page
    if page.image_path is None:
        return
    gray = load_grayscale_image(page.image_path)
    page_height, page_width = gray.shape[:2]
    reason_label = REASON_LABELS.get(case.reason, case.reason)
    color = REASON_COLORS.get(case.reason, "#C00000")

    polygons = (case.line.polygon,)
    if case.reason == "duplicate_line_id":
        polygons = tuple(related.polygon for related in case.related_lines) + (case.line.polygon,)

    x0, y0, x1, y1 = _zoom_bounds(polygons, image_width=page_width, image_height=page_height)
    zoom = gray[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=FIGURE_DPI)
    for ax, image, title in (
        (axes[0], gray, "整页视图"),
        (axes[1], zoom, "局部放大"),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        if ax is axes[0] and case.reason == "out_of_bounds":
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    page_width,
                    page_height,
                    linewidth=1.5,
                    edgecolor="#00B050",
                    facecolor="none",
                    linestyle="--",
                    label="图像边界",
                )
            )

    if case.reason == "duplicate_line_id":
        for related in case.related_lines:
            _draw_polygon(
                axes[0],
                related.polygon,
                edge_color="#00B050",
                face_color="#00B050",
                label=f"先出现 {related.line_id}",
                show_vertices=True,
            )
            _draw_polygon(
                axes[1],
                related.polygon,
                edge_color="#00B050",
                face_color="#00B050",
                show_vertices=True,
            )
        _draw_polygon(
            axes[0],
            case.line.polygon,
            edge_color="#C00000",
            face_color="#C00000",
            label=f"重复 {case.line.line_id}",
            show_vertices=True,
        )
        _draw_polygon(
            axes[1],
            case.line.polygon,
            edge_color="#C00000",
            face_color="#C00000",
            show_vertices=True,
        )
    elif case.reason == "insufficient_points":
        vx, vy = _polygon_xy(case.line.polygon)
        for ax in axes:
            ax.scatter(vx, vy, s=60, c=color, label="标注点")
            for index, (x, y) in enumerate(zip(vx, vy, strict=True)):
                ax.text(x, y, str(index), color="white", fontsize=8, ha="center", va="center")
    else:
        _draw_polygon(
            axes[0],
            case.line.polygon,
            edge_color=color,
            face_color=color,
            label="问题 line",
        )
        _draw_polygon(
            axes[1],
            case.line.polygon,
            edge_color=color,
            face_color=color,
        )

    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
        ax.set_xticks([])
        ax.set_yticks([])

    title_lines = [
        reason_label,
        f"image_id={case.line.image_id}  line_id={case.line.line_id}  block={case.line.block_type}",
        f"点数={case.polygon_point_count}  OCR={_ocr_preview(case.line.ocr)}",
    ]
    if case.detail:
        title_lines.append(case.detail)
    fig.suptitle("\n".join(title_lines), fontsize=10, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_case(case: LineValidationErrorCase, output_root: Path) -> dict[str, str]:
    reason_dir = output_root / case.reason
    filename = _safe_filename(case.line.image_id, case.line.line_id)
    output_path = reason_dir / filename
    _save_error_figure(case, output_path)
    return {
        "reason": case.reason,
        "reason_label": REASON_LABELS.get(case.reason, case.reason),
        "image_id": case.line.image_id,
        "line_id": case.line.line_id,
        "block_type": case.line.block_type,
        "polygon_point_count": str(case.polygon_point_count),
        "ocr_preview": _ocr_preview(case.line.ocr),
        "image_path": str(case.page.image_path) if case.page.image_path else "",
        "json_path": str(case.page.json_path),
        "figure_path": output_path.relative_to(output_root).as_posix(),
        "detail": case.detail,
    }


def export_line_validation_error_figures(
    config: LineLevelConfig,
    output_root: Path,
) -> dict[str, object]:
    """Discover validation failures and write per-reason visualizations."""
    output_root.mkdir(parents=True, exist_ok=True)
    for reason in REASON_LABELS:
        (output_root / reason).mkdir(parents=True, exist_ok=True)

    pages = discover_pages_from_benchmark(config)
    cases = collect_line_validation_errors(pages)

    index_rows: list[dict[str, str]] = []
    if cases:
        if config.workers is not None and config.workers <= 1:
            index_rows = [_render_case(case, output_root) for case in cases]
        else:
            index_rows = parallel_map(
                lambda case: _render_case(case, output_root),
                cases,
                description="Rendering line validation error figures",
                show_progress=config.show_progress,
                workers=config.workers,
            )

    counts = {reason: 0 for reason in REASON_LABELS}
    for case in cases:
        counts[case.reason] = counts.get(case.reason, 0) + 1

    summary = {
        "total_error_count": len(cases),
        "counts_by_reason": counts,
        "reason_labels": REASON_LABELS,
        "output_root": str(output_root.resolve()),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if index_rows:
        frame = pd.DataFrame(index_rows)
        frame.to_csv(output_root / "error_index.csv", index=False)

    return summary
