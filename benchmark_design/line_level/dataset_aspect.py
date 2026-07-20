"""Cross-dataset line width / height / aspect-ratio statistics (AABB)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from benchmark_design.progress import parallel_map


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif")


@dataclass(frozen=True, slots=True)
class DatasetLineGeometryRow:
    """One line (or one expression image treated as a single line).

    Horizontal/vertical sizes use the original image axes (no WH swap):
    - width_px = horizontal span
    - height_px = vertical span
    - aspect_ratio = width_px / height_px
    """

    dataset: str
    relative_path: str
    width_px: float
    height_px: float
    aspect_ratio: float


# Back-compat alias used by earlier call sites / tests.
DatasetImageSizeRow = DatasetLineGeometryRow


def list_dataset_roots(external_root: Path) -> list[tuple[str, Path]]:
    if not external_root.is_dir():
        return []
    return [(path.name, path) for path in sorted(external_root.iterdir()) if path.is_dir()]


def _iter_image_paths(dataset_root: Path, extensions: tuple[str, ...]) -> list[Path]:
    allowed = {ext.lower() for ext in extensions}
    paths: list[Path] = []
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            paths.append(path)
    paths.sort()
    return paths


def _read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def geometry_from_image_size(width: float, height: float) -> tuple[float, float, float]:
    """Keep original horizontal/vertical sizes; never swap or force aspect >= 1.

    Returns ``(width_px, height_px, aspect_ratio)`` with aspect = width/height.
    """
    width_px = float(width)
    height_px = float(height)
    if width_px <= 0 or height_px <= 0:
        return 0.0, 0.0, 0.0
    return width_px, height_px, width_px / height_px


def geometry_from_box(width: float, height: float) -> tuple[float, float, float]:
    """Alias for ``geometry_from_image_size`` (width, height, aspect)."""
    return geometry_from_image_size(width, height)


def _measure_one(
    args: tuple[str, Path, Path],
) -> DatasetLineGeometryRow | None:
    dataset, dataset_root, path = args
    size = _read_image_size(path)
    if size is None:
        return None
    image_width, image_height = size
    width_px, height_px, aspect = geometry_from_image_size(image_width, image_height)
    try:
        relative = path.relative_to(dataset_root).as_posix()
    except ValueError:
        relative = path.name
    return DatasetLineGeometryRow(
        dataset=dataset,
        relative_path=relative,
        width_px=width_px,
        height_px=height_px,
        aspect_ratio=aspect,
    )


def rows_from_line_metrics(
    line_rows: list,
    *,
    dataset: str = "ours",
) -> list[DatasetLineGeometryRow]:
    """Build geometry rows from valid line_level AABB metrics."""
    rows: list[DatasetLineGeometryRow] = []
    for line in line_rows:
        if not bool(getattr(line, "is_valid", False)):
            continue
        height = float(getattr(line, "bbox_height_px", 0.0) or 0.0)
        width = float(getattr(line, "bbox_width_px", 0.0) or 0.0)
        aspect = getattr(line, "aspect_ratio", None)
        if height <= 0 or width <= 0:
            continue
        if aspect is None:
            aspect = width / height
        image_id = str(getattr(line, "image_id", ""))
        line_id = str(getattr(line, "line_id", ""))
        rows.append(
            DatasetLineGeometryRow(
                dataset=dataset,
                relative_path=f"{image_id}:{line_id}",
                width_px=width,
                height_px=height,
                aspect_ratio=float(aspect),
            )
        )
    return rows


def rows_from_page_metrics(
    page_rows: list,
    *,
    dataset: str = "ours",
) -> list[DatasetLineGeometryRow]:
    """Deprecated page-level path; prefer rows_from_line_metrics."""
    rows: list[DatasetLineGeometryRow] = []
    for page in page_rows:
        width = float(getattr(page, "width", 0) or 0)
        height = float(getattr(page, "height", 0) or 0)
        if width <= 0 or height <= 0:
            continue
        width_px, height_px, aspect = geometry_from_image_size(width, height)
        rows.append(
            DatasetLineGeometryRow(
                dataset=dataset,
                relative_path=str(getattr(page, "image_id", "")),
                width_px=width_px,
                height_px=height_px,
                aspect_ratio=aspect,
            )
        )
    return rows


def collect_dataset_image_sizes(
    external_root: Path,
    *,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    workers: int | None = None,
    show_progress: bool = True,
    ours_rows: list[DatasetLineGeometryRow] | None = None,
) -> list[DatasetLineGeometryRow]:
    """Collect line geometry for each external expression image (+ optional ours lines)."""
    tasks: list[tuple[str, Path, Path]] = []
    for dataset, root in list_dataset_roots(external_root):
        for path in _iter_image_paths(root, extensions):
            tasks.append((dataset, root, path))
    if not tasks:
        rows: list[DatasetLineGeometryRow] = []
    elif workers is not None and workers <= 1:
        rows = [row for row in (_measure_one(task) for task in tasks) if row is not None]
    else:
        measured = parallel_map(
            _measure_one,
            tasks,
            description="Measuring external dataset line geometry",
            show_progress=show_progress,
            workers=workers,
        )
        rows = [row for row in measured if row is not None]

    if ours_rows:
        rows = [row for row in rows if row.dataset != "ours"]
        rows = list(ours_rows) + rows
    return rows


def dataset_size_summary_frame(rows: list[DatasetLineGeometryRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "line_count",
                "height_px_median",
                "height_px_p05",
                "height_px_p95",
                "width_px_median",
                "width_px_p05",
                "width_px_p95",
                "aspect_ratio_median",
                "aspect_ratio_p05",
                "aspect_ratio_p95",
            ]
        )
    frame = pd.DataFrame([asdict(row) for row in rows])
    summaries: list[dict[str, float | int | str]] = []
    for dataset, group in frame.groupby("dataset", sort=True):
        heights = group["height_px"].to_numpy(dtype=np.float64)
        widths = group["width_px"].to_numpy(dtype=np.float64)
        ratios = group["aspect_ratio"].to_numpy(dtype=np.float64)
        summaries.append(
            {
                "dataset": str(dataset),
                "line_count": int(len(group)),
                "height_px_median": float(np.median(heights)),
                "height_px_p05": float(np.quantile(heights, 0.05)),
                "height_px_p95": float(np.quantile(heights, 0.95)),
                "width_px_median": float(np.median(widths)),
                "width_px_p05": float(np.quantile(widths, 0.05)),
                "width_px_p95": float(np.quantile(widths, 0.95)),
                "aspect_ratio_median": float(np.median(ratios)),
                "aspect_ratio_p05": float(np.quantile(ratios, 0.05)),
                "aspect_ratio_p95": float(np.quantile(ratios, 0.95)),
            }
        )
    return pd.DataFrame(summaries)


def validate_external_geometry_rows(rows: list[DatasetLineGeometryRow]) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not np.isfinite(row.width_px) or row.width_px <= 0:
            errors.append(f"row {index}: width_px invalid")
        if not np.isfinite(row.height_px) or row.height_px <= 0:
            errors.append(f"row {index}: height_px invalid")
        expected = row.width_px / row.height_px if row.height_px > 0 else float("nan")
        if not np.isfinite(row.aspect_ratio) or abs(row.aspect_ratio - expected) > 1e-6:
            errors.append(f"row {index}: aspect_ratio mismatch")
        if len(errors) >= 20:
            errors.append("... truncated")
            break
    return errors
