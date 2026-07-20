"""Representative extreme sample selection for line-level geometry metrics."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_design.line_level.models import LineLevelConfig, LineMetricsRow


@dataclass(frozen=True, slots=True)
class LineSampleSelection:
    sample_type: str
    image_id: str
    line_id: str
    metric: str
    metric_value: float
    rank: str


SAMPLE_CATEGORIES: tuple[tuple[str, str, bool], ...] = (
    ("smallest_lines", "bbox_height_px", True),
    ("extreme_aspect_ratio", "aspect_ratio", False),
)


def _pick_extremes(
    rows: list[LineMetricsRow],
    *,
    metric: str,
    count: int,
    lowest: bool,
) -> list[LineSampleSelection]:
    candidates: list[LineMetricsRow] = []
    for row in rows:
        if not row.is_valid:
            continue
        value = getattr(row, metric)
        if value is None:
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: float(getattr(row, metric)), reverse=not lowest)
    selections: list[LineSampleSelection] = []
    for index, row in enumerate(candidates[:count]):
        selections.append(
            LineSampleSelection(
                sample_type=f"{metric}_{'low' if lowest else 'high'}",
                image_id=row.image_id,
                line_id=row.line_id,
                metric=metric,
                metric_value=float(getattr(row, metric)),
                rank=f"{'low' if lowest else 'high'}_{index + 1}",
            )
        )
    return selections


def select_all_line_samples(
    rows: list[LineMetricsRow],
    config: LineLevelConfig,
) -> list[LineSampleSelection]:
    selections: list[LineSampleSelection] = []
    count = config.extreme_sample_count
    for _category, metric, pick_low in SAMPLE_CATEGORIES:
        selections.extend(_pick_extremes(rows, metric=metric, count=count, lowest=pick_low))
        selections.extend(_pick_extremes(rows, metric=metric, count=count, lowest=not pick_low))
    return selections
