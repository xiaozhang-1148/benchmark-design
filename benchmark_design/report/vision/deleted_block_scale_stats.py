"""Continuous R_del statistics for Deleted-Block Scale exports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from benchmark_design.vision.deleted_block_scale.models import PageDeletedBlockScaleResult

R_DEL_TAIL_CUTOFFS: tuple[float, ...] = (0.2, 0.3, 0.5)
HIGH_R_DEL_EXAMPLE_COUNT = 40

R_DEL_BAND_SPECS: tuple[tuple[str, float | None, float | None], ...] = (
    ("0-0.1", None, 0.1),
    ("0.1-0.2", 0.1, 0.2),
    ("0.2-0.3", 0.2, 0.3),
    ("0.3-0.5", 0.3, 0.5),
    (">=0.5", 0.5, None),
)


@dataclass(frozen=True, slots=True)
class RDelBand:
    range_label: str
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class RDelBandSummary:
    total_pages: int
    bands: tuple[RDelBand, ...]


@dataclass(frozen=True, slots=True)
class DeletedInstanceBand:
    instance_count: int
    page_count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class DeletedBlockScaleSummaryStats:
    pages_analyzed: int
    pages_with_deleted: int
    pages_without_deleted: int
    deleted_block_prevalence: float
    total_deleted_instances: int
    mean_deleted_count_affected: float
    max_deleted_count_affected: int
    total_deleted_area: int
    total_answer_related_area: int
    dataset_level_deleted_area_ratio: float
    mean_r_del: float
    max_r_del: float
    pages_r_del_ge_0_2: int
    pages_r_del_ge_0_3: int
    pages_r_del_ge_0_5: int
    manual_review_pages: int
    review_counts: Counter[str]


def _r_del_in_band(value: float, *, lower: float | None, upper: float | None) -> bool:
    if lower is not None and value < lower:
        return False
    if upper is not None and value >= upper:
        return False
    return True


def compute_r_del_bands(values: list[float]) -> RDelBandSummary:
    total = len(values)
    bands: list[RDelBand] = []
    for range_label, lower, upper in R_DEL_BAND_SPECS:
        count = sum(1 for value in values if _r_del_in_band(value, lower=lower, upper=upper))
        ratio = (count / total) if total else 0.0
        bands.append(RDelBand(range_label=range_label, count=count, ratio=ratio))
    return RDelBandSummary(total_pages=total, bands=tuple(bands))


def compute_deleted_instance_bands(
    results: list[PageDeletedBlockScaleResult],
) -> tuple[DeletedInstanceBand, ...]:
    affected = [result for result in results if result.has_deleted_text_block]
    total = len(affected)
    max_count = max((result.num_deleted_text_block for result in affected), default=0)
    rows: list[DeletedInstanceBand] = []
    for instance_count in range(1, max(max_count, 4) + 1):
        page_count = sum(
            1 for result in affected if result.num_deleted_text_block == instance_count
        )
        if page_count == 0 and instance_count > max_count:
            continue
        ratio = (page_count / total) if total else 0.0
        rows.append(
            DeletedInstanceBand(
                instance_count=instance_count,
                page_count=page_count,
                ratio=ratio,
            )
        )
    return tuple(rows)


def _affected_r_del_values(results: list[PageDeletedBlockScaleResult]) -> list[float]:
    return [
        result.r_del
        for result in results
        if result.has_deleted_text_block and result.r_del is not None
    ]


def compute_deleted_block_scale_summary_stats(
    results: list[PageDeletedBlockScaleResult],
) -> DeletedBlockScaleSummaryStats:
    total = len(results)
    pages_with_deleted = sum(1 for result in results if result.has_deleted_text_block)
    pages_without_deleted = total - pages_with_deleted
    prevalence = (pages_with_deleted / total) if total else 0.0
    total_instances = sum(result.num_deleted_text_block for result in results)

    affected = [result for result in results if result.has_deleted_text_block]
    deleted_counts = [result.num_deleted_text_block for result in affected]
    mean_deleted_count = sum(deleted_counts) / len(deleted_counts) if deleted_counts else 0.0
    max_deleted_count = max(deleted_counts) if deleted_counts else 0

    total_deleted_area = sum(result.deleted_area for result in results)
    total_answer_related_area = sum(result.answer_related_area for result in results)
    dataset_ratio = (
        total_deleted_area / total_answer_related_area if total_answer_related_area else 0.0
    )

    r_del_values = _affected_r_del_values(results)
    mean_r_del = sum(r_del_values) / len(r_del_values) if r_del_values else 0.0
    max_r_del = max(r_del_values) if r_del_values else 0.0

    pages_r_del_ge_0_2 = sum(1 for value in r_del_values if value >= 0.2)
    pages_r_del_ge_0_3 = sum(1 for value in r_del_values if value >= 0.3)
    pages_r_del_ge_0_5 = sum(1 for value in r_del_values if value >= 0.5)

    review_counts: Counter[str] = Counter()
    for result in results:
        for reason in result.review_reason.split(";"):
            if reason:
                review_counts[reason] += 1

    return DeletedBlockScaleSummaryStats(
        pages_analyzed=total,
        pages_with_deleted=pages_with_deleted,
        pages_without_deleted=pages_without_deleted,
        deleted_block_prevalence=prevalence,
        total_deleted_instances=total_instances,
        mean_deleted_count_affected=mean_deleted_count,
        max_deleted_count_affected=max_deleted_count,
        total_deleted_area=total_deleted_area,
        total_answer_related_area=total_answer_related_area,
        dataset_level_deleted_area_ratio=dataset_ratio,
        mean_r_del=mean_r_del,
        max_r_del=max_r_del,
        pages_r_del_ge_0_2=pages_r_del_ge_0_2,
        pages_r_del_ge_0_3=pages_r_del_ge_0_3,
        pages_r_del_ge_0_5=pages_r_del_ge_0_5,
        manual_review_pages=sum(1 for result in results if result.needs_manual_review),
        review_counts=review_counts,
    )


def top_r_del_results(
    results: list[PageDeletedBlockScaleResult],
    *,
    top_k: int = HIGH_R_DEL_EXAMPLE_COUNT,
) -> list[PageDeletedBlockScaleResult]:
    candidates = [
        result
        for result in results
        if result.has_deleted_text_block and result.r_del is not None
    ]
    return sorted(candidates, key=lambda result: result.r_del or 0.0, reverse=True)[:top_k]
