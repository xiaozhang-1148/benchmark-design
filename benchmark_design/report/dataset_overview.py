"""Dataset overview statistics for HMER and block-level benchmarks."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.io.benchmark_loader import iter_benchmark_json_paths, load_expressions
from benchmark_design.ocr.length_distribution import percentile
from benchmark_design.ocr.processing import EnrichedCorpus
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.block_level.dataset import load_block_level_benchmark_dataset
from benchmark_design.block_level.flow_structure.block_roles import (
    is_deleted_text_block,
    is_txt_block,
    normalize_block_type,
)
from benchmark_design.block_level.flow_structure.models import PageAnnotation
from benchmark_design.block_level.processing_options import VisionProcessingOptions
from benchmark_design.block_level.sample_record import ImageSampleRecord


@dataclass(frozen=True, slots=True)
class HmerOverviewMetrics:
    page_count: int
    expression_count: int
    total_characters: int
    avg_expressions_per_page: float
    expressions_per_page: tuple[int, ...]
    page_ids: tuple[str, ...]

    @property
    def avg_chars_per_expression(self) -> float:
        if self.expression_count == 0:
            return 0.0
        return self.total_characters / self.expression_count

    @property
    def max_expressions_per_page(self) -> int:
        if not self.expressions_per_page:
            return 0
        return max(self.expressions_per_page)


@dataclass(frozen=True, slots=True)
class BlockOverviewMetrics:
    txtblock_count: int
    deleted_text_block_count: int
    chart_count: int
    figure_count: int
    total_block_count: int

    @property
    def other_block_count(self) -> int:
        return self.total_block_count - (
            self.txtblock_count
            + self.deleted_text_block_count
            + self.chart_count
            + self.figure_count
        )


@dataclass(frozen=True, slots=True)
class VisionOverviewMetrics:
    page_count: int
    sample_ids: tuple[str, ...]
    aspect_ratios: tuple[float, ...]
    widths: tuple[int, ...]
    heights: tuple[int, ...]
    megapixels: tuple[float, ...]

    @property
    def avg_aspect_ratio(self) -> float:
        if not self.aspect_ratios:
            return 0.0
        return sum(self.aspect_ratios) / len(self.aspect_ratios)

    @property
    def avg_megapixels(self) -> float:
        if not self.megapixels:
            return 0.0
        return sum(self.megapixels) / len(self.megapixels)

    @property
    def median_aspect_ratio(self) -> float:
        return percentile(self.aspect_ratios, 50)

    @property
    def portrait_count(self) -> int:
        return sum(1 for ratio in self.aspect_ratios if ratio < 1.0)

    @property
    def landscape_count(self) -> int:
        return sum(1 for ratio in self.aspect_ratios if ratio > 1.0)

    def orientation_share(self, count: int) -> float:
        total = len(self.aspect_ratios)
        if total == 0:
            return 0.0
        return count / total * 100.0


@dataclass(frozen=True, slots=True)
class DatasetOverviewMetrics:
    hmer: HmerOverviewMetrics
    vision: VisionOverviewMetrics
    block: BlockOverviewMetrics


def compute_hmer_overview_from_enriched(enriched: EnrichedCorpus) -> HmerOverviewMetrics:
    json_paths = iter_benchmark_json_paths(enriched.input_dir)
    page_count = len(json_paths)
    page_ids = [path.stem for path in json_paths]
    per_page = [0] * page_count
    path_to_index = {path.resolve(): index for index, path in enumerate(json_paths)}
    expressions = enriched.expressions
    total_characters = sum(len(record.ocr) for record in expressions)
    expression_count = len(expressions)
    for record in expressions:
        index = path_to_index.get(Path(record.source_file).resolve())
        if index is None:
            continue
        per_page[index] += 1
    avg_expressions = expression_count / page_count if page_count else 0.0
    return HmerOverviewMetrics(
        page_count=page_count,
        expression_count=expression_count,
        total_characters=total_characters,
        avg_expressions_per_page=avg_expressions,
        expressions_per_page=tuple(per_page),
        page_ids=tuple(page_ids),
    )


def compute_block_overview_from_pages(pages: Sequence[PageAnnotation]) -> BlockOverviewMetrics:
    txtblock_count = 0
    deleted_text_block_count = 0
    chart_count = 0
    figure_count = 0
    total_block_count = 0
    for page in pages:
        for block in page.blocks:
            total_block_count += 1
            block_type = block.block_type
            if is_txt_block(block_type):
                txtblock_count += 1
            elif is_deleted_text_block(block_type):
                deleted_text_block_count += 1
            elif normalize_block_type(block_type) == "chart":
                chart_count += 1
            elif normalize_block_type(block_type) == "figure":
                figure_count += 1
    return BlockOverviewMetrics(
        txtblock_count=txtblock_count,
        deleted_text_block_count=deleted_text_block_count,
        chart_count=chart_count,
        figure_count=figure_count,
        total_block_count=total_block_count,
    )


def compute_block_counts_from_pages(pages: Sequence[PageAnnotation]) -> tuple[int, int]:
    block = compute_block_overview_from_pages(pages)
    return block.txtblock_count, block.total_block_count


def compute_vision_overview_from_samples(
    samples: Sequence[ImageSampleRecord],
) -> VisionOverviewMetrics:
    aspect_ratios: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    megapixels: list[float] = []
    sample_ids: list[str] = []
    for sample in samples:
        width = sample.width_px
        height = sample.height_px
        if width is None or height is None or width <= 0 or height <= 0:
            continue
        sample_ids.append(sample.sample_id)
        widths.append(width)
        heights.append(height)
        aspect_ratios.append(width / height)
        megapixels.append((width * height) / 1_000_000.0)
    return VisionOverviewMetrics(
        page_count=len(samples),
        sample_ids=tuple(sample_ids),
        aspect_ratios=tuple(aspect_ratios),
        widths=tuple(widths),
        heights=tuple(heights),
        megapixels=tuple(megapixels),
    )


def compute_hmer_overview(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
) -> HmerOverviewMetrics:
    _ = processing or ProcessingOptions()
    json_paths = iter_benchmark_json_paths(input_dir)
    page_count = len(json_paths)
    expressions = load_expressions(
        input_dir,
        json_paths=json_paths,
        show_progress=False,
    )
    total_characters = sum(len(record.ocr) for record in expressions)
    expression_count = len(expressions)
    page_ids = [path.stem for path in json_paths]
    per_page = [0] * page_count
    path_to_index = {path.resolve(): index for index, path in enumerate(json_paths)}
    for record in expressions:
        index = path_to_index.get(Path(record.source_file).resolve())
        if index is None:
            continue
        per_page[index] += 1
    avg_expressions = expression_count / page_count if page_count else 0.0
    return HmerOverviewMetrics(
        page_count=page_count,
        expression_count=expression_count,
        total_characters=total_characters,
        avg_expressions_per_page=avg_expressions,
        expressions_per_page=tuple(per_page),
        page_ids=tuple(page_ids),
    )


def _resolve_page_annotations(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions,
    pages: Sequence[PageAnnotation] | None = None,
) -> list[PageAnnotation]:
    if pages is not None:
        return list(pages)
    dataset = load_block_level_benchmark_dataset(input_dir, processing=processing)
    return list(dataset.pages)


def compute_vision_overview(
    input_dir: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    samples: Sequence[ImageSampleRecord] | None = None,
) -> VisionOverviewMetrics:
    processing = processing or VisionProcessingOptions()
    if samples is not None:
        return compute_vision_overview_from_samples(samples)
    dataset = load_block_level_benchmark_dataset(input_dir, processing=processing)
    return compute_vision_overview_from_samples(dataset.samples)


def compute_dataset_overview(
    input_dir: Path,
    *,
    processing: ProcessingOptions | None = None,
    vision_processing: VisionProcessingOptions | None = None,
    enriched: EnrichedCorpus | None = None,
    vision_samples: Sequence[ImageSampleRecord] | None = None,
    vision_pages: Sequence[PageAnnotation] | None = None,
) -> DatasetOverviewMetrics:
    hmer = (
        compute_hmer_overview_from_enriched(enriched)
        if enriched is not None
        else compute_hmer_overview(input_dir, processing=processing)
    )
    vision_processing = vision_processing or VisionProcessingOptions()
    vision = compute_vision_overview(
        input_dir,
        processing=vision_processing,
        samples=vision_samples,
    )
    pages = _resolve_page_annotations(
        input_dir,
        processing=vision_processing,
        pages=vision_pages,
    )
    block = compute_block_overview_from_pages(pages)
    return DatasetOverviewMetrics(hmer=hmer, vision=vision, block=block)


def _histogram(
    values: Sequence[float | int],
    *,
    bin_edges: Sequence[float],
) -> list[tuple[str, int]]:
    if not values:
        return [
            (f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}", 0)
            for i in range(len(bin_edges) - 1)
        ]
    counts = [0] * (len(bin_edges) - 1)
    for value in values:
        for index in range(len(bin_edges) - 1):
            lower = bin_edges[index]
            upper = bin_edges[index + 1]
            if index == len(bin_edges) - 2:
                if lower <= float(value) <= upper:
                    counts[index] += 1
                    break
            elif lower <= float(value) < upper:
                counts[index] += 1
                break
    return [
        (f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}", counts[i])
        for i in range(len(bin_edges) - 1)
    ]


def _aspect_ratio_bins() -> list[float]:
    return [0.0, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 3.0, 10.0]


def _resolution_histogram(
    widths: Sequence[int],
    heights: Sequence[int],
) -> list[tuple[str, int]]:
    if not widths:
        return [
            ("0-0.5M", 0),
            ("0.5-1M", 0),
            ("1-2M", 0),
            ("2-4M", 0),
            ("4-10M", 0),
            (">=10M", 0),
        ]
    counts = [0, 0, 0, 0, 0, 0]
    for width, height in zip(widths, heights, strict=True):
        pixels = width * height
        if pixels < 500_000:
            counts[0] += 1
        elif pixels < 1_000_000:
            counts[1] += 1
        elif pixels < 2_000_000:
            counts[2] += 1
        elif pixels < 4_000_000:
            counts[3] += 1
        elif pixels < 10_000_000:
            counts[4] += 1
        else:
            counts[5] += 1
    return [
        ("0-0.5M", counts[0]),
        ("0.5-1M", counts[1]),
        ("1-2M", counts[2]),
        ("2-4M", counts[3]),
        ("4-10M", counts[4]),
        (">=10M", counts[5]),
    ]


def _write_csv(
    path: Path,
    rows: Sequence[tuple[str, int | float]],
    *,
    headers: tuple[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for label, value in rows:
            writer.writerow([label, value])


def _format_count(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}"


def _format_ratio_share(count: int, share_percent: float) -> str:
    return f"{count:,}，占 {share_percent:.2f}%"


def _summary_table_rows(metrics: DatasetOverviewMetrics) -> list[str]:
    hmer = metrics.hmer
    vision = metrics.vision
    portrait_share = vision.orientation_share(vision.portrait_count)
    landscape_share = vision.orientation_share(vision.landscape_count)
    return [
        "| 指标 | 统计 |",
        "| --- | ---: |",
        f"| 图像总数 | {_format_count(vision.page_count)} |",
        f"| 图像长宽比均值 / 中位数 | "
        f"{vision.avg_aspect_ratio:.3f} / {vision.median_aspect_ratio:.3f} |",
        f"| 纵向图像数量 | "
        f"{_format_ratio_share(vision.portrait_count, portrait_share)} |",
        f"| 横向图像数量 | "
        f"{_format_ratio_share(vision.landscape_count, landscape_share)} |",
        f"| 表达式实例总数 | {_format_count(hmer.expression_count)} |",
        f"| 平均每页表达式数量 | {hmer.avg_expressions_per_page:.4f} |",
        f"| 单页最多表达式数量 | {_format_count(hmer.max_expressions_per_page)} |",
    ]


def _block_table_rows(metrics: BlockOverviewMetrics) -> list[str]:
    rows = [
        "| 类型 | 数量 |",
        "| --- | ---: |",
        f"| Txtblock | {_format_count(metrics.txtblock_count)} |",
        f"| deleted_text_block | {_format_count(metrics.deleted_text_block_count)} |",
        f"| chart | {_format_count(metrics.chart_count)} |",
        f"| figure | {_format_count(metrics.figure_count)} |",
    ]
    if metrics.other_block_count > 0:
        rows.append(f"| 其他 | {_format_count(metrics.other_block_count)} |")
    rows.append(f"| 总数 | {_format_count(metrics.total_block_count)} |")
    return rows


def _distribution_table_rows(
    histogram: Sequence[tuple[str, int]],
    total: int,
) -> list[str]:
    rows = ["| 区间 | 数量 | 占比 |", "| --- | ---: | ---: |"]
    for label, count in histogram:
        ratio = (count / total * 100.0) if total else 0.0
        rows.append(f"| {label} | {count:,} | {ratio:.2f}% |")
    return rows


def write_hmer_overview(
    metrics: HmerOverviewMetrics,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    scale_rows = [
        ("page_count", metrics.page_count),
        ("expression_count", metrics.expression_count),
        ("total_characters", metrics.total_characters),
        ("avg_expressions_per_page", round(metrics.avg_expressions_per_page, 4)),
        ("max_expressions_per_page", metrics.max_expressions_per_page),
        ("avg_chars_per_expression", round(metrics.avg_chars_per_expression, 4)),
    ]
    _write_csv(
        tables_dir / "hmer_scale.csv",
        scale_rows,
        headers=("metric", "value"),
    )

    per_page_rows = [
        (metrics.page_ids[index], count)
        for index, count in enumerate(metrics.expressions_per_page)
    ]
    _write_csv(
        tables_dir / "expressions_per_page.csv",
        per_page_rows,
        headers=("page", "expressions"),
    )

    lines = [
        "# HMER 数据集概览",
        "",
        "| 指标 | 统计 |",
        "| --- | ---: |",
        f"| 表达式实例总数 | {_format_count(metrics.expression_count)} |",
        f"| 平均每页表达式数量 | {metrics.avg_expressions_per_page:.4f} |",
        f"| 单页最多表达式数量 | {_format_count(metrics.max_expressions_per_page)} |",
        f"| 字符总数 | {_format_count(metrics.total_characters)} |",
        "",
        "## 详细表格",
        "",
        "- `tables/hmer_scale.csv`",
        "- `tables/expressions_per_page.csv`",
        "",
    ]
    overview_path = output_dir / "overview.md"
    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return overview_path


def write_vision_overview(
    metrics: VisionOverviewMetrics,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    aspect_hist = _histogram(metrics.aspect_ratios, bin_edges=_aspect_ratio_bins())
    _write_csv(
        tables_dir / "aspect_ratio_distribution.csv",
        aspect_hist,
        headers=("aspect_ratio_bin", "count"),
    )

    resolution_hist = _resolution_histogram(metrics.widths, metrics.heights)
    _write_csv(
        tables_dir / "resolution_distribution.csv",
        resolution_hist,
        headers=("resolution_bin", "count"),
    )

    orientation_rows = [
        ("portrait", metrics.portrait_count),
        ("landscape", metrics.landscape_count),
    ]
    _write_csv(
        tables_dir / "orientation_distribution.csv",
        orientation_rows,
        headers=("orientation", "count"),
    )

    dimension_path = tables_dir / "sample_dimensions.csv"
    dimension_path.parent.mkdir(parents=True, exist_ok=True)
    with dimension_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "width_px", "height_px", "aspect_ratio", "megapixels"])
        for index in range(len(metrics.widths)):
            writer.writerow(
                [
                    metrics.sample_ids[index],
                    metrics.widths[index],
                    metrics.heights[index],
                    round(metrics.aspect_ratios[index], 4),
                    round(metrics.megapixels[index], 4),
                ]
            )

    aspect_total = len(metrics.aspect_ratios)
    resolution_total = len(metrics.widths)
    portrait_share = metrics.orientation_share(metrics.portrait_count)
    landscape_share = metrics.orientation_share(metrics.landscape_count)
    lines = [
        "# Block-level 数据集概览",
        "",
        "| 指标 | 统计 |",
        "| --- | ---: |",
        f"| 图像总数 | {_format_count(metrics.page_count)} |",
        f"| 图像长宽比均值 / 中位数 | "
        f"{metrics.avg_aspect_ratio:.3f} / {metrics.median_aspect_ratio:.3f} |",
        f"| 纵向图像数量 | "
        f"{_format_ratio_share(metrics.portrait_count, portrait_share)} |",
        f"| 横向图像数量 | "
        f"{_format_ratio_share(metrics.landscape_count, landscape_share)} |",
        f"| 分辨率（均值 / P50） | "
        f"{metrics.avg_megapixels:.3f} MP / "
        f"{percentile(metrics.megapixels, 50):.3f} MP |",
        "",
        "## 图像长宽比分布",
        "",
        *_distribution_table_rows(aspect_hist, aspect_total),
        "",
        "## 图像分辨率分布",
        "",
        *_distribution_table_rows(resolution_hist, resolution_total),
        "",
        "## 详细表格",
        "",
        "- `tables/aspect_ratio_distribution.csv`",
        "- `tables/resolution_distribution.csv`",
        "- `tables/orientation_distribution.csv`",
        "- `tables/sample_dimensions.csv`",
        "",
    ]
    overview_path = output_dir / "overview.md"
    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return overview_path


def write_block_overview(
    metrics: BlockOverviewMetrics,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    block_rows = [
        ("Txtblock", metrics.txtblock_count),
        ("deleted_text_block", metrics.deleted_text_block_count),
        ("chart", metrics.chart_count),
        ("figure", metrics.figure_count),
    ]
    if metrics.other_block_count > 0:
        block_rows.append(("其他", metrics.other_block_count))
    block_rows.append(("总数", metrics.total_block_count))
    _write_csv(
        tables_dir / "block_counts.csv",
        block_rows,
        headers=("block_type", "count"),
    )

    lines = [
        "# Block 数据集概览",
        "",
        "按标注块类型统计 block 数量。",
        "",
        *_block_table_rows(metrics),
        "",
        "## 详细表格",
        "",
        "- `tables/block_counts.csv`",
        "",
    ]
    overview_path = output_dir / "overview.md"
    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return overview_path


def write_dataset_overview(
    metrics: DatasetOverviewMetrics,
    output_root: Path,
    *,
    skip_domain_overviews: bool = False,
    page_level_detail_md: str | None = None,
    line_level_detail_md: str | None = None,
    structure_layout_detail_md: str | None = None,
    summary_json_path: Path | None = None,
    pipeline_doc_path: Path | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if skip_domain_overviews:
        hmer_detail_md = "HMER/summary.md"
        hmer_tables = "HMER/tables/"
        vision_detail_md = structure_layout_detail_md or (
            "block_level/structure_layout/block_level_summary.md"
        )
        vision_tables = "block_level/structure_layout/tables/"
        block_detail_md = "block/overview.md"
        block_tables = "block/tables/"
    else:
        hmer_dir = output_root / "HMER"
        block_level_dir = output_root / "block_level"
        block_dir = output_root / "block"
        write_hmer_overview(metrics.hmer, hmer_dir)
        write_vision_overview(metrics.vision, block_level_dir)
        write_block_overview(metrics.block, block_dir)
        hmer_detail_md = "HMER/overview.md"
        hmer_tables = "HMER/tables/"
        vision_detail_md = "block_level/overview.md"
        vision_tables = "block_level/tables/"
        block_detail_md = "block/overview.md"
        block_tables = "block/tables/"

    hmer = metrics.hmer
    avg_chars_per_page = (
        hmer.total_characters / hmer.page_count if hmer.page_count else 0.0
    )
    lines = [
        "# 数据集总纲",
        "",
    ]
    if summary_json_path is not None:
        summary_rel = summary_json_path.name
        if summary_json_path.parent.resolve() == output_root.resolve():
            lines.extend(
                [
                    f"> 机器可读项目总览：[`{summary_rel}`]({summary_rel})",
                    "",
                ]
            )
    lines.extend(
        [
            *_summary_table_rows(metrics),
            "",
            "## Block（标注块）",
            "",
            *_block_table_rows(metrics.block),
            "",
            "## 明细",
            "",
            "### HMER（文本段）",
            "",
            f"- 字符总数：{_format_count(hmer.total_characters)}",
            f"- 平均每页字符数：{avg_chars_per_page:.2f}",
            "",
            f"详见 [{hmer_detail_md}]({hmer_detail_md}) 与 `{hmer_tables}`。",
            "",
            "### Block（标注块）",
            "",
            f"- Block 标注总数：{_format_count(metrics.block.total_block_count)}",
            f"- Txtblock 数量：{_format_count(metrics.block.txtblock_count)}",
            "",
            f"详见 [{block_detail_md}]({block_detail_md}) 与 `{block_tables}`。",
            "",
            "### Block-level（块级图像）",
            "",
            f"- 页面流结构与块组成：见 [{vision_detail_md}]({vision_detail_md})",
            "",
            f"详见 [{vision_detail_md}]({vision_detail_md}) 与 `{vision_tables}`。",
            "",
        ]
    )
    if pipeline_doc_path is not None and pipeline_doc_path.is_file():
        pipeline_rel = pipeline_doc_path.name
        if pipeline_doc_path.parent.resolve() == output_root.resolve():
            lines.extend(
                [
                    f"- 分层导出与 page_id 连接说明：[`{pipeline_rel}`]({pipeline_rel})",
                    "",
                ]
            )
    if page_level_detail_md is not None:
        lines.extend(
            [
                "### Page-level（纯图像）",
                "",
                f"- 整页前景密度、对比度与热力图：见 [{page_level_detail_md}]({page_level_detail_md})",
                "",
            ]
        )
    if line_level_detail_md is not None:
        lines.extend(
            [
                "### Line-level（行级标注）",
                "",
                f"- 几何、笔画、布局与扫描质量：见 [{line_level_detail_md}]({line_level_detail_md})",
                "",
            ]
        )
    overview_path = output_root / "dataset_overview.md"
    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return overview_path


def run_dataset_overview_export(
    input_dir: Path,
    output_root: Path,
    *,
    processing: ProcessingOptions | None = None,
    vision_processing: VisionProcessingOptions | None = None,
    enriched: EnrichedCorpus | None = None,
    vision_samples: Sequence[ImageSampleRecord] | None = None,
    vision_pages: Sequence[PageAnnotation] | None = None,
    metrics: DatasetOverviewMetrics | None = None,
    skip_domain_overviews: bool = False,
    page_level_detail_md: str | None = None,
    line_level_detail_md: str | None = None,
    structure_layout_detail_md: str | None = None,
    summary_json_path: Path | None = None,
    pipeline_doc_path: Path | None = None,
) -> Path:
    if metrics is None:
        metrics = compute_dataset_overview(
            input_dir,
            processing=processing,
            vision_processing=vision_processing,
            enriched=enriched,
            vision_samples=vision_samples,
            vision_pages=vision_pages,
        )
    return write_dataset_overview(
        metrics,
        output_root,
        skip_domain_overviews=skip_domain_overviews,
        page_level_detail_md=page_level_detail_md,
        line_level_detail_md=line_level_detail_md,
        structure_layout_detail_md=structure_layout_detail_md,
        summary_json_path=summary_json_path,
        pipeline_doc_path=pipeline_doc_path,
    )
