"""Block-level benchmark export pipeline (flow structure classification)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from benchmark_design.block_level.dataset import (
    BlockLevelBenchmarkDataset,
    load_block_level_benchmark_dataset,
)
from benchmark_design.block_level.flow_structure.flow_group import FLOW_GROUP_HIERARCHY, FLOW_GROUP_LABELS
from benchmark_design.block_level.flow_structure.models import PageAnnotation, PageFlowStructureResult
from benchmark_design.block_level.page_metrics import compute_block_level_benchmark_results
from benchmark_design.block_level.block_foreground_density import (
    BlockForegroundDensityRow,
    compute_block_foreground_densities,
)
from benchmark_design.block_level.processing_options import BlockLevelProcessingOptions
from benchmark_design.block_level.sample_record import ImageSampleRecord
from benchmark_design.export_layout import (
    HYBRID_LAYOUT_FLOW_STRUCTURES,
    STRUCTURE_LAYOUT_FLOW_STRUCTURES,
    export_flow_structure_label,
)
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.page_level.models import CalibrationResult
from benchmark_design.progress import run_parallel_tasks
from benchmark_design.report.block_level.block_density_export import write_block_foreground_density_csv
from benchmark_design.report.block_level.block_density_plotting import export_block_foreground_density_figure
from benchmark_design.report.block_level.block_density_summary import write_block_level_density_summary
from benchmark_design.report.block_level.flow_structure_export import (
    write_flow_group_summary_csv,
    write_flow_structure_block_geometry_csv,
    write_flow_structure_decisions_jsonl,
    write_flow_structure_page_metrics_csv,
)
from benchmark_design.report.block_level.flow_structure_figures import export_flow_structure_figures
from benchmark_design.report.block_level.flow_structure_summary import write_flow_structure_summary_md
from benchmark_design.report.block_level.output_layout import BlockLevelOutputLayout
from benchmark_design.report.dataset_overview import run_dataset_overview_export
from benchmark_design.report.output_layout import relative_input_path

LayoutKind = Literal["structure_layout", "hybrid_layout"]

_STRUCTURE_HIERARCHY = tuple(
    group for group in FLOW_GROUP_HIERARCHY if group[0] in {"Single-flow", "Columnar-flow"}
)
_HYBRID_HIERARCHY = tuple(group for group in FLOW_GROUP_HIERARCHY if group[0] == "Hybrid-flow")


def _filter_flow_results(
    flow_results: Sequence[PageFlowStructureResult],
    *,
    flow_structures: frozenset[str] | None,
) -> list[PageFlowStructureResult]:
    if flow_structures is None:
        return list(flow_results)
    return [result for result in flow_results if result.flow_structure in flow_structures]


def _filter_samples_for_results(
    samples: Sequence[ImageSampleRecord],
    flow_results: Sequence[PageFlowStructureResult],
) -> list[ImageSampleRecord]:
    page_ids = {result.page_id for result in flow_results}
    return [sample for sample in samples if sample.sample_id in page_ids]


def _layout_title(layout_kind: LayoutKind) -> str:
    if layout_kind == "hybrid_layout":
        return "Hybrid Layout Benchmark Summary"
    return "Structure Layout Benchmark Summary"


def _write_sample_index_csv(samples: list[ImageSampleRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["sample_id,image_path,width_px,height_px,dataset,source_file"]
    for sample in samples:
        lines.append(
            f"{sample.sample_id},{sample.image_path},{sample.width_px or ''},"
            f"{sample.height_px or ''},{sample.dataset},{sample.source_file}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_md(
    samples: list[ImageSampleRecord],
    output_path: Path,
    *,
    input_dir: Path,
    flow_group_counts: Counter[str],
    manual_review_count: int,
    flow_summary_path: Path,
    layout_kind: LayoutKind,
) -> None:
    widths = [sample.width_px for sample in samples if sample.width_px is not None]
    heights = [sample.height_px for sample in samples if sample.height_px is not None]
    mean_w = sum(widths) / len(widths) if widths else 0.0
    mean_h = sum(heights) / len(heights) if heights else 0.0
    lines = [
        f"# {_layout_title(layout_kind)}",
        "",
        f"- Input: `{relative_input_path(input_dir)}`",
        f"- Layout domain: **{layout_kind}**",
        f"- Image samples indexed: **{len(samples):,}**",
        f"- Mean width (px): **{mean_w:.1f}**" if widths else "- Mean width (px): *unavailable*",
        f"- Mean height (px): **{mean_h:.1f}**" if heights else "- Mean height (px): *unavailable*",
        "",
        f"See [`{flow_summary_path.name}`]({flow_summary_path.name}) for the flow-structure breakdown.",
        "",
        "## Answer-Block Flow Structure (flow_group overview)",
        "",
        "| Flow group | Pages |",
        "| --- | ---: |",
    ]
    for label in FLOW_GROUP_LABELS:
        count = flow_group_counts.get(label, 0)
        if count:
            lines.append(f"| {label} | {count:,} |")
    if flow_group_counts.get("no_valid_answer_block", 0):
        lines.append(f"| no_valid_answer_block | {flow_group_counts.get('no_valid_answer_block', 0):,} |")
    lines.extend(
        [
            "",
            f"- Pages flagged for manual review: **{manual_review_count:,}**",
            "",
            "## Outputs",
            "",
            "- `tables/sample_index.csv` — one row per page",
            "- `tables/flow_structure_page_metrics.csv` — page-level metrics with flow_group",
            "- `tables/flow_group_summary.csv` — hierarchical flow_group summary",
            "- `tables/flow_structure_block_geometry.csv` — block annotation geometry audit table",
            "- `tables/block_foreground_density.csv` — Txtblock foreground density (denominator = annotation mask pixels)",
            "- `../block_foreground_density_distribution.png` — dataset-level Txtblock density interval distribution",
            "- `details/flow_structure_decisions.jsonl` — per-page rule outcomes and metrics",
            f"- `{flow_summary_path.name}` — flow group + diagnostic breakdown",
        ]
    )
    if layout_kind == "hybrid_layout":
        lines.append("- `figures/hybrid_layout/{flow_group_id}/` — mask overlay PNGs")
    else:
        lines.append(
            "- `figures/flow_structure/{single_flow|columnar_flow|na}/{flow_group_id}/` — mask overlay PNGs"
        )
    lines.extend(
        [
            "- `metadata.json` — export provenance",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_block_level_export(
    input_dir: Path,
    output_root: Path,
    *,
    processing: BlockLevelProcessingOptions | None = None,
    dataset: BlockLevelBenchmarkDataset | None = None,
    flow_results: Sequence[PageFlowStructureResult] | None = None,
    flow_structures: frozenset[str] | None = None,
    layout_kind: LayoutKind = "structure_layout",
    skip_flow_figures: bool = False,
    skip_dataset_overview: bool = True,
    # Accepted for backward compatibility; ignored.
    skip_foreground_load_figures: bool = False,
    skip_deleted_block_scale_figures: bool = False,
    skip_page_intrinsic: bool = False,
    dataset_version: str = "",
) -> dict[str, str]:
    """Run block-level export for answer-block flow structure classification."""
    del skip_foreground_load_figures, skip_deleted_block_scale_figures, skip_page_intrinsic, dataset_version
    processing = processing or BlockLevelProcessingOptions()
    layout = BlockLevelOutputLayout(output_root)
    layout.ensure()

    if dataset is None:
        dataset = load_block_level_benchmark_dataset(input_dir, processing=processing)
    pages = list(dataset.pages)
    samples = list(dataset.samples)

    if flow_results is None:
        block_results = compute_block_level_benchmark_results(
            input_dir,
            processing=processing,
            input_dir_for_images=input_dir,
            pages=pages,
        )
        all_flow_results = block_results.flow_structure
    else:
        all_flow_results = list(flow_results)

    if flow_structures is None:
        if layout_kind == "hybrid_layout":
            flow_structures = HYBRID_LAYOUT_FLOW_STRUCTURES
        elif layout_kind == "structure_layout":
            flow_structures = STRUCTURE_LAYOUT_FLOW_STRUCTURES

    filtered_results = _filter_flow_results(all_flow_results, flow_structures=flow_structures)
    filtered_samples = _filter_samples_for_results(samples, filtered_results)

    sample_index = layout.tables / "sample_index.csv"
    _write_sample_index_csv(filtered_samples, sample_index)

    flow_group_counts = Counter(result.flow_group for result in filtered_results)
    manual_review_count = sum(1 for result in filtered_results if result.needs_manual_review)

    page_metrics = layout.tables / "flow_structure_page_metrics.csv"
    flow_group_summary = layout.tables / "flow_group_summary.csv"
    block_geometry = layout.tables / "flow_structure_block_geometry.csv"
    decisions_jsonl = layout.details / "flow_structure_decisions.jsonl"

    def _write_flow_tables() -> None:
        write_flow_structure_page_metrics_csv(filtered_results, page_metrics)
        write_flow_group_summary_csv(filtered_results, flow_group_summary)
        write_flow_structure_block_geometry_csv(
            [record for result in filtered_results for record in result.block_records],
            block_geometry,
        )
        write_flow_structure_decisions_jsonl(filtered_results, decisions_jsonl)
        write_flow_structure_summary_md(filtered_results, layout.flow_structure_summary_md)
        _write_summary_md(
            filtered_samples,
            layout.summary_md,
            input_dir=input_dir,
            flow_group_counts=flow_group_counts,
            manual_review_count=manual_review_count,
            flow_summary_path=layout.flow_structure_summary_md,
            layout_kind=layout_kind,
        )

    run_parallel_tasks(
        {"flow": _write_flow_tables},
        description="Writing block-level export tables",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )

    figure_tasks: dict[str, Callable[[], dict[str, int]]] = {}
    if not skip_flow_figures:
        if layout_kind == "hybrid_layout":
            figure_tasks["flow"] = lambda: export_flow_structure_figures(
                filtered_results,
                input_dir=input_dir,
                figures_root=layout.figures_hybrid_layout,
                show_progress=processing.show_progress,
                flow_hierarchy=_HYBRID_HIERARCHY,
                include_na_figures=False,
            )
        else:
            figure_tasks["flow"] = lambda: export_flow_structure_figures(
                filtered_results,
                input_dir=input_dir,
                figures_root=layout.figures_flow_group_examples,
                show_progress=processing.show_progress,
                flow_hierarchy=_STRUCTURE_HIERARCHY,
                include_na_figures=True,
            )

    figure_results = run_parallel_tasks(
        figure_tasks,
        description="Writing block-level export figures",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
    figure_counts = figure_results.get("flow", {})

    manifest = {
        "summary": layout.summary_md.name,
        "flow_structure_summary": layout.flow_structure_summary_md.name,
        "sample_index": f"tables/{sample_index.name}",
        "flow_structure_page_metrics": f"tables/{page_metrics.name}",
        "flow_group_summary": f"tables/{flow_group_summary.name}",
        "flow_structure_block_geometry": f"tables/{block_geometry.name}",
        "flow_structure_decisions": f"details/{decisions_jsonl.name}",
    }
    if figure_counts:
        manifest["flow_group_figures"] = (
            "figures/hybrid_layout/" if layout_kind == "hybrid_layout" else "figures/flow_structure/"
        )
    flow_structure_counts = Counter(
        export_flow_structure_label(result.flow_structure) for result in filtered_results
    )
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "domain": "block_level",
        "layout_kind": layout_kind,
        "input_dir": relative_input_path(input_dir),
        "sample_count": len(filtered_samples),
        "page_count": len(filtered_results),
        "flow_group_counts": dict(flow_group_counts),
        "flow_structure_counts": dict(flow_structure_counts),
        "manual_review_count": manual_review_count,
        "flow_group_figure_counts": figure_counts,
        "manifest": manifest,
    }
    layout.metadata_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["metadata"] = layout.metadata_json.name

    if not skip_dataset_overview:
        overview_path = run_dataset_overview_export(
            input_dir,
            output_root,
            processing=ProcessingOptions(
                show_progress=processing.show_progress,
                workers=processing.workers,
            ),
            vision_processing=processing,
            vision_samples=samples,
            vision_pages=pages,
        )
        manifest["dataset_overview"] = overview_path.name

    return manifest


def write_block_density_exports(
    *,
    pages: Sequence[PageAnnotation],
    flow_results: Sequence[PageFlowStructureResult],
    input_dir: Path,
    calibration: CalibrationResult,
    structure_layout_dir: Path,
    hybrid_layout_dir: Path,
    block_level_dir: Path | None = None,
    processing: BlockLevelProcessingOptions | None = None,
    gray_cache_root: Path | None = None,
    density_rows: list[BlockForegroundDensityRow] | None = None,
) -> dict[str, str]:
    """Write Txtblock foreground density tables after page-level calibration is available."""
    processing = processing or BlockLevelProcessingOptions()
    rows = density_rows
    if rows is None:
        rows = compute_block_foreground_densities(
            list(pages),
            input_dir=input_dir,
            calibration=calibration,
            show_progress=processing.show_progress,
            workers=processing.workers,
            gray_cache_root=gray_cache_root,
        )
    flow_by_page = {result.page_id: result.flow_structure for result in flow_results}
    structure_rows = [
        row
        for row in rows
        if flow_by_page.get(row.page_id) in STRUCTURE_LAYOUT_FLOW_STRUCTURES
    ]
    hybrid_rows = [
        row for row in rows if flow_by_page.get(row.page_id) in HYBRID_LAYOUT_FLOW_STRUCTURES
    ]
    structure_path = structure_layout_dir / "tables" / "block_foreground_density.csv"
    hybrid_path = hybrid_layout_dir / "tables" / "block_foreground_density.csv"
    write_block_foreground_density_csv(structure_rows, structure_path)
    write_block_foreground_density_csv(hybrid_rows, hybrid_path)

    manifest = {
        "structure_layout_block_foreground_density": structure_path.relative_to(
            structure_layout_dir
        ).as_posix(),
        "hybrid_layout_block_foreground_density": hybrid_path.relative_to(hybrid_layout_dir).as_posix(),
    }
    combined_rows = structure_rows + hybrid_rows
    if combined_rows and block_level_dir is not None:
        figure_path = export_block_foreground_density_figure(
            combined_rows,
            block_level_dir,
            title_suffix="Txtblock",
            gray_threshold=calibration.gray_threshold,
        )
        manifest["block_foreground_density_figure"] = figure_path.relative_to(block_level_dir).as_posix()
        summary_paths = write_block_level_density_summary(
            output_dir=block_level_dir,
            rows=combined_rows,
            structure_rows=structure_rows,
            hybrid_rows=hybrid_rows,
            flow_results=flow_results,
            calibration=calibration,
            input_dir=input_dir,
        )
        for key, path in summary_paths.items():
            manifest[key] = path.relative_to(block_level_dir).as_posix()
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export block-level benchmark tables, figures, and reports.")
    parser.add_argument("input", type=Path, help="Benchmark JSON page export directory")
    parser.add_argument("output", type=Path, help="Block-level output root directory")
    parser.add_argument("--workers", type=int, default=None, help="Worker threads for parallel export stages")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--skip-dimensions", action="store_true", help="Skip reading image width/height")
    parser.add_argument("--skip-flow-figures", action="store_true", help="Skip flow structure figure export")
    args = parser.parse_args(argv)

    processing = BlockLevelProcessingOptions(
        show_progress=not args.no_progress,
        workers=args.workers,
        read_image_dimensions=not args.skip_dimensions,
    )
    manifest = run_block_level_export(
        args.input,
        args.output,
        processing=processing,
        skip_flow_figures=args.skip_flow_figures,
    )
    print(f"wrote {len(manifest)} block-level artifacts under {args.output.resolve()}")
    for key, rel_path in manifest.items():
        print(f"{key}: {rel_path}")
    return 0


# Backward-compatible alias (deprecated).
run_vision_benchmark_export = run_block_level_export


if __name__ == "__main__":
    raise SystemExit(main())
