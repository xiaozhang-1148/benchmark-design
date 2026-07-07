"""Vision benchmark export pipeline."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmark_design.report.output_layout import relative_input_path
from benchmark_design.progress import run_parallel_tasks
from benchmark_design.report.vision.deleted_block_scale_export import (
    write_deleted_block_scale_block_geometry_csv,
    write_deleted_block_scale_diagnostics_json,
    write_deleted_block_scale_page_metrics_csv,
)
from benchmark_design.report.vision.deleted_block_scale_stats import (
    compute_deleted_block_scale_summary_stats,
)
from benchmark_design.report.vision.deleted_block_scale_figures import export_deleted_block_scale_figures
from benchmark_design.report.vision.deleted_block_scale_summary import write_deleted_block_scale_summary_md
from benchmark_design.report.vision.flow_structure_export import (
    write_flow_group_summary_csv,
    write_flow_structure_block_geometry_csv,
    write_flow_structure_decisions_jsonl,
    write_flow_structure_page_metrics_csv,
)
from benchmark_design.report.vision.flow_structure_figures import export_flow_structure_figures
from benchmark_design.report.vision.flow_structure_summary import write_flow_structure_summary_md
from benchmark_design.report.vision.foreground_pixel_density_diagnostics_figures import (
    export_foreground_pixel_density_diagnostics_figures,
)
from benchmark_design.report.vision.foreground_pixel_density_comparison_figures import (
    export_foreground_pixel_density_comparison_figures,
)
from benchmark_design.report.vision.foreground_pixel_density_export import (
    compute_and_write_flow_structure_density_csv,
    write_foreground_pixel_density_block_metrics_csv,
    write_foreground_pixel_density_diagnostics_json,
    write_foreground_pixel_density_overall_csv,
    write_foreground_pixel_density_page_metrics_csv,
    write_foreground_pixel_density_region_metrics_csv,
)
from benchmark_design.report.vision.foreground_pixel_density_figures import (
    export_foreground_pixel_density_figures,
)
from benchmark_design.report.vision.foreground_pixel_density_summary import (
    write_foreground_pixel_density_summary_md,
)
from benchmark_design.report.dataset_overview import run_dataset_overview_export
from benchmark_design.report.vision.output_layout import VisionBenchmarkOutputLayout
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.vision.dataset import VisionBenchmarkDataset, load_vision_benchmark_dataset
from benchmark_design.vision.deleted_block_scale.thresholds import (
    DELETED_TEXT_BLOCK_TYPE,
    VALID_BLOCK_TYPES,
    VALID_BLOCK_TYPE,
)
from benchmark_design.vision.page_metrics import compute_vision_benchmark_results
from benchmark_design.vision.flow_structure.flow_group import FLOW_GROUP_LABELS
from benchmark_design.vision.processing_options import VisionProcessingOptions
from benchmark_design.vision.sample_record import ImageSampleRecord


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
) -> None:
    widths = [sample.width_px for sample in samples if sample.width_px is not None]
    heights = [sample.height_px for sample in samples if sample.height_px is not None]
    mean_w = sum(widths) / len(widths) if widths else 0.0
    mean_h = sum(heights) / len(heights) if heights else 0.0
    lines = [
        "# Vision Benchmark Summary",
        "",
        f"- Input: `{relative_input_path(input_dir)}`",
        f"- Image samples indexed: **{len(samples):,}**",
        f"- Mean width (px): **{mean_w:.1f}**" if widths else "- Mean width (px): *unavailable*",
        f"- Mean height (px): **{mean_h:.1f}**" if heights else "- Mean height (px): *unavailable*",
        "",
        "See [`flow_structure_summary.md`](flow_structure_summary.md), "
        "[`foreground_pixel_density_summary.md`](foreground_pixel_density_summary.md), and "
        "[`deleted_block_scale_summary.md`](deleted_block_scale_summary.md) for metric breakdowns.",
        "",
        "## Answer-Block Flow Structure (flow_group overview)",
        "",
        "| Flow group | Pages |",
        "| --- | ---: |",
    ]
    for label in FLOW_GROUP_LABELS:
        lines.append(f"| {label} | {flow_group_counts.get(label, 0):,} |")
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
            "- `tables/flow_structure_block_geometry.csv` — txtBlock geometry audit table",
            "- `tables/foreground_pixel_density_page_metrics.csv` — page-level foreground pixel density",
            "- `tables/foreground_pixel_density_block_metrics.csv` — txtBlock foreground pixel density",
            "- `tables/foreground_pixel_density_overall.csv` — continuous distribution summary",
            "- `tables/foreground_pixel_density_by_flow_structure.csv` — density by flow structure",
            "- `tables/deleted_block_scale_page_metrics.csv` — deleted-block scale (page)",
            "- `tables/deleted_block_scale_block_geometry.csv` — Txtblock/deleted_text_block audit",
            "- `metadata/foreground_pixel_density_diagnostics.json` — internal QA thresholds/tags",
            "- `details/flow_structure_decisions.jsonl` — per-page rule outcomes and metrics",
            f"- `{flow_summary_path.name}` — flow group + diagnostic breakdown",
            "- `foreground_pixel_density_summary.md` — continuous density summary",
            "- `deleted_block_scale_summary.md` — R_del summary (section 2.3.2 table)",
            "- `metadata/deleted_block_scale_diagnostics.json` — area totals and tail cutoffs",
            "- `figures/flow_structure/{single_flow|columnar_flow|hybrid_flow|na}/{flow_group_id}/` — mask overlay PNGs",
            "- `figures/d_page_density_bands.png` — page-level density band distribution",
            "- `figures/d_block_density_bands.png` — block-level density band distribution",
            "- `figures/density_level_comparison.png` — page vs block overall density comparison",
            "- `figures/deleted_block_scale/r_del_histogram.png` — R_del distribution among affected pages",
            "- `figures/deleted_block_scale/deleted_instance_histogram.png` — deleted block count per affected page",
            "- `figures/deleted_block_scale/high_r_del_examples/` — high-burden page overlays",
            "- `metadata.json` — export provenance",
            "",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_vision_benchmark_export(
    input_dir: Path,
    output_root: Path,
    *,
    processing: VisionProcessingOptions | None = None,
    dataset: VisionBenchmarkDataset | None = None,
    skip_flow_figures: bool = False,
    skip_foreground_load_figures: bool = False,
    skip_deleted_block_scale_figures: bool = False,
    skip_dataset_overview: bool = True,
) -> dict[str, str]:
    """Run vision export including flow structure, foreground load, and deleted-block scale."""
    processing = processing or VisionProcessingOptions()
    layout = VisionBenchmarkOutputLayout(output_root)
    layout.ensure()

    if dataset is None:
        dataset = load_vision_benchmark_dataset(input_dir, processing=processing)
    pages = list(dataset.pages)
    samples = list(dataset.samples)
    sample_index = layout.tables / "sample_index.csv"
    _write_sample_index_csv(samples, sample_index)

    vision_results = compute_vision_benchmark_results(
        input_dir,
        processing=processing,
        input_dir_for_images=input_dir,
        pages=pages,
    )
    flow_results = vision_results.flow_structure
    fg_results = vision_results.foreground_load
    fg_thresholds = vision_results.foreground_load_thresholds
    global_fg_config = vision_results.global_foreground_config
    dbs_results = vision_results.deleted_block_scale

    flow_group_counts = Counter(result.flow_group for result in flow_results)
    manual_review_count = sum(1 for result in flow_results if result.needs_manual_review)

    page_metrics = layout.tables / "flow_structure_page_metrics.csv"
    flow_group_summary = layout.tables / "flow_group_summary.csv"
    block_geometry = layout.tables / "flow_structure_block_geometry.csv"
    decisions_jsonl = layout.details / "flow_structure_decisions.jsonl"
    fg_page_metrics = layout.tables / "foreground_pixel_density_page_metrics.csv"
    fg_block_metrics = layout.tables / "foreground_pixel_density_block_metrics.csv"
    fg_region_metrics = layout.tables / "foreground_pixel_density_region_metrics.csv"
    fg_overall = layout.tables / "foreground_pixel_density_overall.csv"
    fg_by_flow = layout.tables / "foreground_pixel_density_by_flow_structure.csv"
    fg_diagnostics_json = layout.metadata_dir / "foreground_pixel_density_diagnostics.json"
    dbs_page_metrics = layout.tables / "deleted_block_scale_page_metrics.csv"
    dbs_block_geometry = layout.tables / "deleted_block_scale_block_geometry.csv"
    dbs_diagnostics_json = layout.metadata_dir / "deleted_block_scale_diagnostics.json"

    def _write_flow_tables() -> None:
        write_flow_structure_page_metrics_csv(flow_results, page_metrics)
        write_flow_group_summary_csv(flow_results, flow_group_summary)
        write_flow_structure_block_geometry_csv(
            [record for result in flow_results for record in result.block_records],
            block_geometry,
        )
        write_flow_structure_decisions_jsonl(flow_results, decisions_jsonl)
        write_flow_structure_summary_md(flow_results, layout.flow_structure_summary_md)
        _write_summary_md(
            samples,
            layout.summary_md,
            input_dir=input_dir,
            flow_group_counts=flow_group_counts,
            manual_review_count=manual_review_count,
            flow_summary_path=layout.flow_structure_summary_md,
        )

    def _write_foreground_pixel_density_tables() -> None:
        page_stats, block_stats = write_foreground_pixel_density_overall_csv(fg_results, fg_overall)
        flow_stats = compute_and_write_flow_structure_density_csv(fg_results, flow_results, fg_by_flow)
        write_foreground_pixel_density_page_metrics_csv(fg_results, fg_page_metrics)
        write_foreground_pixel_density_block_metrics_csv(
            [block for result in fg_results for block in result.block_results],
            fg_block_metrics,
        )
        write_foreground_pixel_density_region_metrics_csv(fg_results, fg_region_metrics)
        write_foreground_pixel_density_diagnostics_json(
            fg_results,
            fg_thresholds,
            fg_diagnostics_json,
            global_config=global_fg_config,
        )
        write_foreground_pixel_density_summary_md(
            fg_results,
            layout.foreground_pixel_density_summary_md,
            page_stats=page_stats,
            block_stats=block_stats,
            flow_stats=flow_stats,
            thresholds=fg_thresholds,
            diagnostics_json_path="metadata/foreground_pixel_density_diagnostics.json",
        )

    def _write_deleted_block_scale_tables() -> None:
        dbs_stats = compute_deleted_block_scale_summary_stats(dbs_results)
        write_deleted_block_scale_page_metrics_csv(dbs_results, dbs_page_metrics)
        write_deleted_block_scale_block_geometry_csv(
            [record for result in dbs_results for record in result.block_records],
            dbs_block_geometry,
        )
        write_deleted_block_scale_diagnostics_json(dbs_stats, dbs_diagnostics_json)
        write_deleted_block_scale_summary_md(
            dbs_results,
            layout.deleted_block_scale_summary_md,
            stats=dbs_stats,
        )

    run_parallel_tasks(
        {
            "flow": _write_flow_tables,
            "foreground_pixel_density": _write_foreground_pixel_density_tables,
            "deleted_block_scale": _write_deleted_block_scale_tables,
        },
        description="Writing vision export tables",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )

    figure_tasks: dict[str, Callable[[], dict[str, int]]] = {}
    if not skip_flow_figures:
        figure_tasks["flow"] = lambda: export_flow_structure_figures(
            flow_results,
            input_dir=input_dir,
            figures_root=layout.figures_flow_group_examples,
            show_progress=processing.show_progress,
        )
    if not skip_foreground_load_figures:
        def _export_fg_figures() -> dict[str, int | bool]:
            counts = export_foreground_pixel_density_figures(
                fg_results,
                pages,
                flow_results,
                input_dir=input_dir,
                figures_root=layout.figures,
            )
            comparison_counts = export_foreground_pixel_density_comparison_figures(
                fg_results,
                pages,
                input_dir=input_dir,
                figures_root=layout.figures,
                global_config=global_fg_config,
            )
            diagnostics_counts = export_foreground_pixel_density_diagnostics_figures(
                fg_results,
                pages,
                input_dir=input_dir,
                figures_root=layout.figures,
                global_config=global_fg_config,
                calibration_histogram=np.array(global_fg_config.calibration_histogram, dtype=np.int64),
            )
            merged: dict[str, int | bool] = dict(counts)
            merged.update(comparison_counts)
            merged.update(diagnostics_counts)
            return merged

        figure_tasks["foreground_pixel_density"] = _export_fg_figures
    if not skip_deleted_block_scale_figures:
        figure_tasks["deleted_block_scale"] = lambda: export_deleted_block_scale_figures(
            dbs_results,
            pages,
            input_dir=input_dir,
            figures_root=layout.figures_deleted_block_scale,
            show_progress=processing.show_progress,
        )

    figure_results = run_parallel_tasks(
        figure_tasks,
        description="Writing vision export figures",
        show_progress=processing.show_progress,
        workers=processing.workers,
    )
    figure_counts = figure_results.get("flow", {})
    fg_figure_counts = figure_results.get("foreground_pixel_density", {})
    dbs_figure_counts = figure_results.get("deleted_block_scale", {})

    manifest = {
        "summary": layout.summary_md.name,
        "flow_structure_summary": layout.flow_structure_summary_md.name,
        "foreground_pixel_density_summary": layout.foreground_pixel_density_summary_md.name,
        "deleted_block_scale_summary": layout.deleted_block_scale_summary_md.name,
        "sample_index": f"tables/{sample_index.name}",
        "flow_structure_page_metrics": f"tables/{page_metrics.name}",
        "flow_group_summary": f"tables/{flow_group_summary.name}",
        "flow_structure_block_geometry": f"tables/{block_geometry.name}",
        "flow_structure_decisions": f"details/{decisions_jsonl.name}",
        "foreground_pixel_density_page_metrics": f"tables/{fg_page_metrics.name}",
        "foreground_pixel_density_block_metrics": f"tables/{fg_block_metrics.name}",
        "foreground_pixel_density_region_metrics": f"tables/{fg_region_metrics.name}",
        "foreground_pixel_density_overall": f"tables/{fg_overall.name}",
        "foreground_pixel_density_by_flow_structure": f"tables/{fg_by_flow.name}",
        "foreground_pixel_density_diagnostics": f"metadata/{fg_diagnostics_json.name}",
        "deleted_block_scale_page_metrics": f"tables/{dbs_page_metrics.name}",
        "deleted_block_scale_block_geometry": f"tables/{dbs_block_geometry.name}",
        "deleted_block_scale_diagnostics": f"metadata/{dbs_diagnostics_json.name}",
    }
    if figure_counts:
        manifest["flow_group_figures"] = "figures/flow_structure/"
    if fg_figure_counts:
        manifest["foreground_pixel_density_figures"] = "figures/d_block_density_bands.png"
    if dbs_figure_counts:
        manifest["deleted_block_scale_figures"] = "figures/deleted_block_scale/"
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "domain": "vision",
        "input_dir": relative_input_path(input_dir),
        "sample_count": len(samples),
        "page_count": len(flow_results),
        "flow_group_counts": dict(flow_group_counts),
        "flow_structure_counts": dict(Counter(result.flow_structure for result in flow_results)),
        "manual_review_count": manual_review_count,
        "flow_group_figure_counts": figure_counts,
        "foreground_pixel_density_page_count": len(fg_results),
        "foreground_pixel_density_manual_review_count": sum(
            1 for result in fg_results if result.needs_manual_review
        ),
        "foreground_pixel_density_figure_counts": fg_figure_counts,
        "deleted_block_scale_page_count": len(dbs_results),
        "deleted_block_scale_pages_with_deleted": sum(
            1 for result in dbs_results if result.has_deleted_text_block
        ),
        "deleted_block_scale_manual_review_count": sum(
            1 for result in dbs_results if result.needs_manual_review
        ),
        "deleted_block_scale_figure_counts": dbs_figure_counts,
        "deleted_block_scale": {
            "valid_block_types": sorted(VALID_BLOCK_TYPES),
            "valid_block_type": VALID_BLOCK_TYPE,
            "deleted_text_block_type": DELETED_TEXT_BLOCK_TYPE,
            "r_del_formula": "deleted_area / answer_related_area",
            "area_mode": "polygon_mask_union",
            "A_valid": "Txtblock ∪ chart ∪ figure",
            "A_deleted": "deleted_text_block",
            "A_ans": "A_valid ∪ A_deleted",
        },
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
