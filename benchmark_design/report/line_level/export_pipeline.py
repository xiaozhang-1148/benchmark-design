"""Line-level analysis export pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_design.line_level.config import load_line_level_config
from benchmark_design.line_level.dataset_aspect import (
    collect_dataset_image_sizes,
    validate_external_geometry_rows,
)
from benchmark_design.line_level.geometry import validate_line_metrics_aabb
from benchmark_design.line_level.pipeline import run_line_level_analysis
from benchmark_design.page_level.models import CalibrationResult
from benchmark_design.report.line_level.chapter_figures import export_chapter_distribution_figures
from benchmark_design.report.line_level.chapter_samples import export_chapter_sample_figures
from benchmark_design.report.line_level.chapter_tables import write_chapter_tables
from benchmark_design.report.line_level.export_tables import (
    write_config_snapshot,
    write_dataset_summary,
    write_external_dataset_aspect_tables,
    write_invalid_annotations,
    write_line_metrics,
    write_page_metrics,
    write_processing_errors,
    write_target_pairs,
)
from benchmark_design.report.line_level.output_layout import LineLevelOutputLayout
from benchmark_design.report.line_level.plotting import export_external_dataset_aspect_plots
from benchmark_design.report.line_level.report import (
    configure_analysis_logger,
    write_figure_manifest,
    write_line_analysis_report,
    write_run_manifest,
)


@dataclass(frozen=True, slots=True)
class LineLevelExportResult:
    output_root: Path
    manifest: dict[str, str]


def _relative_manifest(paths: dict[str, Path], output_root: Path) -> dict[str, str]:
    return {key: path.relative_to(output_root).as_posix() for key, path in paths.items()}


def run_line_level_export(
    input_dir: Path,
    output_root: Path,
    *,
    config_path: Path | None = None,
    workers: int | None = None,
    show_progress: bool = True,
    skip_figures: bool = False,
    calibration_path: Path | None = None,
    calibration: CalibrationResult | None = None,
) -> dict[str, str]:
    config = load_line_level_config(
        config_path,
        input_root=input_dir,
        output_root=output_root,
        workers=workers,
        show_progress=show_progress,
        calibration_path=calibration_path,
        calibration=calibration,
    )
    return _run_export_with_config(config, config_path=config_path, skip_figures=skip_figures).manifest


def _run_export_with_config(
    config,
    *,
    config_path: Path | None,
    skip_figures: bool,
) -> LineLevelExportResult:
    layout = LineLevelOutputLayout(config.output_root)
    layout.ensure()
    logger = configure_analysis_logger()
    logger.info("Starting line-level export for %s", config.input_root)

    result = run_line_level_analysis(config)
    line_rows = list(result.line_metrics)
    page_rows = list(result.page_metrics)
    pair_rows = list(result.pair_rows)

    aabb_errors = validate_line_metrics_aabb(line_rows)
    if aabb_errors:
        raise ValueError("AABB geometry validation failed:\n" + "\n".join(aabb_errors[:20]))
    taller = sum(
        1
        for row in line_rows
        if row.is_valid and row.bbox_width_px < row.bbox_height_px
    )
    logger.info(
        "AABB checklist OK: valid lines keep width/height axes "
        "(%d taller-than-wide targets allowed)",
        taller,
    )

    write_line_metrics(line_rows, layout.root)
    write_page_metrics(page_rows, layout.root)
    pair_path = write_target_pairs(pair_rows, layout.root)
    write_invalid_annotations(list(result.invalid_rows), layout.root)
    write_processing_errors(list(result.processing_errors), layout.root)
    write_config_snapshot(config, config_path, layout.root)
    dataset_summary = write_dataset_summary(result, layout.root)

    figure_manifest: dict[str, str] = {}
    table_manifest: dict[str, str] = {}
    if pair_path is not None:
        table_manifest["target_pairs"] = pair_path.relative_to(layout.root).as_posix()

    chapter_table_paths = write_chapter_tables(line_rows, pair_rows, config, layout.root)
    table_manifest.update(_relative_manifest(chapter_table_paths, layout.root))

    if (
        config.external_dataset_aspect_enabled
        and config.external_dataset_root is not None
        and config.external_dataset_root.is_dir()
    ):
        logger.info(
            "Collecting line geometry distributions from %s (ours from line metrics)",
            config.external_dataset_root,
        )
        from benchmark_design.line_level.dataset_aspect import rows_from_line_metrics

        ours_rows = rows_from_line_metrics(line_rows, dataset="ours")
        external_rows = collect_dataset_image_sizes(
            config.external_dataset_root,
            extensions=config.image_extensions,
            workers=config.workers,
            show_progress=config.show_progress,
            ours_rows=ours_rows,
        )
        external_errors = validate_external_geometry_rows(external_rows)
        if external_errors:
            raise ValueError(
                "External geometry validation failed:\n" + "\n".join(external_errors[:20])
            )
        mw_heights = {
            row.height_px
            for row in external_rows
            if row.dataset in {"MathWritting", "MathWriting"}
        }
        if mw_heights:
            logger.info(
                "MathWriting / MathWritting unique height_px values: %s",
                sorted(mw_heights),
            )
        table_paths = write_external_dataset_aspect_tables(external_rows, layout.root)
        table_manifest.update(_relative_manifest(table_paths, layout.root))
        if not skip_figures:
            external_plot_paths = export_external_dataset_aspect_plots(external_rows, layout.plots)
            figure_manifest.update(_relative_manifest(external_plot_paths, layout.root))

    if not skip_figures:
        logger.info("Exporting chapter distribution figures")
        chapter_plot_paths = export_chapter_distribution_figures(line_rows, pair_rows, layout.plots)
        figure_manifest.update(_relative_manifest(chapter_plot_paths, layout.root))
        logger.info("Exporting chapter sample overlays")
        sample_paths = export_chapter_sample_figures(line_rows, pair_rows, config, layout.samples)
        figure_manifest.update(_relative_manifest(sample_paths, layout.root))

    report_path = write_line_analysis_report(result, dataset_summary, layout, figure_manifest)
    write_figure_manifest(figure_manifest, layout.report)
    run_manifest_path = write_run_manifest(result, layout, figure_manifest, config_path=config_path)
    logger.info("Wrote report to %s", report_path)

    manifest = {
        "report": report_path.relative_to(layout.root).as_posix(),
        "run_manifest": run_manifest_path.relative_to(layout.root).as_posix(),
        "line_metrics": "line_metrics.csv",
        "page_metrics": "page_metrics.csv",
        "dataset_summary": "dataset_summary.json",
        "line_bbox_outside_ink": "line_bbox_outside_ink.csv",
    }
    manifest.update(table_manifest)
    manifest.update(figure_manifest)
    return LineLevelExportResult(output_root=layout.root, manifest=manifest)
