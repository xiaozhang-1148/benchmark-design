"""Page-level image analysis export pipeline (aspect ratio + foreground density)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark_design.foreground.analysis import export_foreground_analysis
from benchmark_design.foreground.calibration import calibration_to_threshold_config
from benchmark_design.foreground.threshold import save_foreground_threshold_config
from benchmark_design.page_level.config import load_page_level_config
from benchmark_design.page_level.models import PageLevelConfig
from benchmark_design.page_level.pipeline import run_page_level_analysis
from benchmark_design.report.page_level.export_tables import (
    write_aspect_ratio_groups,
    write_calibration_outputs,
    write_feature_tables,
    write_inventory_tables,
    write_statistics_tables,
)
from benchmark_design.export_layout import BenchmarkExportLayout
from benchmark_design.report.page_level.output_layout import PageLevelOutputLayout
from benchmark_design.report.page_level.plotting import export_paper_figures
from benchmark_design.report.page_level.report import (
    configure_analysis_logger,
    write_figure_manifest,
    write_image_analysis_report,
    write_run_manifest,
)


@dataclass(frozen=True, slots=True)
class PageLevelExportResult:
    output_root: Path
    image_count: int
    manifest: dict[str, str]


def _relative_manifest(paths: dict[str, Path], output_root: Path) -> dict[str, str]:
    return {
        key: path.relative_to(output_root).as_posix()
        for key, path in paths.items()
    }


def run_page_level_export(
    input_dir: Path,
    output_root: Path,
    *,
    config_path: Path | None = None,
    workers: int | None = None,
    show_progress: bool = True,
    skip_figures: bool = False,
    gray_cache_root: Path | None = None,
    cleanup_gray_cache: bool = True,
) -> dict[str, str]:
    """Run the page-level export and return a manifest of key artifacts."""
    config = load_page_level_config(
        config_path,
        input_root=input_dir,
        output_root=output_root,
        workers=workers,
        show_progress=show_progress,
    )
    result = _run_export_with_config(
        config,
        skip_figures=skip_figures,
        config_path=config_path,
        gray_cache_root=gray_cache_root,
        cleanup_gray_cache=cleanup_gray_cache,
    )
    return result.manifest


def _run_export_with_config(
    config: PageLevelConfig,
    *,
    skip_figures: bool,
    config_path: Path | None = None,
    gray_cache_root: Path | None = None,
    cleanup_gray_cache: bool = True,
) -> PageLevelExportResult:
    layout = PageLevelOutputLayout(config.output_root)
    layout.ensure()
    logger = configure_analysis_logger(layout.logs / "analysis.log")
    logger.info("Starting page-level export for %s", config.input_root)

    analysis = run_page_level_analysis(
        config,
        gray_cache_root=gray_cache_root,
        cleanup_gray_cache=cleanup_gray_cache,
    )
    inventory = list(analysis.inventory)
    features = list(analysis.features)
    calibration = analysis.calibration
    logger.info("Analyzed %s images", len(features))

    write_inventory_tables(inventory, layout.tables)
    write_feature_tables(features, layout.tables)
    write_calibration_outputs(calibration, layout.calibration)

    threshold_config = calibration_to_threshold_config(calibration)
    export_layout = BenchmarkExportLayout(config.output_root.parent)
    foreground_dir = export_layout.foreground
    save_foreground_threshold_config(threshold_config, export_layout.foreground_threshold)
    export_foreground_analysis(
        output_dir=foreground_dir,
        calibration=calibration,
        threshold_config=threshold_config,
        features=features,
    )

    calibration_payload = {
        "dark_reference": calibration.dark_reference,
        "light_reference": calibration.light_reference,
        "gray_threshold": calibration.gray_threshold,
        "darkness_threshold": calibration.tau_d,
        "tau_D": calibration.tau_d,
        "global_threshold": calibration.global_threshold,
        "foreground_valley_threshold": calibration.foreground_valley_threshold,
        "dark_percentile": calibration.dark_percentile,
        "light_percentile": calibration.light_percentile,
        "threshold_method": calibration.threshold_method,
        "image_count": calibration.image_count,
    }
    dataset_summary = write_statistics_tables(
        inventory,
        features,
        layout.tables,
        layout.report,
        calibration_payload,
    )

    if config.aspect_ratio_groups_enabled:
        write_aspect_ratio_groups(features, layout.tables)

    figure_manifest: dict[str, str] = {}
    if not skip_figures:
        paper_paths = export_paper_figures(
            features,
            layout.figures_paper,
            gray_threshold=calibration.gray_threshold,
        )
        figure_manifest = _relative_manifest(paper_paths, layout.root)

    report_path = write_image_analysis_report(
        layout=layout,
        config=config,
        inventory=inventory,
        features=features,
        calibration=calibration,
        dataset_summary=dataset_summary,
        figure_manifest=figure_manifest,
    )
    write_figure_manifest(figure_manifest, layout.report)
    run_manifest_path = write_run_manifest(
        layout=layout,
        config=config,
        image_count=len(features),
        calibration=calibration,
        figure_manifest=figure_manifest,
        config_path=config_path,
    )
    logger.info("Wrote report to %s", report_path)

    manifest = {
        "report": report_path.relative_to(layout.root).as_posix(),
        "run_manifest": run_manifest_path.relative_to(layout.root).as_posix(),
        "image_inventory": "tables/image_inventory.parquet",
        "image_features": "tables/image_features.parquet",
        "dataset_summary": "report/dataset_summary.json",
        "calibration": "calibration/calibration.json",
        "foreground_threshold": "../foreground/threshold.json",
        "foreground_analysis": "../foreground/gray_hist_log.png",
    }
    manifest.update(figure_manifest)
    return PageLevelExportResult(
        output_root=layout.root,
        image_count=len(features),
        manifest=manifest,
    )
