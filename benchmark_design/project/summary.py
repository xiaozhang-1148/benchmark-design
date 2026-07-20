"""Build machine-readable project summary.json from pipeline manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark_design.export_layout import (
    BenchmarkExportLayout,
    BLOCK_LEVEL_DIR,
    BLOCK_LEVEL_HYBRID_LAYOUT_DIR,
    BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR,
    PAGE_LEVEL_DIR,
)
from benchmark_design.project.models import ProjectSummary
from benchmark_design.report.dataset_overview import DatasetOverviewMetrics


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _pipeline_entry(
    *,
    manifest_rel: str | None,
    payload: dict[str, Any] | None,
    key_metrics: dict[str, Any],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "manifest": manifest_rel,
        "key_metrics": key_metrics,
    }
    if payload is not None:
        entry["metadata"] = payload
    return entry


def build_project_summary(
    *,
    input_root: Path,
    output_root: Path,
    hmer_output: Path,
    structure_layout_output: Path,
    hybrid_layout_output: Path,
    page_level_output: Path | None,
    line_level_output: Path | None,
    page_level_hmer_output: Path | None = None,
    page_level_latex_split_output: Path | None = None,
    overview_metrics: DatasetOverviewMetrics | None = None,
    split_selected_seed: int | None = None,
    # Backward-compatible aliases (deprecated).
    block_level_output: Path | None = None,
    density_output: Path | None = None,
) -> ProjectSummary:
    warnings: list[str] = []
    layout = BenchmarkExportLayout(output_root)
    structure_layout_output = block_level_output or structure_layout_output
    page_level_output = page_level_output or density_output

    hmer_metadata_path = hmer_output / "metadata.json"
    structure_metadata_path = structure_layout_output / "metadata.json"
    hybrid_metadata_path = hybrid_layout_output / "metadata.json"
    page_level_run_manifest_path = (
        page_level_output / "report" / "run_manifest.json" if page_level_output is not None else None
    )
    page_level_dataset_summary_path = (
        page_level_output / "report" / "dataset_summary.json" if page_level_output is not None else None
    )
    line_run_manifest_path = (
        line_level_output / "report" / "run_manifest.json" if line_level_output is not None else None
    )
    line_dataset_summary_path = (
        line_level_output / "dataset_summary.json" if line_level_output is not None else None
    )
    split_manifest_path = (
        page_level_latex_split_output / "split_manifest.csv"
        if page_level_latex_split_output is not None
        else None
    )

    hmer_metadata = _read_json(hmer_metadata_path)
    if hmer_metadata is None:
        warnings.append(f"missing or invalid HMER metadata: {hmer_metadata_path}")

    structure_metadata = _read_json(structure_metadata_path)
    if structure_metadata is None:
        warnings.append(f"missing or invalid structure_layout metadata: {structure_metadata_path}")

    hybrid_metadata = _read_json(hybrid_metadata_path)
    if hybrid_metadata is None:
        warnings.append(f"missing or invalid hybrid_layout metadata: {hybrid_metadata_path}")

    page_level_run_manifest = (
        _read_json(page_level_run_manifest_path) if page_level_run_manifest_path else None
    )
    if page_level_output is not None and page_level_run_manifest is None:
        warnings.append(f"missing or invalid page_level run manifest: {page_level_run_manifest_path}")

    page_level_dataset_summary = (
        _read_json(page_level_dataset_summary_path) if page_level_dataset_summary_path else None
    )

    line_run_manifest = _read_json(line_run_manifest_path) if line_run_manifest_path else None
    if line_level_output is not None and line_run_manifest is None:
        warnings.append(f"missing or invalid line_level run manifest: {line_run_manifest_path}")

    line_dataset_summary = _read_json(line_dataset_summary_path) if line_dataset_summary_path else None

    page_count = None
    if overview_metrics is not None:
        page_count = overview_metrics.hmer.page_count
    elif hmer_metadata is not None:
        page_count = hmer_metadata.get("json_file_count")
    elif structure_metadata is not None:
        page_count = structure_metadata.get("page_count")

    overview: dict[str, Any] = {}
    if overview_metrics is not None:
        overview = {
            "hmer": {
                "page_count": overview_metrics.hmer.page_count,
                "expression_count": overview_metrics.hmer.expression_count,
                "total_characters": overview_metrics.hmer.total_characters,
            },
            "block": {
                "txtblock_count": overview_metrics.block.txtblock_count,
                "total_block_count": overview_metrics.block.total_block_count,
            },
            "structure_layout_page_metrics": {
                "page_count": overview_metrics.vision.page_count,
                "avg_aspect_ratio": round(overview_metrics.vision.avg_aspect_ratio, 4),
                "avg_megapixels": round(overview_metrics.vision.avg_megapixels, 4),
            },
        }

    structure_rel = layout.structure_layout.relative_to(output_root).as_posix()
    hybrid_rel = layout.hybrid_layout.relative_to(output_root).as_posix()
    page_level_rel = layout.page_level.relative_to(output_root).as_posix()

    pipelines: dict[str, Any] = {
        "HMER": _pipeline_entry(
            manifest_rel=f"{layout.hmer.name}/metadata.json",
            payload=hmer_metadata,
            key_metrics={
                "expression_count": (hmer_metadata or {}).get("expression_count"),
                "parse_success_rate": (hmer_metadata or {}).get("parse_success_rate"),
            },
        ),
        f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}": _pipeline_entry(
            manifest_rel=f"{structure_rel}/metadata.json",
            payload=structure_metadata,
            key_metrics={
                "page_count": (structure_metadata or {}).get("page_count"),
                "flow_group_counts": (structure_metadata or {}).get("flow_group_counts"),
            },
        ),
        f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_HYBRID_LAYOUT_DIR}": _pipeline_entry(
            manifest_rel=f"{hybrid_rel}/metadata.json",
            payload=hybrid_metadata,
            key_metrics={
                "page_count": (hybrid_metadata or {}).get("page_count"),
                "flow_group_counts": (hybrid_metadata or {}).get("flow_group_counts"),
            },
        ),
    }

    if page_level_output is not None:
        pipelines[PAGE_LEVEL_DIR] = _pipeline_entry(
            manifest_rel=f"{page_level_rel}/report/run_manifest.json",
            payload=page_level_run_manifest,
            key_metrics={
                "image_count": (page_level_run_manifest or {}).get("image_count"),
                "scope": (page_level_run_manifest or {}).get("scope"),
                "dataset_summary": (
                    f"{page_level_rel}/report/dataset_summary.json"
                    if page_level_dataset_summary is not None
                    else None
                ),
            },
        )

    if line_level_output is not None:
        pipelines["line_level"] = _pipeline_entry(
            manifest_rel=f"{layout.line_level.name}/report/run_manifest.json",
            payload=line_run_manifest,
            key_metrics={
                "line_count": (line_dataset_summary or {}).get("line_count"),
                "page_count": (line_dataset_summary or {}).get("discovered_page_count"),
                "dataset_summary": (
                    f"{layout.line_level.name}/dataset_summary.json"
                    if line_dataset_summary is not None
                    else None
                ),
            },
        )

    if page_level_hmer_output is not None:
        pipelines["page_level_HMER"] = _pipeline_entry(
            manifest_rel=f"{layout.page_level_hmer.name}/metrics/page_latex_metrics.csv",
            payload=None,
            key_metrics={
                "output_root": str(page_level_hmer_output.resolve()),
            },
        )

    if page_level_latex_split_output is not None:
        split_metrics: dict[str, Any] = {}
        if split_selected_seed is not None:
            split_metrics["selected_seed"] = split_selected_seed
        if split_manifest_path is not None and split_manifest_path.is_file():
            split_metrics["split_manifest"] = split_manifest_path.relative_to(output_root).as_posix()
        pipelines["page_level_latex_split"] = _pipeline_entry(
            manifest_rel=(
                split_metrics.get("split_manifest")
                if split_manifest_path is not None and split_manifest_path.is_file()
                else None
            ),
            payload=None,
            key_metrics=split_metrics,
        )

    return ProjectSummary(
        generated_at=datetime.now(tz=UTC).isoformat(),
        input_root=str(input_root.resolve()),
        output_root=str(output_root.resolve()),
        page_count=page_count,
        pipelines=pipelines,
        overview=overview,
        warnings=tuple(warnings),
    )


def write_project_summary(summary: ProjectSummary, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "summary.json"
    payload = {
        "generated_at": summary.generated_at,
        "input_root": summary.input_root,
        "output_root": summary.output_root,
        "page_count": summary.page_count,
        "pipelines": summary.pipelines,
        "overview": summary.overview,
    }
    if summary.warnings:
        payload["warnings"] = list(summary.warnings)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
