"""Dataclasses for project-level benchmark export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectSummary:
    generated_at: str
    input_root: str
    output_root: str
    page_count: int | None
    pipelines: dict[str, Any]
    overview: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectExportResult:
    output_root: Path
    hmer_output: Path
    structure_layout_output: Path
    hybrid_layout_output: Path
    page_level_output: Path | None
    line_level_output: Path | None
    page_level_hmer_output: Path | None
    page_level_latex_split_output: Path | None
    hmer_manifest: dict[str, Path]
    structure_layout_manifest: dict[str, str]
    hybrid_layout_manifest: dict[str, str]
    page_level_manifest: dict[str, str] | None
    block_density_manifest: dict[str, str] | None
    line_level_manifest: dict[str, str] | None
    page_level_hmer_manifest: dict[str, str] | None
    page_level_latex_split_manifest: dict[str, str] | None
    dataset_overview: Path
    summary_json: Path
    pipeline_doc: Path

    @property
    def block_level_output(self) -> Path:
        return self.structure_layout_output

    @property
    def density_output(self) -> Path | None:
        """Deprecated alias for page_level_output."""
        return self.page_level_output

    @property
    def density_manifest(self) -> dict[str, str] | None:
        """Deprecated alias for page_level_manifest."""
        return self.page_level_manifest

    @property
    def block_level_manifest(self) -> dict[str, str]:
        return self.structure_layout_manifest

    @property
    def vision_output(self) -> Path:
        return self.structure_layout_output

    @property
    def vision_manifest(self) -> dict[str, str]:
        return self.structure_layout_manifest


# Backward-compatible alias (deprecated).
UnifiedExportResult = ProjectExportResult
