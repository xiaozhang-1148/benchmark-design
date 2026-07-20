"""Backward-compatible unified export wrapper."""

from __future__ import annotations

from benchmark_design.project.export import run_project_export
from benchmark_design.project.models import ProjectExportResult, UnifiedExportResult

__all__ = [
    "ProjectExportResult",
    "UnifiedExportResult",
    "run_project_export",
    "run_unified_benchmark_export",
]

run_unified_benchmark_export = run_project_export
