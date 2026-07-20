"""Unified benchmark export orchestration facade.

The full multi-level pipeline lives in ``benchmark_design.project.export``;
this module exposes a stable entry point, path layout helpers, and the
documented dependency graph for tests and downstream tooling.
"""

from __future__ import annotations

from benchmark_design.project.export import run_project_export
from benchmark_design.export_layout import (
    BENCHMARK_EXPORT_DEPENDENCIES,
    BENCHMARK_EXPORT_PRIMARY_DIRS,
    PAGE_LEVEL_HMER_DIR,
    PAGE_LEVEL_LATEX_ALIAS,
    PAGE_LEVEL_LATEX_SPLIT_DIR,
    BenchmarkExportLayout,
    cross_domain_inputs_available,
    write_export_pipeline_doc,
)
from benchmark_design.project.models import ProjectExportResult

__all__ = [
    "BENCHMARK_EXPORT_DEPENDENCIES",
    "BENCHMARK_EXPORT_PRIMARY_DIRS",
    "BenchmarkExportLayout",
    "PAGE_LEVEL_HMER_DIR",
    "PAGE_LEVEL_LATEX_ALIAS",
    "PAGE_LEVEL_LATEX_SPLIT_DIR",
    "ProjectExportResult",
    "cross_domain_inputs_available",
    "run_benchmark_export_pipeline",
    "write_export_pipeline_doc",
]

# Alias: same orchestrator as ``project export``.
run_benchmark_export_pipeline = run_project_export
