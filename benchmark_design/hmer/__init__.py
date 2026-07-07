"""HMER domain — handwritten mathematical expression recognition benchmarks.

Implementation lives under ``benchmark_design.ocr`` (LaTeX tokenization, structure
metrics, cross-benchmark reports). Import from ``benchmark_design.hmer`` for the
domain-oriented API; ``benchmark_design.ocr`` remains stable for existing code.
"""

from __future__ import annotations

from benchmark_design.ocr import processing as processing
from benchmark_design.ocr.consolidated import compute_ocr_consolidated_metrics
from benchmark_design.ocr.cross_benchmark import compute_cross_benchmark_profiles
from benchmark_design.ocr.expression_features import ExpressionFeatures
from benchmark_design.ocr.processing import ProcessingOptions, build_enriched_corpus
from benchmark_design.report.export_pipeline import run_benchmark_export

__all__ = [
    "ExpressionFeatures",
    "ProcessingOptions",
    "build_enriched_corpus",
    "compute_cross_benchmark_profiles",
    "compute_ocr_consolidated_metrics",
    "processing",
    "run_benchmark_export",
]
