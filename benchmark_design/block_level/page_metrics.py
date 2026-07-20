"""Block-level metric computation (flow structure classification)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmark_design.block_level.flow_structure.models import PageAnnotation, PageFlowStructureResult
from benchmark_design.block_level.flow_structure.pipeline import compute_flow_structure_results
from benchmark_design.block_level.processing_options import BlockLevelProcessingOptions


@dataclass(frozen=True, slots=True)
class BlockLevelBenchmarkResults:
    flow_structure: list[PageFlowStructureResult]


def compute_block_level_benchmark_results(
    input_dir: Path,
    *,
    processing: BlockLevelProcessingOptions | None = None,
    dataset: str = "ours",
    input_dir_for_images: Path | None = None,
    pages: Sequence[PageAnnotation] | None = None,
) -> BlockLevelBenchmarkResults:
    """Compute block-level results (answer-block flow structure classification)."""
    flow_results = compute_flow_structure_results(
        input_dir,
        processing=processing,
        dataset=dataset,
        input_dir_for_images=input_dir_for_images,
        pages=pages,
    )
    return BlockLevelBenchmarkResults(flow_structure=list(flow_results))


# Backward-compatible aliases (deprecated).
VisionBenchmarkResults = BlockLevelBenchmarkResults
compute_vision_benchmark_results = compute_block_level_benchmark_results
