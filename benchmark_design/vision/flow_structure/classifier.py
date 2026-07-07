"""Answer-Block Flow Structure page classifier."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.vision.flow_structure.decision import decide_flow_structure
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageFlowStructureResult


def classify_page_flow_structure(
    page: PageAnnotation,
    *,
    input_dir: Path | None = None,
) -> PageFlowStructureResult:
    return decide_flow_structure(page, input_dir=input_dir)
