"""Machine-readable pipeline linkage for unified benchmark export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark_design.export_layout import (
    BLOCK_LEVEL_DIR,
    BLOCK_LEVEL_HYBRID_LAYOUT_DIR,
    BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR,
    PAGE_LEVEL_DIR,
    PAGE_LEVEL_HMER_DIR,
    PAGE_LEVEL_LATEX_SPLIT_DIR,
    SPLIT_INPUTS_DIR,
)


def build_pipeline_manifest(
    *,
    output_root: Path,
    hmer_output: Path,
    structure_layout_output: Path,
    hybrid_layout_output: Path,
    page_level_output: Path | None,
    line_level_output: Path | None,
    page_level_hmer_output: Path | None,
    page_level_latex_split_output: Path | None,
    # Backward-compatible aliases (deprecated).
    block_level_output: Path | None = None,
    density_output: Path | None = None,
) -> dict[str, Any]:
    """Describe export stages, final deliverables, and cross-level dependencies."""
    structure_layout_output = block_level_output or structure_layout_output
    page_level_output = density_output or page_level_output

    stages: list[dict[str, Any]] = [
        {
            "id": "HMER",
            "role": "deliverable",
            "path": _rel(output_root, hmer_output),
            "feeds": [PAGE_LEVEL_HMER_DIR, PAGE_LEVEL_LATEX_SPLIT_DIR],
        },
        {
            "id": f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}",
            "role": "supporting",
            "path": _rel(output_root, structure_layout_output),
            "feeds": [PAGE_LEVEL_LATEX_SPLIT_DIR],
            "note": "Single-flow and columnar-flow layout metrics.",
        },
        {
            "id": f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_HYBRID_LAYOUT_DIR}",
            "role": "supporting",
            "path": _rel(output_root, hybrid_layout_output),
            "feeds": [PAGE_LEVEL_LATEX_SPLIT_DIR],
            "note": "Hybrid-layout page metrics.",
        },
    ]
    if page_level_output is not None:
        stages.append(
            {
                "id": PAGE_LEVEL_DIR,
                "role": "supporting",
                "path": _rel(output_root, page_level_output),
                "feeds": [PAGE_LEVEL_LATEX_SPLIT_DIR, "line_level"],
                "note": "Full-page foreground density and calibration.",
            }
        )
    if line_level_output is not None:
        stages.append(
            {
                "id": "line_level",
                "role": "deliverable",
                "path": _rel(output_root, line_level_output),
                "feeds": [PAGE_LEVEL_LATEX_SPLIT_DIR],
            }
        )
    if page_level_hmer_output is not None:
        stages.append(
            {
                "id": PAGE_LEVEL_HMER_DIR,
                "role": "deliverable",
                "path": _rel(output_root, page_level_hmer_output),
                "depends_on": ["HMER"],
                "feeds": [PAGE_LEVEL_LATEX_SPLIT_DIR],
            }
        )
    if page_level_latex_split_output is not None:
        split_inputs = page_level_latex_split_output / SPLIT_INPUTS_DIR
        depends_on = [PAGE_LEVEL_HMER_DIR, "HMER", "line_level"]
        if page_level_output is not None:
            depends_on.append(PAGE_LEVEL_DIR)
        depends_on.extend(
            [
                f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}",
                f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_HYBRID_LAYOUT_DIR}",
            ]
        )
        stages.append(
            {
                "id": PAGE_LEVEL_LATEX_SPLIT_DIR,
                "role": "deliverable",
                "path": _rel(output_root, page_level_latex_split_output),
                "depends_on": depends_on,
                "inputs": {
                    "split_inputs_dir": _rel(output_root, split_inputs),
                },
            }
        )

    deliverables = [stage["id"] for stage in stages if stage["role"] == "deliverable"]
    return {
        "version": 2,
        "output_root": str(output_root.resolve()),
        "primary_directories": [
            PAGE_LEVEL_DIR,
            BLOCK_LEVEL_DIR,
            "line_level",
            "HMER",
            PAGE_LEVEL_HMER_DIR,
            PAGE_LEVEL_LATEX_SPLIT_DIR,
        ],
        "deliverables": deliverables,
        "stages": stages,
    }


def write_pipeline_manifest(payload: dict[str, Any], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "pipeline_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
