"""Load project-level YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ProjectPipelineConfig:
    hmer: dict[str, Any] = field(default_factory=dict)
    block_level: dict[str, Any] = field(default_factory=dict)
    page_level: dict[str, Any] = field(default_factory=dict)
    line_level: dict[str, Any] = field(default_factory=dict)
    page_level_hmer: dict[str, Any] = field(default_factory=dict)
    page_level_latex_split: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    input_root: Path
    output_root: Path
    workers: int | None
    pipelines: ProjectPipelineConfig


def load_project_config(path: Path) -> ProjectConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pipelines_payload = payload.get("pipelines", {}) or {}
    return ProjectConfig(
        input_root=Path(payload["input_root"]),
        output_root=Path(payload.get("output_root", "benchmark_export")),
        workers=payload.get("workers"),
        pipelines=ProjectPipelineConfig(
            hmer=dict(pipelines_payload.get("hmer", {}) or {}),
            block_level=dict(pipelines_payload.get("block_level", {}) or {}),
            page_level=dict(pipelines_payload.get("page_level", {}) or {}),
            line_level=dict(pipelines_payload.get("line_level", {}) or {}),
            page_level_hmer=dict(pipelines_payload.get("page_level_hmer", {}) or {}),
            page_level_latex_split=dict(pipelines_payload.get("page_level_latex_split", {}) or {}),
        ),
    )
