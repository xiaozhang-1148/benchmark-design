"""Resolved v2 output directories under output_root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import ensure_dir


def analysis_v2_dir(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["paths"]["outputs_dir"]) / "analysis_v2"
    return ensure_dir(p)


def transformers_dir(cfg: dict[str, Any]) -> Path:
    return ensure_dir(analysis_v2_dir(cfg) / "transformers")


def reports_v2_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["paths"]["output_root"])
    override = (cfg.get("paths") or {}).get("reports_v2_dir")
    p = Path(override) if override else root / "reports_v2"
    return ensure_dir(p)
