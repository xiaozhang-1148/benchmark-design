"""Block-level report writers."""

from __future__ import annotations

from typing import Any

__all__ = ["run_block_level_export", "run_vision_benchmark_export"]


def __getattr__(name: str) -> Any:
    if name in {"run_block_level_export", "run_vision_benchmark_export"}:
        from benchmark_design.report.block_level.export_pipeline import (
            run_block_level_export,
            run_vision_benchmark_export,
        )

        return run_block_level_export if name == "run_block_level_export" else run_vision_benchmark_export
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
