"""V2 layout / OCR-quality / recognition analysis (does not overwrite v1 artifacts)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


def run_analysis_v2(cfg: dict[str, Any]) -> None:
    from .run import run_analysis_v2 as _run

    _run(cfg)


__all__ = ["run_analysis_v2"]
