"""Shared processing configuration for parallel OCR analysis."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_design.progress import default_worker_count


@dataclass(frozen=True, slots=True)
class ProcessingOptions:
    show_progress: bool = False
    workers: int | None = None

    @property
    def worker_count(self) -> int:
        return self.workers if self.workers is not None else default_worker_count()
