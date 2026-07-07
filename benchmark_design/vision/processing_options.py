"""Vision pipeline runtime options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VisionProcessingOptions:
    show_progress: bool = True
    workers: int | None = None
    read_image_dimensions: bool = True
