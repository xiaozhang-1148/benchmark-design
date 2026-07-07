"""One image sample in the vision benchmark corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImageSampleRecord:
    """Minimal record for vision-side benchmark analysis."""

    sample_id: str
    image_path: Path
    dataset: str
    source_file: str
    width_px: int | None = None
    height_px: int | None = None
    page_id: str = ""
    expression_id: str = ""

    def resolved_image_path(self) -> Path:
        return self.image_path.resolve()
