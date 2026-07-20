"""Output directory layout for line-level analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LineLevelOutputLayout:
    root: Path

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def samples(self) -> Path:
        return self.root / "samples"

    @property
    def report(self) -> Path:
        return self.root / "report"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.plots.mkdir(parents=True, exist_ok=True)
