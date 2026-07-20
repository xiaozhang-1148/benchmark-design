"""Output directory layout for page-level image analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PageLevelOutputLayout:
    root: Path

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def figures_paper(self) -> Path:
        return self.figures / "paper"

    @property
    def report(self) -> Path:
        return self.root / "report"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.calibration,
            self.tables,
            self.figures,
            self.figures_paper,
            self.report,
            self.logs,
        ):
            path.mkdir(parents=True, exist_ok=True)
