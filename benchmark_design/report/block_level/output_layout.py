"""Block-level benchmark output directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BlockLevelOutputLayout:
    root: Path

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def details(self) -> Path:
        return self.root / "details"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def docs_metadata(self) -> Path:
        return self.docs / "metadata"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def figures_flow_structure(self) -> Path:
        return self.figures / "flow_structure"

    @property
    def figures_hybrid_layout(self) -> Path:
        return self.figures / "hybrid_layout"

    @property
    def figures_flow_group_examples(self) -> Path:
        """Alias for hierarchical flow_structure figure root."""
        return self.figures_flow_structure

    @property
    def figures_paper(self) -> Path:
        return self.figures / "paper"

    @property
    def flow_structure_summary_md(self) -> Path:
        return self.root / "flow_structure_summary.md"

    @property
    def metadata_json(self) -> Path:
        return self.root / "metadata.json"

    @property
    def block_level_summary_md(self) -> Path:
        return self.root / "block_level_summary.md"

    @property
    def vision_benchmark_summary_md(self) -> Path:
        """Deprecated alias."""
        return self.block_level_summary_md

    def ensure(self) -> None:
        for path in (
            self.root,
            self.tables,
            self.figures,
            self.details,
            self.docs,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def summary_md(self) -> Path:
        return self.block_level_summary_md


VisionBenchmarkOutputLayout = BlockLevelOutputLayout
