"""Vision benchmark output directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VisionBenchmarkOutputLayout:
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
    def deleted_block_scale_summary_md(self) -> Path:
        return self.root / "deleted_block_scale_summary.md"

    @property
    def figures_deleted_block_scale(self) -> Path:
        return self.figures / "deleted_block_scale"

    @property
    def foreground_pixel_density_summary_md(self) -> Path:
        return self.root / "foreground_pixel_density_summary.md"

    @property
    def foreground_load_summary_md(self) -> Path:
        """Deprecated alias."""
        return self.foreground_pixel_density_summary_md

    @property
    def figures_foreground_load(self) -> Path:
        """Deprecated alias for figures root (density figures live directly under figures/)."""
        return self.figures

    @property
    def figures_flow_structure(self) -> Path:
        return self.figures / "flow_structure"

    @property
    def figures_flow_group_examples(self) -> Path:
        """Alias for hierarchical flow_structure figure root."""
        return self.figures_flow_structure

    @property
    def flow_structure_summary_md(self) -> Path:
        return self.root / "flow_structure_summary.md"

    @property
    def metadata_json(self) -> Path:
        return self.root / "metadata.json"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.tables,
            self.figures,
            self.metadata_dir,
            self.figures_flow_group_examples,
            self.figures_deleted_block_scale,
            self.details,
            self.docs,
            self.docs_metadata,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def summary_md(self) -> Path:
        return self.root / "vision_benchmark_summary.md"
