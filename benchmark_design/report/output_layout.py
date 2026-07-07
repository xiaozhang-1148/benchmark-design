"""Standard benchmark output directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkOutputLayout:
    root: Path

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def tables_appendix(self) -> Path:
        return self.tables / "appendix"

    @property
    def cross_benchmark(self) -> Path:
        return self.root / "cross_benchmark"

    @property
    def details(self) -> Path:
        return self.root / "details"

    @property
    def examples(self) -> Path:
        return self.root / "examples"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def resources(self) -> Path:
        return self.root / "resources"

    @property
    def docs_metadata(self) -> Path:
        return self.docs / "metadata"

    @property
    def summary_md(self) -> Path:
        return self.root / "summary.md"

    @property
    def ocr_benchmark_summary_md(self) -> Path:
        return self.root / "ocr_benchmark_summary.md"

    @property
    def cross_benchmark_summary_md(self) -> Path:
        return self.root / "cross_benchmark_summary.md"

    @property
    def metadata_json(self) -> Path:
        return self.root / "metadata.json"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.tables,
            self.tables_appendix,
            self.cross_benchmark,
            self.details,
            self.examples,
            self.figures,
            self.docs,
            self.resources,
            self.docs_metadata,
        ):
            path.mkdir(parents=True, exist_ok=True)


def tables_dir(output_root: Path) -> Path:
    path = output_root / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_output_path(path: Path, output_root: Path) -> str:
    """Return *path* relative to *output_root* for manifest and metadata."""
    path = Path(path)
    output_root = Path(output_root)
    for base in (output_root.resolve(), output_root):
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            try:
                return str(path.relative_to(base))
            except ValueError:
                continue
    return str(path)


def relative_input_path(path: Path, *, anchor: Path | None = None) -> str:
    """Return *path* relative to *anchor* or the current working directory when possible."""
    path = Path(path)
    anchors: list[Path] = []
    if anchor is not None:
        anchors.append(Path(anchor))
    anchors.append(Path.cwd())
    for base in anchors:
        for candidate in (base.resolve(), base):
            try:
                return str(path.resolve().relative_to(candidate.resolve()))
            except ValueError:
                try:
                    return str(path.relative_to(candidate))
                except ValueError:
                    continue
    return str(path)


def relativize_source_file(source_file: str, *, input_dir: Path | None = None) -> str:
    if not source_file:
        return source_file
    path = Path(source_file)
    if input_dir is not None:
        for base in (input_dir.resolve(), input_dir):
            try:
                return str(path.resolve().relative_to(base.resolve()))
            except ValueError:
                try:
                    return str(path.relative_to(base))
                except ValueError:
                    pass
    return relative_input_path(path)
