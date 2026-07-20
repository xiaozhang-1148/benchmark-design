"""Unified benchmark export directory layout and cross-layer path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HMER_DIR = "HMER"
LINE_LEVEL_DIR = "line_level"
PAGE_LEVEL_DIR = "page_level"
BLOCK_LEVEL_DIR = "block_level"
BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR = "structure_layout"
BLOCK_LEVEL_HYBRID_LAYOUT_DIR = "hybrid_layout"
PAGE_LEVEL_HMER_DIR = "page_level_HMER"
PAGE_LEVEL_LATEX_SPLIT_DIR = "page_level_latex_split"
FOREGROUND_DIR = "foreground"
SPLIT_INPUTS_DIR = "inputs"

# Standalone module name; orchestrator writes PAGE_LEVEL_HMER_DIR.
PAGE_LEVEL_LATEX_ALIAS = "page_level_latex"

STRUCTURE_LAYOUT_FLOW_STRUCTURES = frozenset({"Single-flow", "Columnar-flow", "NA"})
HYBRID_LAYOUT_FLOW_STRUCTURES = frozenset({"Hybrid-flow"})

FLOW_STRUCTURE_EXPORT_LABELS = {
    "Hybrid-flow": "Hybrid-layout",
}

# Dependency edges: stage -> prerequisites that must finish first.
BENCHMARK_EXPORT_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    PAGE_LEVEL_DIR: (),
    LINE_LEVEL_DIR: (),
    HMER_DIR: (),
    f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}": (),
    f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_HYBRID_LAYOUT_DIR}": (),
    PAGE_LEVEL_HMER_DIR: (HMER_DIR,),
    f"{PAGE_LEVEL_LATEX_SPLIT_DIR}/{SPLIT_INPUTS_DIR}": (PAGE_LEVEL_HMER_DIR,),
    PAGE_LEVEL_LATEX_SPLIT_DIR: (
        f"{PAGE_LEVEL_LATEX_SPLIT_DIR}/{SPLIT_INPUTS_DIR}",
        PAGE_LEVEL_DIR,
        LINE_LEVEL_DIR,
        HMER_DIR,
        f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}",
        f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_HYBRID_LAYOUT_DIR}",
    ),
}

# Top-level export roots referenced in papers and cross-domain joins.
BENCHMARK_EXPORT_PRIMARY_DIRS: tuple[str, ...] = (
    PAGE_LEVEL_DIR,
    BLOCK_LEVEL_DIR,
    LINE_LEVEL_DIR,
    HMER_DIR,
    PAGE_LEVEL_HMER_DIR,
    PAGE_LEVEL_LATEX_SPLIT_DIR,
)


@dataclass(frozen=True, slots=True)
class BenchmarkExportLayout:
    """Canonical layout under a single benchmark export root."""

    export_root: Path

    @property
    def hmer(self) -> Path:
        return self.export_root / HMER_DIR

    @property
    def line_level(self) -> Path:
        return self.export_root / LINE_LEVEL_DIR

    @property
    def page_level(self) -> Path:
        return self.export_root / PAGE_LEVEL_DIR

    @property
    def density(self) -> Path:
        """Deprecated alias for the page-level export root."""
        return self.page_level

    @property
    def block_level(self) -> Path:
        return self.export_root / BLOCK_LEVEL_DIR

    @property
    def structure_layout(self) -> Path:
        return self.block_level / BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR

    @property
    def hybrid_layout(self) -> Path:
        return self.block_level / BLOCK_LEVEL_HYBRID_LAYOUT_DIR

    @property
    def page_level_hmer(self) -> Path:
        return self.export_root / PAGE_LEVEL_HMER_DIR

    @property
    def page_level_latex_split(self) -> Path:
        return self.export_root / PAGE_LEVEL_LATEX_SPLIT_DIR

    @property
    def split_inputs(self) -> Path:
        return self.page_level_latex_split / SPLIT_INPUTS_DIR

    @property
    def foreground(self) -> Path:
        return self.export_root / FOREGROUND_DIR

    @property
    def foreground_threshold(self) -> Path:
        return self.foreground / "threshold.json"

    @property
    def density_calibration(self) -> Path:
        return self.page_level / "calibration" / "calibration.json"

    def relative_density_report_md(self) -> str:
        return f"{PAGE_LEVEL_DIR}/report/image_analysis_report.md"

    def relative_structure_layout_summary_md(self) -> str:
        return (
            f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}/block_level_summary.md"
        )

    def relative_hybrid_layout_summary_md(self) -> str:
        return f"{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_HYBRID_LAYOUT_DIR}/block_level_summary.md"

    def relative_line_level_report_md(self) -> str:
        return f"{LINE_LEVEL_DIR}/report/line_analysis_report.md"


def resolve_export_artifact(export_root: Path, candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def prune_empty_directories(root: Path) -> int:
    """Remove empty directories under *root* (deepest first). Returns directories removed."""
    if not root.is_dir():
        return 0
    removed = 0
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in directories:
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()
            removed += 1
    return removed


def image_features_csv(export_root: Path) -> Path:
    layout = BenchmarkExportLayout(export_root)
    return resolve_export_artifact(
        export_root,
        (
            layout.page_level / "tables" / "image_features.csv",
            layout.page_level / "density" / "tables" / "image_features.csv",
            export_root / PAGE_LEVEL_DIR / "tables" / "image_features.csv",
        ),
    )


def _flow_structure_page_metrics_paths(export_root: Path) -> tuple[Path, ...]:
    layout = BenchmarkExportLayout(export_root)
    return (
        layout.structure_layout / "tables" / "flow_structure_page_metrics.csv",
        layout.hybrid_layout / "tables" / "flow_structure_page_metrics.csv",
        export_root
        / PAGE_LEVEL_DIR
        / BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR
        / "tables"
        / "flow_structure_page_metrics.csv",
        export_root
        / PAGE_LEVEL_DIR
        / BLOCK_LEVEL_HYBRID_LAYOUT_DIR
        / "tables"
        / "flow_structure_page_metrics.csv",
        export_root / BLOCK_LEVEL_DIR / "tables" / "flow_structure_page_metrics.csv",
    )


def flow_structure_page_metrics_csv(export_root: Path) -> Path:
    return resolve_export_artifact(export_root, _flow_structure_page_metrics_paths(export_root))


def load_flow_structure_page_metrics(export_root: Path) -> pd.DataFrame:
    paths = [path for path in _flow_structure_page_metrics_paths(export_root) if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No flow_structure_page_metrics.csv under {export_root}")
    frames = [pd.read_csv(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["page_id"], keep="first")


def _flow_structure_block_geometry_paths(export_root: Path) -> tuple[Path, ...]:
    layout = BenchmarkExportLayout(export_root)
    return (
        layout.structure_layout / "tables" / "flow_structure_block_geometry.csv",
        layout.hybrid_layout / "tables" / "flow_structure_block_geometry.csv",
        export_root
        / PAGE_LEVEL_DIR
        / BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR
        / "tables"
        / "flow_structure_block_geometry.csv",
        export_root
        / PAGE_LEVEL_DIR
        / BLOCK_LEVEL_HYBRID_LAYOUT_DIR
        / "tables"
        / "flow_structure_block_geometry.csv",
        export_root / BLOCK_LEVEL_DIR / "tables" / "flow_structure_block_geometry.csv",
    )


def flow_structure_block_geometry_csv(export_root: Path) -> Path:
    return resolve_export_artifact(export_root, _flow_structure_block_geometry_paths(export_root))


def load_flow_structure_block_geometry(export_root: Path) -> pd.DataFrame:
    paths = [path for path in _flow_structure_block_geometry_paths(export_root) if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No flow_structure_block_geometry.csv under {export_root}")
    frames = [pd.read_csv(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def line_metrics_csv(export_root: Path) -> Path:
    return BenchmarkExportLayout(export_root).line_level / "line_metrics.csv"


def expression_level_statistics_csv(export_root: Path) -> Path:
    return BenchmarkExportLayout(export_root).hmer / "details" / "expression_level_statistics.csv"


def cross_domain_inputs_available(export_root: Path) -> bool:
    flow_paths = _flow_structure_page_metrics_paths(export_root)
    flow_available = any(path.is_file() for path in flow_paths)
    return all(
        condition
        for condition in (
            image_features_csv(export_root).is_file(),
            flow_available,
            line_metrics_csv(export_root).is_file(),
            expression_level_statistics_csv(export_root).is_file(),
        )
    )


def export_flow_structure_label(flow_structure: str) -> str:
    return FLOW_STRUCTURE_EXPORT_LABELS.get(flow_structure, flow_structure)


def write_export_pipeline_doc(export_root: Path) -> Path:
    """Document how export layers connect via page_id."""
    export_root.mkdir(parents=True, exist_ok=True)
    layout = BenchmarkExportLayout(export_root)
    path = export_root / "PIPELINE.md"
    lines = [
        "# Benchmark export pipeline",
        "",
        "All layers share the benchmark JSON input directory. Rows join on `page_id` "
        "(line-level tables use `image_id`, which matches `page_id` without extension).",
        "",
        "## Directory layout",
        "",
        "```",
        f"{export_root.name}/",
        f"  {PAGE_LEVEL_DIR}/                     # full-page foreground density + calibration",
        f"  {BLOCK_LEVEL_DIR}/",
        f"    {BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}/  # single-flow + columnar-flow layout",
        f"    {BLOCK_LEVEL_HYBRID_LAYOUT_DIR}/     # hybrid-layout pages",
        f"  {LINE_LEVEL_DIR}/                      # line geometry / interference",
        f"  {HMER_DIR}/                            # expression-level OCR metrics",
        f"  {PAGE_LEVEL_HMER_DIR}/                 # page-level LaTeX / expression aggregates",
        f"  {PAGE_LEVEL_LATEX_SPLIT_DIR}/          # stratified split manifest + Ch.7 tables",
        "  summary.json",
        "  dataset_overview.md",
        "```",
        "",
        "## Data flow",
        "",
        "1. **HMER** — tokenizes expressions; `expression_id` embeds page id "
        "(`dataset:page.jpg:…`).",
        "2. **page_level** — writes `tables/image_features.csv` keyed by `image_id` "
        "(= page stem). Calibration JSON is consumed by line-level ink metrics and "
        "block-level density (Txtblock annotation mask; global pooled Otsu gray threshold t_I).",
        f"3. **{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_STRUCTURE_LAYOUT_DIR}** — single-flow and "
        "columnar-flow pages; `tables/flow_structure_page_metrics.csv` keyed by `page_id`.",
        f"4. **{BLOCK_LEVEL_DIR}/{BLOCK_LEVEL_HYBRID_LAYOUT_DIR}** — hybrid-layout pages "
        "with the same table schema.",
        "5. **line_level** — per-line metrics in `line_metrics.csv` (`image_id` = page id).",
        "6. **page_level_HMER** — page/expression LaTeX metrics used to build split inputs.",
        "7. **page_level_latex_split** — joins sibling exports for cross-domain Ch.7 tables "
        "(fig 7-3 density, 7-4 layout, 7-5 lines, 7-6 expression difficulty).",
        "",
        "## Key join files",
        "",
        f"- Page density features: `{(layout.page_level / 'tables/image_features.csv').relative_to(export_root).as_posix()}`",
        f"- Block density: `{(layout.structure_layout / 'tables/block_foreground_density.csv').relative_to(export_root).as_posix()}` (+ hybrid sibling; Txtblock only)",
        f"- Block density figure: `{(layout.block_level / 'block_foreground_density_distribution.png').relative_to(export_root).as_posix()}`",
        f"- Block density summary: `{(layout.block_level / 'block_level_summary.md').relative_to(export_root).as_posix()}`",
        f"- Flow structure: `{(layout.structure_layout / 'tables/flow_structure_page_metrics.csv').relative_to(export_root).as_posix()}` + hybrid sibling",
        f"- Line metrics: `{(layout.line_level / 'line_metrics.csv').relative_to(export_root).as_posix()}`",
        f"- Expression stats: `{(layout.hmer / 'details/expression_level_statistics.csv').relative_to(export_root).as_posix()}`",
        f"- Split manifest: `{(layout.page_level_latex_split / 'split_manifest.csv').relative_to(export_root).as_posix()}`",
        "",
        "## CLI",
        "",
        "```bash",
        "python -m benchmark_design project export --config config/project.yaml",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
