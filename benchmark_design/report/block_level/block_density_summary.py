"""Summary for block-level Txtblock foreground density across layout domains."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from benchmark_design.block_level.block_foreground_density import BlockForegroundDensityRow
from benchmark_design.block_level.flow_structure.flow_group import FLOW_GROUP_LABELS
from benchmark_design.block_level.flow_structure.models import PageFlowStructureResult
from benchmark_design.export_layout import (
    HYBRID_LAYOUT_FLOW_STRUCTURES,
    STRUCTURE_LAYOUT_FLOW_STRUCTURES,
    export_flow_structure_label,
)
from benchmark_design.page_level.models import CalibrationResult
from benchmark_design.page_level.statistics import _continuous_stats
from benchmark_design.report.output_layout import relative_input_path

LayoutDomain = str


def _density_values(rows: Sequence[BlockForegroundDensityRow]) -> np.ndarray:
    if not rows:
        return np.array([], dtype=np.float64)
    return np.array([row.foreground_density for row in rows], dtype=np.float64)


def _format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value * 100:.2f}%"


def _format_density(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{_format_percent(value)} ({value:.6f})"


def _overall_density_table(stats: dict[str, float | int], *, metric: str = "D_block") -> list[str]:
    return [
        "| Metric | N | Mean | Median |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| {metric} | {int(stats['count']):,} | "
            f"{_format_percent(float(stats['mean']))} | "
            f"{_format_percent(float(stats['median']))} |"
        ),
    ]


def _detail_density_table(stats: dict[str, float | int]) -> list[str]:
    return [
        "| Statistic | Value |",
        "| --- | --- |",
        f"| std | {_format_density(float(stats['std']))} |",
        f"| P05 | {_format_density(float(stats['p05']))} |",
        f"| P95 | {_format_density(float(stats['p95']))} |",
        f"| min | {_format_density(float(stats['min']))} |",
        f"| max | {_format_density(float(stats['max']))} |",
    ]


def _extended_density_table(stats: dict[str, float | int], *, metric: str = "D_block") -> list[str]:
    return [
        *_overall_density_table(stats, metric=metric),
        "",
        *_detail_density_table(stats),
    ]


def _format_count_ratio(count: int, total: int) -> str:
    if total <= 0:
        return f"{count:,}"
    return f"{count:,} ({100.0 * count / total:.1f}%)"


def _layout_domain_for_flow_structure(flow_structure: str) -> LayoutDomain | None:
    if flow_structure in STRUCTURE_LAYOUT_FLOW_STRUCTURES:
        return "structure_layout"
    if flow_structure in HYBRID_LAYOUT_FLOW_STRUCTURES:
        return "hybrid_layout"
    return None


def _page_counts_by_layout(
    flow_results: Sequence[PageFlowStructureResult],
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    layout_counts: Counter[str] = Counter()
    flow_structure_counts: Counter[str] = Counter()
    flow_group_counts: Counter[str] = Counter()
    for result in flow_results:
        domain = _layout_domain_for_flow_structure(result.flow_structure)
        if domain is None:
            continue
        layout_counts[domain] += 1
        flow_structure_counts[export_flow_structure_label(result.flow_structure)] += 1
        flow_group_counts[result.flow_group] += 1
    return layout_counts, flow_structure_counts, flow_group_counts


def _stats_rows_for_values(values: np.ndarray) -> dict[str, float | int]:
    return _continuous_stats(values)


def _write_density_statistics_csv(
    *,
    output_path: Path,
    overall: dict[str, float | int],
    by_layout: dict[str, dict[str, float | int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stat_names = ("count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max")
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "layout_domain", *stat_names])
        writer.writerow(["overall", "all", *(overall[name] for name in stat_names)])
        for layout_domain, stats in by_layout.items():
            writer.writerow(["layout_domain", layout_domain, *(stats[name] for name in stat_names)])


def write_block_level_density_summary(
    *,
    output_dir: Path,
    rows: Sequence[BlockForegroundDensityRow],
    structure_rows: Sequence[BlockForegroundDensityRow],
    hybrid_rows: Sequence[BlockForegroundDensityRow],
    flow_results: Sequence[PageFlowStructureResult],
    calibration: CalibrationResult,
    input_dir: Path | None = None,
) -> dict[str, Path]:
    """Write block-level root summary describing layout counts and density statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_md = output_dir / "block_level_summary.md"
    stats_csv = output_dir / "tables" / "block_density_statistics.csv"
    metadata_json = output_dir / "metadata.json"

    layout_page_counts, flow_structure_counts, flow_group_counts = _page_counts_by_layout(flow_results)
    total_pages = sum(layout_page_counts.values())
    total_blocks = len(rows)
    structure_block_count = len(structure_rows)
    hybrid_block_count = len(hybrid_rows)

    overall_stats = _stats_rows_for_values(_density_values(rows))
    structure_stats = _stats_rows_for_values(_density_values(structure_rows))
    hybrid_stats = _stats_rows_for_values(_density_values(hybrid_rows))
    by_layout_stats = {
        "structure_layout": structure_stats,
        "hybrid_layout": hybrid_stats,
    }
    _write_density_statistics_csv(
        output_path=stats_csv,
        overall=overall_stats,
        by_layout=by_layout_stats,
    )

    lines = [
        "# Block-Level Density Summary",
        "",
    ]
    if input_dir is not None:
        lines.append(f"- Input: `{relative_input_path(input_dir)}`")
    lines.extend(
        [
            f"- Foreground rule: `I <= t_I` with global pooled Otsu gray threshold "
            f"**t_I = {calibration.gray_threshold:.0f}**",
            f"- Density denominator: Txtblock annotation mask pixels",
            f"- Indexed pages: **{total_pages:,}**",
            f"- Txtblocks with density: **{total_blocks:,}**",
            "",
            "## Overall statistical results",
            "",
            "Txtblock foreground density overall statistics (`D_block = |F ∩ A| / |A|`).",
            "",
            *_overall_density_table(overall_stats),
            "",
            "## Layout domain overview",
            "",
            "| Layout domain | Pages | Txtblocks |",
            "| --- | ---: | ---: |",
            f"| structure_layout | {_format_count_ratio(layout_page_counts.get('structure_layout', 0), total_pages)} | "
            f"{_format_count_ratio(structure_block_count, total_blocks)} |",
            f"| hybrid_layout | {_format_count_ratio(layout_page_counts.get('hybrid_layout', 0), total_pages)} | "
            f"{_format_count_ratio(hybrid_block_count, total_blocks)} |",
            f"| **Total** | **{total_pages:,}** | **{total_blocks:,}** |",
            "",
            "## Flow structure (page counts)",
            "",
            "| Flow structure | Pages |",
            "| --- | ---: |",
        ]
    )
    for label, count in flow_structure_counts.most_common():
        lines.append(f"| {label} | {count:,} |")
    lines.extend(
        [
            "",
            "## Flow group (page counts)",
            "",
            "| Flow group | Pages |",
            "| --- | ---: |",
        ]
    )
    for label in FLOW_GROUP_LABELS:
        count = flow_group_counts.get(label, 0)
        if count:
            lines.append(f"| {label} | {count:,} |")
    if flow_group_counts.get("no_valid_answer_block", 0):
        lines.append(f"| no_valid_answer_block | {flow_group_counts['no_valid_answer_block']:,} |")

    lines.extend(
        [
            "",
            "## Txtblock foreground density by layout domain",
            "",
            "Density = foreground pixels / annotation mask pixels, using the shared gray threshold above.",
            "",
            "### structure_layout",
            "",
            *_overall_density_table(structure_stats),
            "",
            "### hybrid_layout",
            "",
            *_overall_density_table(hybrid_stats),
            "",
            "## Extended density statistics",
            "",
            "### Overall",
            "",
            *_detail_density_table(overall_stats),
        ]
    )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `block_foreground_density_distribution.png` — dataset-level Txtblock density bins",
            "- `tables/block_density_statistics.csv` — numeric density summary (overall + by layout domain)",
            "- `structure_layout/tables/block_foreground_density.csv` — per-Txtblock density (structure pages)",
            "- `hybrid_layout/tables/block_foreground_density.csv` — per-Txtblock density (hybrid pages)",
            "- `structure_layout/block_level_summary.md` — structure-layout flow export summary",
            "- `hybrid_layout/block_level_summary.md` — hybrid-layout flow export summary",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "domain": "block_level",
        "scope": "txtblock_foreground_density",
        "gray_threshold": calibration.gray_threshold,
        "threshold_method": calibration.threshold_method,
        "foreground_rule": "gray <= gray_threshold",
        "page_count": total_pages,
        "txtblock_count": total_blocks,
        "layout_page_counts": dict(layout_page_counts),
        "layout_txtblock_counts": {
            "structure_layout": structure_block_count,
            "hybrid_layout": hybrid_block_count,
        },
        "flow_structure_counts": dict(flow_structure_counts),
        "flow_group_counts": dict(flow_group_counts),
        "density_statistics": {
            "overall": overall_stats,
            "by_layout_domain": by_layout_stats,
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "block_level_summary": summary_md,
        "block_density_statistics": stats_csv,
        "block_level_metadata": metadata_json,
    }


def _load_density_rows_from_csv(csv_path: Path) -> list[BlockForegroundDensityRow]:
    if not csv_path.is_file():
        return []
    rows: list[BlockForegroundDensityRow] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            rows.append(
                BlockForegroundDensityRow(
                    page_id=row["page_id"],
                    block_id=row["block_id"],
                    block_type=row["block_type"],
                    block_order=index,
                    foreground_density=float(row["density"]),
                    annotation_pixel_count=int(row["mask_area"]),
                    foreground_pixel_count=int(row["foreground_pixels"]),
                )
            )
    return rows


def _merge_metadata_counters(export_root: Path) -> tuple[Counter[str], Counter[str], Counter[str], int]:
    from benchmark_design.export_layout import BenchmarkExportLayout

    layout = BenchmarkExportLayout(export_root)
    layout_page_counts: Counter[str] = Counter()
    flow_structure_counts: Counter[str] = Counter()
    flow_group_counts: Counter[str] = Counter()
    for layout_dir, domain in (
        (layout.structure_layout, "structure_layout"),
        (layout.hybrid_layout, "hybrid_layout"),
    ):
        metadata_path = layout_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        layout_page_counts[domain] += int(payload.get("page_count", 0))
        flow_structure_counts.update(payload.get("flow_structure_counts", {}))
        flow_group_counts.update(payload.get("flow_group_counts", {}))
    return layout_page_counts, flow_structure_counts, flow_group_counts, sum(layout_page_counts.values())


def regenerate_block_level_density_summary_from_export(export_root: Path) -> dict[str, Path]:
    """Rebuild block-level root summary from an existing project export."""
    from benchmark_design.export_layout import BenchmarkExportLayout
    from benchmark_design.line_level.bbox_ink import load_calibration_result

    layout = BenchmarkExportLayout(export_root)
    calibration = load_calibration_result(layout.density_calibration)
    structure_rows = _load_density_rows_from_csv(
        layout.structure_layout / "tables" / "block_foreground_density.csv"
    )
    hybrid_rows = _load_density_rows_from_csv(
        layout.hybrid_layout / "tables" / "block_foreground_density.csv"
    )
    combined_rows = structure_rows + hybrid_rows
    if not combined_rows:
        raise FileNotFoundError(
            "No block_foreground_density.csv files found under structure_layout/ or hybrid_layout/"
        )

    layout_page_counts, flow_structure_counts, flow_group_counts, _total_pages = _merge_metadata_counters(
        export_root
    )

    output_dir = layout.block_level
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_md = output_dir / "block_level_summary.md"
    stats_csv = output_dir / "tables" / "block_density_statistics.csv"
    metadata_json = output_dir / "metadata.json"

    total_pages = sum(layout_page_counts.values())
    total_blocks = len(combined_rows)
    structure_block_count = len(structure_rows)
    hybrid_block_count = len(hybrid_rows)

    overall_stats = _stats_rows_for_values(_density_values(combined_rows))
    structure_stats = _stats_rows_for_values(_density_values(structure_rows))
    hybrid_stats = _stats_rows_for_values(_density_values(hybrid_rows))
    by_layout_stats = {
        "structure_layout": structure_stats,
        "hybrid_layout": hybrid_stats,
    }
    _write_density_statistics_csv(
        output_path=stats_csv,
        overall=overall_stats,
        by_layout=by_layout_stats,
    )

    input_dir_text = ""
    structure_metadata = layout.structure_layout / "metadata.json"
    if structure_metadata.is_file():
        payload = json.loads(structure_metadata.read_text(encoding="utf-8"))
        input_dir_text = str(payload.get("input_dir", ""))

    lines = [
        "# Block-Level Density Summary",
        "",
    ]
    if input_dir_text:
        lines.append(f"- Input: `{input_dir_text}`")
    lines.extend(
        [
            f"- Foreground rule: `I <= t_I` with global pooled Otsu gray threshold "
            f"**t_I = {calibration.gray_threshold:.0f}**",
            f"- Density denominator: Txtblock annotation mask pixels",
            f"- Indexed pages: **{total_pages:,}**",
            f"- Txtblocks with density: **{total_blocks:,}**",
            "",
            "## Overall statistical results",
            "",
            "Txtblock foreground density overall statistics (`D_block = |F ∩ A| / |A|`).",
            "",
            *_overall_density_table(overall_stats),
            "",
            "## Layout domain overview",
            "",
            "| Layout domain | Pages | Txtblocks |",
            "| --- | ---: | ---: |",
            f"| structure_layout | {_format_count_ratio(layout_page_counts.get('structure_layout', 0), total_pages)} | "
            f"{_format_count_ratio(structure_block_count, total_blocks)} |",
            f"| hybrid_layout | {_format_count_ratio(layout_page_counts.get('hybrid_layout', 0), total_pages)} | "
            f"{_format_count_ratio(hybrid_block_count, total_blocks)} |",
            f"| **Total** | **{total_pages:,}** | **{total_blocks:,}** |",
            "",
            "## Flow structure (page counts)",
            "",
            "| Flow structure | Pages |",
            "| --- | ---: |",
        ]
    )
    for label, count in flow_structure_counts.most_common():
        lines.append(f"| {label} | {count:,} |")
    lines.extend(
        [
            "",
            "## Flow group (page counts)",
            "",
            "| Flow group | Pages |",
            "| --- | ---: |",
        ]
    )
    for label in FLOW_GROUP_LABELS:
        count = flow_group_counts.get(label, 0)
        if count:
            lines.append(f"| {label} | {count:,} |")
    if flow_group_counts.get("no_valid_answer_block", 0):
        lines.append(f"| no_valid_answer_block | {flow_group_counts['no_valid_answer_block']:,} |")
    lines.extend(
        [
            "",
            "## Txtblock foreground density by layout domain",
            "",
            "### structure_layout",
            "",
            *_overall_density_table(structure_stats),
            "",
            "### hybrid_layout",
            "",
            *_overall_density_table(hybrid_stats),
            "",
            "## Extended density statistics",
            "",
            "### Overall",
            "",
            *_detail_density_table(overall_stats),
            "",
            "## Outputs",
            "",
            "- `block_foreground_density_distribution.png` — dataset-level Txtblock density bins",
            "- `tables/block_density_statistics.csv` — numeric density summary (overall + by layout domain)",
            "- `structure_layout/tables/block_foreground_density.csv` — per-Txtblock density (structure pages)",
            "- `hybrid_layout/tables/block_foreground_density.csv` — per-Txtblock density (hybrid pages)",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "domain": "block_level",
        "scope": "txtblock_foreground_density",
        "gray_threshold": calibration.gray_threshold,
        "threshold_method": calibration.threshold_method,
        "foreground_rule": "gray <= gray_threshold",
        "page_count": total_pages,
        "txtblock_count": total_blocks,
        "layout_page_counts": dict(layout_page_counts),
        "layout_txtblock_counts": {
            "structure_layout": structure_block_count,
            "hybrid_layout": hybrid_block_count,
        },
        "flow_structure_counts": dict(flow_structure_counts),
        "flow_group_counts": dict(flow_group_counts),
        "density_statistics": {
            "overall": overall_stats,
            "by_layout_domain": by_layout_stats,
        },
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "block_level_summary": summary_md,
        "block_density_statistics": stats_csv,
        "block_level_metadata": metadata_json,
    }
