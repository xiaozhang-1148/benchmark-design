"""Review overlay figures grouped by hierarchical flow_group labels."""

from __future__ import annotations

import random
from pathlib import Path

from benchmark_design.progress import parallel_map
from benchmark_design.report.pyplot_lock import with_locked_pyplot
from benchmark_design.report.export_figures import _configure_matplotlib_fonts
from benchmark_design.block_level.flow_structure.flow_group import FLOW_GROUP_HIERARCHY, flow_group_figure_parts
from benchmark_design.block_level.flow_structure.models import PageFlowStructureResult

SAMPLE_SEED = 42
SAMPLE_PER_GROUP = 40


def _require_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    _configure_matplotlib_fonts(plt)
    return plt, MplPolygon


def _safe_filename(page_id: str) -> str:
    return page_id.replace("/", "_").replace("\\", "_")


def _mask_points(record) -> list[tuple[float, float]]:
    if record.polygon and len(record.polygon) >= 3:
        return list(record.polygon)
    return [
        (record.bbox_x1, record.bbox_y1),
        (record.bbox_x2, record.bbox_y1),
        (record.bbox_x2, record.bbox_y2),
        (record.bbox_x1, record.bbox_y2),
    ]


@with_locked_pyplot
def _draw_overlay(
    result: PageFlowStructureResult,
    *,
    input_dir: Path,
    output_path: Path,
) -> bool:
    plt, MplPolygon = _require_matplotlib()
    image_path = input_dir / result.image_name
    if not image_path.is_file():
        image_path = Path(result.review_image_path)
    if not image_path.is_file():
        return False

    fig, axis = plt.subplots(figsize=(10, 14))
    image = plt.imread(str(image_path))
    axis.imshow(image)
    colors = ("#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6")
    for record in result.block_records:
        color = colors[(record.assigned_column_id or 0) % len(colors)]
        polygon = MplPolygon(
            _mask_points(record),
            closed=True,
            fill=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.35,
            linewidth=2,
        )
        axis.add_patch(polygon)
        label = f"#{record.sort_index} c{record.assigned_column_id}"
        axis.text(
            record.center_x,
            record.center_y,
            label,
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1},
        )
    title = (
        f"{result.flow_structure} / {result.flow_group} ({result.flow_confidence})\n"
        f"{result.flow_reason}"
    )
    if result.flow_tags:
        title += f"\ntags: {result.flow_tags}"
    axis.set_title(title, fontsize=9)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _sample_by_group(
    results: list[PageFlowStructureResult],
    *,
    flow_group: str,
    limit: int,
    rng: random.Random,
) -> list[PageFlowStructureResult]:
    candidates = [result for result in results if result.flow_group == flow_group]
    if len(candidates) <= limit:
        return candidates
    return rng.sample(candidates, limit)


def _draw_flow_task(args: tuple) -> bool:
    result, input_dir, output_path = args
    return _draw_overlay(result, input_dir=input_dir, output_path=output_path)


def export_flow_structure_figures(
    results: list[PageFlowStructureResult],
    *,
    input_dir: Path,
    figures_root: Path,
    show_progress: bool = False,
    flow_hierarchy: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] | None = None,
    include_na_figures: bool = True,
) -> dict[str, int]:
    rng = random.Random(SAMPLE_SEED)
    counts: dict[str, int] = {}
    hierarchy = flow_hierarchy if flow_hierarchy is not None else FLOW_GROUP_HIERARCHY

    for flow_structure, groups in hierarchy:
        for flow_group, group_id in groups:
            bucket_results = _sample_by_group(
                results,
                flow_group=flow_group,
                limit=SAMPLE_PER_GROUP,
                rng=rng,
            )
            structure_slug, bucket_id = flow_group_figure_parts(
                flow_structure=flow_structure,
                flow_group_id=group_id,
            )
            out_dir = figures_root / structure_slug / bucket_id
            tasks = [
                (result, input_dir, out_dir / f"{_safe_filename(result.page_id)}.png")
                for result in bucket_results
            ]
            written = 0
            if tasks:
                outcomes = parallel_map(
                    _draw_flow_task,
                    tasks,
                    description=f"Flow structure figures ({structure_slug}/{bucket_id})",
                    show_progress=show_progress,
                )
                written = sum(1 for ok in outcomes if ok)
            counts[f"{structure_slug}/{bucket_id}"] = written

    na_results = [result for result in results if result.flow_structure == "NA"]
    if include_na_figures and na_results:
        out_dir = figures_root / "na" / "no_valid_answer_block"
        tasks = [
            (result, input_dir, out_dir / f"{_safe_filename(result.page_id)}.png")
            for result in na_results[:SAMPLE_PER_GROUP]
        ]
        written = 0
        if tasks:
            outcomes = parallel_map(
                _draw_flow_task,
                tasks,
                description="Flow structure figures (na/na)",
                show_progress=show_progress,
            )
            written = sum(1 for ok in outcomes if ok)
        counts["na/no_valid_answer_block"] = written

    return counts
