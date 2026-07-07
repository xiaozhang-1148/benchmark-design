"""Tests for unified single-pass vision page metrics."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from benchmark_design.vision.deleted_block_scale.compute import compute_page_deleted_block_scale
from benchmark_design.vision.flow_structure.classifier import classify_page_flow_structure
from benchmark_design.vision.flow_structure.models import PageAnnotation, PageBlockAnnotation
from benchmark_design.vision.foreground_load.pipeline import compute_foreground_load_from_pages
from benchmark_design.vision.masks import build_unified_page_masks
from benchmark_design.vision.page_metrics import compute_page_vision_metrics
from benchmark_design.vision.processing_options import VisionProcessingOptions


def _page(page_id: str, blocks: list[tuple[str, list[list[float]]]]) -> PageAnnotation:
    page_blocks = tuple(
        PageBlockAnnotation(
            page_id=page_id,
            block_id=f"{page_id}:block_{index}",
            block_type=block_type,
            block_order=index,
            polygon=tuple((float(x), float(y)) for x, y in polygon),
        )
        for index, (block_type, polygon) in enumerate(blocks)
    )
    return PageAnnotation(
        page_id=page_id,
        image_name=f"{page_id}.jpg",
        source_file=f"/tmp/{page_id}.json",
        image_width=200,
        image_height=200,
        blocks=page_blocks,
    )


def test_unified_masks_match_separate_builders() -> None:
    page = _page(
        "p_unified",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("deleted_text_block", [[20, 20], [40, 20], [40, 40], [20, 40]]),
            ("figure", [[0, 0], [50, 0], [50, 50], [0, 50]]),
        ],
    )
    unified = build_unified_page_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    )
    fg_masks = unified.foreground_masks()
    dbs_masks = unified.deleted_block_masks()
    assert int(fg_masks.effective_union.sum()) > 0
    assert int(dbs_masks.valid_union.sum()) > 0
    assert int(dbs_masks.deleted_union.sum()) > 0
    assert int(dbs_masks.answer_related_union.sum()) >= int(dbs_masks.deleted_union.sum())


def test_page_vision_metrics_matches_separate_compute(tmp_path: Path) -> None:
    page = _page(
        "p_pass",
        [
            ("Txtblock", [[20, 20], [180, 20], [180, 180], [20, 180]]),
            ("deleted_text_block", [[20, 20], [40, 20], [40, 40], [20, 40]]),
        ],
    )
    Image.new("RGB", (200, 200), color=(255, 255, 255)).save(tmp_path / page.image_name)
    processing = VisionProcessingOptions(show_progress=False, workers=1)
    fg_results, _, global_config = compute_foreground_load_from_pages(
        [page],
        input_dir=tmp_path,
        processing=processing,
    )
    combined = compute_page_vision_metrics(page, input_dir=tmp_path, global_config=global_config)
    flow = classify_page_flow_structure(page, input_dir=tmp_path)
    fg = fg_results[0]
    unified = build_unified_page_masks(
        page.blocks,
        image_width=page.image_width,
        image_height=page.image_height,
    )
    dbs = compute_page_deleted_block_scale(page, input_dir=tmp_path, unified_masks=unified)

    assert combined.flow_structure.flow_structure == flow.flow_structure
    assert combined.foreground_load.D_page_eff == fg.D_page_eff
    assert combined.deleted_block_scale.r_del == dbs.r_del
    assert combined.deleted_block_scale.has_deleted_text_block == dbs.has_deleted_text_block
