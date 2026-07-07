"""Consistency tests for benchmark pipeline performance optimizations."""

from __future__ import annotations

import shutil
from pathlib import Path

from benchmark_design.ocr.cross_benchmark import compute_cross_benchmark_results
from benchmark_design.ocr.processing import build_enriched_corpus_cached, clear_enriched_corpus_cache
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.structure_distribution import (
    compute_ocr_structure_distribution_from_features,
    compute_ocr_structure_distribution_from_token_sequences,
)
from benchmark_design.report.dataset_overview import (
    compute_dataset_overview,
    compute_hmer_overview,
)
from benchmark_design.vision.dataset import (
    clear_vision_benchmark_dataset_cache,
    load_vision_benchmark_dataset,
    load_vision_benchmark_dataset_cached,
)
from benchmark_design.vision.processing_options import VisionProcessingOptions

SAMPLE_PAGE = Path(__file__).parent / "fixtures" / "sample_page.json"


def test_enriched_corpus_cache_reuses_build(tmp_path: Path) -> None:
    clear_enriched_corpus_cache()
    clear_vision_benchmark_dataset_cache()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")
    processing = ProcessingOptions(show_progress=False)

    first = build_enriched_corpus_cached("ours", input_dir, processing)
    second = build_enriched_corpus_cached("ours", input_dir, processing)
    assert first is second


def test_vision_dataset_cache_reuses_build(tmp_path: Path) -> None:
    clear_vision_benchmark_dataset_cache()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")
    processing = VisionProcessingOptions(show_progress=False, read_image_dimensions=False)

    first = load_vision_benchmark_dataset_cached(input_dir, processing=processing)
    second = load_vision_benchmark_dataset_cached(input_dir, processing=processing)
    assert first is second


def test_hmer_overview_cached_matches_cold_path(tmp_path: Path) -> None:
    clear_enriched_corpus_cache()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")
    processing = ProcessingOptions(show_progress=False)

    enriched = build_enriched_corpus_cached("ours", input_dir, processing)
    from benchmark_design.report.dataset_overview import compute_hmer_overview_from_enriched

    cached = compute_hmer_overview_from_enriched(enriched)
    cold = compute_hmer_overview(input_dir, processing=processing)

    assert cached.page_count == cold.page_count
    assert cached.expression_count == cold.expression_count
    assert cached.total_characters == cold.total_characters
    assert cached.expressions_per_page == cold.expressions_per_page


def test_dataset_overview_injected_matches_cold_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")
    processing = ProcessingOptions(show_progress=False)
    vision_processing = VisionProcessingOptions(show_progress=False, read_image_dimensions=False)

    clear_enriched_corpus_cache()
    enriched = build_enriched_corpus_cached("ours", input_dir, processing)
    dataset = load_vision_benchmark_dataset(input_dir, processing=vision_processing)

    injected = compute_dataset_overview(
        input_dir,
        processing=processing,
        vision_processing=vision_processing,
        enriched=enriched,
        vision_samples=dataset.samples,
        vision_pages=dataset.pages,
    )
    cold = compute_dataset_overview(
        input_dir,
        processing=processing,
        vision_processing=vision_processing,
    )

    assert injected.hmer.expression_count == cold.hmer.expression_count
    assert injected.vision.page_count == cold.vision.page_count
    assert injected.vision.portrait_count == cold.vision.portrait_count
    assert injected.block.txtblock_count == cold.block.txtblock_count
    assert injected.block.total_block_count == cold.block.total_block_count


def test_structure_distribution_from_features_matches_tokens(tmp_path: Path) -> None:
    clear_enriched_corpus_cache()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")
    enriched = build_enriched_corpus_cached(
        "ours",
        input_dir,
        ProcessingOptions(show_progress=False),
    )
    from_features = compute_ocr_structure_distribution_from_features(list(enriched.features))
    from_tokens = compute_ocr_structure_distribution_from_token_sequences(enriched.token_sequences)
    assert from_features.expression_count == from_tokens.expression_count
    assert from_features.structural_token_count == from_tokens.structural_token_count
    assert from_features.as_rows() == from_tokens.as_rows()


def test_cross_benchmark_corpus_cache_populated(tmp_path: Path) -> None:
    clear_enriched_corpus_cache()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    shutil.copy(SAMPLE_PAGE, input_dir / "sample.jpg.json")

    from benchmark_design.config import CROSS_BENCHMARK_SETS
    from benchmark_design.ocr import processing as processing_module

    original_path = CROSS_BENCHMARK_SETS["ours"]
    CROSS_BENCHMARK_SETS["ours"] = input_dir
    try:
        processing = ProcessingOptions(show_progress=False)
        cache: dict = {}
        _profiles, rows = compute_cross_benchmark_results(
            processing=processing,
            dataset_names=["ours"],
            corpus_cache=cache,
        )
        assert len(rows) == 1
        assert rows[0].expression_count == 3
        assert "ours" in cache
        assert cache["ours"] is processing_module._CORPUS_CACHE[("ours", str(input_dir.resolve()))]
    finally:
        CROSS_BENCHMARK_SETS["ours"] = original_path
