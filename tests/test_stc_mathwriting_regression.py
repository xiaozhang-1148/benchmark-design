"""Regression tests for MathWriting duplicate sample stems across splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from benchmark_design.config import CROSS_BENCHMARK_SETS
from benchmark_design.io.dataset_loaders import load_mathwriting
from benchmark_design.ocr.duplicates import normalize_expression_latex
from benchmark_design.ocr.processing import build_enriched_corpus
from benchmark_design.ocr.processing_options import ProcessingOptions
from benchmark_design.ocr.structure_stc import is_stc_export_cohort
from benchmark_design.report.confusable_token_figures import build_expression_record_index
from benchmark_design.report.stc_figures import _load_expression_image, build_ntc_cbc_cohort_metrics

ROOT = CROSS_BENCHMARK_SETS["MathWriting"]


def test_mathwriting_expression_ids_are_unique() -> None:
    records = load_mathwriting(ROOT)
    counts = Counter(record.expression_id for record in records)
    duplicates = [expression_id for expression_id, count in counts.items() if count > 1]
    assert duplicates == []


def test_mathwriting_train_val_same_stem_have_distinct_ids() -> None:
    records = load_mathwriting(ROOT)
    by_stem: dict[str, list[str]] = {}
    for record in records:
        stem = Path(record.source_file).stem
        by_stem.setdefault(stem, []).append(record.expression_id)
    overlap = {stem: ids for stem, ids in by_stem.items() if len(ids) > 1}
    assert overlap
    for ids in overlap.values():
        assert len(set(ids)) == len(ids)


def test_mathwriting_record_image_matches_feature_latex() -> None:
    enriched = build_enriched_corpus("MathWriting", ROOT, ProcessingOptions(workers=4))
    record_index = build_expression_record_index(enriched.expressions)
    cohort = [feature for feature in enriched.features if is_stc_export_cohort(feature)]
    assert cohort
    for feature in cohort[:50]:
        record = record_index[feature.expression_id]
        assert normalize_expression_latex(record.ocr) == feature.normalized_latex
        image = _load_expression_image(record, input_dir=ROOT)
        assert image is not None


def test_mathwriting_top_cohort_entry_000003277_train_matches_image() -> None:
    enriched = build_enriched_corpus("MathWriting", ROOT, ProcessingOptions(workers=4))
    record_index = build_expression_record_index(enriched.expressions)
    train_id = "MathWriting:train/shard-000001/000003277"
    val_id = "MathWriting:val/shard-000001/000003277"
    assert train_id in record_index
    assert val_id in record_index
    train_record = record_index[train_id]
    val_record = record_index[val_id]
    assert train_record.ocr != val_record.ocr
    assert r"\frac" in train_record.ocr
    assert r"\notin" in val_record.ocr or "notin" in val_record.ocr

    feature_by_id = {feature.expression_id: feature for feature in enriched.features}
    prepared = build_ntc_cbc_cohort_metrics(enriched.features)
    top_ids = {feature.expression_id for feature, _ in prepared[:100]}
    if train_id in top_ids:
        feature = feature_by_id[train_id]
        assert feature.normalized_latex == normalize_expression_latex(train_record.ocr)
