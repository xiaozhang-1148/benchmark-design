"""Tests for page-level HMER multilabel stratified split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from benchmark_design.page_level_latex_split.audit import manifest_hash, SplitAcceptanceError
from benchmark_design.page_level_latex_split.config import load_split_config
from benchmark_design.page_level_latex_split.io_validate import load_and_validate_inputs
from benchmark_design.page_level_latex_split.labels import assign_bin, build_page_labels
from benchmark_design.page_level_latex_split.pipeline import run_page_level_latex_split
from benchmark_design.page_level_latex_split.stratify import largest_remainder_counts
from benchmark_design.page_level_latex_split.vocab_cover import VocabIndex, build_train_vocab_cover


def _write_tiny_inputs(tmp_path: Path, n: int = 40) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    ann_dir = tmp_path / "anns"
    ann_dir.mkdir()

    manifest_rows = []
    feature_rows = []
    token_rows = []
    for i in range(n):
        page_id = f"page_{i:03d}"
        img = image_dir / f"{page_id}.png"
        Image.new("L", (8, 8), color=200).save(img)
        ann = ann_dir / f"{page_id}.jpg.json"
        ann.write_text(json.dumps({"image_name": f"{page_id}.png", "blocks": []}), encoding="utf-8")
        manifest_rows.append(
            {
                "page_id": page_id,
                "image_path": str(img.resolve()),
                "annotation_path": str(ann.resolve()),
            }
        )
        expr = 1 + (i % 45)
        tokens = 50 + 30 * (i % 20)
        maxlen = 5 + (i % 90)
        depth = i % 6
        has = [int((i + k) % 3 == 0) for k in range(6)]
        feature_rows.append(
            {
                "page_id": page_id,
                "expression_count": expr,
                "page_token_count": tokens,
                "max_expression_token_count": maxlen,
                "max_ast_depth": depth,
                "has_frac": has[0],
                "has_sup": has[1],
                "has_sub": has[2],
                "has_sqrt": has[3],
                "has_sum": has[4],
                "has_env": has[5],
                "structure_type_count": sum(has),
                "has_rare8": int(i % 7 == 0),
                "rare8_token_count": int(i % 7 == 0) * (1 + i % 3),
                "has_digit_letter_pair": int(i % 5 == 0),
                "has_circle_like_pair": int(i % 8 == 0),
                "has_latin_greek_pair": int(i % 9 == 0),
                "has_greek_variant_pair": int(i % 11 == 0),
                "has_operator_variable_pair": int(i % 6 == 0),
                "has_relation_stroke_pair": int(i % 10 == 0),
                **{f"has_expr_depth_{d}": int(depth == d) for d in range(6)},
            }
        )
        # Shared tokens + one page-specific token that also appears on train-heavy pages
        token_rows.append({"page_id": page_id, "token": "x", "count": 2})
        token_rows.append({"page_id": page_id, "token": "1", "count": 1})
        token_rows.append({"page_id": page_id, "token": f"t{i % 5}", "count": 1})

    pd.DataFrame(manifest_rows).to_csv(inputs / "dataset_manifest.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(inputs / "page_hmer_features.csv", index=False)
    pd.DataFrame(token_rows).to_csv(inputs / "page_token_counts.csv", index=False)
    (inputs / "dataset_meta.json").write_text(
        json.dumps({"dataset_version": "test_fixture", "page_count": n}) + "\n",
        encoding="utf-8",
    )
    return inputs


def _tiny_config(tmp_path: Path) -> Path:
    text = """
dataset_version: test_fixture
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
random_seed: 42
candidate_seeds: [42, 43]
expression_count_bins:
  - {label: expr_1_10, min: 1, max: 11}
  - {label: expr_11_20, min: 11, max: 21}
  - {label: expr_21_40, min: 21, max: 41}
  - {label: expr_gt_40, min: 41, max: null}
max_expression_token_bins:
  - {label: maxlen_1_10, min: 1, max: 11}
  - {label: maxlen_11_20, min: 11, max: 21}
  - {label: maxlen_21_40, min: 21, max: 41}
  - {label: maxlen_41_80, min: 41, max: 81}
  - {label: maxlen_gt_80, min: 81, max: null}
page_token_bins:
  - {label: page_token_bin_1, min: 0, max: 200}
  - {label: page_token_bin_2, min: 200, max: 400}
  - {label: page_token_bin_3, min: 400, max: 700}
  - {label: page_token_bin_4, min: 700, max: 1200}
  - {label: page_token_tail, min: 1200, max: null}
ast_depth_labels:
  depth_0: [0]
  depth_1: [1]
  depth_2: [2]
  depth_3: [3]
  depth_4_5: [4, 5, 6]
min_support_pages: 5
min_expected_per_split: 1
min_split_ratio_for_support: 0.05
require_train_vocab_coverage: true
tie_break: seeded_hash
global_swap_max_iterations: 50
repair_top_k_train: 50
max_relative_deviation_tolerance: 0.5
family_tolerances:
  scale: 0.5
  structure: 0.5
  rare8: 0.5
  similar_token: 0.5
  vocabulary: 0.0
algorithm_version: test_v1
"""
    path = tmp_path / "split.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_largest_remainder_conserves_total() -> None:
    counts = largest_remainder_counts(100, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert counts == {"train": 80, "val": 10, "test": 10}
    counts2 = largest_remainder_counts(11, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert sum(counts2.values()) == 11


def test_assign_bin_edges() -> None:
    from benchmark_design.page_level_latex_split.config import BinSpec

    bins = (
        BinSpec("a", 1, 11),
        BinSpec("b", 11, 21),
        BinSpec("c", 21, None),
    )
    assert assign_bin(1, bins) == "a"
    assert assign_bin(10, bins) == "a"
    assert assign_bin(11, bins) == "b"
    assert assign_bin(100, bins) == "c"


def test_validate_rejects_structure_mismatch(tmp_path: Path) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=10)
    features = pd.read_csv(inputs / "page_hmer_features.csv")
    features.loc[0, "structure_type_count"] = 99
    features.to_csv(inputs / "page_hmer_features.csv", index=False)
    config = load_split_config(_tiny_config(tmp_path))
    with pytest.raises(ValueError, match="structure_type_count"):
        load_and_validate_inputs(inputs, config)


def test_validate_rejects_missing_page(tmp_path: Path) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=10)
    features = pd.read_csv(inputs / "page_hmer_features.csv").iloc[:-1]
    features.to_csv(inputs / "page_hmer_features.csv", index=False)
    config = load_split_config(_tiny_config(tmp_path))
    with pytest.raises(ValueError, match="mismatch"):
        load_and_validate_inputs(inputs, config)


def test_split_pipeline_deterministic(tmp_path: Path) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=40)
    config_path = _tiny_config(tmp_path)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    r1 = run_page_level_latex_split(inputs, out1, config_path=config_path, skip_figures=True, workers=1)
    r2 = run_page_level_latex_split(inputs, out2, config_path=config_path, skip_figures=True, workers=1)
    assert r1.manifest_sha256 == r2.manifest_sha256
    assert (out1 / "split_manifest.csv").is_file()
    assert (out1 / "train.txt").is_file()
    assert (out1 / "val.txt").is_file()
    assert (out1 / "test.txt").is_file()
    assert (out1 / "split_quota_audit.csv").is_file()
    assert (out1 / "low_support_features.csv").is_file()
    assert (out1 / "vocabulary_audit.csv").is_file()
    assert (out1 / "tables" / "table1_split_scale.csv").is_file()
    assert (out1 / "split_metadata.json").is_file()

    manifest = pd.read_csv(out1 / "split_manifest.csv")
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert len(manifest) == 40
    assert manifest["page_id"].nunique() == 40
    assert r1.acceptance.checks["all_pages_assigned_once"]
    assert r1.acceptance.checks["splits_disjoint"]
    assert r1.acceptance.checks["capacities_match"]


def test_labels_include_bins(tmp_path: Path) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=12)
    config = load_split_config(_tiny_config(tmp_path))
    bundle = load_and_validate_inputs(inputs, config)
    labels = build_page_labels(bundle.features, config)
    assert len(labels) == 12
    assert all(page.expr_bin.startswith("expr_") for page in labels)


def test_vocab_cover_yields_zero_unseen_before_swap(tmp_path: Path) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=40)
    config = load_split_config(_tiny_config(tmp_path))
    bundle = load_and_validate_inputs(inputs, config)
    page_labels = build_page_labels(bundle.features, config)
    vocab_index = VocabIndex.from_token_counts(bundle.token_counts)
    cover = build_train_vocab_cover(vocab_index, page_labels, config, seed=42)
    assignment = dict(cover.locked_assignment)
    # Remaining pages unassigned — unseen only defined once val/test exist; locked cover
    # must include every corpus token on at least one train page.
    train_vocab: set[str] = set()
    page_tokens = vocab_index.page_to_tokens
    for page_id in cover.locked_train_pages:
        train_vocab.update(page_tokens.get(page_id, frozenset()))
    assert train_vocab == set(vocab_index.all_tokens)


def test_always_writes_diagnostic_artifacts(tmp_path: Path) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=40)
    config_path = _tiny_config(tmp_path)
    out = tmp_path / "out_diag"
    run_page_level_latex_split(inputs, out, config_path=config_path, skip_figures=True, workers=1)
    assert (out / "diagnostic_split_manifest.csv").is_file()
    assert (out / "candidate_scores.csv").is_file()
    assert (out / "split_manifest.csv").is_file()
    assert (out / "train.txt").is_file()


def test_rejection_skips_official_lists(tmp_path: Path, monkeypatch) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=40)
    config_path = _tiny_config(tmp_path)
    out = tmp_path / "out_reject"

    from benchmark_design.page_level_latex_split import pipeline as pipeline_mod
    from benchmark_design.page_level_latex_split.audit import AcceptanceResult

    def _fail_acceptance(*_args, **_kwargs):
        return AcceptanceResult(passed=False, checks={"vocab_coverage": False}, messages=("forced",))

    monkeypatch.setattr(pipeline_mod, "run_acceptance_checks", _fail_acceptance)
    with pytest.raises(SplitAcceptanceError):
        run_page_level_latex_split(inputs, out, config_path=config_path, skip_figures=True, workers=1)
    assert (out / "diagnostic_split_manifest.csv").is_file()
    assert (out / "split_rejected.json").is_file()
    assert not (out / "train.txt").is_file()


def test_seeded_hash_reduces_prefix_batch_bias(tmp_path: Path) -> None:
    inputs = _write_tiny_inputs(tmp_path, n=40)
    config_path = _tiny_config(tmp_path)
    out = tmp_path / "out_hash"
    result = run_page_level_latex_split(
        inputs, out, config_path=config_path, skip_figures=True, workers=1
    )
    manifest = __import__("pandas").read_csv(out / "split_manifest.csv")
    holdout = manifest.loc[manifest["split"].isin(["val", "test"]), "page_id"].astype(str)
    # With seeded_hash, val/test should not all come from one decile of sorted page ids.
    sorted_ids = sorted(manifest["page_id"].astype(str))
    n = len(sorted_ids)
    last_decile = set(sorted_ids[int(n * 0.9) :])
    assert len(holdout) > 0
    assert len(set(holdout) & last_decile) < len(holdout)
    assert result.acceptance.passed
