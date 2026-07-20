"""Unit tests for DeepSeek-OCR-2 feature pipeline (CPU, no model required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.feature_store import EmbeddingStore
from src.parse_layout import blocks_to_features, parse_grounding_blocks
from src.parse_recognition import extract_recognition_features
from src.utils import l2_normalize, make_image_id, sha256_file


def test_l2_normalize_error():
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = l2_normalize(v)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-6


def test_layout_coord_range():
    text = "<|ref|>text<|/ref|><|det|>[[0, 0, 999, 500], [100, 100, 200, 200]]<|/det|>"
    blocks = parse_grounding_blocks(text)
    assert len(blocks) == 2
    for b in blocks:
        for k in ("x1", "y1", "x2", "y2", "cx", "cy"):
            assert 0.0 <= b[k] <= 1.0
    feat = blocks_to_features("abc", blocks)
    assert feat["layout_available"] is True
    assert feat["block_count"] == 2
    for i in range(4):
        for j in range(4):
            assert 0.0 <= feat[f"occupancy_{i}_{j}"] <= 1.0


def test_layout_no_forge_when_missing():
    feat = blocks_to_features("abc", [])
    assert feat["layout_available"] is False
    assert feat["layout_missing_reason"]


def test_recognition_features():
    text = "# Title\nHello 你好 123 $$x^2$$\n"
    feat = extract_recognition_features("id1", text, output_token_count=10, mean_logprob=None)
    assert feat["logprob_available"] is False
    assert feat["mean_generated_token_logprob"] is None
    assert feat["markdown_heading_count"] >= 1
    assert feat["chinese_ratio"] > 0
    assert feat["digit_ratio"] > 0


def test_embedding_store_roundtrip(tmp_path: Path):
    store = EmbeddingStore(tmp_path / "e.f32.mmap", tmp_path / "e.parquet", dim=4)
    rng = np.random.default_rng(0)
    vecs = np.stack([l2_normalize(rng.normal(size=4).astype(np.float32)) for _ in range(3)])
    rows = [
        {
            "image_id": f"i{i}",
            "selected_layer": 12,
            "token_count": 10,
            "embedding_norm_before_normalization": 1.0,
        }
        for i in range(3)
    ]
    store.append_many(rows, vecs)
    mat, df = store.load_matrix()
    assert mat.shape == (3, 4)
    assert list(df["image_id"]) == ["i0", "i1", "i2"]
    assert store.done_ids == {"i0", "i1", "i2"}


def test_image_id_alignment(tmp_path: Path):
    p = tmp_path / "a.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(p)
    digest = sha256_file(p)
    iid = make_image_id(str(p), digest)
    assert len(iid) == 16
    p2 = tmp_path / "b.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(p2)
    assert make_image_id(str(p2), sha256_file(p2)) == iid


def test_channels_not_fused():
    from src.analyze_features import LAYOUT_FEATURE_COLS, RECOG_FEATURE_COLS

    assert "embedding" not in LAYOUT_FEATURE_COLS
    assert "visual_layout_concat" not in LAYOUT_FEATURE_COLS
    assert "fused" not in " ".join(LAYOUT_FEATURE_COLS + RECOG_FEATURE_COLS)


def test_corrupt_image_manifest(tmp_path: Path):
    from src import build_manifest as bm

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not-an-image")
    good = tmp_path / "good.png"
    Image.new("RGB", (16, 16), color=(0, 128, 0)).save(good)
    cfg = {
        "data": {"input_dir": str(tmp_path), "image_extensions": [".jpg", ".png"], "recursive": True},
        "paths": {"outputs_dir": str(tmp_path / "out")},
        "model": {"name_or_path": "x", "revision": "y"},
        "prompt": "p",
        "visual": {"base_size": 1024, "image_size": 768, "crop_mode": True},
        "pipeline": {"limit": None},
    }
    Path(cfg["paths"]["outputs_dir"]).mkdir(parents=True, exist_ok=True)
    df = bm.build_manifest(cfg, workers=2)
    assert (df["status"] == "corrupt").any()
    assert (df["status"] == "pending").any()


def test_no_clustering_in_pipeline_stages():
    from src.config import load_config

    cfg = load_config("configs/default.yaml")
    stages = cfg["pipeline"]["stages"]
    assert "cluster" not in stages
    assert "split" not in stages


def test_resume_skips_done(tmp_path: Path):
    store = EmbeddingStore(tmp_path / "e.f32.mmap", tmp_path / "e.parquet", dim=2)
    v = l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    store.append_many(
        [{"image_id": "x", "selected_layer": 1, "token_count": 1, "embedding_norm_before_normalization": 1.0}],
        v[None, :],
    )
    assert "x" in store.done_ids
