"""Tests for multi-dataset loaders."""

from __future__ import annotations

from pathlib import Path

from benchmark_design.io.dataset_loaders import (
    load_caption_txt,
    load_hme100k,
    load_mathwriting,
    load_mne_split,
)

FIXTURES = Path(__file__).parent / "fixtures" / "datasets"


def test_load_caption_txt_tab() -> None:
    records = load_caption_txt(FIXTURES / "crohme" / "caption.txt", dataset="CROHME2014")
    assert len(records) == 2
    assert records[0].ocr == "x^{2}"
    assert records[0].expression_id == "CROHME2014:sample1"


def test_load_crohme_train_caption() -> None:
    records = load_caption_txt(FIXTURES / "crohme" / "caption.txt", dataset="CROHMEtrain")
    assert len(records) == 2
    assert records[0].expression_id == "CROHMEtrain:sample1"


def test_load_hme100k() -> None:
    records = load_hme100k(FIXTURES / "hme100k", dataset="HME100K")
    assert len(records) == 2
    assert records[0].image_name == "a"
    assert records[1].ocr == "c=d"


def test_load_mne_split() -> None:
    records = load_mne_split(FIXTURES / "mne", dataset="MNE_N1")
    assert len(records) == 2
    assert records[0].expression_id == "MNE_N1:id1"


def test_load_mathwriting() -> None:
    records = load_mathwriting(FIXTURES / "mathwriting", dataset="MathWriting")
    assert len(records) == 1
    assert records[0].expression_id == "MathWriting:train/shard-000/expr001"
    assert records[0].image_name == "train/shard-000/expr001"
