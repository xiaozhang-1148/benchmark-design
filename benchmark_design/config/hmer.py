"""HMER (handwritten math expression recognition) benchmark configuration."""

from __future__ import annotations

from pathlib import Path

DEFAULT_BENCHMARK_INPUT = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark"
)

CROSS_BENCHMARK_ROOT = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/dataset"
)

CROSS_BENCHMARK_SETS: dict[str, Path] = {
    "ours": DEFAULT_BENCHMARK_INPUT,
    "CROHME2014": CROSS_BENCHMARK_ROOT / "CROHME" / "2014",
    "CROHME2016": CROSS_BENCHMARK_ROOT / "CROHME" / "2016",
    "CROHME2019": CROSS_BENCHMARK_ROOT / "CROHME" / "2019",
    "HME100K": CROSS_BENCHMARK_ROOT / "HME100K",
    "MathWriting": CROSS_BENCHMARK_ROOT / "MathWritting",
    "MNE_N1": CROSS_BENCHMARK_ROOT / "data_MNE" / "N1",
    "MNE_N2": CROSS_BENCHMARK_ROOT / "data_MNE" / "N2",
    "MNE_N3": CROSS_BENCHMARK_ROOT / "data_MNE" / "N3",
}

CROSS_BENCHMARK_REPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "CROHME": ("CROHME2014", "CROHME2016", "CROHME2019"),
    "HME100K": ("HME100K",),
    "MathWriting": ("MathWriting",),
    "MNE": ("MNE_N1", "MNE_N2", "MNE_N3"),
    "Ours": ("ours",),
}

CROSS_BENCHMARK_REPORT_ORDER: tuple[str, ...] = (
    "CROHME",
    "HME100K",
    "MathWriting",
    "MNE",
    "Ours",
)

CROSS_BENCHMARK_NOTES: dict[str, str] = {
    "CROHME": "mostly formula-centric",
    "HME100K": "formula-centric",
    "MathWriting": "clean writing-oriented",
    "MNE": "level-based formula difficulty",
    "Ours": "includes Chinese solution text and OCR correction traces",
}

CROSS_BENCHMARK_PROVENANCE: tuple[dict[str, str], ...] = (
    {
        "dataset": "ours",
        "version_or_year": "internal benchmark v1",
        "split": "full",
        "source": "internal benchmark JSON export",
        "license_or_access": "internal",
        "preprocessing_note": "Unified JSON page format; OCR LaTeX normalized by benchmark loader",
    },
    {
        "dataset": "CROHME2014",
        "version_or_year": "2014",
        "split": "test",
        "source": "CROHME",
        "license_or_access": "research use (CROHME terms)",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
    {
        "dataset": "CROHME2016",
        "version_or_year": "2016",
        "split": "test",
        "source": "CROHME",
        "license_or_access": "research use (CROHME terms)",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
    {
        "dataset": "CROHME2019",
        "version_or_year": "2019",
        "split": "test",
        "source": "CROHME",
        "license_or_access": "research use (CROHME terms)",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
    {
        "dataset": "HME100K",
        "version_or_year": "HME100K",
        "split": "test",
        "source": "HME100K",
        "license_or_access": "dataset-specific terms",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
    {
        "dataset": "MathWriting",
        "version_or_year": "MathWriting",
        "split": "test",
        "source": "MathWritting (processed folder name)",
        "license_or_access": "dataset-specific terms",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
    {
        "dataset": "MNE_N1",
        "version_or_year": "MNE N1",
        "split": "N1",
        "source": "data_MNE/N1",
        "license_or_access": "dataset-specific terms",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
    {
        "dataset": "MNE_N2",
        "version_or_year": "MNE N2",
        "split": "N2",
        "source": "data_MNE/N2",
        "license_or_access": "dataset-specific terms",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
    {
        "dataset": "MNE_N3",
        "version_or_year": "MNE N3",
        "split": "N3",
        "source": "data_MNE/N3",
        "license_or_access": "dataset-specific terms",
        "preprocessing_note": "Converted to unified JSON; same LATEX_DICT tokenizer",
    },
)
