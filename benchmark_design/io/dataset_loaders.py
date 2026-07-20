"""Load expressions from multiple benchmark dataset formats."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from benchmark_design.io.benchmark_loader import ExpressionRecord, load_expressions
from benchmark_design.progress import parallel_map_flatten


def load_caption_txt(
    caption_path: Path,
    *,
    dataset: str,
    delimiter: str = "\t",
) -> list[ExpressionRecord]:
    records: list[ExpressionRecord] = []
    source_file = str(caption_path.resolve())
    with caption_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if delimiter == "\t":
                parts = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            sample_id, latex = parts[0].strip(), parts[1].strip()
            if not latex:
                continue
            records.append(
                ExpressionRecord(
                    image_name=sample_id,
                    block_order=0,
                    line_order=line_no,
                    block_type="",
                    ocr=latex,
                    dataset=dataset,
                    source_file=source_file,
                    expression_id=f"{dataset}:{sample_id}",
                    line_id=str(line_no),
                )
            )
    return records


def load_hme100k(root: Path, *, dataset: str = "HME100K") -> list[ExpressionRecord]:
    train_txt = root / "train.txt"
    if not train_txt.is_file():
        msg = f"HME100K train.txt not found: {train_txt}"
        raise FileNotFoundError(msg)
    records: list[ExpressionRecord] = []
    source_file = str(train_txt.resolve())
    with train_txt.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            image_path, latex = parts[0].strip(), parts[1].strip()
            sample_id = Path(image_path).stem
            records.append(
                ExpressionRecord(
                    image_name=sample_id,
                    block_order=0,
                    line_order=line_no,
                    block_type="",
                    ocr=latex,
                    dataset=dataset,
                    source_file=source_file,
                    expression_id=f"{dataset}:{sample_id}",
                    line_id=str(line_no),
                )
            )
    return records


def load_mne_split(root: Path, *, dataset: str) -> list[ExpressionRecord]:
    caption_path = root / "caption.txt"
    return load_caption_txt(caption_path, dataset=dataset, delimiter=" ")


def _load_mathwriting_txt(txt_path: Path, *, dataset: str, root: Path) -> list[ExpressionRecord]:
    latex = txt_path.read_text(encoding="utf-8").strip()
    if not latex:
        return []
    relative_key = txt_path.relative_to(root).with_suffix("").as_posix()
    source_file = str(txt_path.resolve())
    return [
        ExpressionRecord(
            image_name=relative_key,
            block_order=0,
            line_order=0,
            block_type="",
            ocr=latex,
            dataset=dataset,
            source_file=source_file,
            expression_id=f"{dataset}:{relative_key}",
            line_id="0",
        )
    ]


def load_mathwriting(root: Path, *, dataset: str = "MathWriting") -> list[ExpressionRecord]:
    txt_paths = sorted(root.glob("train/shard-*/*.txt")) + sorted(root.glob("val/shard-*/*.txt"))
    if not txt_paths:
        msg = f"No MathWriting shard txt files under: {root}"
        raise FileNotFoundError(msg)

    def loader(path: Path) -> list[ExpressionRecord]:
        return _load_mathwriting_txt(path, dataset=dataset, root=root)

    return parallel_map_flatten(
        loader,
        txt_paths,
        description=f"Loading {dataset}",
        show_progress=False,
        workers=None,
    )


def load_ours(input_dir: Path, *, show_progress: bool = False, workers: int | None = None) -> list[ExpressionRecord]:
    return load_expressions(input_dir, dataset="ours", show_progress=show_progress, workers=workers)


DATASET_LOADERS: dict[str, Callable[..., list[ExpressionRecord]]] = {
    "ours": load_ours,
    "CROHMEtrain": lambda p: load_caption_txt(p / "caption.txt", dataset="CROHMEtrain"),
    "CROHME2014": lambda p: load_caption_txt(p / "caption.txt", dataset="CROHME2014"),
    "CROHME2016": lambda p: load_caption_txt(p / "caption.txt", dataset="CROHME2016"),
    "CROHME2019": lambda p: load_caption_txt(p / "caption.txt", dataset="CROHME2019"),
    "HME100K": load_hme100k,
    "MathWriting": load_mathwriting,
    "MNE_N1": lambda p: load_mne_split(p, dataset="MNE_N1"),
    "MNE_N2": lambda p: load_mne_split(p, dataset="MNE_N2"),
    "MNE_N3": lambda p: load_mne_split(p, dataset="MNE_N3"),
}


def load_dataset(
    dataset_name: str,
    input_dir: Path,
    *,
    show_progress: bool = False,
    workers: int | None = None,
) -> list[ExpressionRecord]:
    if dataset_name not in DATASET_LOADERS:
        msg = f"Unknown dataset: {dataset_name}. Known: {sorted(DATASET_LOADERS)}"
        raise ValueError(msg)
    loader = DATASET_LOADERS[dataset_name]
    if dataset_name == "ours":
        return loader(input_dir, show_progress=show_progress, workers=workers)
    return loader(input_dir)
