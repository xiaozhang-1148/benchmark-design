"""Build sample index from image directory + paired LaTeX/JSON sidecars."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class SampleRecord:
    row_index: int
    sample_id: str
    image_path: str
    latex_path_or_id: str
    original_width: int | None = None
    original_height: int | None = None
    image_format: str | None = None
    file_size: int | None = None
    status: str = "pending"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_json_path(image_path: Path) -> Path:
    """``foo.jpg`` ↔ ``foo.jpg.json`` (dataset convention)."""
    return image_path.with_name(image_path.name + ".json")


def discover_samples(
    input_dir: str | Path,
    *,
    recursive: bool = False,
) -> list[SampleRecord]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir not found: {root}")

    if recursive:
        images = sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    else:
        images = sorted(
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    # Deduplicate by resolved path
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    records: list[SampleRecord] = []
    dup_path = 0
    dup_id = 0
    missing_json = 0

    for img in images:
        ap = str(img.resolve())
        if ap in seen_paths:
            dup_path += 1
            continue
        seen_paths.add(ap)
        sample_id = img.stem  # keep human-readable stem; uniqueness checked below
        # Prefer full filename stem uniqueness for this flat dataset
        sample_id = img.name  # unique among .jpg files
        if sample_id in seen_ids:
            dup_id += 1
            continue
        seen_ids.add(sample_id)

        jpath = paired_json_path(img)
        if not jpath.is_file():
            missing_json += 1
            latex = ""
            status = "missing_latex"
            err = f"missing paired json: {jpath}"
        else:
            latex = str(jpath.resolve())
            status = "pending"
            err = None

        records.append(
            SampleRecord(
                row_index=len(records),
                sample_id=sample_id,
                image_path=ap,
                latex_path_or_id=latex,
                image_format=img.suffix.lower(),
                file_size=img.stat().st_size,
                status=status,
                error_message=err,
            )
        )

    if missing_json:
        print(f"[manifest] WARNING: {missing_json} images missing paired .json")
    if dup_path or dup_id:
        print(f"[manifest] WARNING: skipped dup_path={dup_path} dup_id={dup_id}")
    return records


def enrich_image_sizes(
    records: list[SampleRecord],
    *,
    workers: int = 64,
) -> dict[str, Any]:
    """Fill width/height; count small/corrupt. Mutates records."""

    def _one(rec: SampleRecord) -> tuple[int, int | None, int | None, str | None]:
        try:
            with Image.open(rec.image_path) as im:
                im = ImageOps.exif_transpose(im)
                w, h = im.size
            return rec.row_index, w, h, None
        except Exception as e:  # noqa: BLE001
            return rec.row_index, None, None, f"{type(e).__name__}: {e}"

    by_idx = {r.row_index: r for r in records}
    small = 0
    corrupt = 0
    widths: list[int] = []
    heights: list[int] = []
    fmt = Counter(r.image_format for r in records)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, w, h, err in ex.map(_one, records):
            rec = by_idx[idx]
            if err:
                corrupt += 1
                rec.status = "corrupt"
                rec.error_message = err
                continue
            rec.original_width = w
            rec.original_height = h
            assert w is not None and h is not None
            widths.append(w)
            heights.append(h)
            if w <= 768 and h <= 768:
                small += 1

    import numpy as np

    stats: dict[str, Any] = {
        "n_total": len(records),
        "n_ok_readable": len(widths),
        "n_corrupt": corrupt,
        "n_small_both_le_768": small,
        "format_counts": dict(fmt),
        "duplicate_sample_ids": 0,
        "width": _summarize(np.array(widths, dtype=np.int64)) if widths else None,
        "height": _summarize(np.array(heights, dtype=np.int64)) if heights else None,
    }
    return stats


def _summarize(arr: Any) -> dict[str, float]:
    import numpy as np

    a = np.asarray(arr)
    return {
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def write_manifest_jsonl(records: list[SampleRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    tmp.replace(path)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records_from_jsonl(path: str | Path) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for d in iter_jsonl(path):
        out.append(SampleRecord(**{k: d[k] for k in SampleRecord.__dataclass_fields__ if k in d}))
    out.sort(key=lambda r: r.row_index)
    return out
