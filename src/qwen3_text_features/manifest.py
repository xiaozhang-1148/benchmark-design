"""Discover JSON samples and build manifests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class SampleRecord:
    row_index: int
    sample_id: str
    json_path: str
    image_name: str | None = None
    status: str = "pending"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_json_samples(input_dir: str | Path, *, recursive: bool = False) -> list[SampleRecord]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir not found: {root}")
    if recursive:
        paths = sorted(p for p in root.rglob("*.json") if p.is_file())
    else:
        # Prefer paired page JSON: *.jpg.json etc. Also accept plain *.json
        paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".json")

    # Stable dict-order by resolved path string
    paths = sorted(paths, key=lambda p: str(p.resolve()))
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    records: list[SampleRecord] = []
    for p in paths:
        ap = str(p.resolve())
        if ap in seen_paths:
            continue
        seen_paths.add(ap)
        # sample_id = filename (unique in flat dir)
        sid = p.name
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        # image_name heuristic: strip trailing .json
        image_name = p.name[:-5] if p.name.endswith(".json") else p.stem
        records.append(
            SampleRecord(
                row_index=len(records),
                sample_id=sid,
                json_path=ap,
                image_name=image_name,
            )
        )
    return records


def write_jsonl(records: list[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            obj = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(path)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: str | Path) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    for d in iter_jsonl(path):
        out.append(SampleRecord(**{k: d[k] for k in SampleRecord.__dataclass_fields__ if k in d}))
    out.sort(key=lambda r: r.row_index)
    return out
