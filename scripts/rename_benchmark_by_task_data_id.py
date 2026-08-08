#!/usr/bin/env python3
"""Rename ALL_Benchmark short ids to change_name_Folder names via taskId+dataId.

1. Read taskId/dataId from Folders-Doc/*.json (short stem)
2. Read taskId/dataId from change_name_Folder/*.json (long stem)
3. Join on (taskId, dataId) → name mapping table
4. In ALL_Benchmark: rename {old}.jpg / {old}.jpg.json → {new}.*,
   and update JSON field image_name.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEFAULT_DOC = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/tempt_data/Folders-Doc"
)
DEFAULT_CHANGE = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/tempt_data/change_name_Folder"
)
DEFAULT_BENCH = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/ALL-data/ALL_Benchmark"
)
DEFAULT_MAP = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/tempt_data/"
    "name_mapping_folders_doc_to_change_name.csv"
)


def load_stem_by_ids(folder: Path) -> dict[tuple[str, str], str]:
    """Return {(taskId, dataId): stem} from *.json under folder."""
    out: dict[tuple[str, str], str] = {}
    for p in sorted(folder.glob("*.json")):
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if "taskId" not in data or "dataId" not in data:
            raise KeyError(f"missing taskId/dataId in {p}")
        key = (str(data["taskId"]), str(data["dataId"]))
        if key in out:
            raise ValueError(f"duplicate (taskId,dataId)={key} in {folder}: {out[key]} vs {p.stem}")
        out[key] = p.stem
    return out


def build_mapping(
    doc_dir: Path, change_dir: Path
) -> list[dict[str, str]]:
    doc_map = load_stem_by_ids(doc_dir)
    chg_map = load_stem_by_ids(change_dir)
    common = set(doc_map) & set(chg_map)
    if len(common) != len(doc_map) or len(common) != len(chg_map):
        print(
            f"[warn] doc={len(doc_map)} change={len(chg_map)} "
            f"common={len(common)} doc_only={len(set(doc_map)-set(chg_map))} "
            f"change_only={len(set(chg_map)-set(doc_map))}",
            flush=True,
        )
    rows: list[dict[str, str]] = []
    for key in sorted(common, key=lambda k: doc_map[k]):
        task_id, data_id = key
        rows.append(
            {
                "taskId": task_id,
                "dataId": data_id,
                "old_stem": doc_map[key],
                "new_stem": chg_map[key],
                "old_jpg": f"{doc_map[key]}.jpg",
                "new_jpg": f"{chg_map[key]}.jpg",
                "old_json": f"{doc_map[key]}.jpg.json",
                "new_json": f"{chg_map[key]}.jpg.json",
            }
        )
    return rows


def write_mapping(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "taskId",
                "dataId",
                "old_stem",
                "new_stem",
                "old_jpg",
                "new_jpg",
                "old_json",
                "new_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    # also json for convenience
    json_path = path.with_suffix(".json")
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[map] wrote {len(rows)} rows → {path}", flush=True)
    print(f"[map] wrote {json_path}", flush=True)


def apply_renames(
    bench: Path, rows: list[dict[str, str]], *, dry_run: bool
) -> tuple[int, int, list[str]]:
    renamed = 0
    skipped = 0
    errors: list[str] = []

    for i, row in enumerate(rows, 1):
        old_stem = row["old_stem"]
        new_stem = row["new_stem"]
        old_jpg = bench / f"{old_stem}.jpg"
        new_jpg = bench / f"{new_stem}.jpg"
        old_json = bench / f"{old_stem}.jpg.json"
        new_json = bench / f"{new_stem}.jpg.json"

        if not old_jpg.is_file() and not old_json.is_file():
            # already renamed or missing
            if new_jpg.is_file() and new_json.is_file():
                skipped += 1
                continue
            errors.append(f"missing both old files for {old_stem}")
            continue

        if new_jpg.exists() or new_json.exists():
            errors.append(f"target exists for {old_stem} → {new_stem}")
            continue

        if not old_jpg.is_file() or not old_json.is_file():
            errors.append(f"incomplete pair for {old_stem}: jpg={old_jpg.is_file()} json={old_json.is_file()}")
            continue

        if dry_run:
            renamed += 1
        else:
            # update image_name inside json then rename both
            with old_json.open(encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "image_name" in payload:
                payload["image_name"] = f"{new_stem}.jpg"
                with old_json.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
            old_jpg.rename(new_jpg)
            old_json.rename(new_json)
            renamed += 1

        if i % 500 == 0 or i == len(rows):
            print(f"[rename] {i}/{len(rows)} processed (renamed={renamed} skipped={skipped})", flush=True)

    return renamed, skipped, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-dir", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--change-dir", type=Path, default=DEFAULT_CHANGE)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--mapping-out", type=Path, default=DEFAULT_MAP)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build mapping and validate, do not rename",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="only write mapping table, skip rename",
    )
    args = parser.parse_args()

    for p in (args.doc_dir, args.change_dir, args.benchmark_dir):
        if not p.is_dir():
            print(f"[error] not a directory: {p}", file=sys.stderr)
            return 1

    print("[1] building name mapping …", flush=True)
    rows = build_mapping(args.doc_dir, args.change_dir)
    write_mapping(rows, args.mapping_out)

    if args.map_only:
        print("[done] map-only", flush=True)
        return 0

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"[2] renaming under {args.benchmark_dir} ({mode}) …", flush=True)
    renamed, skipped, errors = apply_renames(
        args.benchmark_dir, rows, dry_run=args.dry_run
    )
    print(
        f"[done] renamed={renamed} skipped={skipped} errors={len(errors)}",
        flush=True,
    )
    if errors:
        err_path = args.mapping_out.with_name("rename_errors.txt")
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"[warn] first errors: {errors[:5]}", flush=True)
        print(f"[warn] all errors → {err_path}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
