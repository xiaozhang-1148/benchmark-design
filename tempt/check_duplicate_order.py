#!/usr/bin/env python3
"""Check JSON annotation files for duplicate ``order`` values."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "/mnt/nvme_metadata/private/qitian/labeled_data/Batch01_qitian_9911/"
    "20260630-Batch01_remove_operatorname/Folders-Doc"
)


def find_duplicate_orders(json_path: Path) -> dict[str, int]:
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    orders = [
        str(annotation.get("order"))
        for annotation in payload.get("annotations", [])
        if annotation.get("order") is not None
    ]
    return {order: count for order, count in Counter(orders).items() if count > 1}


def check_directory(input_dir: Path) -> list[tuple[Path, dict[str, int]]]:
    results: list[tuple[Path, dict[str, int]]] = []
    for json_path in sorted(input_dir.glob("*.json")):
        duplicates = find_duplicate_orders(json_path)
        if duplicates:
            results.append((json_path, duplicates))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find JSON files whose annotations contain duplicate order values."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing paired .jpg/.json annotation files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print duplicate order values and counts for each file",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")

    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"no JSON files found in: {input_dir}")

    duplicates = check_directory(input_dir)
    if not duplicates:
        print(f"No duplicate order values found in {len(json_paths)} JSON files under {input_dir}")
        return

    print(f"Found {len(duplicates)} file(s) with duplicate order values:")
    for json_path, duplicate_orders in duplicates:
        print(json_path.name)
        if args.verbose:
            for order, count in sorted(duplicate_orders.items()):
                print(f"  order={order!r} count={count}")


if __name__ == "__main__":
    main()
