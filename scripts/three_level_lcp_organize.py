#!/usr/bin/env python3
"""两层最长字符匹配：按文件名分层归类并复制图片。

对源目录中的图片文件名（不含扩展名）做两次最长字符匹配：
  第 1 层：全体文件的最长匹配分组键
  第 2 层：在第 1 层内对剩余后缀再做一次最长匹配
  图片直接放入第 2 层目录（不再建第 3 层）

输出目录结构：
  Batch01/
    <L1>/
      <L2>/
        <image>

默认路径：
  SRC  = .../tempt_data/Folders-Doc
  DEST = .../tempt_data/Batch01
"""

from __future__ import annotations

import argparse
import bisect
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

DEFAULT_SRC = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/ALL-data/ALL_Benchmark"
)
DEFAULT_DEST = Path(
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/tempt_data/Batch02"
)


def lcp2(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def nearest_lcp(name: str, sorted_names: list[str]) -> str:
    """与其它名称（经排序邻居）的最长公共前缀。"""
    i = bisect.bisect_left(sorted_names, name)
    cands: list[str] = []

    j = i - 1
    while j >= 0 and sorted_names[j] == name:
        j -= 1
    if j >= 0:
        cands.append(lcp2(name, sorted_names[j]))

    j = i
    while j < len(sorted_names) and sorted_names[j] == name:
        j += 1
    if j < len(sorted_names):
        cands.append(lcp2(name, sorted_names[j]))

    if not cands:
        return ""
    return max(cands, key=len)


def level_key(name: str, sorted_names: list[str]) -> str:
    """一层最长字符匹配键。

    - 先取与最近其它名的 LCP；
    - 若 LCP 覆盖至少一个完整的 ``_`` 分段，则用该分段作为本层键；
    - 否则该名的首个 ``_`` 分段各自成类。
    """
    shared = nearest_lcp(name, sorted_names)

    if "_" in name:
        first = name.split("_", 1)[0]
        if len(shared) >= len(first) and shared.startswith(first) and (
            len(shared) == len(first)
            or shared[len(first)] == "_"
            or len(name) == len(first)
        ):
            return first
        return first

    return shared if shared else name


def partition_by_lcp(names: list[str]) -> dict[str, list[str]]:
    names_sorted = sorted(names)
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        groups[level_key(n, names_sorted)].append(n)
    return dict(groups)


def strip_level(name: str, key: str) -> str:
    if name.startswith(key):
        rest = name[len(key) :]
        return rest[1:] if rest.startswith("_") else rest
    return name


def safe_dir_name(name: str) -> str:
    """文件系统安全的目录名。"""
    name = name.strip() or "(empty)"
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name


def collect_images(src: Path) -> list[Path]:
    return sorted(
        p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_hierarchy(stems: list[str]) -> dict[str, dict[str, list[str]]]:
    """返回 hierarchy[L1][L2] = [stem, ...]。"""
    hierarchy: dict[str, dict[str, list[str]]] = {}

    level1 = partition_by_lcp(stems)
    for l1_key, l1_names in level1.items():
        rem_map: dict[str, list[str]] = defaultdict(list)
        for n in l1_names:
            rem_map[strip_level(n, l1_key)].append(n)

        level2 = partition_by_lcp(list(rem_map.keys()))
        hierarchy[l1_key] = {}
        for l2_key, l2_rems in level2.items():
            originals: list[str] = []
            for r in l2_rems:
                originals.extend(rem_map[r])
            hierarchy[l1_key][l2_key] = originals

    return hierarchy


def print_stats(hierarchy: dict[str, dict[str, list[str]]]) -> None:
    n_l1 = len(hierarchy)
    n_l2 = sum(len(l2) for l2 in hierarchy.values())
    n_imgs = sum(len(stems) for l2 in hierarchy.values() for stems in l2.values())

    print("=" * 60)
    print("两层最长字符匹配统计")
    print("=" * 60)
    print(f"第一层类别数: {n_l1}")
    print(f"第二层类别数: {n_l2}")
    print(f"图片总数:     {n_imgs}")
    print("-" * 60)

    l1_sizes = sorted(
        ((k, sum(len(s) for s in v.values())) for k, v in hierarchy.items()),
        key=lambda x: -x[1],
    )
    print("第一层各类别图片数:")
    for k, cnt in l1_sizes:
        n2 = len(hierarchy[k])
        print(f"  [{cnt:5d} 图 | L2目录={n2:5d}]  {k}")

    print("-" * 60)
    print("第二层各类别文件数（按所属第一层）:")
    for l1_key, _ in l1_sizes:
        l2 = hierarchy[l1_key]
        sizes = sorted(((k2, len(stems)) for k2, stems in l2.items()), key=lambda x: -x[1])
        total_files = sum(c for _, c in sizes)
        print(f"  L1={l1_key}  →  {len(sizes)} 个第二层 | 合计 {total_files} 文件")
        for k2, cnt in sizes[:5]:
            print(f"      [{cnt:4d} 文件]  {k2}")
        if len(sizes) > 5:
            print(f"      ... 另有 {len(sizes) - 5} 个第二层类别")
    print("=" * 60)


def copy_hierarchy(
    hierarchy: dict[str, dict[str, list[str]]],
    stem_to_path: dict[str, Path],
    dest: Path,
    *,
    dry_run: bool = False,
) -> int:
    copied = 0
    for l1_key, l2_map in hierarchy.items():
        for l2_key, stems in l2_map.items():
            out_dir = dest / safe_dir_name(l1_key) / safe_dir_name(l2_key)
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
            for stem in stems:
                src_path = stem_to_path[stem]
                dst_path = out_dir / src_path.name
                if dry_run:
                    copied += 1
                    continue
                shutil.copy2(src_path, dst_path)
                copied += 1
                if copied % 500 == 0:
                    print(f"  已复制 {copied} 张...", flush=True)
    return copied


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="两层最长字符匹配归类并复制图片")
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="源图片目录")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="输出根目录")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计分层，不复制文件",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="复制前清空目标目录",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src: Path = args.src
    dest: Path = args.dest

    if not src.is_dir():
        print(f"源目录不存在: {src}", file=sys.stderr)
        return 1

    images = collect_images(src)
    if not images:
        print(f"源目录中未找到图片: {src}", file=sys.stderr)
        return 1

    stem_to_path = {p.stem: p for p in images}
    if len(stem_to_path) != len(images):
        print("警告: 存在同名（不同扩展名）图片，后者将覆盖映射。", file=sys.stderr)

    stems = list(stem_to_path.keys())
    print(f"源目录: {src}")
    print(f"输出目录: {dest}")
    print(f"读取到图片: {len(stems)}")
    print()

    hierarchy = build_hierarchy(stems)
    print_stats(hierarchy)

    if args.dry_run:
        n = copy_hierarchy(hierarchy, stem_to_path, dest, dry_run=True)
        print(f"[dry-run] 将复制 {n} 张图片到 {dest}")
        return 0

    if args.clean and dest.exists():
        print(f"清空目标目录: {dest}")
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n开始复制到 {dest} ...")
    n = copy_hierarchy(hierarchy, stem_to_path, dest, dry_run=False)
    print(f"完成: 共复制 {n} 张图片 → {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
