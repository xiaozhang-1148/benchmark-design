"""Shard save / resume / atomic merge for large-scale extraction."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, ensure_dir


GLOBAL_DIM = 896
LOCAL_DIM = 896
CONCAT_DIM = 1792


def shard_dir(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "shards" / f"shard_{shard_id:02d}"


def init_shard_store(sdir: Path, capacity: int) -> dict[str, Any]:
    """Preallocate memmap arrays + progress json for one shard."""
    ensure_dir(sdir)
    g_path = sdir / "global.f32.mmap"
    l_path = sdir / "local.f32.mmap"
    c_path = sdir / "concat.f32.mmap"
    # Always create fresh maps sized to this shard's assigned rows
    g = np.memmap(g_path, dtype=np.float32, mode="w+", shape=(capacity, GLOBAL_DIM))
    l = np.memmap(l_path, dtype=np.float32, mode="w+", shape=(capacity, LOCAL_DIM))
    c = np.memmap(c_path, dtype=np.float32, mode="w+", shape=(capacity, CONCAT_DIM))
    g.flush()
    l.flush()
    c.flush()
    progress = {
        "capacity": capacity,
        "n_ok": 0,
        "n_fail": 0,
        "last_row_index": None,
        "last_local_index": None,
    }
    atomic_write_json(sdir / "progress.json", progress)
    (sdir / "index.jsonl").write_text("", encoding="utf-8")
    (sdir / "errors.jsonl").write_text("", encoding="utf-8")
    return {"global": g, "local": l, "concat": c, "progress": progress}


def open_shard_store(sdir: Path) -> dict[str, Any]:
    prog = json.loads((sdir / "progress.json").read_text(encoding="utf-8"))
    cap = int(prog["capacity"])
    return {
        "global": np.memmap(sdir / "global.f32.mmap", dtype=np.float32, mode="r+", shape=(cap, GLOBAL_DIM)),
        "local": np.memmap(sdir / "local.f32.mmap", dtype=np.float32, mode="r+", shape=(cap, LOCAL_DIM)),
        "concat": np.memmap(sdir / "concat.f32.mmap", dtype=np.float32, mode="r+", shape=(cap, CONCAT_DIM)),
        "progress": prog,
    }


def _flush_progress(sdir: Path, prog: dict[str, Any], *, force: bool = False) -> None:
    """Write compact progress.json (no ever-growing done_* lists)."""
    pending = int(prog.pop("_dirty", 0))
    if not force and pending < 32:
        prog["_dirty"] = pending
        return
    compact = {
        "capacity": int(prog["capacity"]),
        "n_ok": int(prog.get("n_ok", 0)),
        "n_fail": int(prog.get("n_fail", 0)),
        # Keep short tails only for debugging; authoritative resume source is index.jsonl
        "last_row_index": prog.get("last_row_index"),
        "last_local_index": prog.get("last_local_index"),
    }
    atomic_write_json(sdir / "progress.json", compact)
    prog["_dirty"] = 0


def write_shard_success(
    store: dict[str, Any],
    sdir: Path,
    *,
    local_index: int,
    row_index: int,
    sample: dict[str, Any],
    global_vec: np.ndarray,
    local_vec: np.ndarray,
    concat_vec: np.ndarray,
) -> None:
    store["global"][local_index] = global_vec
    store["local"][local_index] = local_vec
    store["concat"][local_index] = concat_vec
    # Avoid fsync-every-sample (was a major CPU/IO bottleneck under 16 workers).
    if int(local_index) % 8 == 0:
        store["global"].flush()
        store["local"].flush()
        store["concat"].flush()

    with (sdir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    prog = store["progress"]
    prog["n_ok"] = int(prog.get("n_ok", 0)) + 1
    prog["last_row_index"] = int(row_index)
    prog["last_local_index"] = int(local_index)
    prog["_dirty"] = int(prog.get("_dirty", 0)) + 1
    _flush_progress(sdir, prog, force=False)


def write_shard_error(sdir: Path, store: dict[str, Any], error_row: dict[str, Any]) -> None:
    with (sdir / "errors.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(error_row, ensure_ascii=False) + "\n")
    prog = store["progress"]
    prog["n_fail"] = int(prog.get("n_fail", 0)) + 1
    prog["_dirty"] = int(prog.get("_dirty", 0)) + 32  # force flush soon
    _flush_progress(sdir, prog, force=True)


def done_row_set(sdir: Path) -> set[int]:
    """Resume from index.jsonl (authoritative), not from progress.json lists."""
    done: set[int] = set()
    idx = sdir / "index.jsonl"
    if not idx.exists():
        return done
    with idx.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["row_index"]))
            except Exception:
                continue
    return done


def finalize_shard_progress(sdir: Path, store: dict[str, Any]) -> None:
    store["global"].flush()
    store["local"].flush()
    store["concat"].flush()
    _flush_progress(sdir, store["progress"], force=True)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    # Use a stem without .npy so np.save writes exactly ``tmp.npy``.
    tmp_stem = path.parent / f".{path.stem}.{os.getpid()}.{time.time_ns()}.tmp"
    tmp_path = Path(str(tmp_stem) + ".npy")
    try:
        np.save(str(tmp_stem), arr)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists() and tmp_path.resolve() != path.resolve():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def merge_shards(
    output_dir: Path,
    *,
    n_shards: int,
    n_expected: int,
) -> dict[str, Any]:
    """Merge shard memmaps into final contiguous npy files (row_index order)."""
    output_dir = Path(output_dir)
    global_out = np.zeros((n_expected, GLOBAL_DIM), dtype=np.float32)
    local_out = np.zeros((n_expected, LOCAL_DIM), dtype=np.float32)
    concat_out = np.zeros((n_expected, CONCAT_DIM), dtype=np.float32)
    index_rows: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    seen_rows: set[int] = set()

    for sid in range(n_shards):
        sdir = shard_dir(output_dir, sid)
        if not (sdir / "progress.json").exists():
            raise RuntimeError(f"missing shard progress: {sdir}")
        prog = json.loads((sdir / "progress.json").read_text(encoding="utf-8"))
        # Rebuild mapping local_index → row_index from index.jsonl (authoritative)
        local_to_row: dict[int, int] = {}
        if (sdir / "index.jsonl").exists():
            with (sdir / "index.jsonl").open("r", encoding="utf-8") as f:
                # index.jsonl may have duplicate lines if resume rewrote; last wins
                tmp_rows: list[dict[str, Any]] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tmp_rows.append(json.loads(line))
                # Deduplicate by row_index keeping last
                by_row: dict[int, dict[str, Any]] = {}
                for r in tmp_rows:
                    by_row[int(r["row_index"])] = r
                # Need local_index: store it in sample records
                for r in by_row.values():
                    li = int(r["shard_local_index"])
                    ri = int(r["row_index"])
                    local_to_row[li] = ri
                    if ri in seen_rows:
                        raise RuntimeError(f"duplicate row_index across shards: {ri}")
                    seen_rows.add(ri)
                    index_rows[ri] = r

        cap = int(prog["capacity"])
        g = np.memmap(sdir / "global.f32.mmap", dtype=np.float32, mode="r", shape=(cap, GLOBAL_DIM))
        l = np.memmap(sdir / "local.f32.mmap", dtype=np.float32, mode="r", shape=(cap, LOCAL_DIM))
        c = np.memmap(sdir / "concat.f32.mmap", dtype=np.float32, mode="r", shape=(cap, CONCAT_DIM))
        for li, ri in local_to_row.items():
            if not (0 <= ri < n_expected):
                raise RuntimeError(f"row_index {ri} out of range [0,{n_expected})")
            global_out[ri] = g[li]
            local_out[ri] = l[li]
            concat_out[ri] = c[li]

        if (sdir / "errors.jsonl").exists():
            with (sdir / "errors.jsonl").open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        errors.append(json.loads(line))

    missing = sorted(set(range(n_expected)) - seen_rows)
    if missing:
        raise RuntimeError(
            f"merge incomplete: missing {len(missing)} rows e.g. {missing[:20]}; "
            f"refusing to write final feature files"
        )

    g_path = output_dir / "deepseek_ocr2_global_mean_fp32.npy"
    l_path = output_dir / "deepseek_ocr2_local_mean_fp32.npy"
    c_path = output_dir / "deepseek_ocr2_global_local_concat_fp32.npy"
    atomic_save_npy(g_path, global_out)
    atomic_save_npy(l_path, local_out)
    atomic_save_npy(c_path, concat_out)

    # Write sample_index.jsonl in row order
    idx_path = output_dir / "sample_index.jsonl"
    tmp_idx = idx_path.with_suffix(".jsonl.tmp")
    with tmp_idx.open("w", encoding="utf-8") as f:
        for ri in range(n_expected):
            f.write(json.dumps(index_rows[ri], ensure_ascii=False) + "\n")
    os.replace(tmp_idx, idx_path)

    err_path = output_dir / "extraction_errors.jsonl"
    with err_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return {
        "n_merged": n_expected,
        "n_errors_logged": len(errors),
        "global_path": str(g_path),
        "local_path": str(l_path),
        "concat_path": str(c_path),
        "index_path": str(idx_path),
        "errors_path": str(err_path),
        "merged_utc": datetime.now(timezone.utc).isoformat(),
    }
