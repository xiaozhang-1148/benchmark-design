"""Shard IO for text embeddings."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, ensure_dir

EMB_DIM = 1024


def shard_dir(output_dir: Path, shard_id: int) -> Path:
    return output_dir / "shards" / f"shard_{shard_id:02d}"


def init_shard_store(sdir: Path, capacity: int) -> dict[str, Any]:
    ensure_dir(sdir)
    emb = np.memmap(sdir / "emb.f32.mmap", dtype=np.float32, mode="w+", shape=(capacity, EMB_DIM))
    emb.flush()
    prog = {"capacity": capacity, "n_ok": 0, "n_fail": 0, "last_row_index": None, "_dirty": 0}
    atomic_write_json(sdir / "progress.json", {k: v for k, v in prog.items() if k != "_dirty"})
    (sdir / "index.jsonl").write_text("", encoding="utf-8")
    (sdir / "assembled.jsonl").write_text("", encoding="utf-8")
    (sdir / "errors.jsonl").write_text("", encoding="utf-8")
    return {"emb": emb, "progress": prog}


def open_shard_store(sdir: Path) -> dict[str, Any]:
    prog = json.loads((sdir / "progress.json").read_text(encoding="utf-8"))
    cap = int(prog["capacity"])
    prog["_dirty"] = 0
    return {
        "emb": np.memmap(sdir / "emb.f32.mmap", dtype=np.float32, mode="r+", shape=(cap, EMB_DIM)),
        "progress": prog,
    }


def done_row_set(sdir: Path) -> set[int]:
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


def _flush_progress(sdir: Path, prog: dict[str, Any], *, force: bool = False) -> None:
    dirty = int(prog.get("_dirty", 0))
    if not force and dirty < 64:
        return
    compact = {
        "capacity": int(prog["capacity"]),
        "n_ok": int(prog.get("n_ok", 0)),
        "n_fail": int(prog.get("n_fail", 0)),
        "last_row_index": prog.get("last_row_index"),
    }
    atomic_write_json(sdir / "progress.json", compact)
    prog["_dirty"] = 0


def write_success(
    store: dict[str, Any],
    sdir: Path,
    *,
    local_index: int,
    row_index: int,
    index_row: dict[str, Any],
    assembled_row: dict[str, Any],
    emb: np.ndarray,
) -> None:
    store["emb"][local_index] = emb
    if local_index % 16 == 0:
        store["emb"].flush()
    with (sdir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(index_row, ensure_ascii=False) + "\n")
    with (sdir / "assembled.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(assembled_row, ensure_ascii=False) + "\n")
    prog = store["progress"]
    prog["n_ok"] = int(prog.get("n_ok", 0)) + 1
    prog["last_row_index"] = int(row_index)
    prog["_dirty"] = int(prog.get("_dirty", 0)) + 1
    _flush_progress(sdir, prog)


def write_error(sdir: Path, store: dict[str, Any], err: dict[str, Any]) -> None:
    with (sdir / "errors.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(err, ensure_ascii=False) + "\n")
    prog = store["progress"]
    prog["n_fail"] = int(prog.get("n_fail", 0)) + 1
    prog["_dirty"] = 64
    _flush_progress(sdir, prog, force=True)


def finalize_shard(sdir: Path, store: dict[str, Any]) -> None:
    store["emb"].flush()
    _flush_progress(sdir, store["progress"], force=True)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
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


def merge_shards(output_dir: Path, *, n_shards: int, n_expected: int) -> dict[str, Any]:
    output_dir = Path(output_dir)
    emb_out = np.zeros((n_expected, EMB_DIM), dtype=np.float32)
    index_rows: dict[int, dict[str, Any]] = {}
    assembled: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    seen: set[int] = set()

    for sid in range(n_shards):
        sdir = shard_dir(output_dir, sid)
        if not (sdir / "progress.json").exists():
            raise RuntimeError(f"missing shard: {sdir}")
        by_row_idx: dict[int, dict[str, Any]] = {}
        by_row_asm: dict[int, dict[str, Any]] = {}
        if (sdir / "index.jsonl").exists():
            for line in (sdir / "index.jsonl").open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                by_row_idx[int(r["row_index"])] = r
        if (sdir / "assembled.jsonl").exists():
            for line in (sdir / "assembled.jsonl").open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                by_row_asm[int(r["row_index"])] = r
        cap = int(json.loads((sdir / "progress.json").read_text())["capacity"])
        emb = np.memmap(sdir / "emb.f32.mmap", dtype=np.float32, mode="r", shape=(cap, EMB_DIM))
        for ri, r in by_row_idx.items():
            if ri in seen:
                raise RuntimeError(f"duplicate row_index {ri}")
            seen.add(ri)
            li = int(r["shard_local_index"])
            emb_out[ri] = emb[li]
            index_rows[ri] = r
            if ri in by_row_asm:
                assembled[ri] = by_row_asm[ri]
        if (sdir / "errors.jsonl").exists():
            for line in (sdir / "errors.jsonl").open(encoding="utf-8"):
                if line.strip():
                    errors.append(json.loads(line))

    missing = sorted(set(range(n_expected)) - seen)
    if missing:
        raise RuntimeError(f"merge incomplete: missing {len(missing)} rows e.g. {missing[:20]}")

    emb_path = output_dir / "qwen3_embedding_0.6b_last_token_raw_fp32.npy"
    if emb_path.exists():
        raise RuntimeError(f"refusing to overwrite existing {emb_path}")
    atomic_save_npy(emb_path, emb_out)

    idx_path = output_dir / "text_sample_index.jsonl"
    asm_path = output_dir / "assembled_page_texts.jsonl"
    tmp_idx = idx_path.with_suffix(".jsonl.tmp")
    tmp_asm = asm_path.with_suffix(".jsonl.tmp")
    with tmp_idx.open("w", encoding="utf-8") as fi, tmp_asm.open("w", encoding="utf-8") as fa:
        for ri in range(n_expected):
            fi.write(json.dumps(index_rows[ri], ensure_ascii=False) + "\n")
            fa.write(json.dumps(assembled[ri], ensure_ascii=False) + "\n")
    os.replace(tmp_idx, idx_path)
    os.replace(tmp_asm, asm_path)

    err_path = output_dir / "text_extraction_errors.jsonl"
    with err_path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    return {
        "n_merged": n_expected,
        "n_errors": len(errors),
        "emb_path": str(emb_path),
        "index_path": str(idx_path),
        "assembled_path": str(asm_path),
        "errors_path": str(err_path),
    }
