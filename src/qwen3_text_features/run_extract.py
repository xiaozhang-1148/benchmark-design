"""Multi-GPU Qwen3-Embedding page-text feature extraction CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ..utils import atomic_write_json, ensure_dir
from .assemble import AssembleError, assemble_from_json_path
from .audit import audit_outputs
from .encoder import Qwen3EmbeddingEncoder
from .manifest import SampleRecord, discover_json_samples, load_records, write_jsonl
from .shard_io import (
    EMB_DIM,
    done_row_set,
    finalize_shard,
    init_shard_store,
    merge_shards,
    open_shard_store,
    shard_dir,
    write_error,
    write_success,
)

DEFAULT_MODEL = (
    "/mnt/nvme_model/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
)
DEFAULT_INPUT = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/ALL-data/ALL_Benchmark"
)
DEFAULT_OUTPUT = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/text_feature"
)
SOFT_TOKEN_LIMIT = 2048


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen3-Embedding-0.6B page OCR text features")
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--num-gpus", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--soft-token-limit", type=int, default=SOFT_TOKEN_LIMIT)
    p.add_argument("--allow-unexpected-long", action="store_true",
                   help="Continue even if token_count > soft limit (still never truncate)")
    p.add_argument("--token-scan-only", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--skip-discover-scan", action="store_true")
    p.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--gpu-id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--records-jsonl", type=str, default=None, help=argparse.SUPPRESS)
    return p


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _select(records: list[SampleRecord], args: argparse.Namespace) -> list[SampleRecord]:
    start = int(args.start_index)
    end = int(args.end_index) if args.end_index is not None else len(records)
    if args.limit is not None:
        end = min(end, start + int(args.limit))
    end = min(end, len(records))
    start = max(0, min(start, end))
    selected = []
    for i, r in enumerate(records[start:end]):
        selected.append(
            SampleRecord(
                row_index=i,
                sample_id=r.sample_id,
                json_path=r.json_path,
                image_name=r.image_name,
                status=r.status,
                error_message=r.error_message,
            )
        )
    return selected


def _split(rows: list[SampleRecord], n: int) -> list[list[SampleRecord]]:
    buckets: list[list[SampleRecord]] = [[] for _ in range(n)]
    for i, r in enumerate(rows):
        buckets[i % n].append(r)
    return buckets


def run_token_scan(
    records: list[SampleRecord],
    model_path: str,
    out_path: Path,
    *,
    soft_limit: int,
) -> dict[str, Any]:
    """CPU tokenizer-only length stats for the whole selected set."""
    os.environ.setdefault("HF_HOME", "/mnt/nvme_model/.cache/huggingface")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", trust_remote_code=True, local_files_only=True
    )
    tok.padding_side = "left"
    counts: list[int] = []
    unexpected: list[dict[str, Any]] = []
    hard = 32768
    errors: list[dict[str, Any]] = []
    for r in tqdm(records, desc="token-scan"):
        try:
            page = assemble_from_json_path(r.json_path, sample_id=r.sample_id)
            n = len(
                tok(page.page_text, padding=False, truncation=False, add_special_tokens=True)[
                    "input_ids"
                ]
            )
            counts.append(n)
            if n > soft_limit:
                unexpected.append(
                    {
                        "row_index": r.row_index,
                        "sample_id": r.sample_id,
                        "json_path": r.json_path,
                        "token_count": n,
                        "flag": "unexpected_long_sample",
                    }
                )
            if n > hard:
                unexpected.append(
                    {
                        "row_index": r.row_index,
                        "sample_id": r.sample_id,
                        "token_count": n,
                        "flag": "exceeds_model_context",
                    }
                )
        except Exception as e:  # noqa: BLE001
            errors.append(
                {
                    "row_index": r.row_index,
                    "sample_id": r.sample_id,
                    "json_path": r.json_path,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    arr = np.array(counts, dtype=np.int64) if counts else np.array([], dtype=np.int64)
    stats = {
        "n_scanned": len(counts),
        "n_errors": len(errors),
        "min": int(arr.min()) if arr.size else None,
        "max": int(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "p50": float(np.percentile(arr, 50)) if arr.size else None,
        "p90": float(np.percentile(arr, 90)) if arr.size else None,
        "p95": float(np.percentile(arr, 95)) if arr.size else None,
        "p99": float(np.percentile(arr, 99)) if arr.size else None,
        "gt_1600": int((arr > 1600).sum()) if arr.size else 0,
        "gt_2048": int((arr > soft_limit).sum()) if arr.size else 0,
        "gt_model_ctx": int((arr > hard).sum()) if arr.size else 0,
        "soft_limit": soft_limit,
        "hard_limit": hard,
        "unexpected_long_count": len([u for u in unexpected if u.get("flag") == "unexpected_long_sample"]),
        "unexpected_examples": unexpected[:20],
        "errors_examples": errors[:20],
    }
    atomic_write_json(out_path, stats)
    return stats


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    assert args.worker_id is not None and args.gpu_id is not None
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("HF_HOME", "/mnt/nvme_model/.cache/huggingface")
    os.environ.pop("HF_ENDPOINT", None)

    worker_id = int(args.worker_id)
    output_dir = Path(args.output_dir)
    sdir = shard_dir(output_dir, worker_id)
    records = load_records(args.records_jsonl)

    resume = bool(args.resume) and not bool(args.no_resume)
    if resume and (sdir / "progress.json").exists():
        store = open_shard_store(sdir)
        if int(store["progress"]["capacity"]) != len(records):
            store = init_shard_store(sdir, capacity=len(records))
            done: set[int] = set()
        else:
            done = done_row_set(sdir)
    else:
        store = init_shard_store(sdir, capacity=len(records))
        done = set()

    row_to_local = {r.row_index: i for i, r in enumerate(records)}
    pending = [r for r in records if r.row_index not in done]
    print(
        f"[worker {worker_id}] assigned={len(records)} done={len(done)} pending={len(pending)} "
        f"batch={args.batch_size}",
        flush=True,
    )

    enc = Qwen3EmbeddingEncoder(
        args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=_dtype(args.dtype),
        local_files_only=True,
    )
    info = enc.info.__dict__
    soft = int(args.soft_token_limit)
    allow_long = bool(args.allow_unexpected_long)

    t0 = time.time()
    ok = fail = 0
    bs = max(1, int(args.batch_size))

    for i in tqdm(range(0, len(pending), bs), desc=f"gpu{args.gpu_id}", mininterval=2.0):
        batch = pending[i : i + bs]
        pages = []
        try:
            for r in batch:
                pages.append(assemble_from_json_path(r.json_path, sample_id=r.sample_id))
            texts = [p.page_text for p in pages]
            # pre-check soft limit with tokenizer counts from embed path
            out = enc.embed_texts(texts)
            for j, (r, page) in enumerate(zip(batch, pages)):
                tc = int(out["token_counts"][j])
                if tc > soft and not allow_long:
                    raise RuntimeError(
                        f"unexpected_long_sample row={r.row_index} tokens={tc} > soft={soft}"
                    )
                emb = out["embeddings"][j].numpy().astype(np.float32, copy=False)
                idx = page.to_index_dict(row_index=r.row_index, token_count=tc, status="success")
                idx["shard_id"] = worker_id
                idx["shard_local_index"] = row_to_local[r.row_index]
                asm = page.to_assembled_dict(row_index=r.row_index)
                write_success(
                    store,
                    sdir,
                    local_index=row_to_local[r.row_index],
                    row_index=r.row_index,
                    index_row=idx,
                    assembled_row=asm,
                    emb=emb,
                )
                ok += 1
        except Exception:
            # fall back per sample
            for r in batch:
                if r.row_index in done_row_set(sdir):
                    continue
                try:
                    page = assemble_from_json_path(r.json_path, sample_id=r.sample_id)
                    out = enc.embed_texts([page.page_text])
                    tc = int(out["token_counts"][0])
                    if tc > soft and not allow_long:
                        raise RuntimeError(
                            f"unexpected_long_sample tokens={tc} > soft={soft}"
                        )
                    emb = out["embeddings"][0].numpy().astype(np.float32, copy=False)
                    idx = page.to_index_dict(row_index=r.row_index, token_count=tc, status="success")
                    idx["shard_id"] = worker_id
                    idx["shard_local_index"] = row_to_local[r.row_index]
                    write_success(
                        store,
                        sdir,
                        local_index=row_to_local[r.row_index],
                        row_index=r.row_index,
                        index_row=idx,
                        assembled_row=page.to_assembled_dict(row_index=r.row_index),
                        emb=emb,
                    )
                    ok += 1
                except Exception as e2:  # noqa: BLE001
                    err = {
                        "row_index": r.row_index,
                        "sample_id": r.sample_id,
                        "json_path": r.json_path,
                        "status": "failed",
                        "error": f"{type(e2).__name__}: {e2}",
                        "traceback": traceback.format_exc(),
                    }
                    write_error(sdir, store, err)
                    fail += 1
                    finalize_shard(sdir, store)
                    enc.close()
                    return {
                        "ok": False,
                        "worker_id": worker_id,
                        "n_ok": ok,
                        "n_fail": fail,
                        "fatal_error": err,
                        "elapsed_sec": time.time() - t0,
                        "model_info": info,
                    }

    finalize_shard(sdir, store)
    enc.close()
    return {
        "ok": fail == 0,
        "worker_id": worker_id,
        "gpu_id": args.gpu_id,
        "n_ok": ok,
        "n_fail": fail,
        "elapsed_sec": time.time() - t0,
        "model_info": info,
    }


def _launch_workers(
    args: argparse.Namespace, buckets: list[list[SampleRecord]], n_gpus: int
) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "worker_logs")
    ensure_dir(output_dir / "worker_inputs")
    procs: list[subprocess.Popen] = []
    result_paths: list[Path] = []
    for wid in range(n_gpus):
        rows = buckets[wid]
        if not rows:
            continue
        rec_path = output_dir / "worker_inputs" / f"worker_{wid:02d}.jsonl"
        write_jsonl(rows, rec_path)
        result_path = output_dir / "worker_logs" / f"worker_{wid:02d}.result.json"
        if result_path.exists():
            result_path.unlink()
        result_paths.append(result_path)
        cmd = [
            sys.executable,
            "-m",
            "src.qwen3_text_features.run_extract",
            "--input-dir",
            args.input_dir,
            "--output-dir",
            str(output_dir),
            "--model-path",
            args.model_path,
            "--batch-size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
            "--soft-token-limit",
            str(args.soft_token_limit),
            "--worker-id",
            str(wid),
            "--gpu-id",
            str(wid),
            "--records-jsonl",
            str(rec_path),
        ]
        if args.no_resume:
            cmd.append("--no-resume")
        if args.allow_unexpected_long:
            cmd.append("--allow-unexpected-long")
        log_f = open(output_dir / "worker_logs" / f"worker_{wid:02d}.log", "w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(wid)
        env["HF_HOME"] = "/mnt/nvme_model/.cache/huggingface"
        env.pop("HF_ENDPOINT", None)
        procs.append(
            subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        )
        time.sleep(0.5)

    print(f"[main] launched {len(procs)} workers", flush=True)
    while True:
        alive = sum(1 for p in procs if p.poll() is None)
        total_ok = 0
        for wid in range(n_gpus):
            pp = shard_dir(output_dir, wid) / "progress.json"
            if pp.exists():
                try:
                    total_ok += int(json.loads(pp.read_text()).get("n_ok", 0))
                except Exception:
                    pass
        print(f"[main] alive={alive}/{len(procs)} total_n_ok≈{total_ok}", flush=True)
        if alive == 0:
            break
        time.sleep(20.0)

    exit_codes = [p.wait() for p in procs]
    results = []
    for rp, ec in zip(result_paths, exit_codes):
        if rp.exists():
            results.append(json.loads(rp.read_text(encoding="utf-8")))
        else:
            results.append({"ok": False, "error": f"missing result exit={ec}"})
    return results


def write_manifest(
    args: argparse.Namespace,
    *,
    n_total: int,
    n_success: int,
    n_fail: int,
    model_info: dict[str, Any] | None,
    token_stats: dict[str, Any] | None,
    worker_results: list[dict[str, Any]],
    merge_info: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    started: str,
    elapsed: float,
) -> Path:
    output_dir = Path(args.output_dir)
    emb_path = output_dir / "qwen3_embedding_0.6b_last_token_raw_fp32.npy"
    out_shape = None
    if emb_path.exists():
        arr = np.load(emb_path, mmap_mode="r")
        out_shape = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    manifest = {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "model_path": args.model_path,
        "model_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "tokenizer_name": "Qwen/Qwen3-Embedding-0.6B",
        "tokenizer_revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "embedding_dimension": EMB_DIM,
        "model_context_length": (model_info or {}).get("max_position_embeddings", 32768),
        "hidden_size": (model_info or {}).get("hidden_size", 1024),
        "padding_side": "left",
        "truncation": False,
        "pooling": "last_valid_token",
        "inference_dtype": args.dtype,
        "output_dtype": "float32",
        "instruction_used": False,
        "chat_template_used": False,
        "text_generation_used": False,
        "ocr_content_modified": False,
        "l2_normalized": False,
        "dataset_sample_count": n_total,
        "success_count": n_success,
        "failure_count": n_fail,
        "token_length_stats": token_stats,
        "output_file": out_shape,
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "worker_results": worker_results,
        "merge_info": merge_info,
        "audit": audit,
        "model_info": model_info,
        "sort_rule": "resolved_json_path ascending (dictionary order)",
    }
    path = output_dir / "text_extraction_manifest.json"
    atomic_write_json(path, manifest)
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    os.environ.setdefault("HF_HOME", "/mnt/nvme_model/.cache/huggingface")
    os.environ.pop("HF_ENDPOINT", None)

    if args.worker_id is not None:
        result = run_worker(args)
        atomic_write_json(
            output_dir / "worker_logs" / f"worker_{int(args.worker_id):02d}.result.json",
            result,
        )
        return 0 if result.get("ok") else 1

    if args.audit_only:
        audit = audit_outputs(output_dir)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 0 if audit.get("passed") else 1

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    man_path = output_dir / "dataset_manifest.jsonl"
    if man_path.exists() and not args.no_resume:
        records = load_records(man_path)
        print(f"[main] loaded manifest n={len(records)}")
    else:
        records = discover_json_samples(args.input_dir, recursive=False)
        write_jsonl(records, man_path)
        print(f"[main] discovered n={len(records)} sort=resolved_path")

    selected = _select(records, args)
    write_jsonl(selected, output_dir / "run_sample_manifest.jsonl")
    print(f"[main] selected {len(selected)}", flush=True)

    token_stats = None
    if args.token_scan_only or not args.skip_discover_scan:
        token_stats = run_token_scan(
            selected,
            args.model_path,
            output_dir / "token_length_stats.json",
            soft_limit=int(args.soft_token_limit),
        )
        print(json.dumps({k: token_stats[k] for k in token_stats if k != "unexpected_examples"}, indent=2))
        if token_stats.get("unexpected_long_count", 0) > 0 and not args.allow_unexpected_long:
            print(
                "[main] STOP: unexpected_long_sample detected (token > soft limit). "
                "Re-run with --allow-unexpected-long after review, or raise --soft-token-limit.",
                flush=True,
            )
            write_manifest(
                args,
                n_total=len(selected),
                n_success=0,
                n_fail=0,
                model_info=None,
                token_stats=token_stats,
                worker_results=[],
                merge_info=None,
                audit=None,
                started=started,
                elapsed=time.time() - t0,
            )
            return 2
        if args.token_scan_only:
            return 0

    if args.merge_only:
        merge_info = merge_shards(output_dir, n_shards=int(args.num_gpus), n_expected=len(selected))
        audit = audit_outputs(output_dir, n_expected=len(selected))
        write_manifest(
            args,
            n_total=len(selected),
            n_success=audit.get("n", 0),
            n_fail=0,
            model_info=None,
            token_stats=token_stats,
            worker_results=[],
            merge_info=merge_info,
            audit=audit,
            started=started,
            elapsed=time.time() - t0,
        )
        print(json.dumps({"merge": merge_info, "audit": audit}, indent=2, ensure_ascii=False))
        return 0 if audit.get("passed") else 1

    n_gpus = max(1, min(int(args.num_gpus), max(1, torch.cuda.device_count())))
    buckets = _split(selected, n_gpus)
    print(f"[main] using {n_gpus} GPUs batch_size={args.batch_size}", flush=True)

    if n_gpus == 1:
        rec_path = output_dir / "worker_inputs" / "worker_00.jsonl"
        ensure_dir(rec_path.parent)
        write_jsonl(buckets[0], rec_path)
        wargs = argparse.Namespace(**vars(args))
        wargs.worker_id = 0
        wargs.gpu_id = 0
        wargs.records_jsonl = str(rec_path)
        ensure_dir(output_dir / "worker_logs")
        result = run_worker(wargs)
        atomic_write_json(output_dir / "worker_logs" / "worker_00.result.json", result)
        worker_results = [result]
    else:
        worker_results = _launch_workers(args, buckets, n_gpus)

    if not all(r.get("ok") for r in worker_results):
        print("[main] FATAL: worker failure; refusing merge", flush=True)
        write_manifest(
            args,
            n_total=len(selected),
            n_success=sum(int(r.get("n_ok", 0)) for r in worker_results),
            n_fail=sum(int(r.get("n_fail", 0)) for r in worker_results),
            model_info=worker_results[0].get("model_info") if worker_results else None,
            token_stats=token_stats,
            worker_results=worker_results,
            merge_info=None,
            audit=None,
            started=started,
            elapsed=time.time() - t0,
        )
        return 1

    merge_info = merge_shards(output_dir, n_shards=n_gpus, n_expected=len(selected))
    audit = audit_outputs(output_dir, n_expected=len(selected))
    write_manifest(
        args,
        n_total=len(selected),
        n_success=audit.get("n", 0),
        n_fail=sum(int(r.get("n_fail", 0)) for r in worker_results),
        model_info=worker_results[0].get("model_info") if worker_results else None,
        token_stats=token_stats,
        worker_results=worker_results,
        merge_info=merge_info,
        audit=audit,
        started=started,
        elapsed=time.time() - t0,
    )
    print(json.dumps({"merge": merge_info, "audit_passed": audit.get("passed"), "audit": audit}, indent=2))
    return 0 if audit.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
