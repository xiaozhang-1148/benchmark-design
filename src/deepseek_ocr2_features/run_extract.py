"""CLI entry: multi-GPU DeepSeek-OCR2 Projector-before feature extraction."""

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
from .audit import audit_outputs
from .encoder import CausalFlowVisionEncoder
from .extract import extract_page_features, extract_pages_batched
from .manifest import SampleRecord, discover_samples, enrich_image_sizes, write_manifest_jsonl
from .preprocess import MEAN, STD
from .shard_io import (
    CONCAT_DIM,
    GLOBAL_DIM,
    LOCAL_DIM,
    done_row_set,
    finalize_shard_progress,
    init_shard_store,
    merge_shards,
    open_shard_store,
    shard_dir,
    write_shard_error,
    write_shard_success,
)

DEFAULT_MODEL = (
    "/mnt/nvme_model/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-OCR-2/snapshots/"
    "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
)
DEFAULT_INPUT = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/ALL-data/ALL_Benchmark"
)
DEFAULT_OUTPUT = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/features/vision_deatures"
)


def _git_rev() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[2]),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DeepSeek-OCR2 global+local mean features (896+896=1792, Projector-before)"
    )
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default="aaa02f3811945a91062062994c5c4a3f4c0af2b0")
    p.add_argument("--num-gpus", type=int, default=16)
    p.add_argument(
        "--page-batch-size",
        type=int,
        default=2,
        help="Pages per GPU forward (keep small on A10; local crops are chunked separately)",
    )
    p.add_argument("--max-local-batch", type=int, default=8, help="Max 768px crops per SAM/Qwen forward")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=None, help="Exclusive end; default=all")
    p.add_argument("--limit", type=int, default=None, help="Optional max samples from start")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--attn-implementation",
        default="eager",
        help="HF language-tower attn impl (eager required for DeepseekOCR2ForCausalLM)",
    )
    p.add_argument("--skip-size-scan", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--gpu-id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--records-jsonl", type=str, default=None, help=argparse.SUPPRESS)
    return p


def _select_records(records: list[SampleRecord], args: argparse.Namespace) -> list[SampleRecord]:
    start = int(args.start_index)
    end = int(args.end_index) if args.end_index is not None else len(records)
    if args.limit is not None:
        end = min(end, start + int(args.limit))
    end = min(end, len(records))
    start = max(0, min(start, end))
    return records[start:end]


def _split_for_workers(rows: list[SampleRecord], n_workers: int) -> list[list[SampleRecord]]:
    buckets: list[list[SampleRecord]] = [[] for _ in range(n_workers)]
    for i, r in enumerate(rows):
        buckets[i % n_workers].append(r)
    return buckets


def _write_success(
    store: dict[str, Any],
    sdir: Path,
    *,
    rec: SampleRecord,
    out: dict[str, Any],
    worker_id: int,
    local_index: int,
) -> None:
    meta = out["meta"]
    sample = {
        "row_index": rec.row_index,
        "sample_id": rec.sample_id,
        "image_path": rec.image_path,
        "latex_path_or_id": rec.latex_path_or_id,
        "original_width": meta.original_width,
        "original_height": meta.original_height,
        "local_crop_count": meta.local_crop_count,
        "crop_grid_width": meta.crop_grid_width,
        "crop_grid_height": meta.crop_grid_height,
        "crop_order": meta.crop_order,
        "small_image_fallback": meta.small_image_fallback,
        "status": "success",
        "shard_id": worker_id,
        "shard_local_index": local_index,
        "shapes": out["shapes"],
    }
    write_shard_success(
        store,
        sdir,
        local_index=local_index,
        row_index=rec.row_index,
        sample=sample,
        global_vec=out["global"],
        local_vec=out["local"],
        concat_vec=out["concat"],
    )


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Single-GPU worker process.

    Parent launcher sets ``CUDA_VISIBLE_DEVICES`` to one physical GPU.
    This process always uses logical ``cuda:0``.
    """
    assert args.worker_id is not None and args.gpu_id is not None
    worker_id = int(args.worker_id)
    output_dir = Path(args.output_dir)
    sdir = shard_dir(output_dir, worker_id)
    # If launched without parent env pinning, pin here.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    records: list[SampleRecord] = []
    with open(args.records_jsonl, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            records.append(
                SampleRecord(**{k: d[k] for k in SampleRecord.__dataclass_fields__ if k in d})
            )

    resume = bool(args.resume) and not bool(args.no_resume)
    if resume and (sdir / "progress.json").exists():
        store = open_shard_store(sdir)
        done = done_row_set(sdir)
        # Capacity must match assigned rows
        if int(store["progress"]["capacity"]) != len(records):
            store = init_shard_store(sdir, capacity=len(records))
            done = set()
    else:
        store = init_shard_store(sdir, capacity=len(records))
        done = set()

    row_to_local = {r.row_index: i for i, r in enumerate(records)}
    pending = [r for r in records if r.row_index not in done]
    print(
        f"[worker {worker_id}] gpu={args.gpu_id} assigned={len(records)} "
        f"done={len(done)} pending={len(pending)} page_batch={args.page_batch_size}",
        flush=True,
    )

    encoder = CausalFlowVisionEncoder(
        args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    module_paths = dict(encoder.module_paths.__dict__)

    t0 = time.time()
    ok = 0
    fail = 0
    bs = max(1, int(args.page_batch_size))

    for i in tqdm(range(0, len(pending), bs), desc=f"gpu{args.gpu_id}", mininterval=2.0):
        batch = pending[i : i + bs]
        paths = [r.image_path for r in batch]
        sids = [r.sample_id for r in batch]
        batch_ok = False
        try:
            if len(batch) == 1:
                outs = [
                    extract_page_features(
                        encoder, paths[0], sample_id=sids[0], retry_on_nonfinite=True
                    )
                ]
            else:
                outs = extract_pages_batched(
                    encoder,
                    paths,
                    sids,
                    max_local_batch=int(getattr(args, "max_local_batch", 8)),
                )
            if len(outs) != len(batch):
                raise RuntimeError("batch output length mismatch")
            for rec, out in zip(batch, outs):
                _write_success(
                    store,
                    sdir,
                    rec=rec,
                    out=out,
                    worker_id=worker_id,
                    local_index=row_to_local[rec.row_index],
                )
                ok += 1
            batch_ok = True
        except Exception as batch_exc:  # noqa: BLE001
            batch_ok = False
            batch_err = f"{type(batch_exc).__name__}: {batch_exc}"

        if batch_ok:
            continue

        for rec in batch:
            if rec.row_index in done_row_set(sdir):
                continue
            try:
                out = extract_page_features(
                    encoder, rec.image_path, sample_id=rec.sample_id, retry_on_nonfinite=True
                )
                _write_success(
                    store,
                    sdir,
                    rec=rec,
                    out=out,
                    worker_id=worker_id,
                    local_index=row_to_local[rec.row_index],
                )
                ok += 1
            except Exception as e2:  # noqa: BLE001
                err = {
                    "row_index": rec.row_index,
                    "sample_id": rec.sample_id,
                    "image_path": rec.image_path,
                    "status": "failed",
                    "error": f"{type(e2).__name__}: {e2}",
                    "batch_error": batch_err,
                    "traceback": traceback.format_exc(),
                }
                write_shard_error(sdir, store, err)
                fail += 1
                encoder.close()
                return {
                    "ok": False,
                    "worker_id": worker_id,
                    "gpu_id": args.gpu_id,
                    "n_ok": ok,
                    "n_fail": fail,
                    "fatal_error": err,
                    "elapsed_sec": time.time() - t0,
                    "module_paths": module_paths,
                }

    finalize_shard_progress(sdir, store)
    encoder.close()
    return {
        "ok": fail == 0,
        "worker_id": worker_id,
        "gpu_id": args.gpu_id,
        "n_ok": ok,
        "n_fail": fail,
        "elapsed_sec": time.time() - t0,
        "module_paths": module_paths,
    }


def _launch_workers(
    args: argparse.Namespace,
    worker_records: list[list[SampleRecord]],
    n_gpus: int,
) -> list[dict[str, Any]]:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "worker_logs")
    ensure_dir(output_dir / "worker_inputs")
    procs: list[subprocess.Popen] = []
    result_paths: list[Path] = []

    for wid in range(n_gpus):
        rows = worker_records[wid]
        if not rows:
            continue
        rec_path = output_dir / "worker_inputs" / f"worker_{wid:02d}.jsonl"
        write_manifest_jsonl(rows, rec_path)
        result_path = output_dir / "worker_logs" / f"worker_{wid:02d}.result.json"
        result_paths.append(result_path)
        if result_path.exists():
            result_path.unlink()
        cmd = [
            sys.executable,
            "-m",
            "src.deepseek_ocr2_features.run_extract",
            "--input-dir",
            args.input_dir,
            "--output-dir",
            args.output_dir,
            "--model-path",
            args.model_path,
            "--model-revision",
            args.model_revision,
            "--page-batch-size",
            str(args.page_batch_size),
            "--max-local-batch",
            str(getattr(args, "max_local_batch", 8)),
            "--attn-implementation",
            args.attn_implementation,
            "--worker-id",
            str(wid),
            "--gpu-id",
            str(wid),
            "--records-jsonl",
            str(rec_path),
        ]
        if args.no_resume:
            cmd.append("--no-resume")
        log_path = output_dir / "worker_logs" / f"worker_{wid:02d}.log"
        log_f = open(log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(wid)
        procs.append(
            subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        )
        # Stagger model loads to reduce CPU/disk contention across 16 workers.
        time.sleep(2.0)

    print(f"[main] launched {len(procs)} workers", flush=True)
    print(
        f"[main] progress logs: {output_dir / 'worker_logs'}/worker_XX.log ; "
        f"shards: {output_dir / 'shards'}/shard_XX/progress.json",
        flush=True,
    )

    # Poll aggregate progress while workers run (main terminal otherwise stays silent).
    while True:
        alive = sum(1 for p in procs if p.poll() is None)
        total_ok = 0
        for wid in range(n_gpus):
            prog_path = shard_dir(output_dir, wid) / "progress.json"
            if prog_path.exists():
                try:
                    total_ok += int(json.loads(prog_path.read_text()).get("n_ok", 0))
                except Exception:
                    pass
        print(
            f"[main] alive_workers={alive}/{len(procs)} total_n_ok≈{total_ok}",
            flush=True,
        )
        if alive == 0:
            break
        time.sleep(30.0)

    exit_codes = [p.wait() for p in procs]
    results: list[dict[str, Any]] = []
    for rp, ec in zip(result_paths, exit_codes):
        if rp.exists():
            results.append(json.loads(rp.read_text(encoding="utf-8")))
        else:
            results.append({"ok": False, "error": f"missing result file, exit={ec}", "path": str(rp)})
    return results


def _write_worker_result(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def write_extraction_manifest(
    args: argparse.Namespace,
    *,
    n_total: int,
    n_success: int,
    n_fail: int,
    size_stats: dict[str, Any] | None,
    module_paths: dict[str, Any] | None,
    worker_results: list[dict[str, Any]],
    merge_info: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    started_utc: str,
    elapsed_sec: float,
) -> Path:
    output_dir = Path(args.output_dir)
    g_path = output_dir / "deepseek_ocr2_global_mean_fp32.npy"
    l_path = output_dir / "deepseek_ocr2_local_mean_fp32.npy"
    c_path = output_dir / "deepseek_ocr2_global_local_concat_fp32.npy"

    def _shape_dtype(p: Path) -> dict[str, Any]:
        if not p.exists():
            return {"exists": False}
        arr = np.load(p, mmap_mode="r")
        return {"exists": True, "shape": list(arr.shape), "dtype": str(arr.dtype), "path": str(p)}

    manifest = {
        "model_name": "deepseek-ai/DeepSeek-OCR-2",
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed_sec,
        "code_version": _git_rev(),
        "dataset_sample_count": n_total,
        "success_count": n_success,
        "failure_count": n_fail,
        "global_size": 1024,
        "local_size": 768,
        "mean": list(MEAN),
        "std": list(STD),
        "min_num_crops": 2,
        "max_num_crops": 6,
        "inference_dtype": "bfloat16",
        "pooling_dtype": "float32",
        "output_dtype": "float32",
        "projector_used": False,
        "language_decoder_used": False,
        "ocr_prompt_used": False,
        "text_generation_used": False,
        "l2_normalized": False,
        "feature_dim_global": GLOBAL_DIM,
        "feature_dim_local": LOCAL_DIM,
        "feature_dim_concat": CONCAT_DIM,
        "num_gpus": args.num_gpus,
        "page_batch_size": args.page_batch_size,
        "attn_implementation": args.attn_implementation,
        "module_paths": module_paths,
        "size_stats": size_stats,
        "worker_results": worker_results,
        "merge_info": merge_info,
        "audit": audit,
        "output_files": {
            "global": _shape_dtype(g_path),
            "local": _shape_dtype(l_path),
            "concat": _shape_dtype(c_path),
            "sample_index": str(output_dir / "sample_index.jsonl"),
            "errors": str(output_dir / "extraction_errors.jsonl"),
        },
    }
    path = output_dir / "extraction_manifest.json"
    atomic_write_json(path, manifest)
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Worker mode
    if args.worker_id is not None:
        result = run_worker(args)
        _write_worker_result(
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

    # Discover / load samples
    manifest_path = output_dir / "dataset_manifest.jsonl"
    if manifest_path.exists() and not args.no_resume:
        from .manifest import load_records_from_jsonl

        records = load_records_from_jsonl(manifest_path)
        print(f"[main] loaded existing manifest n={len(records)} from {manifest_path}")
        size_stats = None
    else:
        records = discover_samples(args.input_dir, recursive=False)
        size_stats = None
        if not args.skip_size_scan:
            print("[main] scanning image sizes...", flush=True)
            size_stats = enrich_image_sizes(records)
            print(
                f"[main] n={size_stats['n_total']} small_le_768={size_stats['n_small_both_le_768']} "
                f"corrupt={size_stats['n_corrupt']}",
                flush=True,
            )
            atomic_write_json(output_dir / "checkpoint0_data_stats.json", size_stats)
        write_manifest_jsonl(records, manifest_path)

    selected_raw = _select_records(records, args)
    # Dense row_index 0..N-1 for this run (required for contiguous merge).
    selected: list[SampleRecord] = []
    for i, r in enumerate(selected_raw):
        selected.append(
            SampleRecord(
                row_index=i,
                sample_id=r.sample_id,
                image_path=r.image_path,
                latex_path_or_id=r.latex_path_or_id,
                original_width=r.original_width,
                original_height=r.original_height,
                image_format=r.image_format,
                file_size=r.file_size,
                status=r.status,
                error_message=r.error_message,
            )
        )
    write_manifest_jsonl(selected, output_dir / "run_sample_manifest.jsonl")
    print(
        f"[main] selected {len(selected)} / {len(records)} "
        f"start={args.start_index} end={args.end_index} limit={args.limit}",
        flush=True,
    )

    if args.merge_only:
        merge_info = merge_shards(output_dir, n_shards=int(args.num_gpus), n_expected=len(selected))
        audit = audit_outputs(output_dir, n_expected=len(selected))
        write_extraction_manifest(
            args,
            n_total=len(selected),
            n_success=audit.get("n_success", 0),
            n_fail=0,
            size_stats=size_stats,
            module_paths=None,
            worker_results=[],
            merge_info=merge_info,
            audit=audit,
            started_utc=started,
            elapsed_sec=time.time() - t0,
        )
        print(json.dumps({"merge": merge_info, "audit": audit}, indent=2))
        return 0 if audit.get("passed") else 1

    n_gpus = max(1, min(int(args.num_gpus), max(1, torch.cuda.device_count())))
    print(f"[main] using {n_gpus} GPUs, page_batch_size={args.page_batch_size}", flush=True)
    buckets = _split_for_workers(selected, n_gpus)

    # If only one GPU requested or workers launched inline for tiny runs
    if n_gpus == 1:
        rec_path = output_dir / "worker_inputs" / "worker_00.jsonl"
        ensure_dir(rec_path.parent)
        write_manifest_jsonl(buckets[0], rec_path)
        wargs = argparse.Namespace(**vars(args))
        wargs.worker_id = 0
        wargs.gpu_id = 0
        wargs.records_jsonl = str(rec_path)
        ensure_dir(output_dir / "worker_logs")
        result = run_worker(wargs)
        _write_worker_result(output_dir / "worker_logs" / "worker_00.result.json", result)
        worker_results = [result]
    else:
        worker_results = _launch_workers(args, buckets, n_gpus)

    if not all(r.get("ok") for r in worker_results):
        print("[main] FATAL: one or more workers failed; refusing final merge", flush=True)
        atomic_write_json(output_dir / "worker_results.json", worker_results)
        write_extraction_manifest(
            args,
            n_total=len(selected),
            n_success=sum(int(r.get("n_ok", 0)) for r in worker_results),
            n_fail=sum(int(r.get("n_fail", 0)) for r in worker_results),
            size_stats=size_stats,
            module_paths=worker_results[0].get("module_paths") if worker_results else None,
            worker_results=worker_results,
            merge_info=None,
            audit=None,
            started_utc=started,
            elapsed_sec=time.time() - t0,
        )
        return 1

    merge_info = merge_shards(output_dir, n_shards=n_gpus, n_expected=len(selected))
    audit = audit_outputs(output_dir, n_expected=len(selected))
    write_extraction_manifest(
        args,
        n_total=len(selected),
        n_success=audit.get("n_success", 0),
        n_fail=sum(int(r.get("n_fail", 0)) for r in worker_results),
        size_stats=size_stats,
        module_paths=worker_results[0].get("module_paths") if worker_results else None,
        worker_results=worker_results,
        merge_info=merge_info,
        audit=audit,
        started_utc=started,
        elapsed_sec=time.time() - t0,
    )
    print(json.dumps({"merge": merge_info, "audit_passed": audit.get("passed"), "audit": audit}, indent=2))
    return 0 if audit.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
