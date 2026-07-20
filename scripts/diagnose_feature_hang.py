#!/usr/bin/env python3
"""Locate expression-feature extraction hangs (chunk/expression bisection with timeout)."""

from __future__ import annotations

import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from benchmark_design.io.dataset_loaders import load_dataset
from benchmark_design.ocr.expression_features import build_corpus_feature_context, extract_single_features
from benchmark_design.ocr.processing import _chunk_records
from benchmark_design.ocr.tokenizer import build_latex_vocab, tokenize_greedy


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    def _handler(_signum, _frame):
        raise TimeoutError(f"timeout after {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main() -> int:
    input_dir = Path("/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark")
    chunk_timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    expr_timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    print("Loading corpus...", flush=True)
    t0 = time.perf_counter()
    exprs = load_dataset("ours", input_dir, show_progress=False, workers=8)
    vocab = build_latex_vocab()
    tokens = [tuple(tokenize_greedy(record.ocr, vocab)) for record in exprs]
    chunks = _chunk_records(exprs, workers=128)
    print(f"Loaded {len(exprs)} expressions in {len(chunks)} chunks ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Building duplicate/rare context...", flush=True)
    t0 = time.perf_counter()
    duplicate_index, rare_sets = build_corpus_feature_context(exprs, tokens)
    print(f"Context built in {time.perf_counter() - t0:.1f}s", flush=True)

    slow_chunks: list[tuple[float, int, int]] = []
    hung_chunks: list[int] = []
    offset = 0
    for chunk_idx, records in enumerate(chunks):
        chunk_tokens = tokens[offset : offset + len(records)]
        try:
            with time_limit(chunk_timeout):
                start = time.perf_counter()
                for record_index, (record, token_seq) in enumerate(zip(records, chunk_tokens, strict=True)):
                    global_index = offset + record_index
                    extract_single_features(
                        record,
                        token_seq,
                        duplicate_group_id=duplicate_index.group_id_by_index[global_index],
                        duplicate_count=duplicate_index.group_size_by_index[global_index],
                        rare_sets=rare_sets,
                    )
                elapsed = time.perf_counter() - start
        except TimeoutError:
            hung_chunks.append(chunk_idx)
            print(f"CHUNK TIMEOUT chunk={chunk_idx} offset={offset} size={len(records)}", flush=True)
            offset += len(records)
            continue
        if elapsed > 0.5:
            slow_chunks.append((elapsed, chunk_idx, len(records)))
        offset += len(records)

    slow_chunks.sort(reverse=True)
    print(f"\nSlow chunks (>0.5s): {len(slow_chunks)}", flush=True)
    for elapsed, chunk_idx, size in slow_chunks[:10]:
        print(f"  chunk {chunk_idx}: {elapsed:.2f}s ({size} expr)", flush=True)
    print(f"Hung chunks (>{chunk_timeout}s): {hung_chunks}", flush=True)

    for chunk_idx in hung_chunks:
        records = chunks[chunk_idx]
        start_offset = sum(len(chunks[i]) for i in range(chunk_idx))
        print(f"\nBisecting chunk {chunk_idx} (size={len(records)})...", flush=True)
        for local_index, (record, token_seq) in enumerate(zip(records, tokens[start_offset : start_offset + len(records)], strict=True)):
            global_index = start_offset + local_index
            try:
                with time_limit(expr_timeout):
                    t1 = time.perf_counter()
                    extract_single_features(
                        record,
                        token_seq,
                        duplicate_group_id=duplicate_index.group_id_by_index[global_index],
                        duplicate_count=duplicate_index.group_size_by_index[global_index],
                        rare_sets=rare_sets,
                    )
                    dt = time.perf_counter() - t1
            except TimeoutError:
                print("HANG expression:", flush=True)
                print(f"  global_index={global_index}", flush=True)
                print(f"  expression_id={record.expression_id}", flush=True)
                print(f"  token_length={len(token_seq)}", flush=True)
                print(f"  ocr={record.ocr[:500]!r}", flush=True)
                return 1
            if dt > 0.1:
                print(
                    f"  slow expr {dt*1000:.0f}ms id={record.expression_id} len={len(token_seq)}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
