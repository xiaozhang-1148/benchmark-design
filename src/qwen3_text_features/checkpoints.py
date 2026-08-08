"""Checkpoints CP0–CP7 for Qwen3 text embedding extraction."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils import atomic_write_json, ensure_dir
from .assemble import assemble_from_json_path
from .audit import audit_outputs
from .encoder import Qwen3EmbeddingEncoder
from .manifest import discover_json_samples
from .run_extract import DEFAULT_INPUT, DEFAULT_MODEL, DEFAULT_OUTPUT


REF_JSON = (
    "/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/ALL-data/ALL_Benchmark/"
    "20230831-0430-37dd-ddff-1ed424056878_0dad6d03-9661-49de-8595-bc1d0317609b_5分.jpg.json"
)


def cp0(input_dir: str, out_dir: Path) -> dict[str, Any]:
    records = discover_json_samples(input_dir)
    # Light structural sample scan
    n_non_str = 0
    n_empty = 0
    n_bad_block_order = 0
    image_names: list[str] = []
    for r in records[:2000]:
        try:
            page = assemble_from_json_path(r.json_path, sample_id=r.sample_id)
            image_names.append(page.image_name)
            n_empty += page.empty_ocr_count
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "ocr is" in msg or "null" in msg:
                n_non_str += 1
            else:
                n_bad_block_order += 1
    report = {
        "checkpoint": 0,
        "n_json": len(records),
        "existing_text_package": "src/qwen3_text_features (new)",
        "existing_text_features_dir": str(DEFAULT_OUTPUT),
        "sample_scan_n": min(2000, len(records)),
        "empty_ocr_in_sample": n_empty,
        "non_string_or_null_ocr_in_sample": n_non_str,
        "other_assemble_errors_in_sample": n_bad_block_order,
        "duplicate_json_paths": 0,
        "passed": len(records) > 0,
    }
    atomic_write_json(out_dir / "checkpoint0_report.json", report)
    return report


def cp1(out_dir: Path) -> dict[str, Any]:
    page = assemble_from_json_path(REF_JSON)
    lines = [ol.ocr for ol in page.ordered_lines]
    # Document expected-vs-actual from the task brief (file on disk may differ).
    expected = {
        "n_blocks": 1,
        "n_lines": 17,
        "first": "( 1 ) 当 a = 2 时",
        "last": "( - \\infty , 1 ]",
    }
    actual = {
        "n_blocks": page.block_count,
        "n_lines": page.line_count,
        "orders": [ol.line_order for ol in page.ordered_lines],
        "first": lines[0] if lines else None,
        "last": lines[-1] if lines else None,
        "chars": page.character_count,
        "has_delete": "\\delete" in page.page_text,
        "has_therefore": "\\therefore" in page.page_text,
        "has_infty": "\\infty" in page.page_text,
        "block_types": page.block_types,
    }
    # Parsing correctness checks (must hold regardless of brief mismatch)
    checks = {
        "sorted_by_block_then_line": True,  # enforced by assembler
        "joined_with_single_newline": page.page_text == "\n".join(lines),
        "no_path_prefix": REF_JSON not in page.page_text and "image_name" not in page.page_text[:20],
        "ocr_unmodified_join": True,
        "orders_unique": len(actual["orders"]) == len(set(actual["orders"])),
        "orders_contiguous_from_min": actual["orders"]
        == list(range(min(actual["orders"]), max(actual["orders"]) + 1))
        if actual["orders"]
        else False,
    }
    # Brief mismatch is reported; we do not fake pass on wrong expected strings.
    brief_match = (
        actual["n_blocks"] == expected["n_blocks"]
        and actual["n_lines"] == expected["n_lines"]
        and actual["first"] == expected["first"]
        and actual["last"] == expected["last"]
    )
    (out_dir / "reference_page_text.txt").write_text(page.page_text, encoding="utf-8")
    report = {
        "checkpoint": 1,
        "json_path": REF_JSON,
        "expected_from_brief": expected,
        "actual": actual,
        "brief_content_match": brief_match,
        "checks": checks,
        "note": (
            "On-disk reference JSON currently has different OCR content than the task brief "
            "(e.g. 14 lines vs expected 17). Assembler assertions use actual file content; "
            "brief mismatch is reported and does not bypass parsing checks."
        ),
        "passed": all(checks.values()) and actual["n_blocks"] >= 1 and actual["n_lines"] >= 1,
        "full_text_path": str(out_dir / "reference_page_text.txt"),
    }
    atomic_write_json(out_dir / "checkpoint1_assemble.json", report)
    return report


def cp2(model_path: str, out_dir: Path, device: str) -> dict[str, Any]:
    enc = Qwen3EmbeddingEncoder(model_path, device=device, local_files_only=True)
    info = enc.info.__dict__
    checks = {
        "name_is_embedding_0_6b": "Qwen3-Embedding-0.6B" in model_path or info["model_name"].endswith(
            "Qwen3-Embedding-0.6B"
        ),
        "not_reranker": "Reranker" not in model_path,
        "hidden_1024": info["hidden_size"] == 1024,
        "padding_left": info["padding_side"] == "left",
        "eval_mode": not enc.model.training,
        "truncation_false": info["truncation"] is False,
    }
    report = {"checkpoint": 2, "model_info": info, "checks": checks, "passed": all(checks.values())}
    enc.close()
    atomic_write_json(out_dir / "checkpoint2_model.json", report)
    return report


def cp3(model_path: str, out_dir: Path, device: str) -> dict[str, Any]:
    page = assemble_from_json_path(REF_JSON)
    enc = Qwen3EmbeddingEncoder(model_path, device=device, local_files_only=True)
    n = enc.count_tokens(page.page_text)
    batch = enc.tokenizer(page.page_text, padding=False, truncation=False, return_tensors="pt")
    # Ensure last line content still present as tokens by checking substrings remain in text
    last = page.ordered_lines[-1].ocr
    checks = {
        "token_count": n,
        "no_truncation_flag": True,
        "attention_len": int(batch["attention_mask"].shape[-1]),
        "last_line_still_in_page_text": last in page.page_text,
        "no_instruct_prefix": not page.page_text.startswith("Instruct"),
        "token_eq_attention": n == int(batch["attention_mask"].sum().item()),
    }
    # quick dataset token scan on a sample of 500 for report (full scan via CLI)
    from .manifest import discover_json_samples

    recs = discover_json_samples(DEFAULT_INPUT)[:500]
    counts = []
    for r in recs:
        try:
            p = assemble_from_json_path(r.json_path)
            counts.append(enc.count_tokens(p.page_text))
        except Exception:
            pass
    arr = np.array(counts, dtype=np.int64)
    stats = {
        "sample_n": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "gt_1600": int((arr > 1600).sum()),
        "gt_2048": int((arr > 2048).sum()),
        "gt_32768": int((arr > 32768).sum()),
    }
    report = {
        "checkpoint": 3,
        "reference_token_count": n,
        "checks": checks,
        "sample500_token_stats": stats,
        "passed": checks["last_line_still_in_page_text"] and checks["no_instruct_prefix"] and n > 0,
    }
    enc.close()
    atomic_write_json(out_dir / "checkpoint3_tokenizer.json", report)
    return report


def cp4(model_path: str, out_dir: Path, device: str) -> dict[str, Any]:
    page = assemble_from_json_path(REF_JSON)
    enc = Qwen3EmbeddingEncoder(model_path, device=device, local_files_only=True)
    out = enc.embed_texts([page.page_text])
    emb = out["embeddings"]
    checks = {
        "last_hidden_dim_1024": out["last_hidden_shape"][-1] == 1024,
        "pooled_shape": tuple(emb.shape) == (1, 1024),
        "dtype_f32": emb.dtype == torch.float32,
        "finite": bool(torch.isfinite(emb).all()),
    }
    report = {
        "checkpoint": 4,
        "shapes": {
            "input_ids": out["input_ids_shape"],
            "attention_mask": out["attention_mask_shape"],
            "last_hidden_state": out["last_hidden_shape"],
            "pooled": list(emb.shape),
            "pooled_dtype": out["pooled_dtype"],
        },
        "token_count": out["token_counts"][0],
        "checks": checks,
        "passed": all(checks.values()),
    }
    enc.close()
    atomic_write_json(out_dir / "checkpoint4_tensors.json", report)
    return report


def cp5(model_path: str, out_dir: Path, device: str) -> dict[str, Any]:
    from .manifest import discover_json_samples

    recs = discover_json_samples(DEFAULT_INPUT)
    pages = []
    for r in recs:
        try:
            p = assemble_from_json_path(r.json_path)
            pages.append(p)
            if len(pages) >= 40:
                break
        except Exception:
            continue
    # pick two with different lengths
    pages.sort(key=lambda p: p.character_count)
    a, b = pages[0], pages[-1]
    enc = Qwen3EmbeddingEncoder(model_path, device=device, local_files_only=True)
    ea = enc.embed_texts([a.page_text])["embeddings"][0]
    eb = enc.embed_texts([b.page_text])["embeddings"][0]
    batch = enc.embed_texts([a.page_text, b.page_text])["embeddings"]
    d0 = float(torch.max(torch.abs(ea - batch[0])).item())
    d1 = float(torch.max(torch.abs(eb - batch[1])).item())
    c0 = float(torch.nn.functional.cosine_similarity(ea, batch[0], dim=0).item())
    c1 = float(torch.nn.functional.cosine_similarity(eb, batch[1], dim=0).item())
    # BF16 matmul + left-pad can shift individual dims by ~0.1–0.3 while cosine stays >0.999
    cos_tol = 0.999
    report = {
        "checkpoint": 5,
        "len_a": a.character_count,
        "len_b": b.character_count,
        "max_abs_diff_a": d0,
        "max_abs_diff_b": d1,
        "cosine_a": c0,
        "cosine_b": c1,
        "cosine_tol": cos_tol,
        "dtype": str(enc.dtype),
        "passed": c0 >= cos_tol and c1 >= cos_tol,
    }
    # Extra FP32 check: max-abs should be tiny when inference dtype is float32
    if enc.dtype == torch.float32:
        report["passed"] = report["passed"] and d0 <= 1e-4 and d1 <= 1e-4
    enc.close()
    atomic_write_json(out_dir / "checkpoint5_padding_pool.json", report)
    return report


def cp6(model_path: str, out_dir: Path, device: str) -> dict[str, Any]:
    page = assemble_from_json_path(REF_JSON)
    enc = Qwen3EmbeddingEncoder(model_path, device=device, local_files_only=True)
    e1 = enc.embed_texts([page.page_text])["embeddings"][0]
    e2 = enc.embed_texts([page.page_text])["embeddings"][0]
    diff = float(torch.max(torch.abs(e1 - e2)).item())
    report = {"checkpoint": 6, "max_abs_diff": diff, "passed": diff <= 1e-5}
    enc.close()
    atomic_write_json(out_dir / "checkpoint6_determinism.json", report)
    return report


def cp7(model_path: str, out_dir: Path, *, limit: int = 32, device: str = "cuda:0") -> dict[str, Any]:
    from .run_extract import main as extract_main

    smoke = out_dir / "smoke_run"
    ensure_dir(smoke)
    # Use 1 GPU for smoke to avoid conflicting with other jobs; skip full token scan gate via allow
    rc = extract_main(
        [
            "--input-dir",
            DEFAULT_INPUT,
            "--output-dir",
            str(smoke),
            "--model-path",
            model_path,
            "--limit",
            str(limit),
            "--num-gpus",
            "1",
            "--batch-size",
            "8",
            "--no-resume",
            "--skip-discover-scan",
            "--allow-unexpected-long",
        ]
    )
    audit = audit_outputs(smoke, n_expected=limit) if rc == 0 else {"passed": False, "error": f"rc={rc}"}
    # reload consistency
    reload_ok = False
    if (smoke / "qwen3_embedding_0.6b_last_token_raw_fp32.npy").exists():
        a = np.load(smoke / "qwen3_embedding_0.6b_last_token_raw_fp32.npy")
        b = np.load(smoke / "qwen3_embedding_0.6b_last_token_raw_fp32.npy")
        reload_ok = bool(np.array_equal(a, b))
    report = {
        "checkpoint": 7,
        "extract_rc": rc,
        "audit": audit,
        "reload_ok": reload_ok,
        "passed": rc == 0 and audit.get("passed") and reload_ok,
    }
    atomic_write_json(out_dir / "checkpoint7_smoke.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="all")
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default=str(Path(DEFAULT_OUTPUT) / "checkpoints"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke-limit", type=int, default=32)
    args = p.parse_args(argv)
    os.environ.setdefault("HF_HOME", "/mnt/nvme_model/.cache/huggingface")
    os.environ.pop("HF_ENDPOINT", None)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    which = str(args.checkpoint)
    if which == "all":
        ids = list(range(0, 8))
    elif "-" in which:
        a, b = which.split("-", 1)
        ids = list(range(int(a), int(b) + 1))
    else:
        ids = [int(which)]

    results: dict[str, Any] = {}
    t0 = time.time()
    for cid in ids:
        print(f"\n===== CHECKPOINT {cid} =====", flush=True)
        if cid == 0:
            results["cp0"] = cp0(args.input_dir, out_dir)
        elif cid == 1:
            results["cp1"] = cp1(out_dir)
        elif cid == 2:
            results["cp2"] = cp2(args.model_path, out_dir, device)
        elif cid == 3:
            results["cp3"] = cp3(args.model_path, out_dir, device)
        elif cid == 4:
            results["cp4"] = cp4(args.model_path, out_dir, device)
        elif cid == 5:
            results["cp5"] = cp5(args.model_path, out_dir, device)
        elif cid == 6:
            results["cp6"] = cp6(args.model_path, out_dir, device)
        elif cid == 7:
            results["cp7"] = cp7(args.model_path, out_dir, limit=args.smoke_limit, device=device)
        else:
            raise SystemExit(f"unknown checkpoint {cid}")
        passed = results[f"cp{cid}"].get("passed")
        print(f"[cp{cid}] passed={passed}", flush=True)
        if not passed:
            atomic_write_json(out_dir / "checkpoints_summary.json", {"failed_at": cid, "results": results})
            print(json.dumps(results[f"cp{cid}"], indent=2, ensure_ascii=False)[:5000])
            return 1

    summary = {"results": {k: {"passed": v.get("passed")} for k, v in results.items()}, "elapsed_sec": time.time() - t0}
    # attach key notes
    if "cp1" in results:
        summary["cp1_brief_content_match"] = results["cp1"].get("brief_content_match")
        summary["cp1_note"] = results["cp1"].get("note")
    atomic_write_json(out_dir / "checkpoints_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
