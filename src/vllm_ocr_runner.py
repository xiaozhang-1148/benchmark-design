"""High-throughput DeepSeek-OCR-2 generation via vLLM (channel A)."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import fingerprint_config, load_config, model_resolved_path
from .feature_store import atomic_replace_parquet, merge_status_parquet
from .utils import atomic_write_json, atomic_write_text, ensure_dir, load_image_rgb


def _detect_tp(cfg_tp: int | None) -> int:
    if cfg_tp is not None:
        return int(cfg_tp)
    try:
        import torch

        n = torch.cuda.device_count()
        return max(1, min(n, 1))  # default single-GPU for A10 memory headroom
    except Exception:
        return 1


def _load_tuned(out_dir: Path) -> dict[str, Any]:
    p = out_dir.parent / "reports" / "benchmark.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def build_llm(cfg: dict[str, Any]):
    from vllm import LLM

    vcfg = cfg["vllm"]
    tuned = _load_tuned(Path(cfg["paths"]["outputs_dir"]))
    best = tuned.get("best_config") or {}

    gpu_util = best.get("gpu_memory_utilization", vcfg.get("gpu_memory_utilization", 0.9))
    max_num_seqs = best.get("max_num_seqs", vcfg.get("max_num_seqs") or 8)
    tp = _detect_tp(vcfg.get("tensor_parallel_size"))

    kwargs: dict[str, Any] = {
        "model": model_resolved_path(cfg),
        "dtype": vcfg.get("dtype", "bfloat16"),
        "trust_remote_code": bool(vcfg.get("trust_remote_code", True)),
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": float(gpu_util),
        "max_model_len": int(vcfg.get("max_model_len", 8192)),
        "max_num_seqs": int(max_num_seqs),
        "enforce_eager": bool(vcfg.get("enforce_eager", False)),
        "hf_overrides": vcfg.get("hf_overrides") or {"architectures": ["DeepseekOCR2ForCausalLM"]},
        "enable_prefix_caching": bool(vcfg.get("enable_prefix_caching", False)),
    }
    if vcfg.get("quantization"):
        kwargs["quantization"] = vcfg["quantization"]

    print(f"[vllm_ocr] LLM kwargs: {json.dumps({k: v for k, v in kwargs.items() if k != 'hf_overrides'})}")
    return LLM(**kwargs), kwargs


def build_sampling_params(cfg: dict[str, Any]):
    from vllm import SamplingParams

    vcfg = cfg["vllm"]
    # Prefer simple SamplingParams; ngram processor via extra_args if supported
    kwargs: dict[str, Any] = {
        "temperature": float(vcfg.get("temperature", 0.0)),
        "top_p": float(vcfg.get("top_p", 1.0)),
        "max_tokens": int(vcfg.get("max_tokens", 4096)),
        "skip_special_tokens": False,
        "logprobs": 1,  # attempt; may be ignored / expensive but once only
    }
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("logprobs", None)
        return SamplingParams(**kwargs)


def _mean_logprob_from_output(output) -> float | None:
    try:
        lps = output.outputs[0].logprobs
        if not lps:
            return None
        vals = []
        for step in lps:
            if step is None:
                continue
            # step: dict[token_id, Logprob]
            if isinstance(step, dict) and step:
                # chosen token usually first / matching output token
                lp = next(iter(step.values()))
                vals.append(float(getattr(lp, "logprob", lp)))
        if not vals:
            return None
        return float(sum(vals) / len(vals))
    except Exception:
        return None


def _prefetch_images(rows: list[dict[str, Any]], workers: int) -> dict[str, Image.Image | Exception]:
    out: dict[str, Image.Image | Exception] = {}

    def _one(r):
        try:
            return r["image_id"], load_image_rgb(r["absolute_path"])
        except Exception as e:  # noqa: BLE001
            return r["image_id"], e

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for image_id, img in ex.map(_one, rows):
            out[image_id] = img
    return out


def run_vllm_ocr(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    raw_dir = out_dir / "recognition_raw"
    ensure_dir(raw_dir)
    manifest_path = out_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError("manifest.parquet missing; run build_manifest first")

    man = pd.read_parquet(manifest_path)
    man = man[man["status"] != "corrupt"].copy()
    fp = fingerprint_config(cfg)
    ocr_path = out_dir / "ocr_generations.parquet"

    done = set()
    if ocr_path.exists() and cfg.get("pipeline", {}).get("resume", True):
        prev = pd.read_parquet(ocr_path)
        if "config_fingerprint" in prev.columns:
            prev_ok = prev[(prev["status"] == "ok") & (prev["config_fingerprint"] == fp)]
            done = set(prev_ok["image_id"].astype(str))

    pending = man[~man["image_id"].astype(str).isin(done)].to_dict("records")
    limit = cfg.get("pipeline", {}).get("limit")
    if limit is not None:
        # honor limit on total successful+pending target
        pending = pending[: max(0, int(limit) - len(done))]

    print(f"[vllm_ocr] pending={len(pending)} done={len(done)} fingerprint={fp}")
    if not pending:
        return pd.read_parquet(ocr_path) if ocr_path.exists() else pd.DataFrame()

    llm, llm_kwargs = build_llm(cfg)
    sampling = build_sampling_params(cfg)
    prompt = cfg["prompt"]
    prefetch = int(cfg["vllm"].get("prefetch_workers", 8))
    retry_max = int(cfg["vllm"].get("retry_max", 3))
    # Process in chunks to bound memory of decoded images
    chunk_size = int(llm_kwargs.get("max_num_seqs", 8)) * 4

    all_new_rows: list[dict[str, Any]] = []
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        images = _prefetch_images(chunk, prefetch)
        requests = []
        meta = []
        for r in chunk:
            iid = str(r["image_id"])
            img = images.get(iid)
            if isinstance(img, Exception) or img is None:
                all_new_rows.append(
                    {
                        "image_id": iid,
                        "status": "failed",
                        "error_message": f"image_load: {img}",
                        "text": "",
                        "output_token_count": 0,
                        "mean_generated_token_logprob": None,
                        "latency_sec": None,
                        "config_fingerprint": fp,
                    }
                )
                continue
            requests.append({"prompt": prompt, "multi_modal_data": {"image": img}})
            meta.append(r)

        if not requests:
            continue

        attempt = 0
        outputs = None
        last_err = None
        while attempt < retry_max:
            attempt += 1
            try:
                t0 = time.perf_counter()
                outputs = llm.generate(requests, sampling_params=sampling)
                batch_dt = time.perf_counter() - t0
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(float(cfg["vllm"].get("retry_backoff_sec", 2.0)) * attempt)
        if outputs is None:
            for r in meta:
                all_new_rows.append(
                    {
                        "image_id": str(r["image_id"]),
                        "status": "failed",
                        "error_message": f"vllm_generate: {type(last_err).__name__}: {last_err}",
                        "text": "",
                        "output_token_count": 0,
                        "mean_generated_token_logprob": None,
                        "latency_sec": None,
                        "config_fingerprint": fp,
                    }
                )
            continue

        per = batch_dt / max(len(outputs), 1)
        for r, out in zip(meta, outputs):
            iid = str(r["image_id"])
            try:
                text = out.outputs[0].text if out.outputs else ""
                tok_ids = out.outputs[0].token_ids if out.outputs else []
                mean_lp = _mean_logprob_from_output(out)
                atomic_write_text(raw_dir / f"{iid}.txt", text)
                all_new_rows.append(
                    {
                        "image_id": iid,
                        "status": "ok",
                        "error_message": None,
                        "text": text[:2000],  # short preview in parquet
                        "output_token_count": len(tok_ids) if tok_ids is not None else None,
                        "mean_generated_token_logprob": mean_lp,
                        "latency_sec": per,
                        "config_fingerprint": fp,
                    }
                )
            except Exception as e:  # noqa: BLE001
                all_new_rows.append(
                    {
                        "image_id": iid,
                        "status": "failed",
                        "error_message": f"{type(e).__name__}: {e}",
                        "text": "",
                        "output_token_count": 0,
                        "mean_generated_token_logprob": None,
                        "latency_sec": per,
                        "config_fingerprint": fp,
                    }
                )

        # checkpoint after each chunk
        new_df = pd.DataFrame(all_new_rows)
        merge_status_parquet(ocr_path, new_df, key="image_id")
        print(f"[vllm_ocr] checkpointed {len(all_new_rows)} new rows (chunk {start // chunk_size + 1})")

    # free engine
    del llm
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return pd.read_parquet(ocr_path)


def run_benchmark(cfg: dict[str, Any]) -> dict[str, Any]:
    """Small auto-benchmark over max_num_seqs / gpu_memory_utilization."""
    reports = Path(cfg["paths"]["reports_dir"])
    ensure_dir(reports)
    out_json = reports / "benchmark.json"
    out_md = reports / "benchmark.md"
    bcfg = cfg.get("benchmark", {})
    if bcfg.get("skip_if_exists") and out_json.exists():
        return json.loads(out_json.read_text())

    man_path = Path(cfg["paths"]["outputs_dir"]) / "manifest.parquet"
    if not man_path.exists():
        raise FileNotFoundError("Need manifest before benchmark")
    man = pd.read_parquet(man_path)
    man = man[man["status"] != "corrupt"].head(int(bcfg.get("sample_size", 100)))
    rows = man.to_dict("records")
    if len(rows) < 4:
        result = {
            "best_config": {
                "max_num_seqs": cfg["vllm"].get("max_num_seqs") or 4,
                "gpu_memory_utilization": cfg["vllm"].get("gpu_memory_utilization", 0.9),
                "prefetch_workers": cfg["vllm"].get("prefetch_workers", 8),
            },
            "trials": [],
            "note": "insufficient samples; using defaults",
        }
        atomic_write_json(out_json, result)
        atomic_write_text(out_md, "# Benchmark\n\nInsufficient samples; defaults used.\n")
        return result

    from vllm import LLM, SamplingParams
    import torch

    seqs_cands = bcfg.get("max_num_seqs_candidates") or [4, 8, 16]
    mem_cands = bcfg.get("gpu_memory_utilization_candidates") or [0.85, 0.90]
    pref_cands = bcfg.get("prefetch_workers_candidates") or [8]
    prompt = cfg["prompt"]
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=min(1024, int(cfg["vllm"].get("max_tokens", 4096))),
        skip_special_tokens=False,
    )

    trials = []
    best = None
    # Use a small fixed subset for each trial
    sample = rows[: min(16, len(rows))]
    for max_num_seqs in seqs_cands:
        for gpu_util in mem_cands:
            for prefetch in pref_cands:
                trial = {
                    "max_num_seqs": max_num_seqs,
                    "gpu_memory_utilization": gpu_util,
                    "prefetch_workers": prefetch,
                }
                llm = None
                try:
                    torch.cuda.reset_peak_memory_stats()
                    llm = LLM(
                        model=model_resolved_path(cfg),
                        dtype=cfg["vllm"].get("dtype", "bfloat16"),
                        trust_remote_code=True,
                        tensor_parallel_size=_detect_tp(cfg["vllm"].get("tensor_parallel_size")),
                        gpu_memory_utilization=float(gpu_util),
                        max_model_len=int(cfg["vllm"].get("max_model_len", 8192)),
                        max_num_seqs=int(max_num_seqs),
                        enforce_eager=bool(cfg["vllm"].get("enforce_eager", False)),
                        hf_overrides=cfg["vllm"].get("hf_overrides")
                        or {"architectures": ["DeepseekOCR2ForCausalLM"]},
                        enable_prefix_caching=False,
                    )
                    imgs = _prefetch_images(sample, prefetch)
                    reqs = []
                    for r in sample:
                        img = imgs[str(r["image_id"])]
                        if isinstance(img, Exception):
                            continue
                        reqs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
                    if not reqs:
                        raise RuntimeError("no valid images in benchmark sample")
                    t0 = time.perf_counter()
                    outs = llm.generate(reqs, sampling_params=sampling)
                    dt = time.perf_counter() - t0
                    n_tok = 0
                    fails = 0
                    latencies = []
                    for o in outs:
                        try:
                            n_tok += len(o.outputs[0].token_ids or [])
                            latencies.append(dt / len(outs))
                        except Exception:
                            fails += 1
                    peak = torch.cuda.max_memory_allocated() / (1024**3)
                    trial.update(
                        {
                            "images_per_sec": len(outs) / max(dt, 1e-6),
                            "tokens_per_sec": n_tok / max(dt, 1e-6),
                            "mean_latency_sec": float(sum(latencies) / max(len(latencies), 1)),
                            "p50_latency_sec": float(sorted(latencies)[len(latencies) // 2]) if latencies else None,
                            "p95_latency_sec": float(sorted(latencies)[int(0.95 * (len(latencies) - 1))])
                            if latencies
                            else None,
                            "peak_gpu_mem_gb": float(peak),
                            "failure_rate": fails / max(len(outs), 1),
                            "n_images": len(outs),
                            "ok": True,
                            "error": None,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    trial.update(
                        {
                            "ok": False,
                            "error": f"{type(e).__name__}: {e}",
                            "images_per_sec": 0.0,
                            "failure_rate": 1.0,
                        }
                    )
                finally:
                    if llm is not None:
                        del llm
                    torch.cuda.empty_cache()
                trials.append(trial)
                print(f"[benchmark] {trial}")
                if trial.get("ok") and (best is None or trial["images_per_sec"] > best["images_per_sec"]):
                    best = trial

    if best is None:
        best = {
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.85,
            "prefetch_workers": 8,
            "images_per_sec": 0.0,
        }

    result = {"best_config": best, "trials": trials, "sample_size_requested": bcfg.get("sample_size", 100)}
    atomic_write_json(out_json, result)
    lines = [
        "# vLLM Throughput Benchmark",
        "",
        f"Best config: `{json.dumps(best, ensure_ascii=False)}`",
        "",
        "| max_num_seqs | gpu_mem | prefetch | img/s | tok/s | peakGB | fail | ok |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for t in trials:
        lines.append(
            f"| {t.get('max_num_seqs')} | {t.get('gpu_memory_utilization')} | {t.get('prefetch_workers')} | "
            f"{t.get('images_per_sec', 0):.3f} | {t.get('tokens_per_sec', 0):.1f} | "
            f"{t.get('peak_gpu_mem_gb', 0):.2f} | {t.get('failure_rate', 1):.2f} | {t.get('ok')} |"
        )
    # also project reports
    proj = Path(cfg["paths"].get("project_reports_dir", "reports"))
    ensure_dir(proj)
    atomic_write_text(out_md, "\n".join(lines) + "\n")
    atomic_write_text(proj / "benchmark.md", "\n".join(lines) + "\n")
    atomic_write_json(proj / "benchmark.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.benchmark_only or (cfg.get("benchmark", {}).get("enabled") and not (
        Path(cfg["paths"]["reports_dir"]) / "benchmark.json"
    ).exists()):
        run_benchmark(cfg)
        if args.benchmark_only:
            return 0
    df = run_vllm_ocr(cfg)
    ok = int((df["status"] == "ok").sum()) if len(df) else 0
    print(f"[vllm_ocr] total={len(df)} ok={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
