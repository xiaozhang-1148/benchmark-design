"""Parallel GPU-accelerated per-image extraction worker."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from heatmap_analysis.config import AnalysisConfig, load_config
from heatmap_analysis.gpu import GpuContext, init_worker_gpu, to_numpy
from heatmap_analysis.handwriting import extract_ink_mask
from heatmap_analysis.heatmap import build_heatmaps
from heatmap_analysis.io import (
    ImageRecord,
    cache_path_for_image,
    load_template,
    read_grayscale,
    validate_image,
)
from heatmap_analysis.metrics import compute_metrics

logger = logging.getLogger("heatmap_analysis.extract_worker")

# Process-local GPU context set by pool initializer
_WORKER_GPU: GpuContext | None = None
_WORKER_CFG: AnalysisConfig | None = None


def _init_pool_worker(device_id: int, cfg_path: str, gpu_enabled: bool) -> None:
    global _WORKER_GPU, _WORKER_CFG
    _WORKER_GPU = init_worker_gpu(device_id, enabled=gpu_enabled)
    _WORKER_CFG = load_config(Path(cfg_path))


@dataclass
class ExtractJob:
    image_id: str
    image_path: str
    rel_path: str
    template_id: str | None
    metadata: dict[str, Any]
    skip_if_cached: bool


@dataclass
class ExtractJobResult:
    image_id: str
    success: bool
    skipped: bool = False
    metrics_row: dict[str, Any] | None = None
    issues: list[dict[str, str]] | None = None
    error: str | None = None
    trace: str | None = None


def _record_from_job(job: ExtractJob) -> ImageRecord:
    return ImageRecord(
        image_id=job.image_id,
        image_path=Path(job.image_path),
        rel_path=job.rel_path,
        template_id=job.template_id,
        metadata=job.metadata,
    )


def extract_jobs_on_gpu(gpu_id: int, jobs: list[ExtractJob], cfg_path: str, gpu_enabled: bool) -> list[ExtractJobResult]:
    """Process a chunk of jobs on one GPU with threaded image prefetch."""
    from concurrent.futures import ThreadPoolExecutor

    global _WORKER_GPU, _WORKER_CFG
    _WORKER_GPU = init_worker_gpu(gpu_id, enabled=gpu_enabled)
    _WORKER_CFG = load_config(Path(cfg_path))

    results: list[ExtractJobResult] = []
    with ThreadPoolExecutor(max_workers=2) as reader_pool:
        pending_read = None
        for i, job in enumerate(jobs):
            if pending_read is not None:
                try:
                    pending_read.result()
                except Exception:
                    pass
            if i + 1 < len(jobs):
                nxt = jobs[i + 1]
                pending_read = reader_pool.submit(read_grayscale, Path(nxt.image_path))
            results.append(process_extract_job(job))
    return results


def process_extract_job(job: ExtractJob) -> ExtractJobResult:
    """Process one image; intended for ProcessPoolExecutor."""
    global _WORKER_GPU, _WORKER_CFG
    cfg = _WORKER_CFG
    gpu_ctx = _WORKER_GPU
    if cfg is None:
        return ExtractJobResult(job.image_id, False, error="worker not initialized")

    cache_npz = cache_path_for_image(cfg, job.image_id, "npz")
    if job.skip_if_cached and cache_npz.exists():
        mpath = cfg.output.output_dir / "tables" / "per_image_metrics.csv"
        if mpath.exists():
            import pandas as pd

            existing = pd.read_csv(mpath)
            if job.image_id in existing["image_id"].astype(str).values:
                return ExtractJobResult(job.image_id, True, skipped=True)
        try:
            np.load(cache_npz)
            return ExtractJobResult(job.image_id, True, skipped=True)
        except Exception:
            pass

    rec = _record_from_job(job)
    ink_gpu = gpu_ctx is not None and gpu_ctx.on_gpu and cfg.gpu.preprocessing
    hm_gpu = gpu_ctx is not None and gpu_ctx.on_gpu and cfg.gpu.enabled

    try:
        gray = read_grayscale(rec.image_path)
        template = load_template(cfg.input.template_dir, rec.template_id)
        aligned = False
        if template is not None and cfg.preprocessing.align_to_template:
            from heatmap_analysis.alignment import align_to_template

            gray, template = align_to_template(gray, template)
            aligned = True

        ink, region, info = extract_ink_mask(
            gray, cfg.preprocessing, template, aligned=aligned, use_gpu=ink_gpu, xp=gpu_ctx.xp if gpu_ctx else np
        )
        ink_for_validate = to_numpy(ink)
        hm = build_heatmaps(
            ink,
            cfg.heatmap,
            cfg.preprocessing.blank_ink_threshold,
            use_gpu=hm_gpu,
            xp=gpu_ctx.xp if gpu_ctx else np,
        )
        m = compute_metrics(rec.image_id, hm, ink_for_validate, cfg.heatmap, rec.template_id)

        save_dict = {
            "d_abs": to_numpy(hm.d_abs),
            "d_rel": to_numpy(hm.d_rel),
            "is_blank": hm.is_blank,
        }
        if cfg.heatmap.save_smoothed_grid:
            save_dict["d_abs_smooth"] = to_numpy(hm.d_abs_smooth)
            save_dict["d_rel_smooth"] = to_numpy(hm.d_rel_smooth)
        np.savez_compressed(cache_npz, **save_dict)

        row = m.to_dict()
        row["image_path"] = str(rec.image_path)
        row["rel_path"] = rec.rel_path
        row["extraction_mode"] = info.get("mode")
        for k, v in rec.metadata.items():
            if k not in row:
                row[k] = v

        issues_raw = validate_image(rec, cfg, (ink_for_validate > 0).astype(np.uint8) * 255)
        issues = [
            {
                "image_id": i.image_id,
                "image_path": i.image_path,
                "issue_type": i.issue_type,
                "message": i.message,
            }
            for i in issues_raw
        ]
        return ExtractJobResult(job.image_id, True, metrics_row=row, issues=issues or None)

    except Exception as e:
        return ExtractJobResult(
            job.image_id,
            False,
            error=str(e),
            trace=traceback.format_exc(),
        )
