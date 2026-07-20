"""End-to-end pipeline steps."""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from heatmap_analysis.aggregation import run_aggregation
from heatmap_analysis.cluster_study import run_cluster_study
from heatmap_analysis.comparison import compare_groups
from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.extract_worker import ExtractJob, extract_jobs_on_gpu
from heatmap_analysis.gpu import detect_device_ids, gpu_device, is_gpu_available, to_numpy
from heatmap_analysis.handwriting import extract_ink_mask
from heatmap_analysis.heatmap import build_heatmaps
from heatmap_analysis.io import (
    build_metadata_index,
    cache_path_for_image,
    load_template,
    read_grayscale,
    records_from_metadata,
    run_preprocessing_checks,
    validate_image,
)
from heatmap_analysis.metrics import compute_metrics
from heatmap_analysis.reporting import generate_report
from heatmap_analysis.utils import ensure_dir, save_json
from heatmap_analysis.visualization import generate_all_visualizations

logger = logging.getLogger("heatmap_analysis.pipeline")


def _finalize_extract_results(
    cfg: AnalysisConfig,
    metrics_rows: list[dict],
    all_issues: list[dict],
    failures: list[dict],
    skipped: int,
    processed: int,
) -> pd.DataFrame:
    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        out_csv = cfg.output.output_dir / "tables" / "per_image_metrics.csv"
        if out_csv.exists() and cfg.output.resume:
            mdf = pd.concat([pd.read_csv(out_csv), mdf], ignore_index=True).drop_duplicates(
                subset=["image_id"], keep="last"
            )
        mdf.to_csv(out_csv, index=False)
    elif (cfg.output.output_dir / "tables" / "per_image_metrics.csv").exists():
        mdf = pd.read_csv(cfg.output.output_dir / "tables" / "per_image_metrics.csv")
    else:
        mdf = pd.DataFrame()

    if all_issues:
        fail_path = cfg.output.output_dir / "tables" / "validation_issues.csv"
        issue_df = pd.DataFrame(all_issues)
        if fail_path.exists():
            issue_df = pd.concat([pd.read_csv(fail_path), issue_df], ignore_index=True)
        issue_df.to_csv(fail_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(cfg.output.output_dir / "tables" / "failures.csv", index=False)

    save_json(
        cfg.output.output_dir / "report" / "extract_summary.json",
        {"processed": processed, "skipped": skipped, "failures": len(failures), "gpu": cfg.gpu.enabled},
    )
    logger.info("Extract done: %d new, %d skipped, %d failed", processed, skipped, len(failures))
    return mdf


def _extract_parallel_gpu(cfg: AnalysisConfig, records: list, cfg_path: Path) -> pd.DataFrame:
    device_ids = detect_device_ids(cfg.gpu.device_ids)
    if not device_ids:
        logger.warning("GPU requested but no devices found; falling back to CPU")
        return _extract_sequential(cfg, records, use_gpu=False)

    n_workers = cfg.gpu.num_workers or len(device_ids)
    n_workers = min(n_workers, len(device_ids), max(len(records), 1))
    active_gpus = device_ids[:n_workers]
    logger.info("GPU parallel extract: %d processes on GPUs %s", n_workers, active_gpus)

    jobs = [
        ExtractJob(
            image_id=rec.image_id,
            image_path=str(rec.image_path),
            rel_path=rec.rel_path,
            template_id=rec.template_id,
            metadata=rec.metadata,
            skip_if_cached=cfg.output.resume,
        )
        for rec in records
    ]

    # Partition jobs round-robin across GPUs
    chunks: list[list[ExtractJob]] = [[] for _ in range(n_workers)]
    for i, job in enumerate(jobs):
        chunks[i % n_workers].append(job)

    metrics_rows: list[dict] = []
    all_issues: list[dict] = []
    failures: list[dict] = []
    skipped = 0
    processed = 0

    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
        futures = {
            executor.submit(
                extract_jobs_on_gpu,
                active_gpus[i],
                chunks[i],
                str(cfg_path),
                cfg.gpu.enabled,
            ): (active_gpus[i], len(chunks[i]))
            for i in range(n_workers)
            if chunks[i]
        }
        with tqdm(total=len(jobs), desc=f"Extracting heatmaps (GPU×{n_workers})") as pbar:
            for future in as_completed(futures):
                gpu_id, chunk_len = futures[future]
                try:
                    results = future.result()
                except Exception as e:
                    logger.error("GPU worker %d crashed: %s", gpu_id, e)
                    pbar.update(chunk_len)
                    continue
                for res in results:
                    pbar.update(1)
                    if res.skipped:
                        skipped += 1
                    elif res.success and res.metrics_row:
                        processed += 1
                        metrics_rows.append(res.metrics_row)
                        if res.issues:
                            all_issues.extend(res.issues)
                    elif not res.success:
                        failures.append(
                            {"image_id": res.image_id, "error": res.error, "trace": res.trace}
                        )

    return _finalize_extract_results(cfg, metrics_rows, all_issues, failures, skipped, processed)


def _extract_sequential(cfg: AnalysisConfig, records: list, use_gpu: bool) -> pd.DataFrame:
    metrics_rows: list[dict] = []
    all_issues: list[dict] = []
    failures: list[dict] = []
    skipped = 0
    processed = 0

    desc = "Extracting heatmaps (GPU)" if use_gpu else "Extracting heatmaps (CPU)"
    ctx = gpu_device(0, enabled=use_gpu) if use_gpu else gpu_device(None, enabled=False)

    with ctx as gctx:
        for rec in tqdm(records, desc=desc):
            cache_npz = cache_path_for_image(cfg, rec.image_id, "npz")
            if cfg.output.resume and cache_npz.exists():
                skipped += 1
                try:
                    np.load(cache_npz)
                    mpath = cfg.output.output_dir / "tables" / "per_image_metrics.csv"
                    if mpath.exists():
                        existing = pd.read_csv(mpath)
                        if rec.image_id in existing["image_id"].astype(str).values:
                            continue
                except Exception:
                    pass

            try:
                gray = read_grayscale(rec.image_path)
                template = load_template(cfg.input.template_dir, rec.template_id)
                aligned = False
                if template is not None and cfg.preprocessing.align_to_template:
                    from heatmap_analysis.alignment import align_to_template

                    gray, template = align_to_template(gray, template)
                    aligned = True

                ink, region, info = extract_ink_mask(
                    gray,
                    cfg.preprocessing,
                    template,
                    aligned=aligned,
                    use_gpu=gctx.on_gpu and cfg.gpu.preprocessing,
                    xp=gctx.xp,
                )
                ink_for_validate = to_numpy(ink)
                hm = build_heatmaps(
                    ink,
                    cfg.heatmap,
                    cfg.preprocessing.blank_ink_threshold,
                    use_gpu=gctx.on_gpu and cfg.gpu.enabled,
                    xp=gctx.xp,
                )
                m = compute_metrics(rec.image_id, hm, ink_for_validate, cfg.heatmap, rec.template_id)

                save_dict = {"d_abs": hm.d_abs, "d_rel": hm.d_rel, "is_blank": hm.is_blank}
                if cfg.heatmap.save_smoothed_grid:
                    save_dict["d_abs_smooth"] = hm.d_abs_smooth
                    save_dict["d_rel_smooth"] = hm.d_rel_smooth
                np.savez_compressed(cache_npz, **save_dict)

                row = m.to_dict()
                row["image_path"] = str(rec.image_path)
                row["rel_path"] = rec.rel_path
                row["extraction_mode"] = info.get("mode")
                for k, v in rec.metadata.items():
                    if k not in row:
                        row[k] = v
                metrics_rows.append(row)
                processed += 1

                issues = validate_image(rec, cfg, (ink_for_validate > 0).astype(np.uint8) * 255)
                for i in issues:
                    all_issues.append(
                        {
                            "image_id": i.image_id,
                            "image_path": i.image_path,
                            "issue_type": i.issue_type,
                            "message": i.message,
                        }
                    )

            except Exception as e:
                failures.append({"image_id": rec.image_id, "error": str(e), "trace": traceback.format_exc()})
                logger.error("Failed %s: %s", rec.image_id, e)

    return _finalize_extract_results(cfg, metrics_rows, all_issues, failures, skipped, processed)


def extract_all(cfg: AnalysisConfig, limit: int | None = None, cfg_path: Path | None = None) -> pd.DataFrame:
    """Extract heatmaps and metrics for all images (multi-GPU parallel when enabled)."""
    df = build_metadata_index(cfg)
    records = records_from_metadata(cfg, df)
    if limit is not None:
        records = records[:limit]

    ensure_dir(cfg.cache_dir / "per_image")
    ensure_dir(cfg.output.output_dir / "tables")

    use_gpu = cfg.gpu.enabled and is_gpu_available()
    device_ids = detect_device_ids(cfg.gpu.device_ids) if use_gpu else []

    if use_gpu and len(device_ids) > 1 and len(records) >= cfg.gpu.min_images_for_parallel:
        path = cfg_path or Path("config/heatmap_analysis.yaml")
        return _extract_parallel_gpu(cfg, records, path)

    if use_gpu and device_ids:
        logger.info("Single-GPU extract on device %d", device_ids[0])
        return _extract_sequential(cfg, records, use_gpu=True)

    if cfg.gpu.enabled and not is_gpu_available():
        logger.warning("gpu.enabled=true but CuPy/CUDA unavailable; using CPU")
    return _extract_sequential(cfg, records, use_gpu=False)


def run_all(cfg: AnalysisConfig, limit: int | None = None, group_by: list[str] | None = None, cfg_path: Path | None = None) -> None:
    ensure_dir(cfg.output.output_dir)
    setup_log = cfg.output.output_dir / "heatmap_analysis.log"
    from heatmap_analysis.utils import setup_logging

    setup_logging(setup_log)

    run_preprocessing_checks(cfg)
    extract_all(cfg, limit=limit, cfg_path=cfg_path)
    run_aggregation(cfg)
    fields = group_by if group_by is not None else cfg.report.group_by
    if fields:
        compare_groups(cfg, fields)
    run_cluster_study(cfg, skip_extract=True, cfg_path=cfg_path)
    generate_all_visualizations(cfg)
    report_path = generate_report(cfg)
    logger.info("Pipeline complete. Report: %s", report_path)
