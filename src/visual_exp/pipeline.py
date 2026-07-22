"""Four-stage visual experiment pipeline with ordered barriers."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .clustering import run_clustering
from .config import dump_frozen_run_config, load_run_config
from .diagnostics import run_diagnostics
from .extract_run import run_extract
from .galleries import run_galleries
from .io_util import atomic_write_npy, atomic_write_parquet, file_sha256, stamp_run_id, write_run_meta
from .manifest import build_manifest
from .projections import run_projections
from .report import build_html_report
from .verify import run_verify


def import_embeddings_from(
    cfg: dict[str, Any],
    *,
    source_root: str | Path,
) -> dict[str, Any]:
    """
    Copy existing embeddings/index/manifest into this run directory (no re-extract).
    source_root: legacy experiment root or another run_dir.
    """
    src = Path(source_root)
    # accept either experiment/ or experiment/runs/<id>/
    candidates = [
        (
            src / "embeddings" / "deepseek_ocr2_mean_l2.npy",
            src / "metadata" / "embedding_index.parquet",
            src / "metadata" / "manifest.parquet",
        ),
        (
            src / "deepseek_ocr2_mean_l2.npy",
            src / "embedding_index.parquet",
            src / "manifest.parquet",
        ),
    ]
    emb_src = idx_src = man_src = None
    for e, i, m in candidates:
        if e.exists() and i.exists():
            emb_src, idx_src, man_src = e, i, m if m.exists() else None
            break
    if emb_src is None:
        raise FileNotFoundError(f"No embeddings found under {src}")

    save_name = (cfg.get("extract") or {}).get("save_name", "deepseek_ocr2_mean_l2.npy")
    emb_dst = Path(cfg["paths"]["embeddings_dir"]) / save_name
    idx_dst = Path(cfg["paths"]["metadata_dir"]) / "embedding_index.parquet"
    man_dst = Path(cfg["paths"]["metadata_dir"]) / "manifest.parquet"

    import numpy as np

    X = np.load(emb_src)
    idx = pd.read_parquet(idx_src)
    idx = stamp_run_id(idx, cfg["run_id"])
    if "embedding_row" not in idx.columns:
        idx = idx.reset_index(drop=True)
        idx["embedding_row"] = np.arange(len(idx))
    atomic_write_npy(emb_dst, X.astype(np.float32))
    atomic_write_parquet(idx, idx_dst)

    if man_src is not None:
        man = pd.read_parquet(man_src)
        man = stamp_run_id(man, cfg["run_id"])
        atomic_write_parquet(man, man_dst)
    else:
        # minimal manifest from index
        man = stamp_run_id(pd.DataFrame({"image_id": idx["image_id"], "status": "ok"}), cfg["run_id"])
        atomic_write_parquet(man, man_dst)

    sha = file_sha256(emb_dst)
    cfg["embedding_sha256"] = sha
    summary = {
        "run_id": cfg["run_id"],
        "imported_from": str(src),
        "n_embeddings": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "embedding_sha256": sha,
        "path": str(emb_dst),
    }
    write_run_meta(cfg, stage="import_embeddings", **summary)
    dump_frozen_run_config(cfg)
    print(f"[import] {X.shape} sha256={sha[:12]}… -> run {cfg['run_id']}")
    return summary


def run_visual_experiment(cfg: dict[str, Any], stages: list[str] | None = None) -> None:
    dump_frozen_run_config(cfg)
    write_run_meta(cfg, stage="pipeline_start")
    mapping = {
        "manifest": lambda: build_manifest(cfg),
        "verify": lambda: _verify(cfg),
        "smoke": lambda: _smoke(cfg),
        "extract": lambda: run_extract(cfg, limit=None),
        "import_embeddings": lambda: None,  # handled via CLI flag
        "diagnostics": lambda: run_diagnostics(cfg),
        "projections": lambda: run_projections(cfg),
        "clustering": lambda: run_clustering(cfg),
        "galleries": lambda: run_galleries(cfg),
        "report": lambda: build_html_report(cfg),
        "analyze": lambda: _analyze(cfg),
    }
    stages = stages or ["manifest", "verify", "extract", "analyze"]
    for s in stages:
        print(f"===== STAGE: {s} =====")
        if s not in mapping:
            raise KeyError(f"Unknown stage: {s}")
        mapping[s]()
        print(f"===== DONE: {s} =====")
    write_run_meta(cfg, stage="pipeline_done", embedding_sha256=cfg.get("embedding_sha256"))


def _verify(cfg: dict[str, Any]) -> None:
    n = int(cfg["stages"]["verify_n"])
    build_manifest(cfg, limit=n)
    run_extract(cfg, limit=n, resume=False)
    run_verify(cfg)


def _smoke(cfg: dict[str, Any]) -> None:
    n = int(cfg["stages"]["smoke_n"])
    build_manifest(cfg, limit=n)
    run_extract(cfg, limit=n, resume=False)
    # barrier: extract must finish before analyze
    _analyze(cfg)


def _analyze(cfg: dict[str, Any]) -> None:
    # Ordered: diagnostics → projections → clustering → galleries → report
    run_diagnostics(cfg)
    run_projections(cfg)
    run_clustering(cfg)
    run_galleries(cfg)
    build_html_report(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR2 mean-pooled projected-token embedding experiment"
    )
    parser.add_argument("--config", default="configs/experiment/run_config.yaml")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="manifest verify smoke extract diagnostics projections clustering galleries report analyze",
    )
    parser.add_argument(
        "--import-from",
        default=None,
        help="Copy embeddings from legacy experiment root or another run_dir into a NEW run, then analyze",
    )
    parser.add_argument("--run-id", default=None, help="Optional fixed run_id")
    args = parser.parse_args(argv)
    cfg = load_run_config(args.config, run_id=args.run_id)

    if args.import_from:
        import_embeddings_from(cfg, source_root=args.import_from)
        stages = args.stages or ["analyze"]
        run_visual_experiment(cfg, stages=stages)
        return 0

    run_visual_experiment(cfg, stages=args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
