"""Build image manifest with stable ids and resume-friendly status."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import fingerprint_config, load_config
from .feature_store import atomic_replace_parquet
from .utils import iter_images, make_image_id, safe_image_meta, sha256_file


def _process_one(path: Path, root: Path) -> dict[str, Any]:
    abs_path = str(path.resolve())
    rel = str(path.relative_to(root))
    meta = safe_image_meta(path)
    row: dict[str, Any] = {
        "absolute_path": abs_path,
        "relative_path": rel,
        "sha256": None,
        "width": meta["width"],
        "height": meta["height"],
        "file_size": meta["file_size"],
        "status": "pending",
        "error_message": meta["error_message"],
    }
    if meta["error_message"]:
        row["status"] = "corrupt"
        row["image_id"] = make_image_id(abs_path)
        return row
    try:
        digest = sha256_file(path)
        row["sha256"] = digest
        row["image_id"] = make_image_id(abs_path, digest)
    except Exception as e:  # noqa: BLE001
        row["status"] = "corrupt"
        row["error_message"] = f"{type(e).__name__}: {e}"
        row["image_id"] = make_image_id(abs_path)
    return row


def build_manifest(cfg: dict[str, Any], workers: int = 32) -> pd.DataFrame:
    root = Path(cfg["data"]["input_dir"])
    exts = cfg["data"].get("image_extensions")
    recursive = bool(cfg["data"].get("recursive", True))
    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_path = out_dir / "manifest.parquet"
    fp = fingerprint_config(cfg)

    paths = list(iter_images(root, exts=exts, recursive=recursive))
    limit = cfg.get("pipeline", {}).get("limit")
    if limit:
        paths = paths[: int(limit)]

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, p, root): p for p in paths}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="manifest"):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    if df.empty:
        atomic_replace_parquet(df, out_path)
        return df

    # Resume: preserve prior non-pending statuses if same fingerprint
    meta_path = out_dir / "manifest_meta.json"
    if out_path.exists() and meta_path.exists():
        import json

        try:
            old_meta = json.loads(meta_path.read_text())
            if old_meta.get("config_fingerprint") == fp:
                old = pd.read_parquet(out_path)
                if not old.empty and "image_id" in old.columns:
                    keep_cols = [
                        c
                        for c in old.columns
                        if c
                        in {
                            "image_id",
                            "status",
                            "error_message",
                            "ocr_status",
                            "visual_status",
                            "layout_status",
                            "recognition_status",
                        }
                    ]
                    merged = df.merge(old[keep_cols], on="image_id", how="left", suffixes=("", "_old"))
                    for col in ["status", "error_message", "ocr_status", "visual_status", "layout_status", "recognition_status"]:
                        old_c = f"{col}_old"
                        if old_c in merged.columns:
                            merged[col] = merged[old_c].combine_first(merged[col])
                            merged = merged.drop(columns=[old_c])
                    df = merged
        except Exception:
            pass

    df["config_fingerprint"] = fp
    atomic_replace_parquet(df, out_path)
    from .utils import atomic_write_json

    atomic_write_json(
        meta_path,
        {
            "config_fingerprint": fp,
            "n_images": int(len(df)),
            "n_corrupt": int((df["status"] == "corrupt").sum()) if "status" in df else 0,
            "input_dir": str(root),
        },
    )
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    df = build_manifest(cfg, workers=args.workers)
    print(f"[build_manifest] n={len(df)} -> {cfg['paths']['outputs_dir']}/manifest.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
