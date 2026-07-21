"""Full / limited embedding extraction into experiment/embeddings/."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils import atomic_write_json, ensure_dir, load_image_rgb
from .config import dump_frozen_run_config
from .extractor import ProjectedTokenExtractor


def _resolve_dp(cfg: dict[str, Any]) -> int:
    raw = (cfg.get("extract") or {}).get("data_parallel_size", "auto")
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 1
    if raw is None or raw == "auto":
        return max(1, n)
    return max(1, min(int(raw), max(1, n)))


def _extract_shard(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    shard_dir: Path,
    worker_tag: str = "",
) -> dict[str, Any]:
    ensure_dir(shard_dir)
    extractor = ProjectedTokenExtractor(cfg)
    dim = None
    vecs: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    fails: list[dict[str, Any]] = []

    for r in tqdm(rows, desc=f"extract{worker_tag}"):
        iid = str(r["image_id"])
        try:
            img = load_image_rgb(r["image_path"])
            out = extractor.embed_image(img, debug=False)
            if dim is None:
                dim = int(out["embedding_dim"])
            elif out["embedding_dim"] != dim:
                raise RuntimeError(f"dim mismatch {out['embedding_dim']} vs {dim}")
            vecs.append(out["embedding"])
            meta_rows.append(
                {
                    "image_id": iid,
                    "token_count": out["token_count"],
                    "n_local_patches": out["n_local_patches"],
                    "norm_before_l2": out["norm_before"],
                    "embedding_dim": out["embedding_dim"],
                    "status": "ok",
                    "error_message": None,
                }
            )
        except Exception as e:  # noqa: BLE001
            fails.append({"image_id": iid, "error_message": f"{type(e).__name__}: {e}", "status": "fail"})

    if vecs:
        arr = np.stack(vecs, axis=0).astype(np.float32)
        np.save(shard_dir / "embeddings.npy", arr)
    else:
        arr = np.zeros((0, dim or 1280), dtype=np.float32)
        np.save(shard_dir / "embeddings.npy", arr)
    pd.DataFrame(meta_rows).to_parquet(shard_dir / "index.parquet", index=False)
    if fails:
        pd.DataFrame(fails).to_parquet(shard_dir / "failures.parquet", index=False)
    extractor.close()
    return {"ok": True, "n_ok": len(meta_rows), "n_fail": len(fails), "dim": dim, "shard": str(shard_dir)}


def _dp_entry(payload: dict[str, Any], result_path: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["gpu_id"])
    try:
        meta = _extract_shard(payload["cfg"], payload["rows"], Path(payload["shard_dir"]), f"[w{payload['worker_id']}]")
        meta.update({"worker_id": payload["worker_id"], "gpu_id": payload["gpu_id"]})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        meta = {"ok": False, "error": f"{type(e).__name__}: {e}", "worker_id": payload["worker_id"]}
    Path(result_path).write_text(json.dumps(meta), encoding="utf-8")


def run_extract(cfg: dict[str, Any], *, limit: int | None = None, resume: bool | None = None) -> dict[str, Any]:
    dump_frozen_run_config(cfg)
    meta_dir = Path(cfg["paths"]["metadata_dir"])
    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    man_path = meta_dir / "manifest.parquet"
    if not man_path.exists():
        raise FileNotFoundError(f"Missing {man_path}; build manifest first")

    man = pd.read_parquet(man_path)
    man = man[man["status"] != "corrupt"].copy()
    if limit is not None:
        man = man.head(int(limit)).copy()

    save_name = (cfg.get("extract") or {}).get("save_name", "deepseek_ocr2_mean_l2.npy")
    out_npy = emb_dir / save_name
    out_index = meta_dir / "embedding_index.parquet"
    do_resume = bool(cfg.get("extract", {}).get("resume", True) if resume is None else resume)

    done: set[str] = set()
    if do_resume and out_index.exists() and out_npy.exists():
        prev = pd.read_parquet(out_index)
        done = set(prev.loc[prev["status"] == "ok", "image_id"].astype(str))
        print(f"[extract] resume: {len(done)} already done")

    pending = man[~man["image_id"].astype(str).isin(done)].to_dict("records")
    print(f"[extract] pending={len(pending)} total_target={len(man)}")

    n_workers = _resolve_dp(cfg)
    shard_root = emb_dir / "shards"
    ensure_dir(shard_root)

    if not pending:
        print("[extract] nothing to do")
        return {"n_ok": len(done), "path": str(out_npy)}

    if n_workers == 1:
        shard = shard_root / "w0"
        _extract_shard(cfg, pending, shard)
        shard_metas = [shard]
    else:
        import multiprocessing as mp

        shards_rows: list[list[dict]] = [[] for _ in range(n_workers)]
        for i, r in enumerate(pending):
            shards_rows[i % n_workers].append(r)
        ctx = mp.get_context("spawn")
        procs = []
        result_paths = []
        shard_metas = []
        for wid, rows in enumerate(shards_rows):
            if not rows:
                continue
            shard = shard_root / f"w{wid}"
            ensure_dir(shard)
            result_path = str(shard / "result.json")
            payload = {"worker_id": wid, "gpu_id": wid, "cfg": cfg, "rows": rows, "shard_dir": str(shard)}
            p = ctx.Process(target=_dp_entry, args=(payload, result_path), daemon=False)
            p.start()
            procs.append(p)
            result_paths.append(result_path)
            shard_metas.append(shard)
        for p in procs:
            p.join()
        for rp in result_paths:
            meta = json.loads(Path(rp).read_text())
            if not meta.get("ok"):
                raise RuntimeError(f"worker failed: {meta}")

    # Merge shards (+ previous if resume)
    parts_v = []
    parts_i = []
    if do_resume and out_npy.exists() and out_index.exists():
        parts_v.append(np.load(out_npy))
        parts_i.append(pd.read_parquet(out_index))
    for shard in shard_metas:
        vpath = shard / "embeddings.npy"
        ipath = shard / "index.parquet"
        if vpath.exists() and ipath.exists() and len(pd.read_parquet(ipath)):
            parts_v.append(np.load(vpath))
            parts_i.append(pd.read_parquet(ipath))
        fpath = shard / "failures.parquet"
        if fpath.exists():
            fails = pd.read_parquet(fpath)
            # merge fail status into manifest later
            parts_i.append(fails)

    if not parts_v:
        raise RuntimeError("no embeddings produced")

    X = np.concatenate(parts_v, axis=0).astype(np.float32)
    idx = pd.concat(parts_i, ignore_index=True)
    # keep last status per image_id preferring ok
    idx["_ok"] = (idx["status"] == "ok").astype(int)
    idx = idx.sort_values(["image_id", "_ok"]).drop_duplicates("image_id", keep="last").drop(columns=["_ok"])
    ok = idx[idx["status"] == "ok"].reset_index(drop=True)

    # Reorder X to match ok rows from concatenated shards carefully:
    # Rebuild by reading shard ok rows only in merge order of ok ids from new extract + old
    # Safer: rebuild from shards only for pending, then concat previous ok
    rebuilt_ids: list[str] = []
    rebuilt_vecs: list[np.ndarray] = []
    if do_resume and out_npy.exists() and out_index.exists():
        prev_idx = pd.read_parquet(out_index)
        prev_ok = prev_idx[prev_idx["status"] == "ok"].reset_index(drop=True)
        prev_X = np.load(out_npy)
        id_to_row = {str(i): r for r, i in enumerate(prev_ok["image_id"].astype(str))}
        for iid in prev_ok["image_id"].astype(str):
            if iid in done:
                rebuilt_ids.append(iid)
                rebuilt_vecs.append(prev_X[id_to_row[iid]])

    for shard in shard_metas:
        ipath = shard / "index.parquet"
        vpath = shard / "embeddings.npy"
        if not ipath.exists() or not vpath.exists():
            continue
        si = pd.read_parquet(ipath)
        sv = np.load(vpath)
        for j, iid in enumerate(si["image_id"].astype(str)):
            rebuilt_ids.append(iid)
            rebuilt_vecs.append(sv[j])

    X_final = np.stack(rebuilt_vecs, axis=0).astype(np.float32)
    # Build final index aligned with X_final
    meta_map = {str(r.image_id): r for r in idx.itertuples()}
    final_rows = []
    for row_i, iid in enumerate(rebuilt_ids):
        r = meta_map.get(iid)
        final_rows.append(
            {
                "image_id": iid,
                "embedding_row": row_i,
                "token_count": getattr(r, "token_count", None) if r else None,
                "n_local_patches": getattr(r, "n_local_patches", None) if r else None,
                "norm_before_l2": getattr(r, "norm_before_l2", None) if r else None,
                "embedding_dim": int(X_final.shape[1]),
                "status": "ok",
                "error_message": None,
            }
        )
    final_idx = pd.DataFrame(final_rows)
    np.save(out_npy, X_final)
    final_idx.to_parquet(out_index, index=False)

    # Update manifest with audit fields
    man_all = pd.read_parquet(man_path)
    upd = final_idx.set_index("image_id")
    for col in ("token_count", "n_local_patches", "embedding_row", "norm_before_l2"):
        if col in upd.columns:
            man_all[col] = man_all["image_id"].astype(str).map(upd[col].to_dict())
    man_all.loc[man_all["image_id"].astype(str).isin(set(final_idx["image_id"].astype(str))), "status"] = "ok"
    # mark fails
    fail_ids = set(idx.loc[idx["status"] != "ok", "image_id"].astype(str))
    if fail_ids:
        man_all.loc[man_all["image_id"].astype(str).isin(fail_ids), "status"] = "fail"
        err_map = idx.set_index("image_id")["error_message"].to_dict()
        man_all.loc[man_all["image_id"].astype(str).isin(fail_ids), "error_message"] = (
            man_all["image_id"].astype(str).map(err_map)
        )
    man_all.to_parquet(man_path, index=False)

    summary = {
        "n_embeddings": int(X_final.shape[0]),
        "dim": int(X_final.shape[1]),
        "path": str(out_npy),
        "method": cfg.get("method_name"),
        "success_rate": float(X_final.shape[0] / max(len(man), 1)),
    }
    atomic_write_json(Path(cfg["paths"]["diagnostics_dir"]) / "extraction_summary.json", summary)
    print(f"[extract] saved {X_final.shape} -> {out_npy}")
    return summary
