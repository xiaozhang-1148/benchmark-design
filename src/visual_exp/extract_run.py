"""Full / limited embedding extraction with shard→barrier→atomic merge."""

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
from .io_util import (
    assert_same_id_set,
    assert_same_n,
    atomic_write_npy,
    atomic_write_parquet,
    file_sha256,
    stamp_run_id,
    write_run_meta,
)


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
    run_id = str(cfg["run_id"])

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
                    "run_id": run_id,
                }
            )
        except Exception as e:  # noqa: BLE001
            fails.append(
                {
                    "image_id": iid,
                    "error_message": f"{type(e).__name__}: {e}",
                    "status": "fail",
                    "run_id": run_id,
                }
            )

    if vecs:
        arr = np.stack(vecs, axis=0).astype(np.float32)
    else:
        arr = np.zeros((0, dim or 1280), dtype=np.float32)
    atomic_write_npy(shard_dir / "embeddings.npy", arr)
    atomic_write_parquet(pd.DataFrame(meta_rows), shard_dir / "index.parquet")
    if fails:
        atomic_write_parquet(pd.DataFrame(fails), shard_dir / "failures.parquet")
    extractor.close()
    return {
        "ok": True,
        "n_ok": len(meta_rows),
        "n_fail": len(fails),
        "dim": dim,
        "shard": str(shard_dir),
        "ok_ids": [r["image_id"] for r in meta_rows],
        "fail_ids": [r["image_id"] for r in fails],
    }


def _dp_entry(payload: dict[str, Any], result_path: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["gpu_id"])
    try:
        meta = _extract_shard(
            payload["cfg"], payload["rows"], Path(payload["shard_dir"]), f"[w{payload['worker_id']}]"
        )
        meta.update({"worker_id": payload["worker_id"], "gpu_id": payload["gpu_id"]})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        meta = {"ok": False, "error": f"{type(e).__name__}: {e}", "worker_id": payload["worker_id"]}
    Path(result_path).write_text(json.dumps(meta), encoding="utf-8")


def _merge_shards(
    cfg: dict[str, Any],
    shard_metas: list[Path],
    *,
    expected_pending_ids: set[str],
    prev_X: np.ndarray | None,
    prev_idx: pd.DataFrame | None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Barrier: all shards must exist and succeed before merge."""
    run_id = str(cfg["run_id"])
    rebuilt_ids: list[str] = []
    rebuilt_vecs: list[np.ndarray] = []
    fail_rows: list[dict[str, Any]] = []

    if prev_X is not None and prev_idx is not None and len(prev_idx):
        prev_ok = prev_idx[prev_idx["status"] == "ok"].reset_index(drop=True)
        assert_same_n("prev_index", len(prev_ok), "prev_X", int(prev_X.shape[0]))
        for j, iid in enumerate(prev_ok["image_id"].astype(str)):
            rebuilt_ids.append(iid)
            rebuilt_vecs.append(prev_X[j])

    shard_ok_ids: set[str] = set()
    for shard in shard_metas:
        result_path = shard / "result.json"
        if not result_path.exists():
            # single-worker path may not write result.json — synthesize check via index
            if not (shard / "index.parquet").exists():
                raise RuntimeError(f"shard incomplete (missing result/index): {shard}")
        else:
            meta = json.loads(result_path.read_text())
            if not meta.get("ok"):
                raise RuntimeError(f"shard failed before merge: {meta}")

        ipath = shard / "index.parquet"
        vpath = shard / "embeddings.npy"
        if not ipath.exists() or not vpath.exists():
            raise RuntimeError(f"shard missing embeddings/index: {shard}")
        si = pd.read_parquet(ipath)
        sv = np.load(vpath)
        assert_same_n(f"{shard.name}_index", len(si), f"{shard.name}_npy", int(sv.shape[0]))
        for j, iid in enumerate(si["image_id"].astype(str)):
            rebuilt_ids.append(iid)
            rebuilt_vecs.append(sv[j])
            shard_ok_ids.add(iid)

        fpath = shard / "failures.parquet"
        if fpath.exists():
            fails = pd.read_parquet(fpath)
            fail_rows.extend(fails.to_dict("records"))

    # Pending IDs must be accounted for as ok or fail
    fail_ids = {str(r["image_id"]) for r in fail_rows}
    accounted = shard_ok_ids | fail_ids
    if expected_pending_ids and accounted != expected_pending_ids:
        raise RuntimeError(
            f"pending ID coverage failed after shards: "
            f"missing={sorted(expected_pending_ids - accounted)[:5]} "
            f"extra={sorted(accounted - expected_pending_ids)[:5]}"
        )

    if not rebuilt_vecs:
        raise RuntimeError("no embeddings to merge")

    X_final = np.stack(rebuilt_vecs, axis=0).astype(np.float32)
    # Deduplicate by image_id keeping last (newest shard wins)
    seen: dict[str, int] = {}
    for i, iid in enumerate(rebuilt_ids):
        seen[iid] = i
    order = list(seen.values())
    X_final = X_final[order]
    rebuilt_ids = [rebuilt_ids[i] for i in order]

    # Build index from shard indices
    meta_map: dict[str, dict[str, Any]] = {}
    for shard in shard_metas:
        si = pd.read_parquet(shard / "index.parquet")
        for r in si.to_dict("records"):
            meta_map[str(r["image_id"])] = r
    if prev_idx is not None:
        for r in prev_idx[prev_idx["status"] == "ok"].to_dict("records"):
            iid = str(r["image_id"])
            if iid not in meta_map:
                meta_map[iid] = r

    final_rows = []
    for row_i, iid in enumerate(rebuilt_ids):
        r = meta_map.get(iid, {})
        final_rows.append(
            {
                "image_id": iid,
                "embedding_row": row_i,
                "token_count": r.get("token_count"),
                "n_local_patches": r.get("n_local_patches"),
                "norm_before_l2": r.get("norm_before_l2"),
                "embedding_dim": int(X_final.shape[1]),
                "status": "ok",
                "error_message": None,
                "run_id": run_id,
            }
        )
    final_idx = pd.DataFrame(final_rows)
    fail_df = stamp_run_id(pd.DataFrame(fail_rows), run_id) if fail_rows else pd.DataFrame()
    assert_same_n("final_index", len(final_idx), "final_X", int(X_final.shape[0]))
    return X_final, final_idx, fail_df


def run_extract(cfg: dict[str, Any], *, limit: int | None = None, resume: bool | None = None) -> dict[str, Any]:
    dump_frozen_run_config(cfg)
    write_run_meta(cfg, stage="extract_start")
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
    prev_X = prev_idx = None
    if do_resume and out_index.exists() and out_npy.exists():
        prev_idx = pd.read_parquet(out_index)
        prev_X = np.load(out_npy)
        done = set(prev_idx.loc[prev_idx["status"] == "ok", "image_id"].astype(str))
        print(f"[extract] resume: {len(done)} already done")

    pending_df = man[~man["image_id"].astype(str).isin(done)]
    pending = pending_df.to_dict("records")
    pending_ids = set(pending_df["image_id"].astype(str))
    print(f"[extract] pending={len(pending)} total_target={len(man)} run_id={cfg['run_id']}")

    n_workers = _resolve_dp(cfg)
    shard_root = emb_dir / "shards" / cfg["run_id"]
    ensure_dir(shard_root)

    if not pending:
        print("[extract] nothing to do")
        sha = file_sha256(out_npy) if out_npy.exists() else None
        cfg["embedding_sha256"] = sha
        return {"n_ok": len(done), "path": str(out_npy), "embedding_sha256": sha}

    if n_workers == 1:
        shard = shard_root / "w0"
        meta = _extract_shard(cfg, pending, shard)
        atomic_write_json(shard / "result.json", meta)
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
            payload = {
                "worker_id": wid,
                "gpu_id": wid,
                "cfg": cfg,
                "rows": rows,
                "shard_dir": str(shard),
            }
            p = ctx.Process(target=_dp_entry, args=(payload, result_path), daemon=False)
            p.start()
            procs.append(p)
            result_paths.append(result_path)
            shard_metas.append(shard)

        # Barrier: wait for ALL workers before merge
        exit_codes = []
        for p in procs:
            p.join()
            exit_codes.append(p.exitcode)
        if any(c != 0 for c in exit_codes):
            raise RuntimeError(f"shard worker non-zero exit: {exit_codes}")
        for rp in result_paths:
            if not Path(rp).exists():
                raise RuntimeError(f"missing shard result after join: {rp}")
            meta = json.loads(Path(rp).read_text())
            if not meta.get("ok"):
                raise RuntimeError(f"worker failed before merge: {meta}")

    X_final, final_idx, fail_df = _merge_shards(
        cfg,
        shard_metas,
        expected_pending_ids=pending_ids,
        prev_X=prev_X if do_resume else None,
        prev_idx=prev_idx if do_resume else None,
    )

    # Atomic publish
    atomic_write_npy(out_npy, X_final)
    atomic_write_parquet(final_idx, out_index)
    if len(fail_df):
        atomic_write_parquet(fail_df, meta_dir / "extract_failures.parquet")

    emb_sha = file_sha256(out_npy)
    cfg["embedding_sha256"] = emb_sha

    # Update manifest (only rows in this run's target set)
    man_all = pd.read_parquet(man_path)
    man_all = stamp_run_id(man_all, cfg["run_id"])
    upd = final_idx.set_index("image_id")
    for col in ("token_count", "n_local_patches", "embedding_row", "norm_before_l2"):
        man_all[col] = man_all["image_id"].astype(str).map(upd[col].to_dict())
    ok_ids = set(final_idx["image_id"].astype(str))
    man_all.loc[man_all["image_id"].astype(str).isin(ok_ids), "status"] = "ok"
    if len(fail_df):
        fail_ids = set(fail_df["image_id"].astype(str))
        man_all.loc[man_all["image_id"].astype(str).isin(fail_ids), "status"] = "fail"
        err_map = fail_df.set_index("image_id")["error_message"].to_dict()
        man_all.loc[man_all["image_id"].astype(str).isin(fail_ids), "error_message"] = (
            man_all["image_id"].astype(str).map(err_map)
        )
    atomic_write_parquet(man_all, man_path)

    # Consistency: embedding IDs ⊆ manifest ok
    assert_same_id_set("embeddings", final_idx["image_id"], "index", final_idx["image_id"])

    summary = {
        "run_id": cfg["run_id"],
        "n_embeddings": int(X_final.shape[0]),
        "dim": int(X_final.shape[1]),
        "path": str(out_npy),
        "embedding_sha256": emb_sha,
        "method": cfg.get("method_name"),
        "use_local_patches": cfg.get("use_local_patches", False),
        "success_rate": float(X_final.shape[0] / max(len(man), 1)),
        "n_fail": int(len(fail_df)),
    }
    atomic_write_json(Path(cfg["paths"]["diagnostics_dir"]) / "extraction_summary.json", summary)
    write_run_meta(cfg, stage="extract_done", **summary)
    dump_frozen_run_config(cfg)
    print(f"[extract] saved {X_final.shape} sha256={emb_sha[:12]}… -> {out_npy}")
    return summary
