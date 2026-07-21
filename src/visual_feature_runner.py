"""Channel B: Transformers BF16 visual-layer pooled embedding extraction."""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import fingerprint_config, load_config, model_resolved_path
from .feature_store import EmbeddingStore
from .model_compat import patch_transformers_for_deepseek_ocr2
from .utils import atomic_write_json, ensure_dir, l2_normalize, load_image_rgb


def _resolve_visual_data_parallel(cfg: dict[str, Any]) -> int:
    raw = (cfg.get("visual") or {}).get("data_parallel_size", "auto")
    try:
        n_gpu = int(torch.cuda.device_count())
    except Exception:
        n_gpu = 1
    if raw is None or raw == "auto":
        return max(1, n_gpu)
    return max(1, min(int(raw), max(1, n_gpu)))


def _dynamic_preprocess(image: Image.Image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    """Port of DeepSeek-OCR-2 dynamic crop logic (simplified from official)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / max(orig_height, 1)

    target_ratios = []
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j == n:
                    target_ratios.append((i, j))
    # find closest aspect
    best = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0] / r[1]))
    target_width = image_size * best[0]
    target_height = image_size * best[1]
    resized = image.resize((target_width, target_height))
    crops = []
    for i in range(best[1]):
        for j in range(best[0]):
            box = (j * image_size, i * image_size, (j + 1) * image_size, (i + 1) * image_size)
            crops.append(resized.crop(box))
    return crops, best


class BasicImageTransform:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        from torchvision import transforms

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.tf(img)


def prepare_image_tensors(
    image: Image.Image,
    base_size: int = 1024,
    image_size: int = 768,
    crop_mode: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Returns:
      patches: [P, 3, image_size, image_size] or zeros
      global_view: [1, 3, base_size, base_size]
      crop_ratio: [w_crops, h_crops]
    """
    transform = BasicImageTransform()
    if crop_mode and (image.size[0] > 768 or image.size[1] > 768):
        crops, crop_ratio = _dynamic_preprocess(image, image_size=image_size)
        patches = torch.stack([transform(c) for c in crops], dim=0)
        spatial = [crop_ratio[0], crop_ratio[1]]
    else:
        patches = torch.zeros(1, 3, image_size, image_size)
        spatial = [1, 1]

    global_view = ImageOps.pad(
        image,
        (base_size, base_size),
        color=tuple(int(x * 255) for x in transform.mean),
    )
    global_t = transform(global_view).unsqueeze(0)
    return patches.to(torch.bfloat16), global_t.to(torch.bfloat16), spatial


class VisualLayerExtractor:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.device = cfg["visual"].get("device", "cuda")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.model_path = model_resolved_path(cfg)
        self.selected_layer = cfg["model"].get("selected_layer")
        self.pool_causal = bool(cfg["visual"].get("pool_causal_flow_only", True))
        self._load()

    def _load(self) -> None:
        from transformers import AutoModel

        patch_transformers_for_deepseek_ocr2()
        print(f"[visual] loading {self.model_path}")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation=self.cfg["model"].get("attn_implementation", "sdpa"),
        )
        self.model.eval()
        self.model.to(self.device)
        m = self.model.model if hasattr(self.model, "model") else self.model
        self.sam = m.sam_model
        self.qwen = m.qwen2_model
        self.projector = m.projector

        layers = None
        for chain in [("model", "model", "layers"), ("model", "layers")]:
            cur = self.qwen
            ok = True
            for a in chain:
                if not hasattr(cur, a):
                    ok = False
                    break
                cur = getattr(cur, a)
            if ok:
                layers = cur
                break
        if layers is None:
            raise RuntimeError("Cannot locate qwen2 transformer layers")
        self.layers = layers
        n = len(layers)
        if self.selected_layer is None:
            self.selected_layer = n // 2
        self.selected_layer = int(max(0, min(self.selected_layer, n - 1)))
        self.num_layers = n
        self.hidden_dim = None  # set on first forward
        print(f"[visual] qwen2 layers={n} selected={self.selected_layer}")

    @torch.inference_mode()
    def embed_image(self, image: Image.Image) -> dict[str, Any]:
        patches, global_t, spatial = prepare_image_tensors(
            image,
            base_size=int(self.cfg["visual"].get("base_size", 1024)),
            image_size=int(self.cfg["visual"].get("image_size", 768)),
            crop_mode=bool(self.cfg["visual"].get("crop_mode", True)),
        )
        patches = patches.to(self.device)
        global_t = global_t.to(self.device)

        captured: dict[str, torch.Tensor] = {}

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h.detach()

        handle = self.layers[self.selected_layer].register_forward_hook(hook)
        try:
            # Encode global view through SAM -> Qwen2 (hook fires)
            g1 = self.sam(global_t)
            _ = self.qwen(g1)
            h_global = captured.get("h")
            if h_global is None:
                raise RuntimeError("hook did not capture global features")

            token_vecs = []
            # Sequence layout: [noncausal image tokens | causal-flow query tokens]
            # Final qwen output returns only causal half; intermediate has both.
            seq = h_global.shape[1]
            if self.pool_causal and seq % 2 == 0:
                causal = h_global[:, seq // 2 :, :]
            else:
                # If odd / unexpected, use all tokens but document
                causal = h_global
            token_vecs.append(causal.reshape(-1, causal.shape[-1]))

            # Local patches if present
            if spatial[0] > 1 or spatial[1] > 1:
                if patches.ndim == 4 and patches.shape[0] > 0 and float(patches.abs().sum()) > 0:
                    captured.clear()
                    l1 = self.sam(patches)
                    _ = self.qwen(l1)
                    h_local = captured.get("h")
                    if h_local is not None:
                        seq_l = h_local.shape[1]
                        if self.pool_causal and seq_l % 2 == 0:
                            causal_l = h_local[:, seq_l // 2 :, :]
                        else:
                            causal_l = h_local
                        token_vecs.append(causal_l.reshape(-1, causal_l.shape[-1]))

            H = torch.cat(token_vecs, dim=0).float()
            # Exclude NaN rows if any
            finite = torch.isfinite(H).all(dim=-1)
            H = H[finite]
            if H.numel() == 0:
                raise RuntimeError("no valid visual tokens")
            token_count = int(H.shape[0])
            pooled = H.mean(dim=0)
            norm_before = float(torch.linalg.vector_norm(pooled).item())
            vec = l2_normalize(pooled.cpu().numpy())
            self.hidden_dim = int(vec.shape[0])
            return {
                "embedding": vec,
                "token_count": token_count,
                "norm_before": norm_before,
                "selected_layer": self.selected_layer,
                "embedding_dim": int(vec.shape[0]),
            }
        finally:
            handle.remove()


def _run_visual_pending_on_device(
    cfg: dict[str, Any],
    pending: list[dict[str, Any]],
    fp: str,
    shard_dir: Path,
    worker_tag: str = "",
) -> dict[str, Any]:
    """Embed pending rows; write shard index + npy. Returns meta dict."""
    ensure_dir(shard_dir)
    tag = f"[{worker_tag}] " if worker_tag else ""
    # After CUDA_VISIBLE_DEVICES, always use cuda:0
    cfg = dict(cfg)
    cfg["visual"] = dict(cfg.get("visual") or {})
    cfg["visual"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = VisualLayerExtractor(cfg)
    rows: list[dict[str, Any]] = []
    vecs: list[np.ndarray] = []
    fails: list[dict[str, Any]] = []
    dim = None
    for r in tqdm(pending, desc=f"visual_embed{tag}"):
        iid = str(r["image_id"])
        try:
            img = load_image_rgb(r["absolute_path"])
            out = extractor.embed_image(img)
            if dim is None:
                dim = int(out["embedding_dim"])
            elif out["embedding_dim"] != dim:
                raise RuntimeError(f"dim mismatch {out['embedding_dim']} vs {dim}")
            rows.append(
                {
                    "image_id": iid,
                    "selected_layer": out["selected_layer"],
                    "token_count": out["token_count"],
                    "embedding_norm_before_normalization": out["norm_before"],
                    "config_fingerprint": fp,
                }
            )
            vecs.append(out["embedding"].astype(np.float32))
        except Exception as e:  # noqa: BLE001
            fails.append(
                {
                    "image_id": iid,
                    "error_message": f"{type(e).__name__}: {e}",
                    "config_fingerprint": fp,
                }
            )

    index_path = shard_dir / "index.parquet"
    vec_path = shard_dir / "embeddings.f32.npy"
    fail_path = shard_dir / "failures.parquet"
    if rows:
        pd.DataFrame(rows).to_parquet(index_path, index=False)
        np.save(vec_path, np.stack(vecs, axis=0))
    else:
        pd.DataFrame(columns=["image_id"]).to_parquet(index_path, index=False)
        np.save(vec_path, np.zeros((0, dim or 896), dtype=np.float32))
    if fails:
        pd.DataFrame(fails).to_parquet(fail_path, index=False)

    meta = {
        "ok": True,
        "n_ok": len(rows),
        "n_fail": len(fails),
        "dim": int(dim or 896),
        "selected_layer": int(extractor.selected_layer),
        "num_layers": int(extractor.num_layers),
        "index_path": str(index_path),
        "vec_path": str(vec_path),
        "fail_path": str(fail_path) if fails else None,
    }
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def _visual_dp_entry(payload: dict[str, Any], result_path: str) -> None:
    gpu_id = int(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        meta = _run_visual_pending_on_device(
            payload["cfg"],
            payload["pending"],
            payload["fp"],
            Path(payload["shard_dir"]),
            worker_tag=f"w{payload['worker_id']}/gpu{gpu_id}",
        )
        meta.update({"worker_id": payload["worker_id"], "gpu_id": gpu_id})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        meta = {
            "worker_id": payload["worker_id"],
            "gpu_id": gpu_id,
            "ok": False,
            "n_ok": 0,
            "n_fail": 0,
            "error": f"{type(e).__name__}: {e}",
        }
    Path(result_path).write_text(json.dumps(meta), encoding="utf-8")


def run_visual_features(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    man = pd.read_parquet(out_dir / "manifest.parquet")
    man = man[man["status"] != "corrupt"].copy()
    fp = fingerprint_config(cfg)

    # Visual-only pipeline: embed all readable images (do not gate on OCR).

    # Load introspection for selected layer
    intro = Path(cfg["paths"]["reports_dir"]) / "model_introspection.json"
    if intro.exists():
        info = json.loads(intro.read_text())
        if cfg["model"].get("selected_layer") is None:
            cfg["model"]["selected_layer"] = info.get("selected_layer_index")

    dim = 896
    if intro.exists():
        shape = json.loads(intro.read_text()).get("selected_layer_output_shape")
        if shape and len(shape) >= 3:
            dim = int(shape[-1])

    resume = bool(cfg.get("pipeline", {}).get("resume", True))
    store = EmbeddingStore(
        mmap_path=out_dir / "visual_embeddings.f32.mmap",
        index_path=out_dir / "visual_index.parquet",
        dim=dim,
    )
    done = store.done_ids if resume else set()
    if not resume:
        # Full rebuild: wipe previous visual store
        for p in (
            out_dir / "visual_embeddings.f32.mmap",
            out_dir / "visual_index.parquet",
            out_dir / "visual_embeddings.f32.meta.json",
            out_dir / "visual_failures.parquet",
        ):
            if p.exists():
                p.unlink()
        store = EmbeddingStore(
            mmap_path=out_dir / "visual_embeddings.f32.mmap",
            index_path=out_dir / "visual_index.parquet",
            dim=dim,
        )
        done = set()

    meta_path = out_dir / "visual_run_meta.json"
    pending = man[~man["image_id"].astype(str).isin(done)].to_dict("records")
    limit = cfg.get("pipeline", {}).get("limit")
    if limit is not None:
        pending = pending[: max(0, int(limit) - len(done))]

    n_workers = _resolve_visual_data_parallel(cfg)
    print(f"[visual] pending={len(pending)} done={len(done)} dim={dim} data_parallel={n_workers}", flush=True)
    if not pending:
        return store._index

    if n_workers == 1:
        shard_dir = out_dir / "visual_shards" / "w0"
        meta = _run_visual_pending_on_device(cfg, pending, fp, shard_dir, worker_tag="w0")
        results = [meta]
    else:
        import multiprocessing as mp

        shards: list[list[dict[str, Any]]] = [[] for _ in range(n_workers)]
        for i, row in enumerate(pending):
            shards[i % n_workers].append(row)
        shard_root = out_dir / "visual_shards"
        ensure_dir(shard_root)
        payloads = []
        for wid, shard in enumerate(shards):
            if not shard:
                continue
            payloads.append(
                {
                    "worker_id": wid,
                    "gpu_id": wid,
                    "cfg": cfg,
                    "pending": shard,
                    "fp": fp,
                    "shard_dir": str(shard_root / f"w{wid}"),
                }
            )
        ctx = mp.get_context("spawn")
        print(f"[visual] launching {len(payloads)} non-daemon GPU workers (spawn)", flush=True)
        procs = []
        result_paths = []
        for payload in payloads:
            rp = shard_root / f"result.w{payload['worker_id']}.json"
            if rp.exists():
                rp.unlink()
            result_paths.append(rp)
            p = ctx.Process(
                target=_visual_dp_entry,
                args=(payload, str(rp)),
                daemon=False,
                name=f"visual_gpu{payload['gpu_id']}",
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        results = []
        for rp, p in zip(result_paths, procs):
            if rp.exists():
                results.append(json.loads(rp.read_text(encoding="utf-8")))
            else:
                results.append({"ok": False, "error": f"no result exit={p.exitcode}", "n_ok": 0})

    # Merge shards into EmbeddingStore (single-writer)
    fail_rows = []
    selected_layer = cfg["model"].get("selected_layer")
    num_layers = None
    for meta in results:
        status = "OK" if meta.get("ok") else f"FAIL {meta.get('error')}"
        print(
            f"[visual] worker={meta.get('worker_id')} gpu={meta.get('gpu_id')} "
            f"ok={meta.get('n_ok')} fail={meta.get('n_fail')} {status}",
            flush=True,
        )
        if not meta.get("ok"):
            continue
        if meta.get("selected_layer") is not None:
            selected_layer = meta["selected_layer"]
        if meta.get("num_layers") is not None:
            num_layers = meta["num_layers"]
        idx_p = Path(meta["index_path"])
        vec_p = Path(meta["vec_path"])
        if not idx_p.exists() or not vec_p.exists():
            continue
        idx_df = pd.read_parquet(idx_p)
        if len(idx_df) == 0:
            continue
        vecs = np.load(vec_p)
        if store.dim != int(meta.get("dim") or store.dim) and len(store.done_ids) == 0:
            store = EmbeddingStore(
                mmap_path=out_dir / "visual_embeddings.f32.mmap",
                index_path=out_dir / "visual_index.parquet",
                dim=int(meta["dim"]),
            )
        store.append_many(idx_df.to_dict("records"), vecs)
        if meta.get("fail_path") and Path(meta["fail_path"]).exists():
            fail_rows.append(pd.read_parquet(meta["fail_path"]))

    if fail_rows:
        fail_df = pd.concat(fail_rows, ignore_index=True).drop_duplicates("image_id", keep="last")
        fail_path = out_dir / "visual_failures.parquet"
        if fail_path.exists() and resume:
            oldf = pd.read_parquet(fail_path)
            fail_df = pd.concat([oldf, fail_df], ignore_index=True).drop_duplicates("image_id", keep="last")
        fail_df.to_parquet(fail_path, index=False)

    failed_workers = [r for r in results if not r.get("ok")]
    if len(failed_workers) == len(results):
        raise RuntimeError(f"all visual GPU workers failed: {failed_workers[0] if failed_workers else None}")

    atomic_write_json(
        meta_path,
        {
            "config_fingerprint": fp,
            "selected_layer": selected_layer,
            "num_layers": num_layers,
            "embedding_dim": store.dim,
            "n_embeddings": len(store.done_ids),
            "data_parallel": n_workers,
        },
    )
    return store._index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    df = run_visual_features(cfg)
    print(f"[visual_feature_runner] n={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
