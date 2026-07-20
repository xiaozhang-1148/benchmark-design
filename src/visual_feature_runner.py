"""Channel B: Transformers BF16 visual-layer pooled embedding extraction."""

from __future__ import annotations

import argparse
import json
import math
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
from .utils import ensure_dir, l2_normalize, load_image_rgb


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


def run_visual_features(cfg: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(cfg["paths"]["outputs_dir"])
    man = pd.read_parquet(out_dir / "manifest.parquet")
    man = man[man["status"] != "corrupt"].copy()
    fp = fingerprint_config(cfg)

    # Prefer images that already have OCR ok, else all
    ocr_path = out_dir / "ocr_generations.parquet"
    if ocr_path.exists():
        ocr = pd.read_parquet(ocr_path)
        ok_ids = set(ocr.loc[ocr["status"] == "ok", "image_id"].astype(str))
        if ok_ids:
            man = man[man["image_id"].astype(str).isin(ok_ids)]

    # Load introspection for selected layer
    intro = Path(cfg["paths"]["reports_dir"]) / "model_introspection.json"
    if intro.exists():
        info = json.loads(intro.read_text())
        if cfg["model"].get("selected_layer") is None:
            cfg["model"]["selected_layer"] = info.get("selected_layer_index")

    # Dim unknown until first sample — probe from introspection pooled shape or 896
    dim = 896
    if intro.exists():
        shape = json.loads(intro.read_text()).get("selected_layer_output_shape")
        if shape and len(shape) >= 3:
            dim = int(shape[-1])

    store = EmbeddingStore(
        mmap_path=out_dir / "visual_embeddings.f32.mmap",
        index_path=out_dir / "visual_index.parquet",
        dim=dim,
    )
    done = store.done_ids if cfg.get("pipeline", {}).get("resume", True) else set()
    # Also check fingerprint in a side table
    meta_path = out_dir / "visual_run_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("config_fingerprint") != fp:
            print("[visual] config fingerprint changed; not deleting old store, but will rewrite overlapping ids")
            # For safety we skip only exact done ids; user can delete outputs to fully invalidate

    pending = man[~man["image_id"].astype(str).isin(done)].to_dict("records")
    limit = cfg.get("pipeline", {}).get("limit")
    if limit is not None:
        pending = pending[: max(0, int(limit) - len(done))]

    print(f"[visual] pending={len(pending)} done={len(done)} dim={dim}")
    if not pending:
        return store._index

    extractor = VisualLayerExtractor(cfg)
    # If actual dim differs, rebuild store
    if extractor.hidden_dim and extractor.hidden_dim != dim:
        # will be set after first embed
        pass

    buf_rows = []
    buf_vecs = []
    flush_every = 32
    for r in tqdm(pending, desc="visual_embed"):
        iid = str(r["image_id"])
        try:
            img = load_image_rgb(r["absolute_path"])
            out = extractor.embed_image(img)
            if out["embedding_dim"] != store.dim:
                # Recreate store with correct dim if empty
                if len(store.done_ids) == 0 and not buf_rows:
                    store = EmbeddingStore(
                        mmap_path=out_dir / "visual_embeddings.f32.mmap",
                        index_path=out_dir / "visual_index.parquet",
                        dim=out["embedding_dim"],
                    )
                else:
                    raise RuntimeError(
                        f"embedding dim mismatch: got {out['embedding_dim']} expected {store.dim}"
                    )
            buf_rows.append(
                {
                    "image_id": iid,
                    "selected_layer": out["selected_layer"],
                    "token_count": out["token_count"],
                    "embedding_norm_before_normalization": out["norm_before"],
                    "config_fingerprint": fp,
                }
            )
            buf_vecs.append(out["embedding"])
        except Exception as e:  # noqa: BLE001
            # record failure in a sidecar; do not stop
            fail_path = out_dir / "visual_failures.parquet"
            fail_row = pd.DataFrame(
                [{"image_id": iid, "error_message": f"{type(e).__name__}: {e}", "config_fingerprint": fp}]
            )
            if fail_path.exists():
                oldf = pd.read_parquet(fail_path)
                fail_row = pd.concat([oldf, fail_row], ignore_index=True).drop_duplicates("image_id", keep="last")
            fail_row.to_parquet(fail_path, index=False)
            continue

        if len(buf_rows) >= flush_every:
            store.append_many(buf_rows, np.stack(buf_vecs, axis=0))
            buf_rows, buf_vecs = [], []

    if buf_rows:
        store.append_many(buf_rows, np.stack(buf_vecs, axis=0))

    from .utils import atomic_write_json

    atomic_write_json(
        meta_path,
        {
            "config_fingerprint": fp,
            "selected_layer": extractor.selected_layer,
            "num_layers": extractor.num_layers,
            "embedding_dim": store.dim,
            "n_embeddings": len(store.done_ids),
        },
    )
    del extractor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
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
