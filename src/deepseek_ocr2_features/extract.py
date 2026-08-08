"""FP32 pooling and single-page / batched page feature extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .encoder import CausalFlowVisionEncoder, NonFiniteError, assert_finite
from .preprocess import PageViewMeta, load_image_rgb, prepare_page_views


def pool_global_local(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mean-pool Projector-before tokens in FP32.

    Args:
      global_tokens: [1, 256, 896] or [256, 896]
      local_tokens:  [m, 144, 896]

    Returns:
      g: [896], ell: [896], concat: [1792]  (all float32, CPU)
    """
    g_tok = global_tokens.float()
    if g_tok.ndim == 2:
        g_tok = g_tok.unsqueeze(0)
    if g_tok.shape != (1, 256, 896):
        raise RuntimeError(f"global tokens expected [1,256,896], got {tuple(g_tok.shape)}")

    l_tok = local_tokens.float()
    if l_tok.ndim != 3 or l_tok.shape[1:] != (144, 896):
        raise RuntimeError(f"local tokens expected [m,144,896], got {tuple(l_tok.shape)}")

    g = g_tok.mean(dim=1).squeeze(0)  # [896]
    # Two-level mean == direct mean over all local tokens
    ell = l_tok.mean(dim=(0, 1))  # [896]
    assert_finite(g, "global_descriptor")
    assert_finite(ell, "local_descriptor")
    concat = torch.cat([g, ell], dim=0)
    assert_finite(concat, "concat_descriptor")
    if concat.shape != (1792,):
        raise RuntimeError(f"concat expected [1792], got {tuple(concat.shape)}")
    return g.cpu(), ell.cpu(), concat.cpu()


def verify_pooling_equivalence(local_tokens: torch.Tensor, atol: float = 1e-5) -> bool:
    a = local_tokens.float().mean(dim=(0, 1))
    b = local_tokens.float().mean(dim=1).mean(dim=0)
    return bool(torch.allclose(a, b, atol=atol, rtol=1e-4))


@torch.inference_mode()
def extract_page_features(
    encoder: CausalFlowVisionEncoder,
    image_path: str,
    *,
    sample_id: str = "",
    retry_on_nonfinite: bool = True,
) -> dict[str, Any]:
    """Extract one page; optionally retry once on NonFiniteError."""
    try:
        return _extract_once(encoder, image_path, sample_id=sample_id)
    except NonFiniteError as e:
        if not retry_on_nonfinite:
            raise
        print(f"[extract] non-finite retry sample_id={sample_id} stage={e.stage}")
        return _extract_once(encoder, image_path, sample_id=sample_id)


def _extract_once(
    encoder: CausalFlowVisionEncoder,
    image_path: str,
    *,
    sample_id: str,
) -> dict[str, Any]:
    image = load_image_rgb(image_path)
    global_t, local_t, meta = prepare_page_views(image)
    _, g_tokens = encoder.encode_views(
        global_t,
        expected_tokens=256,
        stage_prefix="global",
        sample_id=sample_id,
        image_path=image_path,
    )
    _, l_tokens = encoder.encode_views(
        local_t,
        expected_tokens=144,
        stage_prefix="local",
        sample_id=sample_id,
        image_path=image_path,
    )
    g, ell, concat = pool_global_local(g_tokens, l_tokens)
    return {
        "global": g.numpy().astype(np.float32, copy=False),
        "local": ell.numpy().astype(np.float32, copy=False),
        "concat": concat.numpy().astype(np.float32, copy=False),
        "meta": meta,
        "shapes": {
            "global_tokens": tuple(g_tokens.shape),
            "local_tokens": tuple(l_tokens.shape),
            "global_descriptor": tuple(g.shape),
            "local_descriptor": tuple(ell.shape),
            "concat": tuple(concat.shape),
        },
        "projector_used": False,
        "language_decoder_used": False,
    }


@torch.inference_mode()
def extract_pages_batched(
    encoder: CausalFlowVisionEncoder,
    image_paths: list[str],
    sample_ids: list[str],
    *,
    max_local_batch: int = 8,
) -> list[dict[str, Any]]:
    """
    Batch globals across pages; encode local crops in chunks to avoid OOM.

    Pages may have different local crop counts; locals are concatenated then split.
    """
    if len(image_paths) != len(sample_ids):
        raise ValueError("image_paths and sample_ids length mismatch")
    if not image_paths:
        return []

    prepared: list[tuple[torch.Tensor, torch.Tensor, PageViewMeta, str, str]] = []
    for path, sid in zip(image_paths, sample_ids):
        img = load_image_rgb(path)
        g, loc, meta = prepare_page_views(img)
        prepared.append((g, loc, meta, sid, path))

    globals_b = torch.cat([p[0] for p in prepared], dim=0)  # [B,3,1024,1024]
    local_counts = [int(p[1].shape[0]) for p in prepared]
    locals_b = torch.cat([p[1] for p in prepared], dim=0)  # [sum_m,3,768,768]

    try:
        _, g_all = encoder.encode_views(
            globals_b, expected_tokens=256, stage_prefix="global_batch"
        )
        # Chunk local crops so page_batch_size*max_crops does not OOM A10.
        l_parts: list[torch.Tensor] = []
        for start in range(0, locals_b.shape[0], max_local_batch):
            chunk = locals_b[start : start + max_local_batch]
            _, l_tok = encoder.encode_views(
                chunk, expected_tokens=144, stage_prefix="local_batch"
            )
            l_parts.append(l_tok)
        l_all = torch.cat(l_parts, dim=0)
    except NonFiniteError:
        return [
            extract_page_features(encoder, path, sample_id=sid, retry_on_nonfinite=True)
            for path, sid in zip(image_paths, sample_ids)
        ]

    outs: list[dict[str, Any]] = []
    offset = 0
    for i, (g_t, loc_t, meta, sid, path) in enumerate(prepared):
        m = local_counts[i]
        g_tok = g_all[i : i + 1]
        l_tok = l_all[offset : offset + m]
        offset += m
        try:
            g, ell, concat = pool_global_local(g_tok, l_tok)
        except NonFiniteError:
            outs.append(
                extract_page_features(encoder, path, sample_id=sid, retry_on_nonfinite=True)
            )
            continue
        outs.append(
            {
                "global": g.numpy().astype(np.float32, copy=False),
                "local": ell.numpy().astype(np.float32, copy=False),
                "concat": concat.numpy().astype(np.float32, copy=False),
                "meta": meta,
                "shapes": {
                    "global_tokens": tuple(g_tok.shape),
                    "local_tokens": tuple(l_tok.shape),
                    "global_descriptor": tuple(g.shape),
                    "local_descriptor": tuple(ell.shape),
                    "concat": tuple(concat.shape),
                },
                "projector_used": False,
                "language_decoder_used": False,
            }
        )
    return outs
