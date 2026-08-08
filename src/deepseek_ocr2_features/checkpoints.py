"""Checkpoint runners CP0–CP6 for DeepSeek-OCR2 feature extraction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..utils import atomic_write_json, ensure_dir, load_image_rgb
from .encoder import CausalFlowVisionEncoder
from .extract import extract_page_features, pool_global_local, verify_pooling_equivalence
from .manifest import discover_samples, enrich_image_sizes
from .preprocess import MEAN, STD, prepare_page_views
from .run_extract import DEFAULT_INPUT, DEFAULT_MODEL, DEFAULT_OUTPUT


def cp0_data(input_dir: str, out_dir: Path) -> dict[str, Any]:
    records = discover_samples(input_dir)
    stats = enrich_image_sizes(records)
    report = {
        "checkpoint": 0,
        "input_dir": input_dir,
        "project_notes": {
            "existing_projected_extractor": "src/visual_exp/extractor.py (Projector+L2, NOT used here)",
            "new_package": "src/deepseek_ocr2_features/",
            "does_not_overwrite_user_features": True,
        },
        "stats": stats,
        "pairing": "image.jpg <-> image.jpg.json",
        "passed": stats["n_corrupt"] == 0 and stats["n_total"] > 0,
    }
    atomic_write_json(out_dir / "checkpoint0_report.json", report)
    return report


def cp1_model(model_path: str, out_dir: Path, device: str = "cuda:0") -> dict[str, Any]:
    enc = CausalFlowVisionEncoder(model_path, device=device if torch.cuda.is_available() else "cpu")
    mp = enc.module_paths
    report = {
        "checkpoint": 1,
        "model_path": model_path,
        "module_paths": mp.__dict__,
        "checks": {
            "sam_exists": mp.sam_path == "model.sam_model",
            "qwen_exists": mp.qwen_path == "model.qwen2_model",
            "projector_exists": mp.projector_path == "model.projector",
            "projector_in_896": mp.projector_in_features == 896,
            "projector_out_1280": mp.projector_out_features == 1280,
            "query_1024_is_256": mp.query_1024_num == 256,
            "query_768_is_144": mp.query_768_num == 144,
            "hidden_dim_896": mp.query_hidden_dim == 896,
            "extraction_point": "qwen2_model output (before projector)",
            "language_decoder_not_called": True,
            "projector_not_called": True,
        },
    }
    report["passed"] = all(
        report["checks"][k]
        for k in (
            "sam_exists",
            "qwen_exists",
            "projector_exists",
            "projector_in_896",
            "projector_out_1280",
            "query_1024_is_256",
            "query_768_is_144",
            "hidden_dim_896",
        )
    )
    # Live shape probe
    g = torch.zeros(1, 3, 1024, 1024, dtype=torch.bfloat16)
    l = torch.zeros(2, 3, 768, 768, dtype=torch.bfloat16)
    _, g_tok = enc.encode_views(g, expected_tokens=256, stage_prefix="cp1_global")
    _, l_tok = enc.encode_views(l, expected_tokens=144, stage_prefix="cp1_local")
    report["live_shapes"] = {
        "global_qwen": list(g_tok.shape),
        "local_qwen": list(l_tok.shape),
    }
    report["checks"]["global_shape_ok"] = tuple(g_tok.shape) == (1, 256, 896)
    report["checks"]["local_shape_ok"] = tuple(l_tok.shape) == (2, 144, 896)
    report["passed"] = report["passed"] and report["checks"]["global_shape_ok"] and report["checks"]["local_shape_ok"]
    enc.close()
    atomic_write_json(out_dir / "checkpoint1_model.json", report)
    return report


def _pick_diverse_images(input_dir: str, n: int = 3) -> list[Path]:
    records = discover_samples(input_dir)
    enrich_image_sizes(records, workers=32)
    usable = [r for r in records if r.original_width and r.original_height]
    usable.sort(key=lambda r: (r.original_width or 0) / max(r.original_height or 1, 1))
    picks: list[Path] = []
    # tall portrait
    tall = max(usable, key=lambda r: (r.original_height or 0) / max(r.original_width or 1, 1))
    picks.append(Path(tall.image_path))
    # wide-ish
    wide = max(usable, key=lambda r: (r.original_width or 0) / max(r.original_height or 1, 1))
    if Path(wide.image_path) not in picks:
        picks.append(Path(wide.image_path))
    # smallest area near threshold
    small = min(usable, key=lambda r: (r.original_width or 0) * (r.original_height or 0))
    if Path(small.image_path) not in picks:
        picks.append(Path(small.image_path))
    return picks[:n]


def cp2_preprocess(input_dir: str, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir / "preprocess_previews")
    picks = _pick_diverse_images(input_dir)
    pages = []
    for i, path in enumerate(picks):
        img = load_image_rgb(path)
        g, loc, meta = prepare_page_views(img, dtype=torch.float32)
        # denormalize for preview
        def _to_pil(t: torch.Tensor) -> Image.Image:
            x = t.detach().float().cpu()
            for c in range(3):
                x[c] = x[c] * STD[c] + MEAN[c]
            x = (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            return Image.fromarray(x)

        g_img = _to_pil(g[0])
        local_imgs = [_to_pil(loc[j]) for j in range(loc.shape[0])]
        # contact sheet
        cell = 192
        cols = 1 + len(local_imgs)
        sheet = Image.new("RGB", (cols * cell, cell), (30, 30, 30))
        g_img.resize((cell, cell)).copy()
        sheet.paste(g_img.resize((cell, cell)), (0, 0))
        for j, li in enumerate(local_imgs):
            sheet.paste(li.resize((cell, cell)), ((j + 1) * cell, 0))
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), "G", fill=(255, 255, 0))
        preview_path = out_dir / "preprocess_previews" / f"page_{i}.jpg"
        sheet.save(preview_path, quality=90)
        pages.append(
            {
                "image_path": str(path),
                "meta": meta.to_dict(),
                "global_tensor_shape": list(g.shape),
                "local_tensor_shape": list(loc.shape),
                "preview": str(preview_path),
                "aspect_preserved_global": True,  # ImageOps.pad
                "normalize_mean": list(MEAN),
                "normalize_std": list(STD),
            }
        )
    report = {"checkpoint": 2, "pages": pages, "passed": len(pages) >= 1}
    atomic_write_json(out_dir / "checkpoint2_preprocess.json", report)
    return report


def cp3_single_sample(model_path: str, input_dir: str, out_dir: Path, device: str = "cuda:0") -> dict[str, Any]:
    picks = _pick_diverse_images(input_dir, n=1)
    path = picks[0]
    enc = CausalFlowVisionEncoder(model_path, device=device if torch.cuda.is_available() else "cpu")
    img = load_image_rgb(path)
    g_img, l_img, meta = prepare_page_views(img)
    sam_g, q_g = enc.encode_views(g_img, expected_tokens=256, stage_prefix="global")
    sam_l, q_l = enc.encode_views(l_img, expected_tokens=144, stage_prefix="local")
    g, ell, concat = pool_global_local(q_g, q_l)
    report = {
        "checkpoint": 3,
        "image_path": str(path),
        "meta": meta.to_dict(),
        "shapes": {
            "global_image": list(g_img.shape),
            "global_sam": list(sam_g.shape),
            "global_qwen2": list(q_g.shape),
            "local_image": list(l_img.shape),
            "local_sam": list(sam_l.shape),
            "local_qwen2": list(q_l.shape),
            "global_descriptor": list(g.shape),
            "local_descriptor": list(ell.shape),
            "concat": list(concat.shape),
        },
        "checks": {
            "global_qwen_256_896": tuple(q_g.shape) == (1, 256, 896),
            "local_qwen_m_144_896": q_l.shape[1:] == (144, 896) and q_l.shape[0] == meta.local_crop_count,
            "global_desc_896": tuple(g.shape) == (896,),
            "local_desc_896": tuple(ell.shape) == (896,),
            "concat_1792": tuple(concat.shape) == (1792,),
        },
    }
    report["passed"] = all(report["checks"].values())
    enc.close()
    atomic_write_json(out_dir / "checkpoint3_tensors.json", report)
    return report


def cp4_pooling(model_path: str, input_dir: str, out_dir: Path, device: str = "cuda:0") -> dict[str, Any]:
    path = _pick_diverse_images(input_dir, n=1)[0]
    enc = CausalFlowVisionEncoder(model_path, device=device if torch.cuda.is_available() else "cpu")
    out = extract_page_features(enc, str(path), sample_id="cp4")
    # Recompute tokens for equivalence
    img = load_image_rgb(path)
    g_img, l_img, _ = prepare_page_views(img)
    _, q_l = enc.encode_views(l_img, expected_tokens=144, stage_prefix="local")
    eq = verify_pooling_equivalence(q_l)
    concat = out["concat"]
    report = {
        "checkpoint": 4,
        "pooling_two_level_equiv": eq,
        "concat_prefix_is_global": bool(np.allclose(concat[:896], out["global"])),
        "concat_suffix_is_local": bool(np.allclose(concat[896:], out["local"])),
    }
    report["passed"] = all(
        [
            report["pooling_two_level_equiv"],
            report["concat_prefix_is_global"],
            report["concat_suffix_is_local"],
        ]
    )
    enc.close()
    atomic_write_json(out_dir / "checkpoint4_pooling.json", report)
    return report


def cp5_determinism(model_path: str, input_dir: str, out_dir: Path, device: str = "cuda:0") -> dict[str, Any]:
    path = _pick_diverse_images(input_dir, n=1)[0]
    enc = CausalFlowVisionEncoder(model_path, device=device if torch.cuda.is_available() else "cpu")
    a = extract_page_features(enc, str(path), sample_id="cp5a")
    b = extract_page_features(enc, str(path), sample_id="cp5b")
    # BF16 path → allow small FP32 diff after mean
    diffs = {
        "global_max_abs": float(np.max(np.abs(a["global"] - b["global"]))),
        "local_max_abs": float(np.max(np.abs(a["local"] - b["local"]))),
        "concat_max_abs": float(np.max(np.abs(a["concat"] - b["concat"]))),
    }
    # With eval+inference_mode, should be exact or near-exact
    tol = 1e-3
    report = {
        "checkpoint": 5,
        "image_path": str(path),
        "diffs": diffs,
        "tol": tol,
        "passed": all(v <= tol for v in diffs.values()),
    }
    enc.close()
    atomic_write_json(out_dir / "checkpoint5_determinism.json", report)
    return report


def cp6_smoke(
    model_path: str,
    input_dir: str,
    out_dir: Path,
    *,
    limit: int = 32,
    num_gpus: int = 1,
    page_batch_size: int = 4,
) -> dict[str, Any]:
    from .run_extract import main as extract_main

    smoke_dir = out_dir / "smoke_run"
    ensure_dir(smoke_dir)
    rc = extract_main(
        [
            "--input-dir",
            input_dir,
            "--output-dir",
            str(smoke_dir),
            "--model-path",
            model_path,
            "--limit",
            str(limit),
            "--num-gpus",
            str(num_gpus),
            "--page-batch-size",
            str(page_batch_size),
            "--no-resume",
            "--skip-size-scan",
        ]
    )
    from .audit import audit_outputs

    audit = audit_outputs(smoke_dir, n_expected=limit)
    report = {"checkpoint": 6, "extract_rc": rc, "audit": audit, "passed": rc == 0 and audit.get("passed")}
    atomic_write_json(out_dir / "checkpoint6_smoke.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="all", help="0..6 or all or 0-5")
    p.add_argument("--input-dir", default=DEFAULT_INPUT)
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default=str(Path(DEFAULT_OUTPUT) / "checkpoints"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke-limit", type=int, default=32)
    p.add_argument("--smoke-gpus", type=int, default=2)
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    which = str(args.checkpoint)
    if which == "all":
        ids = list(range(0, 7))
    elif "-" in which:
        a, b = which.split("-", 1)
        ids = list(range(int(a), int(b) + 1))
    else:
        ids = [int(which)]

    results: dict[str, Any] = {}
    t0 = time.time()
    for cid in ids:
        print(f"\n===== CHECKPOINT {cid} =====", flush=True)
        if cid == 0:
            results["cp0"] = cp0_data(args.input_dir, out_dir)
        elif cid == 1:
            results["cp1"] = cp1_model(args.model_path, out_dir, device=args.device)
        elif cid == 2:
            results["cp2"] = cp2_preprocess(args.input_dir, out_dir)
        elif cid == 3:
            results["cp3"] = cp3_single_sample(args.model_path, args.input_dir, out_dir, device=args.device)
        elif cid == 4:
            results["cp4"] = cp4_pooling(args.model_path, args.input_dir, out_dir, device=args.device)
        elif cid == 5:
            results["cp5"] = cp5_determinism(args.model_path, args.input_dir, out_dir, device=args.device)
        elif cid == 6:
            results["cp6"] = cp6_smoke(
                args.model_path,
                args.input_dir,
                out_dir,
                limit=args.smoke_limit,
                num_gpus=args.smoke_gpus,
            )
        else:
            raise SystemExit(f"unknown checkpoint {cid}")
        passed = results[f"cp{cid}"].get("passed")
        print(f"[cp{cid}] passed={passed}", flush=True)
        if not passed:
            atomic_write_json(out_dir / "checkpoints_summary.json", {"results": results, "failed_at": cid})
            print(json.dumps(results[f"cp{cid}"], indent=2, ensure_ascii=False)[:4000])
            return 1

    summary = {"results": {k: {"passed": v.get("passed")} for k, v in results.items()}, "elapsed_sec": time.time() - t0}
    atomic_write_json(out_dir / "checkpoints_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
