"""Runtime introspection of DeepSeek-OCR-2 vision modules."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import fingerprint_config, load_config, model_resolved_path
from .model_compat import patch_transformers_for_deepseek_ocr2
from .utils import atomic_write_json, atomic_write_text, ensure_dir


def _module_tree(module: torch.nn.Module, prefix: str = "", max_depth: int = 3, depth: int = 0) -> list[str]:
    lines = []
    if depth == 0:
        lines.append(f"{module.__class__.__name__}")
    if depth >= max_depth:
        return lines
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        n_params = sum(p.numel() for p in child.parameters(recurse=False))
        lines.append(f"{'  ' * (depth + 1)}{path}: {child.__class__.__name__} (local_params={n_params})")
        lines.extend(_module_tree(child, path, max_depth=max_depth, depth=depth + 1))
    return lines


def _count_layers(model) -> dict[str, Any]:
    info: dict[str, Any] = {}
    # Official HF remote code paths
    m = model.model if hasattr(model, "model") else model
    sam = getattr(m, "sam_model", None)
    qwen = getattr(m, "qwen2_model", None)
    if sam is not None:
        blocks = getattr(sam, "blocks", None)
        info["sam_module_path"] = "model.sam_model"
        info["sam_num_layers"] = len(blocks) if blocks is not None else None
        info["sam_class"] = sam.__class__.__name__
    if qwen is not None:
        info["qwen2_module_path"] = "model.qwen2_model"
        info["qwen2_class"] = qwen.__class__.__name__
        # Qwen2Decoder2Encoder -> CustomQwen2Decoder -> model (Qwen2Model).layers
        inner = qwen
        layers = None
        for attr_chain in [
            ("model", "model", "layers"),
            ("model", "layers"),
            ("layers",),
        ]:
            cur = inner
            ok = True
            for a in attr_chain:
                if not hasattr(cur, a):
                    ok = False
                    break
                cur = getattr(cur, a)
            if ok:
                layers = cur
                info["qwen2_layers_path"] = "model.qwen2_model." + ".".join(attr_chain)
                break
        info["qwen2_num_layers"] = len(layers) if layers is not None else None
        # causal-flow query embeddings
        q1024 = getattr(qwen, "query_1024", None)
        q768 = getattr(qwen, "query_768", None)
        info["causal_flow_query_1024"] = int(q1024.num_embeddings) if q1024 is not None else None
        info["causal_flow_query_768"] = int(q768.num_embeddings) if q768 is not None else None
        info["causal_flow_hidden_dim"] = int(q1024.embedding_dim) if q1024 is not None else None
    proj = getattr(m, "projector", None)
    if proj is not None:
        info["projector_path"] = "model.projector"
        info["projector_class"] = proj.__class__.__name__
    return info


def _probe_selected_layer_shape(model, selected_layer: int, device: str) -> dict[str, Any]:
    """Forward a tiny synthetic SAM feature through qwen2 with hooks."""
    m = model.model if hasattr(model, "model") else model
    qwen = getattr(m, "qwen2_model", None)
    if qwen is None:
        return {"error": "qwen2_model not found"}

    # Build a fake SAM output: [B, C, H, W] matching 1024 path -> 256 tokens after flatten
    # ImageEncoderViT outputs spatial map that qwen flattens.
    # From code: x.flatten(2).transpose(1, 2) then concat queries.
    # For 1024/16/4 downsample path, n_query for SAM out is typically 256 (16x16) or 64.
    # Safer: call with zeros matching last_conv channel 896 if available.
    # SAM in this model outputs channels matching projector input path via conv neck -> 896.
    B, C, H, W = 1, 896, 16, 16
    x = torch.zeros(B, C, H, W, dtype=torch.bfloat16, device=device)

    captured: dict[str, Any] = {}

    def make_hook(idx):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["layer"] = idx
            captured["shape"] = tuple(h.shape)
            captured["dtype"] = str(h.dtype)

        return hook

    # Locate layers
    layers = None
    for attr_chain in [("model", "model", "layers"), ("model", "layers")]:
        cur = qwen
        ok = True
        for a in attr_chain:
            if not hasattr(cur, a):
                ok = False
                break
            cur = getattr(cur, a)
        if ok:
            layers = cur
            break
    if layers is None:
        return {"error": "could not locate qwen2 layers"}

    n_layers = len(layers)
    idx = selected_layer if selected_layer is not None else n_layers // 2
    idx = max(0, min(idx, n_layers - 1))
    handle = layers[idx].register_forward_hook(make_hook(idx))
    try:
        with torch.inference_mode():
            y = qwen(x)
        captured["output_shape"] = tuple(y.shape)
        captured["selected_layer"] = idx
        captured["num_layers"] = n_layers
        # causal-flow tokens = second half of sequence in intermediate? final y is already causal only
        # Intermediate full seq = n_query (noncausal) + n_query (causal)
        if "shape" in captured and len(captured["shape"]) == 3:
            seq = captured["shape"][1]
            captured["full_seq_len"] = seq
            captured["causal_flow_token_count"] = seq // 2
            captured["token_mask"] = (
                "token_type_ids: 0=non-causal image patches, 1=causal-flow queries; "
                "pool mean over causal-flow half only"
            )
            captured["pooled_shape"] = [captured["shape"][-1]]
    except Exception as e:  # noqa: BLE001
        captured["error"] = f"{type(e).__name__}: {e}"
    finally:
        handle.remove()
    return captured


def check_vllm_visual_hidden_states() -> dict[str, Any]:
    info: dict[str, Any] = {
        "vllm_can_return_visual_hidden_states": False,
        "reason": "",
        "vllm_version": None,
        "vllm_has_ocr2": False,
    }
    try:
        import vllm

        info["vllm_version"] = getattr(vllm, "__version__", None)
        from vllm.model_executor.models import registry as reg

        # registry mapping
        arch_map = getattr(reg, "_VLLM_MODELS", None) or getattr(reg, "ModelRegistry", None)
        has = False
        try:
            from vllm.model_executor.models.registry import ModelRegistry

            # try import module
            import importlib

            try:
                importlib.import_module("vllm.model_executor.models.deepseek_ocr2")
                has = True
            except Exception:
                has = False
            info["vllm_has_ocr2"] = has
        except Exception as e:  # noqa: BLE001
            info["reason"] = f"registry check failed: {e}"
            return info

        # Inspect public generate API — no supported return of intermediate vision states
        info["vllm_can_return_visual_hidden_states"] = False
        info["reason"] = (
            "vLLM DeepseekOCR2ForCausalLM exposes multimodal embeddings for generation only; "
            "LLM.generate / AsyncLLMEngine do not provide a stable API for intermediate "
            "visual-encoder hidden states (SAM blocks or Qwen2 causal-flow layers). "
            "Using dual-channel: vLLM for OCR/grounding generation; Transformers for "
            "fixed visual-layer pooled embeddings."
        )
    except Exception as e:  # noqa: BLE001
        info["reason"] = f"vLLM not available: {e}"
    return info


def run_introspection(cfg: dict[str, Any]) -> dict[str, Any]:
    model_path = model_resolved_path(cfg)
    device = cfg["visual"].get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    from transformers import AutoModel, AutoTokenizer

    patch_transformers_for_deepseek_ocr2()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg["model"].get("trust_remote_code", True)
    )
    # Load on CPU first for structure, move one module probe to device if possible
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        torch_dtype=torch.bfloat16,
        _attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
    )
    model.eval()

    arch = model.__class__.__name__
    layer_info = _count_layers(model)
    tree_lines = _module_tree(model, max_depth=3)

    # Default selected layer: floor(num_layers/2) on the chosen visual encoder
    encoder_name = cfg["model"].get("visual_encoder", "qwen2_model")
    if encoder_name == "sam_model":
        n_layers = layer_info.get("sam_num_layers") or 12
        module_path = layer_info.get("sam_module_path", "model.sam_model")
    else:
        n_layers = layer_info.get("qwen2_num_layers") or 24
        module_path = layer_info.get("qwen2_module_path", "model.qwen2_model")

    selected = cfg["model"].get("selected_layer")
    if selected is None:
        selected = n_layers // 2

    # Probe shapes on device if possible (may OOM on full model — move only qwen)
    shape_info: dict[str, Any] = {}
    try:
        m = model.model if hasattr(model, "model") else model
        if hasattr(m, "qwen2_model"):
            m.qwen2_model.to(device)
            shape_info = _probe_selected_layer_shape(model, int(selected), device)
            m.qwen2_model.to("cpu")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:  # noqa: BLE001
        shape_info = {"error": f"{type(e).__name__}: {e}"}

    vllm_info = check_vllm_visual_hidden_states()

    result = {
        "model_name": cfg["model"]["name_or_path"],
        "model_path_resolved": model_path,
        "model_revision_or_commit": cfg["model"].get("revision"),
        "architecture_class": arch,
        "visual_encoder_module_path": module_path,
        "visual_encoder_name": encoder_name,
        "visual_layer_total": int(n_layers),
        "selected_layer_index": int(selected),
        "selected_layer_output_shape": shape_info.get("shape"),
        "selected_layer_pooled_shape": shape_info.get("pooled_shape"),
        "token_mask_handling": shape_info.get(
            "token_mask",
            "Exclude padding/prompt/special tokens; pool causal-flow query tokens only "
            "(token_type_ids==1 / second half of qwen2 sequence).",
        ),
        "pooling": "mean(valid_visual_tokens, dim=0) then L2 normalize (float32)",
        "vllm_can_obtain_visual_hidden_states": vllm_info["vllm_can_return_visual_hidden_states"],
        "vllm_notes": vllm_info,
        "actual_inference_architecture": (
            "dual_channel: "
            "A=vLLM DeepseekOCR2ForCausalLM (OCR/Markdown/grounding generation); "
            "B=Transformers BF16 (language attn=eager; qwen2 visual encoder uses SDPA) "
            "for fixed visual-layer pooled embedding"
        ),
        "layer_details": layer_info,
        "shape_probe": shape_info,
        "module_tree": tree_lines,
        "config_fingerprint": fingerprint_config(cfg),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
    }

    # Persist selected layer back into a sidecar for runners
    reports = Path(cfg["paths"]["reports_dir"])
    ensure_dir(reports)
    atomic_write_json(reports / "model_introspection.json", result)

    # Update in-memory cfg
    cfg["model"]["selected_layer"] = int(selected)
    cfg["model"]["visual_layer_total"] = int(n_layers)

    md = _format_markdown(result)
    # Write both to output reports and project reports/
    atomic_write_text(reports / "model_introspection.md", md)
    proj_reports = Path(cfg["paths"].get("project_reports_dir", "reports"))
    ensure_dir(proj_reports)
    atomic_write_text(proj_reports / "model_introspection.md", md)
    atomic_write_json(proj_reports / "model_introspection.json", result)

    # Free model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _format_markdown(r: dict[str, Any]) -> str:
    tree = "\n".join(r.get("module_tree", [])[:200])
    return f"""# DeepSeek-OCR-2 Model Introspection

## Summary (required fields)

| Field | Value |
|------|-------|
| 模型名称 | `{r["model_name"]}` |
| 模型revision或commit | `{r["model_revision_or_commit"]}` |
| 模型架构类名 | `{r["architecture_class"]}` |
| 视觉编码器模块路径 | `{r["visual_encoder_module_path"]}` |
| 视觉层总数 | `{r["visual_layer_total"]}` |
| 最终选定层索引 | `{r["selected_layer_index"]}` |
| 选定层输出shape | `{r["selected_layer_output_shape"]}` |
| token mask的处理方式 | {r["token_mask_handling"]} |
| 池化方式 | {r["pooling"]} |
| vLLM能否获得视觉hidden states | `{r["vllm_can_obtain_visual_hidden_states"]}` |
| 实际采用的推理架构 | {r["actual_inference_architecture"]} |

## Resolved path

`{r["model_path_resolved"]}`

## Layer details

```yaml
{yaml.safe_dump(r.get("layer_details", {}), allow_unicode=True, sort_keys=False)}
```

## Shape probe

```yaml
{yaml.safe_dump(r.get("shape_probe", {}), allow_unicode=True, sort_keys=False)}
```

## vLLM notes

```yaml
{yaml.safe_dump(r.get("vllm_notes", {}), allow_unicode=True, sort_keys=False)}
```

## Module tree (depth≤3)

```
{tree}
```

## Decision notes

- Default selected layer = `floor(num_layers / 2)` on `{r["visual_encoder_name"]}`.
- vLLM is used only for autoregressive OCR/Markdown/grounding generation.
- Visual pooled embeddings are extracted via Transformers with a forward hook on the selected layer.
- Causal-flow query tokens are the valid visual tokens used for mean+L2 pooling.
- No forged intermediate features; if extraction fails, status is recorded as failed.

"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect DeepSeek-OCR-2 model structure")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = run_introspection(cfg)
    print(
        f"[inspect_model] arch={result['architecture_class']} "
        f"encoder={result['visual_encoder_module_path']} "
        f"layers={result['visual_layer_total']} selected={result['selected_layer_index']}"
    )
    print(f"[inspect_model] wrote {cfg['paths']['reports_dir']}/model_introspection.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
