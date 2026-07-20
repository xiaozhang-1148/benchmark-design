# DeepSeek-OCR-2 Model Introspection

## Summary (required fields)

| Field | Value |
|------|-------|
| 模型名称 | `deepseek-ai/DeepSeek-OCR-2` |
| 模型revision或commit | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` |
| 模型架构类名 | `DeepseekOCR2ForCausalLM` |
| 视觉编码器模块路径 | `model.qwen2_model` |
| 视觉层总数 | `24` |
| 最终选定层索引 | `12` |
| 选定层输出shape | `(1, 512, 896)` |
| token mask的处理方式 | token_type_ids: 0=non-causal image patches, 1=causal-flow queries; pool mean over causal-flow half only |
| 池化方式 | mean(valid_visual_tokens, dim=0) then L2 normalize (float32) |
| vLLM能否获得视觉hidden states | `False` |
| 实际采用的推理架构 | dual_channel: A=vLLM DeepseekOCR2ForCausalLM (OCR/Markdown/grounding generation); B=Transformers BF16+SDPA (fixed visual-layer pooled embedding) |

## Resolved path

`/mnt/nvme_model/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/snapshots/aaa02f3811945a91062062994c5c4a3f4c0af2b0`

## Layer details

```yaml
sam_module_path: model.sam_model
sam_num_layers: 12
sam_class: ImageEncoderViT
qwen2_module_path: model.qwen2_model
qwen2_class: Qwen2Decoder2Encoder
qwen2_layers_path: model.qwen2_model.model.model.layers
qwen2_num_layers: 24
causal_flow_query_1024: 256
causal_flow_query_768: 144
causal_flow_hidden_dim: 896
projector_path: model.projector
projector_class: MlpProjector

```

## Shape probe

```yaml
layer: 12
shape:
- 1
- 512
- 896
dtype: torch.bfloat16
output_shape:
- 1
- 256
- 896
selected_layer: 12
num_layers: 24
full_seq_len: 512
causal_flow_token_count: 256
token_mask: 'token_type_ids: 0=non-causal image patches, 1=causal-flow queries; pool
  mean over causal-flow half only'
pooled_shape:
- 896

```

## vLLM notes

```yaml
vllm_can_return_visual_hidden_states: false
reason: 'vLLM DeepseekOCR2ForCausalLM exposes multimodal embeddings for generation
  only; LLM.generate / AsyncLLMEngine do not provide a stable API for intermediate
  visual-encoder hidden states (SAM blocks or Qwen2 causal-flow layers). Using dual-channel:
  vLLM for OCR/grounding generation; Transformers for fixed visual-layer pooled embeddings.'
vllm_version: 0.19.0
vllm_has_ocr2: true

```

## Module tree (depth≤3)

```
DeepseekOCR2ForCausalLM
  model: DeepseekOCR2Model (local_params=1280)
    model.embed_tokens: Embedding (local_params=165478400)
    model.layers: ModuleList (local_params=0)
      model.layers.0: DeepseekV2DecoderLayer (local_params=0)
      model.layers.1: DeepseekV2DecoderLayer (local_params=0)
      model.layers.2: DeepseekV2DecoderLayer (local_params=0)
      model.layers.3: DeepseekV2DecoderLayer (local_params=0)
      model.layers.4: DeepseekV2DecoderLayer (local_params=0)
      model.layers.5: DeepseekV2DecoderLayer (local_params=0)
      model.layers.6: DeepseekV2DecoderLayer (local_params=0)
      model.layers.7: DeepseekV2DecoderLayer (local_params=0)
      model.layers.8: DeepseekV2DecoderLayer (local_params=0)
      model.layers.9: DeepseekV2DecoderLayer (local_params=0)
      model.layers.10: DeepseekV2DecoderLayer (local_params=0)
      model.layers.11: DeepseekV2DecoderLayer (local_params=0)
    model.norm: DeepseekV2RMSNorm (local_params=1280)
    model.sam_model: ImageEncoderViT (local_params=3145728)
      model.sam_model.patch_embed: PatchEmbed (local_params=0)
      model.sam_model.blocks: ModuleList (local_params=0)
      model.sam_model.neck: Sequential (local_params=0)
      model.sam_model.net_2: Conv2d (local_params=1179648)
      model.sam_model.net_3: Conv2d (local_params=4128768)
    model.qwen2_model: Qwen2Decoder2Encoder (local_params=0)
      model.qwen2_model.model: CustomQwen2Decoder (local_params=0)
      model.qwen2_model.query_768: Embedding (local_params=129024)
      model.qwen2_model.query_1024: Embedding (local_params=229376)
    model.projector: MlpProjector (local_params=0)
      model.projector.layers: Linear (local_params=1148160)
  lm_head: Linear (local_params=165478400)
```

## Decision notes

- Default selected layer = `floor(num_layers / 2)` on `qwen2_model`.
- vLLM is used only for autoregressive OCR/Markdown/grounding generation.
- Visual pooled embeddings are extracted via Transformers with a forward hook on the selected layer.
- Causal-flow query tokens are the valid visual tokens used for mean+L2 pooling.
- No forged intermediate features; if extraction fails, status is recorded as failed.

