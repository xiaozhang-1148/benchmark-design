# DeepSeek-OCR-2 Feature Extraction & Distribution Analysis

四通道：视觉 / 布局 / 识别内容 / OCR 质量（质量只过滤，不进 PCA）。不做聚类与特征融合。

## Runtime

```bash
cd /home/baoquan/benchmark/benchmark-design
uv sync --extra deepseek --extra analysis   # analysis 含 umap-learn（plot 硬依赖）
# OCR 另需: uv pip install "vllm==0.19.0"
export HF_HOME=/mnt/nvme_model/baoquan/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
```

## Commands

```bash
# 全量重跑 + 多卡数据并行（resume=false 会清空 OCR/视觉缓存）
uv run -m src.pipeline --config configs/full_gpu.yaml

# 从已有 OCR 重解析特征（CPU，无需 GPU）
uv run -m src.pipeline --config configs/default.yaml \
  --stages parse_layout parse_ocr_quality parse_recognition analyze plot report
```

## Outputs

Default root: `/home/baoquan/benchmark/deepseek`

```
deepseek/
  outputs/
    manifest.parquet
    ocr_generations.parquet
    ocr_quality.parquet          # quality channel
    recognition_features.parquet # content only
    layout_features.parquet
    visual_embeddings.f32.mmap
    analysis/
  reports/
    quality_gates.json
    feature_analysis.md
    figures/
```

## Architecture

- **Channel A (vLLM)**: OCR / Markdown / grounding
- **Channel B (Transformers)**: fixed visual-layer pooled embedding
- **OCR quality**: truncated / repetitive / empty / parse_failed — filter only
- Align by `image_id`
