# DeepSeek-OCR-2 Feature Extraction & Distribution Analysis

三套特征独立提取与分布分析（视觉层 / 布局结构 / 识别统计）。本阶段不做聚类、不做特征融合、不做 train/val/test 划分。

## Runtime

推荐解释器（含 vLLM≥0.19 + DeepSeek-OCR-2）：

```bash
export PYTHON=/home/baoquan/ocr-process/ocr-pipeline/.venv/bin/python
export HF_HOME=/mnt/nvme_model/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0
```

## Commands

```bash
cd /home/baoquan/ocr-process/benchmark-design

$PYTHON -m src.inspect_model --config configs/default.yaml
$PYTHON -m src.build_manifest --config configs/default.yaml
$PYTHON -m src.vllm_ocr_runner --config configs/default.yaml --benchmark-only
$PYTHON -m src.vllm_ocr_runner --config configs/default.yaml
$PYTHON -m src.visual_feature_runner --config configs/default.yaml
$PYTHON -m src.parse_layout --config configs/default.yaml   # also invoked after OCR
$PYTHON -m src.parse_recognition --config configs/default.yaml
$PYTHON -m src.analyze_features --config configs/default.yaml
$PYTHON -m src.build_report --config configs/default.yaml

# end-to-end
$PYTHON -m src.pipeline --config configs/default.yaml

# smoke (10 images)
$PYTHON -m src.pipeline --config configs/smoke.yaml --limit 10
```

## Outputs

Default root: `/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark_export_1/deepseek`

```
deepseek/
  outputs/
    manifest.parquet
    ocr_generations.parquet
    recognition_raw/
    layout_raw/
    layout_features.parquet
    recognition_features.parquet
    visual_embeddings.f32.mmap
    visual_index.parquet
    analysis/
  reports/
    model_introspection.md
    benchmark.md / benchmark.json
    feature_analysis.md
    feature_metrics.json
    figures/
    contact_sheets/
```

## Architecture

- **Channel A (vLLM)**: OCR / Markdown / grounding generation
- **Channel B (Transformers)**: fixed Qwen2 visual-encoder layer pooled embedding (default layer `floor(24/2)=12`), causal-flow tokens only, mean + L2
- Align by `image_id` (content sha256 prefix)
