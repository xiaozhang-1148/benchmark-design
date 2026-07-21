# DeepSeek-OCR2 Visual Embedding Experiment

**方法**：DeepSeek-OCR2 mean-pooled projected-token embedding  
`SAM → Qwen2 → MLP Projector → mean pool → L2`

不做：OCR 文本生成、GT、布局解析、train/val/test 划分。

## 运行

```bash
cd /home/baoquan/benchmark/benchmark-design
uv sync --extra deepseek --extra analysis
export HF_HOME=/mnt/nvme_model/baoquan/.cache/huggingface

# 1) 接口验证（约 50 张）
uv run -m src.pipeline --config configs/experiment/run_config.yaml --stages verify

# 2) Smoke（约 800 张：提取 + 诊断 + PCA/UMAP）
uv run -m src.pipeline --config configs/experiment/run_config.yaml --stages smoke

# 3) 全量提取
uv run -m src.pipeline --config configs/experiment/run_config.yaml --stages manifest extract

# 4) 全量分析与聚类
uv run -m src.pipeline --config configs/experiment/run_config.yaml --stages analyze
```

## 输出

`/home/baoquan/benchmark/deepseek/experiment/`

```
config/run_config.yaml
metadata/manifest.parquet
embeddings/deepseek_ocr2_mean_l2.npy
diagnostics/
projections/
clustering/
galleries/
report/visual_embedding_analysis.html
```
