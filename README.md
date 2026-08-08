# benchmark-design

Benchmark dataset analysis for **HMER** (handwritten math expression recognition) and **block-level** (image-side block annotation) metrics.

## 项目模块地图（Module Map）

仓库按功能域分层，`benchmark_design/` 与 `src/` 为可导入 Python 包，其余为独立工具/归档目录：

```text
benchmark-design/
├── benchmark_design/          # 核心分析库（可安装包，python -m benchmark_design）
├── src/                       # 实验代码（视觉/文本特征、聚类等；含符号链接）
├── tools/                     # 独立工具域与论文图归档
│   ├── candidate_select_cluster/   # 候选样本筛选与聚类修正
│   ├── seed21pro_tools/            # Seed 2.1 Pro 转录/标注
│   └── figure_data_code/           # 论文/报告用图（数据 + 出图代码 + SVG）
├── config/                    # 核心分析配置（benchmark_design 使用）
├── configs/                   # 实验配置（src/ 各实验使用）
├── scripts/                   # 一次性/运维辅助脚本
├── tests/                     # pytest 测试
├── pyproject.toml
└── uv.lock
```

**注意**：`src/` 下的 `equal_fusion_cluster`、`initial_candidate_select`、`stage2_cluster_adjust`、`global_random_dedupe`、`qa_latex_transcribe` 是指向 `tools/` 内实现目录的**符号链接**，保持 `from src.xxx import` 的兼容导入路径；勿删除。

### 核心分析库（benchmark_design/）

```
benchmark_design/
  config/           # shared + domain config (hmer, block_level, project)
  project/          # unified project export + summary.json
  hmer/             # HMER domain facade (implementation in ocr/)
  ocr/              # LaTeX tokenization, structure metrics, cross-benchmark
  block_level/      # image sample loading and flow-structure classification
  page_level/       # pure image-level page analysis
  line_level/       # line-level geometry and scan quality
  io/               # shared dataset loaders
  report/           # export writers
    block_level/    # block-level output layout + export pipeline
  progress.py       # shared Rich / parallel helpers
```

### 实验代码（src/）

| 子目录 | 功能 |
|--------|------|
| `deepseek_ocr2_features/` | DeepSeek-OCR2 视觉特征提取 |
| `qwen3_text_features/` | Qwen3 文本特征提取 |
| `joint_cluster/` `joint_cluster_v2/` | 联合聚类（v1 / v2） |
| `visual_exp/` | 视觉特征聚类实验 |
| `analysis_v2/` | 特征分析 v2 |
| `equal_fusion_cluster/` 等 | → 指向 `tools/candidate_select_cluster/src/`（符号链接） |
| `qa_latex_transcribe/` | → 指向 `tools/seed21pro_tools/src/`（符号链接） |

### 独立工具域（tools/）

| 目录 | 说明 | 入口 |
|------|------|------|
| `candidate_select_cluster/` | 候选样本筛选与聚类修正（4 阶段） | `tools/candidate_select_cluster/README.md` |
| `seed21pro_tools/` | Seed 2.1 Pro Q/A 转录、学生 OCR 回填、数学标签 | `tools/seed21pro_tools/README.md` |
| `figure_data_code/` | 论文三张图的数据、出图代码与 SVG | `tools/figure_data_code/README.md` |

### 配置

| 目录 | 用途 |
|------|------|
| `config/` | 核心分析：`project.yaml`、`image_analysis.yaml`、`line_analysis.yaml`、`page_level_latex_split.yaml` |
| `configs/` | 实验：`default.yaml`、`full_gpu.yaml`、`smoke.yaml`、`experiment/`、`joint_cluster/` |

`configs/` 为各 `src/` 实验脚本 `--config` 的默认路径，勿与 `config/` 混淆。

## 核心 CLI（benchmark_design）

| Domain | CLI | Output root | Focus |
|--------|-----|-------------|--------|
| **Project** | `project export` | `benchmark_export/` | HMER + page_level/* + line_level + page_level_HMER (+ optional split) |
| HMER | `ocr …` or `hmer …` | `HMER/` | LaTeX tokens, structure, AST depth |
| Block-level | `block-level export` | `block_level/` | answer-block flow structure classification |

`ocr` remains the stable command namespace; `hmer` forwards to the same subcommands. The legacy `export` and `vision` commands remain as deprecation aliases.

## Project export (recommended)

One command runs the full benchmark stack in dependency order, writes `summary.json`, `PIPELINE.md`, and the dataset overview (总纲):

```bash
python -m benchmark_design project export \
  --config config/project.yaml \
  --output ./benchmark_export
```

Full export including the Chapter 7 stratified split:

```bash
python -m benchmark_design project export \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark \
  --output /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark_export \
  --run-page-level-latex-split
```

Output layout:

```
benchmark_export/
  summary.json
  dataset_overview.md
  PIPELINE.md                    # layer linkage and join keys
  pipeline_manifest.json         # machine-readable stage graph
  page_level/
    density/                     # foreground features
    structure_layout/            # flow structure
  line_level/                    # line geometry
  HMER/                          # expression stats
  page_level_HMER/               # page LaTeX metrics (Chapter 6)
```

**Dependency order:** HMER + `page_level` + `block_level/structure_layout` + `block_level/hybrid_layout` (parallel) → `line_level` (uses density calibration) + `page_level_HMER` (parallel).

Standalone module name for page-level HMER is `page_level_latex`; the export directory is `page_level_HMER/`.

Options mirror individual pipelines (`--skip-figures`, `--skip-page-level`, `--skip-line-level`, `--skip-page-level-hmer`, etc.). Legacy flat paths `block_level/` and top-level `page_level/tables/` remain readable via fallbacks in cross-domain joins.

### Legacy unified export

`python -m benchmark_design export` is a deprecated alias for `project export`.

## Full HMER benchmark export

```bash
python -m benchmark_design ocr export \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./HMER
```

Writes a layered output tree:

```
HMER/
  summary.md              # overview report
  ocr_benchmark_summary.md
  cross_benchmark_summary.md
  metadata.json           # provenance, parse rates, manifest
  tables/                 # core summary CSVs (+ README.md)
  tables/appendix/        # detailed distributions and long listings
  cross_benchmark/        # cross-dataset comparison CSVs
  resources/              # latex_vocab.csv, token_taxonomy_map.csv
  details/                # expression-level and derived CSVs
  examples/               # audit sample CSVs
  figures/                # PNG charts
  docs/                   # tokenizer rules and taxonomy examples
```

Options:

```bash
python -m benchmark_design ocr export --skip-figures --skip-cross-benchmark
python -m benchmark_design hmer export --output ./benchmark_output
python -m benchmark_design ocr export --workers 16 --datasets CROHME2014,HME100K
```

## Block-level benchmark export

```bash
python -m benchmark_design block-level export \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./block_level
```

(`vision export` is a deprecated alias.)

Writes:

```
block_level/
  block_level_summary.md
  flow_structure_summary.md
  metadata.json
  tables/sample_index.csv
  tables/flow_structure_page_metrics.csv   # flow_group + diagnostic fields
  tables/flow_group_summary.csv          # five-class primary summary
  tables/flow_structure_block_geometry.csv
  details/flow_structure_decisions.jsonl
  figures/flow_structure/
```

Use `--skip-flow-figures` to skip PNG generation.
Use `--skip-dimensions` when Pillow is unavailable.

### Answer-Block Flow Structure only

```bash
python -m benchmark_design block-level flow-structure \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./block_level
```

Also writes `tables/flow_group_summary.csv`, `details/flow_structure_decisions.jsonl`,
`flow_structure_summary.md`, and optionally `figures/flow_structure/` mask overlays.

Classifies each page into `Single-flow`, `Columnar-flow`, `Hybrid-flow`, or `NA` using
`Txtblock` mask geometry from the benchmark JSON export (`Txtblock`, `figure`,
`deleted_text_block`, `chart` block types). Every `Hybrid-flow` row includes a non-empty
`hybrid_reason`.

Add `--skip-flow-figures` to skip overlay PNG generation.
Add `--skip-dimensions` when Pillow is unavailable (for `block-level export` only).

## Cross-benchmark comparison

```bash
python -m benchmark_design ocr cross-benchmark \
  --output ./HMER \
  --datasets CROHME2014,HME100K,MathWriting
```

Outputs `cross_benchmark/*.csv` (profiles, summary, length bins, tokenizer coverage, provenance) using the unified `LATEX_DICT` tokenizer. The full comparison narrative is written to `cross_benchmark_summary.md` at the output root.

## Consolidated OCR report (Tables 1–7)

```bash
python -m benchmark_design ocr report \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./HMER
```

Outputs under `HMER/`:
- `tables/` — core per-metric CSV summaries (see `tables/README.md` for core vs appendix)
- `ocr_benchmark_summary.md` — all seven tables in one Markdown document (output root)
- `cross_benchmark_summary.md` — cross-dataset comparison narrative (output root)
- `docs/metadata/ocr_benchmark_metadata.json` — generation metadata

All OCR commands support Rich progress bars (enabled by default) and parallel loading/tokenization:

```bash
python -m benchmark_design ocr report --workers 16 --output ./benchmark_output
python -m benchmark_design ocr report --no-progress --output ./benchmark_output
```

## OCR data scale

```bash
python -m benchmark_design ocr scale \
  --input /mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_1/benchmark \
  --output ./HMER
```

## OCR expression length distribution

```bash
python -m benchmark_design ocr length --output ./benchmark_output
```

## OCR fixed length bins

```bash
python -m benchmark_design ocr bins --output ./benchmark_output
```

## OCR token taxonomy composition

```bash
python -m benchmark_design ocr taxonomy --output ./benchmark_output
```

## OCR token long-tail

```bash
python -m benchmark_design ocr longtail --output ./benchmark_output
```

## OCR structure type distribution

```bash
python -m benchmark_design ocr structure --output ./benchmark_output
```

Matrix triggers require a complete `\begin{env} ... \\ ... \end{env}` block where
`env` is one of `cases`, `pmatrix`, `bmatrix`, `Bmatrix`, `vmatrix`, `Vmatrix`,
`matrix`, `array`, or `rcases`. Each valid block counts once. Matrix **Max Depth**
is the maximum nesting depth of such valid blocks. PosFormer AST depth uses the
same matrix-environment rule.

## OCR structure combination complexity

```bash
python -m benchmark_design ocr structure-complexity --output ./benchmark_output
```

Counts how many distinct table-6 structure types co-occur within one expression.

## OCR LaTeX AST depth (PosFormer)

```bash
python -m benchmark_design ocr ast --output ./benchmark_output
```

PosFormer position-forest encoding depth statistics (summary section 7; `tables/ast_depth_summary.csv`).

## Development

```bash
pip install -e ".[dev]"
pytest
```

Integration tests (full benchmark / CROHME2014) run only when datasets are mounted:

```bash
pytest -m integration
```
