# figure_data_code

本目录位于 `tools/figure_data_code/`，为**独立归档**：仅含出图脚本、输入数据与 SVG 输出，不依赖仓库内其他 Python 包或路径。每个图文件夹自成一体（各自的 `code/`、`data/`、`figure/`）。

## 目录结构

```text
.
├── requirements.txt
├── README.md
├── knowledge_coverage_distribution/
│   ├── data/
│   ├── code/
│   │   ├── render.py
│   │   └── _common.py
│   └── figure/
└── page_token_structure_scale/
    ├── data/
    ├── code/
    │   ├── render.py
    │   ├── _common.py
    │   └── _page_level_panel.py
    └── figure/
```

## 图一览

| 文件夹 | 默认输出 | 说明 |
|--------|----------|------|
| `knowledge_coverage_distribution/` | `figure/知识点结构与题目占比.svg` | 知识点覆盖率（n=274） |
| `page_token_structure_scale/` | `figure/page_level_distribution.svg` | 页级整合分布（n=1547） |

### `data/` 文件

**knowledge_coverage_distribution**

| 文件 | 说明 |
|------|------|
| `第二批数据题目_seed21pro_latex.csv` | 知识点标注 |
| `高中数学三级标签框架_一级与二级优化版_v0.2.md` | 标签框架参考（顺序已固化在代码中） |

**page_token_structure_scale**

| 文件 | 说明 |
|------|------|
| `page_latex_metrics.csv` | 页级 LaTeX 指标 |
| `kept_samples.csv` | hard / non-hard 划分 |
| `image_features.csv` | `foreground_density` |

## 环境

在本目录（`figure_data_code`）下创建虚拟环境并安装依赖即可，与仓库根目录的 `.venv` 无关：

```bash
cd figure_data_code
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 如何运行

**请先 `cd` 到本目录**（下文路径均相对于 `figure_data_code/`）：

```bash
cd figure_data_code

# 知识点结构与题目占比
.venv/bin/python knowledge_coverage_distribution/code/render.py

# 页级整合分布
.venv/bin/python page_token_structure_scale/code/render.py
```

成功时终端会打印生成的 SVG 路径（默认写在各图的 `figure/` 下）。

### 自定义输入/输出

```bash
.venv/bin/python knowledge_coverage_distribution/code/render.py \
  --csv knowledge_coverage_distribution/data/第二批数据题目_seed21pro_latex.csv \
  --output knowledge_coverage_distribution/figure/知识点结构与题目占比.svg

.venv/bin/python page_token_structure_scale/code/render.py \
  --metrics page_token_structure_scale/data/page_latex_metrics.csv \
  --features page_token_structure_scale/data/image_features.csv \
  --kept-samples page_token_structure_scale/data/kept_samples.csv \
  --output page_token_structure_scale/figure/page_level_distribution.svg
```

### 使用 `--hard-index`（可选）

传入外部 hard 索引 CSV（列结构须与 `kept_samples.csv` 相同，含
`basename` + `post_selection_source`）时，将用它替代 `--kept-samples` 作为
hard / non-hard 划分来源，并把 `page_level_distribution.svg` 输出到**该 CSV
所在目录**：

```bash
.venv/bin/python page_token_structure_scale/code/render.py \
  --hard-index /mnt/nvme_user/baoquan_datasets/Bench_Folder/hard_index.csv
# => /mnt/nvme_user/baoquan_datasets/Bench_Folder/page_level_distribution.svg
```

`--output` 也可指定目录（末尾带 `/` 或已存在目录）或文件路径：

```bash
.venv/bin/python page_token_structure_scale/code/render.py \
  --hard-index /mnt/nvme_user/baoquan_datasets/Bench_Folder/hard_index.csv \
  --output /mnt/nvme_user/baoquan_datasets/Bench_Folder/out/
# => /mnt/nvme_user/baoquan_datasets/Bench_Folder/out/page_level_distribution.svg
```

未创建 `.venv` 时，也可用已安装相同依赖的系统 `python3` 替换上述命令中的 `.venv/bin/python`。
