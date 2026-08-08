# Seed 2.1 Pro 工具集

本目录汇总基于 **Seed 2.1 Pro**（ByteDance 多模态大模型）的 Q/A LaTeX 转录、学生 OCR 填充与数学三级标签识别脚本。

## 目录结构

```text
seed21pro_tools/
├── README.md                    # 本文档
├── fill_student_ocr.py          # 真题学生 OCR 回填入口
├── label_math_taxonomy.py       # 数学三级标签标注入口
├── 高中数学三级标签框架_一级与二级优化版_v0.2.md   # 标签体系定义
└── src/
    └── qa_latex_transcribe/
        ├── __init__.py
        ├── run_transcribe.py    # Q/A LaTeX 转录（Seed 2.1 Pro API）
        ├── fill_student_ocr.py  # 学生笔迹 OCR 批量填充
        └── label_taxonomy.py   # 数学标签逐题标注
```

本目录位于 `tools/seed21pro_tools/`，为独立归档：仅含出图脚本、输入数据与 SVG 输出。`src/qa_latex_transcribe` 为仓库根 `src/` 下同名符号链接的目标。

## 前置

- 使用 ByteDance Seed 2.1 Pro API（通过 OpenAI 兼容接口）；需设置环境变量：
  ```bash
  export OPENAI_API_KEY="your-ark-api-key"
  export OPENAI_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
  ```
- 图像素材与 CSV 已就绪（参见各脚本 `--help`）。

## 运行方式

在**仓库根目录**执行（保证 `src` 可导入）。本目录位于 `tools/seed21pro_tools/`：

### 1) Q/A LaTeX 转录

将真题 Q/A 图片转为 LaTeX CSV：

```bash
.venv/bin/python tools/seed21pro_tools/src/qa_latex_transcribe/run_transcribe.py \
  --input-dir tempt_data/第二批数据题目 \
  --output-csv tempt_data/第二批数据题目_seed21pro_latex.csv \
  --max-workers 32
```

关键选项：
- `--input-dir`：包含 `q_*.jpg` / `a_*.jpg` 的目录
- `--output-csv`：输出 LaTeX CSV 路径
- `--max-workers`：并发数（Q∥A 并行）
- `--disable-thinking`：关闭推理增强（默认关闭）

### 2) 学生 OCR 填充

将真题的学生笔迹 OCR 填入 CSV：

```bash
.venv/bin/python tools/seed21pro_tools/fill_student_ocr.py \
  --csv tempt_data/第二批数据题目_seed21pro_latex.csv \
  --raw-root .../processed_2/data_set/raw_dataset \
  --output-csv tempt_data/第二批数据题目_with_students.csv
```

规则：仅 Q 和 A 均为空时才从 raw_dataset 提取学生 OCR（最多 10 条）。

### 3) 数学三级标签标注

逐题调用 Seed 2.1 Pro 进行标签分类：

```bash
.venv/bin/python tools/seed21pro_tools/label_math_taxonomy.py \
  --csv tempt_data/第二批数据题目_with_students.csv \
  --output-csv tempt_data/第二批数据题目_labeled.csv \
  --max-workers 32 \
  --resume
```

关键选项：
- `--max-workers`：并发线程数（推荐 32–128）
- `--resume`：跳过已有标签的题目
- `--taxonomy-file`：自定义标签体系文件（默认使用同目录 `高中数学三级标签框架_一级与二级优化版_v0.2.md`）

输出格式：中文标签 `Label_level_1` / `Label_level_2` / `Label_level_3`，多标签时每对扩充为一行。

## 标签体系

详见 `高中数学三级标签框架_一级与二级优化版_v0.2.md`。label_taxonomy.py 仅解析其中的 L1 表、L2 表和边界表，不依赖完整 Markdown。

## 共享常量

`src/qa_latex_transcribe/run_transcribe.py` 中定义了 CSV 列名常量 `CSV_COLUMNS`，供 `fill_student_ocr.py` 与 `label_taxonomy.py` 统一引用。
