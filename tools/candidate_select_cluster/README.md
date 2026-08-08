# 候选样本筛选与聚类修正

本目录汇总文档 [候选样本筛选与聚类修正流程.md](./docs/候选样本筛选与聚类修正流程.md) 中描述的候选筛选与聚类修正代码。

最终对外 select：`.../processed_2/stage2_cluster_adjusted_3`（1547 对）。

仓库根目录 `cluster_by_question.py` 为兼容入口；`tools/candidate_select_cluster/` 根目录下四个流水线脚本亦为薄封装，实际实现位于 `scripts/` 与 `src/`。仓库 `src/{initial_candidate_select,equal_fusion_cluster,stage2_cluster_adjust,global_random_dedupe}` 为指向本目录 `src/` 下同名包的符号链接。

## 目录结构

```text
tools/candidate_select_cluster/
├── README.md
├── paths.py                              # 本目录内数据/文档/图表路径常量
├── bootstrap.py                          # 将仓库根与 cluster 根加入 sys.path
├── _run_script.py                        # 根目录薄封装共用转发器
├── select_initial_candidates.py          # 兼容入口 → scripts/
├── cluster_by_question.py
├── stage2_adjust_candidates.py
├── global_random_dedupe.py
├── render_knowledge_distribution_from_csv.py  # 兼容入口 → scripts/render_knowledge_distribution.py
├── docs/
│   ├── 候选样本筛选与聚类修正流程.md
│   └── 高中数学三级标签框架_一级与二级优化版_v0.2.md
├── data/
│   └── 第二批数据题目_seed21pro_latex.csv
├── figures/
│   └── 知识点结构与题目占比.{png,svg}    # 由 render 脚本生成
├── scripts/                              # 可执行入口（推荐直接调用）
│   ├── select_initial_candidates.py
│   ├── cluster_by_question.py
│   ├── stage2_adjust_candidates.py
│   ├── global_random_dedupe.py
│   └── render_knowledge_distribution.py
└── src/
    ├── initial_candidate_select/         # 每题抽 8 候选（4 random + 2 hard + 2 diverse）
    ├── equal_fusion_cluster/             # 等权融合特征 + 球型 K-means
    ├── stage2_cluster_adjust/            # 题内冻结聚类修正 select/raw（τ if-else）
    └── global_random_dedupe/             # 跨题全局簇：每簇至多 1 个 random
```

共享工具仍位于仓库 `src/utils.py`。第一阶段 hard 样本依赖仓库包 `benchmark_design.ocr`（AST / tokenizer）。

## 前置：多模态特征（本目录不含抽特征代码）

特征需事先落盘，后续阶段只按文件名对齐读取：

| 模态 | 模型 | 默认路径（`BASE=.../processed_2`） |
|------|------|-------------------------------------|
| 图像 | DeepSeek-OCR2 | `BASE/features/vision_deatures` |
| 文本 | Qwen3-Embedding | `BASE/features/text_feature` |

对应实现仍在仓库其他包：`src/deepseek_ocr2_features/`、`src/qwen3_text_features/`。

## 推荐运行顺序

在**仓库根目录**执行（保证 `src` / `benchmark_design` 可导入），或已 `pip install -e .`：

```bash
BASE=/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2

# 1) 第一阶段：合格题（样本数 ≥ 40）每题抽 8 候选 → select / raw
.venv/bin/python tools/candidate_select_cluster/select_initial_candidates.py

# 2) 仅对 raw_dataset 做题内球型聚类（冻结参考模型）
.venv/bin/python tools/candidate_select_cluster/cluster_by_question.py

# 3) 第二阶段题内修正：τ=ceil(K/2) if-else（不重拟合 PCA / 不重选 K）
.venv/bin/python tools/candidate_select_cluster/stage2_adjust_candidates.py --dry-run \
  --output-root "$BASE/stage2_cluster_adjusted_1"
.venv/bin/python tools/candidate_select_cluster/stage2_adjust_candidates.py --materialize \
  --output-root "$BASE/stage2_cluster_adjusted_1"

# 4) 全局簇 random 去重（2592 → 1813；对照金标准 adjusted_2）
.venv/bin/python tools/candidate_select_cluster/global_random_dedupe.py \
  --from-adjusted2-manifests "$BASE/stage2_cluster_adjusted_2/manifests" \
  --verify-against-manifests "$BASE/stage2_cluster_adjusted_2/manifests" \
  --dry-run --output-root /tmp/adjusted2_verify
# 物化（需提供含 2592 对的 select-in，通常为 adjusted_1/select_dataset）：
# .venv/bin/python tools/candidate_select_cluster/global_random_dedupe.py \
#   --assignments-csv /path/to/select_assignments.csv \
#   --select-in "$BASE/stage2_cluster_adjusted_1/select_dataset" \
#   --materialize --output-root "$BASE/stage2_cluster_adjusted_2"
```

根目录入口与 `scripts/` 下同名脚本等价；亦可：

```bash
.venv/bin/python tools/candidate_select_cluster/scripts/stage2_adjust_candidates.py --help
```

## 知识点覆盖率图（可选）

在标注 CSV 落盘后，从 `data/第二批数据题目_seed21pro_latex.csv` 统计并出图：

```bash
.venv/bin/python tools/candidate_select_cluster/scripts/render_knowledge_distribution.py
# 默认输出：figures/知识点结构与题目占比.png 与 .svg
```

## 第二阶段题内触发规则（摘要）

令 \(C\) 为当前 8 候选覆盖的不同**题内**簇数，\(\tau=\lceil K/2\rceil\)：

| 条件 | 动作 |
|------|------|
| \(C \ge \tau\) | 仅处理 random：同簇多个 random 时保留距中心最近的一个，移除其余；不碰 hard/diverse；不补样 |
| \(C < \tau\) | 移除全部 random；从 hard/diverse 尚未覆盖的簇中按簇样本量降序取中心代表补齐 |

全量 324 题复算（当前完整 if-else）：**\(C\ge\tau\) 移除 123** → select ≈ **2469**；**\(C<\tau\) 更换 504**。  
磁盘历史 `adjusted_1` 为旧规则（\(C\ge\tau\) 移除 0）→ 仍 **2592**。

## 全局簇 random 去重（`adjusted_2`）

与题内 \(\tau\) **无关**：按全局 `cluster_id`，每个簇至多保留 1 个 random（余弦最大）。  
金标准：`BASE/stage2_cluster_adjusted_2`（**1813** = 2592 − 779）。

## 阶段与代码对照

| 阶段 | 入口 | 实现包 |
|------|------|--------|
| 第一阶段候选抽取 | `select_initial_candidates.py` | `src/initial_candidate_select/` |
| 剩余样本聚类 | `cluster_by_question.py` | `src/equal_fusion_cluster/` |
| 第二阶段题内修正 | `stage2_adjust_candidates.py` | `src/stage2_cluster_adjust/run_stage2.py` |
| 全局簇 random 去重 | `global_random_dedupe.py` | `src/global_random_dedupe/run_dedupe.py` |
| 知识点标注与筛题 | `tools/seed21pro_tools/label_math_taxonomy.py` | `src/qa_latex_transcribe/label_taxonomy.py` |
| 知识点覆盖率可视化 | `scripts/render_knowledge_distribution.py` | 本目录 `figures/` |

核心阈值（题内）：`τ = ceil(K/2)`。  
原因标记：`random_multi_per_cluster_keep_nearest` / `pre_unique_cluster_count_lt_ceil_K_half`。

## 硬性约束（摘要）

1. 不把 8 个 select 候选加入聚类训练；
2. 不在候选上重新拟合 PCA；
3. 不重新搜索或重跑球型 K-means；
4. 不修改已冻结簇中心；
5. 不覆盖第一阶段 `select_dataset` / `raw_dataset`；
6. 本流程不启动第三阶段全数据集验收聚类。

更多算法细节、审计字段与物化结果摘要见 [流程文档](./docs/候选样本筛选与聚类修正流程.md)。
