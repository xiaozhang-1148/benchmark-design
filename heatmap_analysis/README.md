# 手写答题图像热力图统计与聚类分析系统

基于归一化坐标网格的手写笔迹密度热力图提取、数据集级统计、分组比较与聚类分析工具。

## GPU 加速

系统检测到 **16× NVIDIA A10** 时，默认启用 GPU 加速（需安装 CuPy）：

```bash
pip install cupy-cuda12x
# 或
pip install -e ".[gpu]"
```

配置项（`config/heatmap_analysis.yaml`）：

```yaml
gpu:
  enabled: true
  device_ids: null              # 使用全部 GPU
  num_workers: null             # 每 GPU 一个进程（默认）
  min_images_for_parallel: 500  # 低于此数量时用单 GPU 进程内模式
```

**加速策略：**
- **多 GPU 并行提取**（≥500 张）：每 GPU 一进程，CPU OpenCV 笔迹提取 + GPU 热力图
- **热力图/grid**：CuPy 向量化 + GPU 高斯平滑
- **聚类**（≥64 样本）：GPU SVD-PCA + KMeans（`gpu.clustering: true`）
- **数据集聚合**：≥64 张时在 GPU 上计算统计量
- **可选** `gpu.preprocessing: true`：笔迹提取也走 GPU（大图场景；小答题卡默认 CPU 更快）

禁用 GPU：`--no-gpu` 或 `gpu.enabled: false`

**实测（1000 张，preprocessing=false）：**
| 模式 | 吞吐 |
|------|------|
| CPU 单进程 | ~32 img/s |
| 16 GPU 并行 + GPU 热力图 | ~70 img/s |

## 安装依赖

```bash
cd /home/baoquan/ocr-process/benchmark-design
pip install -e ".[dev]"
# 或
pip install -r requirements-heatmap.txt
```

可选：`pip install hdbscan` 以启用 HDBSCAN 聚类。

## 项目结构

```text
heatmap_analysis/          # 主包
config/heatmap_analysis.yaml  # 生产配置（benchmark 数据集）
config.example.yaml        # 合成数据示例配置
tests/                     # 单元测试
tests/fixtures/            # 合成测试数据生成器
hotmap/                    # 默认输出目录
```

## 数据准备

默认读取：

```text
/mnt/nvme_user/baoquan_datasets/EDA-Data-Folder/processed_2/benchmark
```

- 支持 `.jpg/.jpeg/.png` 等常见格式
- 可选 `metadata.csv`（列：`image_path`, `image_id`, `template_id`, 以及分组字段如 `correct`, `score`, `school`）
- 无 metadata 时自动扫描目录并为每张图片建立索引
- 可选空白模板目录（`template_dir`）用于模板差分提取笔迹

## 运行完整流程

```bash
python -m heatmap_analysis run-all --config config/heatmap_analysis.yaml
```

合成数据验证：

```bash
python tests/fixtures/generate_heatmap_synthetic.py
python -m heatmap_analysis run-all --config config.example.yaml
```

## 分步命令

```bash
python -m heatmap_analysis preprocess --config config/heatmap_analysis.yaml
python -m heatmap_analysis extract --config config/heatmap_analysis.yaml
python -m heatmap_analysis aggregate --config config/heatmap_analysis.yaml
python -m heatmap_analysis compare --config config/heatmap_analysis.yaml --group-by correct
python -m heatmap_analysis cluster --config config/heatmap_analysis.yaml
python -m heatmap_analysis report --config config/heatmap_analysis.yaml
```

调试时可限制处理数量：

```bash
python -m heatmap_analysis extract --config config/heatmap_analysis.yaml --limit 100
```

## 输出目录

默认：`/home/baoquan/ocr-process/benchmark-design/hotmap/`

```text
outputs/
├── per_image_heatmaps/     # 单张热力图与叠加图
├── aggregate/              # 数据集平均/中位数/标准差/使用概率
├── groups/                 # 分组比较与差异图
├── clustering/             # 聚类结果
├── representative_samples/
├── tables/                 # CSV 指标表
├── models/                 # PCA/Scaler/KMeans 模型
├── cache/                  # 中间 npz 缓存
└── report/index.html       # HTML 分析报告
```

## 核心指标定义

| 指标 | 定义 |
|------|------|
| `ink_coverage` | 笔迹像素权重之和 / 页面总像素 |
| `D_abs[i,j]` | 网格内笔迹权重之和 / 网格像素数 |
| `D_rel[i,j]` | `D_abs[i,j] / sum(D_abs)`，空白卷标记 `is_blank` |
| `spatial_entropy` | 归一化空间熵 `H / log(n)`，范围 [0,1] |
| `hotspot_concentration` | 最密集 10% 网格承载的相对笔迹占比 |
| `dense_stroke_overlap_proxy` | 最大网格绝对密度 / 平均绝对密度（笔画重叠代理，≠涂改率） |

## 测试

```bash
pytest tests/test_heatmap.py tests/test_metrics.py tests/test_alignment.py \
       tests/test_aggregation.py tests/test_clustering.py -v
```

## 当前限制

- 无模板时使用 Otsu/自适应阈值，印刷内容可能被计入笔迹
- 大数据集可视化默认最多渲染 500 张单图 PNG（aggregate/cluster 不受影响）
- UMAP/HDBSCAN 为可选依赖；HDBSCAN 在共享 PCA 后再投影到 **2 维** 上聚类（避免高维密度估计过慢）
- 聚类标签仅描述空间布局模式，需结合外部变量谨慎解读
