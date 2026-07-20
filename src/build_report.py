"""Assemble feature_analysis.md report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, model_resolved_path
from .utils import atomic_write_text, ensure_dir


def build_report(cfg: dict[str, Any]) -> str:
    reports = Path(cfg["paths"]["reports_dir"])
    out_dir = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(reports)

    intro = {}
    if (reports / "model_introspection.json").exists():
        intro = json.loads((reports / "model_introspection.json").read_text())
    metrics = {}
    if (reports / "feature_metrics.json").exists():
        metrics = json.loads((reports / "feature_metrics.json").read_text())
    bench = {}
    if (reports / "benchmark.json").exists():
        bench = json.loads((reports / "benchmark.json").read_text())

    man_n = 0
    corrupt_n = 0
    if (out_dir / "manifest.parquet").exists():
        man = pd.read_parquet(out_dir / "manifest.parquet")
        man_n = len(man)
        corrupt_n = int((man["status"] == "corrupt").sum())

    ocr_ok = ocr_fail = 0
    if (out_dir / "ocr_generations.parquet").exists():
        ocr = pd.read_parquet(out_dir / "ocr_generations.parquet")
        ocr_ok = int((ocr["status"] == "ok").sum())
        ocr_fail = int((ocr["status"] != "ok").sum())

    layout_avail = None
    if (out_dir / "layout_features.parquet").exists():
        lay = pd.read_parquet(out_dir / "layout_features.parquet")
        layout_avail = int(lay["layout_available"].sum()) if "layout_available" in lay else None

    vis_n = 0
    if (out_dir / "visual_index.parquet").exists():
        vis_n = len(pd.read_parquet(out_dir / "visual_index.parquet"))

    cross = metrics.get("cross_channel_distance_spearman")

    md = f"""# DeepSeek-OCR-2 Feature Analysis Report

## 1. 数据与模型配置

- 输入目录: `{cfg["data"]["input_dir"]}`
- 输出根目录: `{cfg["paths"]["output_root"]}`
- 清单样本数: {man_n}（损坏: {corrupt_n}）
- 模型: `{cfg["model"]["name_or_path"]}`
- revision/commit: `{cfg["model"].get("revision")}`
- 解析路径: `{model_resolved_path(cfg)}`
- 架构类: `{intro.get("architecture_class")}`
- 视觉编码器: `{intro.get("visual_encoder_module_path")}`（层数={intro.get("visual_layer_total")}，选定层={intro.get("selected_layer_index")}）
- 提示词: `{cfg.get("prompt")}`
- 配置指纹: `{intro.get("config_fingerprint")}`

## 2. 推理架构和性能

- 实际推理架构: {intro.get("actual_inference_architecture")}
- vLLM能否返回视觉hidden states: `{intro.get("vllm_can_obtain_visual_hidden_states")}`
- OCR成功/失败: {ocr_ok} / {ocr_fail}
- 视觉embedding数: {vis_n}
- 布局可用样本: {layout_avail}

### Benchmark best config

```json
{json.dumps(bench.get("best_config", {}), indent=2, ensure_ascii=False)}
```

图表与原始结果见 `reports/benchmark.md` / `reports/benchmark.json`。

## 3. 视觉层特征分析

```json
{json.dumps(metrics.get("channels", {}).get("visual", {}), indent=2, ensure_ascii=False)[:4000]}
```

- 距离: cosine
- 池化: mean(valid causal-flow tokens) + L2
- 图: `figures/visual_pca.png`, `figures/visual_umap.png`, `figures/visual_knn15.png`
- 联系表: `contact_sheets/visual_contact.png`

## 4. 布局结构特征分析

```json
{json.dumps(metrics.get("channels", {}).get("layout", {}), indent=2, ensure_ascii=False)[:4000]}
```

- 若模型输出无可靠坐标: `layout_available=false`，不伪造坐标；仍保留 Markdown 结构回退字段。
- 图: `figures/layout_pca.png`, `figures/layout_occupancy_mean.png`, `figures/layout_feature_spearman.png`

## 5. 最终识别结果特征分析

```json
{json.dumps(metrics.get("channels", {}).get("recognition", {}), indent=2, ensure_ascii=False)[:4000]}
```

- 第一版为显式统计特征；`extract_decoder_embedding=false`（默认）
- 若 logprob 不可用: `logprob_available=false`，`mean_generated_token_logprob=null`
- 图: `figures/recognition_pca.png`, `figures/recognition_pca_target_spearman.png`

## 6. 三通道关系比较

不融合特征。共同成功样本上，距离矩阵上三角 Spearman：

```json
{json.dumps(cross, indent=2, ensure_ascii=False)}
```

图: `figures/cross_channel_distance_spearman.png`

说明：该矩阵只反映三套特征空间的相似关系是否一致，不是类别标签，也不是因果结论。

## 7. 异常与失败样本

- 损坏图片写入 manifest `status=corrupt`，不中断任务
- OCR失败见 `outputs/ocr_generations.parquet`（status!=ok）
- 视觉失败见 `outputs/visual_failures.parquet`（若存在）

## 8. 当前能够得出的结论

- 已分别提取并独立保存视觉 / 布局 / 识别三套特征
- 每套特征可单独做质量检查、PCA、kNN、UMAP 观察
- vLLM 承担批量 OCR/Markdown/grounding 生成；Transformers 通道只提取池化视觉向量
- 断点续跑与配置指纹校验已接入清单与生成缓存

## 9. 当前不能得出的结论

- UMAP团块不自动等于真实类别
- 相关性不自动等于因果关系
- 视觉embedding维度不能直接命名为具体视觉概念
- 当前阶段没有执行特征融合
- 当前阶段没有执行聚类
- 不得根据 UMAP 图主观声明类别
- 三通道 Spearman 高/低都不能直接推出“哪套特征更好用于训练划分”

## 10. 后续是否适合进入聚类阶段

在以下条件满足时可进入聚类预研（仍需单独实验设计）：

1. 三通道各自质量检查无大量 NaN/Inf，常数维可控
2. 联系表显示同通道近邻在视觉/版面/识别上具有可解释一致性
3. 明确选定**单一**特征通道（禁止未经验证的拼接融合）
4. 聚类目标与评估协议预先定义（不是事后根据 UMAP 命名）

当前版本**不执行聚类**，也**不生成 train/val/test**。
"""

    atomic_write_text(reports / "feature_analysis.md", md)
    proj = Path(cfg["paths"].get("project_reports_dir", "reports"))
    ensure_dir(proj)
    atomic_write_text(proj / "feature_analysis.md", md)
    return md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    build_report(cfg)
    print(f"[build_report] -> {cfg['paths']['reports_dir']}/feature_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
