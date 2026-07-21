"""Assemble feature_analysis.md report with hard quality-gate summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, model_resolved_path
from .quality_gates import run_quality_gates
from .utils import atomic_write_text, ensure_dir


def build_report(cfg: dict[str, Any]) -> str:
    reports = Path(cfg["paths"]["reports_dir"])
    out_dir = Path(cfg["paths"]["outputs_dir"])
    ensure_dir(reports)

    gates = run_quality_gates(cfg)
    summary = gates.get("summary") or {}
    failed = [c for c in gates.get("checks", []) if not c.get("ok")]

    intro = {}
    if (reports / "model_introspection.json").exists():
        intro = json.loads((reports / "model_introspection.json").read_text())
    metrics = {}
    if (reports / "feature_metrics.json").exists():
        metrics = json.loads((reports / "feature_metrics.json").read_text())
    bench = {}
    if (reports / "benchmark.json").exists():
        bench = json.loads((reports / "benchmark.json").read_text())

    gate_lines = "\n".join(
        f"- {'PASS' if c['ok'] else 'FAIL'}: `{c['name']}` — {json.dumps(c.get('detail'), ensure_ascii=False)}"
        for c in gates.get("checks", [])
    )
    status_banner = "PASS" if gates.get("pass") else "FAIL — see checks below"

    md = f"""# DeepSeek-OCR-2 Feature Analysis Report

## 0. 样本与质量验收（硬性）

**总状态: {status_banner}**

| 指标 | 数量 |
|------|------|
| 总图片数 | {summary.get("total_images")} |
| 可读图片数 | {summary.get("readable_images")} |
| 视觉有效数 | {summary.get("visual_valid")} |
| 布局总数 / 有效 | {summary.get("layout_total")} / {summary.get("layout_valid")} |
| 识别特征行数 / valid | {summary.get("recognition_total")} / {summary.get("recognition_valid")} |
| 截断 (truncated) | {summary.get("truncated")} |
| 重复生成 (repetitive) | {summary.get("repetitive")} |
| 空输出 (empty) | {summary.get("empty")} |
| 解析失败 (parse_failed) | {summary.get("parse_failed")} |

OCR 质量分布: `{json.dumps(summary.get("ocr_quality_status_counts", {}), ensure_ascii=False)}`

识别 PCA **仅**使用 `ocr_quality_status=valid`。质量字段不进入识别/布局/视觉特征矩阵。

### 检查项

{gate_lines}

失败项数: {len(failed)}（详见 `reports/quality_gates.json`）

---

## 1. 数据与模型配置

- 输入目录: `{cfg["data"]["input_dir"]}`
- 输出根目录: `{cfg["paths"]["output_root"]}`
- 模型: `{cfg["model"]["name_or_path"]}`
- revision: `{cfg["model"].get("revision")}`
- 解析路径: `{model_resolved_path(cfg)}`
- 架构类: `{intro.get("architecture_class")}`
- 视觉编码器: `{intro.get("visual_encoder_module_path")}`（层数={intro.get("visual_layer_total")}，选定层={intro.get("selected_layer_index")}）
- 提示词: `{cfg.get("prompt")}`
- 配置指纹: `{intro.get("config_fingerprint")}`

## 2. 四通道定义

| 通道 | 回答的问题 | 分析特征 | 过滤 |
|------|------------|----------|------|
| 视觉层 | 页面视觉形态、书写密度、笔迹与图形外观差异 | 单层视觉 embedding（L2，余弦） | 图片可读 |
| 布局层 | 内容如何排列 | 显式结构指标 | `layout_available` |
| 识别结果层 | 字符与数学表达结构 | 显式内容统计 | `ocr_quality_status=valid` |
| 质量层 | OCR 是否截断/重复/失败 | 仅过滤与诊断 | **不参与**特征 PCA |

## 3. 推理架构

- 实际架构: {intro.get("actual_inference_architecture")}
- vLLM 能否返回视觉 hidden states: `{intro.get("vllm_can_obtain_visual_hidden_states")}`
- Benchmark best config:

```json
{json.dumps(bench.get("best_config", {}), indent=2, ensure_ascii=False)}
```

## 4. 视觉层

```json
{json.dumps(metrics.get("channels", {}).get("visual", {}), indent=2, ensure_ascii=False)[:4000]}
```

- 图: `figures/visual_pca.png`, `figures/visual_umap.png`, `figures/visual_knn15.png`
- 诊断着色: `figures/visual_pca_by_*.png`（宽高比/覆盖率等为标签，不进向量）
- 联系表: `contact_sheets/visual_contact.png`, `visual_outliers.png`

## 5. 布局层

```json
{json.dumps(metrics.get("channels", {}).get("layout", {}), indent=2, ensure_ascii=False)[:4000]}
```

- 相关图为层次聚类、无满格数字；高相关对见 `outputs/analysis/layout_high_corr_pairs.parquet`
- 分布: `figures/layout_feature_hist.png`

## 6. 识别结果层（内容特征）

```json
{json.dumps(metrics.get("channels", {}).get("recognition", {}), indent=2, ensure_ascii=False)[:4000]}
```

- 内容相关: `figures/recognition_pca_content_spearman.png`（行=PC，列=内容指标）
- 质量相关（诊断，不参与训练特征）: `figures/recognition_pca_quality_spearman.png`
- 形态规则: `recognition_morphology_rules.json`

## 7. OCR 质量层

```json
{json.dumps(metrics.get("ocr_quality", {}), indent=2, ensure_ascii=False)}
```

- 状态分布图: `figures/ocr_quality_status.png`
- 明细: `outputs/ocr_quality.parquet`

## 8. 三通道关系（不融合）

```json
{json.dumps(metrics.get("cross_channel_distance_spearman"), indent=2, ensure_ascii=False)}
```

## 9. 不能得出的结论

- UMAP 空白/团块 ≠ 类别边界
- 相关 ≠ 因果
- 视觉维不可逐维命名
- 本阶段不做聚类与 train/val/test 划分
- 「OCR 接口成功」≠「有效识别」（以 `ocr_quality_status` 为准）

## 10. 后续聚类前提

1. 本节硬性检查尽可能 PASS（截断/重复率达标可能需提高 `max_tokens` 并重跑 OCR）
2. 识别 PC1 不再被质量指标以 |ρ|>0.8 支配（见 quality Spearman 图）
3. 选定单一通道后再设计聚类协议
"""

    atomic_write_text(reports / "feature_analysis.md", md)
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
