"""HTML report generation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Template

from heatmap_analysis.config import AnalysisConfig
from heatmap_analysis.utils import ensure_dir

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="{{ language }}">
<head>
<meta charset="utf-8"/>
<title>手写答题热力图分析报告</title>
<style>
body { font-family: "Segoe UI", Arial, sans-serif; margin: 2rem; line-height: 1.6; color: #222; }
h1,h2,h3 { color: #1a365d; }
table { border-collapse: collapse; margin: 1rem 0; width: 100%; max-width: 900px; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #edf2f7; }
img { max-width: 480px; margin: 0.5rem; border: 1px solid #ddd; }
.warn { background: #fffbeb; padding: 1rem; border-left: 4px solid #d69e2e; }
.note { background: #f0fff4; padding: 1rem; border-left: 4px solid #38a169; }
.grid { display: flex; flex-wrap: wrap; gap: 1rem; }
</style>
</head>
<body>
<h1>手写答题图像热力图统计与聚类分析报告</h1>
<p>生成时间：{{ generated_at }}</p>

<h2>1. 数据集概况</h2>
<ul>
<li>样本总数：{{ summary.total_images }}</li>
<li>成功提取热力图：{{ summary.extracted }}</li>
<li>空白答卷：{{ summary.blank_count }}</li>
<li>网格尺寸：{{ grid_size }}×{{ grid_size }}</li>
</ul>

<h2>2. 数据检查结果</h2>
{% if data_checks %}
<p>可读图片：{{ data_checks.readable_images }} / {{ data_checks.total_images }}</p>
<p>问题记录数：{{ data_checks.issue_count }}，重复文件组：{{ data_checks.duplicate_groups }}</p>
{% else %}<p>未运行预处理检查。</p>{% endif %}

<h2>3. 热力图计算方法</h2>
<div class="note">
<p>整张答卷图像映射到归一化坐标 x,y ∈ [0,1]，划分为 {{ grid_size }}×{{ grid_size }} 网格（不做页面裁切）。</p>
<p><strong>绝对密度</strong> D_abs[i,j] = 网格内笔迹权重之和 / 网格像素数</p>
<p><strong>相对分布</strong> D_rel[i,j] = D_abs[i,j] / sum(D_abs)，空白答卷单独标记。</p>
<p>提取模式：{{ extraction_mode }}。聚类使用：{{ clustering_grid_version }}。</p>
</div>

<h2>4. 数据集总体热力图</h2>
<div class="grid">
{% for img in aggregate_images %}
<img src="{{ img }}" alt="aggregate"/>
{% endfor %}
</div>

<h2>5. 单张答卷统计指标分布</h2>
{% if metrics_stats %}
<table>
<tr><th>指标</th><th>均值</th><th>标准差</th><th>中位数</th></tr>
{% for row in metrics_stats %}
<tr><td>{{ row.name }}</td><td>{{ "%.4f"|format(row.mean) }}</td><td>{{ "%.4f"|format(row.std) }}</td><td>{{ "%.4f"|format(row.median) }}</td></tr>
{% endfor %}
</table>
{% endif %}

<h2>6. 分组比较</h2>
{% if group_summaries %}
{% for field, groups in group_summaries.items() %}
<h3>分组字段：{{ field }}</h3>
<table>
<tr><th>组</th><th>样本数</th><th>平均覆盖率</th><th>平均熵</th></tr>
{% for gname, g in groups.items() %}
<tr><td>{{ gname }}</td><td>{{ g.n_samples }}</td><td>{{ "%.4f"|format(g.mean_ink_coverage) }}</td><td>{{ "%.4f"|format(g.mean_spatial_entropy) }}</td></tr>
{% endfor %}
</table>
{% endfor %}
{% else %}<p>未配置分组比较或未找到分组字段。</p>{% endif %}

<h2>7. 聚类数量选择依据</h2>
{% for cr in clustering_reports %}
<h3>模板组：{{ cr.template_group }}</h3>
<p>选定 k={{ cr.selected_k }}，PCA 累计解释方差={{ "%.3f"|format(cr.pca_cumulative_variance) }}，
Bootstrap 平均 ARI={{ "%.3f"|format(cr.stability.mean_ari) }}。</p>
{% if cr.k_selection_img %}<img src="{{ cr.k_selection_img }}" alt="k selection"/>{% endif %}
{% endfor %}

<h2>8. 聚类平均热力图</h2>
<div class="grid">
{% for img in cluster_images %}
<img src="{{ img }}" alt="cluster"/>
{% endfor %}
</div>

<h2>9. 聚类与外部变量关系</h2>
<div class="warn">
<p>以下相关关系仅为描述性统计，<strong>不得解释为因果关系</strong>。</p>
</div>
{% if cluster_score_table %}
<table>
<tr><th>聚类</th><th>样本数</th><th>平均得分</th><th>平均正确率</th></tr>
{% for row in cluster_score_table %}
<tr><td>{{ row.cluster }}</td><td>{{ row.n }}</td><td>{{ row.mean_score }}</td><td>{{ row.mean_correct }}</td></tr>
{% endfor %}
</table>
{% else %}<p>元数据中无得分/正确率字段或未运行聚类。</p>{% endif %}

<h2>10. 异常样本</h2>
{% if anomalies %}
<ul>
{% for k, v in anomalies.items() %}
<li>{{ k }}：{{ v[:10]|join(", ") }}{% if v|length > 10 %} ... (共{{ v|length }}个){% endif %}</li>
{% endfor %}
</ul>
{% endif %}

<h2>11. 分析限制与注意事项</h2>
<ul>
<li>无模板模式下，印刷内容可能被计入笔迹。</li>
<li>聚类标签仅描述空间布局模式，不代表能力或认知负荷。</li>
<li>dense_stroke_overlap_proxy 仅为笔画重叠代理，不等同于涂改率。</li>
<li>分组差异显著性掩膜基于逐网格 t 检验与 BH-FDR 校正，解释时需谨慎。</li>
</ul>
</body>
</html>
"""


def _rel_path(from_dir: Path, target: Path) -> str:
    try:
        return str(target.relative_to(from_dir))
    except ValueError:
        return str(target)


def generate_report(cfg: AnalysisConfig) -> Path:
    out = cfg.output.output_dir
    report_dir = ensure_dir(out / "report")

    summary = {"total_images": 0, "extracted": 0, "blank_count": 0}
    meta_csv = out / "tables" / "per_image_metrics.csv"
    metrics_stats = []
    if meta_csv.exists():
        df = pd.read_csv(meta_csv)
        summary["total_images"] = len(df)
        summary["extracted"] = len(df)
        summary["blank_count"] = int(df["is_blank"].sum()) if "is_blank" in df.columns else 0
        for col in ["ink_coverage", "spatial_entropy", "centroid_x", "centroid_y", "hotspot_concentration"]:
            if col in df.columns:
                metrics_stats.append(
                    {
                        "name": col,
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "median": float(df[col].median()),
                    }
                )

    data_checks = None
    dc_path = report_dir / "data_checks.json"
    if dc_path.exists():
        with dc_path.open("r", encoding="utf-8") as f:
            data_checks = json.load(f)

    aggregate_images = []
    agg = out / "aggregate"
    for p in sorted(agg.glob("*.png"))[:8]:
        aggregate_images.append(_rel_path(report_dir, p))

    group_summaries = {}
    groups_root = out / "groups"
    if groups_root.exists():
        for gf in groups_root.iterdir():
            sj = gf / "group_summary.json"
            if sj.exists():
                with sj.open("r", encoding="utf-8") as f:
                    group_summaries[gf.name] = json.load(f)

    clustering_reports = []
    cluster_images = []
    cluster_score_rows = []
    anomalies = {}
    cl_root = out / "clustering"
    if cl_root.exists():
        for tg in cl_root.iterdir():
            cr_path = tg / "clustering_report.json"
            if cr_path.exists():
                with cr_path.open("r", encoding="utf-8") as f:
                    cr = json.load(f)
                ks = tg / "k_selection.png"
                if ks.exists():
                    cr["k_selection_img"] = _rel_path(report_dir, ks)
                clustering_reports.append(cr)
            for cpng in sorted(tg.glob("cluster_*/mean_rel.png"))[:6]:
                cluster_images.append(_rel_path(report_dir, cpng))
            an_path = tg / "anomalies.json"
            if an_path.exists():
                with an_path.open("r", encoding="utf-8") as f:
                    anomalies = json.load(f)
            labels_csv = tg / "cluster_labels.csv"
            if labels_csv.exists() and meta_csv.exists():
                labels = pd.read_csv(labels_csv)
                mdf = pd.read_csv(meta_csv)
                merged = labels.merge(mdf, on="image_id", how="left")
                for cid, gdf in merged.groupby("cluster"):
                    row = {"cluster": int(cid), "n": len(gdf), "mean_score": "N/A", "mean_correct": "N/A"}
                    if "score" in gdf.columns:
                        row["mean_score"] = f"{gdf['score'].mean():.2f}"
                    if "correct" in gdf.columns:
                        row["mean_correct"] = f"{gdf['correct'].mean():.2f}"
                    cluster_score_rows.append(row)

    html = Template(REPORT_TEMPLATE).render(
        language=cfg.report.language,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary=summary,
        grid_size=cfg.heatmap.grid_size,
        data_checks=data_checks,
        extraction_mode="template_subtraction" if cfg.preprocessing.use_template_subtraction else "threshold",
        clustering_grid_version="d_abs_smooth + d_rel_smooth (dual track)",
        aggregate_images=aggregate_images,
        metrics_stats=metrics_stats,
        group_summaries=group_summaries,
        clustering_reports=clustering_reports,
        cluster_images=cluster_images,
        cluster_score_table=cluster_score_rows,
        anomalies=anomalies,
    )

    report_path = report_dir / "index.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
