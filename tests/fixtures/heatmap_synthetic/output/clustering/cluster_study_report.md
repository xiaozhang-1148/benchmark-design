# 64×64 双特征轨聚类研究

## 处理流程
1. 读取灰度图 → 页面区域裁切 → 可选模板对齐 → 提取笔迹 mask
2. 生成 64×64 `d_abs` → `d_abs_smooth`（墨迹密度）/ `d_rel_smooth`（空间分布）
3. 分别展开为 4096 维 → PCA 保留 95% 方差
4. 各特征轨分别运行 K-means / GMM / HDBSCAN
5. 评价聚类结果 → 输出类中心热力图与代表原图

## 特征轨
| 目录 | 特征 | 含义 |
|------|------|------|
| abs_smooth/ | d_abs_smooth | 保留墨迹密度 |
| rel_smooth/ | d_rel_smooth | 只保留空间分布 |

## K-means
- k = [2, 3]
- 强制划分所有样本；需结合轮廓系数判断，不能只看类别数

## GMM
- 输出 AIC/BIC、后验概率、过渡样本

## HDBSCAN
- 自动发现类别数；大量噪声提示连续分布

## 评价摘要
### 墨迹密度 (d_abs_smooth)
- K-means 最佳轮廓系数: k=3 sil=0.4943

### 空间分布 (d_rel_smooth)
- K-means 最佳轮廓系数: k=3 sil=0.3829
