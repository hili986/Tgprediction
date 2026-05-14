# 全能 Tg 单一回归模型阶段总结（2026-04-25）

## 目标

本阶段目标是构建一个单一回归器，用同一套算法预测均聚物、一般共聚物和核酸/碱基共聚物 Tg，而不是按任务简单调用不同模型。当前路线为自研 `PhysicsResidualKernelRegressor`：

- 先用物理/组成特征拟合低频先验，包括端点 Tg、Fox 混合、组成熵、组分数和结构类型。
- 再用核残差学习结构非线性，使模型能利用组分 M2M-V 描述符。
- 对均聚物单独启用 186d 高维残差校正，但该校正仍在同一个回归器内部完成，不是外部分流。
- 对共聚物和核酸启用轻量局部残差，补偿小样本体系的局部偏差。

## 关键修正

1. 修正 PoLyInfo 组成解析。

`ratio_1=0.819 mol%` 这类行实际代表 81.9% 小数比例，旧逻辑无条件除以 100，导致组成被错当成 0.819%。现在规则为：`0 < ratio_1 < 1` 视为已归一化比例，其余数值按百分数处理。

2. 修正自研模型训练路径。

之前只有 `physics_kernel` 直接接收带列名的 DataFrame，其它自研变体被 `SimpleImputer` 包进 sklearn pipeline 后丢失列名，导致加性核、高维门控、物理先验选择都不能正确识别特征块。现在所有 `PhysicsResidualKernelRegressor` 变体都直接接收 DataFrame。

3. 修正核残差正则漏洞。

旧 ridge 求解默认把第一列当截距不惩罚，但核矩阵没有截距列，导致第一个 landmark 的核系数无正则。已改为：物理先验截距不惩罚，核残差和均聚物高维残差全部惩罚。

4. 剔除 PoLyInfo 纯端点行。

纯端点 Tg 已由 7k 均聚物数据库提供，PoLyInfo 中的 `w=0/1` 行存在明显冲突。训练一般共聚物时默认跳过这些纯端点行，避免污染共聚物泛化。

## 当前最佳结果

当前最佳实验为：

`results/universal_single_regressor/exp45_homo_local_nopure`

模型配置：

- model: `physics_homo_local_light`
- feature layer: `HYBRID-HOMO186`
- homopolymer: 7486 条
- PolyInfo real copolymer: 149 条，已剔除纯端点
- nucleobase real: 17 条
- virtual copolymer: 0 条
- weights: homopolymer=1, copolymer=10, nucleobase=60

| 任务 | 验证方式 | n | MAE (°C) | RMSE (°C) | R2 |
|---|---:|---:|---:|---:|---:|
| 均聚物 | random holdout | 1498 | 27.16 | 38.09 | 0.887 |
| 一般共聚物 | random holdout | 30 | 10.18 | 15.15 | 0.932 |
| 一般共聚物 | group holdout | 149 | 16.36 | 22.03 | 0.844 |
| 核酸/碱基共聚物 | group holdout | 17 | 6.85 | 9.15 | 0.789 |

说明：一般共聚物 random holdout 已经达到 `R2=0.932`，但更严格、更有意义的是按体系留出的 group holdout，目前为 `R2=0.844`。

## 重要对照实验

| 实验 | 主要变化 | 均聚物 R2 | 一般共聚物 group R2 | 核酸 group R2 | 结论 |
|---|---|---:|---:|---:|---|
| `exp35_fixedratio_kernel_penaltyfix` | M2M-V 单核物理残差 | 0.802 | 0.796 | 0.812 | 共聚物/核酸稳，但均聚物不足 |
| `exp41_fixedratio_local_light_direct` | 加轻局部残差 | 0.822 | 0.798 | 0.813 | 小幅改善，未解决均聚物 |
| `exp42_fixedratio_homo_correction_direct` | 加均聚物 186d 校正 | 0.864 | 0.796 | 0.812 | 均聚物显著提升且共聚物不退化 |
| `exp44_fixedratio_homo_local_light_direct` | 186d 校正 + 轻局部残差 | 0.886 | 0.798 | 0.813 | 三类指标最均衡 |
| `exp45_homo_local_nopure` | 剔除 PoLyInfo 纯端点 | 0.887 | 0.844 | 0.789 | 当前最佳；共聚物显著提升 |
| `exp49_homo_local_nopure_virtual5k_w005` | 加 5k 虚拟共聚物，权重 0.05 | 0.887 | 0.779 | 0.681 | 虚拟数据直接混入会伤害真实泛化 |

## 虚拟数据结论

本轮把 5000 条 Bicerano 虚拟随机共聚物以低权重 `0.05` 加入训练，结果一般共聚物 group R2 从 `0.844` 降到 `0.779`，核酸 group R2 从 `0.789` 降到 `0.681`。

这说明当前虚拟数据不能直接作为普通训练样本混入。更合理的用法应是：

- 只选与真实共聚物组成/端点 Tg 分布接近的虚拟样本。
- 对虚拟样本做 teacher uncertainty 或端点物理一致性过滤。
- 把虚拟数据作为预训练约束，而不是与真实数据同权或低权直接混合。
- 对虚拟标签只学习平滑趋势，不学习高频残差。

## 加性多核结论

加性多核思路值得保留，但当前简单实现没有超过主模型。

已经验证过两种方式：

- 拼接式加性核：每个核块拥有独立系数，容量过大，group holdout 严重过拟合。
- 求和式加性核：把物理核、结构核和全局核加权合成同一相似度矩阵，稳定性恢复，但指标没有超过 `physics_homo_local_light`。

因此下一步不应继续盲目增加核块，而应做“受约束的多核学习”：核权重由验证集或 leave-system-out 目标选择，并对真实共聚物/核酸单独设置公平指标约束。

## 推荐训练命令

当前最佳复现实验命令：

```bash
python -u scripts/train_universal_tg_single_regressor.py \
  --build-table \
  --table results/universal_single_regressor/unified_training_table_fixedratio_homo186_nopure.parquet \
  --output-dir results/universal_single_regressor/exp45_homo_local_nopure \
  --model physics_homo_local_light \
  --feature-layer HYBRID-HOMO186 \
  --max-virtual 0 \
  --virtual-weight 0 \
  --copolymer-weight 10 \
  --nucleobase-weight 60 \
  --group-eval
```

当前最佳模型路径：

```text
results/universal_single_regressor/exp45_homo_local_nopure/model.joblib
```

## 下一步

1. 做虚拟数据筛选，而不是全量混入。

优先筛选与真实 PolyInfo 共聚物端点 Tg、组成范围、组分结构距离接近的虚拟样本，并限制每个真实体系附近的虚拟样本数量。

2. 做 leave-system-out 导向的核权重搜索。

目标函数不看 overall holdout，而看三项公平指标：均聚物 holdout、一般共聚物 group holdout、核酸 group holdout。

3. 单独处理 PolyInfo 高误差体系。

当前一般共聚物误差主要集中在 P900015、P900012、P900008、P900025 等体系。这些体系需要人工核查比例方向、文献条件、结晶/半结晶影响和 Tg 测量方法差异。

4. 核酸数据需要扩充或增强。

核酸只有 17 条，单点如 `AT2012-T-32` 对 group R2 影响很大。短期应以 MAE 和 leave-base-out 稳定性为主，长期需要补充更多碱基共聚物实测数据。
