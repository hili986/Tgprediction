# 深度学习与 GNN — 综合调研

> **合并来源**：
> - 深度学习方法预测聚合物Tg调研.md (2026-03-13)
> - 前沿方法调研.md (2026-03-07) — 经典方法演进与专题部分
> - 聚合物Tg预测前沿方法深度调研-2026-03-14.md (2026-03-14) — 最新进展增量
> **原始文件保留在** `docs/research/_archive/`
> **时效性**：有效
> **推荐阅读**：参考（方案B启动时重点阅读）

---

## 目录

1. [GNN 方法](#1-gnn-方法)
2. [Transformer / 序列模型方法](#2-transformer--序列模型方法)
3. [预训练模型](#3-预训练模型)
4. [多模态融合方法](#4-多模态融合方法)
5. [多尺度建模](#5-多尺度建模)
6. [物理引导 ML 方法](#6-物理引导-ml-方法)
7. [小数据集策略](#7-小数据集策略)
8. [LLM 直接预测](#8-llm-直接预测)
9. [共聚物 Tg 预测](#9-共聚物-tg-预测)
10. [特殊体系 Tg 预测](#10-特殊体系-tg-预测)
11. [BigSMILES 相关方法](#11-bigsmiles-相关方法)
12. [生成式设计与逆设计](#12-生成式设计与逆设计)
13. [方法演进时间线](#13-方法演进时间线)
14. [综合性能对比](#14-综合性能对比)
15. [项目建议](#15-项目建议)
16. [参考文献](#16-参考文献)

---

## 1. GNN 方法

### 1.1 聚合物图表示方法

| 表示方法 | 描述 | 优势 | 劣势 |
|----------|------|------|------|
| **Monomer Graph** | 仅以单体/重复单元构建图 | 简单、通用 | 丢失链级信息 |
| **Repeat Unit Graph** | 重复单元 + 链接点标记 | 保留聚合位点 | 无法表示周期性 |
| **Periodic Polymer Graph** | 在边界原子间添加键形成环 | 捕捉周期性 | 计算复杂 |

关键洞察：periodic polymer graph 通过在重复单元头尾原子间添加化学键，模拟聚合物链的周期性，类似固体物理中的周期性边界条件 (PBC)。

### 1.2 代表性论文

#### polyGNN (Ramprasad Group, 2023)
- **发表**: J. Phys. Chem. Lett., 2023
- **方法**: Periodic graph + message passing + 多任务学习 (Tg, Tm, 密度)
- **性能**: 单任务 RMSE=36.6K; 多任务 RMSE=31.7K (提升 13.4%)
- **小数据适用**: ★★★☆☆ — 多任务显著提升小数据性能，但 GNN 仍需 ~1000+ 数据

#### PolymerGNN (Tao et al., 2023)
- **发表**: J. Chem. Inf. Model., 2023
- **方法**: Periodic graph + MPNN + Deep Set (处理共聚物多单体)
- **性能**: R2 = 0.89 (1,251 条测试集)
- **小数据适用**: ★★☆☆☆ — 纯 GNN，需较大数据集

#### Data-Augmented GCN (Yamada et al., 2023)
- **发表**: ACS Applied Materials & Interfaces, 2023
- **方法**: GCN + SMILES enumeration 数据增强
- **性能**: 无增强 R2=0.88, RMSE=19.4K; **有增强 R2=0.97, RMSE=7.4K** (RMSE 降 62%)
- **小数据适用**: ★★★★★ — SMILES enumeration 是小数据最有效增强策略

#### GRIN: 重复不变图表示 (arXiv, 2025) [高度创新]
- **核心**: 解决标准 GNN 混淆"链更长"与"化学不同"的根本问题
- **方法**: Max-aggregation + MST 对齐 + 1/3-RU 数据增强
- **理论**: 3 个重复单元 = 最小充分增强 (数学证明)
- **性能**: R2=0.896; 60-RU 泛化 R2=0.895; 嵌入余弦相似度 0.999
- **对比**: 基线 GCN 0.882; DISGEN 等在 60-RU 上 R2 为负

#### Lieconv-Tg (2024)
- **方法**: 3D 分子坐标 + Lie 群等变神经网络，考虑旋转/平移不变性
- **性能**: R2=0.90, MAE=24.42K (7,166 条数据)
- **小数据适用**: ★★☆☆☆ — 需要 3D 坐标生成和大数据集

#### GATBoost (2024)
- **方法**: Graph Attention Network 提取子结构特征 → gradient boosting 输入
- **性能**: R2=0.91, 可解释性优于纯 GNN
- **小数据适用**: ★★★☆☆ — 混合架构对小数据更友好

#### Chem-DAGNN (ICML 2024 Workshop)
- **方法**: 有向无环图编码聚合物合成路径
- **性能**: R2=0.95, RMSE=28°C
- **特点**: DAG 结构天然适合表示合成过程

#### MSRGCN-RL (Langmuir, 2024)
- **方法**: 多尺度关系图网络 + 强化学习自动架构搜索
- **小数据适用**: ★★☆☆☆ — 架构复杂

#### Self-Supervised GNN (2024)
- **方法**: Node/Edge/Graph 三级自监督预训练 → Tg 微调
- **小数据适用**: ★★★★☆ — 无标签预训练可利用无标签数据

#### KA-GNN (Nature Machine Intelligence, 2025)
- **创新**: 用 Kolmogorov-Arnold 网络替代传统 MLP 层
- **适用**: 通用聚合物性质预测

### 1.3 GNN 方法性能总结

| 方法 | 年份 | 图表示 | 最佳性能 | 数据量 |
|------|------|--------|---------|--------|
| polyGNN | 2023 | Periodic | RMSE=31.7K | ~5000 |
| PolymerGNN | 2023 | Periodic+DeepSet | R2=0.89 | ~6000 |
| Data-Aug GCN | 2023 | Monomer+Aug | **R2=0.97** | ~500+aug |
| GRIN | 2025 | 重复不变 | R2=0.896 | 7174 |
| Lieconv-Tg | 2024 | 3D coords | R2=0.90 | 7166 |
| GATBoost | 2024 | GAT+XGBoost | R2=0.91 | 6500 |
| Chem-DAGNN | 2022-24 | DAG | R2=0.95 | PolyInfo |

**关键结论**:
1. 小数据场景下，SMILES enumeration 数据增强最有效 (Data-Aug GCN: R2 0.88→0.97)
2. 304 条下纯 GNN 不如树模型 (Kaggle 2025 实证)
3. GNN 嵌入 → 传统 ML 是更务实的混合策略
4. GRIN 的重复不变思想可启发传统描述符设计

---

## 2. Transformer / 序列模型方法

### 2.1 代表性论文

#### TransPolymer (Xu et al., npj Computational Materials 2023)
- **方法**: RoBERTa + PSMILES tokenizer, 5M PSMILES 预训练
- **性能**: Tg MAE=32.9K (5549 条, 80/10/10); 10 个聚合物性质 benchmark SOTA
- **小数据适用**: ★★★★☆ — 预训练+微调范式天然适合小数据

#### TransTg (2024)
- **方法**: Tg 专用 Transformer, SMILES 字符级 tokenization, 无预训练
- **性能**: R2=0.849, MAE=22.55K
- **小数据适用**: ★★★☆☆ — 无预训练，小数据上有限

#### Multimodal Transformer (ACS AMI, 2024)
- **方法**: 双通道 (SMILES 序列 + 分子图) + cross-attention 融合
- **性能**: Tg 优于单模态; Dimer configuration 最优
- **小数据适用**: ★★★☆☆ — 多模态增加数据需求

#### polyBART (EMNLP, 2025)
- **方法**: 首个双向聚合物语言模型 (encoder-decoder), 使用 PSELFIES
- **能力**: 结构→性质 AND 性质→结构 (inverse design)
- **创新**: 首次 ML 设计聚合物→合成→实验验证闭环
- **小数据适用**: ★★★★☆ — 双向能力提供额外训练信号

### 2.2 Transformer 方法对比

| 方法 | 年份 | 预训练 | Tg 性能 | 关键特色 |
|------|------|--------|---------|----------|
| TransPolymer | 2023 | 5M PSMILES | R2~0.85, MAE=32.9K | 10 性质 SOTA |
| TransTg | 2024 | 无 | R2=0.849, MAE=22.55K | Tg 专用 |
| Multimodal Trans | 2024 | 部分 | 优于单模态 | SMILES+图融合 |
| polyBART | 2025 | 是 | 多性质竞争力 | 双向 (结构↔性质) |

**关键结论**: 预训练是 Transformer 在小数据集上成功的关键。

---

## 3. 预训练模型

### 3.1 代表性模型

#### polyBERT (Nature Communications, 2023)
- **架构**: DeBERTa, 1 亿条 PSMILES 预训练, 768-dim fingerprint
- **性能**: 29 个性质平均 R2~0.80; Tg R2=0.82-0.92 (依赖评估方式); 推理速度比 PG FP 快 215 倍
- **小数据适用**: ★★★★★ — fingerprint 直接可用，无需训练神经网络
- **实用方案**: polyBERT fingerprint + ExtraTrees 是最直接可用的 DL 增强方案

#### ChemBERTa (2020, 持续更新至 2024)
- **架构**: RoBERTa, 7700 万条 PubChem SMILES 预训练
- **性能**: Uni-Poly benchmark 中最佳单模态 Tg 预测器
- **局限**: 训练在小分子上，聚合物 `[*]` 标记可能不在词汇表中
- **小数据适用**: ★★★★☆

#### PolyCL (Digital Discovery, 2024-2025)
- **方法**: 对比学习预训练 — SMILES enumeration + masking + dropping 生成正对
- **性能**: 7 性质平均 R2=0.7897 > polyBERT(0.7775) > TransPolymer(0.7830)
- **鲁棒性**: 非标准数据仅下降 2.6% (vs polyBERT 8.3%)
- **训练**: 1M 聚合物, 8xV100, ~22h
- **小数据适用**: ★★★★★ — 无标签预训练 + 对比学习天然适合小数据

#### MMPolymer (CIKM 2024)
- **方法**: 3D 构象 + SMILES 双模态; MLM + 3D denoising + cross-modal alignment
- **性能**: 7/8 数据集 SOTA
- **小数据适用**: ★★★☆☆ — 需要 3D 构象数据

### 3.2 预训练模型对比

| 模型 | 预训练数据 | 架构 | Tg R2 | 聚合物专用 | 开源 |
|------|-----------|------|---------|-----------|------|
| polyBERT | 100M PSMILES | DeBERTa | ~0.82-0.92 | ✔ | ✔ |
| ChemBERTa | 77M PubChem | RoBERTa | 最佳单模态 | ✘ | ✔ |
| PolyCL | 1M 无标签聚合物 | Contrastive | 4/7 最佳 | ✔ | ✔ |
| MMPolymer | 聚合物+3D | Multimodal | 7/8 SOTA | ✔ | 部分 |

**关键结论**: polyBERT fingerprint 是最实用方案 — 预训练 embedding 直接作为传统 ML 输入，不增加模型复杂度。

---

## 4. 多模态融合方法

### 4.1 代表性方法

#### Uni-Poly (npj Computational Materials, 2025)
- **5 模态统一表示**: SMILES 文本 + 2D 图 + 3D 几何 + 指纹 + 文本描述
- **创新**: Poly-Caption 10K+ 文本，文本描述提供结构表示无法捕捉的互补信息
- **性能**: 多 benchmark SOTA

#### GeoALBEF (ACS, 2025)
- **方法**: "先对齐，再融合" (Align before Fuse)，分别编码 SMILES 和 3D 几何
- **性能**: RMSE 降低 8.6% (相比单模态)
- **灵感**: 来自视觉-语言预训练 (ALBEF)

#### PolyLLMem (arXiv, 2025)
- **方法**: LLaMA-3(4096d) + Uni-Mol(1536d) + LoRA + 门控融合
- **性能**: Tg R2=0.89 (5-fold CV, ~6000 条); 无需百万级预训练
- **小数据适用**: ★★★★☆ — LLM 常识可弥补数据不足

### 4.2 多模态方法对比

| 方法 | 年份 | 模态 | 核心创新 | 小数据适用 |
|------|------|------|---------|-----------|
| Uni-Poly | 2025 | 5 模态 | 统一多模态 + 文本描述 | ★★★ |
| GeoALBEF | 2025 | 语言+3D | Align before Fuse | ★★★ |
| PolyLLMem | 2025 | LLM+3D | LLM 常识增强 | ★★★★ |
| MMPolymer | 2024 | SMILES+3D | Star Substitution | ★★★ |

**趋势**: 多模态融合 (文本+图+3D) 是 2025-2026 年主流方向。

---

## 5. 多尺度建模

### 5.1 核心挑战

聚合物 Tg 受多尺度因素影响: 原子级 (官能团、键角) → 单体级 (重复单元结构) → 链级 (分子量、缠结) → 宏观级 (结晶度、共混)。传统描述符通常只捕捉 1-2 个尺度。

### 5.2 代表性方法

#### HAPPY (2024)
- **方法**: 层次化聚合物表示: 原子 → 官能团 → 重复单元 → 聚合物
- **性能**: R2=0.86 (仅 214 条数据)
- **小数据适用**: ★★★★★ — 专为小数据设计

#### Mol-TDL (arXiv, 2024)
- **方法**: 用 simplicial complexes 替代传统图，自然编码多体相互作用
- **理论优势**: 图无法表达的 3-body+ 相互作用
- **小数据适用**: ★★☆☆☆ — 前沿方法，实现复杂

#### 多尺度 ML (J. Phys. Chem. B, 2025)
- **方法**: 预测整个玻璃化转变区间 (非单点 Tg)
- **特点**: 微观+介观+宏观三尺度特征

**关键结论**: 对于 ~300 条数据，HAPPY 的层次化表示最值得关注 — 专门在 214 条数据上验证了 R2=0.86。

---

## 6. 物理引导 ML 方法

### 6.1 TrinityLLM (Nature Computational Science, 2025) [关键]
- **核心**: LLM + 物理建模 + 实验 = 三位一体
- **方法**: GC 法合成数据预训练 (3237 假想聚合物) → 实验微调
- **性能**: 点火时间 -25.6%, 峰值热释放率 -51.2%
- **代码**: github.com/ningliu-iga/TrinityLLM
- **与本项目关系**: **与 Fox 虚拟数据策略高度一致! Nature 验证了方向正确性**

### 6.2 GC-GNN (Macromolecules, 2025) [真正原创]
- **方法**: 高斯链理论嵌入 GNN (Tandem 模型)
- **数据集**: ToPoRg-18k
- **优势**: 迁移性超纯物理+纯 GNN; 系数与溶剂疏水性高度相关 (可解释)
- **启示**: ML 学物理残差 = 方案 A 的 GC 残差学习思路

### 6.3 PerioGT (Nature Computational Science, 2025)
- **方法**: 周期性先验 + 对比学习 + 周期提示微调
- **特点**: 数据稀缺下表现优异

### 6.4 GC+ML 混合模型 (2023-2024)
- **思路**: 传统基团贡献法 (Van Krevelen/Bicerano) 作为 ML 基线或特征
- **方法**: GC 贡献值作为输入 + ML 学习残差
- **优势**: 可解释性好，外推能力较强

### 6.5 物理引导方法对比

| 方法 | 物理先验类型 | 主要优势 |
|------|-------------|---------|
| TrinityLLM | GC 合成数据 | Nature 验证，与 Fox 策略一致 |
| GC-GNN | 高斯链理论 | 外推能力，可解释性 |
| PerioGT | 周期性先验 | 小数据优异 |
| GC+ML | 基团贡献值 | 可解释，残差学习 |
| PINN | 热力学/动力学 PDE | 理论严谨 (早期阶段) |

---

## 7. 小数据集策略

### 7.1 策略概览

#### A: 预训练表示 (最低风险, 最高性价比)
- polyBERT Fingerprint + 传统 ML: 768-dim embedding → RF/ET/GBR
- PolyCL Contrastive Fingerprint: 无标签预训练
- 适用性: ★★★★★

#### B: 数据增强
- **SMILES Enumeration**: 同一分子多个等价 SMILES，扩大 10-100 倍。Data-Aug GCN: R2 0.88→0.97
  - **注意**: 对 DL 有效，对树模型有害 (Kaggle 2025 实证)
- **Fox 方程虚拟数据**: 304 均聚物 → 46,000+ 共聚物，两阶段训练
- **重复单元增强 (GRIN 策略)**: 生成二聚体/三聚体 SMILES 分别算描述符; 3-RU=最小充分; **低成本可立即实施**

#### C: 迁移学习
- **Shotgun TL (XenonPy)**: 140,000+ 预训练模型，自动选择最佳源; 极小数据 (10-50 条) 也能工作
- **Cross-Property TL**: 69% 情况迁移优于从零训练; Tg 作为源/目标任务均有效

#### D: 多任务学习
- 同时预测 Tg + Tm + 密度; polyGNN: RMSE 从 36.6K 降至 31.7K (13.4%)
- CoPolyGNN: 辅助任务 +50% 提升

#### E: Few-Shot / Meta-Learning
- 在聚合物领域尚不成熟; 适用性 ★★☆☆☆

### 7.2 策略推荐排序 (针对 ~300 条数据)

| 优先级 | 策略 | 预期提升 | 实现难度 | 推荐度 |
|--------|------|---------|---------|--------|
| 1 | polyBERT FP + ExtraTrees | R2 +0.03-0.05 | 低 | ★★★★★ |
| 2 | 重复单元增强 (GRIN) | R2 +0.01-0.03 | 低 | ★★★★★ |
| 3 | Fox 虚拟数据预训练 | R2 +0.03-0.05 | 中 (已实现) | ★★★★★ |
| 4 | Cross-property TL | R2 +0.02-0.05 | 中 | ★★★★☆ |
| 5 | 对比学习 (PolyCL) | R2 +0.03-0.05 | 中高 | ★★★★☆ |
| 6 | 多任务学习 | R2 +0.02-0.04 | 高 | ★★★☆☆ |

**核心建议**: 不要从零训练深度学习模型。最有效的策略是用 DL 生成的 embedding 或增强数据来加强传统 ML。

### 7.3 Fox 虚拟数据 vs TrinityLLM 对比

| 维度 | TrinityLLM (Nature 2025) | 本项目 Fox 方案 |
|------|-------------------------|--------------|
| 来源 | GC 法假想聚合物 | Fox 方程虚拟共聚物 |
| 数量 | 3,237 | 46,000+ |
| 精度 | MAE 30-50K | MAE 10-30K |
| 策略 | 预训练→微调 | 两阶段训练 (已实现) |

---

## 8. LLM 直接预测

### 8.1 Benchmark 结果

| 模型 | Tg RMSE (K) | 相对表现 |
|------|------------|---------|
| GPT-3.5 | 47.2 | 最差 |
| LLaMA-3 | 39.48 | 稍好 |
| GNN (from scratch) | 31-37 | 明显更好 |
| polyBERT + ML | ~28-32 | 更好 |
| Data-Aug GCN | 7.4 | 最佳 |

### 8.2 分析

- LLM 对化学 SMILES 的理解是隐式的，缺乏显式化学知识编码
- 定量回归不是 LLM 强项 (vs 分类/排序)
- LLM 的价值在于: 化学常识增强 (PolyLLMem)、inverse design (polyBART)、数据提取 (MatBERT)

**结论**: 不推荐直接用 LLM 做 Tg 回归。LLM 的价值在辅助能力，非替代专用模型。

---

## 9. 共聚物 Tg 预测

### 9.1 经典方程局限

共聚物 Tg 传统依赖 Fox/Gordon-Taylor/Couchman-Karasz/Kwei 方程，但均有重大局限：
- 仅适用于理想无规共聚物
- 无法处理序列效应 (嵌段 vs 无规 vs 交替)
- 忽略共聚单体间相互作用
- 不考虑分子量效应

### 9.2 ML 方法进展

#### WC-SMILES (ACS Appl. Polym. Mater., 2024)
- **方法**: Weighted-Chained SMILES 表示，将共聚物编码为加权单体链
- **性能**: R2 ~0.85-0.921 (优于 Fox 方程)
- **适用**: 无规/嵌段/交替共聚物
- **地位**: 目前共聚物 Tg 预测最强对标方法

#### CoPolyGNN (ECML PKDD, 2025)
- **方法**: GNN + 注意力 + 辅助任务; 性能提升 ~50%
- **特点**: 首个专门针对共聚物的 GNN

#### CNN-LSTM + WGAN-GP (2025)
- **方法**: NLP 处理共聚物序列 + 生成对抗网络增强
- **性能**: R2=0.95

#### 嵌段共聚物 Graph Kernel (Angew. Chem., 2025)
- **方法**: Graph kernel 预测生物基嵌段共聚物热塑性弹性体性质

### 9.3 不同共聚物类型的预测现状

| 共聚物类型 | ML 方法数量 | 数据集规模 | 研究成熟度 |
|-----------|-----------|-----------|-----------|
| 无规共聚物 | 2-3 方法 | <1,000 | 起步 |
| 嵌段共聚物 | 几乎没有 | <100 | 空白 |
| 交替共聚物 | 几乎没有 | <50 | 空白 |
| 接枝共聚物 | 0 | 0 | 空白 |

### 9.4 核心研究空白

1. BigSMILES 对共聚物序列信息的编码效果未经验证
2. BigSMILES vs WC-SMILES vs pSMILES 在共聚物上的系统对比不存在
3. 嵌段共聚物的多 Tg 预测 (微相分离) 几乎没有 ML 工作
4. 组成-序列-Tg 关系的 ML 建模方法论缺失

---

## 10. 特殊体系 Tg 预测

### 10.1 各特殊体系概览

| 体系 | 数据量 | 主要挑战 | Tg R2 | 代表工作 |
|------|--------|---------|-------|---------|
| 可降解聚合物 | ~500 | 降解效应 | 0.85 | Kim et al., Green Chem. 2024 |
| 导电聚合物 | ~200 | 共轭效应 | 0.78 | Li et al., Adv. Funct. Mater. 2024 |
| 聚合物共混物 | ~800 | 相分离/双 Tg | 0.82 | Pilania et al., Macromolecules 2023 |
| 高温聚合物 | ~1,200 | 高温区误差大 | 0.87 | Chen et al., Polymer 2024 |
| 热固性聚合物 | <500 | 交联密度 | 0.80-0.87 | 多尺度 ML 2024; VAE 2025 |
| 核酸 | <30 | 数据极少 | N/A | **文献空白** |

### 10.2 核酸 Tg 的特殊挑战

- 核酸研究主要关注 Tm (熔解温度) 而非 Tg
- 干态 DNA 薄膜有可测 Tg，强烈依赖含水量
- DNA-聚合物共轭体系热性质研究处于起步阶段
- **核酸材料的 Tg 预测是文献空白 — 论文创新价值**

| 维度 | 合成聚合物 | 核酸 |
|------|----------|------|
| 单体 | 1-2 种 | 4 种 |
| 序列 | 统计 | 精确 |
| 主链 | 碳/醚 | 磷酸二酯 |
| 作用力 | VdW/氢键 | 碱基堆积+氢键+磷酸 |
| Tg 数据 | 丰富 | 极少 |

---

## 11. BigSMILES 相关方法

### 11.1 BigSMILES vs SMILES 系统评估

- **来源**: Qiu & Sun, Digital Discovery 2024 / Macromolecules 2025
- **核心发现**:
  - BigSMILES 因 token 复杂度低，训练速度更快
  - 均聚物上两者表现相当
  - **共聚物场景下 BigSMILES 信息编码更准确**
  - 共聚物方向评估仍不充分 — 研究机会

### 11.2 BigSMILES 核心优势

1. **编码随机性**: 天然匹配高分子的随机特性 (一个字符串 = 一个分子集合)
2. **紧凑性**: 比 pSMILES 更短，LLM 训练 token 更少
3. **共聚物信息**: 能编码单体连接方式 ($$ 类型、<> 定向性)

### 11.3 G-BigSMILES (Digital Discovery, 2024)
- 扩展 BigSMILES 以包含反应性比、分子量分布、集合大小
- 概念验证阶段，尚未完整实验验证

### 11.4 BigSMILES 规范化
- Lin et al., 2022: 有确定主链的高分子规范化方案
- 支化/网络结构规范化仍是开放问题

---

## 12. 生成式设计与逆设计

### 12.1 方法概览

| 方法 | 代表工作 | 目标 | 有效结构率 |
|------|---------|------|-----------|
| **VAE** | 共聚物逆设计 (Chem. Sci. 2025) | 生成新单体+配比 | 100% |
| **CharRNN** | 聚合物 Tg 逆设计 | 生成高 Tg 聚合物 | ~60% |
| **REINVENT** (RL) | 强化学习引导 | Tg 目标优化 | ~80% |
| **GraphINVENT** | 图生成 | 分子图直接生成 | ~90% |
| **Transformer** | polyGT (2025) | 通用高分子设计 | 100% (Group SELFIES) |
| **Diffusion** | 散布生成式 (2025) | 从性质到加工条件 | 前沿 |

### 12.2 关键空白

基于 BigSMILES 语法约束的生成模型仍处于概念阶段。BigSMILES-guided VAE/Transformer 是重要研究机会。

---

## 13. 方法演进时间线

```
2002 ── Bicerano 基团贡献法教科书
2019 ── BigSMILES 提出 (Lin et al., ACS Central Science)
2021 ── 79 种 ML 模型 benchmark (Tao et al., JCIM)
2022 ── Chem-DAGNN: R2=0.95 (ACS Polym. Au)
     ── BigSMILES 规范化 (Lin, ACS Polymers Au)
2023 ── polyGNN 多任务学习 (J. Phys. Chem. Lett.)
     ── polyBERT: R2=0.92, 100M 预训练 (Nature Commun.)
     ── TransPolymer (npj Comp. Mater.)
     ── Data-Aug GCN: R2=0.97 (ACS AMI)
     ── Lieconv-Tg: 3D 等变网络 (Digital Discovery)
     ── Shotgun Transfer Learning (ACS Omega)
2024 ── WCS 共聚物框架: R2=0.921 (ACS APM)
     ── BigSMILES vs SMILES 12 任务 benchmark
     ── GATBoost: GAT + XGBoost (npj)
     ── GC-GNN: 物理引导 (JCIM)
     ── MMPolymer: 多模态 (CIKM)
     ── PolyCL: 对比学习 (Digital Discovery)
     ── G-BigSMILES (Digital Discovery)
     ── Locluster: QC 增强少样本
     ── TabPFN v2: <10K 超越 GBDT (arXiv)
2025 ── TrinityLLM: GC 预训练 (Nature Comp. Sci.)
     ── GC-GNN 跨架构迁移 (Macromolecules)
     ── PerioGT: 周期性先验 (Nature Comp. Sci.)
     ── GRIN: 重复不变 (arXiv)
     ── Uni-Poly: 5 模态统一 (npj)
     ── PolyLLMem: LLM+3D (arXiv)
     ── polyBART: 双向语言模型 (EMNLP)
     ── Afsordeh: 4 描述符 R2=0.97 (Chinese J. Polym. Sci.)
     ── Open Polymer Challenge 10,000+ 参赛 (NeurIPS/Kaggle)
     ── KA-GNN (Nature Mach. Intell.)
     ── CoPolyGNN (ECML PKDD)
     ── PolyMetriX 7,367 标准化 Tg (npj)
     ── TabM: MLP 集成匹配 GBDT (ICLR)
2026 ── OPoly26: 657 万 DFT 计算 (arXiv)
```

---

## 14. 综合性能对比

### 14.1 按方法类别

| 类别 | 代表方法 | 最佳 Tg R2 | 优势 | 劣势 |
|------|---------|-----------|------|------|
| 指纹+树模型 | Morgan+XGBoost | 0.88 | 简单、快速 | 缺乏序列信息 |
| 结构特征+ML | ET+GPR | 0.97* | 最高精度、可解释 | 需专家设计特征 |
| GNN | GATBoost, GRIN | 0.91-0.95 | 自动学结构特征 | 需图构建 |
| Transformer | polyBERT, MMPolymer | 0.92-0.94 | 端到端 | 需预训练 |
| 多模态 | Uni-Poly, PolyLLMem | 0.89-0.94 | 多源信息 | 复杂度高 |
| 物理引导 | GC-GNN, TrinityLLM | 0.90 | 外推好 | 需领域知识 |
| 迁移学习 | 自监督预训练 | +10-15% | 小数据友好 | 依赖源任务 |
| LLM 直接 | GPT/LLaMA 微调 | ~0.80 | 多任务 | 数值精度差 |

*注: Afsordeh R2=0.97 基于 90/10 split + 12 测试样本

### 14.2 按聚合物类型覆盖

| 聚合物类型 | ML 方法数量 | 最佳 R2 | 研究成熟度 |
|-----------|-----------|---------|-----------|
| 均聚物 | 30+ 方法 | 0.97 | 非常成熟 |
| 无规共聚物 | 2-3 方法 | ~0.92 | 起步 |
| 嵌段共聚物 | 几乎没有 | N/A | 空白 |
| 交替/接枝 | 0-1 | N/A | 空白 |
| 共混物 | 2-3 | 0.82 | 起步 |
| 交联聚合物 | 1-2 | ~0.87 | 起步 |
| 核酸杂化 | 0 | N/A | **空白** |

### 14.3 SOTA 排行榜 (按评估严格程度加权)

| 排名 | 方法 | R2 | 数据集 | 评估方式 | 年份 |
|------|------|-----|--------|---------|------|
| 1 | Afsordeh ET | 0.97 | 112 | 90/10 split | 2025 |
| 2 | RF + 物理描述符 | 0.98 | ~200 | split | 2024 |
| 3 | GRIN + GIN | 0.896 | 7174 | 80/10/10 | 2025 |
| 4 | Lieconv-Tg | 0.90 | 7166 | 80/20 | 2024 |
| 5 | Uni-Poly | ~0.90 | 7174 | CV | 2025 |
| 6 | PolyLLMem | 0.89 | ~6000 | 5-fold CV | 2025 |
| 7 | Open Polymer Top-1 | 0.86 | 3985 | 5-fold CV | 2025 |

**关键洞察**: R2>0.90 的方法要么使用弱评估 (简单 split)，要么使用大数据集 (>5000)。严格 CV + 小数据集下 R2=0.86 是更可靠的上界。

---

## 15. 项目建议

### 15.1 针对本项目的深度学习整合方案

基于 304 条 Bicerano + ExtraTrees 基线 (R2=0.837)：

**策略 A (推荐优先)**: polyBERT Fingerprint 增强
- 现有物理特征 (34d) + polyBERT embedding (768d) → 降维 (PCA 50-100d) → Stacking
- 预期: R2 0.84→0.87-0.89 | 难度: 低 | 风险: 低

**策略 B**: 重复单元增强 + Fox 虚拟预训练
- 已有基础 (Fox 46K)，加入 GRIN 重复单元增强
- 预期: R2 0.84→0.87-0.90 | 难度: 低-中

**策略 C**: GNN 嵌入 (方案B 启动时)
- GNN 在 GPU 服务器上训练，提取嵌入 → 传统 ML
- 预期: R2 0.84→0.90-0.95 | 难度: 高

### 15.2 最具论文价值的路径

1. GC 残差学习 (对标 GC-GNN, Macromolecules 2025)
2. MAPIE UQ (对标 PolUQBench, JCIM 2025)
3. 核酸 Tg 迁移 (文献空白)
4. Fox 合成预训练 (对标 TrinityLLM, Nature 2025)
5. 重复单元增强 (对标 GRIN, 2025)

### 15.3 不推荐
- 从头训练大型 DL 模型
- 全栈多模态方法
- 对树模型做 SMILES 增强
- DNA 聚合物共轭体系 Tg (数据极少)

---

## 16. 参考文献

### GNN 方法
1. polyGNN — Gurnani et al., J. Phys. Chem. Lett., 2023
2. PolymerGNN — Tao et al., J. Chem. Inf. Model., 2023
3. Data-Augmented GCN — Yamada et al., ACS AMI, 2023
4. Lieconv-Tg — Aldeghi et al., Digital Discovery 2023 / ACS Omega 2024. https://pubs.acs.org/doi/10.1021/acsomega.3c06843
5. MSRGCN-RL — Langmuir, 2024. https://pubs.acs.org/doi/10.1021/acs.langmuir.4c01906
6. GATBoost — Park et al., npj Comput. Mater., 2024
7. Self-Supervised GNN — Wang et al., npj Comput. Mater., 2024
8. GRIN — arXiv, 2025. https://arxiv.org/abs/2505.10726
9. Chem-DAGNN — Hong et al., ACS Polym. Au, 2022 / ICML 2024 Workshop
10. KA-GNN — Nat. Mach. Intell., 2025. https://www.nature.com/articles/s42256-025-01087-7
11. CoPolyGNN — ECML PKDD, 2025. https://link.springer.com/chapter/10.1007/978-3-032-06118-8_25

### Transformer / 序列模型
12. TransPolymer — Xu et al., npj Comp. Mater., 2023. https://www.nature.com/articles/s41524-023-01016-5
13. TransTg — 2024
14. Multimodal Transformer — ACS AMI, 2024
15. polyBART — EMNLP, 2025. https://arxiv.org/abs/2506.04233

### 预训练模型
16. polyBERT — Kuenneth & Ramprasad, Nature Communications, 2023. https://www.nature.com/articles/s41467-023-39868-6
17. ChemBERTa — Chithrananda et al., 2020 (updated 2024)
18. PolyCL — Digital Discovery, 2024. https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00236a
19. MMPolymer — CIKM 2024. https://dl.acm.org/doi/10.1145/3627673.3679684

### 多模态 / 多尺度
20. Uni-Poly — npj Comp. Mater., 2025. https://www.nature.com/articles/s41524-025-01652-z
21. GeoALBEF — ACS, 2025. https://pubs.acs.org/doi/10.1021/cbe.5c00057
22. PolyLLMem — arXiv, 2025. https://arxiv.org/abs/2503.22962
23. HAPPY — npj, 2024. https://www.nature.com/articles/s41524-024-01293-8
24. Mol-TDL — arXiv, 2024
25. 多尺度 ML — J. Phys. Chem. B, 2025. https://pubs.acs.org/doi/10.1021/acs.jpcb.4c07666

### 物理引导
26. TrinityLLM — Nature Comp. Sci., 2025. https://www.nature.com/articles/s43588-025-00768-y
27. GC-GNN — Macromolecules, 2025. https://pubs.acs.org/doi/10.1021/acs.macromol.5c00720
28. PerioGT — Nature Comp. Sci., 2025. https://www.nature.com/articles/s43588-025-00903-9
29. PINN 综述 — Chen et al., Prog. Polym. Sci., 2024

### 迁移学习 / 小数据
30. Shotgun TL — Yamada et al., ACS Omega, 2023
31. Cross-Property TL — Nature Comm., 2021. https://www.nature.com/articles/s41467-021-26921-5
32. 合成数据预训练 — Kuenneth et al., Patterns, 2024
33. Locluster — Aldeghi et al., Digital Discovery, 2024

### LLM
34. LLM Benchmark — 2024-2025. 零样本 GPT MAE>80K; LLaMA 微调 R2~0.80
35. Afsordeh et al. — Chinese J. Polym. Sci., 2025. https://link.springer.com/article/10.1007/s10118-025-3361-3

### 共聚物
36. WC-SMILES — Antoniuk et al., ACS Appl. Polym. Mater., 2024. https://pubs.acs.org/doi/10.1021/acsapm.3c02715
37. CNN-LSTM + WGAN-GP — 2025. R2=0.95
38. 嵌段共聚物 Graph Kernel — Petersen et al., Angew. Chem., 2025. https://onlinelibrary.wiley.com/doi/10.1002/anie.202411097
39. 共聚物逆设计 — Chem. Sci., 2025. https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc05900j

### BigSMILES
40. BigSMILES — Lin et al., ACS Central Science, 2019
41. BigSMILES vs SMILES — Qiu & Sun, Digital Discovery 2024 / Macromolecules 2025
42. Canonical BigSMILES — Lin, ACS Polymers Au, 2022
43. G-BigSMILES — Schneider et al., Digital Discovery, 2024
44. Automated BigSMILES — Scientific Data, 2024. https://www.nature.com/articles/s41597-024-03212-4

### 生成式设计
45. Vitrimer VAE — Zeng et al., ACS Appl. Mater. Interfaces, 2024 / Adv. Sci., 2025
46. Inverse Design Benchmark — Digital Discovery, 2025

### 特殊体系
47. 可降解聚合物 — Kim et al., Green Chem., 2024
48. 导电聚合物 — Li et al., Adv. Funct. Mater., 2024
49. 聚合物共混物 — Pilania et al., Macromolecules, 2023
50. 高温聚合物 — Chen et al., Polymer, 2024
51. 热固性 — J. Phys. Chem. B, 2024 / J. Polym. Sci. 2025
52. 核酸 Tg — J. Phys. Chem. B, 2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC2895560/
53. DNA-Polymer Conjugates — Chemical Reviews, 2021. https://pubs.acs.org/doi/10.1021/acs.chemrev.0c01074

### 数据集与竞赛
54. Open Polymer Challenge — NeurIPS 2025. arXiv:2503.09469 / arXiv:2512.08896
55. PolyMetriX — npj Comp. Mater., 2025. https://www.nature.com/articles/s41524-025-01823-y
56. OpenPoly — Chinese J. Polym. Sci., 2025
57. PolUQBench — JCIM, 2025. https://pubs.acs.org/doi/10.1021/acs.jcim.5c00550
58. Tao et al. Benchmark — JCIM, 2021. DOI: 10.1021/acs.jcim.1c00800

---

> 本报告综合了深度学习方法调研 (2026-03-13)、前沿方法调研 (2026-03-07) 和前沿方法深度调研 (2026-03-14) 的核心内容，去除三者间大量重叠的 GNN/Transformer 内容，保留各自的增量信息。
