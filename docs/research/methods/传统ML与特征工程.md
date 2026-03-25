# 传统 ML 与特征工程 — 综合调研

> **合并来源**：
> - 低成本计算描述符调研.md (2026-03-08)
> - ML最佳实践调研-Tg预测2024-2026.md (2026-03-15)
> **原始文件保留在** `docs/research/_archive/`
> **时效性**：有效
> **推荐阅读**：必读（特征工程和模型选择的核心参考）

---

## 目录

1. [描述符体系总览](#1-描述符体系总览)
2. [基团贡献法 (GC)](#2-基团贡献法)
3. [化学信息学 2D 描述符](#3-化学信息学-2d-描述符)
4. [聚合物专用描述符](#4-聚合物专用描述符)
5. [量子化学/半经验描述符](#5-量子化学半经验描述符)
6. [拓扑/图论描述符](#6-拓扑图论描述符)
7. [物理参数类描述符](#7-物理参数类描述符)
8. [Afsordeh 4-Feature 与 SHAP 共识排序](#8-afsordeh-4-feature-与-shap-共识排序)
9. [描述符对比与分层推荐](#9-描述符对比与分层推荐)
10. [最佳算法选择 (~7,500 样本)](#10-最佳算法选择)
11. [表格数据基准测试：树模型 vs 深度学习](#11-表格数据基准测试)
12. [特征选择与模型组合策略](#12-特征选择与模型组合策略)
13. [不确定性量化 (UQ)](#13-不确定性量化)
14. [综合推荐与项目对比](#14-综合推荐与项目对比)
15. [RDKit 描述符与 Tg 相关系数详表](#15-rdkit-描述符与-tg-相关系数详表)
16. [参考文献](#16-参考文献)

---

## 1. 描述符体系总览

聚合物 Tg 预测的描述符可按计算成本和信息层级分为以下体系：

| 描述符类别 | 典型数量 | 计算成本 | 单独 R2 | 组合 R2 | 可解释性 | 推荐优先级 |
|-----------|---------|----------|---------|---------|---------|-----------|
| 改进 GC 法 | 58 参数 | < 1 sec | 0.99 | - | 高 | 1 (最高) |
| PolyMetriX | 96 | < 1 sec | - | 0.95-0.97 | 高 | 2 |
| RDKit + Morgan FP | 200+2048 | < 1 sec | - | 0.90-0.94 | 中 | 3 |
| Mordred 2D | 1613 | 1-5 sec | - | 0.90-0.94 | 低 | 4 |
| xTB 电子描述符 | 10-20 | 5-10 min | 0.85 | 0.93-0.96 | 高 | 5 |
| DFT 电子描述符 | 10-20 | 30-60 min | 0.85 | 0.93-0.96 | 高 | 6 |
| 拓扑指数 | 20-50 | < 1 sec | 0.70-0.80 | 0.88-0.92 | 中 | 7 |
| 溶解度参数 | 1-3 | < 1 sec | 0.40-0.60 | - | 高 | 补充 |
| 自由体积参数 | 2-3 | < 1 sec | 0.30-0.50 | - | 中 | 补充 |
| 粗粒化参数 | 3-5 | < 1 sec | 0.50-0.70 | - | 高 | 补充 |

**三级特征体系** (来自 ML 最佳实践调研)：

- **Tier 1 (物理描述符)**: Van Krevelen 溶解度参数、Kuhn 长度/特征比、自由体积分数、内聚能密度。Afsordeh 2025 证明仅 4 个物理描述符即可达 R2=0.97。
- **Tier 2 (拓扑/化学描述符)**: NumRotatableBonds (SHAP #1)、RingCount、HBondDonors/Acceptors、MolWt、TPSA、Morgan FP。
- **Tier 3 (学习型表示)**: polyBERT 600-dim fingerprint、GNN embedding (64-256 dim)、3D conformer descriptors。需要预训练或 MD 模拟。

---

## 2. 基团贡献法

### 2.1 主要方法

#### Van Krevelen 方法
- **原理**：定义摩尔 Tg 函数 Yg，Tg = 1000 * Sum(Yg_i) / M
- **基团数**：约 40 个标准基团
- **精度**：R2 约 0.85-0.90，MAE ~30-50 K
- **优势**：简单直观，手工可算
- **局限**：不考虑基团间相互作用、位置效应

#### Bicerano 方法
- **原理**：基于连接性指数 (connectivity indices) 和结构参数
- **核心参数**：N_BB (主链原子数)、N_SG (侧基大小)、sigma (对称数)、零阶/一阶连接性指数
- **精度**：R2 约 0.90-0.93，MAE ~20-35 K
- **参考书**：Bicerano, "Prediction of Polymer Properties" (3rd Ed., 2002)

#### 改进型 GC 法 (Satyanarayana et al., 2020)
- **创新**：基团数扩展到 58 个，区分邻位/间位/对位取代
- **精度**：R2 = 0.9925 (450+ 种聚合物)
- **论文**：Satyanarayana et al., "Polymer Tg prediction by modified group contribution method" (2020)

### 2.2 计算成本

| 方法 | 所需输入 | 计算时间 | 是否需要软件 |
|------|----------|----------|-------------|
| Van Krevelen | 重复单元结构 | 手工 ~5 min | 否（查表） |
| Bicerano | SMILES/结构 | < 1 sec | Python 脚本 |
| 改进 GC | SMILES/结构 | < 1 sec | Python 脚本 |

### 2.3 GC 中间变量

GC 法不仅预测 Tg，还可计算以下中间变量（均与 Tg 相关）：
- **内聚能密度 (CED)**：Ecoh/V，反映分子间作用力强度
- **摩尔体积 (Vm)**：反映分子堆积效率
- **链刚性参数**：主链转动自由度
- **氢键密度**：基团中氢键供/受体数量

### 2.4 性能对比

| 方法 | 数据集大小 | R2 | MAE (K) | 来源 |
|------|-----------|-----|---------|------|
| Van Krevelen 经典 | ~200 | 0.85-0.90 | 30-50 | Van Krevelen (1990) |
| Bicerano | ~320 | 0.90-0.93 | 20-35 | Bicerano (2002) |
| 改进 GC (2020) | 450+ | 0.9925 | ~10 | Satyanarayana (2020) |

---

## 3. 化学信息学 2D 描述符

### 3.1 RDKit 描述符 (~200 个)
- **类型**：宪法性 (分子量、原子计数)、拓扑 (Wiener/BertzCT)、电子 (TPSA/MR/LogP)、氢键 (HBD/HBA)、片段
- **计算**：`from rdkit.Chem import Descriptors`，整套 < 1 sec
- **优势**：开源、成熟、广泛使用

### 3.2 Mordred 描述符 (1826 个，其中 1613 个 2D)
- **类型**：比 RDKit 更全面——自相关、BCUT、信息内容、药效团等
- **计算**：`from mordred import Calculator, descriptors`，整套 ~1-5 sec
- **推荐使用流程**：
  1. 先用全部 1613 个 2D 描述符 + RFE/Boruta 做特征选择
  2. 然后与推荐子集对比，取交集
  3. 最终保留 20-50 个描述符进入模型

### 3.3 Morgan/ECFP 指纹
- **变体**：ECFP4 (半径 2, 2048 位) 最常用；ECFP6 (半径 3) 捕获更大范围
- **用于 Tg**：Pilania et al. (2019) 使用 Morgan FP + GPR 达到 R2 = 0.92
- **特点**：不可解释但性能出色

### 3.4 性能对比

| 描述符方案 | 描述符数 | 模型 | R2 (Tg) | 数据集大小 |
|-----------|---------|------|---------|-----------|
| RDKit 基础 | ~200 | RF | 0.88-0.92 | ~500 |
| Mordred 2D | ~1600 | XGBoost | 0.90-0.94 | ~500 |
| Morgan FP (2048) | 2048 | GPR | 0.92 | ~500 |
| Polymer Genome | ~300 | GPR | 0.90-0.93 | ~800 |
| PolyMetriX | 96 | ET/GPR | 0.95-0.97 | 7367 |

---

## 4. 聚合物专用描述符

### 4.1 Polymer Genome 指纹
- **三级层次结构**：原子级 → 块级 → 链级
- **模型**：GPR，Tg 预测 R2 ~ 0.90-0.93
- **访问**：https://polymergenome.org

### 4.2 PolyMetriX (2025)
- **创新**：分层聚合物特征化器 (hierarchical polymer featurizer)
- **特征分解**：全聚合物层、骨架层、侧链层 (各 25 化学描述符 + 7 拓扑描述符)
- **总描述符数**：(25+7) x 3 = 96 个
- **关键化学描述符**：分子量、原子计数、杂原子比例、不饱和度、氢键供/受体数、可旋转键数、芳香环数、sp3 碳比例
- **关键拓扑描述符**：链长度、分支度、环化度、路径复杂度
- **训练数据**：7367 个 Tg 数据点
- **性能**：配合 Extra Trees/GPR，R2 达 0.95-0.97
- **优势**：明确区分骨架/侧链贡献，可解释性强
- **论文**：PolyMetriX: A hierarchical polymer featurizer (2025)

---

## 5. 量子化学/半经验描述符

### 5.1 主要描述符与 Tg 相关性

| 描述符 | DFT (B3LYP/6-31G*) | xTB (GFN2) | 单独 r(Tg) |
|--------|--------------------|-----------|----|
| 转动势垒 | 10-30 min | 1-5 min | 0.5-0.7 |
| 偶极矩 | 1-5 min | < 30 sec | 0.3-0.5 |
| 极化率 | 5-15 min | < 1 min | 0.3-0.4 |
| HOMO-LUMO | 1-5 min | < 30 sec | 0.2-0.3 |
| 分子体积 | 1-5 min | < 30 sec | 0.4-0.6 |
| 部分电荷 | 1-5 min | < 30 sec | 0.3-0.5 |
| 全套描述符 | 30-60 min | 5-10 min | 组合 R2 ~0.85 |

### 5.2 半经验替代方案：xTB (GFN2-xTB)

**强烈推荐作为 DFT 的低成本替代**：
- 速度比 DFT 快 100-1000 倍
- 精度：几何结构和相对能量接近 DFT (误差 ~2-5 kcal/mol)
- 可算：偶极矩、极化率、部分电荷、转动势垒、振动频率
- 软件：xtb 程序 (开源，Grimme 组开发)
- 适用规模：可处理 500+ 原子体系

**关键发现**：转动势垒是最有信息量的 QM 描述符——如果只算一个 DFT/xTB 描述符，选转动势垒。

---

## 6. 拓扑/图论描述符

### 6.1 主要描述符

| 描述符 | 定义 | 与 Tg 关系 | 计算成本 |
|--------|------|-----------|----------|
| Wiener 指数 (W) | 所有原子对最短路径之和 | 弱到中等 (r ~0.3-0.5) | 毫秒级 |
| Randic 连接性指数 | Sum(delta_i*delta_j)^(-0.5) | 中等 (r ~0.4-0.6)，Bicerano 核心输入 | 毫秒级 |
| Balaban J | 基于距离矩阵的拓扑指数 | 弱相关 | 毫秒级 |
| Kappa 形状指数 | Hall-Kier 形状指数 (1-3) | 捕获分子形状 | 毫秒级 |
| E-state | 拓扑环境 + 电负性 | 比纯拓扑更好 | 毫秒级 |

### 6.2 文献性能

| 研究 | 描述符组合 | 模型 | R2 | 数据集 |
|------|-----------|------|-----|-------|
| Katritzky et al. (1996) | 拓扑 + 几何 + 电子 | MLR | 0.90 | 88 聚合物 |
| Mattioni & Jurs (2002) | 拓扑 + QM | MLR/PLS | 0.93 | 165 聚合物 |
| Liu & Briber (2020) | 拓扑 + 宪法性 | SVR | 0.937 | 536 聚合物 |

### 6.3 局限性

- 单独使用 R2 < 0.80
- 对立体化学、电子效应不敏感
- 最佳实践：与电子/几何描述符组合使用

---

## 7. 物理参数类描述符

### 7.1 溶解度参数

**Hildebrand 参数**：delta = sqrt(Ecoh/V)，与 Tg 中等正相关 (r ~0.4-0.6)。

**Hansen 三分量**：
- delta_d (色散力)、delta_p (极性力)、delta_h (氢键)
- delta_h 与 Tg 相关性最强 (r ~0.5-0.7)
- 三分量组合比 Hildebrand 单参数更准确

**计算方法**：Van Krevelen GC 法精度 +-10%，成本 < 1 sec (推荐)。

### 7.2 粗粒化参数

| 参数 | 与 Tg 关系 | 理论估算方法 | 精度 |
|------|-----------|-------------|------|
| 特征比 C_inf | 强正相关 | RIS 转移矩阵 / Bicerano GC | +-15% / R2~0.85 |
| Kuhn 长度 b_K | 与 C_inf 等效 | 从 C_inf 导出 | 同 C_inf |
| 持久长度 l_p | 强正相关 | l_p = C_inf*l/2 | 同 C_inf |
| 柔性参数 | 中等负相关 (r ~-0.4 到 -0.6) | RDKit 可旋转键比例 | 精确 |

### 7.3 自由体积参数

- **核心注意**：FFV 是 Tg 的结果而非原因——知道 FFV(Tg) 需要先知道 Tg
- **实用价值**：从结构估算的 V_w 和 K_p 可作为描述符输入 ML 模型，但需结合链刚性/相互作用强度

### 7.4 力场参数

从力场参数估算分子间相互作用强度（不需要运行 MD 模拟）：
- **Lennard-Jones epsilon/sigma**：查 OPLS-AA / GAFF 表，< 1 sec
- **部分电荷**：Gasteiger (RDKit, <1s) / AM1-BCC (~1-10s, 推荐) / RESP (DFT)
- **二面角参数**：等效于转动势垒，强相关

---

## 8. Afsordeh 4-Feature 与 SHAP 共识排序

### 8.1 Afsordeh & Shirali (2025) 研究

**论文**: Chinese Journal of Polymer Science (2025)
**数据集**: 112 种聚合物 | **最佳模型**: Extra Trees, R2=0.97, MAE~7K
**可信度注意**: 12 测试样本 CI 较宽；保守估计严格 CV 下 R2=0.87-0.92

| 特征 | 定义 | 物理含义 | 计算方法 |
|------|------|---------|----------|
| **Flexibility (F)** | 可旋转键数/主链原子数 | 链柔性↑ → Tg↓ | RDKit |
| **SOL** | 侧链最长支链长度 | 空间位阻↑ → Tg↓ | 图论最长路径 |
| **H-bond Density** | (HBD+HBA)/重原子数 | 氢键↑ → Tg↑ | RDKit |
| **Polarity Index** | 极性原子/重原子数 | 极性↑ → Tg↑ | 原子计数 |

**SHAP 重要性**：Flexibility > SOL > HBD > PI

### 8.2 SHAP 多研究共识排序 (综合 6+ 研究)

| 排序 | 特征类别 | SHAP 重要性 | 代表性描述符 |
|------|----------|-----------|-------------|
| 1 | **链柔性/刚性** | ★★★★★ | `NumRotatableBonds`, Flexibility |
| 2 | **拓扑复杂度** | ★★★★ | `BertzCT`, `BalabanJ`, `RingCount` |
| 3 | **极性/氢键** | ★★★ | `TPSA`, HBD, HBA, Polarity Index |
| 4 | **芳香环** | ★★★ | `NumAromaticRings`, `nAromAtom` |
| 5 | **侧链效应** | ★★ | SOL, `FractionCSP3` |
| 6 | **分子量/大小** | ★★ | `MolWt`, `nHeavyAtom` |

**关键发现**: 链柔性/刚性在所有研究中一致排名第一。

### 8.3 单体描述符理论天花板

Tg 方差来源：单体结构 (~70%) + 聚合物级信息 (~30%, 链长/立构/交联/结晶/加工)。
单体 SMILES 只捕获第一类：R2 天花板 ≈ 0.70 + 0.30*0.5 = **0.85**。
这解释了为什么不同方法在严格评估下收敛到 R2 约 0.85-0.90。

---

## 9. 描述符对比与分层推荐

### 9.1 全方法对比表

| 方法 | 特征维度 | 计算速度 | 典型 R2 | 优势 | 局限 |
|------|---------|---------|---------|------|------|
| **Afsordeh 4 特征** | 4 | <1s | 0.97* | 简单、可解释 | 仅 112 样本验证 |
| **RDKit 2D 描述符** | 15-30 | <1s | 0.85-0.90 | 免费成熟 | 单独上限约 0.90 |
| **Mordred 2D** | 20-50 | 2-5s | 0.85-0.92 | 覆盖面广 | 大量冗余需筛选 |
| **Morgan/ECFP 指纹** | 1024-2048 | <1s | 0.88-0.90 | 通用性好 | 黑箱 |
| **PG 指纹** | ~3000 | ~1s | 0.90-0.93 | 聚合物专用 | 需外部访问 |
| **GFN2-xTB** | 5-10 | 1-5min | +0.02~0.05 | 电子性质补充 | 计算较慢 |
| **DFT (B3LYP)** | 10-20 | 1-24h | +0.03~0.08 | 最精确 | 成本极高 |

*注: Afsordeh R2=0.97 基于 90/10 split + 12 测试样本，严格 CV 下保守估计 0.87-0.92

### 9.2 推荐的渐进式特征工程路线图

| 层级 | 特征来源 | 特征数 | 计算成本 | 预期 R2 | 阶段 |
|------|---------|--------|---------|---------|------|
| **L0** | Afsordeh 4 特征 | 4 | <0.1s | 0.85-0.90 | Phase 1 |
| **L1** | + RDKit 核心描述符 | +15 | <0.5s | 0.90-0.93 | Phase 1 |
| **L2** | + Mordred 精选子集 | +20-30 | <5s | 0.92-0.95 | Phase 2 |
| **L3** | + GC 物性 (CED, 溶解度参数) | +5-8 | <1s | 0.93-0.96 | Phase 2 |
| **L4** | + GFN2-xTB 电子描述符 | +5-10 | 1-5min | 0.94-0.97 | Phase 3 |
| **L5** | + Morgan 指纹 (并行通道) | +1024 | <1s | 集成提升 | Phase 3 |
| **L6** | + polyBERT 嵌入 (可选) | +768 | ~1s | 集成提升 | Phase 4 |

### 9.3 关键发现

1. **基团贡献法的回归**：改进 GC 法 R2=0.99 可与复杂 ML 媲美，且完全可解释
2. **分层描述最重要**：PolyMetriX 的成功表明，区分骨架/侧链的描述符比单一水平更有效
3. **数据量比描述符精度更重要**：7367 条 + 简单描述符 > 500 条 + 复杂描述符
4. **特征工程质量 > 数据规模 > 模型复杂度**：贯穿全领域的关键规律

---

## 10. 最佳算法选择

### 10.1 树模型家族

| 模型 | 优势 | 劣势 | 7.5K 样本表现 | 推荐度 |
|------|------|------|--------------|--------|
| **CatBoost** | Ordered boosting 防过拟合；原生类别特征支持 | 训练稍慢 | 2024-25 基准平均比 XGBoost 高 ~6% | 5/5 |
| **LightGBM** | 极快训练；低内存；GPU 加速 | 小数据易过拟合 | 略逊 CatBoost 但差距 <2% | 4/5 |
| **XGBoost** | 最成熟生态；GPU 支持好 | 2024+ 非最优 | 稳定但不突出 | 4/5 |
| **ExtraTrees** | 极快；天然正则化 | 大数据不如 boosting | Phase 1-3 R2=0.85 | 3/5 |
| **GBR (sklearn)** | 简单可靠 | 无 GPU；大数据慢 | 被 CatBoost/LightGBM 超越 | 2/5 |

**关键发现**：CatBoost 在 2024-2025 年多项基准中一致领先。Ordered boosting 在 1K-50K 数据集上有效缓解 target leakage。

### 10.2 深度学习模型 (表格数据)

| 模型 | 核心创新 | 7.5K 样本适用性 | 推荐度 |
|------|---------|----------------|--------|
| **TabPFN v2** (2024) | 零调参；合成数据预训练 Transformer | <10K 首次系统性超越 GBDT | 5/5 |
| **TabM** (ICLR 2025) | BatchEnsemble 极简 MLP 集成 | 与 GBDT 持平 | 4/5 |
| **FT-Transformer** | 每个特征一个 token + self-attention | 需 >5K 才有优势 | 3/5 |
| **TabNet** | 稀疏特征选择 | 调参敏感 | 3/5 |

**TabPFN v2 关键**: 限制 <10K 样本 / <500 特征。7,500 样本 + ~50 维完美落在最优区间。零调参 = Nested CV 内层可跳过超参搜索。

### 10.3 AutoML 框架

| 框架 | 2024 Kaggle 表现 | 核心策略 | 推荐度 |
|------|-----------------|---------|--------|
| **AutoGluon** | 15/18 奖牌, 7 金 | 多层 stacking + k-fold bagging | 5/5 |
| **FLAML** | 小众但高效 | 低成本超参搜索 (CFO) | 3/5 |

### 10.4 按数据规模推荐

| 数据规模 | 最佳选择 | 次优选择 |
|----------|---------|---------|
| <500 | ExtraTrees / RF | TabPFN v2 |
| 500-2,000 | CatBoost + Nested CV | TabPFN v2 |
| **2,000-10,000** | **CatBoost / TabPFN v2** | **LightGBM / 多模型 Stacking** |
| 10,000-50,000 | LightGBM / XGBoost | CatBoost / DL 集成 |
| >50,000 | DL 开始有优势 | AutoML (AutoGluon) |

---

## 11. 表格数据基准测试

### 11.1 关键基准论文

| 基准 | 年份 | 结论 | 引用 |
|------|------|------|------|
| Grinsztajn et al. | 2022 | 树模型在 <10K 上系统性优于 DL | NeurIPS 2022 |
| McElfresh et al. | 2024 | GBDT 仍是默认最优但差距缩小 | ICLR 2024 |
| TabPFN v2 | 2024 | 首个在 <10K 上一致超越 GBDT 的 DL | arXiv 2024 |
| TabM | 2025 | 简单 MLP 集成即可匹配 GBDT | ICLR 2025 |
| Kaggle 2025 实证 | 2025 | <5000 条数据树模型 > DL | 竞赛总结 |

### 11.2 评估方式对 R2 的影响

| 评估方式 | R2 膨胀 | 可靠性 |
|---------|--------|--------|
| 90/10 split (12样本) | +0.05~0.15 | 低 |
| 80/20 split | +0.03~0.08 | 中 |
| 5-fold CV | 基准 | 高 |
| Nested CV (5x3, inner 3) | -0.02~0.05 | 最高 |

**结论**: Nested CV R2=0.83 等价于简单 split 下 R2=0.88-0.92。

---

## 12. 特征选择与模型组合策略

### 12.1 特征选择 Pipeline (4 阶段)

1. **VarianceThreshold**：去除常量/准常量特征
2. **Boruta-SHAP**：结合 Boruta 统计严格性 + SHAP 全局重要性 (比原始 Boruta 快 3-5x)
3. **mRMR**：最大相关 + 最小冗余 (mutual information 框架)，避免冗余特征
4. **SHAP 精排**：最终排序

**替代工具**：shap-select (2024) — 超轻量特征选择，用 SHAP 值回归目标变量。

### 12.2 Stacking 架构

7,500 样本可支撑 2 层 stacking (304 样本时 3 次失败，但 7.5K 足够)：

- **L0 基学习器**：CatBoost + LightGBM + TabPFN v2 + ExtraTrees
- **L1 Meta-learner**：Ridge (简单线性组合 + L2 正则化)

**关键设计决策**：
1. Meta-learner 选 Ridge：避免过拟合元特征
2. 传递 top-5 SHAP 原始特征到 L1：平均提升 0.5-1% (AutoGluon 2024)
3. L0 所有模型必须使用相同 fold 划分：确保 OOF 预测对齐
4. 样本量验证：7,500 / 5 folds = 1,500 OOF per fold，足以训练 L1

### 12.3 Stacking vs Blending

| 策略 | 训练复杂度 | 数据利用率 | 过拟合风险 | 推荐 |
|------|-----------|-----------|-----------|------|
| **Stacking (CV-based)** | 高 | 100% | 低 | 5/5 |
| **Blending (holdout)** | 低 | ~80% | 中 | 3/5 |
| **加权平均** | 最低 | 100% | 最低 | 4/5 |

### 12.4 超参搜索策略

| 方法 | 效率 | 推荐场景 |
|------|------|---------|
| **Optuna (TPE)** | 高 | 通用最佳选择，支持剪枝 |
| **FLAML (CFO)** | 最高 | 预算受限时 |
| Grid Search | 低 | 仅 <5 超参时 |
| Random Search | 中 | 探索阶段 |

推荐 **Optuna + Nested CV 内层**：
- 外层 CV: RepeatedKFold(5,3) 评估泛化性能
- 内层 CV: KFold(3) + Optuna(50 trials) 搜索超参
- TabPFN v2 跳过内层（零调参）

---

## 13. 不确定性量化

### 13.1 方法对比

| 方法 | 类型 | 优势 | 劣势 | 适用性 |
|------|------|------|------|--------|
| **Conformal Prediction** | 分布无关 | 理论覆盖率保证；模型无关 | 区间可能过宽 | 5/5 |
| **MAPIE CQR** | CP + 分位数 | 自适应区间宽度；sklearn 兼容 | 需分位数模型 | 5/5 |
| **NGBoost** | 概率梯度提升 | 直接输出分布参数 | 训练慢 | 3/5 |
| **MC Dropout** | 贝叶斯近似 | 简单实现 | 仅适用 NN | 2/5 |
| **Deep Ensemble** | 集成 | 强经验表现 | 5x 训练成本 | 3/5 |

### 13.2 推荐方案：MAPIE CQR

**Conformalized Quantile Regression (CQR)** 是当前最优选择：
1. **分布无关**：不假设误差分布形状
2. **自适应区间**：高不确定性区域更宽，低不确定性更窄
3. **理论保证**：实际覆盖率 >= (1-alpha) - 1/(n+1)
4. **实现简单**：MAPIE 库与 scikit-learn 完全兼容

### 13.3 集成最佳实践

1. 校准集划分：从训练集中划出 15-20% (~1,500 条)
2. 与 Nested CV 结合：在外层 CV 每个 fold 中再划出校准集
3. 多模型 UQ 对比：对 stacking 每个基模型分别做 CQR
4. 核酸域外预测：CQR 区间会自然变宽 (正是期望行为)

### 13.4 PolUQBench (JCIM 2025) 基准

- 9 种 UQ 方法对比：分布内 Ensemble 最优；OOD 用 BNN-MCMC；高 Tg 用 NGBoost
- MAPIE: sklearn 兼容，无需 PyTorch，20+ 方法，~10 行代码即可集成

---

## 14. 综合推荐与项目对比

### 14.1 当前计划 vs 调研推荐

| 维度 | 当前计划 | 调研推荐 | 变更理由 |
|------|---------|---------|---------|
| **主模型** | ExtraTrees | **CatBoost** | 2024-25 基准一致领先 |
| **DL 基模型** | 无 | **TabPFN v2** | 零调参、<10K 超越 GBDT |
| **集成策略** | Stacking ET+GBR+SVR→Ridge | **CatBoost+LGB+TabPFN+ET→Ridge** | 多样化基学习器 |
| **UQ** | 无 | **MAPIE CQR** | 核酸域外预测需可靠区间 |
| **特征选择** | Boruta→mRMR→SHAP | Boruta-SHAP→mRMR→SHAP | 效率 3-5x |
| **超参搜索** | 自动搜索 10-20 组 | **Optuna 50 trials** | 更系统化 |
| **预处理** | PowerTransformer+MinMax | 不变 | 已是最佳实践 |
| **评估** | Nested CV 5x3, 3 | 不变 | 已是最佳实践 |

### 14.2 优先级排序

| 优先级 | 改进项 | 预期收益 | ROI |
|--------|--------|---------|-----|
| **P0** | CatBoost 替换 ExtraTrees | R2 +2-5% | 5/5 |
| **P0** | Van Krevelen 物理描述符 | R2 +5-10% | 5/5 |
| **P1** | TabPFN v2 加入 stacking | R2 +1-3% | 4/5 |
| **P1** | MAPIE CQR 不确定性量化 | 论文亮点 | 4/5 |
| **P2** | Boruta-SHAP 替换 Boruta | 效率 3-5x | 3/5 |
| **P2** | Optuna 系统化超参搜索 | R2 +0.5-1% | 3/5 |
| **P3** | AutoGluon 基线对比 | 基线验证 | 3/5 |

---

## 15. RDKit 描述符与 Tg 相关系数详表

基于 500+ 聚合物数据集的文献汇总 (Pearson 线性相关系数 r)：

### 第一梯队 (|r| > 0.6，强相关)

| 描述符 | Pearson r | 方向 | 物理解释 |
|--------|----------|------|----------|
| NumRotatableBonds | -0.87 to -0.90 | 负 | 可旋转键越多越柔，Tg 越低 |
| fr_bicyclic | +0.65 to +0.75 | 正 | 双环结构增加刚性 |
| BertzCT | +0.60 to +0.70 | 正 | 拓扑复杂度高=链刚 |
| RingCount | +0.60 to +0.68 | 正 | 环结构限制构象自由度 |

### 第二梯队 (|r| = 0.4-0.6，中等相关)

| 描述符 | Pearson r | 方向 | 物理解释 |
|--------|----------|------|----------|
| TPSA | +0.45 to +0.55 | 正 | 极性表面积大=强极性相互作用 |
| MaxEStateIndex | +0.40 to +0.55 | 正 | 电负性原子环境 |
| BalabanJ | +0.40 to +0.50 | 正 | 拓扑形状指数 |
| HBD + HBA | +0.35 to +0.50 | 正 | 氢键能力 |
| NumAromaticRings | +0.40 to +0.50 | 正 | 芳香环刚性 |
| Kappa3 | -0.40 to -0.50 | 负 | 形状灵活性 |

### 第三梯队 (|r| = 0.2-0.4，弱相关但有用)

| 描述符 | Pearson r | 方向 | 物理解释 |
|--------|----------|------|----------|
| MolLogP | -0.25 to -0.35 | 负 | 疏水性与柔性链相关 |
| MolMR | +0.20 to +0.35 | 正 | 摩尔折射率反映极化 |
| MolWt | +0.15 to +0.30 | 混合 | 分子量效应复杂 |
| FractionCSP3 | -0.20 to -0.35 | 负 | sp3 碳比例高=更柔 |
| Chi0n, Chi1n | +0.20 to +0.30 | 正 | 连接性指数 |

**重要提醒**: 单变量 Pearson 相关系数只反映线性相关。许多描述符与 Tg 的关系是非线性的，在 ML 模型中组合使用时贡献远大于单独相关系数所示。

---

## 16. 参考文献

### 经典文献

1. Van Krevelen & Te Nijenhuis (2009). Properties of Polymers (4th Ed.). DOI: 10.1016/B978-0-08-054819-7.X0001-5
2. Bicerano (2002). Prediction of Polymer Properties (3rd Ed.). ISBN: 978-0824708214
3. Mattioni & Jurs (2002). J. Chem. Inf. Comput. Sci. DOI: 10.1021/ci010062o
4. Satyanarayana et al. (2020). Modified Group Contribution for Tg Prediction.

### 近期重要论文 (2019-2026)

5. Pilania et al. (2019). JCIM. DOI: 10.1021/acs.jcim.8b00500
6. Tao et al. (2021). JCIM. DOI: 10.1021/acs.jcim.1c00800
7. Liu & Briber (2020). ACS Macro Letters.
8. Miccio et al. (2020). Polymer. DOI: 10.1016/j.polymer.2020.122341
9. Chen et al. (2021). Materials Science and Engineering: R. DOI: 10.1016/j.mser.2020.100595
10. Kim et al. (2018). J. Phys. Chem. C. DOI: 10.1021/acs.jpcc.8b02913
11. PolyMetriX Team (2025). Hierarchical polymer featurizer. npj Comp. Mater. https://www.nature.com/articles/s41524-025-01823-y
12. Jha et al. (2019). Modelling and Simulation in Materials Science and Engineering. DOI: 10.1088/1361-651X/aaf8bc
13. Afsordeh & Shirali (2025). Chinese J. Polym. Sci. https://link.springer.com/article/10.1007/s10118-025-3361-3

### 算法与基准

14. Prokhorenkova, L. et al. CatBoost: unbiased boosting with categorical features. NeurIPS 2018.
15. Shwartz-Ziv, R. and Armon, A. Tabular data: Deep learning is not all you need. Information Fusion, 2022.
16. Hollmann, N. et al. TabPFN v2. arXiv:2410.18021, 2024.
17. Grinsztajn, L. et al. Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022.
18. McElfresh, D. et al. When Do Neural Nets Outperform Boosted Trees on Tabular Data? ICLR 2024.
19. Gorishniy, Y. et al. TabM. ICLR 2025.
20. AutoGluon Team. AutoGluon-Tabular. ICML 2024 Workshop.
21. Kaggle 2025 表格竞赛总结. <5000 条数据树模型 > DL.
22. Open Polymer Challenge (NeurIPS 2025). arXiv:2503.09469 / arXiv:2512.08896

### UQ 与特征选择

23. Keany, E. BorutaShap. 2024. https://github.com/Ekeany/Boruta-Shap
24. shap-select. 2024. https://github.com/shap-select/shap-select
25. Ding, C. and Peng, H. mRMR. J Bioinformatics, 2005.
26. Taquet, V. et al. MAPIE. arXiv:2207.12274, 2022.
27. Romano, Y. et al. Conformalized Quantile Regression. NeurIPS 2019.
28. Berthon, A. et al. Conformal Prediction for Materials Science. 2024.
29. PolUQBench. JCIM 2025. https://pubs.acs.org/doi/10.1021/acs.jcim.5c00550

### 工具与框架

30. Optuna. https://optuna.org/
31. FLAML. https://microsoft.github.io/FLAML/
32. mrmr-selection. https://github.com/smazzanti/mrmr
33. AutoGluon. https://auto.gluon.ai/
34. TabPFN. https://github.com/automl/TabPFN
35. MAPIE. https://github.com/scikit-learn-contrib/MAPIE

### 数据库

36. PoLyInfo (NIMS). ~30,000 条目. 免费注册.
37. Polymer Genome. https://polymergenome.org
38. CROW Polymer Database. ~2,000 聚合物. 免费在线.

---

> 本报告综合了低成本计算描述符调研 (2026-03-08) 和 ML 最佳实践调研 (2026-03-15) 的核心内容，去除重复部分后整合为特征工程和模型选择的统一参考文档。
