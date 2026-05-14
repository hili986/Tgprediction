# Tg 预测项目 — 完整技术报告

> 为升级答辩准备 | 2026-04-08

## 项目全景：一句话说清楚

**给定一个聚合物的分子结构（SMILES），不做实验，预测它的玻璃化转变温度（Tg）。**

最终成果：R²=0.9165（多尺度186维特征 + TabPFN），核酸迁移 ATP/ADP 误差 <5K。

---

## 技术路线总览

```
Phase 1-3 (小数据探索)          Phase 4 (消融验证)         Phase 5 (大数据SOTA)
Bicerano 304条                 E1-E15                     E16-E26
特征从4d→46d                   验证每组特征贡献             7,486条 + 7种模型
GBR基线 R²~0.82               VPD +4.6% → R²=0.866       TabPFN R²=0.896

         ↓                                ↓                        ↓
         
Phase A-E (多尺度重构)
物理驱动特征金字塔
46d → 186d (PHY-C 58d + GNN 64d + polyBERT 64d)
TabPFN R²=0.9165 (最终SOTA)
```

---

## 第一部分：数据集建设

### 1.1 Bicerano 标准数据集（304条）

| 项目 | 内容 |
|------|------|
| 来源 | Bicerano 教科书 (2002) + Choi et al. (2024) Figshare |
| 规模 | 304 种线型均聚物 |
| 内容 | 聚合物名称 + 重复单元 SMILES + BigSMILES + Tg (K) |
| Tg 范围 | 130K（PDMS，柔软如橡胶）～ 576K（PEEK，坚硬如塑料） |
| 用途 | Phase 1-4 的主数据集，消融实验基础 |

### 1.2 统一大规模数据集（7,486条）

整合 6 个公开数据库：

| 数据源 | 原始条数 | 质量过滤后 | 说明 |
|--------|---------|----------|------|
| Bicerano | 304 | 304 | 经典标准集 |
| PolyMetriX | ~5,500 | ~5,500 | 策展数据，含可靠性分级 |
| NeurIPS OPP 2025 | ~17,000 | ~1,200 | 竞赛数据，严格过滤 |
| Pilania PHA | ~400 | ~400 | 聚羟基链 |
| 其他 | ~100 | ~80 | 文献补充 |
| **合计** | — | **7,486** | 去重 + SMILES 规范化 |

**质量过滤规则**：
- 去除非聚合物（SMILES 中无 `*` 标记）
- 去除 ML 预测值（小数位 >2 判断为计算值而非实验值）
- SMILES 规范化（RDKit canonical）
- 重复数据取中位数

### 1.3 桥梁数据集（205条）

为核酸迁移预测专门构建：
- 精选含丰富氢键的聚合物（聚酰胺、聚氨酯、聚酰亚胺等）
- 在化学空间上"连接"通用聚合物和核酸分子
- 训练时给予额外权重（bridge_weight=0.8~1.0）

---

## 第二部分：特征工程（核心技术）

### 2.1 特征演进路线

```
L0 (4d)     →  L1 (19d)    →  L1H (34d)   →  M2M-V (46d)  →  PHY (48d)   →  186d
Afsordeh       +RDKit 2D      +HBond         +VPD            +GC_Tg         +GNN+polyBERT
基础物理        分子描述符      氢键            虚拟聚合         基团贡献         多尺度融合
R²~0.65        R²~0.75        R²~0.82        R²~0.87         R²~0.90        R²~0.92
```

### 2.2 L0：Afsordeh 基础物理特征（4维）

来自 Afsordeh et al. 论文（声称 R²=0.97，但基于 90/10 split 只有 12 个测试样本）。

| 特征 | 公式 | 物理含义 |
|------|------|---------|
| **FlexibilityIndex** | 旋转键数 / 重原子数 | 主链柔性 — **SHAP #1，遥遥领先** |
| HBondDensity | (H键供体 + H键受体) / 重原子数 | 分子间氢键能力 |
| PolarityIndex | (O + N + S 原子数) / 重原子数 | 极性 |
| SOL | √(Ecoh/Vw)，Van Krevelen 基团贡献法 | Hildebrand 溶解度参数 |

**关键洞察**：FlexibilityIndex 一个特征就贡献了 SHAP=41.51，是第二名的 2.7 倍。这符合 Gibbs-DiMarzio 理论：**Tg 本质上由链的柔性决定**。

### 2.3 L1：RDKit 2D 描述符（15维）

| 分类 | 特征 | 物理含义 |
|------|------|---------|
| 拓扑 | RingCount, NumAromaticRings | 环数 → 刚性来源 |
| 极性 | TPSA, MolLogP | 极性表面积、亲疏水性 |
| 氢键 | NumHDonors, NumHAcceptors | H键供受体计数 |
| 柔性 | NumRotatableBonds, FractionCSP3 | 旋转键数、sp3碳比例 |
| 分子量 | MolWt, HeavyAtomCount | 重复单元大小 |
| 形状 | BalabanJ, Chi0v, Chi1v, Kappa1, Kappa2 | 分子拓扑形状 |

### 2.4 L1H：氢键特征（15维完整版 / 5维精简版）

**完整版（15维）**：
- 10 个 SMARTS 模式匹配计数：酰胺、脲、氨基甲酸酯、酰亚胺、磷酸酯、芳香NH、苯并咪唑、嘧啶、嘌呤、羟基
- 4 个密度指标：总H键密度、强H键密度、核酸相关性、芳香H键密度
- 1 个 CED 加权和：每个基团乘以其内聚能密度（CED）值再求和

**精简版（5维，推荐）**：
| 特征 | 物理含义 |
|------|---------|
| ced_weighted_sum | 内聚能密度加权和（SHAP #7） |
| total_hbond_density | H键位点密度 |
| hbond_network_potential | H键网络形成能力 = (donors × acceptors) / heavy² |
| polar_fraction | 极性表面比例 |
| interaction_types | 不同H键基团类型数 |

### 2.5 VPD：虚拟聚合描述符（12维）— 核心创新

**解决的问题**：传统方法只从单体计算描述符，但聚合物的性质 ≠ 单体的性质。单体连接成链后会产生"聚合效应"——性质发生变化。传统单体描述符的理论天花板是 R²≈0.84。

**核心思路**：
```
单体 SMILES (*CC*)
    ↓ 虚拟聚合（n=3）
3-RU 寡聚体 (*-CC-CC-CC-*)
    ↓ 分别计算描述符
比较聚合前后的差异 → 这个差异就是 VPD
```

**12个特征分3组**：

#### Core（6个）— 三聚体单位平均特征：
| 特征 | 计算方式 | 物理含义 |
|------|---------|---------|
| MolWt_per_RU | 三聚体分子量 / 3 | 单位重复单元质量 |
| TPSA_per_RU | 三聚体极性面积 / 3 | 单位极性 |
| MolLogP_per_RU | 三聚体LogP / 3 | 单位疏水性 |
| HeavyAtom_per_RU | 三聚体重原子 / 3 | 单位大小 |
| RotBonds_per_RU | 三聚体旋转键 / 3 | 单位柔性 |
| RingCount_per_RU | 三聚体环数 / 3 | 单位刚性（**SHAP #2**） |

#### Delta（4个）— 聚合引起的性质变化：
| 特征 | 公式 | 物理含义 |
|------|------|---------|
| MolWt_delta | (二聚体MW/2) - 单体MW | 聚合脱水带来的质量变化 |
| TPSA_delta | (二聚体TPSA/2) - 单体TPSA | 聚合引起的极性变化 |
| LogP_delta | (二聚体LogP/2) - 单体LogP | 聚合引起的亲疏水变化 |
| RotBonds_delta | (二聚体RotBonds/2) - 单体RotBonds | 聚合引起的柔性变化（**SHAP #10**） |

#### Junction（2个）— 连接点特征：
| 特征 | 物理含义 |
|------|---------|
| junction_hbond_count | 连接点附近的H键能力 |
| junction_flex_ratio | 连接点处灵活键比例（**SHAP #4**） |

**VPD 构建算法**（`build_oligomer` 函数）：
1. 找到 SMILES 中的 `*` 标记（两个附着点）
2. 复制 n 份单体分子
3. 用 `CombineMols` 合并
4. 移除虚原子（`*`），在附着点之间添加化学键
5. 得到 n-聚体的完整分子

**效果**：VPD 在 SHAP Top 15 中占 6 席（40%），带来 +4.6% R² 提升。

### 2.6 PPF：物理代理特征（10维）

从物理理论出发设计的特征：

| 特征 | 计算方式 | 物理含义 | 理论来源 |
|------|---------|---------|---------|
| M_per_f | 分子量 / 柔性键数 | 每个柔性键承载的质量，越高越刚 | Schneider-DiMarzio |
| CED_estimate | δ²（溶解度参数平方） | 内聚能密度 | Hildebrand |
| Vf_estimate | 1 - 1.3×(Vvdw/Vm) | 自由体积分数 | Simha-Boyer |
| backbone_rigidity | 刚性键 / 总骨架键 | 主链刚性比例 | — |
| steric_volume | 侧链原子 × 15Å³ | 侧链体积代理 | — |
| flexible_bond_density | 柔性键 / 重原子 | 柔性键密度 | — |
| symmetry_index | 对称取代 / 总取代 | 结构对称性 [0,1] | — |
| side_chain_ratio | 侧链原子 / 骨架原子 | 侧链比例 | — |
| CED_hbond_frac | H键Ecoh / 总Ecoh | H键对CED的贡献 | — |
| ring_strain_proxy | 3-4元环 / 总环数 | 环张力代理 | — |

**柔性键计数规则**（Schneider-DiMarzio）：
- Si-O：1.5（超灵活，这就是 PDMS Tg 极低的原因）
- 非环单键 C-C/C-O/C-N：1.0
- 芳香-芳香连接：0.5
- 末端 -F/-OH：0（旋转不改变分子形状）
- 酰胺 C-N：0（共振限制旋转）

### 2.7 GC_Tg：基团贡献 Tg 预测（2维）

基于 Cao (2020) 的改进型基团贡献法：

**核心公式**：
```
Tg(∞) = 1000 × Σ(Ni × Ygi) / Mw    （单位：K）
```
- Ni = 第 i 个基团在重复单元中出现的次数
- Ygi = 该基团的 Tg 贡献值（K·kg/mol）
- Mw = 重复单元分子量

**基团库**：58 个基团，按大小降序匹配（避免重复计数）。举几个关键基团：

| 基团 | Yg 值 | 代表聚合物 |
|------|-------|----------|
| CH(phenyl) | 38.934 | 聚苯乙烯（PS，Tg=373K） |
| C(CH3)(COOCH3) | 37.503 | PMMA（Tg=378K） |
| 酰胺 -CONH- | 19.247 | 尼龙（Tg 高） |
| -CH2- | 4.026 | 聚乙烯（Tg=195K，低） |
| 醚 -O- | **-14.718** | 聚环氧乙烷（**降低 Tg！**） |

**输出**：GC_Tg（预测值）+ GC_coverage（覆盖率，<30% 返回 NaN）

**与 Tg 的相关性**：r=0.554（中等相关）。作为特征输入模型而非直接用作预测。

### 2.8 多尺度重构新增特征（Phase B2/C）

#### 链间相互作用（8维，Phase B2）

| 类别 | 特征 | 物理含义 |
|------|------|---------|
| 静电 | MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge | Gasteiger 偏电荷 |
| 偶极 | dipole_moment, MolMR, polar_bond_fraction | 偶极矩、极化率、极性键比例 |
| 疏水 | hydrophobic_ratio, hydrophilic_ratio | 疏水/亲水表面积比 |

#### 链段物理（~8维，Phase C）

| 特征 | 物理含义 | 计算方式 |
|------|---------|---------|
| Neff_300K | 300K 有效链段数 | 3-mer 构象采样 + Boltzmann 加权 |
| Neff_500K | 500K 有效链段数 | 同上，不同温度 |
| Neff_ratio | 温度敏感性 | Neff_500K / Neff_300K |
| conf_strain | 构象应变能 | MMFF94 力场优化 |
| Cn_proxy | 链刚度代理 | 图直径（最远重原子对距离） |
| curl_ratio | 卷曲比 | Rg/Ree 比（回转半径/端端距） |
| curl_variance | 卷曲方差 | 构象分布宽度 |

**这就是"CPU 算了好久"的部分**：每个分子需要采样多个 3D 构象，用 MMFF94 力场优化，再做 Boltzmann 统计。304 个分子 × 多个构象 × 力场计算 = 大量 CPU 时间。

### 2.9 GNN 嵌入（64维，Phase D）

用图注意力网络（PhysicsGAT）对分子图编码：

```
分子 SMILES → 分子图（原子=节点, 化学键=边）
    ↓
原子特征（25维）：元素、电荷、度数、芳香性、物理特征
边特征（6维）：键类型、共轭、环成员
    ↓
3层 GAT：25d → 128d(4头) → 128d(4头) → 64d(1头)
    ↓
GRIN 池化：只对中间重复单元的原子做 mean pooling
    ↓
64维 图嵌入向量
```

**GRIN 池化的关键设计**：只聚合"中间重复单元"的原子特征，忽略端基原子。这确保嵌入对重复单元数目不变——因为聚合物是周期性结构。

**GNN 预训练**：只用外部数据（NeurIPS + PolyMetriX ~10-15K），不含 unified_tg 数据集（防泄漏）。微调在 Nested CV 每个 outer fold 内做（零泄漏）。

### 2.10 polyBERT 嵌入（64维，Phase D）

```
SMILES → polyBERT (DeBERTa-v2 预训练语言模型)
    ↓
[CLS] token → 600维 嵌入
    ↓
PCA 降维 → 64维（每个 CV fold 内独立拟合，防泄漏）
```

polyBERT 是在大量聚合物 SMILES 上预训练的语言模型，学到了化学结构的隐含模式。

### 2.11 最终特征集总结

| 阶段 | 特征集 | 维度 | R²（TabPFN） | 用途 |
|------|--------|------|-------------|------|
| Phase 1-3 | L1H | 34d | ~0.82 | 基线 |
| Phase 4 | M2M-V | 46d | 0.866 (GBR) | 消融验证 |
| Phase 5 | M2M-V | 46d | 0.896 (TabPFN) | 大数据SOTA |
| Phase B | PHY | 48d | 0.872 (CatBoost) | 物理特征 |
| Phase C | PHY-C-light | 58d | 0.905 (TabPFN) | 链段物理 |
| **Phase E** | **186d** | **58+64+64** | **0.917 (TabPFN)** | **最终SOTA** |

---

## 第三部分：模型训练与评估

### 3.1 评估框架：嵌套交叉验证（Nested CV）

**为什么用 Nested CV**：普通 CV 在调参时会高估性能（数据泄漏）。Nested CV 用两层 CV 隔离调参和评估。

```
外层 CV：RepeatedKFold(5折 × 3次重复 = 15折)
    │
    │  每个外层折：
    │  ├── 训练集 → 内层 CV → 调参
    │  │   内层 CV：KFold(3折)
    │  │   每折跑 50 次 Optuna 试验
    │  │
    │  └── 用最优参数在训练集上训练 → 在测试集上评估
    │
    └── 汇总 15 折的 R²/MAE → 均值±标准差
```

**配置**：
- 外层：`RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)`
- 内层：`KFold(n_splits=3, shuffle=True)`
- Optuna：TPE 采样器 + 中位数剪枝，50 trials/fold
- 总计算量：15 × 50 = 750 次 Optuna 试验

**零调参模型（TabPFN）**：跳过内层，直接用外层 15 折评估。

### 3.2 预处理流程

```python
PowerTransformer(method='yeo-johnson')  →  MinMaxScaler(feature_range=(0, 1))
```

- Yeo-Johnson 变换：将偏态特征正态化（支持负值）
- MinMaxScaler：缩放到 [0,1]，适配 TabPFN 等模型

### 3.3 模型库

| 模型 | 类型 | 默认配置 | 特点 |
|------|------|---------|------|
| **TabPFN v2** | 元学习 | 零调参 | 在百万表格数据集上预训练，对新数据直接推理 |
| CatBoost | GBDT | iterations=1000, lr=0.05, depth=6 | 有序提升，天然处理类别特征 |
| LightGBM | GBDT | n_estimators=1000, lr=0.05, leaves=31 | 直方图加速，高速 |
| XGBoost | GBDT | n_estimators=1000, lr=0.05, depth=6 | 经典 GBDT |
| ExtraTrees | 随机森林变体 | n_estimators=500 | 极度随机化分割 |
| GBR | sklearn GBDT | n_estimators=300, lr=0.05, depth=4 | Phase 4 基线 |
| Stacking v2 | 堆叠集成 | CatBoost+LightGBM+ET+XGBoost→Ridge | 4 模型集成 |

### 3.4 不确定性量化（MAPIE）

**方法**：CrossConformal CatBoost

**原理**：
1. 用交叉验证在训练集上计算每个样本的残差
2. 用残差的分位数构建置信区间
3. 保形推理保证理论覆盖率

**结果**：
| 指标 | 数值 |
|------|------|
| 目标覆盖率 | 90% |
| 实际覆盖率 | 90% |
| 平均区间宽度 | 129K (±64.5K) |
| 相对宽度 | 30.9% |

**应用**：每个预测附带置信区间。高置信（窄区间）→ 直接用；低置信（宽区间）→ 做实验验证。

### 3.5 层级残差学习（探索性，最终未采用）

4层物理分解：
```
L0：线性基线      LinearRegression(M/f)     → Gibbs-DiMarzio 主干
L1：体积修正      GBR(体积、对称性)          → 学习 L0 的残差
L2：极性修正      GBR(CED、氢键)            → 学习 L0+L1 的残差
L3：残差捕获      GBR(全部特征)             → 学习所有残差
```

**结果**：无额外增益（TabPFN 直接在全特征上表现更好）。作为负面结论写入论文。

---

## 第四部分：实验结果

### 4.1 Phase 4：消融实验（E1-E15，Bicerano 304条）

验证每组特征的贡献：

| 排名 | 实验 | 特征 | 模型 | R² | MAE(K) | 发现 |
|------|------|------|------|-----|--------|------|
| 1 | **E3** | **M2M-V 46d** | GBR | **0.866** | 29.2 | **VPD +4.6%** |
| 2 | E4 | M2M 56d (全部) | GBR | 0.860 | 29.0 | PPF 无额外增益 |
| 3 | E5 | M2M-PV 22d (纯物理) | GBR | 0.860 | 28.5 | 纯物理等效 |
| 4 | E12 | GNN嵌入+表格 | GBR | 0.839 | 29.2 | GNN嵌入可行但不及VPD |
| 5 | E1 | **L1H 34d (基线)** | GBR | **0.820** | 34.9 | **基线** |
| — | E9-15 | GNN 端到端 | — | <0 | >100 | **完全失败** |

**核心结论**：
1. VPD +4.6%（0.820 → 0.866），突破单体描述符天花板
2. GNN 端到端在 304 样本上不可行（参数/样本比 >300:1）
3. 物理特征 + 浅层模型 = 小数据最优策略

### 4.2 Phase 5：大数据模型对比（E16-E24，统一 7,486条）

| 排名 | 模型 | CV R² | CV MAE(K) | 备注 |
|------|------|-------|----------|------|
| 1 | **TabPFN v2** | **0.8955** | **24.05** | **零调参即SOTA** |
| 2 | Stacking v2 | 0.8750 | 27.99 | 4模型集成 |
| 3 | CatBoost (Optuna) | 0.8742 | — | 20轮调参仅+0.5% |
| 4 | XGBoost (Optuna) | 0.8722 | — | +0.1% |
| 5 | LightGBM (Optuna) | 0.8699 | — | — |
| 6 | ExtraTrees | 0.8606 | 30.28 | — |
| 7 | GBR | 0.8555 | 31.03 | — |

**核心结论**：
1. **TabPFN 零调参超越所有 Optuna 调参模型 ~2%** — 特征质量 > 模型复杂度
2. Optuna 调参收益极小（最高 CatBoost +0.5%）
3. Stacking 在 7K+ 数据上首次有效（Phase 1-3 在 304 条上 3 次失败）

### 4.3 SHAP 特征重要度（E23）

| 排名 | 特征 | SHAP | 来源 | 物理含义 |
|------|------|------|------|---------|
| 1 | **FlexibilityIndex** | **41.51** | Afsordeh | 主链柔性 |
| 2 | VPD_RingCount_per_RU | 15.53 | VPD | 每RU环数 |
| 3 | RingCount | 12.55 | RDKit | 单体环数 |
| 4 | VPD_junction_flex_ratio | 10.48 | VPD | 连接柔性 |
| 5 | FractionCSP3 | 10.00 | RDKit | sp3碳比例 |
| 6 | VPD_TPSA_per_RU | 7.98 | VPD | 极性/RU |
| 7 | ced_weighted_sum | 6.89 | HBond | 内聚能密度 |

**物理解读**：Tg 由**链柔性**（#1,4,5）和**分子间作用力**（#2,3,6,7）共同决定，完全吻合 Gibbs-DiMarzio 理论。

**特征选择帕累托最优**：
| 特征数 | R² | 结论 |
|--------|-----|------|
| 46 (全量) | 0.8612 | 基线 |
| **15 (Top-K)** | **0.8619** | **零损失，仅需 1/3** |
| 10 | 0.8541 | 轻微下降 |
| 5 | 0.8220 | 明显下降 |

### 4.4 多尺度重构结果（Phase A-E）

| Phase | 特征集 | 维度 | 最优模型 | R² | vs 上一阶段 |
|-------|--------|------|---------|-----|-----------|
| — | M2M-V (基线) | 46d | CatBoost | 0.867 | — |
| B | PHY | 48d | CatBoost | 0.872 | +0.6% |
| B2 | PHY-B2 (+链间) | 56d | CatBoost | 0.884 | +0.8% |
| C | PHY-C-light (+链段) | 58d | TabPFN | 0.905 | **突破 0.9** |
| D | +GNN+polyBERT | 186d | TabPFN | 0.917 | +1.2% |
| E | 专家委员会 | 186d | TabPFN直接 | **0.917** | **最终SOTA** |

**Phase E 关键发现**：物理专家委员会（多个专门模型的集成）没有超越 TabPFN 直接在 186d 上的表现。TabPFN 的元学习能力已经自动完成了特征选择和加权。

### 4.5 核酸迁移预测（E26）

| 核苷酸 | 预测误差 (K) | 评价 |
|--------|-------------|------|
| **ATP** | **4.7** | 优秀 |
| **ADP** | **0.4** | 优秀 |
| AMP | 8.7 | 良好 |
| GMP | 14.0 | 一般 |
| UMP | 84.5 | 失败 |
| CMP | 77.7 | 失败 |

**关键发现**：
1. L1H (34d) 对核苷酸 > M2M-V (46d)：VPD 基于 3-RU 寡聚体，不适用于核苷酸小分子
2. 嘌呤类（ATP/ADP/AMP）预测好，嘧啶类（UMP/CMP）预测差 — 训练数据缺乏嘧啶结构
3. bridge_weight=1.0 最优
4. **这是文献空白** — 没有人尝试过从通用聚合物预测核酸 Tg

---

## 第五部分：GNN 方案

### 5.1 架构（PhysicsGAT）

```
原子特征 25d
    ↓
GATConv(25→128, heads=4) + BatchNorm + ELU
    ↓
GATConv(128→128, heads=4) + BatchNorm + ELU
    ↓
GATConv(128→64, heads=1) + ELU
    ↓
GRIN 池化（只聚合中间重复单元原子）
    ↓
64维 图嵌入
```

### 5.2 为什么端到端失败

- 304 样本 vs GNN 数万参数 → 参数/样本比 >300:1
- 严重过拟合，训练 R² 接近 1.0，测试 R² < 0
- **教训**：小数据场景，精心设计的物理特征 > 黑箱深度学习

### 5.3 为什么嵌入方式有效

- GNN 编码结构信息 → 64d 嵌入作为特征输入表格模型
- 表格模型（GBR/CatBoost/TabPFN）只需要少量参数
- 信息互补：GNN 提供结构，物理特征提供可解释性
- E12：GNN嵌入+表格 R²=0.839（可行但不及 VPD 的 0.866）
- Phase D：GNN 64d + PHY-C 58d + polyBERT 64d → TabPFN R²=0.917

---

## 第六部分：代码架构

### 6.1 目录结构与行数

```
src/
├── features/                    # 特征工程（~1,750 行）
│   ├── afsordeh_features.py       85行   4维基础物理
│   ├── rdkit_descriptors.py       93行   15维RDKit描述符
│   ├── hbond_features.py         282行   15/5维氢键特征
│   ├── virtual_polymerization.py 332行   12维VPD（核心创新）
│   ├── physical_proxy.py         436行   10维物理代理
│   ├── gc_tg.py                  247行   2维基团贡献
│   ├── feature_pipeline.py       263行   统一特征管道
│   ├── interchain_features.py    212行   8维链间相互作用
│   └── chain_physics_cache.py    118行   8维链段物理
│
├── ml/                          # 机器学习（~1,830 行）
│   ├── evaluation.py             644行   Nested CV 框架
│   ├── sklearn_models.py         428行   模型库 + Stacking + Optuna
│   ├── uncertainty.py            389行   MAPIE 不确定性量化
│   └── hierarchical_model.py     370行   层级残差学习
│
├── gnn/                         # 图神经网络（方案B，待GPU）
│   ├── graph_builder.py                  分子图构建
│   ├── physics_gat.py                    物理感知GAT
│   ├── tandem_m2m.py                     图+表格融合
│   ├── pretrainer.py                     GNN 预训练
│   ├── multitask.py                      多任务学习
│   └── ensemble.py                       集成方法
│
├── data/                        # 数据集
├── analysis/                    # SHAP + 可视化
├── bigsmiles/                   # BigSMILES 工具链（寒假项目搬来）
└── sequence/                    # 核酸序列处理

scripts/                         # 实验脚本
├── exp_phase2.py                  Phase 2 实验
├── exp_phase3.py                  Phase 3 实验
├── exp_phase4.py                  Phase 4 消融 (E1-E15)
├── exp_phase5.py                  Phase 5 模型对比 (E16-E24)
├── exp_phase5b.py                 Phase 5B 收尾 (E25-E26)
└── predict_tg_from_sequence.py    核酸序列预测工具

tests/                           # 14个测试文件
results/                         # 实验结果 JSON + 总结
```

### 6.2 关键数据流

```
SMILES 字符串
    ↓
feature_pipeline.compute_features(smiles, layer="PHY")
    ├── afsordeh_features.compute_afsordeh_4(smiles)     → 4d
    ├── rdkit_descriptors.compute_l1_descriptors(smiles)  → 15d
    ├── hbond_features.compute_hbond_slim(smiles)         → 5d
    ├── physical_proxy.compute_ppf(smiles)                → 10d
    ├── virtual_polymerization.compute_vpd(smiles)        → 12d
    └── gc_tg.compute_gc_tg(smiles)                       → 2d
    ↓
48维特征向量
    ↓
PowerTransformer + MinMaxScaler
    ↓
模型预测 → Tg (K)
```

---

## 第七部分：实验编号索引

| 编号 | Phase | 内容 | 数据 | 关键结果 |
|------|-------|------|------|---------|
| E1 | 4 | L1H 34d 基线 | 304 | R²=0.820 |
| E3 | 4 | **M2M-V 46d + GBR** | 304 | **R²=0.866 (+4.6%)** |
| E9-15 | 4 | GNN 端到端 | 304 | **R²<0（失败）** |
| E12 | 4 | GNN嵌入+GBR | 304 | R²=0.839 |
| E16 | 5 | GBR 基线 | 7,486 | R²=0.856 |
| E17 | 5 | CatBoost | 7,486 | R²=0.869 |
| E19 | 5 | **TabPFN v2** | 7,486 | **R²=0.896（零调参SOTA）** |
| E21 | 5 | Stacking v2 | 7,486 | R²=0.875 |
| E22 | 5 | 不确定性量化 | 7,486 | 90%覆盖率, 129K区间 |
| E23 | 5 | SHAP特征选择 | 7,486 | Top15 无损 |
| E26 | 5B | 核酸迁移预测 | 7,691 | ATP 4.7K, ADP 0.4K |
| — | A-E | 多尺度重构 | 7,486 | **R²=0.917（最终SOTA）** |

---

## 第八部分：两个项目的完整逻辑链

```
寒假：BigSMILES 工具链
"让计算机读懂高分子"

    解析器 → 生成器 → 规范化 → 校验 → 数据集 → 初步指纹 → 初步预测(R²=0.58)
                                                                    │
                                                                    ↓
本学期：Tg 预测
"读懂之后预测性质"

    数据扩展(304→7,486) → 特征工程(4d→186d) → 模型训练(7种) → R²=0.917
                                │
                                ├── VPD 虚拟聚合（核心创新，+4.6%）
                                ├── 物理代理（Schneider-DiMarzio 理论）
                                ├── 基团贡献（Van Krevelen）
                                ├── GNN 嵌入（图结构编码）
                                └── polyBERT（化学语言模型）
                                                    │
                                                    ↓
                                            核酸迁移（ATP/ADP <5K）
                                            "填补文献空白"
```
