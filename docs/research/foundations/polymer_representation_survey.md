# 聚合物分子表示方法调研 — 面向 Tg 预测场景

> 调研日期: 2026-03-13
> 本项目使用单体 SMILES 计算描述符预测聚合物 Tg (当前 R2=0.84). 本报告分析方法论局限并调研替代方案.

---

## 一、单体 SMILES 表示的根本局限

### 1.1 Tg 影响因素多尺度分析

| 层级 | 影响因素 | 可从单体 SMILES 推导? | 说明 |
|------|----------|:-------------------:|------|
| 单体级 | 主链刚性 (backbone stiffness) | 部分 | 双键、芳环增加刚性 |
| 单体级 | 侧基体积与极性 | 是 | Cl/CN/OH 显著提高 Tg |
| 单体级 | 分子间力 (H-bond, dipole) | 部分 | 可推断但无法量化实际分子间距 |
| 链级 | 分子量/聚合度 (DP) | 否 | Flory-Fox: Tg=Tg(inf)-K/Mn |
| 链级 | 链端基效应 | 否 | 链端自由体积大 |
| 链级 | 立构规整度 (tacticity) | 否 | iso/syndio/atactic PP 差 60K |
| 超分子级 | 结晶度 | 否 | 约束非晶区链段运动 |
| 超分子级 | 交联密度 | 否 | 限制链段运动 |
| 超分子级 | 链缠结 | 否 | 影响松弛时间 |
| 超分子级 | pi-pi 堆叠 | 否 | 影响自由体积 |
| 实验级 | 冷却速率 | 否 | Tg 非平衡态 |
| 实验级 | 测量方法 (DSC/DMA) | 否 | 不同方法差 5-15K |
| 环境级 | 增塑剂/含水量 | 否 | 水是天然增塑剂 |

**核心结论**: 单体 SMILES 只能捕获单体级信息 (约占 Tg 变异 60-80%). 链级/超分子级/实验级信息完全丢失. R2 约 0.85-0.90 是理论天花板.

### 1.2 信息论视角

单体 SMILES 映射: `f: 单体结构 -> Tg`

真实映射: `g: (单体结构, DP, 立构, 加工条件, ...) -> Tg`

只用单体建模 = 取条件期望: `f(m) = E[Tg|m] +/- sigma`, sigma 是不可消除噪声.

Bicerano 数据集标注 Tg(inf) 消除了 MW 效应, 但立构/结晶度/实验条件差异仍在.

### 1.3 文献佐证

- **Afsordeh 2025**: 4 个物理描述符 (flexibility, side chain length, polarity, H-bond) -> ET 模型 R2=0.97. 精心设计 > 盲目堆叠.
- **Van Krevelen 改进 (2020)**: 198 种均聚物 R2=0.99, MRE=8%. Polyolefins 误差大.
- 独立验证集通常降到 R2=0.90-0.94 (数据集间测量标准不一致).

### 1.4 数据集问题

1. Tg 测量不确定性 +/-20-30K
2. Tg(inf) vs 实测 Tg 标注不统一
3. 数据量极小 (200-1500 条)
4. 高 Tg 聚合物稀缺

---

## 二、聚合物表示方法全景

### 2.1 方法对比

| 方法 | 编码对象 | 优势 | 劣势 | 适用场景 |
|------|---------|------|------|---------|
| SMILES | 单体 | 简单通用 | 无聚合物级信息 | 快速基线 |
| PSMILES | 重复单元 | [*]标记连接点 | 无DP/立构 | 均聚物预测 |
| BigSMILES | 完整聚合物 | 编码随机性 | 语法复杂/数据少 | 共聚物 |
| HAPPY | 层级抽象 | 紧凑/训练快2x | 新方法 | 小数据DL |
| Group Contribution | 化学基团 | 物理可解释 | 缺组不可用 | 快速估算 |
| Morgan FP | 子结构 | 标准化 | 信息损失 | 传统ML |
| 物理描述符 | 物理量 | 高可解释 | 需领域知识 | 小数据 |
| polyBERT | PSMILES token | 自动表示 | 需预训练 | 大规模 |
| GNN | 分子图 | 拓扑信息 | 图截断难题 | 拓扑敏感 |
| 多模态 | SMILES+3D+text | 信息丰富 | 成本高 | SOTA |

### 2.2 PSMILES

用 `[*]` 标记重复单元连接点: `[*]CC[*]` = 聚乙烯. polyBERT 在 PSMILES 上预训练. 局限: 仍只编码化学结构, 无 DP/MW/tacticity.

### 2.3 BigSMILES

2024 系统比较 (12 任务): 性能可比/略优 SMILES; LLM 训练更快; 共聚物单体连接信息更准确. 本项目处理均聚物, 边际收益有限.

### 2.4 HAPPY (2024)

子结构映射为单字符 + 独立连接符. 300-500 条数据上 R2 高于 SMILES, 训练快 2 倍.

### 2.5 GNN

核心挑战: 如何表示无限长重复结构.
- Monomer Graph: 只用单体图 (丢失连接信息)
- Repeat Unit Graph: + dummy node
- Oligomer Graph: 拼接 n 个重复单元
- Polymer-Unit Graph (2024): 训练时间减 98%

### 2.6 预训练表示

| 模型 | 策略 | 维度 | 小数据 |
|------|------|------|--------|
| polyBERT | Masked LM | 768 | 中等 |
| TransPolymer | Transformer | 512 | 中等 |
| MMPolymer | SMILES+3D | 256 | 较好 |
| PolyLLMem | Llama3+UniMol | 4096 | 好 |
| Mol2Vec | Word2Vec | 300 | 中等 |

300 条数据: 传统ML + 物理描述符仍是最务实选择.

### 2.7 Group Contribution vs ML

Van Krevelen 改进 (2020): R2=0.99/MRE=8% (198 均聚物). Afsordeh 物理描述符: R2=0.97 (ET). 两者证明: 物理知识驱动 > 纯数据驱动.

---

## 三、核酸 Tg 的特殊性

### 3.1 Tm vs Tg — 核心区分

| | Tm (melting) | Tg (glass transition) |
|---|---|---|
| 过程 | 双链解离 | 非晶->橡胶态 |
| 典型值 | 40-100C (溶液) | -50C (223-234K, 干态) |
| 因素 | 碱基序列/离子/链长 | 水含量/链构象 |

DNA Tg 约 -50C (MD 模拟), 远低于合成聚合物.

### 3.2 核酸 Tm 决定因素

1. 碱基配对稳定性 (GC: 3 H-bond > AT: 2 H-bond)
2. 碱基堆叠 (base stacking, pi-pi)
3. 近邻效应 (nearest-neighbor): Tm 取决于相邻碱基对组合
4. 离子浓度 (Na+/Mg2+ 屏蔽磷酸骨架)
5. 链长 (短链 Tm 链长敏感)
6. 序列结构 (hairpin/bulge/mismatch)
7. DNA 浓度

### 3.3 合成聚合物模型无法预测核酸的根本原因

- **物理过程不同**: Tg = 链段运动解冻; Tm = 氢键+堆叠力断裂
- **化学空间不重叠**: C-C/C-O 骨架 vs 糖-磷酸骨架
- **关键信息缺失**: 序列/近邻效应/离子浓度无法从 SMILES 提取

### 3.4 Nearest-Neighbor Model — 核酸 Tm 黄金标准

SantaLucia (1998) 统一近邻参数:
- 10 个 Watson-Crick 近邻参数 (dH + dS)
- `Tm = dH / (dS + R*ln(CT/4))`
- 精度: r=0.96, 误差 +/-2-3C
- 2025: 高通量实验改进 mismatch/bulge/hairpin 参数 (Nature Communications)

### 3.5 核酸固态 Tg

干燥 DNA Tg 约 -50C, 与水含量高度相关. 生物学意义有限 (体内始终水溶液).

---

## 四、前沿趋势与建议

### 4.1 小数据集 (300 条) 最佳策略

| 排名 | 方法 | 预期 R2 | 复杂度 |
|------|------|---------|--------|
| 1 | 物理描述符 + ExtraTrees | 0.90-0.97 | Low |
| 2 | HAPPY + 轻量 NN | 0.88-0.93 | Medium |
| 3 | 迁移学习 (预训练+微调) | 0.85-0.90 | Medium |
| 4 | polyBERT 嵌入 + ML | 0.83-0.88 | Medium |
| 5 | Morgan FP + ML | 0.80-0.85 | Low |
| 6 | GNN | 0.78-0.85 | High |
| 7 | 端到端 DL | 0.70-0.80 | High |

**洞见**: 小数据上, 领域知识驱动特征工程 > 自动化特征学习.

### 4.2 前沿方向

1. **多模态融合** (2024-25 主流): MMPolymer, PolyLLMem, Uni-Poly
2. **LLM 迁移** (2025 新兴): LoRA 微调, 精确数值任务优势待验证
3. **多任务学习** (2023-25): 辅助任务提供正则化
4. **物理信息增强**: group contribution 知识注入 ML

### 4.3 本项目提升建议

| 改进方向 | 预期收益 | 难度 | 优先级 |
|---------|---------|------|--------|
| 优化物理描述符 (参考 Afsordeh) | +5-10% R2 | Low | 最高 |
| Group contribution 特征 | +2-5% R2 | Medium | 高 |
| PSMILES + polyBERT | +1-3% R2 | Medium | 中 |
| 多模态融合 | +3-5% R2 | High | 低 |

### 4.4 核酸预测结论

用合成聚合物 Tg 模型预测核酸热稳定性是**根本性方法论问题**:

1. 核酸热稳定性由 Tm 决定, 不是 Tg
2. Tm 和 Tg 物理机制完全不同
3. 核酸 Tm 已有 Nearest-Neighbor Model (r=0.96)
4. 固态核酸 Tg 约 -50C, 与合成聚合物数据分布不重叠

**建议**: 论文中作为 negative result + 理论分析贡献.

---

## 参考文献

### 聚合物表示
1. Afsordeh & Shirali (2025). Chinese J. Polymer Sci. 43:1661-1670
2. Kim et al. (2024). npj Comput. Mater. [HAPPY]
3. Kuenneth et al. (2023). Nat. Commun. [polyBERT]
4. Lin & Olsen (2019). BigSMILES. MIT
5. Is BigSMILES the Friend of Polymer ML? (2024). ChemRxiv
6. Afzal & Hasan (2020). ACS Omega [Modified Group Contribution]
7. Van Krevelen (2009). Properties of Polymers. Elsevier
8. Xu et al. (2024). MMPolymer. arXiv
9. Bejagam et al. (2023). Chem. Mater. [Multitask GNN]
10. Zhou et al. (2024). PMC [Simple ML for Tg]

### 核酸热力学
11. SantaLucia (1998). PNAS 95:1460-1465
12. Saha et al. (2025). Nat. Commun. [DNA melt]
13. Glass transition in DNA from MD (2001). PMC
14. Le Novere (2012). BMC Bioinf. [MELTING]
15. Chain stability of solid-state DNA (2010). PMC

### Tg 影响因素
16. Fox & Flory (1950). Flory-Fox equation
17. Gao et al. (2024). Macromolecules [MW and Tg]
18. NCSU Advances in Polymer Science [Free Volume]

> **调研结论**: 单体 SMILES 已接近理论天花板 (R2 约 0.85-0.90). 提升路径: 优化物理描述符 (Afsordeh 2025). 核酸预测失败源于物理机制不匹配, 应作为 negative result 讨论.