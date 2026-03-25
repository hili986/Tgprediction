# 深度学习方法预测聚合物玻璃化转变温度 (Tg) — 调研报告

> 调研日期：2026-03-13

> 调研范围：2023-2026

> 覆盖方向：GNN, Transformer, 预训练模型, 多尺度建模, 小数据集策略

---

## 目录

1. [GNN (图神经网络) 方法](#1-gnn-图神经网络-方法)
2. [Transformer / 序列模型方法](#2-transformer--序列模型方法)
3. [预训练模型](#3-预训练模型)
4. [多尺度建模](#4-多尺度建模)
5. [小数据集策略](#5-小数据集策略)
6. [LLM 直接预测](#6-llm-直接预测)
7. [综合性能对比](#7-综合性能对比)
8. [项目建议](#8-项目建议)
9. [参考文献](#9-参考文献)

---

## 1. GNN (\u56fe\u795e\u7ecf\u7f51\u7edc) \u65b9\u6cd5

### 1.1 \u805a\u5408\u7269\u56fe\u8868\u793a\u65b9\u6cd5\u603b\u89c8

\u805a\u5408\u7269\u7684\u56fe\u8868\u793a\u662f GNN \u65b9\u6cd5\u7684\u6838\u5fc3\u95ee\u9898\u3002\u76ee\u524d\u4e3b\u8981\u6709\u4e09\u79cd\u8868\u793a\u7b56\u7565\uff1a

| \u8868\u793a\u65b9\u6cd5 | \u63cf\u8ff0 | \u4f18\u52bf | \u52a3\u52bf |
|----------|------|------|------|
| **Monomer Graph** | \u4ec5\u4ee5\u5355\u4f53/\u91cd\u590d\u5355\u5143\u6784\u5efa\u56fe | \u7b80\u5355\u3001\u901a\u7528 | \u4e22\u5931\u94fe\u7ea7\u4fe1\u606f |
| **Repeat Unit Graph** | \u91cd\u590d\u5355\u5143 + \u94fe\u63a5\u70b9\u6807\u8bb0 | \u4fdd\u7559\u805a\u5408\u4f4d\u70b9 | \u65e0\u6cd5\u8868\u793a\u5468\u671f\u6027 |
| **Periodic Polymer Graph** | \u5728\u8fb9\u754c\u539f\u5b50\u95f4\u6dfb\u52a0\u952e\u5f62\u6210\u73af | \u6355\u6349\u5468\u671f\u6027 | \u8ba1\u7b97\u590d\u6742 |

\u5173\u952e\u6d1e\u5bdf\uff1aperiodic polymer graph \u901a\u8fc7\u5728\u91cd\u590d\u5355\u5143\u7684\u5934\u5c3e\u539f\u5b50\u4e4b\u95f4\u6dfb\u52a0\u5316\u5b66\u952e\uff0c\u5f62\u6210\u73af\u72b6\u7ed3\u6784\uff0c\u4ece\u800c\u6a21\u62df\u805a\u5408\u7269\u94fe\u7684\u5468\u671f\u6027\u3002\u8fd9\u7c7b\u4f3c\u4e8e\u56fa\u4f53\u7269\u7406\u4e2d\u7684\u5468\u671f\u6027\u8fb9\u754c\u6761\u4ef6 (PBC)\u3002

### 1.2 \u4ee3\u8868\u6027\u8bba\u6587

#### 1.2.1 polyGNN (Ramprasad Group, 2023)

- **\u8bba\u6587**: "Polymer Graph Neural Networks for Multitask Property Learning"
- **\u53d1\u8868**: J. Phys. Chem. Lett., 2023
- **\u6838\u5fc3\u65b9\u6cd5**:
  - \u91c7\u7528 periodic polymer graph \u8868\u793a\uff0c\u5728 PSMILES \u7684\u5934\u5c3e `[*]` \u539f\u5b50\u95f4\u6dfb\u52a0\u952e
  - \u591a\u4efb\u52a1\u5b66\u4e60\uff1a\u540c\u65f6\u9884\u6d4b Tg\u3001Tm\u3001\u5bc6\u5ea6\u7b49\u591a\u4e2a\u6027\u8d28
  - message passing + global readout \u67b6\u6784
- **\u6027\u80fd**:
  - \u5355\u4efb\u52a1 Tg RMSE = 36.6 K
  - \u591a\u4efb\u52a1 Tg RMSE = 31.7 K\uff08\u63d0\u5347 13.4%\uff09
  - \u591a\u4efb\u52a1\u5b66\u4e60\u663e\u8457\u63d0\u5347\u5c0f\u6570\u636e\u96c6\u6027\u8d28\u7684\u9884\u6d4b
- **\u5c0f\u6570\u636e\u96c6\u9002\u7528\u6027**: \u2605\u2605\u2605\u2606\u2606 \u2014 \u591a\u4efb\u52a1\u5b66\u4e60\u663e\u8457\u63d0\u5347\u5c0f\u6570\u636e\u96c6\u6027\u80fd\uff0c\u4f46 GNN \u672c\u8eab\u4ecd\u9700 ~1000+ \u6570\u636e\u70b9

#### 1.2.2 PolymerGNN (Tao et al., 2023)

- **\u8bba\u6587**: "Benchmarking GNNs for Polymer Informatics: Accelerating Polymer Property Prediction"
- **\u53d1\u8868**: J. Chem. Inf. Model., 2023 (\u5f15\u7528\u91cf ~100+)
- **\u6838\u5fc3\u65b9\u6cd5**:
  - Periodic graph + MPNN (Message Passing Neural Network) \u67b6\u6784
  - \u5f15\u5165 Deep Set \u5904\u7406\u5171\u805a\u7269\u7684\u591a\u5355\u4f53\u7ec4\u6210
  - \u652f\u6301 PSMILES \u8f93\u5165\uff0c\u81ea\u52a8\u751f\u6210 periodic graph
- **\u6027\u80fd**:
  - R\u00b2 = 0.89\uff0c\u5728 1,251 \u6761\u6d4b\u8bd5\u96c6\u4e0a
  - \u5bf9\u5171\u805a\u7269\u7684\u652f\u6301\u662f\u91cd\u8981\u4f18\u52bf
- **\u5c0f\u6570\u636e\u96c6\u9002\u7528\u6027**: \u2605\u2605\u2606\u2606\u2606 \u2014 \u7eaf GNN\uff0c\u9700\u8981\u8f83\u5927\u6570\u636e\u96c6

#### 1.2.3 Data-Augmented GCN (Yamada et al., ACS AMI 2023)

- **\u8bba\u6587**: "Predicting Glass Transition Temperatures of Polymers Using Graph Convolutional Networks with Data Augmentation"
- **\u53d1\u8868**: ACS Applied Materials & Interfaces, 2023
- **\u6838\u5fc3\u65b9\u6cd5**:
  - GCN + SMILES enumeration data augmentation
  - \u5bf9\u540c\u4e00\u5206\u5b50\u751f\u6210\u591a\u4e2a\u7b49\u4ef7 SMILES\uff0c\u6269\u5927\u8bad\u7ec3\u6570\u636e
  - \u7c7b\u4f3c\u56fe\u50cf\u8bc6\u522b\u4e2d\u7684\u6570\u636e\u589e\u5f3a\uff08\u65cb\u8f6c\u3001\u7ffb\u8f6c\uff09
- **\u6027\u80fd** (\u91cd\u70b9):
  - \u65e0\u589e\u5f3a: R\u00b2 = 0.88, RMSE = 19.4 K
  - **\u6709\u589e\u5f3a: R\u00b2 = 0.97, RMSE = 7.4 K**
  - \u589e\u5f3a\u540e\u6027\u80fd\u63d0\u5347\u5de8\u5927\uff08RMSE \u964d\u4f4e 62%\uff09
- **\u5c0f\u6570\u636e\u96c6\u9002\u7528\u6027**: \u2605\u2605\u2605\u2605\u2605 \u2014 SMILES enumeration \u662f\u5c0f\u6570\u636e\u96c6\u6700\u6709\u6548\u7684\u589e\u5f3a\u7b56\u7565\u4e4b\u4e00

#### 1.2.4 Lieconv-Tg (3D Equivariant NN, 2024)

- **\u8bba\u6587**: "Predicting Tg from 3D Molecular Structures Using Equivariant Neural Networks"
- **\u53d1\u8868**: 2024
- **\u6838\u5fc3\u65b9\u6cd5**:
  - \u4f7f\u7528 3D \u5206\u5b50\u5750\u6807 + Lie \u7fa4\u7b49\u53d8\u795e\u7ecf\u7f51\u7edc
  - \u8003\u8651\u5206\u5b50\u7684\u65cb\u8f6c/\u5e73\u79fb\u4e0d\u53d8\u6027
  - \u4ece conformer ensemble \u4e2d\u63d0\u53d6\u7279\u5f81
- **\u6027\u80fd**: R\u00b2 = 0.90, MAE = 24.42 K\uff08\u57fa\u4e8e 7,166 \u6761\u6570\u636e\uff09
- **\u5c0f\u6570\u636e\u96c6\u9002\u7528\u6027**: \u2605\u2605\u2606\u2606\u2606 \u2014 \u9700\u8981 3D \u5750\u6807\u751f\u6210\u548c\u5927\u6570\u636e\u96c6

#### 1.2.5 MSRGCN-RL (Langmuir, 2024)

- **\u8bba\u6587**: "Multi-Scale Relational Graph Convolutional Network with Reinforcement Learning for Polymer Tg Prediction"
- **\u53d1\u8868**: Langmuir, 2024
- **\u6838\u5fc3\u65b9\u6cd5**:
  - \u591a\u5c3a\u5ea6\u5173\u7cfb\u56fe\u7f51\u7edc + \u5f3a\u5316\u5b66\u4e60\u81ea\u52a8\u67b6\u6784\u641c\u7d22
  - \u540c\u65f6\u5efa\u6a21\u539f\u5b50\u7ea7\u548c\u5b50\u7ed3\u6784\u7ea7\u7279\u5f81
  - RL agent \u81ea\u52a8\u9009\u62e9\u6700\u4f18 GNN \u67b6\u6784
- **\u5c0f\u6570\u636e\u96c6\u9002\u7528\u6027**: \u2605\u2605\u2606\u2606\u2606 \u2014 \u67b6\u6784\u590d\u6742\uff0c\u4e0d\u9002\u5408\u5c0f\u6570\u636e\u96c6

#### 1.2.6 GATBoost (2024)

- **\u6838\u5fc3\u65b9\u6cd5**:
  - Graph Attention Network \u63d0\u53d6\u5b50\u7ed3\u6784\u7279\u5f81
  - GAT \u8f93\u51fa\u4f5c\u4e3a gradient boosting \u7684\u8f93\u5165\u7279\u5f81
  - \u6df1\u5ea6\u5b66\u4e60\u7279\u5f81 + \u4f20\u7edf ML \u7684\u6df7\u5408\u67b6\u6784
- **\u6027\u80fd**: \u4e0e\u7eaf GNN \u76f8\u5f53\uff0c\u4f46 interpretability \u66f4\u597d
- **\u5c0f\u6570\u636e\u96c6\u9002\u7528\u6027**: \u2605\u2605\u2605\u2606\u2606 \u2014 \u6df7\u5408\u67b6\u6784\u5bf9\u5c0f\u6570\u636e\u96c6\u66f4\u53cb\u597d

#### 1.2.7 Self-Supervised GNN (2024)

- **\u6838\u5fc3\u65b9\u6cd5**:
  - Node-level: \u8282\u70b9\u5c5e\u6027\u9884\u6d4b\uff08\u539f\u5b50\u7c7b\u578b\u3001\u5ea6\u6570\uff09
  - Edge-level: \u952e\u7c7b\u578b\u9884\u6d4b
  - Graph-level: \u5b50\u56fe\u5bf9\u6bd4\u5b66\u4e60 (contrastive learning)
  - \u9884\u8bad\u7ec3\u540e\u5fae\u8c03\u5230\u4e0b\u6e38 Tg \u4efb\u52a1
- **\u5c0f\u6570\u636e\u96c6\u9002\u7528\u6027**: \u2605\u2605\u2605\u2605\u2606 \u2014 \u81ea\u76d1\u7763\u9884\u8bad\u7ec3\u53ef\u5229\u7528\u65e0\u6807\u7b7e\u6570\u636e

### 1.3 GNN \u65b9\u6cd5\u5c0f\u7ed3

| \u65b9\u6cd5 | \u5e74\u4efd | \u56fe\u8868\u793a | \u6700\u4f73\u6027\u80fd | \u6570\u636e\u91cf |
|------|------|--------|--------|--------|
| polyGNN | 2023 | Periodic | RMSE=31.7K | ~5000 |
| PolymerGNN | 2023 | Periodic+DeepSet | R\u00b2=0.89 | ~6000 |
| Data-Aug GCN | 2023 | Monomer+Aug | **R\u00b2=0.97** | ~500+aug |
| Lieconv-Tg | 2024 | 3D coords | R\u00b2=0.90 | 7166 |
| MSRGCN-RL | 2024 | Multi-scale | -- | ~5000 |

**\u5173\u952e\u7ed3\u8bba**: \u5728 GNN \u65b9\u6cd5\u4e2d\uff0cSMILES enumeration \u6570\u636e\u589e\u5f3a (Data-Aug GCN) \u662f\u5c0f\u6570\u636e\u96c6\u573a\u666f\u4e0b\u6700\u6709\u6548\u7684\u7b56\u7565\uff0c\u53ef\u5c06 R\u00b2 \u4ece 0.88 \u63d0\u5347\u81f3 0.97\u3002

---

## 2. Transformer / 序列模型方法

### 2.1 核心思路

Transformer 方法将聚合物 SMILES/PSMILES 视为"语言"，利用自注意力机制捕捉原子间的长程依赖关系。核心优势在于可以直接从字符串表示学习，无需手工特征工程。

### 2.2 代表性论文

#### 2.2.1 TransPolymer (Xu et al., 2023)

- **论文**: "TransPolymer: A Transformer-based Language Model for Polymer Property Predictions"
- **发表**: npj Computational Materials, 2023 (引用量 100+)
- **核心方法**:
  - 基于 RoBERTa 架构，PSMILES tokenizer
  - 在 500 万条 hypothetical PSMILES 上预训练 (MLM 任务)
  - 下游微调: Transformer encoder + MLP regression head
  - 10 个聚合物性质 benchmark 上达到 SOTA
- **Tg 性能**:
  - 在标准 Tg benchmark 上优于 GP、RF、XGBoost
  - 具体 R² ~0.85 (依赖数据集划分)
- **小数据集适用性**: ★★★★☆ — 预训练+微调范式天然适合小数据集

#### 2.2.2 TransTg (2024)

- **论文**: "TransTg: A SMILES-to-Tg Transformer for Glass Transition Temperature Prediction"
- **发表**: 2024
- **核心方法**:
  - 专门为 Tg 预测设计的 Transformer
  - SMILES 字符级 tokenization + positional encoding
  - Multi-head self-attention + feed-forward layers
  - 端到端训练，无需预训练
- **性能**: R² = 0.849, MAE = 22.55 K
- **小数据集适用性**: ★★★☆☆ — 无预训练，小数据集上表现有限

#### 2.2.3 Multimodal Transformer (ACS AMI, 2024)

- **论文**: "Multimodal Transformer for Polymer Property Prediction"
- **发表**: ACS Applied Materials & Interfaces, 2024
- **核心方法**:
  - 双通道输入: SMILES 序列 + 分子图
  - Cross-attention 融合文本和图特征
  - Dimer configuration (二聚体) 表现最优
  - 覆盖 5 个聚合物性质
- **性能**: Tg 预测优于单模态模型
- **小数据集适用性**: ★★★☆☆ — 多模态增加数据需求

#### 2.2.4 polyBART (EMNLP, 2025)

- **论文**: "polyBART: A Bidirectional Polymer Language Model"
- **发表**: EMNLP, 2025
- **核心方法**:
  - 首个双向聚合物语言模型 (encoder-decoder)
  - 使用 PSELFIES 表示 (比 PSMILES 更鲁棒)
  - 支持双向: 结构→性质 AND 性质→结构 (inverse design)
  - 生成式 + 判别式统一框架
- **性能**: 多个性质预测 benchmark 上有竞争力
- **小数据集适用性**: ★★★★☆ — 双向能力提供额外训练信号

### 2.3 Transformer 方法小结

| 方法 | 年份 | 预训练 | Tg 性能 | 关键特色 |
|------|------|--------|---------|----------|
| TransPolymer | 2023 | 5M PSMILES | R²~0.85 | 10 性质 SOTA |
| TransTg | 2024 | 无 | R²=0.849, MAE=22.55K | Tg 专用 |
| Multimodal Trans | 2024 | 部分 | 优于单模态 | SMILES+图融合 |
| polyBART | 2025 | 是 | 多性质竞争力 | 双向 (结构↔性质) |

**关键结论**: 预训练是 Transformer 在小数据集上成功的关键。TransPolymer 的 5M PSMILES 预训练 + 小数据集微调是最实用的范式。

---

## 3. 预训练模型

### 3.1 核心思路

预训练模型先在大规模无标签/弱标签聚合物数据上学习通用表示，再微调到具体性质预测任务。这是解决聚合物领域标注数据稀缺问题的主要策略。

### 3.2 代表性模型

#### 3.2.1 polyBERT (Kuenneth & Ramprasad, Nature Communications 2023)

- **论文**: "polyBERT: A Chemical Language Model to Enable Fully Machine-Driven Ultrafast Polymer Informatics"
- **发表**: Nature Communications, 2023 (高影响力)
- **核心方法**:
  - 基于 DeBERTa 架构
  - 在 1 亿条 hypothetical PSMILES 上预训练
  - PSMILES tokenizer，支持聚合物特有的 `[*]` 标记
  - 输出 768 维 polymer fingerprint
  - 下游: fingerprint + 简单回归器 (RF/XGBoost)
- **性能**:
  - 29 个聚合物性质上 R² ~ 0.80 平均
  - Tg: R² 约 0.82-0.85 (依赖数据集)
  - **推理速度**: 比传统 PG fingerprint 快 215 倍
- **小数据集适用性**: ★★★★★ — 预训练 fingerprint 直接可用，无需训练神经网络
- **对本项目的价值**: polyBERT fingerprint + ExtraTrees 是最直接可用的深度学习增强方案

#### 3.2.2 ChemBERTa (Chithrananda et al., 2020; 持续更新至 2024)

- **论文**: "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction"
- **核心方法**:
  - 基于 RoBERTa 架构
  - 在 7700 万条 PubChem SMILES 上预训练
  - 通用分子表示，非聚合物专用
  - Hugging Face 上可直接使用
- **Tg 性能**:
  - 在 Uni-Poly benchmark 中，ChemBERTa 是最佳单模态 Tg 预测器
  - 优于 polyBERT 在某些数据集划分上
- **小数据集适用性**: ★★★★☆ — 预训练充分，但非聚合物专用
- **局限**: 训练在小分子上，聚合物的 `[*]` 标记可能不在词汇表中

#### 3.2.3 PolyCL (Contrastive Learning, Digital Discovery 2025)

- **论文**: "PolyCL: Contrastive Learning for Polymer Representation"
- **发表**: Digital Discovery, 2025
- **核心方法**:
  - 对比学习预训练: 同一聚合物的不同 SMILES 表示为正对，不同聚合物为负对
  - SMILES enumeration 自然生成对比学习的正样本
  - 无需标签数据即可学习有效表示
- **性能**: 7 个聚合物性质数据集中 4 个达到最佳 R²
- **小数据集适用性**: ★★★★★ — 无标签预训练 + 对比学习天然适合小数据

#### 3.2.4 MMPolymer (多模态预训练, 2024)

- **论文**: "MMPolymer: A Multimodal Multitask Pretraining Framework for Polymer Property Prediction"
- **发表**: 2024
- **核心方法**:
  - 3D 构象 + SMILES 双模态
  - 多任务预训练: MLM + 3D denoising + cross-modal alignment
  - 预训练后提取 unified polymer representation
- **性能**: 多个性质预测上优于单模态方法
- **小数据集适用性**: ★★★☆☆ — 需要 3D 构象数据，增加复杂度

#### 3.2.5 MatBERT (Walker et al., 2021; 应用至 2024)

- **核心方法**:
  - 在材料科学文献上预训练的 BERT
  - 主要处理文本描述，非结构化数据
  - 可用于从文献中自动提取 Tg 数据
- **Tg 相关性**: 不直接预测 Tg，但可辅助构建数据集
- **小数据集适用性**: 不适用于直接预测

### 3.3 预训练模型对比

| 模型 | 预训练数据 | 架构 | Tg R² | 聚合物专用 | 开源 |
|------|-----------|------|---------|-----------|------|
| polyBERT | 100M PSMILES | DeBERTa | ~0.82 | ✔ | ✔ |
| ChemBERTa | 77M PubChem | RoBERTa | 最佳单模态 | ✘ | ✔ |
| PolyCL | 无标签聚合物 | Contrastive | 4/7 最佳 | ✔ | ✔ |
| MMPolymer | 聚合物+3D | Multimodal | 多性质优 | ✔ | 部分 |
| MatBERT | 材料文献 | BERT | N/A | ✘ | ✔ |

**关键结论**: polyBERT fingerprint 是最实用的方案 — 预训练 embedding 直接作为传统 ML 的输入特征，不增加模型复杂度，且推理极快。

---

## 4. 多尺度建模

### 4.1 核心挑战

聚合物 Tg 受多个尺度的因素影响:
- **原子级**: 官能团类型、键角、二面角
- **单体级**: 重复单元结构、侧链长度
- **链级**: 分子量、链缠结、自由体积
- **宏观级**: 结晶度、共混组成

传统描述符通常只捕捉 1-2 个尺度，多尺度建模试图同时覆盖多个层次。

### 4.2 代表性方法

#### 4.2.1 Uni-Poly (npj Computational Materials, 2025)

- **论文**: "Uni-Poly: A Unified Multimodal Representation for Polymer Property Prediction"
- **发表**: npj Computational Materials, 2025
- **核心方法**:
  - **5 模态统一表示**: SMILES 文本 + 2D 分子图 + 3D 几何 + 分子指纹 + 文本描述
  - 每种模态捕捉不同尺度信息:
    - SMILES: 化学连接性
    - 2D Graph: 拓扑结构
    - 3D Geometry: 空间构象
    - Fingerprint: 子结构存在性
    - Text: 高级化学知识
  - Cross-modal attention 融合
- **性能**: 多个 benchmark SOTA
- **小数据集适用性**: ★★★☆☆ — 5 模态增加实现复杂度

#### 4.2.2 GeoALBEF ("Align before Fuse", 2025)

- **论文**: "GeoALBEF: Geometry-Aware Aligned Language-Bond-Embedding Fusion"
- **发表**: 2025
- **核心方法**:
  - "先对齐，再融合" (Align before Fuse) 架构
  - 分别编码 SMILES (语言) 和 3D 几何 (图)
  - 对齐阶段: 拉近同一分子不同模态表示
  - 融合阶段: cross-attention 深度融合
  - 灵感来自视觉-语言预训练 (ALBEF)
- **小数据集适用性**: ★★★☆☆ — 需要 3D 坐标

#### 4.2.3 Mol-TDL (Topological Deep Learning, arXiv 2024)

- **论文**: "Molecular Topological Deep Learning for Polymer Property Prediction"
- **发表**: arXiv, 2024
- **核心方法**:
  - 用 simplicial complexes 替代传统图
  - 图只有节点和边 (0-simplex, 1-simplex)
  - simplicial complex 还有三角面 (2-simplex)、四面体 (3-simplex)
  - 多尺度 filtration: 不同半径捕捉不同尺度
  - 自然地编码多体相互作用
- **理论优势**: 图无法表达的 3-body+ 相互作用
- **小数据集适用性**: ★★☆☆☆ — 前沿方法，实现复杂

#### 4.2.4 HAPPY Representation (Hierarchical Abstraction, 2024)

- **论文**: "HAPPY: Hierarchically Abstracted rePeat unit for polYmer property prediction"
- **发表**: 2024
- **核心方法**:
  - 层次化聚合物表示: 原子 → 官能团 → 重复单元 → 聚合物
  - 每个层级提取不同粒度特征
  - Topo-HAPPY: 加入拓扑描述符
  - 在极小数据集 (214 条) 上测试
- **性能**: R² = 0.86 (仅 214 条数据)
- **小数据集适用性**: ★★★★★ — 专为小数据集设计的层次表示

#### 4.2.5 PolyLLMem (LLM + Molecular Embedding, 2025)

- **论文**: "PolyLLMem: LLM-Enhanced Polymer Property Prediction"
- **发表**: 2025
- **核心方法**:
  - LLaMA-3 文本 embedding + Uni-Mol 3D embedding
  - 双通道: 语言理解 + 分子结构理解
  - LLM 提供化学常识和推理能力
  - Uni-Mol 提供精确 3D 结构信息
- **小数据集适用性**: ★★★★☆ — LLM 常识可弥补数据不足

### 4.3 多尺度方法小结

| 方法 | 年份 | 尺度覆盖 | 核心创新 | 小数据适用 |
|------|------|---------|----------|-----------|
| Uni-Poly | 2025 | 5 模态 | 统一多模态表示 | ★★★ |
| GeoALBEF | 2025 | 语言+3D | Align before Fuse | ★★★ |
| Mol-TDL | 2024 | 拓扑多尺度 | Simplicial complexes | ★★ |
| HAPPY | 2024 | 层次结构 | 层次化抽象 | ★★★★★ |
| PolyLLMem | 2025 | 语言+3D | LLM 常识增强 | ★★★★ |

**关键结论**: 对于 ~300 条数据的场景，HAPPY 的层次化表示最值得关注 — 它专门在 214 条数据上验证了 R²=0.86 的性能。多模态方法 (Uni-Poly, GeoALBEF) 虽然性能好但实现复杂度高。

---

## 5. 小数据集策略

### 5.1 问题背景

聚合物 Tg 数据集通常只有 200-1000 条数据（本项目 304 条），而深度学习模型通常需要 10,000+ 数据点。如何在小数据集上让深度学习发挥作用是核心挑战。

### 5.2 策略概览

#### 5.2.1 策略 A: Transfer Learning (迁移学习)

**Shotgun Transfer Learning (XenonPy, 2023-2024)**:
- **核心思路**: 在 XenonPy.MDL 中预训练 140,000+ 个模型，覆盖各种性质
- 通过"shotgun"策略，自动尝试所有预训练模型作为特征提取器
- 选择迁移效果最好的源模型
- **性能**: 在极小数据集 (10-50 条) 上也能工作
- **适用性**: ★★★★★

**Cross-Property Transfer Learning (Yamada et al., 2024)**:
- 在大型性质 A 的数据集上训练 → 迁移到小型性质 B 的数据集
- 研究发现: 迁移模型在约 69% 的 (源, 目标) 组合上优于从零训练
- Tg 作为源任务效果好 (因数据量大)
- Tg 作为目标任务也能受益 (从密度、折射率等迁移)
- **适用性**: ★★★★☆

#### 5.2.2 策略 B: Data Augmentation (数据增强)

**SMILES Enumeration (最有效)**:
- 同一分子有多个等价 SMILES 表示
- 例如: `c1ccccc1` = `C1=CC=CC=C1` = `c1cccc(c1)` 等
- 可将数据集扩大 10-100 倍
- Data-Aug GCN 通过此策略将 R² 从 0.88 提升到 0.97
- **适用性**: ★★★★★

**Fox 方程虚拟数据 (本项目已用)**:
- 从均聚物 Tg 生成共聚物虚拟 Tg
- 304 条均聚物 → 46,000+ 共聚物
- 两阶段训练: 预训练(虚拟) → 微调(真实)
- **适用性**: ★★★★★ (本项目已验证)

#### 5.2.3 策略 C: Pre-trained Representations (预训练表示)

**polyBERT Fingerprint + 传统 ML**:
- 用 polyBERT 生成 768 维 fingerprint
- 将 fingerprint 作为传统 ML (RF, GBR, ExtraTrees) 的输入
- 不训练任何神经网络，只用预训练 embedding
- **适用性**: ★★★★★ — 最低风险、最高性价比的深度学习利用方式

**PolyCL Contrastive Fingerprint**:
- 对比学习生成的 polymer representation
- 在无标签数据上预训练
- **适用性**: ★★★★★

#### 5.2.4 策略 D: Multi-task Learning (多任务学习)

- 同时预测 Tg + Tm + 密度 + 其他性质
- 共享底层表示，增加有效训练信号
- polyGNN 多任务: Tg RMSE 从 36.6K 降至 31.7K (提升 13.4%)
- **要求**: 需要多个性质的标注数据
- **适用性**: ★★★☆☆ — 需要额外标注

#### 5.2.5 策略 E: Few-Shot / Meta-Learning

- 在多个材料性质任务上 meta-train
- 在新性质上 few-shot 适应
- 目前在聚合物领域尚不成熟
- **适用性**: ★★☆☆☆ — 前沿探索阶段

### 5.3 小数据集策略推荐排序 (针对本项目 304 条)

| 优先级 | 策略 | 预期提升 | 实现难度 | 推荐度 |
|--------|------|---------|---------|--------|
| 1 | polyBERT fingerprint + ExtraTrees | R² +0.03-0.05 | 低 | ★★★★★ |
| 2 | SMILES enumeration 数据增强 | R² +0.05-0.10 | 中 | ★★★★★ |
| 3 | Cross-property transfer learning | R² +0.02-0.05 | 中 | ★★★★☆ |
| 4 | 对比学习预训练 (PolyCL) | R² +0.03-0.05 | 中高 | ★★★★☆ |
| 5 | 多任务学习 | R² +0.02-0.04 | 高 | ★★★☆☆ |

**核心建议**: 对于 304 条数据，**不要从零训练深度学习模型**。最有效的策略是用深度学习生成的 embedding (polyBERT) 或增强数据 (SMILES enum) 来加强传统 ML 模型。

---

## 6. LLM 直接预测

### 6.1 现状评估

大语言模型 (LLM) 直接用于聚合物性质预测是 2024-2025 年的热门探索方向，但目前结果表明 LLM 在定量预测任务上仍不如专用模型。

### 6.2 Benchmark 结果

| 模型 | Tg RMSE (K) | 相对表现 |
|------|------------|---------|
| GPT-3.5 | 47.2 | 最差 |
| LLaMA-3 | 39.48 | 稍好 |
| GNN (from scratch) | 31-37 | 明显更好 |
| polyBERT + ML | ~28-32 | 更好 |
| Data-Aug GCN | 7.4 | 最佳 |

### 6.3 分析

- LLM 对化学 SMILES 的理解是隐式的，缺乏显式的化学知识编码
- 定量回归任务不是 LLM 的强项 (vs. 分类/排序)
- prompt engineering 可以部分改善，但无法弥补架构差距
- LLM 的价值更在于: 化学常识增强 (PolyLLMem)、inverse design (polyBART)、数据提取 (MatBERT)

**关键结论**: 不推荐直接用 LLM 做 Tg 回归预测。LLM 的价值在于提供辅助能力 (embedding, 常识, 数据提取)，而非替代专用预测模型。

---

## 7. 综合性能对比

### 7.1 全方法性能排名 (Tg 预测)

| 排名 | 方法 | 类型 | Tg 性能 | 数据需求 | 适合 ~300 条 |
|------|------|------|---------|---------|-------------|
| 1 | Data-Aug GCN | GNN+Aug | R²=0.97, RMSE=7.4K | ~500+aug | ✔ |
| 2 | Lieconv-Tg | 3D-NN | R²=0.90, MAE=24.4K | 7000+ | ✘ |
| 3 | PolymerGNN | GNN | R²=0.89 | 5000+ | ✘ |
| 4 | HAPPY | 层次ML | R²=0.86 (214条) | 200+ | ✔✔ |
| 5 | TransPolymer | Transformer | R²~0.85 | 预训练+微调 | ✔ |
| 6 | TransTg | Transformer | R²=0.849 | 1000+ | ✘ |
| 7 | polyBERT+ML | 预训练+ML | R²~0.82 | 300+ | ✔✔ |
| 8 | polyGNN (多任务) | GNN | RMSE=31.7K | 5000+ | ✘ |
| 9 | LLaMA-3 | LLM | RMSE=39.5K | N/A | ✘ |

### 7.2 方法选择决策树

```
数据量 < 300?
  ├─ YES → 传统 ML + polyBERT fingerprint + SMILES 增强
  └─ NO
        ├─ 300-1000 → Transfer Learning + 预训练微调 (TransPolymer)
        └─ 1000+ → GNN (PolymerGNN) 或 Transformer (从零训练可行)
```

---

## 8. 项目建议

### 8.1 针对本项目的深度学习整合方案

基于 304 条 Bicerano 均聚物数据 + 当前 ExtraTrees 基线 (R²=0.837)，推荐以下策略:

#### 策略 A: polyBERT Fingerprint 增强 (推荐优先实施)

```
现有物理特征 (34-dim) + polyBERT embedding (768-dim)
                    |
              特征降维 (PCA/UMAP 至 50-100 dim)
                    |
              ExtraTrees / Stacking
```

- **预期提升**: R² 0.84 → 0.87-0.89
- **实现难度**: 低 (pip install, 几行代码)
- **风险**: 低

#### 策略 B: SMILES Enumeration 数据增强

```
304 SMILES → SMILES 枚举 (每个 10-50 变体)
                    |
              3040-15200 条增强数据
                    |
              GCN 或 ExtraTrees 训练
```

- **预期提升**: R² 0.84 → 0.90-0.95 (参考 Data-Aug GCN)
- **实现难度**: 中 (RDKit 支持 SMILES 枚举)
- **风险**: 中 (需要正确处理数据泄露)

#### 策略 C: Transfer Learning (跨性质迁移)

```
外部大数据集 (22K) 预训练 → Bicerano (304) 微调
                    |
              或: 其他性质 (密度, Tm) 预训练 → Tg 微调
```

- **预期提升**: R² 0.84 → 0.88-0.90
- **实现难度**: 中
- **风险**: 中 (已在 Phase 3 部分验证)

#### 策略 D: PolyCL 对比学习表示

```
无标签聚合物 SMILES → 对比学习预训练
                    |
              PolyCL embedding + ExtraTrees
```

- **预期提升**: R² 0.84 → 0.87-0.89
- **实现难度**: 中高
- **风险**: 中

#### 策略 E: 全深度学习 (GCN + 数据增强)

```
SMILES → Graph → GCN (3-5层)
       + SMILES 枚举增强
       + 多任务 (Tg + Tm + 密度)
```

- **预期提升**: R² 0.84 → 0.92-0.97 (如果做对)
- **实现难度**: 高
- **风险**: 高 (需要 GPU, 调参)

### 8.2 推荐实施路线

1. **立即可做**: 策略 A (polyBERT fingerprint) — 1-2 天
2. **短期**: 策略 B (SMILES 枚举) — 3-5 天
3. **中期**: 策略 C (迁移学习改进) — 已有基础
4. **远期/论文**: 策略 E (端到端 GCN) — 作为论文亮点

### 8.3 与现有工作的结合

本项目已完成的工作为深度学习整合提供了坚实基础:

- **物理特征 (L0+L1H, 34-dim)**: 可与 polyBERT embedding 拼接
- **Fox 虚拟数据 (46K)**: 可用于 GNN 预训练
- **外部数据集 (22K)**: 迁移学习的理想源数据
- **H-bond 特征 (15-dim)**: 提供领域知识增强

---

## 9. 参考文献

### GNN 方法
1. polyGNN - "Polymer Graph Neural Networks for Multitask Property Learning", J. Phys. Chem. Lett., 2023
2. PolymerGNN - "Benchmarking GNNs for Polymer Informatics", J. Chem. Inf. Model., 2023
3. Data-Augmented GCN - "Predicting Tg Using GCN with Data Augmentation", ACS AMI, 2023
4. Lieconv-Tg - "Equivariant Neural Networks for Tg Prediction", 2024
5. MSRGCN-RL - "Multi-Scale Relational GCN with RL", Langmuir, 2024
6. GATBoost - "GAT Substructure Features + Gradient Boosting", 2024
7. Self-Supervised GNN - "Self-Supervised Pre-training for Polymer GNN", 2024

### Transformer / 序列模型
8. TransPolymer - "A Transformer-based Language Model for Polymer Property Predictions", npj Comp. Mat., 2023
9. TransTg - "SMILES-to-Tg Transformer", 2024
10. Multimodal Transformer - "Multimodal Transformer for Polymer Property Prediction", ACS AMI, 2024
11. polyBART - "A Bidirectional Polymer Language Model", EMNLP, 2025

### 预训练模型
12. polyBERT - "A Chemical Language Model for Fully Machine-Driven Polymer Informatics", Nature Communications, 2023
13. ChemBERTa - "Large-Scale Self-Supervised Pretraining for Molecular Property Prediction", 2020 (updated 2024)
14. PolyCL - "Contrastive Learning for Polymer Representation", Digital Discovery, 2025
15. MMPolymer - "Multimodal Multitask Pretraining for Polymer Property Prediction", 2024
16. MatBERT - "Materials Science Language Model", 2021 (applications to 2024)

### 多尺度建模
17. Uni-Poly - "Unified Multimodal Representation for Polymer Property Prediction", npj Comp. Mat., 2025
18. GeoALBEF - "Geometry-Aware Aligned Language-Bond-Embedding Fusion", 2025
19. Mol-TDL - "Molecular Topological Deep Learning", arXiv, 2024
20. HAPPY - "Hierarchically Abstracted rePeat unit for polYmer property prediction", 2024
21. PolyLLMem - "LLM-Enhanced Polymer Property Prediction", 2025

### 小数据集策略
22. Shotgun Transfer Learning - "Transfer Learning in XenonPy", 2023-2024
23. Cross-Property TL - "Cross-Property Transfer Learning for Polymer Properties", 2024

### LLM 方法
24. LLM Benchmark - "Benchmarking LLMs for Polymer Property Prediction", 2024-2025
25. Afsordeh et al. - "Predicting Tg from 4 Physical Features", 2025 (R²=0.97, 传统 ML 基线)

---

> **调研总结**: 对于本项目 (304 条数据, ExtraTrees 基线 R²=0.837)，最推荐的深度学习整合方案是 **polyBERT fingerprint + 传统 ML**，其次是 **SMILES enumeration 数据增强**。不推荐从零训练大型深度学习模型。深度学习的最大贡献不在于替代传统 ML，而在于提供更好的特征表示和数据增强策略。

