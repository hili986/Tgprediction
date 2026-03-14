# Tg 预测项目 学习指南

> 最后更新：2026-03-12 | 当前里程碑：Phase 3: 桥梁聚合物特征迁移学习完成
> 本指南面向初学者，帮助你理解项目涉及的基础知识、技术实现和学习路径。

## 1. 项目概览

### 1.1 这个项目是什么

本项目是同济大学 SITP（大学生创新训练计划）课题：**AI 辅助高分子材料设计**。核心目标是用机器学习预测聚合物的**玻璃化转变温度 (Tg)**——一个决定聚合物是"硬塑料"还是"软橡胶"的关键温度。

**为什么重要？** Tg 决定了材料的使用温度范围。例如，聚苯乙烯 (PS) 的 Tg≈100°C，常温下是硬的；天然橡胶的 Tg≈-70°C，常温下是软的。传统测量 Tg 需要合成样品 + DSC 实验（成本高、周期长），而 ML 预测可以在秒级完成。

**终极挑战**：从通用聚合物 Tg 预测（数据充足）迁移到核酸分子 Tg 预测（数据极度稀缺，<10 条）。

### 1.2 完成后你将学到什么

- 能够**解释**玻璃化转变温度的物理意义及其分子结构决定因素
- 能够**实现**基于 scikit-learn 的回归预测流水线（特征工程 → 模型训练 → 评估）
- 能够**分析** SHAP 值来解释模型预测的原因
- 能够**设计**迁移学习策略解决小数据问题
- 能够**评价**不同模型和特征组合的优劣，做出数据驱动的决策
- 能够**应用** Nested CV 进行无偏的模型评估

### 1.3 适合谁阅读本指南

- 本科 2-3 年级理工科学生
- 了解 Python 基本语法（列表、函数、类）
- 有基础的数学知识（线性代数、概率统计入门）
- 对机器学习有好奇心，但不需要有实操经验

---

## 2. 前置知识

### 2.1 必需知识

| 知识领域 | 具体内容 | 学习资源 | 预计时间 |
|---------|---------|---------|---------|
| Python 基础 | 变量、函数、列表推导、文件 I/O | [Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/) | 10h |
| NumPy 基础 | 数组操作、广播、矩阵运算 | [NumPy 官方入门](https://numpy.org/doc/stable/user/quickstart.html) | 3h |
| 高中化学 | 化学键、分子结构、官能团 | 高中化学课本 | 复习 2h |

### 2.2 推荐知识

| 知识领域 | 具体内容 | 学习资源 | 预计时间 |
|---------|---------|---------|---------|
| Pandas 基础 | DataFrame 操作 | [Pandas 10 分钟入门](https://pandas.pydata.org/docs/user_guide/10min.html) | 2h |
| 线性回归原理 | 损失函数、梯度下降 | [3Blue1Brown 线性代数](https://www.3blue1brown.com/topics/linear-algebra) | 4h |
| 高分子化学入门 | 聚合物结构、聚合反应 | 任意高分子化学教材第 1-2 章 | 4h |

### 2.3 自测清单

在开始学习前，检查你是否准备好了：

- [ ] 我能用 Python 写一个读取 CSV 文件并计算列平均值的脚本
- [ ] 我知道 `numpy.array` 和 Python `list` 的区别
- [ ] 我能解释什么是"函数"（输入 → 处理 → 输出）
- [ ] 我知道 C-C 单键可以自由旋转，而 C=C 双键不能
- [ ] 我听说过"回归"这个词（即使不完全理解）
- [ ] 我能用命令行运行 `python script.py`

如果 4/6 以上打勾，你已经准备好了！

---

## 3. 学习路线图

### 3.1 总览

```
Phase 0: 环境搭建 + 基础概念
    ↓
Phase 1: 描述符增强 (特征工程 + 模型评估)     ← 核心 ML 技能
    ↓
Phase 2: Fox 共聚物扩展 (数据增强 + 数据泄露)  ← 数据工程思维
    ↓
Phase 3: 桥梁聚合物迁移 (迁移学习 + SHAP)     ← 高级 ML 策略  ✅ 当前
    ↓
Phase 4: 整合 + 论文 (消融实验 + 科学写作)     ← 待完成
```

### 3.2 里程碑列表

| 里程碑 | 学习目标 | 前置依赖 | 预计耗时 | 难度 | 状态 |
|--------|---------|---------|---------|------|------|
| M0 | 理解 Tg 和项目目标 | 无 | 1h | ⭐ | ✅ |
| M1 | 掌握分子描述符和特征工程 | M0 | 4h | ⭐⭐ | ✅ |
| M2 | 理解 Nested CV 和模型评估 | M1 | 3h | ⭐⭐⭐ | ✅ |
| M3 | 理解数据增强和数据泄露 | M2 | 3h | ⭐⭐ | ✅ |
| M4 | 掌握迁移学习和 SHAP | M3 | 4h | ⭐⭐⭐ | ✅ |
| M5 | 消融实验和论文写作 | M4 | 5h | ⭐⭐⭐⭐ | 🔲 |

---

## 4. 基础知识速览

### 4.1 玻璃化转变温度 (Tg)

Tg 是聚合物从"玻璃态"（硬、脆）转变为"橡胶态"（软、弹性）的温度。本质上，Tg 反映的是**链段运动有多困难**——分子链越刚性、分子间作用力越强，Tg 越高。

例子：聚苯乙烯有苯环侧基（增加刚性），Tg≈100°C；聚二甲基硅氧烷骨架极柔软，Tg≈-125°C。

### 4.2 SMILES 分子表示

SMILES (Simplified Molecular Input Line Entry System) 是用字符串表示分子结构的标准方法。例如 `c1ccccc1` 表示苯环，`CC(=O)O` 表示乙酸。ML 模型不能直接"看"分子，需要通过 SMILES 计算数值描述符。

### 4.3 分子描述符

描述符是从分子结构计算得到的数值特征。例如：
- **NumRotatableBonds**（可旋转键数）：直接反映链柔性，与 Tg 负相关（r=-0.87~-0.90）
- **TPSA**（拓扑极性表面积）：反映分子极性，与氢键能力相关
- **RingCount**（环数）：环结构增加刚性，提高 Tg

### 4.4 交叉验证 (CV)

用全部数据训练模型再评估 = 考试前偷看答案。CV 把数据分成 k 折，每次留一折做测试。5-fold CV 意味着模型被训练和测试 5 次，取平均性能。

### 4.5 R² 和 MAE

- **R²（决定系数）**：模型解释了目标变量方差的百分比。R²=0.85 意味着模型解释了 85% 的 Tg 变化规律。1.0 = 完美，0 = 和猜平均值一样差。
- **MAE（平均绝对误差）**：预测值和真实值之间的平均偏差。MAE=33K 意味着预测平均偏差 33 开尔文。

### 4.6 迁移学习

当目标领域数据极少时（如核酸 Tg <10 条），借用相关领域的知识。本项目的策略：用"桥梁聚合物"（与核酸共享氢键模式的合成聚合物，205 条）的数据辅助训练。

### 4.7 SHAP 值

SHAP (SHapley Additive exPlanations) 基于博弈论的 Shapley 值，量化每个特征对单个预测的贡献。例如："这个分子的 RingCount=3 使 Tg 预测提高了 45K"。它让 ML 模型的"黑箱"变得可解释。

---

## 5. 里程碑学习单元

### Milestone 0: 环境搭建 + 项目理解

**学习目标**：完成后你能够搭建 Python ML 环境，理解项目目标和数据集。

**概念讲解（30%）**：

Tg 的物理意义可以用一个类比理解：想象一碗意大利面。冷的时候面条纠缠在一起，很难移动（玻璃态）；加热后面条变软，可以自由滑动（橡胶态）。Tg 就是这个"面条开始能动"的温度。

本项目的数据来自 **Bicerano 数据集**：304 条均聚物的 SMILES 和实验 Tg 值。均聚物是只由一种单体重复构成的聚合物（如聚乙烯 = CH2CH2 重复）。

**动手实践（70%）**：

- Step 1: 安装依赖
  ```bash
  pip install -r requirements.txt
  ```
- Step 2: 加载数据集，查看结构
  ```python
  from src.data.bicerano_tg_dataset import load_bicerano_data
  smiles_list, tg_values = load_bicerano_data()
  print(f"数据集大小: {len(smiles_list)} 条")
  print(f"Tg 范围: {min(tg_values):.0f}K ~ {max(tg_values):.0f}K")
  print(f"示例 SMILES: {smiles_list[0]}")
  ```
- Step 3: 运行全部测试确认环境正常
  ```bash
  python -m unittest discover tests/ -v
  ```
- ✅ 验证：329 个测试全部通过，0 failures

**知识小结**：Tg 是聚合物的核心热力学参数；SMILES 是 ML 预测的输入；304 条数据是一个典型的"小数据集"挑战。

---

### Milestone 1: 描述符增强 — 特征工程的力量（Phase 1）

**学习目标**：完成后你能够解释分子描述符的物理意义，独立构建特征工程流水线。

**概念讲解（30%）**：

特征工程是 ML 中最关键的步骤之一。本项目采用**分层特征策略**：

| 层级 | 描述符 | 维度 | 来源 | 物理意义 |
|------|--------|------|------|---------|
| L0 | Afsordeh 4 特征 | 4 | 论文复现 | 柔性、位阻、极性、氢键 |
| L1 | RDKit 2D 描述符 | 15 | RDKit 计算 | 拓扑、电子、形状 |
| L2 | 全描述符 + Morgan | ~1068 | RDKit + 指纹 | 全面但冗余 |

一个关键发现：**4 个好特征 > 2000 个随机特征**。Afsordeh & Shirali (2025) 证明只用 FlexibilityIndex, SOL, HBondDensity, PolarityIndex 就能达到 R²=0.97（在他们的数据集上）。这说明**特征质量远比特征数量重要**。

**动手实践（70%）**：

- Step 1: 理解 L0 Afsordeh 特征
  ```python
  from src.features.afsordeh_features import compute_afsordeh_features
  # FlexibilityIndex = 可旋转键 / 总键数（归一化的链柔性）
  # SOL = 侧链空间占据长度（位阻效应）
  # HBondDensity = 氢键供体+受体 / 重原子数（分子间作用力）
  # PolarityIndex = 极性键的 Fajan 极性加权和
  ```
- Step 2: 查看特征流水线
  ```python
  from src.features.feature_pipeline import build_dataset_v2
  X, y, smiles, names = build_dataset_v2(layer="L1", verbose=True)
  print(f"特征矩阵: {X.shape}")  # (304, 19)
  print(f"特征名: {names}")
  ```
- Step 3: 运行 Phase 1 基线实验
  ```bash
  # 查看实验结果
  python -c "import json; print(json.dumps(json.load(open('results/phase1/exp-1.2-L1.json')), indent=2))"
  ```
- ✅ 验证：L0 only R²≈0.76, L0+L1 R²≈0.81, L2 selected R²≈0.83

**知识小结**：
- NumRotatableBonds 是 Tg 的头号预测因子（6/6 SHAP 研究一致）
- 分层特征策略让你从简单到复杂逐步添加描述符
- 特征选择（Boruta, mRMR）帮助从 1000+ 维中筛选出 30-50 个有效特征

**常见问题**：
- Q: 为什么不直接用全部 1000+ 个描述符？
  A: 维度灾难。304 条数据 + 1000 维特征 = 过拟合。特征选择是必须的。
- Q: RDKit 是什么？
  A: 开源化学信息学库，可以从 SMILES 计算分子描述符。本项目用它计算 L1/L2 特征。

---

### Milestone 2: 模型评估 — Nested CV 的必要性（Phase 1）

**学习目标**：完成后你能够解释普通 CV 的偏差问题，独立实现 Nested CV 评估。

**概念讲解（30%）**：

**普通 k-fold CV 的隐患**：用同一套数据既选超参又评估性能 = 用模拟卷选考试策略，再用同一套模拟卷评分。结果会高估 0.03-0.05 的 R²（304 样本下不可忽略）。

**Nested CV** 用两层循环分离两个目标：
- 外循环 (outer=RepeatedKFold(5,3))：评估泛化性能
- 内循环 (inner=KFold(3))：选择最佳超参数

这样每次评估都是在"从未见过的"数据上进行，更接近真实部署表现。

**Extra Trees 为何适合小数据集**：ET 在分裂点选择上引入双重随机化（特征随机 + 阈值随机），这种"更笨"的策略反而是天然的正则化手段，减少对训练噪声的过拟合。

**动手实践（70%）**：

- Step 1: 理解评估模块
  ```python
  from src.ml.evaluation import nested_cv_evaluate
  # outer = RepeatedKFold(n_splits=5, n_repeats=3) → 15 次外折
  # inner = KFold(n_splits=3) → 每次外折内部做 3-fold 选参
  # 总计: 15 * 3 = 45 次模型训练
  ```
- Step 2: 查看 Phase 1 调参结果
  ```bash
  python -c "import json; d=json.load(open('results/phase1/exp-1.5a-L1-tuned.json')); print(f'R2={d[\"R2_mean\"]:.4f}, MAE={d[\"MAE_mean\"]:.1f}')"
  ```
- ✅ 验证：Nested CV 的 R² 比普通 CV 低 0.01-0.03（更保守但更可靠）

**知识小结**：
- Nested CV 是小数据集上最可靠的评估方法
- Extra Trees 的双重随机化是一种隐式正则化
- 论文中报告 Nested CV 结果比普通 CV 更有说服力

---

### Milestone 3: 数据增强 — Fox 方程与数据泄露（Phase 2）

**学习目标**：完成后你能够解释 Fox 方程的原理、识别数据泄露问题、理解虚拟数据的局限性。

**概念讲解（30%）**：

**Fox 方程**：预测共聚物 Tg 的经典经验公式。
```
1/Tg_copolymer = w1/Tg1 + w2/Tg2
```
其中 w1, w2 是两种单体的质量分数，Tg1, Tg2 是对应均聚物的 Tg。

用 304 条均聚物可以生成 C(304,2) × 多个组分比 ≈ 数万条虚拟共聚物数据。但有一个陷阱：

**数据泄露**：虚拟共聚物的 Tg 完全由均聚物 Tg 决定。如果 CV 测试集中的均聚物参与了虚拟样本的生成，模型就能"偷看"测试答案。修复方法：每个 CV fold 排除包含测试集均聚物的虚拟样本。

**关键发现**：修复泄露后，虚拟数据仅带来 +1% R² 的边际改善（R²=0.95+ 完全是泄露的假象）。根本原因：Fox 方程不引入新的结构信息。

**动手实践（70%）**：

- Step 1: 查看 Fox 数据生成器
  ```python
  from src.data.fox_copolymer_generator import FoxCopolymerGenerator
  gen = FoxCopolymerGenerator()
  # gen.generate(n_samples=10000) 生成 10K 条虚拟共聚物
  ```
- Step 2: 对比泄露 vs 无泄露结果
  ```bash
  # 查看 Phase 2 总结
  python -c "
  import json
  d = json.load(open('results/phase2/exp-2.2b-leakage-free.json'))
  print('Leakage-free results:')
  for cfg in d.get('results', d.get('experiments', []))[:3]:
      print(f'  {cfg}')
  "
  ```
- ✅ 验证：泄露版 R²=0.95，无泄露版 R²=0.84 — 差距 0.11 全是泄露

**知识小结**：
- Fox 方程是聚合物科学的经典工具，但生成的数据没有新信息
- 数据泄露是 ML 实验中最危险的错误之一，可能导致完全虚假的好结果
- 数据增强的价值取决于是否引入了**新的结构-性质关系**

**常见问题**：
- Q: 既然虚拟数据效果有限，为什么还要做 Phase 2？
  A: 负面结论也有价值——证明了"更多数据"不等于"更好数据"，为论文提供了重要的对照实验。

---

### Milestone 4: 迁移学习 — 桥梁聚合物与核酸预测（Phase 3）

**学习目标**：完成后你能够设计特征迁移策略、解释 SHAP 分析结果、评估迁移学习的效果。

**概念讲解（30%）**：

**核心困境**：核酸 Tg 数据极少（ATP=246K, ADP=244K, AMP=249K, GMP=260K，仅 4 条可靠数据）。直接训练不可能。

**桥梁聚合物策略**：找到与核酸**共享氢键模式**的合成聚合物（如聚酰胺共享 N-H...O=C 模式，聚磷酸酯共享 P-O 骨架），用它们的数据作为"桥梁"辅助训练。

关键假设：**相似氢键模式 → 相似 Tg 贡献规律**。

**H-bond SMARTS 特征**（15 维）：用 SMARTS 模式匹配分子中的氢键结构（amide, urea, urethane 等 10 种模式），加上 4 个密度指标和 1 个 CED 加权和。这 15 个特征让模型"看到"核酸与桥梁聚合物之间的共性。

**加权训练**：合并 Bicerano (304条, weight=1.0) + 桥梁 (205条, weight=0.4~0.8)。权重越高，模型越"关注"桥梁数据中的氢键模式。

**SHAP 分析结果**（本项目验证）：
- RingCount 是头号预测因子（SHAP=45.17），是第二名的 2.5 倍
- H-bond 特征对 Bicerano 贡献小（4.6%），但对核酸迁移至关重要
- GBR 和 ET 的 SHAP 排序不完全一致（不同模型"看问题角度"不同）

**Stacking 失败分析**：ET+GBR+SVR→Ridge 集成在 304 样本上表现为 R²=-0.45（远差于单模型）。原因：(1) 样本太少，内层 CV 后基学习器过拟合；(2) SVR 不支持 sample_weight 透传。教训：<500 样本时 Stacking 通常不如单模型。

**动手实践（70%）**：

- Step 1: 理解桥梁数据集
  ```python
  from src.data.bridge_polymers import build_bridge_dataset
  X_br, y_br, smiles_br, names = build_bridge_dataset(
      layer="L1", include_hbond=True, verbose=True
  )
  print(f"桥梁数据: {X_br.shape[0]} 条, {X_br.shape[1]} 维特征")
  # 预期输出: 205 条, 34 维 (19 L1 + 15 hbond)
  ```
- Step 2: 查看迁移学习结果
  ```bash
  python -c "
  import json
  d = json.load(open('results/phase3/exp-3.3-transfer-learning.json'))
  for k, v in d.items():
      if 'R2' in str(v):
          print(f'{k}: {v}')
  "
  ```
- Step 3: 查看 SHAP 分析
  ```bash
  python -c "
  import json
  d = json.load(open('results/phase3/exp-3.5-shap-analysis.json'))
  print('GBR Top-5 SHAP features:')
  for feat in d['gbr_shap_top15'][:5]:
      print(f'  {feat[\"feature\"]}: {feat[\"mean_abs_shap\"]:.2f}')
  "
  ```
- Step 4: 查看核酸预测效果
  ```bash
  # 查看 Phase 3 总结
  cat results/phase3/phase3-summary.md | head -115 | tail -20
  ```
- ✅ 验证：核酸平均误差从 229K → 3.9K（98.3% 提升），ATP 误差仅 1.0K

**知识小结**：
- 迁移学习的关键是找到源域和目标域之间的"桥梁"
- bridge_weight 控制源-目标权衡：0.4 最优 CV，0.8 最优核酸预测
- SHAP 让 ML 模型可解释，RingCount 是 Tg 的头号预测因子
- 负面结论（Stacking 失败）同样有科学价值

**常见问题**：
- Q: 为什么 bridge_weight=0.8 核酸更好，但 CV R² 反而略低？
  A: 经典的源-目标权衡。权重越高，模型越偏向桥梁数据的氢键模式（对核酸有利），但在通用聚合物上泛化略差。最终选择取决于使用场景。
- Q: DNA backbone 为什么预测不准（误差 169K）？
  A: 单体级 SMILES 无法捕捉 DNA 的高级结构（双螺旋、counterion 效应、链间氢键网络）。这是方法论的根本局限。
- Q: SHAP 和 feature_importances_ 有什么区别？
  A: `feature_importances_` 基于 impurity reduction（不纯度减少），偏向高基数特征；SHAP 基于 Shapley 值（博弈论），理论上更公平可靠。

---

## 6. 常见问题与排错

### 6.1 环境问题

- **Q: `ModuleNotFoundError: No module named 'rdkit'`**
  A: 安装 RDKit：`pip install rdkit` 或 `conda install -c conda-forge rdkit`

- **Q: `ImportError: No module named 'shap'`**
  A: `pip install shap`

- **Q: 运行测试时大量 SMILES 解析警告**
  A: 正常现象。RDKit 对某些 SMILES 会输出警告但不影响结果。

### 6.2 运行时错误

- **Q: SHAP 分析非常慢**
  A: TreeExplainer 对 ET (500 棵树) 需要一定时间。304 样本下通常 1-5 分钟完成。

- **Q: Nested CV 运行时间很长**
  A: 预期行为。5×3 outer + 3 inner + 20 iter RandomizedSearch = 大量模型训练。耐心等待或减少 `n_iter`。

### 6.3 结果不符预期

- **Q: 我的 R² 比报告的低**
  A: 检查 `random_state` 是否一致。不同随机种子会导致 ±0.01-0.02 的波动。

- **Q: 核酸预测误差很大**
  A: 确认使用了 bridge_weight=0.8 和 tuned 超参数。权重选择对核酸预测影响巨大。

---

## 7. 进阶方向

### 7.1 可深入探索的方向

1. **图神经网络 (GNN)**：直接从分子图学习表示，无需手工特征。参考 polyGNN (Kuenneth 2023)
2. **多保真度学习**：用 Delta Learning 融合不同精度的数据源（Fox → MD → 实验）
3. **主动学习**：让模型自动选择最有价值的新实验来做
4. **Transformer 预训练**：在大规模聚合物数据上预训练，然后微调到核酸领域

### 7.2 相关项目推荐

- [PolyMetriX](https://polymetrix.org)：7367 条聚合物 Tg 数据集 + 分层描述符
- [TransPolymer](https://github.com/ChangwenXu98/TransPolymer)：基于 Transformer 的聚合物性质预测
- [Chemprop](https://github.com/chemprop/chemprop)：消息传递神经网络的分子性质预测

### 7.3 学术延伸阅读

- Xu et al. (2024) "Machine learning analysis of a large set of homopolymers to predict glass transition temperatures" — *Communications Chemistry*
- Afsordeh & Shirali (2025) "Machine Learning-assisted Prediction of Polymer Glass Transition Temperature: A Structural Feature Approach" — *Chinese J. Polymer Sci.*
- Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" — NeurIPS（SHAP 原始论文）

---

## 8. 参考资源汇总

| 资源名称 | 类型 | 难度 | 必读/选读 | 链接 |
|---------|------|------|----------|------|
| scikit-learn 官方教程 | 文档 | ⭐⭐ | 必读 | [链接](https://scikit-learn.org/stable/tutorial/basic/tutorial.html) |
| scikit-learn 完整指南 2026 | 教程 | ⭐⭐ | 选读 | [链接](https://nerdleveltech.com/mastering-scikit-learn-a-complete-2026-tutorial-for-machine-learning) |
| DataCamp scikit-learn 教程 | 教程 | ⭐ | 选读 | [链接](https://www.datacamp.com/tutorial/machine-learning-python) |
| SHAP 官方入门 | 文档 | ⭐⭐⭐ | 必读 | [链接](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) |
| DataCamp SHAP 教程 | 教程 | ⭐⭐ | 选读 | [链接](https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability) |
| Interpretable ML Book — SHAP 章节 | 书籍 | ⭐⭐⭐ | 必读 | [链接](https://christophm.github.io/interpretable-ml-book/shap.html) |
| RDKit 入门 | 文档 | ⭐⭐ | 必读 | [链接](https://www.rdkit.org/docs/GettingStartedInPython.html) |
| NumPy 快速入门 | 文档 | ⭐ | 必读 | [链接](https://numpy.org/doc/stable/user/quickstart.html) |
| Geurts et al. (2006) Extra Trees 原文 | 论文 | ⭐⭐⭐⭐ | 选读 | ML 63(1):3-42 |
| Cawley & Talbot (2010) 模型选择过拟合 | 论文 | ⭐⭐⭐ | 选读 | JMLR 11:2079-2107 |
| Pan & Yang (2010) 迁移学习综述 | 论文 | ⭐⭐⭐ | 选读 | IEEE TKDE 22(10) |
| Wolpert (1992) Stacked Generalization | 论文 | ⭐⭐⭐⭐ | 选读 | Neural Networks 5(2):241-259 |

---

## 9. 术语表

| 术语 | 解释 | 首次出现 |
|------|------|---------|
| Tg | 玻璃化转变温度，聚合物从硬变软的临界温度 | M0 |
| SMILES | 用字符串表示分子结构的标准方法 | M0 |
| 描述符 (Descriptor) | 从分子结构计算的数值特征 | M1 |
| RDKit | 开源化学信息学 Python 库 | M1 |
| R² | 决定系数，衡量模型解释力（0~1） | M1 |
| MAE | 平均绝对误差，预测偏差的平均值 | M1 |
| CV | 交叉验证，防止过拟合的评估方法 | M2 |
| Nested CV | 嵌套交叉验证，分离调参和评估 | M2 |
| Extra Trees (ET) | 极端随机树，一种集成学习方法 | M2 |
| GBR | 梯度提升回归，另一种集成学习方法 | M2 |
| Fox 方程 | 预测共聚物 Tg 的经验公式 | M3 |
| 数据泄露 (Data Leakage) | 训练时无意使用了测试信息 | M3 |
| 迁移学习 (Transfer Learning) | 借用相关领域的知识辅助目标任务 | M4 |
| 桥梁聚合物 | 与核酸共享氢键模式的合成聚合物 | M4 |
| SHAP | 基于 Shapley 值的模型解释方法 | M4 |
| SMARTS | 分子子结构匹配的模式语言 | M4 |
| H-bond | 氢键，分子间的一种非共价作用力 | M4 |
| CED | 内聚能密度，衡量分子间作用力强度 | M4 |
| Stacking | 多模型集成方法，用元学习器融合基学习器 | M4 |
| bridge_weight | 桥梁数据的训练权重，控制迁移强度 | M4 |
| 消融实验 (Ablation) | 逐一移除组件观察影响 | M5 |
