# 寒假项目 — BigSMILES 工具链技术报告

> 为升级答辩准备 | 2026-04-08

## 项目全景

**一句话**：用 Python 给高分子世界造了一套"编译器 + 机器学习预测器"。

**总代码**：~5,950 行核心代码（10 个模块）+ ~2,500 行测试（9 个测试文件，415+ 测试用例）

**技术栈**：纯 Python 3.10+，标准库优先，RDKit 可选（没有 RDKit 也能跑大部分功能）

---

## 模块依赖关系图

```
bigsmiles_checker.py (基础层 — 语法检查)
    │
    ├──→ bigsmiles_parser.py (解析+生成+拓扑分析)
    │       │
    │       ├──→ bigsmiles_fingerprint.py (分子指纹+特征提取)
    │       │       │
    │       │       ├──→ ml_models.py (7种纯Python ML模型)
    │       │       │       │
    │       │       │       └──→ ml_experiment.py (实验框架)
    │       │       │
    │       │       └──→ web_demo.py (Web演示，调用全部模块)
    │       │
    │       └──→ bicerano_tg_dataset.py (304聚合物数据集)
    │
    └──→ bigsmiles_annotation.py (属性标注扩展)

sequence_to_bigsmiles.py (独立模块 — 核酸序列转换)
helm_to_3d.py (独立模块 — HELM转3D结构)
```

---

## 模块 1：bigsmiles_checker.py（858 行）— 语法检查器

### 它做什么

输入一个 BigSMILES 字符串，告诉你它是不是合法的。如果不合法，精确定位错误在哪里。

### 怎么做的 — 三阶段流水线

这和编译器的经典架构完全一样：

```
Stage 1: 分词 (Tokenize)     "把文本切成一个个有意义的片段"
    ↓
Stage 2: 解析 (Parse)         "把片段组装成语法树"
    ↓
Stage 3: 语义校验 (Validate)  "检查意义上是否正确"
```

#### Stage 1 分词器

把 `{[$]CC[$]}` 这样的字符串切成 Token 序列：

```
{  →  OPEN_BRACE
[$] → BOND_DESCRIPTOR
C   → ATOM
C   → ATOM
[$] → BOND_DESCRIPTOR
}   → CLOSE_BRACE
```

一共定义了 16 种 Token 类型。关键细节：
- `,` 和 `;` 只在 `{}` 花括号内有特殊含义（上下文敏感分词）
- 两字符原子（Cl, Br）要优先识别，不能误切

#### Stage 2 递归下降解析器

把 Token 序列变成 AST（语法树）。用的是"递归下降"方法：
- 遇到 `{` → 进入随机对象解析
- 遇到 `[$]` → 识别为键描述符
- 遇到 `,` → 分隔重复单元
- 遇到 `;` → 分隔端基
- 如果 `{}` 里面又有 `{}`，递归处理（支持嵌套）

#### Stage 3 七项语义检查

| # | 检查项 | 具体检查什么 |
|---|--------|-------------|
| 1 | 括号匹配 | `{` `}` `(` `)` `[` `]` 是否正确配对（用栈实现） |
| 2 | 描述符语法 | `[$]` `[>]` `[<]` 格式是否正确 |
| 3 | 随机对象结构 | `{}` 内至少有 1 个非空重复单元 |
| 4 | 终端配对 | `[>1]` 必须和 `[<1]` 配对出现 |
| 5 | SMILES 有效性 | 把描述符替换为 `*` 后，用 RDKit 检查 SMILES 是否合法 |
| 6 | 描述符一致性 | 同一个 `{}` 内不能混用 `$` 和 `<>` |
| 7 | 最少描述符数 | 重复单元至少 2 个描述符，端基恰好 1 个 |

### 关键设计：错误消息是双语的（中英文），方便演示。

---

## 模块 2：bigsmiles_parser.py（347 行）— 解析器 + 生成器

### 它做什么

在 checker 的基础上提供更高级的操作：

1. **生成器（Generator）**：AST → BigSMILES 文本（round-trip，验证解析正确性）
2. **提取重复单元**：从 AST 中递归提取所有重复单元的 SMILES
3. **提取键描述符**：列出所有 `[$]` `[>]` `[<]`
4. **拓扑分类**：自动判断聚合物属于 7 种拓扑中的哪一种

### 拓扑分类的决策树

```
输入 BigSMILES AST
│
├─ 没有随机对象 {} → "small_molecule"（小分子，不是聚合物）
│
├─ 1 个随机对象 {}
│   ├─ 有嵌套（{}里还有{}） → "graft_copolymer"（接枝共聚物）
│   ├─ 描述符 > 2 个 → "branched"（支化聚合物）
│   ├─ 1 个重复单元 → "linear_homopolymer"（线型均聚物）
│   └─ >1 个重复单元 → "random_copolymer"（随机共聚物）
│
└─ ≥2 个随机对象
    ├─ 有嵌套 → "graft_block_copolymer"
    └─ 无嵌套 → "block_copolymer"（嵌段共聚物）
```

### 生成器核心逻辑

遍历 AST 的每个节点，按 BigSMILES 语法规则拼接字符串。关键是**按 position 字段排序**，确保嵌套对象恢复到原始位置。

如果 `parse(text) → AST → generate(AST) == text`，说明解析-生成的 round-trip 正确。

---

## 模块 3：bigsmiles_examples.py — 39 种聚合物示例库

覆盖 13 类聚合物结构：

| 类别 | 示例数 | 代表性聚合物 |
|------|--------|-------------|
| 线型均聚物 | 6 | 聚乙烯 `{[$]CC[$]}` |
| 随机共聚物 | 4 | SBR 橡胶 |
| 嵌段共聚物 | 3 | SBS 热塑弹性体 |
| 交替共聚物 | 2 | 尼龙-66 |
| 梯度共聚物 | 1 | — |
| 接枝共聚物 | 3 | ABS |
| 支化聚合物 | 2 | LDPE |
| 超支化聚合物 | 1 | — |
| 网络聚合物 | 2 | 环氧交联 |
| 环状聚合物 | 1 | 环状 PEG |
| 端基修饰 | 3 | PEG-二胺 |
| 星型聚合物 | 2 | 三臂星 PCL |
| 核酸高分子 | 3 | DNA `{[>]...[<]}` |

**作用**：既是语法检查器的测试数据，也是教学示例库。

---

## 模块 4：bicerano_tg_dataset.py（501 行）— 数据集

### 数据来源

Bicerano (2002) 经典教科书 + Choi et al. (2024) Figshare 更新

### 数据内容

- **304 种线型均聚物**
- 每条数据包含：名称 + 重复单元 SMILES + BigSMILES + Tg (K)
- Tg 范围：130K（PDMS，软如橡胶）～ 576K（PEEK，硬如塑料）

### 核心 API

```python
load_dataset()     # → 304 个字典列表
get_smiles()       # → SMILES 列表
get_tg_values()    # → Tg 值列表
validate_all()     # → 用 checker 校验全部 BigSMILES
to_csv() / to_json()  # → 导出
```

---

## 模块 5：bigsmiles_fingerprint.py（507 行）— 特征工程

### 它做什么

把分子结构转成机器学习能用的数字向量。三层特征：

### 第 1 层：Morgan 指纹（默认 2048 位）

```
原理：以每个原子为中心，向外扩展 radius 层邻域
      → 对每个子结构做哈希 → 映射到固定长度的 0/1 向量

类比：给分子拍"指纹照"，不同的局部结构留下不同的指纹
```

相当于 ECFP（Extended-Connectivity Fingerprint），化学信息学标准方法。

### 第 2 层：片段计数（14 维）

用 SMARTS 模式匹配 14 种功能团，计数它们在分子中出现了几次：

```
sp3 碳 (CX4)、sp2 碳 (CX3)、胺 (NX3)、醚氧 (OX2)、
羰基 (C=O)、羟基 (OH)、芳环……等 14 种
```

### 第 3 层：聚合物描述符（14 维）

14 个物理/化学特征：
- 分子量、非氢原子数、可旋转键数
- C/O/N 原子分数、杂原子比例
- 芳香比例、氢键供体/受体
- 极性表面积 (TPSA)、LogP
- 键合类型（AA 型 vs AB 型）

### 组合特征

```
combined_fingerprint = [Morgan 1024 位] + [14 片段] + [14 描述符]
                     = 1052 维向量
```

### 性能（Ridge 回归，Bicerano 304 条）

| 特征组合 | R² |
|---------|-----|
| 仅 Morgan | 0.52 |
| 仅片段计数 | 0.28 |
| 仅描述符 | 0.43 |
| 全部组合 | 0.58 |

**注意**：这是寒假项目里的初步结果。后来 Tg 预测项目里用更好的特征（VPD 等）和更好的模型，达到了 R²=0.896。

---

## 模块 6：ml_models.py（784 行）— 纯 Python 机器学习

### 特别之处

**完全不依赖 numpy/sklearn**，7 种回归模型全用 Python 标准库手写实现。

### 7 种模型

| 类型 | 模型 | 核心算法 |
|------|------|---------|
| 线性 | Ridge 回归 | 梯度下降 + L2 正则化 |
| 线性 | Lasso 回归 | 近端梯度下降 + L1 正则化 |
| 线性 | ElasticNet | L1 + L2 混合正则化 |
| 邻域 | KNN 回归 | 距离加权平均 |
| 树 | 决策树 | 递归分割，最小化 MSE |
| 集成 | 随机森林 | Bagging（多棵树投票） |
| 集成 | 梯度提升 (GBR) | Boosting（逐轮纠正残差） |

### GBR（最优模型）核心算法

```
1. 初始化：f₀(x) = mean(y)      "先猜一个平均值"
2. 第 m 轮：
   a. 计算残差 rₘ = y - fₘ₋₁(x)  "看看上一轮猜错了多少"
   b. 用决策树拟合残差 hₘ(x)       "训练一棵小树去纠正错误"
   c. 更新 fₘ = fₘ₋₁ + α·hₘ       "加上这棵树的修正（乘学习率）"
3. 最终预测 = 初始均值 + Σ(所有修正树)
```

### 评估指标

```python
R² = 1 - SS_res/SS_tot    # 范围(-∞, 1]，越接近1越好
MAE = mean(|真实 - 预测|)   # 平均绝对误差，单位是K
RMSE = sqrt(mean(误差²))    # 均方根误差
```

---

## 模块 7：ml_experiment.py（500 行）— 实验框架

### 三类实验

**实验 1：模型对比**（7 种模型 × 5 折交叉验证）
```
最优：GBR R²=0.85, MAE=31.9K
最差：KNN R²=0.61
```

**实验 2：特征消融**（验证每组特征的贡献）
```
Morgan 独占 52% 预测能力
全部组合出现轻微饱和
```

**实验 3：超参扫描**（Morgan 指纹的 bits × radius 网格搜索）
```
最优平衡：256 bits + radius=3
```

---

## 模块 8：sequence_to_bigsmiles.py（720 行）— 核酸序列转换

### 核心问题

如何用计算机表示 DNA/RNA 的分子结构？

### 解决方案：双表示策略

BigSMILES **不能区分序列顺序**（ACGT 和 TGCA 生成相同的 BigSMILES），所以需要两种表示互补：

| 表示 | 编码什么 | 用途 |
|------|---------|------|
| BigSMILES | 聚合物类别（"由 ATGC 四种核苷酸组成的 DNA"） | 分类 |
| Full SMILES | 精确原子结构（编码序列顺序） | 性质计算 |

### Full SMILES 拼接算法

```
输入：ACGT (DNA)

拼接过程：
  5'-OH  +  dA(含磷酸)  +  dC(含磷酸)  +  dG(含磷酸)  +  dT(无磷酸)
  起始     内部核苷酸     内部核苷酸     内部核苷酸     3'末端

输出：一个 ~170 字符的 SMILES 串，对应 ~80 个原子的完整分子
```

每个核苷酸是一个预定义的 SMILES 构建块（糖环 + 碱基 + 磷酸），逐个拼接。

### DNA vs RNA 的区别

```
DNA：糖环 2' 位是 CC（脱氧）     → 碱基：A T G C
RNA：糖环 2' 位是 C(O)C（含羟基） → 碱基：A U G C
```

---

## 模块 9：bigsmiles_annotation.py（436 行）— 属性标注

为 BigSMILES 字符串附加结构化属性：

```
{[$]CC[$]}|Tg=373K;Mn=50000;source=Bicerano2018|
```

定义了 15 种标准属性（Tg, Tm, Mn, Mw, PDI, 密度等），可以解析、合并、校验。

---

## 模块 10：web_demo.py（642 行）— Web 演示

纯标准库 HTTP 服务器，提供 5 个 API：

```
POST /api/check       → 语法检查
POST /api/parse       → 解析 + 拓扑分析
POST /api/fingerprint → 特征提取
POST /api/predict     → Tg 预测
POST /api/pipeline    → 端到端（上面全做）
```

自带单页面前端，输入 BigSMILES → 实时显示结果。

---

## 模块 11：helm_to_3d.py（655 行）— HELM 转 3D 结构

解析 HELM 核酸记法（如 `RNA1{[R](A)P.[R](C)P}$$$$`），转成 SMILES，用 RDKit 生成 3D SDF 文件。

与 `sequence_to_bigsmiles.py` 互补：前者输出 3D 结构文件，后者输出 2D 表示。

---

## 与 Tg 预测项目的关系

```
寒假项目                              Tg 预测项目
(BigSMILES 工具链)                    (本学期)
                                      
bigsmiles_checker  ──复用──→  src/bigsmiles/ (直接搬过来)
bicerano_dataset   ──扩展──→  src/data/ (304 → 7,486)
fingerprint        ──升级──→  src/features/ (1052d → 46d VPD → 186d 多尺度)
ml_models          ──替换──→  sklearn/CatBoost/TabPFN (工业级模型)
ml_experiment      ──升级──→  Nested CV + 消融 + SHAP (严格评估)
sequence_to_bigsmiles ─延伸─→  核酸 Tg 迁移预测
```

**寒假项目是地基，Tg 预测项目是在地基上盖的楼。**
