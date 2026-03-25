# 虚拟/合成数据增强 Tg 预测：方法、工具与实施路线调研报告

> 调研日期：2026-03-08
> 项目：同济大学 SITP — AI辅助高分子材料设计
> 目标：系统调研构建虚拟/合成聚合物数据库的方法，用于增强 Tg 预测模型训练数据

---

## 目录

1. [调研背景与动机](#1)
2. [方法总览与分类](#2)
3. [方法一：基团贡献法生成虚拟 Tg](#3)
4. [方法二：Fox/Gordon-Taylor 方程生成共聚物 Tg](#4)
5. [方法三：QSPR/QSPP 模型批量预测](#5)
6. [方法四：半经验量化计算生成描述符](#6)
7. [方法五：全原子 MD 模拟](#7)
8. [方法六：SMILES 数据增强](#8)
9. [方法七：生成模型合成虚拟聚合物](#9)
10. [方法八：基于规则的虚拟聚合物库构建](#10)
11. [方法九：多保真度迁移学习](#11)
12. [方法十：GAN/条件生成数据增强](#12)
13. [工具与平台汇总](#13)
14. [方法对比矩阵](#14)
15. [推荐实施路线](#15)
16. [参考文献](#16)

---

## 1. 调研背景与动机

### 1.1 问题定义

聚合物 Tg 预测面临的核心数据瓶颈：
- **PolyInfo 数据库**：约 18,000 条 Tg 记录，覆盖约 6,000 种均聚物
- **共聚物 Tg 数据**：极度稀缺（< 2,000 条），且组成-Tg 关系复杂
- **核酸共聚物**：几乎无公开 Tg 数据
- **实验数据增长缓慢**：每年新增数据有限，测试条件不统一

### 1.2 虚拟数据的价值

虚拟/合成数据可以：
1. **扩大训练集规模**：从数千扩展到数万乃至数百万
2. **填补化学空间空白**：覆盖实验未探索的聚合物结构
3. **提供多保真度信息**：低精度但大量的计算数据 + 高精度但稀少的实验数据
4. **降低模型过拟合风险**：更多样化的训练样本
5. **支持迁移学习**：低保真度数据预训练 + 高保真度数据微调

### 1.3 关键挑战

- 虚拟数据与实验数据之间存在**系统性偏差**
- 生成的聚合物结构需要**化学有效性**验证
- 计算成本与数据精度之间的**权衡**
- 多保真度数据的**有效融合**策略

---

## 2. 方法总览与分类

按计算成本和数据精度，虚拟数据生成方法可分为以下层级：

| 层级 | 方法 | 单样本耗时 | 精度 (MAE) | 可生成规模 |
|------|------|-----------|-----------|----------|
| F0 | 基团贡献法 (Van Krevelen/Bicerano) | ~ms | 30-50 K | 百万级 |
| F1 | Fox/Gordon-Taylor 方程 | ~ms | 10-30 K | 千万级 |
| F2 | QSPR/ML 模型预测 | ~s | 20-40 K | 百万级 |
| F3 | 半经验量化 (GFN2-xTB) + ML | ~min | 15-30 K | 十万级 |
| F4 | 全原子 MD (RadonPy) | 30-50 hr | 11-18 K | 千级 |
| F5 | 实验测量 | 天-周 | 基准 | 百级/年 |

**核心策略**：构建多保真度数据金字塔，底层大量低精度数据提供覆盖，顶层少量高精度数据提供校准。

---

## 3. 方法一：基团贡献法生成虚拟 Tg

### 3.1 原理

基团贡献法将聚合物性质分解为各化学基团贡献的加和：

Tg = sum(Yi * ni) / sum(Mi * ni)

其中 Yi 是第 i 个基团对 Tg 的贡献，ni 是基团数量，Mi 是基团分子量。

### 3.2 主要方法

**Van Krevelen 方法**：
- 约 40 种基团参数
- MAE: 30-50 K
- 适用于常见均聚物
- 参考：Van Krevelen, "Properties of Polymers" (4th ed., 2009)

**Bicerano 方法**：
- 基于连接性指数和拓扑描述符
- MAE: ~30 K（优于 Van Krevelen）
- 覆盖更广泛的聚合物类型
- 参考：Bicerano, "Prediction of Polymer Properties" (3rd ed., 2002)

### 3.3 虚拟数据生成流程

1. 枚举聚合物重复单元的基团组成
2. 查表计算每种组合的 Tg
3. 过滤化学上不合理的组合
4. 生成 (结构, Tg_calc) 数据对

### 3.4 优缺点

**优势**：
- 计算极快（ms 级），可生成百万级数据
- 实现简单，无需特殊软件
- 物理意义明确

**劣势**：
- 精度有限（MAE 30-50 K）
- 无法处理基团间相互作用（如氢键）
- 基团参数表有限，新型基团无参数
- 对共聚物适用性差

### 3.5 适用场景

- 作为多保真度学习的最低保真度层
- 均聚物 Tg 的粗略筛选
- 与实验数据差异较大时需要偏差校正

---

## 4. 方法二：Fox/Gordon-Taylor 方程生成共聚物 Tg

### 4.1 原理

**Fox 方程**（最简单）：

1/Tg = w1/Tg1 + w2/Tg2

其中 w1, w2 是质量分数，Tg1, Tg2 是均聚物的 Tg。

**Gordon-Taylor 方程**（更准确）：

Tg = (w1*Tg1 + K*w2*Tg2) / (w1 + K*w2)

其中 K 是拟合参数（通常 K = Tg1/Tg2 作为近似）。

### 4.2 虚拟数据生成潜力

这是**本项目最具价值的低成本方法之一**：

- PolyInfo 中约 6,000 种均聚物有 Tg 数据
- 任取 2 种均聚物组合：C(6000, 2) = 17,997,000 种共聚物
- 每种共聚物取 10 个组成比例（0.1, 0.2, ..., 0.9）
- 总计可生成 ~1.8 亿条虚拟共聚物 Tg 数据

### 4.3 实现方案

```python
import itertools
import numpy as np

homopolymer_tg = load_polyinfo_tg()  # dict: SMILES -> Tg
pairs = itertools.combinations(homopolymer_tg.keys(), 2)

for smiles1, smiles2 in pairs:
    tg1, tg2 = homopolymer_tg[smiles1], homopolymer_tg[smiles2]
    for w1 in np.arange(0.1, 1.0, 0.1):
        w2 = 1 - w1
        tg_fox = 1.0 / (w1/tg1 + w2/tg2)
```

### 4.4 精度与局限

- Fox 方程 MAE: 10-30 K（对无强相互作用体系较好）
- 无法处理：氢键体系、电荷转移复合物、序列分布效应
- 改进：Gordon-Taylor 方程引入 K 参数，但需要实验拟合
- **对核酸共聚物**：因存在强氢键，Fox 方程偏差较大，需要修正项

### 4.5 与本项目的关系

本项目目标之一是共聚物 Tg 预测。Fox 方程可快速生成大量共聚物虚拟 Tg 作为预训练数据，再用少量实验数据微调，是**性价比最高的数据增强策略**。

---

## 5. 方法三：QSPR/QSPP 模型批量预测

### 5.1 原理

利用已训练好的 QSPR（定量结构-性质关系）模型，对新聚合物结构批量预测 Tg。

### 5.2 代表性工作

**POINT2 数据库** (Ma & Luo, 2024)：
- 基于 PI1M（100万虚拟聚合物 SMILES）
- 使用 Quantile Random Forest、MLP、GNN、LLM 四种模型
- 预测 Tg 等 6 种性质，附带不确定性量化
- 数据规模：~100 万条虚拟聚合物 Tg 预测值
- 公开可用：https://github.com/Yihan222/POINT-2

**Lieconv-Tg** (Tao et al., 2021)：
- 等变神经网络，基于 3D 构象
- 训练集：7,166 种聚合物
- MAE: 24.4 C
- 筛选 ~100 万聚合物，找到 49,435 种 Tg > 200 C 的候选

### 5.3 作为虚拟数据源

- 可直接使用 POINT2 的预测结果作为低保真度训练数据
- 或训练自己的 QSPR 模型，对新化学空间进行预测
- 关键：需要附带**不确定性估计**，高不确定性的预测应降权或丢弃

### 5.4 优缺点

**优势**：
- 速度快（秒级/样本），可大规模生成
- 可利用现有开源模型和数据
- 不确定性量化帮助筛选可靠数据

**劣势**：
- 外推能力有限（训练域外预测不可靠）
- 模型偏差会系统性传播
- 需要高质量训练数据来构建初始模型

---

## 6. 方法四：半经验量化计算生成描述符

### 6.1 原理

使用半经验量子化学方法（如 PM7、GFN2-xTB）计算聚合物重复单元的电子结构描述符，作为 ML 模型的输入特征。

### 6.2 GFN2-xTB 方法

- 开发者：Grimme 课题组
- 特点：速度比 DFT 快 100-1000 倍，精度介于力场和 DFT 之间
- 可计算描述符：HOMO/LUMO 能级、偶极矩、极化率、振动频率、热力学量
- 单分子计算时间：秒到分钟级（取决于原子数）
- 软件：xtb（免费开源）

### 6.3 虚拟数据生成方案

1. 对 10 万种聚合物重复单元进行 GFN2-xTB 优化和性质计算
2. 提取描述符矩阵 [HOMO, LUMO, dipole, polarizability, ...]
3. 结合 Morgan fingerprint 等结构描述符
4. 训练增强的 ML 模型预测 Tg

### 6.4 规模估算

- 10 万种聚合物 x 平均 3 分钟/构象优化
- 总计 ~5,000 CPU 小时 = 单台 40 核服务器约 5 天
- 成本可控，适合学术课题组

### 6.5 优缺点

**优势**：
- 提供物理上有意义的描述符
- 计算成本适中，可达 10 万级规模
- 捕捉电子结构信息（基团贡献法无法获取）

**劣势**：
- 仅计算单体/重复单元，无法反映链间相互作用
- 构象采样可能不充分
- 对大分子（>200 原子）计算变慢

---

## 7. 方法五：全原子 MD 模拟

### 7.1 RadonPy 平台

**RadonPy** 是目前最成熟的聚合物 MD 自动化平台：

- 开发者：日本理化学研究所 (RIKEN)
- 功能：全自动构建聚合物体系、力场分配、平衡态 MD、性质计算
- 力场：GAFF (General Amber Force Field)
- 可计算性质：62 种，包括 Tg、密度、热导率等
- 验证：1,000+ 种聚合物与实验值比较
- GitHub: https://github.com/RadonPy/RadonPy

### 7.2 Tg 计算方法

RadonPy 通过**密度-温度曲线**计算 Tg：
1. 在多个温度点（如 100K-600K，间隔 25K）进行 NPT 模拟
2. 计算每个温度点的平衡密度
3. 对密度-温度数据进行双线性拟合
4. 两条直线的交点即为 Tg

### 7.3 精度与可靠性

- **单次模拟 MAE**：30-50 K（与实验值比较）
- **集成模拟（10+ replicas）MAE**：11-18 K
- **95% 置信区间**：< 20 K（10 个 replicas）
- 关键：需要足够的 replicas 来降低统计噪声

### 7.4 计算成本

- 单个聚合物完整 Tg 计算：30-50 小时（40 核 CPU）
- 集成模拟（10 replicas）：300-500 CPU 小时
- 并行化后实际耗时：~6 小时（使用集群）
- 1,000 种聚合物：约 50 万 CPU 小时

### 7.5 SPACIER 工作流

**SPACIER** = RadonPy + 贝叶斯优化：
- 发表于 npj Computational Materials (2025)
- 目的：按需设计满足目标性质的聚合物
- 流程：RadonPy 计算 -> GP 代理模型 -> 贝叶斯优化选择下一个候选 -> 循环
- 优势：减少总体计算量（不需要遍历所有候选）

### 7.6 其他 MD 工具

| 工具 | 特点 | 适用场景 |
|------|------|--------|
| PYSIMM | Python 接口，支持 LAMMPS | 通用聚合物 MD |
| PolyParGen | 自动力场参数化 | CGenFF 力场 |
| OpenMM | GPU 加速 | 大体系快速模拟 |
| GROMACS | 高性能 | 蛋白质/聚合物 |

### 7.7 适用场景

- 需要**高精度**虚拟 Tg 数据时
- 作为多保真度学习的**最高计算保真度层**
- 验证其他低成本方法的预测结果
- 规模有限（百-千级），适合高价值候选聚合物

---

## 8. 方法六：SMILES 数据增强

### 8.1 原理

同一聚合物可以有多种合法的 SMILES 表示。通过枚举不同的 SMILES 字符串，增加训练集的表观大小和多样性。

### 8.2 实现方法

使用 RDKit 的 SMILES 随机化功能：


```python
from rdkit import Chem

def enumerate_smiles(smiles, n=10):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    return [Chem.MolToSmiles(mol, doRandom=True) for _ in range(n)]
```

### 8.3 效果

- 文献报道：10-20x 增强效果最优
- Tg 预测：R-squared 从 0.56 提升到 0.66（Bjerrum, 2017 方法论）
- 对基于 Transformer/LSTM 的模型效果最显著
- 对基于 fingerprint 的模型无效（因为 fingerprint 与 SMILES 顺序无关）

### 8.4 其他数据增强技术

**噪声注入**：
- 在 Tg 值上添加高斯噪声 N(0, sigma)
- sigma 通常设为实验不确定性的量级（如 5-10 K）

**kNNMTD**：
- 基于 k 近邻的多目标数据增强
- 在聚合物 Tg 预测中验证：RF 模型 R-squared = 0.85
- 通过在描述符空间中插值生成新样本

**Mixup**：
- 在特征空间中线性插值两个样本
- 适用于连续描述符，不适用于 SMILES 字符串

### 8.5 优缺点

**优势**：
- 实现极简单，几行代码
- 零额外计算成本
- 对序列模型效果好

**劣势**：
- 不引入新的化学信息
- 增强倍数有上限（过多反而有害）
- 对非序列模型无效

---

## 9. 方法七：生成模型合成虚拟聚合物

### 9.1 CharRNN -> PI1M

**PI1M** (Polymer Informatics with 1 Million polymers)：
- 开发者：University of Notre Dame (Ramprasad 课题组)
- 方法：CharRNN 在 PolyInfo 的 ~13,000 种聚合物 SMILES 上训练
- 生成：100 万种虚拟聚合物 SMILES
- 验证：生成的 SMILES 化学有效率 > 90%
- 数据公开：https://github.com/Ramprasad-Group/PI1M
- 后续：POINT2 项目使用 PI1M 作为虚拟库，预测 Tg 等性质

### 9.2 VAE（变分自编码器）

**Vitrimer 设计案例**：
- 生成 100 万种虚拟 vitrimer 结构
- 使用 MD 计算其中 8,424 种的 Tg
- 筛选出满足目标 Tg 范围的候选
- 发表于 Nature Computational Science (2024)

### 9.3 REINVENT

- AstraZeneca 开发的分子生成平台
- 基于 RNN + 强化学习
- 可针对目标性质（如特定 Tg 范围）定向生成
- 适用于小分子，需要适配聚合物场景

### 9.4 GraphINVENT

- 基于图神经网络的分子生成
- 直接在分子图空间生成，避免 SMILES 语法错误
- 生成有效率更高

### 9.5 优缺点

**优势**：
- 可探索未知化学空间
- 生成规模大（百万级）
- 可结合性质优化（条件生成/强化学习）

**劣势**：
- 生成结构的**可合成性**不保证
- 训练需要大量数据
- 生成质量依赖模型架构和训练策略
- 聚合物特有的表示问题（重复单元 vs 全链）

---

## 10. 方法八：基于规则的虚拟聚合物库构建

### 10.1 SMiPoly

**SMiPoly** (SMILES to Polymers)：
- 方法：22 种聚合反应规则 + 1,083 种商用单体
- 生成：169,347 种虚拟聚合物
- 与 PolyInfo 覆盖率：48%（验证有效性）
- 新聚合物比例：53%（拓展化学空间）
- 优势：生成的聚合物**理论上可合成**（基于真实反应规则）

### 10.2 反应规则示例

- 加聚（如乙烯基聚合）
- 缩聚（如酰胺键形成）
- 开环聚合
- 自由基聚合
- 离子聚合

### 10.3 与生成模型的比较

| 维度 | 规则法 (SMiPoly) | 生成模型 (PI1M) |
|------|-----------------|----------------|
| 可合成性 | 高（基于反应规则）| 低（无合成约束）|
| 多样性 | 受限于规则和单体 | 高（学习分布）|
| 规模 | 中等（十万级）| 大（百万级）|
| 化学有效性 | 100% | 90-95% |
| 新颖性 | 53% | 高但不保证可合成 |

### 10.4 适用场景

- 需要**可合成性保证**的虚拟库
- 与特定单体供应商对接的实际应用
- 共聚物虚拟库构建（组合两种单体的反应规则）

---

## 11. 方法九：多保真度迁移学习

### 11.1 核心思想

多保真度学习是将不同精度的数据源融合到一个统一框架中：

- 大量低保真度数据（基团贡献法、Fox 方程、QSPR）提供广泛的化学空间覆盖
- 少量高保真度数据（MD、实验）提供精确校准

### 11.2 Delta Learning（差值学习）

核心公式：

y_high = y_low + delta(x)

其中 delta(x) 是 ML 模型学习的高低保真度之间的差值。

**Implicit Delta Learning (IDLe)**：
- 比普通迁移学习需要 **50x 更少**的高保真度数据
- 适用于聚合物性质预测
- 参考：Doan et al., J. Phys. Chem. Lett. (2024)

### 11.3 多保真度迁移学习 + GNN

最新研究进展：
- 使用 GNN 作为基础模型，在低保真度数据上预训练
- 在高保真度数据上微调（自适应 readout 层是关键）
- MAE 降低 20-60%，只需 10x 更少的高保真度数据
- 参考：Venetos et al., npj Comput. Mater. (2024)

### 11.4 Co-Kriging / 多保真度 GP

- 高斯过程框架下的多保真度融合
- 在 382 种聚合物的带隙预测中已验证
- 适用于小数据集场景
- 内置不确定性量化

### 11.5 实施建议

对本项目的推荐多保真度策略：

```
F0 (基团贡献法, ~ms)  --> 百万级均聚物 Tg
        |
F1 (Fox 方程, ~ms)      --> 千万级共聚物 Tg
        |
F2 (QSPR/ML, ~s)          --> 百万级预测 Tg
        |
F3 (GFN2-xTB+ML, ~min)    --> 十万级描述符增强
        |
F4 (MD/RadonPy, ~hr)      --> 千级高精度 Tg
        |
F5 (实验, ~天)           --> 百级基准数据
```

每一层通过 delta learning 或迁移学习与上一层连接。

---

## 12. 方法十：GAN/条件生成数据增强

### 12.1 WGAN-GP 方法

- 使用 Wasserstein GAN with Gradient Penalty 生成合成数据
- 在共聚物 Tg 预测中的应用：
  - 结合 CNN-LSTM 模型
  - R-squared = 0.95
  - 显著优于无增强基线

### 12.2 条件生成

- 条件 VAE/GAN：指定目标 Tg 范围，生成满足条件的聚合物结构
- 优势：生成的数据分布更接近目标分布
- 缺点：需要足够的标注数据训练生成器

### 12.3 优缺点

**优势**：
- 生成的数据分布接近真实数据
- 可以条件控制生成方向
- 不需要额外的物理计算

**劣势**：
- 训练 GAN 不稳定（模式崩溃等问题）
- 生成数据的物理合理性难以保证
- 需要较大的初始数据集

---

## 13. 工具与平台汇总

| 工具/平台 | 类型 | 用途 | 开源 | URL |
|-----------|------|------|------|-----|
| RadonPy | MD 自动化 | 全原子 MD 计算 62 种聚合物性质 | 是 | github.com/RadonPy/RadonPy |
| PYSIMM | MD 接口 | LAMMPS 聚合物模拟 | 是 | pysimm.org |
| PolyParGen | 力场 | 自动力场参数化 | 是 | polypargen.com |
| xtb | 量化 | GFN2-xTB 半经验计算 | 是 | github.com/grimme-lab/xtb |
| PI1M | 数据库 | 100万虚拟聚合物 SMILES | 是 | github.com/Ramprasad-Group/PI1M |
| POINT2 | 数据库 | 100万聚合物性质预测 | 是 | github.com/Yihan222/POINT-2 |
| SMiPoly | 生成器 | 基于反应规则的聚合物生成 | 是 | - |
| REINVENT | 生成器 | RNN+RL 分子生成 | 是 | github.com/MolecularAI/REINVENT |
| RDKit | 化学信息学 | SMILES 处理/fingerprint/描述符 | 是 | rdkit.org |
| OpenMM | MD | GPU 加速 MD 模拟 | 是 | openmm.org |
| LAMMPS | MD | 通用 MD 引擎 | 是 | lammps.org |

---

## 14. 方法对比矩阵

| 维度 | 基团贡献法 | Fox 方程 | QSPR/ML | 半经验 | MD (RadonPy) | SMILES增强 | 生成模型 | 规则法 | 多保真度 | GAN |
|------|----------|--------|---------|------|------------|----------|----------|------|----------|-----|
| 计算成本 | 极低 | 极低 | 低 | 中 | 高 | 极低 | 中 | 低 | 变化 | 中 |
| Tg 精度 | 30-50K | 10-30K | 20-40K | 15-30K | 11-18K | N/A | N/A | N/A | 最优 | 中 |
| 可生成规模 | 百万 | 千万 | 百万 | 十万 | 千 | 10-20x | 百万 | 十万 | - | 万 |
| 实现难度 | 低 | 低 | 中 | 中-高 | 高 | 极低 | 高 | 中 | 高 | 高 |
| 共聚物适用 | 差 | 优 | 中 | 中 | 优 | 中 | 中 | 优 | 优 | 中 |
| 新化学信息 | 无 | 无 | 低 | 中 | 高 | 无 | 高 | 中 | 变化 | 中 |
| 物理可解释 | 高 | 高 | 低 | 中 | 高 | N/A | 低 | 高 | 中 | 低 |
| 本项目优先级 | ★★★ | ★★★★★ | ★★★★ | ★★★ | ★★ | ★★★★ | ★★★ | ★★★ | ★★★★★ | ★★ |

---

## 15. 推荐实施路线

### 阶段一：快速见效（1-2 周）

**目标**：用最低成本获得最大数据增益

1. **SMILES 枚举增强**：对现有训练集做 10-20x SMILES 增强
   - 工具：RDKit
   - 仅对序列模型（Transformer/LSTM）有效

2. **Fox 方程生成共聚物 Tg**：
   - 从 PolyInfo 获取均聚物 Tg 数据
   - 组合生成数十万条共聚物虚拟 Tg
   - 作为共聚物 Tg 模型的预训练数据

3. **下载 POINT2 数据库**：
   - 直接使用 100 万聚合物的预测 Tg 作为低保真度数据
   - 结合不确定性估计筛选可靠数据

### 阶段二：多保真度融合（2-4 周）

**目标**：构建多保真度学习框架

1. **实现 Delta Learning**：
   - F0（基团贡献法）-> F5（实验）的差值学习
   - F1（Fox 方程）-> F5 的差值学习（共聚物）

2. **迁移学习实验**：
   - 在 POINT2 数据上预训练 GNN
   - 在 PolyInfo 实验数据上微调
   - 对比有/无预训练的性能差异

### 阶段三：深度拓展（4-8 周）

**目标**：构建完整的虚拟数据生成管线

1. **GFN2-xTB 描述符计算**：
   - 对 1-10 万聚合物重复单元计算电子结构描述符
   - 将描述符纳入 ML 特征

2. **生成模型探索**：
   - 基于 PI1M/SMiPoly 方法，对核酸共聚物化学空间进行拓展
   - 训练核酸单体专用生成模型

3. **MD 验证**（可选，需要计算资源）：
   - 选取 50-100 种关键聚合物进行 RadonPy MD Tg 计算
   - 作为最高保真度层数据

### 预期效果

| 阶段 | 数据增量 | 预期 MAE 改善 | 资源需求 |
|------|---------|-------------|----------|
| 一 | 10万-100万 | 5-15% | 普通 PC |
| 二 | 融合上述数据 | 15-30% | 普通 PC + GPU |
| 三 | 额外 1万-10万 | 20-40% | 服务器/集群 |

---

## 16. 参考文献

[1] Van Krevelen, D.W. & Te Nijenhuis, K. "Properties of Polymers" 4th ed., Elsevier, 2009.

[2] Bicerano, J. "Prediction of Polymer Properties" 3rd ed., Marcel Dekker, 2002.

[3] Ma, R. & Luo, T. "POINT2: A multi-model ensemble learning database for polymer informatics" (2024). GitHub: Yihan222/POINT-2

[4] Hayashi, Y. et al. "RadonPy: Automated physical property calculation using all-atom classical molecular dynamics simulations for polymer informatics" npj Comput. Mater. 8, 222 (2022).

[5] Hayashi, Y. et al. "SPACIER: On-demand polymer design with fully automated all-atom classical molecular dynamics integrated into machine learning pipelines" npj Comput. Mater. (2025).

[6] Tao, L. et al. "Benchmarking Machine Learning Models for Polymer Informatics: An Example of Glass Transition Temperature" J. Chem. Inf. Model. 61, 5395-5413 (2021).

[7] Tao, L. et al. "Machine learning discovery of high-temperature polymers" Patterns 2, 100225 (2021). (Lieconv-Tg)

[8] Kuenneth, C. et al. "Polymer Informatics with Multi-Task Learning" Patterns 2, 100238 (2021). (PI1M)

[9] Kim, C. et al. "Polymer Genome: A Data-Powered Polymer Informatics Platform" J. Phys. Chem. C 122, 17575-17585 (2018).

[10] Venetos, M. et al. "Multi-fidelity transfer learning for property prediction of polymers using graph neural networks" npj Comput. Mater. (2024).

[11] Doan, H. et al. "Implicit Delta Learning for Polymer Property Prediction" J. Phys. Chem. Lett. (2024).

[12] Fox, T.G. "Influence of diluent and of copolymer composition on the glass temperature of a polymer system" Bull. Am. Phys. Soc. 1, 123 (1956).

[13] Gordon, M. & Taylor, J.S. "Ideal copolymers and the second-order transitions of synthetic rubbers" J. Appl. Chem. 2, 493-500 (1952).

[14] Bannwarth, C. et al. "GFN2-xTB - An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method" J. Chem. Theory Comput. 15, 1652-1671 (2019).

[15] Bjerrum, E.J. "SMILES Enumeration as Data Augmentation for Neural Network Modeling of Molecules" arXiv:1703.07076 (2017).

[16] Pilania, G. et al. "Multi-fidelity machine learning models for accurate bandgap predictions of solids" Comput. Mater. Sci. 129, 156-163 (2017).

[17] Aldeghi, M. & Bhatt, S. "SMiPoly: Generation of a Synthesizable Polymer Virtual Library" (2024).

[18] Lin, T.S. et al. "BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules" ACS Cent. Sci. 5, 1523-1531 (2019).

[19] Patel, R.A. et al. "Featurization strategies for polymer sequence or composition effects" (2022).

[20] Simine, L. et al. "WGAN-GP data augmentation for polymer Tg prediction" (2023).

[21] Xu, C. & Luo, T. "Machine learning interatomic potentials for polymer molecular dynamics" (2023).

[22] Bhowmik, R. et al. "Polymer graph neural networks for multitask property prediction" (2022).

[23] Otsuka, S. et al. "PolyInfo: Polymer database for polymeric materials design" (2011).

[24] Statt, A. et al. "Machine learning-assisted polymer informatics" (2023).

[25] Chen, L. et al. "Polymer informatics: Current status and critical next steps" Mater. Sci. Eng. R 144, 100595 (2021).

[26] Wu, S. et al. "Machine-Learning-Assisted Discovery of Polymers with High Thermal Conductivity" (2019).

[27] Volgin, I.V. et al. "Molecular dynamics of polymer glass transition" (2024).

[28] Afzal, M.A.F. & Hachmann, J. "Benchmarking DFT approaches for the calculation of polarizability inputs for refractive index predictions in organic polymers" (2019).

[29] Audus, D.J. & de Pablo, J.J. "Polymer Informatics: Opportunities and Challenges" ACS Macro Lett. 6, 1078-1082 (2017).

[30] Ramprasad, R. et al. "Machine Learning in Materials Informatics: Recent Applications and Prospects" npj Comput. Mater. 3, 54 (2017).

---

> 本报告基于 2026 年 3 月的文献调研，部分方法和工具可能有更新版本，建议实施前检查最新状态。
