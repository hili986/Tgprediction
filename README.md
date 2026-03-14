# Tg 预测项目：AI 辅助高分子玻璃化转变温度预测

同济大学 SITP（大学生创新训练计划）项目 —— 基于机器学习的聚合物玻璃化转变温度 (Tg) 预测，支持从核酸序列端到端预测 Tg。

## 项目亮点

- **22,674 条聚合物 Tg 数据**：整合 6 个公开数据库（PolyMetriX, POINT-2, NeurIPS OPP 等），304 条 Bicerano 高质量基线
- **多层级特征工程**：Afsordeh 4-dim → RDKit 2D 15-dim → H-bond 15-dim，共 34 维物化描述符
- **桥梁迁移学习**：通过 205 条含氢键聚合物实现核酸分子 Tg 预测，核酸小分子平均误差 7.7K
- **核酸序列 → Tg**：端到端管道，输入 DNA/RNA 序列直接输出 Tg 预测值
- **BigSMILES 工具链**：完整的 BigSMILES 解析、验证、指纹生成模块

## 环境要求

- Python 3.9+
- RDKit 2024.03+

```bash
# 安装依赖
pip install -r requirements.txt

# 验证 RDKit 安装
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO').GetNumAtoms())"
# 预期: 3
```

> **注意**: RDKit 建议通过 conda 安装 —— `conda install -c conda-forge rdkit`

## 快速开始

### 1. 核酸序列 Tg 预测（交互模式）

```bash
python scripts/predict_tg_from_sequence.py
# 预期: 模型加载 (~1s) 后进入交互界面，输入序列即可预测
```

### 2. 核酸序列 Tg 预测（命令行）

```bash
python scripts/predict_tg_from_sequence.py --seq ACGT --type DNA
# 预期: 输出完整序列预测 + 各单体预测 + 置信度评估

python scripts/predict_tg_from_sequence.py --seq AUGC --type RNA --json
# 预期: JSON 格式输出预测结果
```

### 3. 运行实验

```bash
# Phase 3 迁移学习实验
python scripts/exp_phase3_transfer.py
# 预期: 基线对比 + 权重扫描 + 核酸预测，结果保存到 results/phase3/

# SHAP 特征分析
python scripts/exp_phase3_shap.py
# 预期: Top-10 特征排名 + 特征组贡献分析
```

### 4. 运行测试

```bash
python -m unittest discover tests/ -v
# 预期: 374 个测试通过 (~20s)
```

## 功能模块

### 1. 核酸序列预测 (`scripts/predict_tg_from_sequence.py`)

端到端核酸序列 → Tg 预测脚本。自动训练 GBR 模型（Bicerano 304 + 桥梁聚合物 205），支持 DNA/RNA 序列输入。

**CLI 用法：**

```bash
# 交互模式
python scripts/predict_tg_from_sequence.py

# 直接预测
python scripts/predict_tg_from_sequence.py --seq <序列> --type <DNA|RNA> [--json] [--bridge-weight <0-1>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--seq` | 核酸序列（如 `ACGT`） | 无（进入交互模式） |
| `--type` | 序列类型 | `DNA` |
| `--bridge-weight` | 桥梁数据权重 | `0.8` |
| `--json` | JSON 格式输出 | 否 |

**置信度等级：**

| 序列长度 | 置信度 | 说明 |
|----------|--------|------|
| 1 nt | HIGH | 单核苷酸，与训练集化学空间一致 |
| 2-4 nt | MEDIUM | 短序列，分子大小接近训练集 |
| >4 nt | LOW | 远超训练集范围，建议参考单体平均值 |

### 2. 核酸序列 SMILES 转换 (`src/sequence/nucleotide_smiles.py`)

核酸序列到原子级 SMILES 的转换模块，支持 DNA 和 RNA。

> **注意：** 纯库模块，不提供独立 CLI。通过 `predict_tg_from_sequence.py` 调用，或在 Python 中导入使用。

**API 函数：**

| 函数 | 说明 |
|------|------|
| `validate_sequence(seq, seq_type)` | 验证并清洗序列，返回大写序列 |
| `build_full_smiles(seq, seq_type, direction)` | 构建完整磷酸二酯键连接的 SMILES |
| `get_monomer_smiles(base, seq_type)` | 获取单核苷酸单体 SMILES |
| `sequence_to_smiles(seq, seq_type, direction)` | 返回完整 SMILES + 各单体 SMILES 的字典 |

```python
from src.sequence.nucleotide_smiles import sequence_to_smiles

info = sequence_to_smiles("ACGT", "DNA")
print(info["full_smiles"])    # 完整序列 SMILES
print(info["monomers"])       # 各单体信息列表
```

### 3. 特征工程管道 (`src/features/`)

多层级特征提取系统，从 SMILES 计算物化描述符。

> **注意：** 纯库模块。通过实验脚本或预测脚本调用。

**特征层级：**

| 层级 | 维度 | 包含 | 说明 |
|------|------|------|------|
| L0 | 4 | Afsordeh 物理特征 | FlexibilityIndex, HBondDensity, SOL, VdW |
| L1 | 19 | L0 + RDKit 2D | 加入 15 个 RDKit 分子描述符 |
| L1H | 34 | L1 + H-bond | 加入 15 个氢键 SMARTS 特征 |
| L2 | 1043 | L1 + Morgan FP | 加入 1024-bit Morgan 指纹 |
| L2H | 1058 | L2 + H-bond | 完整特征集 |

**关键文件：**

| 文件 | 说明 |
|------|------|
| `feature_pipeline.py` | 统一特征计算入口 (`compute_features`, `build_dataset_v2`) |
| `afsordeh_features.py` | Afsordeh 2025 四大物理特征 |
| `rdkit_descriptors.py` | RDKit 2D 分子描述符 (15-dim) |
| `hbond_features.py` | 氢键 SMARTS 特征 (15-dim) |
| `selection.py` | 四阶段特征选择 (Variance → Boruta → mRMR → SHAP) |

### 4. 数据集 (`src/data/`)

| 文件 | 说明 |
|------|------|
| `bicerano_tg_dataset.py` | Bicerano 304 条均聚物数据集（内嵌，无外部文件依赖） |
| `bridge_polymers.py` | 205 条桥梁聚合物（PA/PU/聚磷腈/PI 等 7 个家族） |
| `fox_copolymer_generator.py` | Fox/Gordon-Taylor 方程生成虚拟共聚物 |
| `external_datasets.py` | 外部数据集统一加载器（6 源，22,674 条去重后） |

**外部数据源：**

| 数据集 | 原始条数 | 来源 |
|--------|---------|------|
| PolyMetriX | 7,367 | Zenodo（含可靠性分级） |
| POINT-2 | 7,208 | PolyInfo 聚合 |
| NeurIPS OPP 2025 | 17,330 | 竞赛聚合数据 |
| OpenPoly | 443 | 复旦大学 |
| Qiu Polymer | 314 | 邱组通用聚合物 |
| Qiu Polyimide | 372 | 邱组聚酰亚胺 |

### 5. 机器学习 (`src/ml/`)

| 文件 | 说明 |
|------|------|
| `sklearn_models.py` | scikit-learn 模型工厂（ET, GBR, SVR, Stacking 等） |
| `evaluation.py` | Nested CV / Simple CV 评估框架 |
| `two_stage_training.py` | 两阶段训练（虚拟数据预训练 → 实验数据微调） |
| `models.py` | 纯 Python 手写 ML 模型（教学用途，不修改） |
| `experiment.py` | 旧版实验框架（不修改） |

### 6. BigSMILES 工具链 (`src/bigsmiles/`)

完整的 BigSMILES 聚合物表示解析工具链（已完成，不修改）。

| 文件 | 说明 |
|------|------|
| `checker.py` | BigSMILES 语法验证器 |
| `parser.py` | AST 解析器（提取重复单元、拓扑） |
| `fingerprint.py` | Morgan 指纹 + 片段向量 + 聚合物描述符 |
| `annotation.py` | 属性标注系统 (`Tg=373K;Mn=50000`) |
| `examples.py` | 39 种常见聚合物 BigSMILES 库 |

## 项目结构

```
Tg预测项目/
├── src/
│   ├── sequence/                # 核酸序列处理
│   │   └── nucleotide_smiles.py #   序列 → SMILES 转换
│   ├── features/                # 特征工程
│   │   ├── feature_pipeline.py  #   统一特征计算入口
│   │   ├── afsordeh_features.py #   Afsordeh 4 维物理特征
│   │   ├── rdkit_descriptors.py #   RDKit 2D 描述符 (15-dim)
│   │   ├── hbond_features.py    #   氢键 SMARTS 特征 (15-dim)
│   │   └── selection.py         #   四阶段特征选择
│   ├── data/                    # 数据集
│   │   ├── bicerano_tg_dataset.py  # Bicerano 304 均聚物
│   │   ├── bridge_polymers.py      # 205 桥梁聚合物
│   │   ├── fox_copolymer_generator.py # Fox 虚拟共聚物
│   │   └── external_datasets.py    # 外部数据集加载 (6 源, 22K+)
│   ├── ml/                      # 机器学习
│   │   ├── sklearn_models.py    #   模型工厂 + 搜索空间
│   │   ├── evaluation.py        #   CV 评估框架
│   │   └── two_stage_training.py#   两阶段训练
│   └── bigsmiles/               # BigSMILES 工具链 (已完成)
│       ├── checker.py           #   语法验证
│       ├── parser.py            #   AST 解析
│       ├── fingerprint.py       #   分子指纹
│       ├── annotation.py        #   属性标注
│       └── examples.py          #   聚合物示例库
├── scripts/                     # 实验脚本
│   ├── predict_tg_from_sequence.py # 核酸 Tg 端到端预测
│   ├── exp_phase3_transfer.py      # Phase 3 迁移学习
│   ├── exp_phase3_tuning.py        # Phase 3 超参调优
│   └── exp_phase3_shap.py          # Phase 3 SHAP 分析
├── tests/                       # 374 个单元测试
├── data/                        # 数据文件
│   ├── bridge_polymers.csv      #   桥梁数据 CSV 导出
│   └── external/                #   外部数据集 (6 个 CSV)
├── results/                     # 实验结果
│   ├── phase1/                  #   描述符增强实验
│   ├── phase2/                  #   Fox 虚拟数据实验
│   └── phase3/                  #   桥梁迁移学习 + 数据扩展
├── docs/
│   ├── plans/                   #   实验计划
│   └── research/                #   8 份调研报告
├── requirements.txt
└── README.md
```

## 实验结果总览

### Phase 1：描述符增强

| 实验 | 特征 | 模型 | R² | MAE (K) |
|------|------|------|-----|---------|
| 1.1 | L0 (4-dim) | ET | 0.745 | 41.4 |
| 1.2 | L1 (19-dim) | GBR | 0.816 | 35.0 |
| 1.3 | L2 (1072-dim) | SVR | 0.858 | 29.6 |
| 1.5a | L1 tuned (Nested CV) | ET | 0.811 | 35.1 |
| 1.5b | L2 tuned (Nested CV) | SVR | **0.874** | **28.1** |

**关键发现**：L0→L1 提升 +12.9%，L1→L2 再提升 +7.1%。RingCount 是 Tg 头号预测因子（SHAP 确认）。Stacking 集成在 304 样本上失败（R²=-0.40）。

详细报告见 `results/phase1/phase1-summary.md`。

### Phase 2：Fox 虚拟数据扩展

| 实验 | 方法 | R² | MAE (K) | 备注 |
|------|------|-----|---------|------|
| 2.2b | ET + weighted 0.05 | 0.842 | 31.4 | 最佳配置 |

**关键发现**：发现并修复数据泄露问题（Fox 方程编码测试集信息），真实提升仅 +1%。虚拟数据不引入新结构信息。

详细报告见 `results/phase2/phase2-summary.md`。

### Phase 3：桥梁迁移学习 + 数据扩展

| 实验 | 配置 | R² | MAE (K) |
|------|------|-----|---------|
| 3.3 | GBR + bridge(0.4) | **0.837** | **33.1** |
| 3.4 | ET tuned + bridge(0.8) | 0.835 | 32.2 |

**核酸预测（GBR + bridge 0.8）：**

| 分子 | 预期 Tg (K) | 预测 Tg (K) | 误差 |
|------|-----------|-----------|------|
| ATP | 246 | 247.0 | 1.0K |
| ADP | 244 | 245.3 | 1.3K |
| AMP | 249 | 258.5 | 9.5K |
| GMP | 260 | 279.0 | 19.0K |

核酸小分子平均误差 **7.7K**，相比基线提升 **98.3%**。

**数据扩展**：整合 6 个外部数据库（33,034 → 22,674 条去重），构建统一加载器 `external_datasets.py`。

详细实验报告见 `results/phase3/phase3-summary.md`。

## 测试

```bash
# 全部测试
python -m unittest discover tests/ -v
# 预期: 374 tests, ~20s, 全部通过 (1 个 skip)

# 单个模块测试
python -m unittest tests/test_hbond_features.py -v
python -m unittest tests/test_bicerano_tg.py -v
python -m unittest tests/test_external_datasets.py -v
```

## 参考文献

1. Afsordeh, M. et al. (2025). Simple 4-descriptor model for Tg prediction (R²=0.97).
2. Bicerano, J. (2002). *Prediction of Polymer Properties*. 3rd ed., Marcel Dekker.
3. Fox, T.G. (1956). Influence of diluent and copolymer composition on Tg. *Bull. Am. Phys. Soc.* 1, 123.
4. Lin, T.S. et al. (2021). BigSMILES: A structurally-based line notation for describing macromolecules. *ACS Cent. Sci.* 5(9), 1523-1531.

## 许可证

本项目为同济大学 SITP 学术研究项目，仅限学术用途。
