# Tg 预测项目 — CLAUDE.md

## 项目概述
同济大学 SITP: AI 辅助高分子材料设计 — 预测玻璃化转变温度 (Tg)
方案A（物理特征增强）已完成: TabPFN R²=0.8955 + 核酸 ATP/ADP 误差 <5K
方案B（GNN 混合）代码就绪，待 A800 服务器启动

## 技术栈
- Python 3.x (use `python` not `python3`)
- scikit-learn, numpy, pandas, shap, matplotlib
- RDKit (已安装 2025.09.4)
- CatBoost, LightGBM, XGBoost, TabPFN v2
- MAPIE (不确定性量化)
- PyTorch + PyG (方案B GNN)

## 项目结构
```
src/
  data/           → 数据集 (bicerano, external, bridge, fox_copolymer)
  features/       → 特征工程 (afsordeh, rdkit, hbond, pipeline, physical_proxy, virtual_polymerization)
  ml/             → 模型 (sklearn_models, evaluation, uncertainty, hierarchical, two_stage, gnn_evaluation)
  gnn/            → GNN 子包 (graph_builder, physics_gat, tandem_m2m, pretrainer, multitask, ensemble)
  analysis/       → 分析 (shap, ablation, visualization)
  bigsmiles/      → BigSMILES 工具链 (已完成, 不修改)
  sequence/       → 核酸序列处理 (nucleotide_smiles)
tests/            → 单元测试 (14 文件)
scripts/          → 实验脚本 (12 个, exp_phase2~5b + predict_tg)
data/             → 原始数据文件
results/          → 实验输出 (phase1-5/ + 方案A总结.md)
docs/
  INDEX.md        → 文档总目录 (统一入口)
  overview/       → 项目全貌 (背景/技术路线/进展)
  research/       → 调研报告库 (按 foundations/methods/domain/decisions 分类)
  experiments/    → 实验统一索引
  plans/          → 实验计划
```

## 当前状态 (2026-03-24)
- **方案A**: 100% 完成 (Phase 1-5B, E1-E26)
- **最优通用预测**: TabPFN v2, R²=0.8955, MAE=24K (零调参)
- **核酸迁移**: CatBoost, ATP 4.7K, ADP 0.4K
- **学习计划**: 4 阶段完成 (Tg 物理→SHAP 解读→文献精读→原创假说)
- **多尺度重构进行中**:
  - Phase A: 数据诊断 ✅ (R²天花板 0.96-0.99)
  - Phase B: 特征工程 ✅ (PHY 48d: R²=0.8724, MAE=28.5K, +0.6% vs M2M-V)
  - Phase B2: 链间相互作用 ✅ (PHY-B2 56d: R²=0.8836, MAE=27.0K, +0.84%)
  - Phase C: 链段物理特征 ✅ (PHY-C-light 58d: R²=0.8831, MAE=26.9K)
  - Phase D: GNN + polyBERT 嵌入 ⏳
  - Phase E: 物理专家委员会 ⏳
- **当前最优特征集 PHY-C-light 58d**: PHY-B2(56d) + chain_physics(3d: curl_ratio, Neff_ratio, conf_strain) - 冗余(1d: IC_hydrophilic_ratio)

## 当前执行方案 — 物理驱动多尺度重构
> 详细方案: `docs/plans/方案待选-物理驱动多尺度算法重构.md` (v3)
> 核心理念: 结果优先，不为简化牺牲效果

### 行动哲学
- **有效数据极致精度**: 不追求 100% 覆盖率，追求有效数据点上最好结果
- **NaN = 信号**: 计算失败填 NaN + 元特征标记，模型自动处理
- **全量推进**: 不逐步回退，全部算完在最终集成中统一评判
- **结果优先**: 不为减少复杂度牺牲效果，A800 算力充足

### 多尺度特征金字塔 (按 Tg 物理因果链)
| 尺度 | 特征 | 物理理论 | Phase |
|------|------|---------|-------|
| 原子/键 | RBP_Mf_corrected (修正柔性键) | Schneider-DiMarzio M/f | B |
| 重复单元 | GC_Tg (基团贡献预测), HBond 5d (精简) | Van Krevelen | B |
| 链段 | curl_ratio, Neff_ratio, conf_strain (3-mer 构象采样) | Gibbs-DiMarzio, Adam-Gibbs | C |
| 聚合物链 | GNN 预训练嵌入 64d (CV 内微调) | 周期性结构 | D |
| 化学统计 | polyBERT PCA 64d | 语言模型 | D |

### 实施路线
```
Phase A (Day 1):  数据诊断 + 清洗
Phase B (Day 2):  尺度 1-2 特征 (RBP, GC, HBond, FV)
Phase C (Day 3-5): 尺度 3 链段物理 (curl_ratio, Neff_ratio, conf_strain) — ✅ 完成
Phase D (Day 6-8): 尺度 4-5 预训练嵌入 (GNN, polyBERT) — A800
Phase E (Day 9-12): 物理专家委员会集成 + 消融
```

### 模型架构: 物理专家委员会
- 专家 1: 链柔性 (CatBoost + 单调约束)
- 专家 2: 分子间力 (CatBoost + 单调约束)
- 专家 3: 物理基线 (Ridge / GC 残差模型)
- 专家 4: 全特征 (TabPFN)
- 专家 5: 图结构 (GNN 嵌入 → CatBoost)
- 专家 6: 语言模型 (polyBERT → CatBoost)
- 元学习器: ElasticNet

### 关键技术要点
- GC Yg 值: 已校准 (酰胺 42→25, 硅氧烷 -5→6, Cl 12→19), r=0.554 with Tg
- MMFF94: 47% 构象在 1 kcal/mol 内 (Halgren)，芳基酰胺势垒过高 (Guba 2019)
- Cn_proxy: 用图直径 (最远重原子对) 找链端，在 AddHs 之前做
- GNN 预训练: 只用外部数据 (NeurIPS + PolyMetriX ~10-15K)，不含 unified_tg
- GNN 微调: 在 Nested CV 每个 outer fold 内做，零泄漏
- 核苷酸迁移: 独立赛道，用 L1H 34d 特征集

## 已锁定决策
- 评估: Nested CV (outer=RepeatedKFold(5,3), inner=KFold(3))
- 预处理: PowerTransformer(Yeo-Johnson) + MinMaxScaler
- UQ: CrossConformal CatBoost (MAPIE), 90% 覆盖率
- 核苷酸特征集: L1H 34d (不受新特征影响)

## 关键文件映射 (测试)
| 源文件 | 测试文件 |
|--------|----------|
| `src/features/afsordeh_features.py` | `tests/test_afsordeh.py` |
| `src/features/rdkit_descriptors.py` | `tests/test_rdkit_descriptors.py` |
| `src/features/feature_pipeline.py` | `tests/test_feature_pipeline.py` |
| `src/features/gc_tg.py` | `tests/test_gc_tg.py` |
| `src/features/hbond_features.py` | `tests/test_hbond_features.py` |
| `src/ml/evaluation.py` | `tests/test_evaluation.py` |
| `src/data/fox_copolymer_generator.py` | `tests/test_fox_generator.py` |
| `src/features/physical_proxy.py` | `tests/test_physical_proxy.py` |
| `src/features/virtual_polymerization.py` | `tests/test_virtual_poly.py` |
| `src/ml/hierarchical_model.py` | `tests/test_hierarchical.py` |
| `src/gnn/graph_builder.py` | `tests/test_graph_builder.py` |
| `src/gnn/tandem_m2m.py` | `tests/test_tandem_m2m.py` (requires PyG) |

## 编码规范
- 纯函数优先, 不可变数据
- 200-400 行/文件, 800 行上限
- 所有输出中文, 代码英文
- Conventional commits: feat/fix/refactor/test
- 不修改 `src/bigsmiles/` 和 `src/ml/models.py` 和 `src/ml/experiment.py`

## 常用命令
```bash
# 运行特定测试
python -m unittest tests/test_afsordeh.py -v

# 运行全部测试
python -m unittest discover tests/ -v

# 运行实验 (Phase 5)
python scripts/exp_phase5.py

# 核酸序列预测
python scripts/predict_tg_from_sequence.py --seq ACGT --type DNA
```

## 文档导航
- **新人入门**: `docs/课题组新人入门指南.md`
- **文档总目录**: `docs/INDEX.md`
- **实验总览**: `docs/experiments/实验总览.md`
- **方案A总结**: `results/方案A总结.md`

## 实验结果记录
每个实验结果保存到 `results/phaseN/` 目录:
- Phase 1-3: `exp_X_Y_description.json`
- Phase 4-5: `E{N}_description.json` (统一编号 E1-E26)
- 格式: `{"R2_mean": ..., "R2_std": ..., "MAE_mean": ..., "features": [...], "model": "..."}`
- 统一索引: `docs/experiments/实验总览.md`

## 文档维护协议 (MANDATORY)

### "河流与河床"模型
- **河床（不变）**：目录结构、overview/01+02、调研原文 — 几乎不改
- **河水（流动）**：overview/03、实验总览 — 随实验推进持续更新

### 何时更新文档
| 触发事件 | 需要更新的文档 |
|---------|---------------|
| 新实验完成 | `docs/experiments/实验总览.md` (追加行) |
| 刷新最优指标 | `docs/overview/03-当前进展与成果.md` |
| 新调研完成 | `docs/research/{分类}/` + `docs/research/_index.md` + `docs/INDEX.md` |
| 里程碑完成 | `results/方案A总结.md` + CLAUDE.md 当前状态 |

### 新文档模板
- 调研报告: `docs/templates/调研报告模板.md`
- 实验记录: `docs/templates/实验记录模板.md`

### 调研分类规则
| 目录 | 内容 |
|------|------|
| `research/foundations/` | 基础理论（物理原理、表示方法） |
| `research/methods/` | 方法调研（ML、DL、数据增强） |
| `research/domain/` | 领域专题（核酸、VPD） |
| `research/decisions/` | 决策记录（失败分析、方案选择） |
| `research/other/` | 非主线（竞赛等） |
| `research/_archive/` | 被合并的原文件 |

### 项目级 hooks
- **PostToolUse (Bash)**: `doc-sync-check.py` — 检测实验脚本执行，提醒更新文档
- **Stop**: `doc-freshness-audit.py` — 会话结束时检查文档是否与最新结果同步
