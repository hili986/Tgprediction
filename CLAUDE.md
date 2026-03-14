# Tg 预测项目 — CLAUDE.md

## 项目概述
同济大学 SITP: AI 辅助高分子材料设计 — 预测玻璃化转变温度 (Tg)
从 304 条均聚物基线 (R²=0.85) → 目标 R²>0.95 + 核酸 Tg 预测

## 技术栈
- Python 3.x (use `python` not `python3`)
- scikit-learn, numpy, pandas, shap, matplotlib
- RDKit (已安装 2025.09.4)
- SQLite (阶段三)

## 项目结构
```
src/
  data/           → 数据集 (bicerano_tg_dataset.py, external_datasets.py)
  features/       → 特征工程 (afsordeh, rdkit, hbond, pipeline, physical_proxy, virtual_polymerization)
  ml/             → 模型 (sklearn_models, evaluation, constrained_gbr, hierarchical_model, gnn_evaluation)
  gnn/            → GNN 子包 (graph_builder, physics_gat, tandem_m2m, pretrainer, multitask, ensemble)
  analysis/       → 分析 (shap, ablation, visualization)
  bigsmiles/      → BigSMILES 工具链 (已完成, 不修改)
tests/            → 测试
scripts/          → 实验脚本 (exp_phase4_m2m.py, exp_phase4_gnn.py)
data/             → 原始数据文件
docs/plans/       → 实验计划
docs/research/    → 6 份调研报告
results/          → 实验输出 (phase1-4/)
```

## 当前执行计划
**精简计划**: `docs/plans/2026-03-12-Tg预测实验计划.md`
**详细方案**: `docs/plans/2026-03-12-Tg预测实验计划.DRAFT.md` (V4, 含完整代码模板)

## 已锁定决策
- 主模型: ExtraTrees (SOTA for small datasets)
- 评估: Nested CV (outer=RepeatedKFold(5,3), inner=KFold(3))
- 预处理: PowerTransformer(Yeo-Johnson) + MinMaxScaler
- 特征选择: VarianceThreshold → Boruta → mRMR → SHAP
- 集成: Stacking(ExtraTrees+GBR+SVR → Ridge)
- 多保真度: 两阶段训练（预训练→微调）
- 迁移: 特征迁移（加权训练）
- 调参: 自动搜索 10-20 组

## 关键文件映射 (测试)
| 源文件 | 测试文件 |
|--------|----------|
| `src/features/afsordeh_features.py` | `tests/test_afsordeh.py` |
| `src/features/rdkit_descriptors.py` | `tests/test_rdkit_descriptors.py` |
| `src/features/feature_pipeline.py` | `tests/test_feature_pipeline.py` |
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

# 运行实验
python -m src.ml.experiment_v2
```

## 实验结果记录
每个实验结果保存到 `results/phaseN/` 目录:
- `exp_1_1_l0_only.json` — L0 Afsordeh 4 特征
- `exp_1_2_l1.json` — L0+L1
- 格式: `{"R2_mean": ..., "R2_std": ..., "MAE_mean": ..., "features": [...], "model": "..."}`
