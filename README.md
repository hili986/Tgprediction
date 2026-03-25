# AI 辅助高分子材料设计 — Tg 智能预测

> 同济大学 SITP 大学生创新训练计划

从分子结构 (SMILES) 预测高分子玻璃化转变温度 (Tg)，并迁移预测核酸 Tg。

## 核心指标

| 场景 | 模型 | 性能 |
|------|------|------|
| 通用聚合物 (7,486 样本) | TabPFN v2 | R²=0.8955, MAE=24K |
| 核苷酸迁移 (ATP/ADP) | CatBoost | 误差 0.4~4.7K |
| 特征压缩 | SHAP Top 15/46 | 零精度损失 |

## 创新点

1. **VPD (虚拟聚合描述符)** — 捕捉聚合效应，Tg 预测提升 4.6%
2. **桥梁聚合物迁移** — 首次从聚合物数据预测核酸 Tg
3. **Top 15 帕累托特征** — 1/3 特征量，同等精度

## 快速开始

```bash
pip install numpy pandas scikit-learn shap matplotlib rdkit catboost lightgbm xgboost
python -m unittest discover tests/ -v
```

## 文档导航

| 你想... | 去看 |
|---------|------|
| 30 秒了解项目 | 本文件 |
| 新人全面入门 | [`docs/课题组新人入门指南.md`](docs/课题组新人入门指南.md) |
| 浏览所有文档 | [`docs/INDEX.md`](docs/INDEX.md) |
| 查实验结果 | [`docs/experiments/实验总览.md`](docs/experiments/实验总览.md) |
| 看方案A完整总结 | [`results/方案A总结.md`](results/方案A总结.md) |

## 项目结构

```
src/
├── data/         数据加载 (Bicerano 304, 统一 7,486, 桥梁 205, Fox 共聚物)
├── features/     特征工程 (M2M-V 46维: Afsordeh + RDKit + HBond + VPD)
├── ml/           模型+评估+UQ (Nested CV, Stacking, MAPIE)
├── gnn/          图神经网络 (方案B, 待 GPU 启动)
├── bigsmiles/    BigSMILES 工具链 (已完成, 冻结)
└── sequence/     核酸序列处理

docs/
├── INDEX.md          文档总目录
├── overview/         项目全貌 (背景/技术路线/进展)
├── research/         调研报告库 (18份, 按主题分类)
├── experiments/      实验统一索引
└── plans/            实验计划

results/              实验输出 (Phase 1-5, JSON + summary)
scripts/              实验脚本 (12个)
tests/                单元测试 (14文件)
```

## 技术栈

Python | RDKit | scikit-learn | CatBoost | XGBoost | LightGBM | TabPFN | SHAP | MAPIE | PyTorch (GNN)

## 许可证

本项目为同济大学 SITP 学术研究项目，仅限学术用途。
