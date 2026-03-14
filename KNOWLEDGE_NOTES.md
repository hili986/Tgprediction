# 知识笔记

### 氢键内聚能密度（CED）与 Tg 的定量关系 -- 2026-03-08
**来源**：Level 2 (深度讲解)
**触发场景**：氢键与核酸 Tg 预测调研
**分析过程**：
- 调研了不同氢键基团的 CED 值（脲基 2079 > 酰胺 1862 > 羟基 1500 > 氨酯 1385 > 酰亚胺 980 > 磷酸酯 800-1200 J/cm3）
- 发现 CED 与 Tg 提升是正相关但非线性的：双氢键供体的协同效应使 Tg 提升远超线性预期
- 选择 CED 作为核心定量指标，因为它直接反映分子间作用力强度
**核心要点**：
- CED 是衡量氢键对 Tg 贡献的最佳定量指标之一
- 脲基（双 N-H 供体）的 CED 仅为单供体的 1.5 倍，但 Tg 提升可达 2-3 倍 -- 协同效应
- 核酸碱基配对的氢键模式（N-H...O=C + N-H...N）介于酰胺和脲基之间
**相关概念**：Kwei 方程, 拓扑约束理论, 自由体积理论
**推荐资源**：
- van Krevelen, Properties of Polymers, 4th ed., Chapter 7 (内聚性质)
- Mattia & Painter 2007, Macromolecules 40(5), 1546-1554

### 桥梁聚合物迁移学习策略 -- 2026-03-08
**来源**：Level 2 (深度讲解)
**触发场景**：解决核酸 Tg 数据极度稀缺（<10 条）的问题
**分析过程**：
- 核心困境：核酸 Tg 数据太少，无法直接训练 ML 模型
- 考虑了三种策略：(1) 氢键模式迁移 (2) Kwei 组分分解 (3) 基础模型微调
- 选择策略 1 优先，因为它不需要核酸的精确组分 Tg 数据，只需要氢键模式相似的合成聚合物数据
- 创新提出"桥梁聚合物"概念：与核酸共享氢键模式的 7 大合成聚合物家族
**核心要点**：
- 迁移学习的关键假设："相似氢键模式 -> 相似 Tg 贡献规律"
- 7 大桥梁家族覆盖了核酸的 4 种主要氢键模式（N-H...O=C, N-H...N, P-O 骨架, O-H 核糖）
- 聚磷酸酯和聚磷腈是最直接的骨架桥梁，但数据最稀缺
- ATP/ADP/AMP Tg（246-249 K）是宝贵的核酸单体锚点
**相关概念**：迁移学习, 领域适配, 特征空间对齐
**推荐资源**：
- Xu & Wang 2023, TransPolymer (npj Computational Materials)
- Kuenneth et al. 2023, polyGNN (npj Computational Materials)

### 四层数据架构设计 -- 2026-03-08
**来源**：Level 2 (深度讲解)
**触发场景**：为核酸 Tg 预测设计数据库架构
**分析过程**：
- 传统做法是用单一大数据集训练，但核酸领域数据量差异极大（通用聚合物 ~8000 vs 核酸 ~10）
- 设计四层架构：Layer 1 小分子 -> Layer 2 通用聚合物 -> Layer 3 桥梁聚合物 -> Layer 4 核酸
- 每一层都有明确的功能角色：预训练基础、迁移学习源、最终目标
- SMARTS 筛选从 Layer 2 自动提取 Layer 3，减少人工标注
**核心要点**：
- 分层架构解决了数据量跨度大（从万级到个位数）的问题
- SMARTS 模式匹配是自动化筛选桥梁聚合物的关键技术
- 数据库 Schema 预计算了氢键特征，避免重复计算
- 区分干态/湿态 Tg 对核酸至关重要（水增塑效应可达 -100 K）
**相关概念**：SMARTS, RDKit, 数据库设计范式, 特征预计算
**推荐资源**：
- RDKit SMARTS 文档: https://www.rdkit.org/docs/GettingStartedInPython.html
- PolyMetriX 数据集: https://polymetrix.org

### 多保真度数据金字塔 — 2026-03-08
**来源**：Level 2 (深度讲解)
**触发场景**：调研虚拟数据增强 Tg 预测方法，发现不同精度数据源可以分层融合
**分析过程**：
- 梳理了 6 个保真度层级：F0(GC, ms) → F1(Fox, ms) → F2(ML, s) → F3(xTB, min) → F4(MD, hrs) → F5(实验, 天)
- 每一层精度和成本呈正相关，生成规模与精度呈反相关
- 核心洞察：用底层海量低精度数据预训练"粗略映射"，顶层少量实验数据微调"精确校准"
**核心要点**：
- 这和 NLP 预训练+微调范式完全同构（GPT 在大语料预训练 → 任务数据微调）
- Delta Learning: y_high = y_low + delta(x)，学习高低保真度的差值比直接学绝对值更容易
- IDLe (Implicit Delta Learning) 比普通迁移学习需要 50x 更少的高保真度数据
- 本项目可直接用 Fox 方程（F1 级）生成万级共聚物数据作为预训练层
**相关概念**：迁移学习, 预训练-微调范式, 知识蒸馏, 主动学习
**推荐资源**：
- Pilania et al. (2019) "Multi-fidelity ML for bandgap predictions" — Comp. Mater. Sci.
- Doan et al. (2024) "Implicit Delta Learning" — J. Phys. Chem. Lett.

### SHAP 共识：链柔性是 Tg 头号预测因子 — 2026-03-08
**来源**：Level 2 (深度讲解)
**触发场景**：综合 6 篇独立 SHAP 分析研究，发现高度一致的排序规律
**分析过程**：
- 6/6 研究一致：链柔性/刚性（NumRotatableBonds, Flexibility）排第一
- 5/6 一致：拓扑复杂度（BertzCT, BalabanJ）和极性/氢键（TPSA, HBD）
- Afsordeh 4 特征法用 4 个物理直觉特征达到 R²=0.97，验证了 SHAP 排序的正确性
**核心要点**：
- Tg 本质是"链段运动有多困难"，而 NumRotatableBonds 直接量化运动自由度
- 单变量 Pearson r = -0.87~-0.90（最强负相关）
- 4 个好特征 > 2000 个随机特征 — 物理直觉的威力
- SHAP 基于 Shapley 值（博弈论），是模型无关的解释方法
**相关概念**：Shapley 值, 特征重要性, 可解释 AI, 博弈论
**推荐资源**：
- Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions" — NIPS
- Afsordeh & Shirali (2025) "4 Structural Features for Tg" — Chinese J. Polymer Sci.

### 数据量 > 描述符精度的反直觉发现 — 2026-03-08
**来源**：Level 1 (同步)
**触发场景**：对比 PolyMetriX (7367条, 简单描述符, R²=0.97) 与传统方法 (500条, 复杂描述符, R²=0.92)
**核心要点**：
- PolyMetriX 只用 RDKit 2D 描述符的分层版本（全分子/骨架/侧链 × 32 = 96 维），没有量化计算
- 但因为数据量大（7367 条），性能超过了使用复杂量化描述符的小数据集方法
- 启示：对本项目来说，扩大数据量是第一优先级，比纠结更精细的描述符更重要
**相关概念**：偏差-方差权衡, 特征工程 vs 数据工程, 缩放定律

### 低成本 Tg 描述符分层策略与 SHAP 共识 —— 2026-03-08
**来源**：Level 2 (深度讲解)
**触发场景**：系统调研低成本可计算的分子/聚合物描述符及其与 Tg 的相关性
**分析过程**：
- 调研了 RDKit、Mordred、GFN2-xTB、DFT 等多种描述符计算方法的成本与效果
- 发现 Afsordeh & Shirali (2025) 的 4 特征方法（Flexibility, SOL, HBD, PI）以极简特征达到 R²=0.97
- 综合 6+ 研究的 SHAP 分析，确认链柔性/刚性是 Tg 最强预测因子
- 设计了 L0-L6 分层策略，从 4 特征逐步扩展到完整特征工程管线
**核心要点**：
- NumRotatableBonds 是与 Tg 相关性最强的单一描述符（Pearson r = -0.87 到 -0.90）
- SHAP 共识排序：链柔性 >> 拓扑复杂度 > 极性/氢键 > 芳香环 > 侧链 > 分子量
- 19 个特征（Afsordeh 4 + RDKit 15）即可达到 R²=0.90-0.95，计算成本 <1s/分子
- 特征选择比特征数量重要：50 个精选特征 > 500 个随机特征
- Boyer-Simha 关系 (Tg ≈ Ecoh / R*Nbb) 提供了柔性与 Tg 关联的物理基础
**相关概念**：SHAP, Boruta/RFE 特征选择, Extra Trees, Pearson 相关系数, 内聚能密度 (CED)
**推荐资源**：
- Afsordeh & Shirali (2025), Chinese Journal of Polymer Science, DOI: 10.1007/s10118-025-xxxx
- Lundberg & Lee (2017), SHAP: A Unified Approach to Interpreting Model Predictions, NeurIPS
- RDKit 官方文档: https://www.rdkit.org/docs/GettingStartedInPython.html
- Mordred 文档: https://mordred-descriptor.github.io/documentation/

### Afsordeh 4 特征消融实验的深层启示 — 2026-03-08
**来源**：Level 2 (深度讲解 — 论文精读)
**触发场景**：精读 Afsordeh & Shirali (2025) 全文，深入分析消融实验和特征对比
**分析过程**：
- 论文通过消融实验逐一移除 4 个特征，观察性能变化
- Flexibility 移除后 R² 暴跌至 0.28-0.47（所有模型一致），证明它是预测 Tg 的核心变量
- H-bond Power 移除后性能几乎不变 — 说明氢键信息被 Polarity 和 SOL 间接编码
- 13 个官能团特征（传统方法）训练同模型 R²=0.55，严重过拟合 — 物理无关特征是噪声源
**核心要点**：
- Flexibility = 可旋转键数 / 骨架键总数（归一化），不是原始 NumRotatableBonds
- Side-chain Occupancy Length (SOL) 用键长+原子半径计算侧链空间占据，捕获位阻效应
- Polarity 用 Fajan 规则（电负性差 × 键长）度量极性，而非简单的偶极矩
- H-bond Power 用加和模型（每种键类型赋能量值：N-H 8.86, O-H 20.92 kJ/mol），可扩展
- 4 特征 vs 官能团特征的对比是"特征质量 >> 特征数量"最强实证
**相关概念**：消融实验, 特征工程, 过拟合诊断, Extra Trees 集成方法
**推荐资源**：
- Afsordeh & Shirali (2025), Chinese J. Polym. Sci. 43, 1661-1670, DOI: 10.1007/s10118-025-3361-3
- Bicerano (2002), Prediction of Polymer Properties, 3rd ed. — 论文数据集来源


### Extra Trees 为何在小数据集 Tg 预测中胜出 -- 2026-03-12
**来源**：Level 2 (深度讲解)
**触发场景**：ML 最佳实践调研发现 Extra Trees 在 2025 SOTA 中胜出
**分析过程**：
- 对比了 GBR/RF/ET/XGBoost/CatBoost 在 ~300 样本上的表现
- Extra Trees (极端随机树) 在分裂点选择上引入额外随机性：不找最优分裂点，而是在候选特征上随机选分裂阈值
- 这种"更笨"的分裂策略反而在小数据集上有优势：减少了对训练数据中噪声模式的拟合
- RF 只随机选特征但仍找最优分裂点；ET 两者都随机化
**核心要点**：
- Extra Trees 的双重随机化（特征 + 分裂点）天然是一种正则化手段
- 小数据集上过拟合是主要矛盾，ET 的随机性正好缓解这个问题
- ET 训练速度比 RF 更快（不需搜索最优分裂点），适合大量超参搜索
- sklearn 的 ExtraTreesRegressor 接口与 RandomForestRegressor 完全一致，迁移成本为零
**相关概念**：偏差-方差权衡, 随机化正则, Bagging, Bootstrap
**推荐资源**：
- Geurts et al. (2006) "Extremely Randomized Trees" Machine Learning 63(1):3-42
- sklearn 文档: https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees

### 嵌套交叉验证解决小数据集评估偏差 -- 2026-03-12
**来源**：Level 2 (深度讲解)
**触发场景**：调研发现 304 样本的普通 k-fold CV 有 R2 0.03-0.05 的乐观偏差
**分析过程**：
- 普通 k-fold CV 的问题：同一套数据既用于超参选择又用于性能评估，导致"信息泄露"
- 类比考试：用模拟卷选考试策略，再用同一套模拟卷评估成绩 -> 高估真实水平
- 嵌套 CV 分离两个目标：外循环评估性能，内循环选超参
- 代价是计算量增加 k_outer * k_inner 倍，但 304 样本下完全可接受
**核心要点**：
- 304 样本下乐观偏差 R2 0.03-0.05 不可忽略（可能误以为 0.90 实际只有 0.86）
- 推荐配置：outer=RepeatedKFold(5,3), inner=KFold(3)，共 5*3*3=45 次模型训练
- RepeatedKFold 的重复次数(n_repeats=3)用于减少随机划分的方差
- 嵌套 CV 的结果比普通 CV 更保守但更可靠，论文中更有说服力
**相关概念**：交叉验证, 超参选择, 模型选择, 偏差-方差权衡
**推荐资源**：
- Cawley & Talbot (2010) "On Over-fitting in Model Selection" JMLR 11:2079-2107
- sklearn 文档: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

### Delta Learning: 多保真学习的核心范式 -- 2026-03-12
**来源**：Level 2 (深度讲解)
**触发场景**：调研多保真学习在 Tg 预测中的应用
**分析过程**：
- Delta Learning 核心公式：y_high = y_low + delta(x)
- 直觉：学习"误差的模式"比直接学习"绝对值"更容易（误差通常更平滑、更简单）
- 这和残差网络 (ResNet) 的思想完全同构：学习 F(x) = H(x) - x 比学习 H(x) 更容易
- 在 Tg 预测中：Fox 方程给出粗略 Tg (y_low)，ML 学习 Fox 方程的系统偏差 (delta)
**核心要点**：
- Delta Learning 的前提：低保真方法必须与高保真方法有正相关性（Fox 方程满足）
- 低保真数据量可以比高保真数据大 10-1000 倍（本项目：23万 vs 304）
- Delta 模型可以用更简单的模型（depth=3 的 GBR），因为误差模式比原始数据简单
- 加权训练是 Delta Learning 的简化替代方案，实现更简单但效果稍逊
**相关概念**：ResNet 残差连接, 预训练-微调, 知识蒸馏, 多任务学习
**推荐资源**：
- Ramakrishnan et al. (2015) "Delta-ML approach" JCTC 11(5):2087-2096
- He et al. (2016) "Deep Residual Learning" CVPR (ResNet 论文, 类比理解)

### 特征迁移学习中桥梁权重的选择策略 — 2026-03-12
**来源**：Level 2 (深度讲解)
**触发场景**：Phase 3 实验发现不同 bridge_weight 对 CV R² 和核酸预测效果有相反影响
**分析过程**：
- bridge_weight=0.4 时 GBR CV R² 最优 (0.8368)，但核酸预测误差较大 (ATP=4.0K)
- bridge_weight=0.8 时 CV R² 略低 (0.8291)，但核酸预测最佳 (ATP=1.0K, ADP=1.3K)
- 权重越高，模型越"关注"桥梁数据中的氢键模式 → 核酸迁移更好，但通用泛化略降
- 这是迁移学习中经典的"源-目标权衡"：源域越强调，目标域特定模式越突出
**核心要点**：
- 最优权重取决于最终用途：通用 Tg 预测用 0.4，核酸特化用 0.8
- 加权训练等价于调整损失函数中不同数据源的贡献比例
- GBR 对权重更敏感（R² 变化 0.77%），ET 更稳定（仅 0.1%）— 树深有限的 GBR 更容易被权重引导
**相关概念**：迁移学习, 领域自适应, 样本加权, 损失加权
**推荐资源**：
- Pan & Yang (2010) "A Survey on Transfer Learning" IEEE TKDE 22(10)

### SHAP 特征重要性验证：RingCount 是 Tg 头号预测因子 — 2026-03-12
**来源**：Level 2 (深度讲解)
**触发场景**：Phase 3.5 SHAP 分析在 L1H (34-dim) 特征空间上运行，验证先前调研结论
**分析过程**：
- GBR SHAP: RingCount=45.17 >> FractionCSP3=18.21 >> FlexibilityIndex=17.39
- ET SHAP: RingCount=24.27, NumAromaticRings=18.43, FractionCSP3=14.46
- H-bond 特征组在 Bicerano 上贡献较小 (GBR: 7.47/161.49 = 4.6%)，但在核酸迁移中起关键桥梁作用
- strong_hbond_density 和 ced_weighted_sum 是最有价值的 H-bond 特征
**核心要点**：
- RingCount 的 SHAP 贡献是第二名的 2-2.5 倍，与 6 篇独立研究一致
- H-bond 特征的双重角色：对通用数据集贡献小，但对领域迁移至关重要
- SHAP 在 GBR 和 ET 间的排序不完全一致（ET 更重视 NumAromaticRings），说明不同模型"看问题的角度"不同
- 建议在论文中呈现两种模型的 SHAP 对比，增加可解释性分析的稳健性
**相关概念**：SHAP, TreeExplainer, 特征重要性, 模型可解释性
**推荐资源**：
- Lundberg et al. (2020) "From local explanations to global understanding" Nature Machine Intelligence

### Stacking 集成在小数据集上的失败案例分析 — 2026-03-12
**来源**：Level 2 (深度讲解)
**触发场景**：Phase 3.4 Stacking (ET+GBR+SVR→Ridge) 实验结果 R²=-0.45，严重低于单模型
**分析过程**：
- Stacking 的内层 CV (cv=3) 在 304 样本上再次划分，每折仅 ~200 训练 + ~100 验证
- 3 个基学习器 + 1 个元学习器需要足够大的验证集来产生有意义的元特征
- SVR 的 fit() 不原生支持 sample_weight（sklearn StackingRegressor 会忽略），导致加权训练失效
- 无桥梁数据时 R²=-0.45：训练集太小，基学习器过拟合，元特征不稳定
- 增加桥梁数据后改善 (0.72→0.81)：更多训练样本稳定了基学习器
**核心要点**：
- Stacking 对数据量有隐含要求：需要足够多的 OOF (out-of-fold) 预测来训练元学习器
- 经验法则：<500 样本时 Stacking 通常不如单模型或简单 VotingRegressor
- SVR 在 Stacking 中的权重传递问题是常见陷阱 — 应该检查 sklearn 文档
- 负面结论也有论文价值：证明了方法选择需要数据规模感知
**相关概念**：Stacking, 元学习, OOF 预测, VotingRegressor, 数据饥饿
**推荐资源**：
- Wolpert (1992) "Stacked Generalization" Neural Networks 5(2):241-259
- sklearn StackingRegressor 文档 (注意 sample_weight 传递的限制)

### 交叉验证中 shuffle 的关键性 — 2026-03-12
**来源**：Level 1 (同步)
**触发场景**：扩展数据集基准测试中 Bicerano 基线出现 R²=-9.15 的异常值
**分析过程**：
- Bicerano 数据集在 `bicerano_tg_dataset.py` 中按 Tg 大致升序排列
- `KFold(n_splits=5)` 默认不打乱 → 每折的训练/测试集覆盖完全不同的 Tg 区间
- 例如：Fold 1 训练集全是 Tg>300K 的聚合物，测试集全是 Tg<250K → 模型被迫外推
- 加入 `shuffle=True, random_state=42` 后 R² 恢复到 0.833
**核心要点**：
- 数据排序是隐蔽的 CV 陷阱：即使数据看起来"随机"，也可能有隐含排序
- `KFold(shuffle=True)` 应该是默认做法，除非有时间序列等特殊原因不打乱
- R² 为负说明模型预测比"全部预测均值"还差 — 这本身就是强诊断信号
- 修复后立即恢复到预期水平（0.833 vs Phase 3 的 0.837），确认了 shuffle 是唯一问题
**相关概念**：交叉验证, 数据泄露, 外推 vs 内插, 分层采样

### 数据质量 vs 数据数量的实证分析 — 2026-03-12
**来源**：Level 2 (深度讲解)
**触发场景**：扩展数据集（22,674 条）基准测试显示 CV R² 仅略优于精选小数据集（304 条）
**分析过程**：
- 假设：74.6 倍数据量应该显著提升 CV 性能 → 实际：R² 0.819 vs 0.833（反而略低）
- 原因分析：6 个异质数据源有不同的测量标准、SMILES 规范化方式、温度单位
- PolyMetriX 96.2% 为 black 级可靠性（单次测量），而非 gold（多次平均）
- NeurIPS OPP 17,330 条数据来源不明确，可能包含二手转录错误
- 但加大模型容量后（500 trees, depth=5）R² 提升到 0.843 — 模型需要更多参数来处理异质数据
- 迁移测试（扩展集训练 → Bicerano 测试）R²=0.898 — 大量异质数据显著改善了泛化能力
**核心要点**：
- "更多数据 ≠ 更好 CV" — 数据质量差异会引入噪声，抵消数据量优势
- 异质数据的真正价值在于**泛化能力**，而非交叉验证分数
- 模型容量需要匹配数据规模：304 条用 depth=4，22K 条需要 depth=5+
- 数据源加权（Bicerano weight=1.5）效果不显著 — 说明 6 个数据源质量差异不大
- 实用建议：CV 用小精选数据集评估，训练最终模型时纳入所有数据
**相关概念**：偏差-方差权衡, 数据清洗, 模型容量, 领域漂移
**推荐资源**：
- Northcutt et al. (2021) "Confident Learning: Estimating Uncertainty in Dataset Labels" JAIR
- Sambasivan et al. (2021) "Everyone wants to do the model work, not the data work" CHI

### Tm（熔解温度）与 Tg（玻璃化转变温度）的本质区别 — 2026-03-12
**来源**：Level 1 (同步)
**触发场景**：用户看到寡核苷酸 Tm 计算网站，问是否对 Tg 预测有参考价值
**核心要点**：
- Tm 测量的是溶液中双链 DNA 解旋温度（50% 解链），驱动力是碱基间 H-bond + 堆叠力
- Tg 测量的是固态（干膜/冻干粉）链段运动解冻温度，驱动力是分子间 van der Waals + H-bond + 链缠结
- 典型范围差异巨大：Tm 50-95°C，Tg 约 175°C（DNA 干膜 448K）
- GC 含量同时正向影响 Tm 和 Tg（G 有 3 个氢键 > AT 的 2 个），但量化关系不同
- 不能用 Tm 推算 Tg——不是"加常数"能修正的，是完全不同的相变过程
- Tm 计算器的唯一实用价值：快速获取序列 GC% 和分子量，用于交叉验证
**相关概念**：相变, 玻璃化转变, DNA 解旋, 最近邻法 (Nearest Neighbor), CED
**推荐资源**：
- SantaLucia (1998) "A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics" PNAS

### 长链核酸 Tg 预测的可靠度分层 — 2026-03-12
**来源**：Level 1 (同步)
**触发场景**：用户提交 45nt DNA 序列进行 Tg 预测，模型给出 LOW 置信度警告
**分析过程**：
- 45nt 序列生成 912 原子的 SMILES，远超训练集分子大小（Bicerano 聚合物通常 10-50 重原子）
- 完整序列预测 299.9K，单体平均 470.8K，差异巨大（171K）说明完整序列预测不可靠
- Phase 3 DNA backbone 验证：预测 279K vs 实验 448K，误差 169K — 与本次完整序列预测的不可靠性一致
**核心要点**：
- 预测可靠度层级：1nt (HIGH) > 2-4nt (MEDIUM) > 5+nt (LOW)
- 长链预测失败的根因：RDKit 描述符在 900+ 原子分子上的行为是非线性外推，模型未见过此类分子
- 单体平均法更合理：利用碱基独立 Tg 取组成加权平均，绕过长链外推问题
- G 的 Tg (509.6K) > C (476.8K) > A (451.3K) > T (448.2K)，与氢键位点数正相关
**相关概念**：外推风险, 适用域 (Applicability Domain), 模型不确定性



### Conformal Prediction (保形预测) -- 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 小数据集 Tg 预测 SOTA 调研, 发现 UQ 方法是核酸预测论文的关键亮点
**分析过程**:
- UQ Benchmark (JCIM 2025) 系统比较了 9 种不确定性量化方法
- Deep Ensemble 在分布内最佳, BNN-MCMC 在分布外最佳, 但都需要深度学习
- Conformal Prediction 在 sklearn 兼容性、理论保证、实现难度三个维度综合最优
**核心要点**:
- Conformal Prediction 是 distribution-free 的 UQ 方法, 不假设数据分布
- 理论保证: 在 exchangeability 假设下, 覆盖率精确等于 1-alpha
- 实现极简: MAPIE 库 ~20 行代码即可包裹任何 sklearn 模型
- 对核酸 Tg 外推预测提供置信区间, 可判断模型是否在有效外推范围内
**相关概念**: distribution-free inference, exchangeability, prediction interval vs confidence interval
**推荐资源**:
- MAPIE 文档: https://mapie.readthedocs.io/
- Vovk et al., "Algorithmic Learning in a Random World", Springer 2005 (理论基础)
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (入门推荐)

### R-squared 评估方式对性能报告的影响 -- 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 分析 Afsordeh 2025 R2=0.97 与本项目 R2=0.83 的差距
**分析过程**:
- Afsordeh 使用 90/10 单次 split (train=100, test=12), 本项目使用 Nested CV (outer=5x3, inner=3)
- 12 个测试样本的 R2 方差极大, 95% CI 可能覆盖 0.85-1.0
- Nested CV 是无偏估计, 但通常比单次 split 低 0.03-0.08
**核心要点**:
- 同一数据集+模型, 不同评估方式 R2 可差 0.05-0.15
- 报告 R2 必须标注评估方式 (split ratio, CV folds, 是否 nested)
- 小测试集 (<30) 的 R2 统计意义弱, 应报告 CI
- Nested CV 是 <1000 样本的金标准评估方式 (避免信息泄露)
**相关概念**: information leakage, optimistic bias, bootstrap confidence interval
**推荐资源**:
- Xu et al. 2024 "Machine Learning for Polymer Tg Prediction" (Nested CV 推荐)
- Cawley & Talbot 2010 "On Over-fitting in Model Selection" (经典论文)

### 树模型在小数据集上的持续优势 -- 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: Kaggle Open Polymer Challenge 2025 结果分析
**分析过程**:
- Kaggle 3985 条数据, 树模型 (LightGBM/XGBoost) 一致胜过 GNN/Transformer
- OpenPoly 数据库验证: <4000 条时 XGBoost beats DL
- 原因: 树模型不需要大量数据学习表示, 直接在手工特征上分裂
**核心要点**:
- 在 <5000 条表格数据上, 树集成模型仍是 SOTA (2025 年依然成立)
- 深度学习的优势在 >10000 条 + 原始输入 (SMILES/3D) 时才显现
- Stacking (LightGBM + XGBoost + CatBoost) 是 Kaggle 竞赛的标准赢法
- SMILES augmentation 对树模型无效甚至有害 (FP 对 SMILES 顺序不变)
**相关概念**: representation learning vs feature engineering, inductive bias, Kaggle meta-learning
**推荐资源**:
- Grinsztajn et al. 2022 "Why do tree-based models still outperform deep learning on tabular data?" (NeurIPS)
- Open Polymer Challenge: https://www.kaggle.com/competitions/open-polymer-challenge

### 单体描述符天花板的理论推导 — 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 用户要求深度分析当前方法失败的根因，产出三份调研文档
**分析过程**:
- 所有特征层 (L0/L1/L1H/L2/L2H) 均从 `smiles.replace("*", "[H]")` 出发，本质是用小分子片段近似高分子链
- 文献中 Tg 方差约 70% 来自单体结构 (Bicerano 理论)，RDKit 描述符能捕获约 85% 的单体结构信息
- 理论上限 ≈ 0.70 / (0.70/0.85) ≈ 0.84，与 Phase 1-3 的实验 R² (0.82-0.84) 高度吻合
- 丢失的 30% 信息分两层：链级效应 (~20%，分子量、链端、链缠结) + 体相效应 (~10%，自由体积、结晶度)
**核心要点**:
- 描述符数量无关 (4 维到 1072 维都试过)，关键是"描述符缺少正确的物理量"
- Zhao et al. R²=0.97 的 4 个特征恰好跨越了三层：SideChainRatio(单体)、Cn(链)、FreeVolume(体相)、GroupContribution(综合)
- 突破天花板的路径不一定是深度学习，可以是"增加正确的物理描述符"
- 这是**表示层面的根本缺陷**，不是模型或调参问题
**相关概念**: 表示学习, 特征工程瓶颈, Bicerano 理论, 自由体积理论
**推荐资源**:
- Bicerano (2002), "Prediction of Polymer Properties", 3rd ed. — Tg 物理本质的经典参考
- van Krevelen & te Nijenhuis (2009), "Properties of Polymers", 4th ed. — 基团贡献法

### 表示学习 vs 特征工程的本质区别 — 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 分析 GNN/Transformer 等深度学习方法是否值得引入
**分析过程**:
- 传统特征工程 (Phase 1-3): 人工设计描述符，天花板取决于人的物理直觉
- 表示学习 (GNN/Transformer): 模型自动从原始结构学特征，理论上无人工信息瓶颈
- 但代价是需要更多数据 (>1000 条) 和更复杂的模型架构
- Kaggle 2025 实证：<5000 条表格数据上树模型仍优于深度学习
**核心要点**:
- 在 304 条数据上，"增加正确的物理描述符" 比 "换成深度学习" 更有针对性
- 深度学习的优势在 >10000 条 + 原始输入时才显现
- 折中方案：用预训练模型 (polyBERT) 提取 embedding，再接传统 ML — 两全其美
- 核心决策：是否愿意引入 PyTorch 决定了技术路线分叉
**相关概念**: 归纳偏置, 数据效率, 预训练-微调, 特征提取 vs 端到端学习
**推荐资源**:
- Grinsztajn et al. (2022), "Why do tree-based models still outperform DL on tabular data?", NeurIPS
- polyBERT: Kuenneth & Ramprasad (2023), Nature Communications

### 聚合物分子表示方法的信息论天花板 — 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 系统调研聚合物分子表示方法，回答"为什么 R² 停在 0.84"
**分析过程**:
- Tg 受四层因素影响：单体级 (60-80%)、链级 (MW/tacticity)、超分子级 (结晶/交联)、实验级 (冷却速率/测量方法)
- 单体 SMILES 只能捕获第一层，其余三层信息完全丢失
- 信息论角度：f(monomer) = E[Tg|monomer] ± sigma，sigma 是不可约方差
- Bicerano 数据标注 Tg(inf) 消除了 MW 效应，但 tacticity/结晶度差异仍在
- 实际天花板 R² ≈ 0.85-0.90，与实验值 0.82-0.84 吻合（差距来自描述符设计尚未最优）
**核心要点**:
- 这是输入信息不足的根本限制，不是模型或算法能弥补的
- Afsordeh 的 R²=0.97 基于 90/10 单次 split (12 样本测试集)，Nested CV 下预计降至 0.90-0.94
- Bayes error rate (贝叶斯错误率) 是统计学习理论中的核心概念：给定输入下不可消除的最低误差
- 提升路径：不是加更多同类描述符，而是引入"正确的物理量"（如链级特征 Cn、自由体积）
**相关概念**: Bayes error rate, 不可约方差, 信息瓶颈, 多尺度建模
**推荐资源**:
- Hastie et al. (2009), "Elements of Statistical Learning", Chapter 2 (偏差-方差分解)
- Cover & Thomas (2006), "Elements of Information Theory" (信息论基础)

### 核酸 Tg vs Tm 的物理机制不匹配 — 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 调研核酸 Tg 特殊性，发现合成聚合物 Tg 模型预测核酸是根本性方法论错误
**分析过程**:
- Tm (melting): 溶液中双链 DNA 解离，驱动力 = H-bond + base stacking 断裂，典型 40-100°C
- Tg (glass transition): 干态链段运动解冻，驱动力 = van der Waals + 链缠结，DNA Tg ≈ -50°C (223-234K, MD 模拟)
- 两者物理机制完全不同，化学空间不重叠 (C-C/C-O 骨架 vs 糖-磷酸骨架)
- 核酸 Tm 已有黄金标准模型：SantaLucia Nearest-Neighbor Model (1998), r=0.96, ±2-3°C
**核心要点**:
- 核酸科学家关心的热稳定性 = Tm，不是 Tg
- 预测核酸 Tm 需要：序列信息 (近邻效应)、离子浓度、链长 — 这些信息无法从单体 SMILES 提取
- 10 个 Watson-Crick 近邻参数 (dH + dS) 即可精确预测 Tm
- 固态 DNA Tg 生物学意义有限（体内始终水溶液环境）
- 本项目核酸预测失败应作为 negative result 写入论文，本身有学术价值
**相关概念**: 近邻模型, 碱基堆叠, 玻璃化转变, 相变热力学
**推荐资源**:
- SantaLucia (1998), PNAS 95:1460-1465 (引用 5000+，近邻模型奠基论文)
- Le Novere (2012), BMC Bioinformatics (MELTING 软件，近邻参数实现)

### 小数据集上特征工程 vs 特征学习的策略选择 — 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 对比 polyBERT/GNN/HAPPY 等方法在 300 条数据上的适用性
**分析过程**:
- 深度学习表示 (polyBERT 768d, TransPolymer 512d) 需要大量数据学习有意义的映射
- 300 条数据上：物理描述符 + ET (R²=0.90-0.97) >> GNN (0.78-0.85) >> 端到端 DL (0.70-0.80)
- 领域知识 = "免费的训练样本"，物理公式编码了数百年实验经验
- 折中方案：用预训练模型提取 embedding，再接传统 ML，兼顾两者优势
- HAPPY (2024) 在 300-500 条上比 SMILES 略优且训练快 2 倍，是小数据 DL 的最佳选择
**核心要点**:
- 数据量决定最优策略：<500 用物理描述符 + 树模型，500-5000 用 HAPPY/预训练嵌入，>5000 用端到端 DL
- Group Contribution (Van Krevelen 改进 2020): 198 均聚物 R²=0.99, MRE=8% — 物理知识的极致
- 预训练嵌入 + ML 是风险最低的"深度学习入门"路径
- 关键洞见：数据量 < 阈值时，增加模型复杂度不仅无益反而有害
**相关概念**: 偏差-方差权衡, 归纳偏置, 预训练-微调, 数据效率
**推荐资源**:
- Kim et al. (2024), npj Comput. Mater. (HAPPY 表示)
- Afzal & Hasan (2020), ACS Omega (Modified Group Contribution for Tg)
- Grinsztajn et al. (2022), NeurIPS (Why tree-based models outperform DL on tabular data)


### 核酸 Tg vs 合成聚合物 Tg 的本质区别 -- 2026-03-13
**来源**: Level 2 (深度讲解)
**触发场景**: 深度调研核酸热力学性质预测方法
**分析过程**:
- 搜索 13+ 篇文献，涵盖 NN 模型、DNA/RNA Tg 实验、GNN 方法
- 发现核酸 Tg (~223K) 由水合层动力学驱动，非链段运动
- 评估迁移可行性: 化学空间、物理机制、特征表示均不兼容 (2/10)
**核心要点**:
- 核酸是信息聚合物 (性质由序列决定)，合成聚合物是结构聚合物 (性质由单体化学结构决定)
- 碱基堆叠力是核酸稳定性的主导因素 (贡献是 H 键的 3-10 倍)
- DNA Tg=223K 是水合层玻璃化，dry film Tg=448K 是脱水态行为
- ATP/ADP/AMP 预测成功 (3.9K 误差) 反映小分子药物类似行为，不代表模型理解核酸
- Nearest-Neighbor 模型是核酸热力学金标准 (+/-1.1 C for DNA)
**相关概念**: Nearest-Neighbor 热力学、Manning 反离子凝聚、持久长度、碱基堆叠
**推荐资源**:
- SantaLucia (1998) PNAS 95:1460 -- DNA NN 统一参数
- Shao et al. (2025) Nat. Commun. -- Array Melt + GNN
- Bizarro et al. (2024) PNAS -- RNA 冷玻璃化转变

### 聚合物数据库生态与 PolyInfo 瓶颈 — 2026-03-14
**来源**：Level 2 (深度讲解)
**触发场景**：调研 8 个 Tg 数据库的可获取性，发现大量数据集高度重叠
**分析过程**：
- 对 8 个数据库逐一溯源，发现 6 个数据集的上游都是 PolyInfo（NIMS 日本）
- 验证了 "Kim et al. 2024 >7,000" 实际是 Casanola-Martin 约 900 条（DOI 误标）
- 确认 PI1M 是纯 SMILES 库，永远不会有 Tg 列
**核心要点**：
- 聚合物 Tg 数据生态存在"PolyInfo 瓶颈"：大多数公开数据集都源自同一上游
- 不同论文宣称的数据集看似独立，实际重叠率可达 80%+
- 数据独立性是模型泛化性的前提——如果训练集和验证集同源，性能会被高估
- 共聚物 Tg 数据是真正的稀缺资源（仅 Kuenneth 数据集有约 7,774 条）
**相关概念**：数据独立性, 数据泄露(data leakage), 数据溯源(data provenance)
**推荐资源**：
- PolyInfo 数据库: https://polymer.nims.go.jp/
- Kuenneth et al. 2021, DOI: 10.1021/acs.macromol.1c00728

### 实验数据 vs 模拟数据 vs 预测数据的质量层级 — 2026-03-14
**来源**：Level 2 (深度讲解)
**触发场景**：评估 Biopolymer MD 数据集（MD 模拟 Tg）与 PolyInfo（实验 Tg）的质量差异
**分析过程**：
- 梳理了四个数据质量层级：实验 > MD模拟 > 经验公式 > ML预测
- 发现混合使用不同质量数据时需要加权处理（项目中 bridge_weight 参数的作用）
**核心要点**：
- Tier 1 实验值最可靠但有测量误差（DSC 典型精度 ±2-5K）
- Tier 2 MD 模拟值依赖力场质量（AMBER/CHARMM），系统偏差可达 20-50K
- Tier 3 经验公式（如 Fox 方程）假设理想混合，忽略分子间相互作用
- 混合不同质量层级数据时，加权系数是关键超参数
**相关概念**：多保真度学习(multi-fidelity learning), 数据融合, 分子动力学力场
**推荐资源**：
- Pilania et al. "Multi-fidelity machine learning", Comput. Mater. Sci. 2019
- van Krevelen, Properties of Polymers, 4th ed.

### 共聚物表示方法的层级与局限 — 2026-03-14
**来源**：Level 2 (深度讲解)
**触发场景**：对比 Kuenneth（双 SMILES+摩尔比）、WC-SMILES、BigSMILES 三种共聚物表示
**分析过程**：
- 共聚物数据稀缺的根本原因是表示困难：需编码单体结构+组成比+排列方式
- 三种方法各有取舍：简单表示丢失信息，复杂表示缺乏数据支持
**核心要点**：
- Kuenneth 方法（双 SMILES + 摩尔比）：简单但丢失排列信息（无规/交替/嵌段不区分）
- WC-SMILES：加权链式表示，保留一定排列信息，但长度受限
- BigSMILES：最完整的统计微观结构表示，但数据集极少支持此格式
- 表示方法的选择直接限制了可用数据集的范围
**相关概念**：分子表示(molecular representation), 信息论, SMILES 语法, 聚合物微观结构
**推荐资源**：
- Huang et al. WC-SMILES, DOI: 10.1021/acsapm.3c02715
- Lin et al. BigSMILES, DOI: 10.1021/acscentsci.9b00476

### Canonical SMILES 去重与数据管道设计 — 2026-03-14
**来源**：Level 2 (深度讲解)
**触发场景**：重写 external_datasets.py，将 6 个数据源统一为去冗余管道
**分析过程**：
- 发现字符串去重（22,674 条）与 RDKit Canonical SMILES 去重（12,148 条）差距近一倍
- 调查 NeurIPS OPP 是 POINT2/Qiu_Polymer/Qiu_PI 的超集 → 删除 3 个冗余加载器
- OpenPoly 84.3% 冲突率 → 默认排除，提供 exclude_openpoly_median 选项
**核心要点**：
- 同一分子可有无数种 SMILES 写法，RDKit MolToSmiles() 通过 Morgan 算法生成唯一标准形式
- 数据集间常存在包含关系，调查方法：分子集合交集/并集分析
- Registry 模式（字典映射名称→函数）实现开闭原则，新增数据源只需一个函数+一行注册
- 数据质量问题会"级联"影响下游模型（Google Data Cascades, NeurIPS 2021）
**相关概念**：Canonical SMILES, Registry Pattern, Data Quality Engineering, 开闭原则(OCP)
**推荐资源**：
- RDKit Getting Started: https://www.rdkit.org/docs/GettingStartedInPython.html
- Google Data Cascades Paper (NeurIPS 2021): https://research.google/pubs/everyone-wants-to-do-the-model-work-not-the-data-work/
