# 共聚物 Tg 的统计预测方法研究：从虚拟数据增强到 DNA 跨域拓展

> 版本：Nature-style 扩展稿 v0.2  
> 日期：2026-04-26  
> 用途：统计建模比赛论文主稿扩展基础  
> 写作标准：按 Nature 研究论文的叙事强度要求组织主文，即标题聚焦主发现，摘要先给领域背景和主要结论，正文以 Introduction、Results、Discussion、Methods 展开，结果依托少量高信息密度图表和可复现 Methods。比赛正式 Word 版可再转换为“绪论、数据、模型、实验、结论”格式。  
> 匿名提醒：正式匿名 PDF 不得出现学校、队员、指导教师、服务器用户名、绝对路径或个人身份信息。

## 摘要

共聚物玻璃化转变温度（glass transition temperature, Tg）预测的关键困难不是缺少单一机器学习模型，而是缺少同时满足物理约束、真实外推和小样本跨域验证的统计建模链条。现有 Tg 预测方法大多围绕均聚物展开，难以直接处理端点 Tg、组成比例、随机/嵌段架构、相容性和实验噪声共同作用下的共聚物体系；当任务进一步拓展到核酸/碱基功能化高分子时，氢键、碱基堆积和极小样本量又会带来更强的分布偏移。这里我们构建了一个从均聚物源模型到虚拟共聚物弱标签、真实共聚物清洗、核酸相关跨域验证和统一预测路由的完整统计框架。该框架以 7486 条均聚物、149 条清洗后 PolyInfo 一般共聚物和 17 条核酸/碱基功能化共聚物为真实数据基础，使用均聚物 Tg 源模型提供端点 Tg 估计，并以 Fox 方程、组成统计量和链物理特征建立物理先验门控残差回归模型。严格 group-holdout 结果显示，当前最佳单一回归模型在均聚物、一般共聚物和核酸相关共聚物上的 R2 分别为 0.887、0.849 和 0.817，跨任务主指标 min-R2 为 0.817。虚拟数据实验进一步表明，结构近邻虚拟样本可将核酸任务 R2 提升至 0.858，但会使真实 PolyInfo group R2 降至 0.827；teacher-consistency 过滤可缓解真实共聚物退化，但同时削弱核酸增益。本文的主要结论是：虚拟共聚物数据可以提供跨域信号，但必须被端点物理一致性、教师一致性和真实分布相似性约束；面向共聚物 Tg 的可信预测应由“严格统计模型”和“证据自适应应用路由”共同构成。

**关键词**：共聚物；玻璃化转变温度；统计预测；虚拟数据增强；DNA 相关高分子；物理先验

## 主文结构说明

本文采用 Nature-style 主文结构以提高故事完整性和论证强度。正式参加统计建模比赛时，可将以下结构映射为中文竞赛论文常用章节：

| Nature-style 主文 | 比赛论文对应章节 | 作用 |
|---|---|---|
| Introduction | 绪论 | 提出问题、技术挑战和贡献 |
| Results | 数据、模型、实验结果 | 展示研究发现而不是流水账 |
| Discussion | 结果解释与建议 | 总结意义、边界和未来工作 |
| Methods | 模型建立与求解 | 给出可复现的统计建模细节 |
| Extended Data | 附录 | 展示脚本、消融、负结果和复现材料 |

## Introduction

玻璃化转变温度 Tg 是高分子材料从玻璃态进入高弹态的重要热转变指标，决定了材料的耐热性、力学稳定性、加工窗口和应用场景。对于新型高分子材料设计而言，Tg 预测不仅是一个回归任务，也是一种降低实验试错成本的筛选工具。随着聚合物信息学的发展，基于结构描述符、图神经网络、预训练表示和小样本机器学习的 Tg 预测已经能够在均聚物数据上取得较好表现。然而，这些进展并不意味着共聚物 Tg 已经被解决，因为均聚物任务中的“一个重复单元对应一个 Tg”在共聚物场景中不再成立。

共聚物 Tg 预测的核心挑战是端点物理、组成比例和真实数据质量同时影响模型泛化。一个共聚物样本通常由两个或多个结构单元构成，其 Tg 取决于端点均聚物 Tg、组分比例、随机或嵌段架构、链段相容性、结晶倾向和实验条件。Fox 方程等传统公式能够提供低方差物理基线，但难以表达复杂的非理想相互作用；纯机器学习模型能够学习结构残差，但在真实共聚物样本很少时容易被均聚物大样本分布主导。更重要的是，共聚物数据库常含有重复体系、端点行泄漏和同一组成下实验值冲突，如果不清洗和分组验证，随机划分指标会显著高估真实外推能力。

DNA 相关高分子材料进一步将共聚物 Tg 预测推向小样本跨域问题。核酸/碱基功能化单元引入强氢键、碱基堆积和特异性相互作用，使其热转变行为与普通碳链或杂链聚合物不同；同时，可用于 Tg 建模的公开实验记录非常有限。本文现有核酸相关数据仅 17 条，不能支撑独立深度模型，却足以作为严苛的跨域检验：一个真正有用的共聚物 Tg 方法，至少应能在避免碱基族泄漏的 group holdout 下保持可解释的预测能力，并揭示剩余误差来自数据稀缺还是模型结构。

本文的核心观点是，共聚物 Tg 预测不应被写成“把均聚物模型换一个输入格式”的工程问题，而应被写成“源模型、虚拟弱监督、真实校正和跨域验证”共同组成的统计建模问题。我们首先利用已验证的均聚物 Tg 源模型获得端点 Tg 估计能力；然后将端点 Tg、Fox 方程、组成比例和链物理规则结合，生成二元、多元、随机和嵌段虚拟共聚物弱标签；接着清洗真实 PolyInfo 共聚物数据并构建核酸/碱基功能化共聚物跨域数据；最后在一个单一回归路径中引入非均聚物门控校准、低 Fox 区域残差收缩和模型相对 Fox 先验的差值校准。

本文的贡献有四点。第一，我们将前期均聚物 BestTg/186d TabPFN 路线、虚拟共聚物生成器、PolyInfo 共聚物清洗流程、核酸跨域诊断和统一预测路由整合为一个自洽的研究链条。第二，我们提出以三任务 min-R2 为核心的公平评价指标，防止均聚物大样本任务掩盖小样本共聚物任务。第三，我们通过 exp45 到 exp58 的受控实验展示了模型改进和虚拟增强的边界，而不是只报告最漂亮的单项结果。第四，我们将严格统计模型和应用型统一路由分开：前者用于论文主结果，后者用于实际预测时根据输入证据选择最合适路线。

## Results

### A source-model-to-copolymer pipeline organizes previously separate work into one chain

本项目的早期工作首先解决了均聚物 Tg 预测问题，并形成了可复用的端点 Tg 源模型。该路线不是简单调用一个黑箱模型，而是将三类互补信息压缩为 186 维聚合物表示：PHY-C-light 58 维物理化学描述符用于表达局部结构和链段物理，GNN embedding 64 维用于表达图结构邻域信息，polyBERT PCA 64 维用于吸收预训练语言模型中的高分子序列表示。上述特征进入 TabPFNRegressor，形成面向小样本材料性质预测的 BestTg 源模型。该模型在统一路由的严格 train/test split 中，均聚物 holdout R2 达到约 0.907、MAE 约 20.60 K；在后续统一单一回归实验中，为了同时覆盖共聚物和核酸任务，均聚物 holdout R2 稳定在 0.887 左右。本文不把均聚物模型写成最终共聚物模型，而把它定位为端点 Tg 估计器和虚拟数据生成器的源能力。

均聚物源模型在本文中承担两个具体角色。第一，当真实共聚物端点 Tg 不完整时，它为各端点 repeat-unit SMILES 预测 Tg，从而补足 Fox 方程和端点统计特征。第二，它为虚拟共聚物生成提供弱标签基础，使候选共聚物不再只依赖手工经验常数。为避免重复计算，项目实现了 component-level 特征缓存：同一端点结构在一次任务中只需完成一次 RDKit/物理特征、GNN embedding 和 polyBERT 表示计算，之后可以被多个二元、多元或嵌段 recipe 复用。这一点对大规模虚拟数据生成很重要，因为虚拟任务的计算瓶颈通常不是 recipe 数量本身，而是端点模型和深度表示的重复加载与重复 featurization。

虚拟共聚物生成器将均聚物源模型转化为共聚物弱监督数据。生成脚本 `generate_virtual_copolymer_dataset.py` 是一个独立的长任务入口，而不是对单条预测 CLI 的外部循环包装。它支持 `auto`、`csv` 和 `hybrid` 三种输入模式：`auto` 模式从内部端点库（当前主要为 Bicerano/7k 去重均聚物相关端点库）枚举候选组合；`csv` 模式读取用户指定 recipe；`hybrid` 模式合并内部枚举和用户给定候选。生成对象覆盖二元和多元共聚物，架构标签支持 `random`、`block` 和 `both`。每条输出记录包含稳定 `recipe_id`、组分序列、归一化权重、架构、预测 Tg、Fox reference、端点 Tg window 和方法来源字段，使后续训练、过滤和复现实验都可以追溯。

虚拟标签的来源是“源模型端点预测 + 共聚物物理规则”，而不是简单的随机造数。对于每个 recipe，脚本先获得各端点均聚物 Tg：若端点已经在 7k 去重数据库或预计算表中出现，则优先复用已有结果；若没有命中，则调用 BestTg 源模型预测端点 Tg。随后，脚本根据组成权重计算线性混合、Fox reference、端点 Tg window 和链物理摘要，并由 `BestTgPredictor.predict_multicomponent()` 给出 random 或 block 架构下的多组分近似 Tg。这里的 Tg 是低保真弱标签，反映“均聚物源模型和物理规则共同认为该组合可能落在的 Tg 区间”，而不是实验真值。因此，本文后续对虚拟数据只做受控弱监督实验，不把虚拟样本无条件混入最终主模型。

生成器本身也体现了大量工程工作，因为它解决了早期大规模生成中最严重的效率问题。最初如果从 shell 循环逐条调用预测脚本，每条样本都会启动新 Python 进程、重新加载 TabPFN、重新拟合 PCA/预处理器、重新加载 polyBERT 和 GNN checkpoint，导致虚拟数据生成几乎不可用。新的专用脚本在一个任务中只创建一次 `BestTgPredictor`，`predictor.fit()` 只执行一次，后续所有 recipe 共享同一模型对象和端点缓存；输出按 chunk 增量写入 CSV/JSONL，并通过 `--resume` 和稳定 recipe ID 跳过已完成样本。该设计使虚拟数据生成从“单样本推理脚本的重复调用”转变为可恢复、可审计、可长时间运行的数据生成流程。

真实共聚物数据清洗构成了从虚拟弱监督走向可信统计验证的关键步骤。PolyInfo 原始共聚物数据包含复制粘贴格式不一致、端点映射不完整、`sample_id` 非唯一、同一体系同一组成下 Tg 冲突等问题。项目中通过 `parse_polyinfo_raw.py`、`filter_polyinfo_copolymer_conflicts.py` 和稳定行标识审计，将共聚物数据整理为可用于 group holdout 的清洗集。例如，冲突过滤默认以 `COID + w1_used` 为组，若 Tg 标准差超过 10 K 则剔除整组；这一规则识别并剔除了 P900016、P900017 等冲突组。该步骤使后续模型评价不再依赖含有明显矛盾的样本。

核酸/碱基功能化共聚物数据提供了 DNA 相关跨域验证。项目不是声称已经拥有完整 DNA 长链 Tg 数据库，而是将 17 条核酸/碱基功能化共聚物记录作为先导跨域场景。早期专用脚本 `predict_nucleobase_excel_with_copolymer_model.py` 和 `evaluate_nucleobase_copolymer_strategies.py` 显示，如果端点 Tg 已知，实际端点 Linear-Fox 或 Physics-Ridge 可在 leave-base-out 场景中取得很高应用精度；但这类结果依赖任务特定端点信息，不应与统一单一回归的严格泛化指标混用。因此，本文将专用核酸结果写作“诊断和应用上界”，而将 exp56 的 group holdout 结果作为主模型结果。

这一链条将项目原本分散的工作组织为一个递进式研究框架：均聚物源模型提供端点 Tg，虚拟生成器扩展候选共聚物空间，真实 PolyInfo 清洗提供严格验证集，核酸数据提供跨域压力测试，单一回归模型给出比赛论文主结果，统一路由系统给出实际应用入口。图1 应展示这一完整流程，而不应只展示最后一个回归器。

### A strict evaluation protocol prevents inflated copolymer performance

本文采用三任务分开评价的协议，以避免随机划分带来的虚高结果。均聚物任务使用 random holdout 检验基础结构-Tg 映射能力；一般共聚物任务使用 PolyInfo group holdout，按共聚物体系外推；核酸相关任务使用 group holdout 或 leave-base-out，按碱基族外推。主指标定义为：

```text
primary min-R2 = min(R2_homopolymer, R2_polyinfo_group, R2_nucleobase_group)
```

这一指标的目的是让模型不能只在样本量最大的均聚物任务上表现好。若使用整体 R2，7486 条均聚物样本会主导模型选择，149 条 PolyInfo 共聚物和 17 条核酸样本的失败会被掩盖。min-R2 则要求模型在三个任务上同时维持可接受表现，适合本文“从均聚物到共聚物再到 DNA 跨域”的题目结构。

应用路由指标和严格泛化指标在本文中被明确区分。统一路由脚本 `predict_polymer_tg_universal.py` 是一个证据自适应预测系统：它先识别输入是单一均聚物、二元 random 共聚物、多元 random 共聚物、block 共聚物、已知 PolyInfo 体系、核酸端点已知体系，还是端点缺失的新体系；再根据可用证据选择预测路线。该脚本能在已知体系中使用 same-system residual IDW，从而在同体系变组成应用中取得更高精度；核酸专用路线在端点 Tg 已知时也能显著优于通用端点预测。然而，这些指标属于“有额外证据的应用场景”，不能作为新体系 group holdout 的主结果。论文主表只报告严格 holdout/group holdout 指标，路由结果放在应用系统和附录中。

### Physics-guided gated residual regression gives the best eligible single-regressor result

当前最佳合规模型为 `exp56_homo_local_fox_pred_delta_nonhomo_cal_lowfox_shrink_nopure`，模型可概括为物理先验门控残差回归。它使用真实数据共 7652 条，包括 7486 条均聚物、149 条一般共聚物和 17 条核酸/碱基功能化共聚物。样本权重用于平衡不同来源：均聚物权重为 1，一般共聚物权重为 10，核酸相关共聚物权重为 60；虚拟样本在最终主模型中不作为无条件训练增益，而用于受控增强实验。

模型的关键不是简单叠加更多特征，而是控制统计残差相对物理先验的偏离。第一，非均聚物门控校准使端点/Fox 校准主要作用于共聚物，避免被均聚物大样本稀释。第二，低 Fox 区域残差收缩在 `endpoint_tg_fox_c < -35 C` 等低 Tg 先验区域将预测部分拉回端点物理基线，降低硬共聚物的错误外推。第三，预测差值校准使用 `base_prediction - endpoint_tg_fox` 表示模型相对 Fox 先验的偏移，对非均聚物进行低容量修正。

严格评估结果显示，exp56 在均聚物 random holdout 上 R2=0.887、MAE=27.164 ℃；在 PolyInfo 一般共聚物 group holdout 上 R2=0.849、MAE=16.654 ℃；在核酸/碱基功能化共聚物 group holdout 上 R2=0.817、MAE=6.266 ℃。三任务主指标 min-R2 为 0.817。该结果没有达到所有任务 R2=0.95，但在小样本共聚物和核酸跨域场景下是当前最可信的合规结果。

消融实验说明性能提升来自受约束的物理校准，而非盲目增加模型容量。exp45 作为无纯端点泄漏的基础物理-局部残差模型，min-R2 为 0.789；exp53 引入非均聚物门控 endpoint/Fox 校准，将 min-R2 提升至 0.792；exp55 加入低 Fox 区域向端点物理收缩，将 min-R2 提升至 0.810；exp56 加入预测值与 Fox 差值的非均聚物校准，将核酸 group R2 提升至 0.817，并成为当前最佳合规模型。虽然 exp56 使 PolyInfo group R2 从 exp55 的 0.852 略降至 0.849，但它提高了最弱的核酸跨域任务，因此在 min-R2 目标下更均衡。

### Virtual copolymer data reveal a real augmentation trade-off rather than a simple gain

虚拟数据实验是本文扣题“从虚拟数据增强”的核心，但其结论必须写得严谨。结构近邻虚拟增强实验 exp57 从已有 5000 条 HYBRID-HOMO186 虚拟行中选择与真实 PolyInfo 样本在非目标特征空间较接近的 400 条，以低权重 0.05 加入训练。该实验将核酸 group R2 从 exp56 的 0.817 提升到 0.858，说明虚拟共聚物弱标签确实携带了对核酸跨域有用的高 Tg 或结构信号。

然而，exp57 同时将 PolyInfo group R2 从 0.849 降至 0.827，说明未充分约束的虚拟标签会损害真实共聚物体系外推。该现象不能被忽略，因为本文的目标不是只优化核酸 17 条样本，而是建立同时覆盖均聚物、一般共聚物和核酸相关共聚物的统计方法。按照预设的虚拟数据规则，只要真实 group holdout 明显退化，就不能将该增强模型作为最终模型。

teacher-consistency 实验 exp58 进一步揭示了虚拟池的标签冲突。该实验先用真实数据 exp56 路径训练教师模型，再筛选虚拟标签与教师预测差异不超过 30 ℃ 的样本。筛选前，虚拟标签与教师模型的中位差异约为 145.05 ℃；筛选后，中位差异降至约 12.65 ℃。这说明现有虚拟池中存在严重弱标签冲突，而 teacher-consistency 能有效提高标签一致性。

但 exp58 并没有超过 exp56 的主指标。它使 PolyInfo group R2 轻微恢复到 0.850，却使核酸 group R2 回落至 0.817。这个结果说明，强一致性筛选会过滤掉一部分对核酸跨域有用的虚拟信号。由此得到的统计结论比简单宣称“虚拟数据提高精度”更有价值：虚拟数据增强的关键不是样本量，而是端点完整性、物理一致性、教师一致性和跨域信号之间的平衡。

### Error analysis identifies data and representation limits

误差诊断显示，当前模型的主要瓶颈不是再调一个权重就能解决的局部问题，而是数据和表示层面的限制。PolyInfo group holdout 中，P900015 是最主要的硬体系。在早期 exp45 诊断中，P900015 的 MAE 约 45.12 ℃，剔除该体系可将计算得到的 PolyInfo group R2 从约 0.844 提高到约 0.895。后续 exp53 诊断中，P900015 仍然是主导误差，且 endpoint Fox prior 在该体系上反而比模型预测更接近实验值，说明学习到的残差有时会把预测从有用的物理先验推开。

P900015 的错误不能简单归因于重复测量冲突或方向错误。诊断显示，该体系同时存在低 Tg/高次要组分比例行被过预测、高 Tg/低次要组分比例行被低预测的混合符号误差，因此系统级截距校正并不安全。更可能的原因是当前特征没有充分表达相态、混溶性、结晶或组成-相行为。这一发现支持后续增加 no-leak phase/reliability 特征，而不是继续叠加自由残差容量。

核酸任务的瓶颈也具有系统性。exp55 之后，T 族样本仍有系统性低估，G 族样本仍有系统性高估。诊断性 leave-base-out 校准显示，如果只在核酸任务内部使用 endpoint/Fox affine calibrator，R2 可显著提高，但这不是合规最终模型，因为它属于任务特定后处理，无法证明对一般共聚物和均聚物同时泛化。exp56 通过通用非均聚物预测差值校准部分吸收了这一信号，但在 17 条样本条件下仍受碱基族偏差限制。

后 exp56 的多轮实验说明当前瓶颈不能靠小幅调参解决。final-calibration lambda 变化几乎只是数值噪声；source-balanced 或 nucleobase-boosted final calibration 最多带来约 0.00022 的 min-R2 改变，还会降低 PolyInfo group R2；低权重 additive physical/embedding sum-kernel 使主指标降至 0.815844；现有虚拟行的结构近邻和 teacher-consistency 筛选也未能产生合规提升。这些负结果应写入论文，因为它们证明本文不是只挑好看的实验，而是系统检验了虚拟增强、校准权重和核容量扩展的边界。

### The application route system translates the strict model into usable predictions

统一预测路由系统是本文的应用层贡献。严格统计模型回答“在无泄漏评估下模型能否泛化”，而路由系统回答“实际给一个新高分子输入时应该调用哪条证据最充分的预测路线”。这两者服务不同目的，应共同呈现但不能混淆。

路由系统的第一步是任务识别，而不是直接回归。输入只有一个 repeat-unit SMILES 时，系统将其判定为均聚物任务，调用 BestTg/186d TabPFN 路线；输入包含 `smiles1, smiles2, w1` 等字段时，系统判定为二元共聚物任务，并继续检查是否存在 `system_id`、是否匹配 clean PolyInfo calibration 表、端点 Tg 是否可用、架构是否为 random 或 block。批量 CSV 输入时，系统逐行执行同样的判定，并在输出中保留 `router_route`、`primary_method`、端点来源、Fox reference 和 fallback 标记。这样做的目的不是让路由器“挑一个最好看的结果”，而是让每个预测值都带有可解释的证据来源。

路由系统的第二步是根据证据强度选择模型族。单个 repeat-unit SMILES 进入 BestTg/186d TabPFN 均聚物路线；未知二元 random 共聚物在端点存在时进入 Global Linear-Fox 或统一残差路线；已知 PolyInfo 体系在允许应用插值时进入 `binary_known_system_local_residual_idw`，即在全局 Linear-Fox 基线之外使用同体系组成点的 residual IDW 修正；多元 random 共聚物采用 endpoint Fox 多组分近似；block 共聚物输出 miscible proxy 和 Tg window，不假装精确处理真实相分离；端点缺失时可选择 BestTg weighted descriptor/embedding fallback，并降低置信度。这个分层设计将“物理先验、统计校准、同体系插值和探索性 fallback”明确分开。

路由系统的第三步是输出可审计预测结果，而不只是一个 Tg 数字。典型输出包括 Tg 预测值、使用路线、primary method、端点 Tg、Fox reference、端点 Tg window、是否命中 clean PolyInfo 已知体系、是否使用 BestTg fallback、是否属于 block/multicomponent 近似，以及可选的 expected Tg 用于误差分析。这样的输出格式使研究者能够判断某个预测是严格外推、已知体系插值、核酸端点校准，还是端点缺失下的探索性估计。本文将路由系统写作应用层，是因为它反映了实际材料筛选中必须面对的证据不完整问题。

核酸相关输入需要单独标注证据等级。如果实际端点 Tg 已知，核酸专用 endpoint Linear-Fox 或 Physics-Ridge 可作为应用路线；若端点缺失，则应回退到统一单一回归模型，并明确提示小样本跨域风险。这样的路由设计使模型在实际使用中更诚实：它不仅输出 Tg 数值，还输出路线、证据来源和适用边界。

## Discussion

本文最重要的结论是，共聚物 Tg 预测的可信建模单位不是一个孤立回归器，而是一条可审计的数据和证据链。均聚物源模型提供端点 Tg 能力，虚拟生成器扩展候选空间，真实共聚物清洗提供严格验证，核酸数据提供跨域压力测试，单一回归模型提供可比较主指标，路由系统提供实际应用接口。只有把这些环节组织在一起，题目中的“共聚物 Tg”“统计预测方法”“虚拟数据增强”和“DNA 跨域拓展”才形成完整闭环。

物理先验在本文中起到降低方差和约束外推的作用。Fox 方程本身不是最终模型，但它提供了端点 Tg 与组成比例之间的物理基线。门控残差回归的价值在于让统计模型只在必要位置修正物理先验，而不是无约束地覆盖物理规律。exp53、exp55 和 exp56 的递进结果说明，针对非均聚物的门控校准和低 Fox 区域收缩比简单增加模型容量更可靠。

虚拟数据增强的结论需要被谨慎表述。本文发现虚拟样本可以显著改善核酸跨域任务，但也会破坏真实 PolyInfo group holdout。这并不是虚拟数据“失败”，而是说明低保真标签必须被物理一致性和真实分布约束。对于统计建模比赛而言，这一负结果具有方法价值：它展示了样本量、标签质量和分布偏移之间的权衡，而不是机械追求更大的训练集。

本文目前的主要限制来自数据和表示。一般共聚物只有 149 条清洗后真实样本，核酸相关样本只有 17 条；P900015 等硬体系可能包含相态、混溶性或结晶行为，而当前特征尚未显式表达这些因素。未来提升模型精度的关键不是继续微调 exp56，而是三类更实质性的改进：重建端点完整的虚拟共聚物数据；加入无泄漏的相容性、结晶或相态可靠性特征；扩充核酸/碱基功能化共聚物实测数据。

对实际材料筛选而言，本文建议输出的不应只有一个 Tg 点估计。更合理的输出包括预测 Tg、预测路线、端点物理一致性、是否属于已知体系插值、是否跨域、以及置信度或风险标记。对于已知体系内组成优化，可以使用残差插值提高应用精度；对于全新体系，应以 group holdout 验证过的统一模型为主；对于核酸/DNA 相关材料，应优先补充实测端点 Tg 和碱基族信息。

## Methods

### Data sources

真实数据由三部分组成：7486 条均聚物 Tg 数据、149 条清洗后 PolyInfo 一般共聚物数据和 17 条核酸/碱基功能化共聚物数据。均聚物数据用于学习基础结构-Tg 映射并提供端点 Tg 源模型；PolyInfo 共聚物数据用于一般共聚物体系外推评估；核酸数据用于 DNA 相关材料先导跨域评估。虚拟共聚物数据由项目生成脚本产生，仅用于弱监督增强实验和风险诊断，不视为实验真值。

### Homopolymer source model

均聚物源模型对应项目中的 BestTg/186d TabPFN 路线。输入为单个 repeat-unit SMILES，输出为均聚物 Tg 预测值以及可用于后续共聚物计算的端点 Tg。特征由三部分组成：PHY-C-light 58 维物理化学特征、GNN embedding 64 维结构表示和 polyBERT PCA 64 维预训练序列表示，共 186 维。训练时，PCA 和回归器只在训练折上拟合；推理和虚拟生成时，源模型可在全量均聚物数据上拟合以提供端点预测能力。

BestTg 源模型被封装为 `BestTgPredictor`，其内部包含模型加载、PCA、TabPFN 回归器、组件 featurization 和组件级缓存。对于共聚物任务，端点结构会反复出现在不同组成和不同架构 recipe 中，因此缓存是必要的。`featurize_component()` 会将端点 SMILES 的 RDKit/物理描述符、GNN embedding 和 polyBERT 表示缓存到 `_component_cache`，后续相同端点复用已有特征。这使端点预测从“每条 recipe 都重算”变为“每个端点只算一次”。

### Virtual copolymer generation

虚拟共聚物生成由 `scripts/generate_virtual_copolymer_dataset.py` 完成，目标是为共聚物模型提供可追溯的低保真弱监督数据。脚本支持三种模式：`auto` 从内部端点库枚举组合，`csv` 从用户提供文件读取 recipe，`hybrid` 合并内部枚举和外部输入。每个 recipe 可包含两个或多个端点，权重会被归一化；架构字段支持 `random`、`block` 或 `both`。输出格式支持 CSV 和 JSONL，长任务按 chunk 增量写入，并可通过 `--resume` 跳过已完成 recipe。

虚拟共聚物的来源包含三个层次。第一，端点来源于内部均聚物端点库、Bicerano/7k 去重库或用户 CSV。第二，端点 Tg 来源于已有去重数据库命中、预计算结果复用或 BestTg 源模型预测。第三，共聚物 Tg 弱标签来源于端点 Tg、组成权重、Fox reference、端点 Tg window、descriptor/embedding 加权混合和 random/block 架构近似。给定候选端点结构、组成权重和架构类型后，脚本首先预测或复用端点 Tg，再基于 Fox 方程和链物理规则生成弱标签。Fox 计算使用 Kelvin 温度：

```text
1 / Tg_Fox = sum_i w_i / Tg_i
```

其中 `Tg_i` 为端点均聚物 Tg，`w_i` 为组成权重。生成结果记录端点、权重、架构和标签来源，用于后续过滤和审计。

输出 schema 保留了虚拟数据的来源和不确定性线索。每行至少包含 `recipe_id`、`mode`、`architecture`、`n_components`、`components_serialized`、`weights_serialized`、`tg_k_pred`、`tg_c_pred`、`primary_method`、`descriptor_mix_tg`、`fox_reference_tg` 和 `component_tg_window` 等字段。这些字段允许后续筛选时判断一个虚拟样本是否端点跨度过大、是否偏离 Fox reference、是否落在真实 PolyInfo 组成范围附近，以及是否与教师模型预测冲突。因此，虚拟数据不是一个不可解释的黑箱 CSV，而是带有方法 provenance 的弱标签表。

### PolyInfo cleaning and conflict filtering

PolyInfo 原始共聚物数据经过专用解析和清洗。核心操作包括单位统一、比例方向判断、mol% 到近似 weight fraction 转换、端点映射、pure endpoint rows 剔除、稳定行标识构造和冲突组剔除。冲突剔除规则为：同一 `COID + w1_used` 下 Tg 标准差超过 10 K 的组被视为高冲突组并整体移除。该规则减少了同一组成下实验值严重矛盾对模型上限的影响。

### Feature representation

统一模型使用结构、组成、端点物理和校准特征。结构特征来自 repeat-unit 表示和嵌入；组成特征包括组分权重、最大/最小组分比例、组成熵和 Herfindahl 指数；端点物理特征包括端点 Tg 最小值、最大值、加权均值、Fox Tg 和端点跨度；校准特征包括 `is_homopolymer`、`endpoint_tg_fox_c`、低 Fox 区域标记和 `base_prediction - endpoint_tg_fox`。

### Single-regressor training

最终主模型使用统一训练表：

```text
D_real = D_homopolymer + D_polyinfo + D_nucleobase
```

不同来源样本使用权重平衡：均聚物权重 1，一般共聚物权重 10，核酸相关共聚物权重 60。当前最佳模型 exp56 不将虚拟样本作为无条件训练增益，而是在 exp57 和 exp58 中单独评估虚拟弱监督影响。模型输出 Tg 摄氏度预测值。

### Gated residual calibration

物理先验门控残差回归由三部分组成。首先，非均聚物门控 endpoint/Fox 校准只在共聚物行中启用低容量校准，防止均聚物大样本主导校准器。其次，低 Fox 区域残差收缩在低 Tg 先验区域将预测向端点物理基线收缩，以降低外推误差。最后，预测差值校准使用模型相对 Fox 先验的偏移作为校准特征，针对非均聚物执行受约束修正。

### Evaluation protocol

均聚物任务使用 random holdout；一般共聚物任务使用 PolyInfo group holdout，按体系外推；核酸任务使用 group holdout 或 leave-base-out，按碱基族外推。主指标为三任务 R2 的最小值：

```text
primary min-R2 = min(R2_homopolymer, R2_polyinfo_group, R2_nucleobase_group)
```

报告指标包括 R2、MAE 和 RMSE。虚拟数据实验必须同时报告 PolyInfo group holdout 和核酸 group holdout；若真实 PolyInfo group holdout 明显退化，则不提升为最终模型。

### Virtual filtering experiments

exp57 使用结构/组成近邻筛选，从现有虚拟池中选择接近真实 PolyInfo 分布的虚拟样本并以低权重加入训练。exp58 使用 teacher-consistency 筛选，先以真实数据路径训练教师模型，再保留虚拟标签与教师预测差异不超过阈值的样本。两者均作为虚拟增强诊断，而非最终主模型。

### Application routing

应用路由脚本 `scripts/predict_polymer_tg_universal.py` 根据输入证据选择预测路线。它首先解析输入形态：若只有 `smiles`，进入均聚物路线；若存在 `smiles1/smiles2/w1`，进入二元共聚物路线；若存在多个 `smiles_i/w_i`，进入多元路线；若 `architecture=block`，进入 block proxy；若包含 `system_id` 或 `COID`，尝试匹配 clean PolyInfo calibration 表；若输入标记为核酸相关体系，则检查实际端点 Tg 是否可用。每条输出均保留路由名称和 primary method。

路由器中的主要路线包括：BestTg 186d TabPFN 均聚物预测；未知二元 random 共聚物的 Global Linear-Fox 或 endpoint/Fox 残差路线；已知体系的 `global_linear_fox_plus_same_system_residual_idw`；多元 random 共聚物的 endpoint Fox 多组分近似；block 共聚物的 miscible proxy 和 Tg window；端点缺失时的 BestTg weighted descriptor/embedding fallback；核酸端点已知时的 actual endpoint Linear-Fox 或 Physics-Ridge。路由器的价值在于把“严格泛化模型”和“应用时可用的额外证据”分开管理，而不是把所有样本强行交给一个不区分场景的模型。

## Display Items

### Figure 1. Overall study design

建议绘制从左到右的流程图：均聚物 Tg 数据和源模型；虚拟共聚物生成；PolyInfo 清洗；核酸/碱基功能化数据；统一单一回归；虚拟增强诊断；应用路由系统。图中应明确虚拟数据是弱标签，不是实验真值。

### Figure 2. Data composition and evaluation splits

建议展示三类真实数据样本量、Tg 分布和评估方式。均聚物为 random holdout，PolyInfo 为 group holdout，核酸为 base-family group holdout。该图用于证明本文指标设计不是整体随机划分。

### Figure 3. Physics-guided gated residual regression

建议展示端点 Tg、Fox 先验、结构/组成特征和三类校准机制：非均聚物门控、低 Fox 收缩、预测差值校准。图中应强调单一回归路径，而不是三个独立模型。

### Figure 4. Ablation trajectory from exp45 to exp56

建议用折线或柱状图展示 exp45、exp53、exp55、exp56 的三任务 R2 和 min-R2。重点突出 exp55 和 exp56 的方法贡献。

### Figure 5. Virtual augmentation gain-risk trade-off

建议同时展示 exp56、exp57、exp58 的 PolyInfo group R2 和 nucleobase group R2。该图应成为“虚拟数据不是越多越好”的核心证据。

### Figure 6. Application routing system

建议绘制输入类型到预测路线的决策树：均聚物、未知共聚物、已知体系、核酸端点已知、核酸端点缺失、多元/嵌段、虚拟生成。

## Extended Data

### Extended Data Table 1. Main model metrics

| 任务 | n | MAE/℃ | RMSE/℃ | R2 |
|---|---:|---:|---:|---:|
| 均聚物 random holdout | 1498 | 27.164 | 38.090 | 0.887 |
| PolyInfo group holdout | 149 | 16.654 | 21.639 | 0.849 |
| 核酸/碱基功能化 group holdout | 17 | 6.266 | 8.511 | 0.817 |
| primary min-R2 | - | - | - | 0.817 |

### Extended Data Table 2. Controlled ablations

| 实验 | 方法要点 | 均聚物 R2 | PolyInfo group R2 | 核酸 group R2 | min-R2 | 状态 |
|---|---|---:|---:|---:|---:|---|
| exp45 | 基础物理-局部残差模型，无 pure endpoint 泄漏 | 0.887 | 0.844 | 0.789 | 0.789 | 基线 |
| exp53 | 非均聚物门控 endpoint/Fox 校准 | 0.887 | 0.845 | 0.792 | 0.792 | 保留 |
| exp55 | 低 Fox 区域向端点物理收缩 | 0.887 | 0.852 | 0.810 | 0.810 | 保留 |
| exp56 | 非均聚物预测差值校准 | 0.887 | 0.849 | 0.817 | 0.817 | 当前最佳 |
| exp57 | 结构近邻虚拟数据增强 | 0.886 | 0.827 | 0.858 | 0.827 | 因 PolyInfo 退化拒绝 |
| exp58 | teacher-consistency 虚拟筛选 | 0.886 | 0.850 | 0.817 | 0.817 | 未超过 exp56 |

### Extended Data Table 3. Application routes

| 输入类型 | 预测路线 | 论文定位 |
|---|---|---|
| 单一 repeat-unit SMILES | BestTg/186d TabPFN | 均聚物源模型和应用路线 |
| 未知二元共聚物 | endpoint Fox / Linear-Fox / 统一残差 | 新体系外推 |
| 已知 PolyInfo 体系新组成 | same-system residual IDW | 应用插值，不作为主泛化指标 |
| 多元随机共聚物 | endpoint Fox 多组分近似 | 保守工程路线 |
| block 共聚物 | miscible proxy + Tg window | 风险提示路线 |
| 核酸端点已知体系 | actual endpoint Linear-Fox / Physics-Ridge | 专用应用路线 |
| 核酸端点缺失体系 | 统一单一回归回退 | 跨域保守预测 |
| 虚拟数据生成 | BestTg endpoint + Fox/chain physics | 弱标签生成 |

### Extended Data Table 4. Virtual copolymer generator schema

| 字段 | 含义 | 用途 |
|---|---|---|
| `recipe_id` | 由组分、权重和架构生成的稳定 ID | 支持 resume 和去重 |
| `mode` | `auto`、`csv` 或 `hybrid` | 记录候选来源 |
| `architecture` | `random`、`block` 或 `both` | 区分链段架构近似 |
| `n_components` | 组分数 | 支持二元和多元共聚物 |
| `components_serialized` | 端点 SMILES 序列 | 复现实验输入 |
| `weights_serialized` | 归一化组成权重 | 复现 Fox 和混合计算 |
| `tg_k_pred` / `tg_c_pred` | 虚拟 Tg 弱标签 | 训练或筛选目标 |
| `primary_method` | 标签生成主方法 | 区分 descriptor/embedding/Fox 等来源 |
| `descriptor_mix_tg_c` | 描述符混合预测 | 诊断标签组成 |
| `fox_reference_tg_c` | Fox reference | 物理一致性筛选 |
| `component_tg_window_*` | 端点 Tg 范围 | 判断端点跨度和外推风险 |

### Extended Data Table 5. Router output fields

| 字段 | 含义 | 为什么重要 |
|---|---|---|
| `predicted_tg_c` | 最终 Tg 预测 | 应用输出 |
| `router_route` | 路由决策名称 | 判断预测是均聚物、外推、插值还是 fallback |
| `primary_method` | 实际主预测方法 | 区分 BestTg、Linear-Fox、IDW、统一回归等 |
| `endpoint_tg_*` | 端点 Tg 估计或实测值 | 解释 Fox 和端点物理先验 |
| `fox_reference_tg_c` | Fox 基线 | 判断模型相对物理先验的偏移 |
| `known_system_hit` | 是否命中 clean PolyInfo 体系 | 区分同体系插值和新体系外推 |
| `used_besttg_fallback` | 是否使用端点缺失 fallback | 标注低置信度预测 |
| `architecture` | random/block/multicomponent | 标注架构近似边界 |
| `confidence_note` | 风险提示 | 防止把探索性预测当作高可信实验值 |

### Extended Data Note 1. Why exp57 is not the final model

exp57 的 raw primary min-R2 看似高于 exp56，是因为核酸 group R2 从 0.817 提升到 0.858；但 PolyInfo group R2 同时从 0.849 降至 0.827。由于本文目标是同时覆盖真实一般共聚物和核酸跨域任务，真实共聚物 group holdout 的明显退化不可接受。因此 exp57 应作为虚拟增强潜力和风险的证据，而不是最终模型。

### Extended Data Note 2. Why the project stopped after exp56

exp56 之后，lambda-only final calibration、source-balanced calibration、nucleobase-boosted calibration、teacher-consistency virtual filtering 和 additive multi-kernel diagnostics 均未带来合规提升。瓶颈指向数据和表示限制，而不是单一超参数。下一步应优先重建端点完整虚拟数据、增加 no-leak phase/reliability 特征或扩充核酸实测数据。

## Code and data availability

正式提交时建议将以下材料作为“数据及其他材料”打包：

| 材料 | 说明 |
|---|---|
| `scripts/generate_virtual_copolymer_dataset.py` | 虚拟共聚物数据生成 |
| `scripts/train_universal_tg_single_regressor.py` | 统一单一回归模型训练 |
| `scripts/predict_polymer_tg_universal.py` | 应用型统一路由预测 |
| `scripts/parse_polyinfo_raw.py` | PolyInfo 原始数据解析 |
| `scripts/filter_polyinfo_copolymer_conflicts.py` | 共聚物冲突样本剔除 |
| `scripts/predict_nucleobase_excel_with_copolymer_model.py` | 核酸数据预测 |
| `results/universal_single_regressor/scoreboard.json` | 主模型 scoreboard |
| `docs/research/universal-tg-bottleneck-report-2026-04-26-round25.md` | 后 exp56 瓶颈报告 |

匿名版注意：提交材料中不得包含服务器用户名、个人目录、SSH 地址、密码、学校或队员信息。

## Claim-evidence map

| 主张 | 证据 | 论文处理 |
|---|---|---|
| 单一回归模型可同时覆盖均聚物、一般共聚物和核酸相关共聚物 | exp56 三任务 R2=0.887/0.849/0.817 | 主结果 |
| 虚拟数据对核酸跨域有潜力 | exp57 核酸 group R2=0.858 | 结果与讨论 |
| 虚拟数据会损害真实共聚物泛化 | exp57 PolyInfo group R2=0.827 | 结果与讨论 |
| teacher-consistency 可降低虚拟标签冲突 | exp58 筛选前后虚拟标签与教师差异 145.05 ℃ -> 12.65 ℃ | 结果与方法 |
| 后 exp56 小调参无效 | lambda/source-balanced/additive-kernel/virtual filtering 均未合规提升 | Discussion |
| 已知体系路由能提高应用预测 | 统一路由和 same-system residual IDW 阶段测试 | 应用系统，不作主指标 |
| 已达到所有任务 R2=0.95 | 无证据，严格 min-R2=0.817 | 不使用 |

## 参考文献占位

正式论文需补充真实参考文献。建议包括：

1. Fox 方程和共聚物 Tg 经典关系文献；
2. 高分子 Tg 物理化学基础文献；
3. 聚合物信息学综述；
4. 聚合物 GNN/Transformer/预训练表示文献；
5. 小样本聚合物 Tg 预测和 TabPFN/物理先验机器学习文献；
6. 虚拟聚合物数据、弱监督和数据增强文献；
7. PolyInfo 数据库或高分子数据库文献；
8. 核酸/碱基功能化高分子热性能文献；
9. 统计建模中 group holdout、数据泄漏和外推验证文献。
