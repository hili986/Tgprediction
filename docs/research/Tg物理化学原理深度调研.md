# 玻璃化转变温度 (Tg) 物理化学原理深度调研

> 日期：2026-03-14
> 项目：同济大学 SITP - AI 辅助高分子材料设计（Tg 预测）
> 目的：深入理解 Tg 的物理本质，为设计物理约束的 ML 模型提供理论基础

---

## 1. Tg 的物理本质：三大理论

### 1.1 自由体积理论 (Free Volume Theory)

**核心思想**：聚合物分子链运动需要空间。自由体积是分子间未被占据的空间。当温度下降到某点，自由体积收缩到临界最小值，链段运动被冻结，即玻璃化转变。

**数学描述**：
```
f = V_f / V = (V - V_0) / V
f(T) = f_g + alpha_f * (T - T_g)
```
- f_g ≈ 0.025（Tg 处自由体积分数，即 2.5%）
- alpha_f ≈ 4.8e-4 /K

**Simha-Boyer 关系**：
```
(alpha_l - alpha_g) * T_g ≈ 0.113
alpha_l * T_g ≈ 0.164
```
Tg 是等自由体积态。

### 1.2 动力学理论 (Kinetic Theory)

Tg 是动力学冻结。冷却速率超过链段重排速率时进入玻璃态。Tg 有 ~5K 固有不确定性。

### 1.3 热力学理论 (Gibbs-DiMarzio)

```
Tg 正比于 E_h / (k_B * ln(f))
```
f = 柔性参数，E_h = 孔穴能。柔性越大 Tg 越低，链间作用越强 Tg 越高。

**三个理论都指向：链段柔性 + 分子间相互作用强度。**
---

## 2. 核心经验方程

### 2.1 Fox-Flory（分子量效应）
`T_g = T_g,inf - K/M_n`
链端贡献额外自由体积，高MW时 K/Mn→0。

### 2.2 WLF 方程
从 Doolittle `ln(eta)=lnA+B/f` 推导：
`log(a_T) = -C1*(T-T0)/(C2+T-T0)`
C1=17.44, C2=51.6K, f_g=0.025, alpha_f=4.8e-4/K。等价 VFT 方程。

### 2.3 Fox 方程
`1/Tg = w1/Tg1 + w2/Tg2`（自由体积加和性）

### 2.4 Gordon-Taylor
`Tg = (w1*Tg1 + k*w2*Tg2)/(w1 + k*w2)`，k = Dalpha1/Dalpha2

---

## 3. Van Krevelen 基团贡献法

`Tg = SUM(Ygi) / Mw`

Liu et al. (2020) 改进：198聚合物 R2=0.9925，58 基团。

| 基团 | 效应 | 对Tg影响 |
|------|------|---------|
| -CH2- | 柔性 | 降低 |
| -C(CH3)2- | 刚性 | 升高 |
| -p-C6H4- | 刚性主链 | 显著升高 |
| -O- | 柔性节点 | 降低 |
| -Si-O- | 超柔性 | 大幅降低 |
| -CONH- | 刚性+氢键 | 显著升高 |

GC+ML策略：残差学习(推荐)、GC作特征、Physics-Informed Loss。
---

## 4. 决定 Tg 的关键结构因素

### 4.1 链骨架柔性（最重要）
| 骨架 | 聚合物 | Tg(K) |
|------|--------|-------|
| -Si-O- | PDMS | 150 |
| -CH2-CH2- | PE | 195 |
| -CH2-CHR- | PP | 253 |
| -C6H4-O- | PPO | 483 |
| 全芳酰胺 | Kevlar | >500 |

M/f 公式：`Tg = A*(M/f)^p + C`，精度 3.7-6.4K。
柔性键：C-C=1, C-OH=0, amide=0, p-phenyl=1.5。

### 4.2 侧基效应
**位阻**：H(195K)->CH3(253K)->C6H5(373K)->CH3+C6H5(441K)
**侧链长度**：短位阻升->中内增塑降->长可结晶
**极性**：PE vs PVC(+58K), PE vs PVA(>+100K)
**对称性**：对称降Tg。PVC 354K vs PVDC 255K(差99K)

### 4.3 分子间相互作用
氢键(10-40kJ/mol), pi-pi堆积, 范德华力(CED度量)

### 4.4 CED 与 Tg
非极性：`Tg ≈ a*CED + b`（线性）。CED 可通过 GC 法计算。

### 4.5 自由体积推算
`V_f ≈ V_m - 1.3*V_vdW`。RDKit: MolVolume, LabuteASA, TPSA, FractionCSP3。
---

## 5. 从单体到聚合物的结构推断

### 5.1 可推断
Tg(高MW均聚物, MAE 20-40K), 溶度参数(~5%), CED(~10%), 密度(~5%)

### 5.2 不可推断
Tm, 力学强度, 电阻率, 低MW Tg, 共聚物序列Tg, 交联Tg, 立构异构体Tg

### 5.3 链刚性指标
`C_inf = <r^2>_0/(n*l^2)`, `l_p = C_inf*l/2`
l_p<1nm柔性, 1-10nm半柔性, >10nm刚性(DNA~50nm)

### 5.4 聚合方式
加聚: C-C主链,侧基决定Tg。缩聚: 杂原子主链,氢键/偶极力使Tg偏高。

---

## 6. 物理约束在 ML 中的应用

### 6.1 可用约束
| 物理关系 | ML 约束 |
|---------|--------|
| GC 加和性 | 残差学习基线 |
| CED/M_f 正相关 | 单调性约束 |
| 柔性负相关 | 单调性约束 |
| 100K<Tg<700K | 硬边界 |

### 6.2 Physics-Informed Loss
`L = L_data + lam1*L_physics + lam2*L_monotonic + lam3*L_range`

### 6.3 PENN 范式
NN预测物理方程参数，物理方程硬编码在架构中。

### 6.4 研究空白
2025综述中无PINN直接用于Tg预测。**创新点。**

### 6.5 GC 残差学习（最适合本项目）
`Tg_pred = Tg_GC + model.predict(features)`

---

## 7. 对本项目的启示

### 7.1 物理特征
| 特征 | 物理依据 | 重要性 |
|------|---------|--------|
| M/f | Gibbs-DiMarzio | 极高 |
| CED_est | CED-Tg线性 | 高 |
| Yg_GC | Van Krevelen加和性 | 极高 |
| symmetry_index | 对称性效应 | 中 |
| steric_volume | 位阻 | 高 |
| Vf_est | 自由体积 | 中 |
| backbone_bond_type | 旋转能垒 | 高 |
| H-bond_density | 分子间力 | 高 |

### 7.2 残差学习
1. GC计算Tg_GC  2. ML学残差  3. Tg = Tg_GC + residual

### 7.3 单调性约束
GBR monotone_constraints, 特征工程, Isotonic Regression

### 7.4 创新点
| 维度 | 方案 | 预期 |
|------|------|------|
| 物理特征 | M/f+CED+Vf | R2+0.03-0.05 |
| 残差学习 | GC+ML | 物理合理性 |
| 单调约束 | monotone | 改善外推 |
| 对称性 | symmetry | 区分异构体 |

---

## 参考文献
1. Van Krevelen. Properties of Polymers, 4th ed., 2009
2. Bicerano. Prediction of Polymer Properties, 3rd ed., 2002
3. Liu et al. ACS Omega 2020 (GC R2=0.99)
4. Schneider & Di Marzio. PMC 2008 (M/f, MAE 3.7-6.4K)
5. Afsordeh et al. Chinese J. Polym. Sci. 2025 (R2=0.97)
6. Williams, Landel & Ferry. JACS 1955
7. Fox & Flory. J. Appl. Phys. 1950
8. Gordon & Taylor. J. Appl. Chem. 1952
9. Simha & Boyer. J. Chem. Phys. 1962
10. PINNs in Polymers. Polymers 2025
11. PENN viscosity. npj Comput. Mater. 2025

## 在线资源
- [Free Volume and Tg](https://www.eng.uc.edu/~beaucag/Classes/Processing/FreeVolumeTghtml/FreeVolumeTg.html)
- [Glass Transition](https://ncstate.pressbooks.pub/advancesinpolymerscience/chapter/glass-transition-free-volume-and-plasticization/)
- [Modified GC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7676332/)
- [M/f Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC2203329/)
- [Tg Factors](https://eng.libretexts.org/Bookshelves/Materials_Science/Supplemental_Modules_(Materials_Science)/Polymer_Chemistry/Polymer_Chemistry:_Transitions/Polymer_Chemistry:_Factors_Influencing_Tg)
- [PINNs Review](https://www.mdpi.com/2073-4360/17/8/1108)
- [Afsordeh 2025](https://link.springer.com/article/10.1007/s10118-025-3361-3)