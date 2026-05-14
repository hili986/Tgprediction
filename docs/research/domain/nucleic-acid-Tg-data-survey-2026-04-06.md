# 核酸高分子 Tg 实验数据调研

> 调研日期: 2026-04-06
> 调研目的: 搜集 DNA/RNA 及核苷酸的玻璃化转变温度 (Tg) 实验数据
> 调研范围: 1996-2025 年文献

---

## 0. 核心发现摘要

| # | 发现 | 置信度 |
|---|------|--------|
| 1 | 核酸 Tg 实验数据**极度稀缺**，全球文献中仅少数研究组系统测量过 | 高 |
| 2 | DNA Tg 水溶液 ~223-234 K (MD); 干燥状态 >>室温 | 高 |
| 3 | ATP/ADP 冻干 Tg 较高(~350-400K); 项目用 246/244/249K 可能是 Tg-prime | 中-高 |
| 4 | Cytosine Tg = **388+/-3 K** (FSC 实验, 2017) | 高 |
| 5 | 水是核酸最强增塑剂，Tg 对含水量极敏感 | 高 |
| 6 | mRNA 疑苗领域关注赋形剂 Tg，非核酸本身 | 高 |
---

## 1. DNA Tg Data

### 1.1 MD Simulation -- Lee & Olson (1996)

- **Paper**: Glass transition in DNA from molecular dynamics simulations
- **Journal**: PNAS, 1996, 93(19), 10173-10176
- **PMID**: [8816771](https://pubmed.ncbi.nlm.nih.gov/8816771/)
- **DNA**: d(CGCGCG)2 hexamer duplex
- **Method**: MD simulation, 20-340 K
- **Tg**: **223-234 K** (-50 to -39 C)
- **Condition**: aqueous solution
- **Findings**: Max H-bonds at Tg; harmonic behavior at low T

### 1.2 DSC Experiment -- Mrevlishvili (1999-2001)

- **Paper1**: Thermostability of DNA and the glassing process (1999) [PMID:10643052](https://pubmed.ncbi.nlm.nih.gov/10643052/)
- **Paper2**: Glass Transition in Humid Proteins and DNA (2001) [Springer](https://link.springer.com/article/10.1023/A:1010110727782)
- **DNA**: Calf thymus DNA
- **Method**: DSC, water content 12-92%
- **Temperature**: -30 C to 130 C
- **Heat capacity jump**: ~1.0 cal/(g*C)
- **Findings**: Tg depends on water content; at ~25% humidity Tg > RT

**Water content vs Tg estimate**:

| Water (wt%) | Est. Tg (K) | Note |
|-------------|------------|------|
| ~0% | >400 | Far above RT |
| ~12% | ~370-400 | Glassy |
| ~25% | ~300-340 | Near RT |
| ~50% | ~250-280 | Below RT |
| ~75%+ | <250 | Solution-like |

### 1.3 Project DNA dry film Tg source issue

Project E26 uses DNA dry film Tg = 448 K, attributed to "Simperler 2006".
But Simperler 2006 studied glucose/sucrose/trehalose, **NOT DNA**.
**Source needs re-tracing.**

---

## 2. Nucleotide Tg Data

### 2.1 ATP/ADP -- Kawai et al. (2002)

- **Paper**: Glass transition and enthalpy relaxation of polyphosphate compounds
- **Journal**: CryoLetters, 2002, 23(2), 79-88
- **PMID**: [12050775](https://pubmed.ncbi.nlm.nih.gov/12050775/)
- **Compounds**: ATP, ADP, di-/tri-polyphosphates
- **Method**: DSC
- **Tg**: "relatively high", depends on moisture (exact values in full text)
- **Fragility**: ATP/ADP more fragile than trehalose/sucrose
- **Estimated dry Tg**: 350-420 K

### 2.2 Project validation data: Tg vs Tg-prime issue

| Compound | Project value (K) | Source label | Audit result |
|----------|------------------|-------------|-------------|
| ATP | 246 | Simperler 2006 | **Suspicious** - Simperler did not study nucleotides |
| ADP | 244 | Simperler 2006 | **Suspicious** |
| AMP | 249 | Simperler 2006 | **Suspicious** |
| GMP | 260 | estimated | No literature support |
| UMP | 255 | estimated | No literature support |
| CMP | 252 | estimated | No literature support |

**Tg-prime hypothesis**: ATP=246K matches trehalose Tg-prime=244K closely.
These values may be Tg-prime of maximally freeze-concentrated solution, not dry-state Tg.

---

## 3. Nucleobase Tg Data

### 3.1 Cytosine (only high-reliability data)

- **Paper**: Melting temperature and heat of fusion of cytosine (2017)
- **Source**: [Thermochimica Acta](https://www.sciencedirect.com/science/article/abs/pii/S0040603117302344)
- **Method**: Fast scanning calorimetry (FSC), 6000 K/s
- **Tg**: **388 +/- 3 K** (115 C)
- **Tm**: 606 +/- 4 K
- **Heat of fusion**: 35 +/- 4 kJ/mol
- **Cold crystallization**: 448 +/- 8 K

### 3.2 Other nucleobase Tg estimates

| Nucleobase | Tg (K) | Tm (K) | Reliability |
|-----------|--------|--------|------------|
| Cytosine | 388+/-3 | 606 | High (FSC) |
| Adenine | ~400-430 | ~633 | Low (Tg/Tm est.) |
| Thymine | ~350-380 | ~583 | Low |
| Uracil | ~340-370 | ~578 | Low |
| Guanine | missing | ~633+ | None |

### 3.3 Nucleobase-drug mixtures -- MD simulation (2021)

- [PMC8400648](https://pmc.ncbi.nlm.nih.gov/articles/PMC8400648/)
- Adenine + Cytosine with APIs via MD
- Adenine increases ibuprofen Tg, slows diffusion ~10x

---

## 4. mRNA/RNA Related Data

### 4.1 Excipient Tg (indirect relevance)

| Excipient | Tg (K) | Tg-prime (K) |
|-----------|--------|-------------|
| Trehalose | 379 | 244 |
| Sucrose | 333 | 240 |
| Glucose | 310 | -- |

### 4.2 RNA dry-state stability (2025)

- [Nature](https://www.nature.com/articles/s43246-025-00850-y)
- RNA degradation correlates with water mass and crystallization
- Focuses on excipient Tg, not RNA itself

### 4.3 RNA cold glass transition (2024)

- [PNAS](https://www.pnas.org/doi/10.1073/pnas.2408313121)
- RNA hairpins undergo glass-like transition at low T

---

## 5. Data Gaps and Root Causes

| Needed Data | Status | Root Cause |
|------------|--------|-----------|
| Dry DNA polymer Tg (exp.) | Almost nonexistent | Thermal decomp. < Tg |
| poly(dA)/poly(dT) Tg | Completely missing | Same |
| RNA dry-state Tg | Nonexistent | Decomp. + hydrolysis |
| Nucleotide precise Tg | Very few | Could use FSC |
| PNA/LNA/Morpholino Tg | Completely missing | Research gap |

---

## 6. Bridge Dataset Audit

| Entry | Project (K) | FSC exp. (K) | Gap |
|-------|------------|-------------|-----|
| Cytosine monomer | 470 | 388 | +82 K |
| Adenine monomer | 500 | ~400-430 | +70-100 K |
| Thymine monomer | 460 | ~350-380 | +80-110 K |
| Uracil monomer | 465 | ~340-370 | +95-125 K |
| Guanine monomer | 510 | No data | -- |

---

## 7. Urgent Action Items

1. **Trace ATP/ADP/AMP Tg source** -- Simperler 2006 likely wrong attribution
2. **Verify bridge nucleobase Tg** -- cytosine 470K vs experimental 388K
3. **Get Kawai 2002 full text** -- precise ATP/ADP Tg and Tg-prime values
4. **Discuss validation limitations in paper**
5. **Add cytosine 388K as new validation point**

---

## 8. Key References

1. Lee & Olson (1996) PNAS 93(19) 10173. [PMID:8816771](https://pubmed.ncbi.nlm.nih.gov/8816771/)
2. Kawai et al. (2002) CryoLetters 23(2) 79. [PMID:12050775](https://pubmed.ncbi.nlm.nih.gov/12050775/)
3. Mrevlishvili (1999) Biofizika. [PMID:10643052](https://pubmed.ncbi.nlm.nih.gov/10643052/)
4. Mrevlishvili (2001) J Thermal Anal Calorim. [Springer](https://link.springer.com/article/10.1023/A:1010110727782)
5. Abdelaziz et al. (2017) Thermochim Acta. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0040603117302344)
6. Simperler et al. (2006) J Phys Chem B 110(39). [ACS](https://pubs.acs.org/doi/10.1021/jp063134t)
7. Knapik-Kowalczuk et al. (2021) Pharmaceutics 13(8). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8400648/)
8. FSC nucleobases (2020) PCCP. [RSC](https://pubs.rsc.org/en/content/articlelanding/2020/cp/c9cp04761a)
9. RNA dry-state (2025) Commun Mater. [Nature](https://www.nature.com/articles/s43246-025-00850-y)
10. RNA cold transitions (2024) PNAS. [PNAS](https://www.pnas.org/doi/10.1073/pnas.2408313121)
