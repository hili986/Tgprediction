"""
BigSMILES 示例库 — 39 个聚合物结构的 BigSMILES 编码复现
BigSMILES Example Library — 39 polymer structures with BigSMILES encodings

覆盖 13 类聚合物结构，每个条目包含：
- BigSMILES 编码
- 重复单元 SMILES（可被 RDKit 解析）
- 中英文结构说明
- ASCII 结构式
- 文献来源
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# 示例库数据 / Example Library Data
# ---------------------------------------------------------------------------

EXAMPLES = [
    # =========================================================================
    # 1. 线型均聚物 / Linear Homopolymers (8)
    # =========================================================================
    {
        "id": "1.1",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚乙烯",
        "name_en": "Polyethylene (PE)",
        "mechanism_cn": "自由基/配位聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC[$]}",
        "smiles_repeat_unit": "*CC*",
        "structure_ascii": "—[CH2-CH2]n—",
        "explanation_cn": "聚乙烯是最简单的聚合物，由乙烯单体经自由基或配位聚合而成。"
                          "重复单元为 -CH2CH2-，两端用 [$] 表示等价键连接（AA 型）。",
        "explanation_en": "Polyethylene is the simplest polymer, formed by radical or coordination "
                          "polymerization of ethylene. The repeat unit is -CH2CH2- with [$] "
                          "denoting equivalent bonding descriptors (AA-type).",
        "source": "Lin et al. 2019, Fig. 2",
    },
    {
        "id": "1.2",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚环氧乙烷/聚乙二醇",
        "name_en": "Poly(ethylene oxide) / PEG",
        "mechanism_cn": "开环聚合",
        "bonding_type": "AB",
        "bigsmiles": "{[>]CCO[<]}",
        "smiles_repeat_unit": "*CCO*",
        "structure_ascii": "—[CH2-CH2-O]n—",
        "explanation_cn": "PEG 由环氧乙烷开环聚合而成。重复单元为 -CH2CH2O-，"
                          "由于 C-O 键和 C-C 键化学不等价，使用 AB 型描述符 [>]/[<]。",
        "explanation_en": "PEG is formed by ring-opening polymerization of ethylene oxide. "
                          "The repeat unit -CH2CH2O- uses AB-type descriptors [>]/[<] because "
                          "the C-O and C-C bonds are chemically inequivalent.",
        "source": "Lin et al. 2019, Fig. 3",
    },
    {
        "id": "1.3",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚苯乙烯",
        "name_en": "Polystyrene (PS)",
        "mechanism_cn": "自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(c1ccccc1)[$]}",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "—[CH2-CH(C6H5)]n—",
        "explanation_cn": "聚苯乙烯由苯乙烯经自由基聚合而成。侧基为苯环 c1ccccc1，"
                          "主链 C-C 键等价，使用 AA 型描述符 [$]。",
        "explanation_en": "Polystyrene is formed by radical polymerization of styrene. "
                          "The pendant phenyl group is c1ccccc1, and the backbone C-C bonds "
                          "are equivalent, using AA-type [$] descriptors.",
        "source": "Lin et al. 2019, Fig. 2",
    },
    {
        "id": "1.4",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚丙烯",
        "name_en": "Polypropylene (PP)",
        "mechanism_cn": "配位聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(C)[$]}",
        "smiles_repeat_unit": "*CC(C)*",
        "structure_ascii": "—[CH2-CH(CH3)]n—",
        "explanation_cn": "聚丙烯由丙烯配位聚合而成。重复单元含甲基侧基，"
                          "主链 C-C 键等价，使用 AA 型描述符。",
        "explanation_en": "Polypropylene is formed by coordination polymerization of propylene. "
                          "The repeat unit has a methyl side group, with equivalent backbone "
                          "C-C bonds using AA-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "1.5",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚氯乙烯",
        "name_en": "Poly(vinyl chloride) (PVC)",
        "mechanism_cn": "自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(Cl)[$]}",
        "smiles_repeat_unit": "*CC(Cl)*",
        "structure_ascii": "—[CH2-CHCl]n—",
        "explanation_cn": "PVC 由氯乙烯经自由基聚合而成。氯原子为侧基，"
                          "主链 C-C 键等价，使用 AA 型描述符。",
        "explanation_en": "PVC is formed by radical polymerization of vinyl chloride. "
                          "Chlorine is the pendant group, with equivalent backbone C-C bonds "
                          "using AA-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "1.6",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚甲基丙烯酸甲酯",
        "name_en": "Poly(methyl methacrylate) (PMMA)",
        "mechanism_cn": "自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(C)(C(=O)OC)[$]}",
        "smiles_repeat_unit": "*CC(C)(C(=O)OC)*",
        "structure_ascii": "—[CH2-C(CH3)(COOCH3)]n—",
        "explanation_cn": "PMMA 由甲基丙烯酸甲酯自由基聚合而成。侧基包含甲基和甲酯基团，"
                          "主链 C-C 键等价，使用 AA 型描述符。",
        "explanation_en": "PMMA is formed by radical polymerization of methyl methacrylate. "
                          "Side groups include methyl and methyl ester, with equivalent "
                          "backbone C-C bonds using AA-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "1.7",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚四氟乙烯",
        "name_en": "Polytetrafluoroethylene (PTFE)",
        "mechanism_cn": "自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]C(F)(F)C(F)(F)[$]}",
        "smiles_repeat_unit": "*C(F)(F)C(F)(F)*",
        "structure_ascii": "—[CF2-CF2]n—",
        "explanation_cn": "PTFE 由四氟乙烯自由基聚合而成。四个氟原子取代氢，"
                          "主链 C-C 键等价，使用 AA 型描述符。",
        "explanation_en": "PTFE is formed by radical polymerization of tetrafluoroethylene. "
                          "All four hydrogens are replaced by fluorine, with equivalent "
                          "backbone C-C bonds using AA-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "1.8",
        "category_cn": "线型均聚物",
        "category_en": "Linear Homopolymer",
        "name_cn": "聚丙烯腈",
        "name_en": "Polyacrylonitrile (PAN)",
        "mechanism_cn": "自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(C#N)[$]}",
        "smiles_repeat_unit": "*CC(C#N)*",
        "structure_ascii": "—[CH2-CH(CN)]n—",
        "explanation_cn": "PAN 由丙烯腈自由基聚合而成。氰基 (-CN) 为侧基，"
                          "主链 C-C 键等价，使用 AA 型描述符。",
        "explanation_en": "PAN is formed by radical polymerization of acrylonitrile. "
                          "The cyano group (-CN) is the pendant group, with equivalent "
                          "backbone C-C bonds using AA-type descriptors.",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 2. 聚异戊二烯立体异构 / Polyisoprene Stereoisomers (4)
    # =========================================================================
    {
        "id": "2.1",
        "category_cn": "聚异戊二烯立体异构",
        "category_en": "Polyisoprene Stereoisomer",
        "name_cn": "cis-1,4-聚异戊二烯（天然橡胶）",
        "name_en": "cis-1,4-Polyisoprene (Natural Rubber)",
        "mechanism_cn": "配位聚合（Ziegler-Natta）",
        "bonding_type": "AA",
        "bigsmiles": r"{[$]C/C=C(\C)C[$]}",
        "smiles_repeat_unit": r"*C/C=C(\C)C*",
        "structure_ascii": "—[CH2-C(=CH-CH2)(CH3)]n— (cis)",
        "explanation_cn": "cis-1,4-聚异戊二烯为天然橡胶的主要成分。"
                          "双键处 cis 构型用 / 和 \\ 表示。1,4-加成产物。",
        "explanation_en": "cis-1,4-Polyisoprene is the main component of natural rubber. "
                          "The cis configuration at the double bond is denoted by / and \\. "
                          "This is a 1,4-addition product.",
        "source": "Lin et al. 2019, Fig. 5",
    },
    {
        "id": "2.2",
        "category_cn": "聚异戊二烯立体异构",
        "category_en": "Polyisoprene Stereoisomer",
        "name_cn": "trans-1,4-聚异戊二烯（杜仲胶）",
        "name_en": "trans-1,4-Polyisoprene (Gutta-percha)",
        "mechanism_cn": "配位聚合",
        "bonding_type": "AA",
        "bigsmiles": r"{[$]C/C=C(/C)C[$]}",
        "smiles_repeat_unit": r"*C/C=C(/C)C*",
        "structure_ascii": "—[CH2-C(=CH-CH2)(CH3)]n— (trans)",
        "explanation_cn": "trans-1,4-聚异戊二烯为杜仲胶的主要成分。"
                          "双键处 trans 构型用 / 和 / 表示。",
        "explanation_en": "trans-1,4-Polyisoprene is the main component of gutta-percha. "
                          "The trans configuration at the double bond is denoted by / and /.",
        "source": "Lin et al. 2019, Fig. 5",
    },
    {
        "id": "2.3",
        "category_cn": "聚异戊二烯立体异构",
        "category_en": "Polyisoprene Stereoisomer",
        "name_cn": "3,4-加成聚异戊二烯",
        "name_en": "3,4-Addition Polyisoprene",
        "mechanism_cn": "自由基/阴离子聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(C(C)=C)[$]}",
        "smiles_repeat_unit": "*CC(C(C)=C)*",
        "structure_ascii": "—[CH2-CH(C(CH3)=CH2)]n—",
        "explanation_cn": "3,4-加成聚异戊二烯中，双键保留在侧基上。"
                          "主链为饱和碳链，侧基含有异丙烯基。",
        "explanation_en": "In 3,4-addition polyisoprene, the double bond is retained "
                          "in the side group. The backbone is saturated with an "
                          "isopropenyl pendant group.",
        "source": "Lin et al. 2019, Fig. 5",
    },
    {
        "id": "2.4",
        "category_cn": "聚异戊二烯立体异构",
        "category_en": "Polyisoprene Stereoisomer",
        "name_cn": "混合微观结构聚异戊二烯",
        "name_en": "Mixed Microstructure Polyisoprene",
        "mechanism_cn": "自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": r"{[$]C/C=C(\C)C[$],[$]C/C=C(/C)C[$],[$]CC(C(C)=C)[$]}",
        "smiles_repeat_unit": r"*C/C=C(\C)C*",
        "structure_ascii": "—[cis-1,4 / trans-1,4 / 3,4]n— (random mix)",
        "explanation_cn": "混合微观结构聚异戊二烯包含 cis-1,4、trans-1,4 和 3,4 三种重复单元"
                          "的随机共聚。逗号分隔的多个重复单元表示随机排列。",
        "explanation_en": "Mixed microstructure polyisoprene contains a random copolymer of "
                          "cis-1,4, trans-1,4, and 3,4 repeat units. Comma-separated repeat "
                          "units denote random arrangement.",
        "source": "Lin et al. 2019, Fig. 5",
    },

    # =========================================================================
    # 3. 无规共聚物 / Random Copolymers (3)
    # =========================================================================
    {
        "id": "3.1",
        "category_cn": "无规共聚物",
        "category_en": "Random Copolymer",
        "name_cn": "聚(乙烯-co-丁烯)",
        "name_en": "Poly(ethylene-co-butylene)",
        "mechanism_cn": "配位共聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC[$],[$]CC(CC)[$]}",
        "smiles_repeat_unit": "*CC*",
        "structure_ascii": "—[CH2CH2, CH2CH(C2H5)]n— (random)",
        "explanation_cn": "乙烯与丁烯的无规共聚物。两种重复单元用逗号分隔，"
                          "均使用 AA 型描述符 [$]，表示随机排列。",
        "explanation_en": "Random copolymer of ethylene and butylene. Two repeat units "
                          "separated by comma, both using AA-type [$] descriptors, "
                          "indicating random arrangement.",
        "source": "Lin et al. 2019, Fig. 6",
    },
    {
        "id": "3.2",
        "category_cn": "无规共聚物",
        "category_en": "Random Copolymer",
        "name_cn": "聚(苯乙烯-co-甲基丙烯酸甲酯)",
        "name_en": "Poly(styrene-co-methyl methacrylate)",
        "mechanism_cn": "自由基共聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(c1ccccc1)[$],[$]CC(C)(C(=O)OC)[$]}",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "—[CH2CH(Ph), CH2C(CH3)(COOCH3)]n— (random)",
        "explanation_cn": "苯乙烯与甲基丙烯酸甲酯的无规共聚物。"
                          "两种乙烯基单体重复单元随机排列。",
        "explanation_en": "Random copolymer of styrene and methyl methacrylate. "
                          "Two vinyl monomer repeat units in random arrangement.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "3.3",
        "category_cn": "无规共聚物",
        "category_en": "Random Copolymer",
        "name_cn": "丁苯橡胶 (SBR)",
        "name_en": "Styrene-Butadiene Rubber (SBR)",
        "mechanism_cn": "乳液共聚合",
        "bonding_type": "AA",
        "bigsmiles": r"{[$]CC(c1ccccc1)[$],[$]C/C=C\C[$]}",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "—[CH2CH(Ph), CH2CH=CHCH2]n— (random)",
        "explanation_cn": "丁苯橡胶是苯乙烯与丁二烯的无规共聚物，"
                          "广泛用于轮胎制造。丁二烯单元含双键。",
        "explanation_en": "SBR is a random copolymer of styrene and butadiene, "
                          "widely used in tire manufacturing. The butadiene unit "
                          "contains a double bond.",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 4. 缩聚物/交替共聚物 / Condensation / Alternating Copolymers (4)
    # =========================================================================
    {
        "id": "4.1",
        "category_cn": "缩聚物/交替共聚物",
        "category_en": "Condensation / Alternating Copolymer",
        "name_cn": "尼龙-6,6",
        "name_en": "Nylon-6,6 (PA66)",
        "mechanism_cn": "缩聚反应",
        "bonding_type": "AB",
        "bigsmiles": "{[>]C(=O)CCCCC(=O)NCCCCCCN[<]}",
        "smiles_repeat_unit": "*C(=O)CCCCC(=O)NCCCCCCN*",
        "structure_ascii": "—[CO-(CH2)4-CO-NH-(CH2)6-NH]n—",
        "explanation_cn": "尼龙-6,6 由己二酸与己二胺缩聚而成。"
                          "重复单元包含酰胺键 (-CONH-)，使用 AB 型描述符 [>]/[<]"
                          "因为两端化学环境不同（羰基端 vs 胺端）。",
        "explanation_en": "Nylon-6,6 is formed by condensation of adipic acid and "
                          "hexamethylenediamine. The repeat unit contains amide bonds "
                          "(-CONH-), using AB-type [>]/[<] descriptors because the two "
                          "ends are chemically distinct (carbonyl vs amine).",
        "source": "Lin et al. 2019, Fig. 3",
    },
    {
        "id": "4.2",
        "category_cn": "缩聚物/交替共聚物",
        "category_en": "Condensation / Alternating Copolymer",
        "name_cn": "聚对苯二甲酸乙二醇酯",
        "name_en": "Poly(ethylene terephthalate) (PET)",
        "mechanism_cn": "缩聚反应",
        "bonding_type": "AB",
        "bigsmiles": "{[>]OC(=O)c1ccc(C(=O)OCC)cc1[<]}",
        "smiles_repeat_unit": "*OC(=O)c1ccc(C(=O)OCC)cc1*",
        "structure_ascii": "—[O-CO-C6H4-CO-O-CH2CH2]n—",
        "explanation_cn": "PET 由对苯二甲酸与乙二醇缩聚而成。"
                          "重复单元包含酯键和芳香环，使用 AB 型描述符。",
        "explanation_en": "PET is formed by condensation of terephthalic acid and "
                          "ethylene glycol. The repeat unit contains ester bonds and "
                          "an aromatic ring, using AB-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "4.3",
        "category_cn": "缩聚物/交替共聚物",
        "category_en": "Condensation / Alternating Copolymer",
        "name_cn": "双酚A型聚碳酸酯",
        "name_en": "Bisphenol A Polycarbonate (BPA-PC)",
        "mechanism_cn": "缩聚反应",
        "bonding_type": "AB",
        "bigsmiles": "{[>]Oc1ccc(C(C)(C)c2ccc(OC(=O))cc2)cc1[<]}",
        "smiles_repeat_unit": "*Oc1ccc(C(C)(C)c2ccc(OC(=O))cc2)cc1*",
        "structure_ascii": "—[O-C6H4-C(CH3)2-C6H4-O-CO]n—",
        "explanation_cn": "BPA-PC 由双酚A与碳酰氯（光气）缩聚而成。"
                          "重复单元包含碳酸酯键和双酚A核，使用 AB 型描述符。",
        "explanation_en": "BPA-PC is formed by condensation of bisphenol A and phosgene. "
                          "The repeat unit contains a carbonate linkage and BPA core, "
                          "using AB-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "4.4",
        "category_cn": "缩聚物/交替共聚物",
        "category_en": "Condensation / Alternating Copolymer",
        "name_cn": "聚乳酸",
        "name_en": "Polylactic Acid (PLA)",
        "mechanism_cn": "开环聚合/缩聚",
        "bonding_type": "AB",
        "bigsmiles": "{[>]OC(C)C(=O)[<]}",
        "smiles_repeat_unit": "*OC(C)C(=O)*",
        "structure_ascii": "—[O-CH(CH3)-CO]n—",
        "explanation_cn": "PLA 由乳酸缩聚或丙交酯开环聚合而成。"
                          "重复单元含手性碳和酯键，使用 AB 型描述符。",
        "explanation_en": "PLA is formed by condensation of lactic acid or ring-opening "
                          "polymerization of lactide. The repeat unit has a chiral center "
                          "and ester bond, using AB-type descriptors.",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 5. 嵌段共聚物 / Block Copolymers (3)
    # =========================================================================
    {
        "id": "5.1",
        "category_cn": "嵌段共聚物",
        "category_en": "Block Copolymer",
        "name_cn": "聚(苯乙烯-b-甲基丙烯酸甲酯)",
        "name_en": "Poly(styrene-b-methyl methacrylate) (PS-b-PMMA)",
        "mechanism_cn": "活性阴离子聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(c1ccccc1)[$]}{[$]CC(C)(C(=O)OC)[$]}",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "—[PS]—[PMMA]—",
        "explanation_cn": "PS-b-PMMA 是由两个独立的随机对象相邻表示的嵌段共聚物。"
                          "每个 {} 代表一个嵌段，共享的键连接将它们连接起来。",
        "explanation_en": "PS-b-PMMA is a block copolymer represented by two adjacent "
                          "stochastic objects. Each {} represents one block, connected "
                          "by shared bonding descriptors.",
        "source": "Lin et al. 2019, Fig. 7",
    },
    {
        "id": "5.2",
        "category_cn": "嵌段共聚物",
        "category_en": "Block Copolymer",
        "name_cn": "Pluronic (PEO-PPO-PEO 三嵌段)",
        "name_en": "Pluronic (PEO-PPO-PEO Triblock)",
        "mechanism_cn": "阴离子开环聚合",
        "bonding_type": "AB",
        "bigsmiles": "{[>]CCO[<]}{[>]CC(C)O[<]}{[>]CCO[<]}",
        "smiles_repeat_unit": "*CCO*",
        "structure_ascii": "—[PEO]—[PPO]—[PEO]—",
        "explanation_cn": "Pluronic 是 PEO-PPO-PEO 三嵌段共聚物，由三个相邻随机对象表示。"
                          "中间嵌段为聚丙二醇 (PPO)，两端为聚乙二醇 (PEO)。",
        "explanation_en": "Pluronic is a PEO-PPO-PEO triblock copolymer, represented by "
                          "three adjacent stochastic objects. The middle block is PPO, "
                          "flanked by PEO blocks.",
        "source": "Lin et al. 2019, Fig. 7",
    },
    {
        "id": "5.3",
        "category_cn": "嵌段共聚物",
        "category_en": "Block Copolymer",
        "name_cn": "SBS 三嵌段热塑性弹性体",
        "name_en": "SBS Triblock Thermoplastic Elastomer",
        "mechanism_cn": "阴离子聚合",
        "bonding_type": "AA",
        "bigsmiles": r"{[$]CC(c1ccccc1)[$]}{[$]C/C=C\C[$]}{[$]CC(c1ccccc1)[$]}",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "—[PS]—[PB]—[PS]—",
        "explanation_cn": "SBS 是苯乙烯-丁二烯-苯乙烯三嵌段共聚物，"
                          "由三个相邻随机对象表示。中间嵌段为聚丁二烯弹性体段。",
        "explanation_en": "SBS is a styrene-butadiene-styrene triblock copolymer, "
                          "represented by three adjacent stochastic objects. The middle "
                          "block is the polybutadiene elastomer segment.",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 6. 接枝共聚物 / Graft Copolymers (2)
    # =========================================================================
    {
        "id": "6.1",
        "category_cn": "接枝共聚物",
        "category_en": "Graft Copolymer",
        "name_cn": "聚(异丁烯-g-甲基丙烯酸甲酯)",
        "name_en": "Poly(isobutylene-g-methyl methacrylate) (PIB-g-PMMA)",
        "mechanism_cn": "接枝共聚",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(C)(C)[$],[$]CC(C)(C)([$]){[$]CC(C)(C(=O)OC)[$]}[$]}",
        "smiles_repeat_unit": "*CC(C)(C)*",
        "structure_ascii": "—[PIB main chain]—with PMMA branches",
        "explanation_cn": "PIB-g-PMMA 是在聚异丁烯主链上接枝 PMMA 侧链的共聚物。"
                          "嵌套的随机对象 {[$]CC(C)(C(=O)OC)[$]} 表示 PMMA 接枝链。"
                          "主链含有普通 PIB 单元和带接枝点的 PIB 单元两种重复单元。",
        "explanation_en": "PIB-g-PMMA is a graft copolymer with PMMA side chains on a "
                          "PIB backbone. The nested stochastic object represents the "
                          "PMMA graft. The backbone has both plain PIB units and "
                          "PIB units with graft points.",
        "source": "Lin et al. 2019, Fig. 8",
    },
    {
        "id": "6.2",
        "category_cn": "接枝共聚物",
        "category_en": "Graft Copolymer",
        "name_cn": "聚(苯乙烯-g-聚乙二醇)",
        "name_en": "Poly(styrene-g-PEG) (PS-g-PEG)",
        "mechanism_cn": "接枝共聚",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(c1ccccc1)[$],[$]CC(c1ccc(O{[>]CCO[<]})cc1)[$]}",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "—[PS main chain]—with PEG branches on phenyl",
        "explanation_cn": "PS-g-PEG 是在聚苯乙烯苯环上接枝 PEG 的共聚物。"
                          "PEG 接枝链通过苯环上的氧原子连接，用嵌套随机对象 {[>]CCO[<]} 表示。",
        "explanation_en": "PS-g-PEG is a graft copolymer with PEG side chains on PS "
                          "phenyl rings. The PEG graft connects via oxygen on the phenyl "
                          "ring, represented by a nested stochastic object {[>]CCO[<]}.",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 7. 支化聚合物 / Branched Polymers (2)
    # =========================================================================
    {
        "id": "7.1",
        "category_cn": "支化聚合物",
        "category_en": "Branched Polymer",
        "name_cn": "低密度聚乙烯 (LDPE)",
        "name_en": "Low-Density Polyethylene (LDPE)",
        "mechanism_cn": "高压自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC[$],[$]CC([$])[$]}",
        "smiles_repeat_unit": "*CC*",
        "structure_ascii": "—[CH2CH2]n— with branches",
        "explanation_cn": "LDPE 含有支化结构。除了线性 PE 重复单元外，"
                          "还有三功能度支化点单元 CC([$])[$]，其中碳原子连接三个链段。",
        "explanation_en": "LDPE contains branching structures. In addition to linear PE "
                          "repeat units, it has trifunctional branch point units CC([$])[$] "
                          "where a carbon connects to three chain segments.",
        "source": "Lin et al. 2019, Fig. 9",
    },
    {
        "id": "7.2",
        "category_cn": "支化聚合物",
        "category_en": "Branched Polymer",
        "name_cn": "超支化聚酯",
        "name_en": "Hyperbranched Polyester",
        "mechanism_cn": "缩聚反应",
        "bonding_type": "AB",
        "bigsmiles": "{[>]OCC(COC(=O))(COC(=O))[<],[>]OCC(CO)(COC(=O))[<]}",
        "smiles_repeat_unit": "*OCC(COC(=O))(COC(=O))*",
        "structure_ascii": "Hyperbranched polyester (dendritic + linear units)",
        "explanation_cn": "超支化聚酯包含树枝状单元（全反应三醇）和线性单元（部分反应三醇）。"
                          "两种重复单元的混合产生超支化拓扑结构。",
        "explanation_en": "Hyperbranched polyester contains dendritic units (fully reacted "
                          "triol) and linear units (partially reacted triol). The mixture "
                          "of both repeat units produces a hyperbranched topology.",
        "source": "Lin et al. 2019, Fig. 9",
    },

    # =========================================================================
    # 8. 网状/交联聚合物 / Network / Crosslinked Polymers (2)
    # =========================================================================
    {
        "id": "8.1",
        "category_cn": "网状/交联聚合物",
        "category_en": "Network / Crosslinked Polymer",
        "name_cn": "硫化天然橡胶",
        "name_en": "Vulcanized Natural Rubber",
        "mechanism_cn": "硫化交联",
        "bonding_type": "AA",
        "bigsmiles": r"{[$]C/C=C(\C)C[$],[$]CC(C)(CS[$])C[$]}",
        "smiles_repeat_unit": r"*C/C=C(\C)C*",
        "structure_ascii": "—[cis-1,4-PI]n— with sulfur crosslinks",
        "explanation_cn": "硫化橡胶由天然橡胶经硫化交联而成。"
                          "包含正常 cis-1,4 聚异戊二烯重复单元和含硫交联点的单元。"
                          "硫原子通过 [$] 连接到其他链段形成网络结构。",
        "explanation_en": "Vulcanized rubber is formed by sulfur crosslinking of natural rubber. "
                          "It contains normal cis-1,4 PI repeat units and units with sulfur "
                          "crosslink points. Sulfur atoms connect to other chains via [$].",
        "source": "Lin et al. 2019, Fig. 10",
    },
    {
        "id": "8.2",
        "category_cn": "网状/交联聚合物",
        "category_en": "Network / Crosslinked Polymer",
        "name_cn": "环氧-胺网络",
        "name_en": "Epoxy-Amine Network",
        "mechanism_cn": "逐步聚合/交联",
        "bonding_type": "AB",
        "bigsmiles": "{[>]CC(O)CN([<])[<]}",
        "smiles_repeat_unit": "*CC(O)CN(*)*",
        "structure_ascii": "Epoxy-amine crosslinked network",
        "explanation_cn": "环氧-胺网络由环氧基团与胺基反应形成。"
                          "氮原子上有两个 [<] 描述符，形成交联点（多功能度节点）。",
        "explanation_en": "Epoxy-amine network is formed by reaction of epoxy and amine groups. "
                          "The nitrogen has two [<] descriptors, creating crosslink points "
                          "(multifunctional nodes).",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 9. 环形聚合物 / Cyclic Polymers (2)
    # =========================================================================
    {
        "id": "9.1",
        "category_cn": "环形聚合物",
        "category_en": "Cyclic Polymer",
        "name_cn": "环形聚苯乙烯",
        "name_en": "Cyclic Polystyrene (click chemistry)",
        "mechanism_cn": "活性聚合 + 点击化学环化",
        "bonding_type": "AA",
        "bigsmiles": "C1(CCOCC1)n2cc(nn2){[$]CC(c1ccccc1)[$]}n1cc(nn1)COCCOCC",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "cyclic-[PS]—triazole—linker—triazole (ring)",
        "explanation_cn": "环形聚苯乙烯通过点击化学环化合成。SMILES 中的环闭合"
                          "（通过原子编号对应）使整个结构形成环。"
                          "随机对象 {[$]CC(c1ccccc1)[$]} 嵌入在环形骨架中。",
        "explanation_en": "Cyclic polystyrene is synthesized by click chemistry cyclization. "
                          "The ring closure in SMILES (via matching atom numbers) makes the "
                          "entire structure cyclic. The stochastic object is embedded in "
                          "the cyclic backbone.",
        "source": "Lin et al. 2019, Fig. 12",
    },
    {
        "id": "9.2",
        "category_cn": "环形聚合物",
        "category_en": "Cyclic Polymer",
        "name_cn": "环形聚乙二醇",
        "name_en": "Cyclic PEG",
        "mechanism_cn": "环化缩合",
        "bonding_type": "AB",
        "bigsmiles": "C1(CC1){[>]CCO[<]}",
        "smiles_repeat_unit": "*CCO*",
        "structure_ascii": "cyclic-[PEG] (ring)",
        "explanation_cn": "环形 PEG 通过链端环化形成。起始和终止原子通过环闭合数字配对"
                          "形成封闭环路，随机对象表示 PEG 链段。",
        "explanation_en": "Cyclic PEG is formed by end-to-end cyclization. The starting "
                          "and ending atoms pair via ring closure digits to form a closed "
                          "loop, with the stochastic object representing the PEG segment.",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 10. 端基指定 / End-Group Specification (2)
    # =========================================================================
    {
        "id": "10.1",
        "category_cn": "端基指定",
        "category_en": "End-Group Specification",
        "name_cn": "AIBN引发的聚苯乙烯（确定性端基）",
        "name_en": "AIBN-initiated PS (deterministic end groups)",
        "mechanism_cn": "自由基聚合 (AIBN)",
        "bonding_type": "AA",
        "bigsmiles": "CC(C)(C#N)CC(C)(C#N){[$]CC(c1ccccc1)[$]}CC(C)(C#N)CC(C)(C#N)",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "AIBN—[PS]n—AIBN (deterministic ends)",
        "explanation_cn": "AIBN 引发的 PS，端基为确定性的 AIBN 残基片段。"
                          "端基直接写在 {} 外部作为普通 SMILES 片段，"
                          "与随机对象的键连接描述符相连。",
        "explanation_en": "AIBN-initiated PS with deterministic AIBN-derived end groups. "
                          "End groups are written outside {} as ordinary SMILES fragments, "
                          "connected to the stochastic object via bonding descriptors.",
        "source": "Lin et al. 2019, Fig. 11",
    },
    {
        "id": "10.2",
        "category_cn": "端基指定",
        "category_en": "End-Group Specification",
        "name_cn": "含随机端基的聚苯乙烯",
        "name_en": "PS with stochastic end groups",
        "mechanism_cn": "自由基聚合",
        "bonding_type": "AA",
        "bigsmiles": "{[$]CC(c1ccccc1)[$];[$]CC(C)(C#N)CC(C)(C#N),[$]Cl}",
        "smiles_repeat_unit": "*CC(c1ccccc1)*",
        "structure_ascii": "{[PS]n; AIBN-frag | Cl} (stochastic ends)",
        "explanation_cn": "含随机端基的 PS。分号 ';' 后列出可能的端基，"
                          "逗号分隔多种端基选择（AIBN 残基或 Cl）。"
                          "每个端基只含一个描述符。",
        "explanation_en": "PS with stochastic end groups. After semicolon ';' are possible "
                          "end groups, comma-separated (AIBN fragment or Cl). "
                          "Each end group has exactly one descriptor.",
        "source": "Lin et al. 2019, Fig. 11",
    },

    # =========================================================================
    # 11. 嵌套随机对象 / Nested Stochastic Object (1)
    # =========================================================================
    {
        "id": "11.1",
        "category_cn": "嵌套随机对象",
        "category_en": "Nested Stochastic Object",
        "name_cn": "聚氨酯（PEG软段）",
        "name_en": "Polyurethane with PEG Soft Segment",
        "mechanism_cn": "逐步聚合",
        "bonding_type": "AB",
        "bigsmiles": "{[>]C(=O)Nc1ccc(CC2=CC=C(NC(=O){[>]CCO[<]})C=C2)cc1[<]}",
        "smiles_repeat_unit": "*C(=O)Nc1ccc(CC2=CC=C(NC(=O))C=C2)cc1*",
        "structure_ascii": "—[MDI-urethane—PEG]n—",
        "explanation_cn": "聚氨酯含嵌套随机对象：外层为 MDI-异氰酸酯与多元醇的缩聚结构，"
                          "内层 {[>]CCO[<]} 表示 PEG 软段。"
                          "嵌套结构体现了多级聚合物层次。",
        "explanation_en": "Polyurethane with nested stochastic object: the outer layer is "
                          "MDI-isocyanate/polyol condensation, the inner {[>]CCO[<]} "
                          "represents the PEG soft segment. Nesting captures multi-level "
                          "polymer hierarchy.",
        "source": "Lin et al. 2019, Fig. 8",
    },

    # =========================================================================
    # 12. 其他重要聚合物 / Other Important Polymers (4)
    # =========================================================================
    {
        "id": "12.1",
        "category_cn": "其他重要聚合物",
        "category_en": "Other Important Polymer",
        "name_cn": "尼龙-6",
        "name_en": "Nylon-6 (PA6)",
        "mechanism_cn": "开环聚合（己内酰胺）",
        "bonding_type": "AB",
        "bigsmiles": "{[>]CCCCCNC(=O)[<]}",
        "smiles_repeat_unit": "*CCCCCNC(=O)*",
        "structure_ascii": "—[(CH2)5-NH-CO]n—",
        "explanation_cn": "尼龙-6 由己内酰胺开环聚合而成。"
                          "重复单元含 5 个亚甲基和一个酰胺键，使用 AB 型描述符。",
        "explanation_en": "Nylon-6 is formed by ring-opening polymerization of caprolactam. "
                          "The repeat unit has 5 methylene groups and one amide bond, "
                          "using AB-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "12.2",
        "category_cn": "其他重要聚合物",
        "category_en": "Other Important Polymer",
        "name_cn": "聚己内酯",
        "name_en": "Polycaprolactone (PCL)",
        "mechanism_cn": "开环聚合",
        "bonding_type": "AB",
        "bigsmiles": "{[>]CCCCCOC(=O)[<]}",
        "smiles_repeat_unit": "*CCCCCOC(=O)*",
        "structure_ascii": "—[(CH2)5-O-CO]n—",
        "explanation_cn": "PCL 由己内酯开环聚合而成。"
                          "重复单元含 5 个亚甲基和一个酯键，使用 AB 型描述符。",
        "explanation_en": "PCL is formed by ring-opening polymerization of caprolactone. "
                          "The repeat unit has 5 methylene groups and one ester bond, "
                          "using AB-type descriptors.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "12.3",
        "category_cn": "其他重要聚合物",
        "category_en": "Other Important Polymer",
        "name_cn": "聚酰胺酰亚胺",
        "name_en": "Polyamide-imide (PAI)",
        "mechanism_cn": "缩聚反应",
        "bonding_type": "AB",
        "bigsmiles": "{[>]c1cc(C(=O)Nc2ccc(cc2))c2c(c1)C(=O)N(C2=O)[<]}",
        "smiles_repeat_unit": "*c1cc(C(=O)Nc2ccc(cc2))c2c(c1)C(=O)N(C2=O)*",
        "structure_ascii": "—[TMA-imide—ODA-amide]n—",
        "explanation_cn": "聚酰胺酰亚胺包含酰亚胺环和酰胺键。"
                          "由苯偏三酸酐和二胺缩聚而成，兼具聚酰亚胺的耐热性和聚酰胺的加工性。",
        "explanation_en": "Polyamide-imide contains imide rings and amide bonds. "
                          "Formed by condensation of trimellitic anhydride and diamine, "
                          "combining thermal stability of polyimide with processability of polyamide.",
        "source": "Lin et al. 2019",
    },
    {
        "id": "12.4",
        "category_cn": "其他重要聚合物",
        "category_en": "Other Important Polymer",
        "name_cn": "聚二甲基硅氧烷",
        "name_en": "Polydimethylsiloxane (PDMS)",
        "mechanism_cn": "开环聚合/缩聚",
        "bonding_type": "AB",
        "bigsmiles": "{[>]O[Si](C)(C)[<]}",
        "smiles_repeat_unit": "*O[Si](C)(C)*",
        "structure_ascii": "—[O-Si(CH3)2]n—",
        "explanation_cn": "PDMS 由环硅氧烷开环聚合或氯硅烷水解缩聚而成。"
                          "重复单元含 Si-O 键和两个甲基侧基，使用 AB 型描述符。",
        "explanation_en": "PDMS is formed by ring-opening polymerization of cyclosiloxanes "
                          "or hydrolytic condensation of chlorosilanes. The repeat unit "
                          "has Si-O bonds and two methyl groups, using AB-type descriptors.",
        "source": "Lin et al. 2019",
    },

    # =========================================================================
    # 13. 核酸聚合物 / Nucleic Acid Polymers (2)
    # =========================================================================
    {
        "id": "13.1",
        "category_cn": "核酸聚合物",
        "category_en": "Nucleic Acid Polymer",
        "name_cn": "单链DNA（PD-L1适配体）",
        "name_en": "ssDNA (PD-L1 Aptamer)",
        "mechanism_cn": "固相亚磷酰胺合成",
        "bonding_type": "AB",
        "bigsmiles": "O{[>]CC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)O[<],"
                     "[>]CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3OP(=O)(O)O[<],"
                     "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3OP(=O)(O)O[<],"
                     "[>]CC3OC(n1ccc(N)nc1=O)CC3OP(=O)(O)O[<];"
                     "[>]CC3OC(n1cnc2c(N)ncnc12)CC3O,"
                     "[>]CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3O,"
                     "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3O,"
                     "[>]CC3OC(n1ccc(N)nc1=O)CC3O}",
        "smiles_repeat_unit": "*CC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)O*",
        "structure_ascii": "5'-HO—[dNMP]n—OH-3'  (dN = dA, dT, dG, dC)",
        "explanation_cn": "单链DNA（ssDNA）由脱氧核苷酸通过3',5'-磷酸二酯键连接而成。"
                          "4种重复单元分别对应腺嘌呤(dA)、胸腺嘧啶(dT)、"
                          "鸟嘌呤(dG)、胞嘧啶(dC)脱氧核苷酸，逗号分隔表示随机排列。"
                          "每个重复单元包含脱氧核糖五元环(环闭合编号3)和磷酸二酯桥连基团。"
                          "5'端为确定性羟基端基(O)，3'端为随机核苷端基(无尾部磷酸)。"
                          "此示例对应PD-L1适配体序列 "
                          "5'-ACGGGCCACATCAACTCATTGATAGACAATGCGTCCACTGCCCGT-3'（44 nt），"
                          "但BigSMILES仅能表示聚合物类别，无法编码特定碱基序列。"
                          "使用AB型描述符[>]/[<]反映磷酸二酯键的方向性(3'→5')。",
        "explanation_en": "Single-stranded DNA (ssDNA) consists of deoxyribonucleotides linked "
                          "by 3',5'-phosphodiester bonds. The 4 repeat units correspond to "
                          "deoxyadenosine (dA), deoxythymidine (dT), deoxyguanosine (dG), and "
                          "deoxycytidine (dC), comma-separated to indicate stochastic arrangement. "
                          "Each repeat unit contains a deoxyribose furanose ring (ring closure 3) "
                          "and a phosphodiester bridging group. The 5' end has a deterministic "
                          "hydroxyl (O), while the 3' end has stochastic nucleoside end groups "
                          "(without trailing phosphate). This example corresponds to the PD-L1 "
                          "aptamer sequence "
                          "5'-ACGGGCCACATCAACTCATTGATAGACAATGCGTCCACTGCCCGT-3' (44 nt), "
                          "but BigSMILES can only represent the polymer class, not the specific "
                          "base sequence. AB-type descriptors [>]/[<] reflect the directionality "
                          "of the phosphodiester backbone (3'->5').",
        "source": "BigSMILES nucleic acid extension (this work)",
    },
    {
        "id": "13.2",
        "category_cn": "核酸聚合物",
        "category_en": "Nucleic Acid Polymer",
        "name_cn": "单链RNA",
        "name_en": "ssRNA (Single-Stranded RNA)",
        "mechanism_cn": "固相合成/体外转录",
        "bonding_type": "AB",
        "bigsmiles": "O{[>]CC3OC(n1cnc2c(N)ncnc12)C(O)C3OP(=O)(O)O[<],"
                     "[>]CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3OP(=O)(O)O[<],"
                     "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3OP(=O)(O)O[<],"
                     "[>]CC3OC(n1ccc(N)nc1=O)C(O)C3OP(=O)(O)O[<];"
                     "[>]CC3OC(n1cnc2c(N)ncnc12)C(O)C3O,"
                     "[>]CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3O,"
                     "[>]CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3O,"
                     "[>]CC3OC(n1ccc(N)nc1=O)C(O)C3O}",
        "smiles_repeat_unit": "*CC3OC(n1cnc2c(N)ncnc12)C(O)C3OP(=O)(O)O*",
        "structure_ascii": "5'-HO—[NMP]n—OH-3'  (N = A, U, G, C)",
        "explanation_cn": "单链RNA（ssRNA）与ssDNA的区别在于两点：(1) 糖基为核糖而非脱氧核糖，"
                          "C2'位含羟基 C(O)（对比DNA的 CC）；(2) 碱基中以尿嘧啶(U)替代胸腺嘧啶(T)，"
                          "尿嘧啶无5位甲基。4种重复单元对应腺苷(A)、尿苷(U)、鸟苷(G)、胞苷(C)。"
                          "主链同样通过3',5'-磷酸二酯键连接，使用AB型描述符。"
                          "2'-OH的存在使RNA比DNA更易水解，也赋予其更丰富的二级结构。",
        "explanation_en": "Single-stranded RNA (ssRNA) differs from ssDNA in two ways: "
                          "(1) the sugar is ribose instead of deoxyribose, with a 2'-hydroxyl "
                          "C(O) (vs CC in DNA); (2) uracil (U) replaces thymine (T), lacking "
                          "the 5-methyl group. The 4 repeat units correspond to adenosine (A), "
                          "uridine (U), guanosine (G), and cytidine (C). The backbone is also "
                          "linked by 3',5'-phosphodiester bonds using AB-type descriptors. "
                          "The 2'-OH makes RNA more susceptible to hydrolysis than DNA and "
                          "enables richer secondary structures.",
        "source": "BigSMILES nucleic acid extension (this work)",
    },
]


# ---------------------------------------------------------------------------
# 辅助函数 / Utility Functions
# ---------------------------------------------------------------------------

def get_examples():
    """返回全部示例列表。/ Return all examples."""
    return EXAMPLES


def to_json(filepath=None):
    """将示例库导出为 JSON 文件。/ Export examples to JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "output", "bigsmiles_examples.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(EXAMPLES, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 已导出 {len(EXAMPLES)} 个示例到 / Exported {len(EXAMPLES)} examples to: {filepath}")
    return filepath


def generate_images(output_dir=None):
    """
    用 RDKit 为每个重复单元生成 PNG 结构图。
    Generate PNG structure images for each repeat unit using RDKit.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        print("[WARN] RDKit 未安装，跳过图片生成。/ RDKit not installed, skipping image generation.")
        return []

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output", "images")
    os.makedirs(output_dir, exist_ok=True)

    generated = []
    for ex in EXAMPLES:
        smiles = ex["smiles_repeat_unit"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[WARN] 无法解析 / Cannot parse SMILES for {ex['id']} ({ex['name_en']}): {smiles}")
            continue
        filename = f"{ex['id']}_{ex['name_en'].replace(' ', '_').replace('/', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        Draw.MolToFile(mol, filepath, size=(400, 300))
        generated.append(filepath)
        print(f"[OK] {ex['id']} {ex['name_en']} -> {filename}")

    print(f"\n[INFO] 共生成 / Generated {len(generated)}/{len(EXAMPLES)} 张结构图 / images in: {output_dir}")
    return generated


def print_library():
    """终端格式化输出全部示例。/ Print all examples in formatted terminal output."""
    current_category = None
    for ex in EXAMPLES:
        if ex["category_cn"] != current_category:
            current_category = ex["category_cn"]
            print("\n" + "=" * 80)
            print(f"  {ex['category_cn']} / {ex['category_en']}")
            print("=" * 80)

        print(f"\n  [{ex['id']}] {ex['name_cn']} / {ex['name_en']}")
        print(f"  BigSMILES:    {ex['bigsmiles']}")
        print(f"  Repeat SMILES:{ex['smiles_repeat_unit']}")
        print(f"  Bonding Type: {ex['bonding_type']}")
        print(f"  Mechanism:    {ex['mechanism_cn']}")
        print(f"  Structure:    {ex['structure_ascii']}")
        print(f"  Source:       {ex['source']}")
        print(f"  CN: {ex['explanation_cn']}")
        print(f"  EN: {ex['explanation_en']}")
        print("-" * 80)

    print(f"\n  Total: {len(EXAMPLES)} examples across "
          f"{len(set(e['category_cn'] for e in EXAMPLES))} categories\n")


# ---------------------------------------------------------------------------
# 主入口 / Main Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        to_json()
    elif len(sys.argv) > 1 and sys.argv[1] == "--images":
        generate_images()
    else:
        print_library()
        print("\n[提示/Tip] 使用 --json 导出 JSON, 使用 --images 生成结构图")
