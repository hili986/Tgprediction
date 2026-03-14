"""
Bridge Polymer Dataset for H-bond Transfer Learning
桥梁聚合物数据集 — 氢键迁移学习

7 families sharing H-bond patterns with nucleic acids:
    PA (polyamide), PU (polyurethane), PI (polyimide),
    polyurea, polyphosphoester, polyphosphazene, nucleoside

Data sources: van Krevelen (2009), PolyInfo, Bicerano (2002), literature.
Generated variations use established structure-property correlations.

Public API:
    load_all_bridge_data() -> list[dict]
    build_bridge_dataset(layer) -> (X, y, names, feat_names)
    get_bridge_smiles() -> list[str]
    export_csv(path)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data entry format: (name, smiles, tg_k, family, hbond_pattern, source)
# ---------------------------------------------------------------------------

# Core hand-curated bridge polymer data with literature Tg values
_CORE_DATA: Tuple[Tuple[str, str, int, str, str, str], ...] = (
    # ===== Polyamides (PA) — aromatic & semi-aromatic =====
    ("PPTA (Kevlar)", "*Nc1ccc(cc1)NC(=O)c1ccc(cc1)C(*)=O", 630, "PA", "amide", "Bicerano 2002"),
    ("PMIA (Nomex)", "*Nc1cccc(c1)NC(=O)c1cccc(c1)C(*)=O", 546, "PA", "amide", "Mark 1999"),
    ("Poly(p-benzamide)", "*NC(=O)c1ccc(*)cc1", 570, "PA", "amide", "Mark 1999"),
    ("PA 6T", "*NCCCCCCNC(=O)c1ccc(cc1)C(*)=O", 440, "PA", "amide", "Kohan 2003"),
    ("PA 6I", "*NCCCCCCNC(=O)c1cccc(c1)C(*)=O", 400, "PA", "amide", "Kohan 2003"),
    ("PA 4T", "*NCCCCNC(=O)c1ccc(cc1)C(*)=O", 460, "PA", "amide", "estimated"),
    ("PA 10T", "*NCCCCCCCCCCNC(=O)c1ccc(cc1)C(*)=O", 410, "PA", "amide", "Kohan 2003"),
    ("PA 12T", "*NCCCCCCCCCCCCNC(=O)c1ccc(cc1)C(*)=O", 400, "PA", "amide", "estimated"),
    ("PA MXD6", "*NCc1cccc(c1)CNC(=O)CCCCC(*)=O", 358, "PA", "amide", "Mitsubishi Gas"),
    ("PA MXDI-6", "*NCCCCCCNC(=O)c1cccc(c1)C(*)=O", 395, "PA", "amide", "estimated"),
    ("Poly(m-phenylene adipamide)", "*Nc1cccc(c1)NC(=O)CCCCC(*)=O", 380, "PA", "amide", "estimated"),
    ("Poly(p-phenylene adipamide)", "*Nc1ccc(cc1)NC(=O)CCCCC(*)=O", 400, "PA", "amide", "estimated"),
    ("PA 6-3T", "*NCCCCCCNC(=O)c1cc(C)cc(c1)C(*)=O", 420, "PA", "amide", "estimated"),
    ("Poly(trimethylhexamethylene terephthalamide)", "*NC(C)(C)CCCC(C)(C)NC(=O)c1ccc(cc1)C(*)=O", 470, "PA", "amide", "estimated"),
    ("Poly(p-phenylene sebacamide)", "*Nc1ccc(cc1)NC(=O)CCCCCCCCC(*)=O", 360, "PA", "amide", "estimated"),

    # ===== Polyurethanes (PU) =====
    # HDI-based (aliphatic)
    ("PU HDI-EG", "*OC(=O)NCCCCCCNC(=O)OCC*", 290, "PU", "urethane", "Szycher 2013"),
    ("PU HDI-PG", "*OC(=O)NCCCCCCNC(=O)OC(C)C*", 280, "PU", "urethane", "estimated"),
    ("PU HDI-BDO", "*OC(=O)NCCCCCCNC(=O)OCCCC*", 278, "PU", "urethane", "Szycher 2013"),
    ("PU HDI-HD", "*OC(=O)NCCCCCCNC(=O)OCCCCCC*", 265, "PU", "urethane", "estimated"),
    ("PU HDI-OD", "*OC(=O)NCCCCCCNC(=O)OCCCCCCCC*", 255, "PU", "urethane", "estimated"),
    ("PU HDI-DD", "*OC(=O)NCCCCCCNC(=O)OCCCCCCCCCC*", 245, "PU", "urethane", "estimated"),
    ("PU HDI-NPG", "*OC(=O)NCCCCCCNC(=O)OCC(C)(C)C*", 295, "PU", "urethane", "estimated"),
    # MDI-based (aromatic)
    ("PU MDI-EG", "*OC(=O)Nc1ccc(Cc2ccc(NC(=O)OCC*)cc2)cc1", 393, "PU", "urethane", "Szycher 2013"),
    ("PU MDI-BDO", "*OC(=O)Nc1ccc(Cc2ccc(NC(=O)OCCCC*)cc2)cc1", 373, "PU", "urethane", "Szycher 2013"),
    ("PU MDI-HD", "*OC(=O)Nc1ccc(Cc2ccc(NC(=O)OCCCCCC*)cc2)cc1", 355, "PU", "urethane", "estimated"),
    ("PU MDI-DEG", "*OC(=O)Nc1ccc(Cc2ccc(NC(=O)OCCOCC*)cc2)cc1", 340, "PU", "urethane", "estimated"),
    ("PU MDI-NPG", "*OC(=O)Nc1ccc(Cc2ccc(NC(=O)OCC(C)(C)C*)cc2)cc1", 385, "PU", "urethane", "estimated"),
    # TDI-based
    ("PU TDI-EG", "*OC(=O)Nc1ccc(C)c(NC(=O)OCC*)c1", 370, "PU", "urethane", "estimated"),
    ("PU TDI-BDO", "*OC(=O)Nc1ccc(C)c(NC(=O)OCCCC*)c1", 353, "PU", "urethane", "estimated"),
    ("PU TDI-HD", "*OC(=O)Nc1ccc(C)c(NC(=O)OCCCCCC*)c1", 335, "PU", "urethane", "estimated"),
    # IPDI-based (cycloaliphatic)
    ("PU IPDI-EG", "*OC(=O)NC1CC(CC(C1)(C)C)NC(=O)OCC*", 350, "PU", "urethane", "estimated"),
    ("PU IPDI-BDO", "*OC(=O)NC1CC(CC(C1)(C)C)NC(=O)OCCCC*", 340, "PU", "urethane", "estimated"),
    ("PU IPDI-HD", "*OC(=O)NC1CC(CC(C1)(C)C)NC(=O)OCCCCCC*", 325, "PU", "urethane", "estimated"),
    # HMDI-based (H12MDI, cycloaliphatic)
    ("PU HMDI-EG", "*OC(=O)NC1CCC(CC1)CC1CCC(CC1)NC(=O)OCC*", 345, "PU", "urethane", "estimated"),
    ("PU HMDI-BDO", "*OC(=O)NC1CCC(CC1)CC1CCC(CC1)NC(=O)OCCCC*", 330, "PU", "urethane", "estimated"),

    # ===== Polyimides (PI) =====
    # Simple imide-containing structures: C(=O)N(R)C(=O) pattern
    ("PI PMDA-ODA", "*N(C(=O)c1cc(C(=O)*)cc(c1)C(=O)*)c1ccc(Oc2ccc(*)cc2)cc1", 673, "PI", "imide", "DuPont (Kapton)"),
    ("PI PMDA-PDA (p)", "*N1C(=O)c2cc3C(=O)N(c4ccc(*)cc4)C(=O)c3cc2C1=O", 700, "PI", "imide", "estimated"),
    ("PI PMDA-MDA", "*N1C(=O)c2cc3C(=O)N(Cc4ccc(*)cc4)C(=O)c3cc2C1=O", 650, "PI", "imide", "estimated"),
    ("Poly(succinimide)", "*N1C(=O)CCC1=O", 480, "PI", "imide", "estimated"),
    ("Poly(maleimide)", "*N1C(=O)C=CC1=O", 520, "PI", "imide", "estimated"),
    ("Poly(phthalimide-phenyl)", "*c1ccc(cc1)N1C(=O)c2ccccc2C1=O", 580, "PI", "imide", "Mark 1999"),
    ("Poly(phthalimide-ether-phenyl)", "*c1ccc(Oc2ccc(N3C(=O)c4ccccc4C3=O)cc2)cc1", 550, "PI", "imide", "estimated"),
    ("Poly(phthalimide-methylene)", "*CN1C(=O)c2ccccc2C1=O", 490, "PI", "imide", "estimated"),
    ("Poly(phthalimide-hexamethylene)", "*CCCCCCN1C(=O)c2ccccc2C1=O", 460, "PI", "imide", "estimated"),
    ("Poly(naphthalimide-phenyl)", "*c1ccc(cc1)N1C(=O)c2cc3ccccc3cc2C1=O", 620, "PI", "imide", "estimated"),
    ("Poly(4,4-ODA phthalimide-sulfone)", "*c1ccc(S(=O)(=O)c2ccc(N3C(=O)c4ccccc4C3=O)cc2)cc1", 560, "PI", "imide", "estimated"),
    ("Poly(imide-carbonate)", "*OC(=O)Oc1ccc(cc1)N1C(=O)c2ccccc2C1=O", 510, "PI", "imide", "estimated"),
    ("Poly(ether imide) PEI", "*c1ccc(Oc2ccc(N3C(=O)c4ccccc4C3=O)cc2)cc1", 489, "PI", "imide", "SABIC (Ultem)"),
    ("Poly(imide siloxane)", "*N1C(=O)c2ccccc2C1=O", 470, "PI", "imide", "estimated"),

    # ===== Polyureas =====
    ("Poly(hexamethylene urea)", "*NCCCCCCNC(=O)N*", 365, "polyurea", "urea", "Mattia 2007"),
    ("Poly(tetramethylene urea)", "*NCCCCNC(=O)N*", 385, "polyurea", "urea", "estimated"),
    ("Poly(ethylene urea)", "*NCCNC(=O)N*", 410, "polyurea", "urea", "estimated"),
    ("Poly(octamethylene urea)", "*NCCCCCCCCNC(=O)N*", 345, "polyurea", "urea", "estimated"),
    ("Poly(decamethylene urea)", "*NCCCCCCCCCCNC(=O)N*", 335, "polyurea", "urea", "estimated"),
    ("Poly(dodecamethylene urea)", "*NCCCCCCCCCCCCNC(=O)N*", 330, "polyurea", "urea", "estimated"),
    ("Poly(MDI urea)", "*NC(=O)Nc1ccc(Cc2ccc(NC(=O)N*)cc2)cc1", 450, "polyurea", "urea", "estimated"),
    ("Poly(TDI urea)", "*NC(=O)Nc1ccc(C)c(NC(=O)N*)c1", 430, "polyurea", "urea", "estimated"),
    ("Poly(IPDI urea)", "*NC(=O)NC1CC(CC(C1)(C)C)NC(=O)N*", 395, "polyurea", "urea", "estimated"),
    ("Poly(XDI urea)", "*NC(=O)NCc1ccc(CNC(=O)N*)cc1", 415, "polyurea", "urea", "estimated"),
    ("Poly(piperazine urea)", "*NC(=O)N1CCNCC1*", 420, "polyurea", "urea", "estimated"),
    ("Poly(m-phenylene urea)", "*Nc1cccc(c1)NC(=O)N*", 440, "polyurea", "urea", "estimated"),
    ("Poly(p-phenylene urea)", "*Nc1ccc(cc1)NC(=O)N*", 460, "polyurea", "urea", "estimated"),
    ("Poly(HMDI urea)", "*NC(=O)NC1CCC(CC1)CC1CCC(CC1)NC(=O)N*", 380, "polyurea", "urea", "estimated"),

    # ===== Polyphosphoesters =====
    ("Poly(methyl ethylene phosphonate)", "*CCOP(=O)(OC)O*", 260, "polyphosphoester", "phosphoester", "Wang 2001"),
    ("Poly(ethyl ethylene phosphonate)", "*CCOP(=O)(OCC)O*", 245, "polyphosphoester", "phosphoester", "Wang 2001"),
    ("Poly(isopropyl ethylene phosphonate)", "*CCOP(=O)(OC(C)C)O*", 235, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(phenyl ethylene phosphonate)", "*CCOP(=O)(Oc1ccccc1)O*", 290, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(methyl propylene phosphonate)", "*CCCOP(=O)(OC)O*", 240, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(ethyl propylene phosphonate)", "*CCCOP(=O)(OCC)O*", 230, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(methyl butylene phosphonate)", "*CCCCOP(=O)(OC)O*", 230, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(ethyl butylene phosphonate)", "*CCCCOP(=O)(OCC)O*", 225, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(methyl pentylene phosphonate)", "*CCCCCOP(=O)(OC)O*", 222, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(methyl hexylene phosphonate)", "*CCCCCCOP(=O)(OC)O*", 218, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(phenyl propylene phosphonate)", "*CCCOP(=O)(Oc1ccccc1)O*", 275, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(phenyl butylene phosphonate)", "*CCCCOP(=O)(Oc1ccccc1)O*", 265, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(bisphenol A ethyl phosphonate)", "*c1ccc(cc1)C(C)(c1ccc(OP(=O)(OCC)O*)cc1)C", 325, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(ethylene phosphate)", "*CCOP(=O)(O)O*", 270, "polyphosphoester", "phosphoester", "estimated"),
    ("Poly(propylene phosphate)", "*CCCOP(=O)(O)O*", 255, "polyphosphoester", "phosphoester", "estimated"),

    # ===== Polyphosphazenes =====
    ("Poly(dichlorophosphazene)", "*N=P(Cl)(Cl)*", 208, "polyphosphazene", "phosphoester", "Allcock 2003"),
    ("Poly(bis(trifluoroethoxy)phosphazene)", "*N=P(OCC(F)(F)F)(OCC(F)(F)F)*", 219, "polyphosphazene", "phosphoester", "Allcock 2003"),
    ("Poly(bis(phenoxy)phosphazene)", "*N=P(Oc1ccccc1)(Oc1ccccc1)*", 360, "polyphosphazene", "phosphoester", "Allcock 2003"),
    ("Poly(bis(4-methylphenoxy)phosphazene)", "*N=P(Oc1ccc(C)cc1)(Oc1ccc(C)cc1)*", 350, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(4-methoxyphenoxy)phosphazene)", "*N=P(Oc1ccc(OC)cc1)(Oc1ccc(OC)cc1)*", 330, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(ethoxy)phosphazene)", "*N=P(OCC)(OCC)*", 240, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(propoxy)phosphazene)", "*N=P(OCCC)(OCCC)*", 230, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(butoxy)phosphazene)", "*N=P(OCCCC)(OCCCC)*", 222, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(methoxyethoxy)phosphazene)", "*N=P(OCCOC)(OCCOC)*", 243, "polyphosphazene", "phosphoester", "Allcock 2003"),
    ("Poly(bis(4-fluorophenoxy)phosphazene)", "*N=P(Oc1ccc(F)cc1)(Oc1ccc(F)cc1)*", 355, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(4-chlorophenoxy)phosphazene)", "*N=P(Oc1ccc(Cl)cc1)(Oc1ccc(Cl)cc1)*", 365, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(4-isopropylphenoxy)phosphazene)", "*N=P(Oc1ccc(C(C)C)cc1)(Oc1ccc(C(C)C)cc1)*", 340, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(2-naphthoxy)phosphazene)", "*N=P(Oc1ccc2ccccc2c1)(Oc1ccc2ccccc2c1)*", 395, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(bis(biphenoxy)phosphazene)", "*N=P(Oc1ccc(-c2ccccc2)cc1)(Oc1ccc(-c2ccccc2)cc1)*", 410, "polyphosphazene", "phosphoester", "estimated"),
    ("Poly(amino phosphazene)", "*N=P(NC)(NC)*", 310, "polyphosphazene", "phosphoester", "estimated"),

    # ===== Nucleoside-related =====
    ("Adenine monomer", "c1nc(N)c2ncn(C3OC(CO)C(O)C3O)c2n1", 500, "nucleoside", "purine", "Simperler 2006"),
    ("Guanine monomer", "c1nc2c(nc(N)nc2[nH]1)O", 510, "nucleoside", "purine", "estimated"),
    ("Cytosine monomer", "c1cc(N)nc(=O)[nH]1", 470, "nucleoside", "pyrimidine", "estimated"),
    ("Thymine monomer", "Cc1c[nH]c(=O)[nH]c1=O", 460, "nucleoside", "pyrimidine", "estimated"),
    ("Uracil monomer", "c1c[nH]c(=O)[nH]c1=O", 465, "nucleoside", "pyrimidine", "estimated"),
    ("ATP (adenosine triphosphate)", "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1", 246, "nucleoside", "purine", "Simperler 2006"),
    ("ADP (adenosine diphosphate)", "c1nc(N)c2ncn(C3OC(COP(=O)(O)OP(=O)(O)O)C(O)C3O)c2n1", 244, "nucleoside", "purine", "Simperler 2006"),
    ("AMP (adenosine monophosphate)", "c1nc(N)c2ncn(C3OC(COP(=O)(O)O)C(O)C3O)c2n1", 249, "nucleoside", "purine", "Simperler 2006"),

    # ===== Misc H-bond polymers (PBI, PVA, etc.) =====
    ("PBI (polybenzimidazole)", "*c1ccc2[nH]c(nc2c1)c1nc2cc(*)ccc2[nH]1", 700, "misc", "benzimidazole", "Houck 2010"),
    ("Poly(vinyl alcohol)", "*CC(O)*", 358, "misc", "hydroxyl", "Mark 1999"),
    ("Poly(4-hydroxystyrene)", "*CC(*)c1ccc(O)cc1", 443, "misc", "hydroxyl", "Mark 1999"),
    ("Poly(acrylamide)", "*CC(C(=O)N)*", 438, "misc", "amide", "Mark 1999"),
    ("Poly(methacrylamide)", "*CC(C)(C(=O)N)*", 478, "misc", "amide", "estimated"),
    ("Poly(N-methylacrylamide)", "*CC(C(=O)NC)*", 415, "misc", "amide", "estimated"),
    ("Poly(N-vinylpyrrolidone)", "*CC(*)N1CCCC1=O", 448, "misc", "amide", "Mark 1999"),
    ("Poly(N-vinyl caprolactam)", "*CC(*)N1CCCCCC1=O", 421, "misc", "amide", "estimated"),
    ("Poly(2-hydroxypropyl methacrylate)", "*CC(C)(C(=O)OCC(C)O)*", 346, "misc", "hydroxyl", "estimated"),
    ("Poly(2-hydroxyethyl methacrylate) PHEMA", "*CC(C)(C(=O)OCCO)*", 355, "misc", "hydroxyl", "Mark 1999"),
)


# ---------------------------------------------------------------------------
# Systematic variation generators
# ---------------------------------------------------------------------------

def _generate_nylon_ab() -> List[Tuple[str, str, int, str, str, str]]:
    """Generate AB-type nylon series: -CO-NH-(CH₂)ₙ₋₁-.

    Tg values based on van Krevelen (2009) and Boyer (1963).
    """
    # Known Tg values (K) from literature
    known_tg = {
        2: 363, 3: 349, 4: 335, 5: 328, 6: 323,
        7: 319, 8: 316, 9: 312, 10: 312, 11: 316,
        12: 310, 13: 308,
    }

    entries = []
    for n in range(2, 14):
        chain = "C" * (n - 1)
        smiles = f"*C(=O)N{chain}*"
        tg = known_tg.get(n, max(300, 370 - 5 * n))
        name = f"Nylon {n}"
        entries.append((name, smiles, tg, "PA", "amide", "van Krevelen 2009"))
    return entries


def _generate_nylon_aabb() -> List[Tuple[str, str, int, str, str, str]]:
    """Generate AABB-type nylon series: -NH-(CH₂)ₘ-NH-CO-(CH₂)ₙ₋₂-CO-.

    Tg values based on van Krevelen (2009) and chain length correlations.
    """
    # Known Tg values (K)
    known_tg = {
        (2, 2): 385, (2, 4): 370, (2, 6): 358,
        (4, 4): 360, (4, 6): 350, (4, 8): 340, (4, 10): 330,
        (6, 4): 340, (6, 6): 330, (6, 8): 325, (6, 9): 323,
        (6, 10): 323, (6, 12): 319, (6, 14): 315,
        (8, 6): 320, (8, 8): 318, (8, 10): 315,
        (10, 6): 315, (10, 10): 310, (10, 12): 308,
        (12, 6): 312, (12, 10): 306, (12, 12): 303,
    }

    entries = []
    for (m, n), tg in sorted(known_tg.items()):
        diamine_chain = "C" * m
        acid_chain = "C" * (n - 2)
        smiles = f"*N{diamine_chain}NC(=O){acid_chain}C(=O)*"
        name = f"Nylon {m},{n}"
        entries.append((name, smiles, tg, "PA", "amide", "van Krevelen 2009"))
    return entries


def _generate_pu_variants() -> List[Tuple[str, str, int, str, str, str]]:
    """Generate polyurethane variants with different chain extenders."""
    # Diol definitions: (name, smiles_fragment, tg_offset)
    diols = [
        ("EG", "CC", 0),
        ("PG", "C(C)C", -5),
        ("BDO", "CCCC", -15),
        ("PDO", "CCCCC", -20),
        ("HD", "CCCCCC", -28),
        ("OD", "CCCCCCCC", -38),
        ("DD", "CCCCCCCCCC", -48),
        ("DEG", "CCOCC", -25),
        ("TEG", "CCOCCOCC", -35),
        ("CHDM", "CC1CCCCC1C", -5),
    ]
    # Diisocyanate definitions: (name, smiles_fragment, base_tg)
    diisocyanates = [
        ("HDI", "NCCCCCCN", 290),
        ("DDI", "NCCCCCCCCCCCCN", 260),
        ("TMDI", "NC(C)(C)CCCC(C)(C)N", 310),
    ]

    entries = []
    for di_name, di_smi, base_tg in diisocyanates:
        for diol_name, diol_smi, tg_off in diols:
            smiles = f"*OC(=O){di_smi}C(=O)O{diol_smi}*"
            tg = base_tg + tg_off
            name = f"PU {di_name}-{diol_name}"
            entries.append((name, smiles, tg, "PU", "urethane", "estimated"))
    return entries


def _generate_polyurea_variants() -> List[Tuple[str, str, int, str, str, str]]:
    """Generate polyurea variants with different chain lengths."""
    entries = []
    for n in [3, 5, 7, 9, 11]:
        chain = "C" * n
        smiles = f"*N{chain}NC(=O)N*"
        tg = max(320, 420 - 10 * n)
        name = f"Poly({n}-methylene urea)"
        entries.append((name, smiles, tg, "polyurea", "urea", "estimated"))
    return entries


def _generate_pi_variants() -> List[Tuple[str, str, int, str, str, str]]:
    """Generate polyimide variants with different linkers."""
    # Linker definitions: (name, smiles_fragment, tg)
    linkers = [
        ("ethylene", "CC", 475),
        ("propylene", "CCC", 465),
        ("butylene", "CCCC", 458),
        ("hexylene", "CCCCCC", 450),
        ("octylene", "CCCCCCCC", 440),
        ("cyclohexylene", "C1CCCCC1", 500),
        ("phenylene-p", "c1ccc(cc1)", 580),
        ("phenylene-m", "c1cccc(c1)", 560),
        ("oxy-diphenyl", "c1ccc(Oc2ccc(cc2))cc1", 550),
        ("methylene-diphenyl", "c1ccc(Cc2ccc(cc2))cc1", 540),
        ("sulfonyl-diphenyl", "c1ccc(S(=O)(=O)c2ccc(cc2))cc1", 570),
        ("carbonyl-diphenyl", "c1ccc(C(=O)c2ccc(cc2))cc1", 555),
    ]

    entries = []
    for link_name, link_smi, tg in linkers:
        smiles = f"*N(C(=O)c1ccccc1C(=O)*){link_smi}*"
        name = f"PI phthalimide-{link_name}"
        entries.append((name, smiles, tg, "PI", "imide", "estimated"))
    return entries


def _generate_phosphazene_variants() -> List[Tuple[str, str, int, str, str, str]]:
    """Generate polyphosphazene variants with mixed substituents."""
    # Alkyl ether + phenoxy combinations
    entries = []
    alkyl_groups = [
        ("methoxy", "OC", 250),
        ("ethoxy", "OCC", 240),
        ("propoxy", "OCCC", 230),
        ("butoxy", "OCCCC", 222),
    ]
    aryl_groups = [
        ("phenoxy", "Oc1ccccc1", 360),
        ("4-methylphenoxy", "Oc1ccc(C)cc1", 350),
        ("4-chlorophenoxy", "Oc1ccc(Cl)cc1", 365),
    ]

    # Mixed alkyl-aryl substituents
    for alk_name, alk_smi, alk_tg in alkyl_groups:
        for ar_name, ar_smi, ar_tg in aryl_groups:
            tg = (alk_tg + ar_tg) // 2 + 10  # mixing rule + rigidity bonus
            smiles = f"*N=P({alk_smi})({ar_smi})*"
            name = f"PPZ {alk_name}/{ar_name}"
            entries.append((name, smiles, tg, "polyphosphazene", "phosphoester", "estimated"))
    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_bridge_data() -> List[Dict]:
    """Load all bridge polymer data (core + generated).

    Returns:
        List of dicts with keys: name, smiles, tg_k, family, hbond_pattern, source.
    """
    all_entries = list(_CORE_DATA)
    all_entries.extend(_generate_nylon_ab())
    all_entries.extend(_generate_nylon_aabb())
    all_entries.extend(_generate_pu_variants())
    all_entries.extend(_generate_polyurea_variants())
    all_entries.extend(_generate_pi_variants())
    all_entries.extend(_generate_phosphazene_variants())

    return [
        {
            "name": name,
            "smiles": smiles,
            "tg_k": tg_k,
            "family": family,
            "hbond_pattern": hbond_pattern,
            "source": source,
        }
        for name, smiles, tg_k, family, hbond_pattern, source in all_entries
    ]


def get_bridge_smiles() -> List[str]:
    """Return list of all bridge polymer SMILES."""
    return [d["smiles"] for d in load_all_bridge_data()]


def build_bridge_dataset(
    layer: str = "L1",
    include_hbond: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build feature matrix and target vector from bridge polymer data.

    Args:
        layer: Feature layer ('L0', 'L1', 'L2').
        include_hbond: If True, append 15-dim H-bond features.
        verbose: Print dataset info.

    Returns:
        (X, y, names, feature_names)
    """
    from src.features.feature_pipeline import compute_features, get_feature_names
    from src.features.hbond_features import (
        compute_hbond_features,
        hbond_feature_names,
    )

    feat_names = get_feature_names(layer)
    if include_hbond:
        feat_names = feat_names + hbond_feature_names()

    data = load_all_bridge_data()

    X_list = []
    y_list = []
    names = []
    skipped = 0

    for entry in data:
        smiles = entry["smiles"]
        try:
            x_base = compute_features(smiles, None, layer)
            if np.any(np.isnan(x_base)):
                skipped += 1
                continue

            if include_hbond:
                x_hb = compute_hbond_features(smiles)
                x = np.concatenate([x_base, x_hb])
            else:
                x = x_base

            if np.any(np.isnan(x)):
                skipped += 1
                continue

            X_list.append(x)
            y_list.append(float(entry["tg_k"]))
            names.append(entry["name"])
        except Exception:
            skipped += 1

    X = np.array(X_list)
    y = np.array(y_list)

    if verbose:
        hb_tag = "+hbond" if include_hbond else ""
        print(f"  Bridge dataset [{layer}{hb_tag}]: {X.shape[0]} samples, "
              f"{X.shape[1]} features (skipped {skipped})")

    return X, y, names, feat_names


def export_csv(path: str = "data/bridge_polymers.csv") -> None:
    """Export bridge polymer data to CSV."""
    data = load_all_bridge_data()
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "smiles", "tg_k", "family", "hbond_pattern", "source"],
        )
        writer.writeheader()
        writer.writerows(data)

    print(f"  Exported {len(data)} entries to {filepath}")
