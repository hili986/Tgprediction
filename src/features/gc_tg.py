"""
Group Contribution Method for Tg Prediction (Cao 2020 + Van Krevelen)
基团贡献法 — Tg 预测值

Formula: Tg(inf) = 1000 * sum(Ni * Ygi) / Mw   (K)
  where Ygi in K·kg/mol (= 10^3 g·K/mol), Mw in g/mol

Source: Cao et al., ACS Omega 2020, 5(44), 29538-29546, Table S3
        R² = 0.9925 on 198 polymers, 58 structural groups
        Key improvement: composite groups (-CH(X)- as one unit)
        capture backbone-substituent interaction effects

Priority matching: larger groups matched first to avoid double-counting.
Coverage < 30% → return NaN (unreliable prediction).

Public API:
    compute_gc_tg(smiles) -> dict  {'GC_Tg': float, 'GC_coverage': float}
    gc_tg_feature_names() -> list[str]
"""

from typing import Dict, List, Set, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors


# ---------------------------------------------------------------------------
# Cao 2020 Yg table (K·kg/mol) — Table S3 from Supporting Information
# Tg(inf) = 1000 * ΣYg / Mw  (K)
#
# Groups marked * are from Van Krevelen (adopted by Cao).
# Others are new groups introduced by Cao.
#
# COMPOSITE groups: -CH(X)- counts the backbone C + its substituent X as
# one unit. This captures the steric/electronic interaction that simple
# atomic fragments miss (e.g., -CH(CH3)- as a unit vs -CH- + -CH3-).
# ---------------------------------------------------------------------------

TG_GC_GROUPS: Dict[str, Tuple[str, float]] = {
    # SMARTS:                        (name,                   Ygi)
    #
    # === VERY LARGE: heterocyclic composite groups (matched first) ===
    # 47*: pyromellitic diimide (PMDA-based polyimides)
    "[CX3](=O)[NX3]c1cc2[CX3](=O)[NX3][CX3](=O)c2cc1[CX3](=O)":
                                     ("PMDA_diimide",         187.952),
    # 46*: naphthalene
    "c1ccc2ccccc2c1":                ("naphthalene",          111.805),
    # 49*: phthalimide (5-ring, on aromatic)
    "[CX3](=O)[NX3][CX3](=O)c":     ("phthalimide",          103.180),
    # 48: phthalimide (5-ring, aliphatic N)
    "[CX3](=O)[NX3][CX3](=O)":      ("imide_aliph",           44.226),
    # 42*: p-phenylene (generic aromatic ring)
    "c1ccccc1":                      ("phenylene",             42.182),

    # === LARGE: composite backbone-substituent groups ===
    # 22*: -C(CH3)(phenyl)-  (poly-alpha-methylstyrene)
    "[CX4;!R]([CH3])(c1ccccc1)":     ("C_CH3_phenyl",          54.475),
    # 8*: -CH(phenyl)-  (polystyrene backbone)
    "[CX4H1;!R](c1ccccc1)":         ("CH_phenyl",             38.934),
    # 20*: -C(CH3)(COOCH3)-  (PMMA!)
    "[CX4;!R]([CH3])([CX3](=O)[OX2])": ("C_CH3_COOMe",        37.503),
    # 6: -CH(CONH2)-  (polyacrylamide)
    "[CX4H1;!R]([CX3](=O)[NX3H2])": ("CH_CONH2",             29.918),
    # 28: -CH(CF3)-
    "[CX4H1;!R]([CX4](F)(F)F)":     ("CH_CF3",               25.254),
    # 4*: -CH(COOH)-  (polyacrylic acid)
    "[CX4H1;!R]([CX3](=O)[OX2H])":  ("CH_COOH",              18.820),
    # 10*: -CHCl-  (PVC backbone)
    "[CX4H1;!R]([Cl])":             ("CHCl",                  17.911),
    # 17*: -CH(CN)-  (polyacrylonitrile)
    "[CX4H1;!R]([CX2]#[NX1])":      ("CH_CN",                15.637),
    # 3*: -CH(OCH3)-
    "[CX4H1;!R]([OX2][CH3])":       ("CH_OCH3",              10.611),
    # 11: -CH(CH=CH2)-  (polybutadiene 1,2)
    "[CX4H1;!R]([CH]=[CH2])":       ("CH_vinyl",             10.770),
    # 2*: -CH(CH3)-  (PP backbone — KEY FIX for PP!)
    "[CX4H1;!R]([CH3])":            ("CH_CH3",                8.222),
    # 5*: -CH(OH)-  (polyvinyl alcohol)
    "[CX4H1;!R]([OX2H])":           ("CH_OH",                 4.697),
    # 19*: -C(CH3)2-  (polyisobutylene, BPA)
    "[CX4;!R]([CH3])([CH3])":        ("C_diCH3",              18.234),
    # 29: -C(CF3)2-  (hexafluoroisopropylidene)
    "[CX4;!R]([CX4](F)(F)F)([CX4](F)(F)F)": ("C_diCF3",     80.103),
    # 25*: -CCl2-
    "[CX4;!R](Cl)(Cl)":             ("CCl2",                  23.910),
    # 24*: -CF2-
    "[CX4;!R](F)(F)":               ("CF2",                   17.503),
    # 26*: -CFCl-
    "[CX4;!R](F)(Cl)":              ("CFCl",                  26.884),

    # === MEDIUM: functional group units ===
    # 40*: -OCOO-  (carbonate)
    "[OX2][CX3](=O)[OX2]":          ("carbonate",             13.663),
    # 41*: -OCONH-  (urethane)
    "[OX2][CX3](=O)[NX3]":          ("urethane",              16.108),
    # 39*: -CONH-  (amide)
    "[CX3](=O)[NX3]":               ("amide",                 19.247),
    # 30: -N(COCH3)-
    "[NX3]([CX3](=O)[CH3])":        ("N_acetyl",              23.331),
    # 37: -O-CS-O-  (thiocarbonate)
    "[OX2][CX3](=S)[OX2]":          ("thiocarbonate",         14.676),
    # 38*: -COO-  (ester)
    "[CX3](=O)[OX2]":               ("ester",                  7.025),
    # 36*: -SO2-
    "[SX4](=O)(=O)":                ("sulfone",               15.373),
    # 35*: -CO-  (carbonyl/ketone)
    "[CX3](=O)":                    ("carbonyl",               4.370),
    # 14*: -C(Cl)=CH-  (polychloroprene)
    "[CX3;!R](Cl)=[CH]":            ("C_Cl_vinyl",            13.807),
    # 13*: -C(CH3)=CH-  (polyisoprene)
    "[CX3;!R]([CH3])=[CH]":         ("C_CH3_vinyl",            6.228),
    # 12*: -CH=CH-  (polybutadiene 1,4)
    "[CH]=[CH]":                     ("trans_vinylene",         1.344),

    # === SMALL: single-atom / backbone groups ===
    # 1*: -CH2-  (ubiquitous)
    "[CH2;!R]":                      ("CH2",                    4.026),
    # Also match ring CH2 (e.g., cyclohexane units)
    "[CH2;R]":                       ("CH2_ring",              4.026),
    # 31*: -O-  (ether, normal)
    "[OX2;!$([OX2][CX3]=O)]":       ("ether_O",             -14.718),
    # 34*: -S-  (thioether)
    "[SX2]":                         ("thioether",             -2.887),
    # 55*: -Si(CH3)2-
    "[Si]([CH3])([CH3])":            ("Si_diCH3",             -1.059),
    # Fallback single atoms
    "[CH3]":                         ("CH3_terminal",           4.026),
    "[CH1;!R]":                      ("CH_bare",               4.026),
    "[CX4H0;!R]":                    ("C_quat_bare",           4.026),
    "[F]":                           ("F_single",              0.0),
    "[Cl]":                          ("Cl_single",             0.0),
    "[Br]":                          ("Br_single",             0.0),
    "[NX3H1]":                       ("NH",                     4.0),
    "[NX3H2]":                       ("NH2",                    4.0),
    "[OX2H]":                        ("OH",                     4.0),
}


FEATURE_NAMES = ["GC_Tg", "GC_coverage"]


# ---------------------------------------------------------------------------
# Pre-compile and sort by pattern size (descending) for priority matching
# ---------------------------------------------------------------------------

_COMPILED_TG_GROUPS: List[Tuple[Chem.Mol, str, float, int]] = []


def _init_patterns():
    """Compile SMARTS and sort by atom count (largest first)."""
    global _COMPILED_TG_GROUPS
    entries = []
    for smarts, (name, yg) in TG_GC_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            n_atoms = pattern.GetNumAtoms()
            entries.append((pattern, name, yg, n_atoms))
        else:
            import warnings
            warnings.warn(f"Invalid SMARTS for {name}: {smarts}")
    # Sort by atom count descending — larger groups matched first
    entries.sort(key=lambda x: x[3], reverse=True)
    _COMPILED_TG_GROUPS = entries


_init_patterns()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gc_tg(smiles: str) -> Dict[str, float]:
    """Compute GC-predicted Tg from repeat unit SMILES.

    Uses Cao 2020 composite group contributions with priority matching.
    Larger groups matched first; already-matched atoms excluded.

    Args:
        smiles: Polymer repeat unit SMILES (may contain * for attachment).

    Returns:
        Dict with:
          GC_Tg:       Predicted Tg in K. NaN if coverage < 30%.
          GC_coverage:  Fraction of heavy atoms covered by GC groups [0, 1].
    """
    # Use [At] (astatine) as dummy cap instead of [H] to preserve
    # backbone H-counts. H-capping adds extra H to terminal carbons,
    # breaking composite group matching (e.g., -CHCl- becomes -CH2Cl).
    clean = smiles.replace("*", "[At]")
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        # Fallback: try H-capping for SMILES that don't support [At]
        mol = Chem.MolFromSmiles(smiles.replace("*", "[H]"))
    if mol is None:
        return {"GC_Tg": float("nan"), "GC_coverage": 0.0}

    # Count only non-dummy heavy atoms
    n_heavy = sum(1 for a in mol.GetAtoms()
                  if a.GetAtomicNum() > 1 and a.GetAtomicNum() != 85)
    if n_heavy == 0:
        return {"GC_Tg": float("nan"), "GC_coverage": 0.0}

    # Mw of the repeat unit (exclude dummy At atoms: At = 210 g/mol)
    mw = Descriptors.MolWt(mol)
    n_at = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 85)
    mw -= n_at * 210.0   # subtract At mass
    mw += n_at * 1.008    # add back H mass (each * was a bond to neighbor)
    if mw < 10:
        return {"GC_Tg": float("nan"), "GC_coverage": 0.0}

    total_yg = 0.0
    matched_atoms: Set[int] = set()

    # Pre-mark At dummy atoms as "matched" so they don't affect coverage
    at_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 85}

    for pattern, name, yg, n_pat_atoms in _COMPILED_TG_GROUPS:
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            # Skip if ANY atom in this match was already claimed
            if any(a in matched_atoms for a in match):
                continue
            # Skip matches that include dummy atoms
            if any(a in at_atoms for a in match):
                continue
            total_yg += yg
            matched_atoms.update(match)

    coverage = len(matched_atoms) / n_heavy

    if coverage < 0.3:
        return {"GC_Tg": float("nan"), "GC_coverage": coverage}

    tg_gc = 1000.0 * total_yg / mw
    return {"GC_Tg": tg_gc, "GC_coverage": coverage}


def gc_tg_feature_names() -> List[str]:
    """Return GC_Tg feature names in fixed order."""
    return list(FEATURE_NAMES)


def gc_tg_vector(smiles: str) -> List[float]:
    """Compute GC_Tg features as a fixed-order numeric vector."""
    features = compute_gc_tg(smiles)
    return [features[name] for name in FEATURE_NAMES]
