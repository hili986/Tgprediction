"""
Van Krevelen Group Contribution Method for Solubility Parameter
Van Krevelen 基团贡献法 -- 溶解度参数 delta 估算

Formula: delta = sqrt(sum(Ecoh_i) / sum(Vw_i))  (unit: (J/cm3)^0.5)
Source: Van Krevelen, Properties of Polymers, 4th Ed.
        低成本计算描述符调研.md S7

Public API:
    compute_solubility_param(mol) -> float
    compute_solubility_param_from_smiles(smiles) -> float
"""

import math
from typing import Dict, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors


# ---------------------------------------------------------------------------
# Van Krevelen group contribution parameters (13 common groups)
# {SMARTS: (name, Ecoh_J_per_mol, Vw_cm3_per_mol)}
# ---------------------------------------------------------------------------

GC_GROUPS: Dict[str, Tuple[str, float, float]] = {
    "[CH3]":              ("-CH3",         4710,   33.5),
    "[CH2]":              ("-CH2-",        4940,   16.1),
    "[CH1]":              (">CH-",         3430,   -1.0),
    "[CX4H0]":            (">C<",          1470,  -19.2),
    "c1ccccc1":            ("phenyl C6H4", 31940,  52.4),
    "[CX3](=O)[OX2]":     ("ester -COO-", 18000,  18.0),
    "[CX3](=O)":          ("carbonyl >C=O", 17370, 10.8),
    "[OX2H0]":            ("ether -O-",    3350,    3.8),
    "[OX2H]":             ("hydroxyl -OH", 21850,  10.0),
    "[NX3H2]":            ("primary amine -NH2", 12560, 19.2),
    "[NX3H1]":            ("secondary amine >NH", 8370,  4.5),
    "[NX3]([CX3]=O)":     ("amide -CONH-", 33490,  9.5),
    "[F]":                ("-F",           2280,   18.0),
}

# Pre-compiled SMARTS patterns (created once at import time)
_COMPILED_PATTERNS = []
for smarts, (name, ecoh, vw) in GC_GROUPS.items():
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is not None:
        _COMPILED_PATTERNS.append((pattern, name, ecoh, vw))


def compute_solubility_param(mol) -> float:
    """Compute Hildebrand solubility parameter using Van Krevelen GC method.
    使用 Van Krevelen 基团贡献法计算 Hildebrand 溶解度参数。

    Args:
        mol: RDKit Mol object.

    Returns:
        Solubility parameter delta in (J/cm3)^0.5.
        Falls back to MolLogP-based approximation if no groups matched.
    """
    if mol is None:
        return float("nan")

    total_ecoh = 0.0
    total_vw = 0.0
    matched_any = False

    for pattern, name, ecoh, vw in _COMPILED_PATTERNS:
        matches = mol.GetSubstructMatches(pattern)
        count = len(matches)
        if count > 0:
            matched_any = True
            total_ecoh += ecoh * count
            total_vw += vw * count

    if not matched_any or total_vw <= 0:
        # Fallback: rough approximation from MolLogP
        logp = Descriptors.MolLogP(mol)
        return 8.0 + logp * 0.5

    return math.sqrt(total_ecoh / total_vw)


def compute_solubility_param_from_smiles(smiles: str) -> float:
    """Compute solubility parameter from SMILES string.
    从 SMILES 字符串计算溶解度参数。

    Handles polymer repeat unit SMILES (replaces * with [H]).

    Args:
        smiles: SMILES string (may contain * for attachment points).

    Returns:
        Solubility parameter delta.
    """
    clean = smiles.replace("*", "[H]")
    mol = Chem.MolFromSmiles(clean)
    return compute_solubility_param(mol)
