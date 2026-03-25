"""
H-bond SMARTS Features for Tg Prediction
氢键 SMARTS 特征

Two versions:
    Full (15-dim): L1 counts + L2 density + L3 CED (legacy, for compatibility)
    Slim (5-dim):  ced_weighted_sum, total_hbond_density,
                   hbond_network_potential, polar_fraction, interaction_types
                   (SHAP-guided: removes 10 low-value SMARTS counts)

Source: 氢键与Tg关系调研-核酸数据库构建策略.md

Public API:
    compute_hbond_features(smiles) -> np.ndarray  (15-dim, legacy)
    compute_hbond_slim(smiles) -> dict             (5-dim, recommended)
    hbond_feature_names() -> list[str]             (15-dim names)
    hbond_slim_feature_names() -> list[str]        (5-dim names)
    count_hbond_groups(smiles) -> dict
    hbond_density(smiles) -> dict
    ced_weighted_sum(smiles) -> float
"""

from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


# ---------------------------------------------------------------------------
# 10 SMARTS patterns + CED contribution values (J/cm³)
# ---------------------------------------------------------------------------

HBOND_SMARTS: Dict[str, Tuple[str, float]] = {
    "amide":        ("[NX3H][CX3](=O)",            1862.0),  # 酰胺键
    "urea":         ("[NX3H][CX3](=O)[NX3H]",      2079.0),  # 脲键（最强氢键）
    "urethane":     ("[OX2][CX3](=O)[NX3H]",       1385.0),  # 氨基甲酸酯键
    "imide":        ("[CX3](=O)[NX3][CX3](=O)",     980.0),  # 酰亚胺键
    "phosphoester": ("[OX2]P(=O)([OX2])[OX1,OX2]", 1000.0),  # 磷酸酯键（核酸骨架）
    "aromatic_nh":  ("[nH]",                         600.0),  # 芳香 N-H
    "benzimidazole": ("[nH]1cnc2ccccc12",           1200.0),  # 苯并咪唑
    "pyrimidine":   ("c1ccnc(n1)",                   500.0),  # 嘧啶环（碱基）
    "purine":       ("c1nc2[nH]cnc2c(n1)",           800.0),  # 嘌呤环（碱基）
    "hydroxyl":     ("[OX2H]",                      1500.0),  # 羟基
}

# Pre-compile SMARTS patterns for performance
_COMPILED_SMARTS: Dict[str, Chem.rdchem.Mol] = {}


def _get_compiled_smarts() -> Dict[str, Chem.rdchem.Mol]:
    """Lazy-compile SMARTS patterns (cached)."""
    if not _COMPILED_SMARTS:
        for name, (smarts, _) in HBOND_SMARTS.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                _COMPILED_SMARTS[name] = pattern
    return _COMPILED_SMARTS


# ---------------------------------------------------------------------------
# L1: SMARTS match counts (10 dim)
# ---------------------------------------------------------------------------

def count_hbond_groups(smiles: str) -> Dict[str, int]:
    """Count H-bond SMARTS pattern matches.

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        Dict mapping pattern name to match count.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: 0 for k in HBOND_SMARTS}

    compiled = _get_compiled_smarts()
    counts = {}
    for name in HBOND_SMARTS:
        pattern = compiled.get(name)
        if pattern is None:
            counts[name] = 0
        else:
            counts[name] = len(mol.GetSubstructMatches(pattern))
    return counts


# ---------------------------------------------------------------------------
# L2: Normalized density indicators (4 dim)
# ---------------------------------------------------------------------------

def hbond_density(smiles: str) -> Dict[str, float]:
    """Compute normalized H-bond density features.

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        Dict with 4 density features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "total_hbond_density": 0.0,
            "strong_hbond_density": 0.0,
            "nucleic_relevance": 0.0,
            "aromatic_hbond_density": 0.0,
        }

    heavy = Descriptors.HeavyAtomCount(mol)
    if heavy == 0:
        heavy = 1

    counts = count_hbond_groups(smiles)

    total = sum(counts.values())
    strong = counts["amide"] + counts["urea"] + counts["hydroxyl"]
    nucleic = counts["phosphoester"] + counts["pyrimidine"] + counts["purine"]
    aromatic = counts["aromatic_nh"] + counts["benzimidazole"]

    return {
        "total_hbond_density": total / heavy,
        "strong_hbond_density": strong / heavy,
        "nucleic_relevance": nucleic / heavy,
        "aromatic_hbond_density": aromatic / heavy,
    }


# ---------------------------------------------------------------------------
# L3: CED-weighted sum (1 dim) — intermolecular force estimate
# ---------------------------------------------------------------------------

def ced_weighted_sum(smiles: str) -> float:
    """CED-weighted sum of H-bond groups, normalized by heavy atom count.

    Higher values indicate stronger intermolecular forces → higher Tg.

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        Normalized CED-weighted sum (J/cm³ per heavy atom).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    heavy = Descriptors.HeavyAtomCount(mol)
    if heavy == 0:
        heavy = 1

    compiled = _get_compiled_smarts()
    total_ced = 0.0
    for name, (_, ced) in HBOND_SMARTS.items():
        pattern = compiled.get(name)
        if pattern is not None:
            count = len(mol.GetSubstructMatches(pattern))
            total_ced += ced * count

    return total_ced / heavy


# ---------------------------------------------------------------------------
# Combined: 15-dim feature vector
# ---------------------------------------------------------------------------

_HBOND_COUNT_NAMES = [f"hbond_count_{name}" for name in HBOND_SMARTS]
_HBOND_DENSITY_NAMES = [
    "total_hbond_density",
    "strong_hbond_density",
    "nucleic_relevance",
    "aromatic_hbond_density",
]
_HBOND_CED_NAMES = ["ced_weighted_sum"]


def hbond_feature_names() -> List[str]:
    """Return ordered list of 15 H-bond feature names."""
    return _HBOND_COUNT_NAMES + _HBOND_DENSITY_NAMES + _HBOND_CED_NAMES


def compute_hbond_features(smiles: str) -> np.ndarray:
    """Compute full 15-dim H-bond feature vector.

    Layout:
        [0:10]  L1 SMARTS match counts
        [10:14] L2 density indicators
        [14:15] L3 CED-weighted sum

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        1D numpy array of shape (15,).
    """
    counts = count_hbond_groups(smiles)
    density = hbond_density(smiles)
    ced = ced_weighted_sum(smiles)

    features = []
    # L1: 10 counts (in HBOND_SMARTS order)
    for name in HBOND_SMARTS:
        features.append(float(counts[name]))
    # L2: 4 densities
    for key in _HBOND_DENSITY_NAMES:
        features.append(float(density[key]))
    # L3: 1 CED
    features.append(float(ced))

    return np.array(features, dtype=float)


# ---------------------------------------------------------------------------
# Slim version: 5-dim (SHAP-guided selection, recommended for new models)
# ---------------------------------------------------------------------------

_HBOND_SLIM_NAMES = [
    "ced_weighted_sum",        # SHAP #7 (6.89): CED-weighted H-bond strength
    "total_hbond_density",     # (donors+acceptors)/heavy: overall H-bond capacity
    "hbond_network_potential",  # donors*acceptors/heavy²: network formation ability
    "polar_fraction",           # TPSA/(TPSA+nonpolar): polar surface ratio
    "interaction_types",        # count of distinct H-bond group types present
]


def hbond_slim_feature_names() -> List[str]:
    """Return ordered list of 5 slim H-bond feature names."""
    return list(_HBOND_SLIM_NAMES)


def compute_hbond_slim(smiles: str) -> Dict[str, float]:
    """Compute slim 5-dim H-bond features (replaces 15-dim for new models).

    Physical rationale for each feature:
      ced_weighted_sum:        intermolecular force strength (→ higher Tg)
      total_hbond_density:     H-bond sites per atom (→ more interchain links)
      hbond_network_potential: donors AND acceptors needed for a network;
                               one without the other cannot form H-bond networks
      polar_fraction:          polar surface → stronger intermolecular attraction
      interaction_types:       diversity of H-bond motifs (amide+OH > pure amide)

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        Dict with 5 float values.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: 0.0 for name in _HBOND_SLIM_NAMES}

    heavy = Descriptors.HeavyAtomCount(mol) or 1

    # Feature 1: ced_weighted_sum (reuse existing function)
    ced = ced_weighted_sum(smiles)

    # Feature 2: total_hbond_density (reuse from hbond_density)
    counts = count_hbond_groups(smiles)
    total_hb = sum(counts.values()) / heavy

    # Feature 3: hbond_network_potential
    n_donors = Descriptors.NumHDonors(mol)
    n_acceptors = Descriptors.NumHAcceptors(mol)
    network = (n_donors * n_acceptors) / (heavy * heavy)

    # Feature 4: polar_fraction = TPSA / (TPSA + non-polar SA)
    tpsa = Descriptors.TPSA(mol)
    labute_asa = Descriptors.LabuteASA(mol)
    total_sa = max(labute_asa, 1.0)
    polar_frac = tpsa / (tpsa + total_sa)

    # Feature 5: interaction_types = number of distinct H-bond group types
    n_types = sum(1 for c in counts.values() if c > 0)

    return {
        "ced_weighted_sum": ced,
        "total_hbond_density": total_hb,
        "hbond_network_potential": network,
        "polar_fraction": polar_frac,
        "interaction_types": float(n_types),
    }
