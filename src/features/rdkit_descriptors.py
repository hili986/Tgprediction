"""
RDKit 2D Descriptors for Tg Prediction (L1 Layer)
RDKit 2D 描述符 -- Tg 预测（L1 层）

15 SHAP-validated descriptors ranked by consensus across 6+ studies:
    P0 (Top 3): NumRotatableBonds, TPSA, MolLogP
    P1 (Top 10): NumHDonors, NumHAcceptors, FractionCSP3, MolWt,
                  RingCount, NumAromaticRings, HeavyAtomCount
    P2 (Topology): BalabanJ, Chi0v, Chi1v, Kappa1, Kappa2

Source: 低成本计算描述符调研.md S4 + SHAP consensus analysis

Public API:
    compute_l1_descriptors(smiles) -> dict of 15 features
    l1_descriptor_names() -> list of 15 feature names
    l1_descriptor_vector(smiles) -> list of 15 float values
"""

from typing import Dict, List

from rdkit import Chem
from rdkit.Chem import Descriptors


# L1 descriptors: name -> RDKit function
# Sorted alphabetically for consistent vector ordering
_L1_DESCRIPTOR_FUNCS = {
    "BalabanJ": Descriptors.BalabanJ,
    "Chi0v": Descriptors.Chi0v,
    "Chi1v": Descriptors.Chi1v,
    "FractionCSP3": Descriptors.FractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "MolLogP": Descriptors.MolLogP,
    "MolWt": Descriptors.MolWt,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "TPSA": Descriptors.TPSA,
}

FEATURE_NAMES = sorted(_L1_DESCRIPTOR_FUNCS.keys())


def compute_l1_descriptors(smiles: str) -> Dict[str, float]:
    """Compute 15 RDKit 2D descriptors from repeat unit SMILES.
    计算 15 个 RDKit 2D 描述符。

    Args:
        smiles: Polymer repeat unit SMILES (may contain * for attachment points).

    Returns:
        Dict with 15 descriptor values.
        Returns NaN for all if SMILES parsing fails.
    """
    clean = smiles.replace("*", "[H]")
    mol = Chem.MolFromSmiles(clean)

    if mol is None:
        return {name: float("nan") for name in FEATURE_NAMES}

    result = {}
    for name in FEATURE_NAMES:
        func = _L1_DESCRIPTOR_FUNCS[name]
        try:
            result[name] = float(func(mol))
        except Exception:
            result[name] = float("nan")
    return result


def l1_descriptor_names() -> List[str]:
    """Return descriptor names in alphabetical order.
    返回按字母顺序排列的描述符名称。
    """
    return list(FEATURE_NAMES)


def l1_descriptor_vector(smiles: str) -> List[float]:
    """Compute L1 descriptors as a fixed-order vector.
    计算 L1 描述符并返回固定顺序的向量。

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        List of 15 float values in alphabetical name order.
    """
    features = compute_l1_descriptors(smiles)
    return [features[name] for name in FEATURE_NAMES]
