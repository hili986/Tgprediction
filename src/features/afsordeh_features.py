"""
Afsordeh 2025 Physical Features for Tg Prediction
Afsordeh 2025 物理特征 -- Tg 预测

4 features achieving R2=0.97 with ExtraTrees (Afsordeh & Shirali, 2025):
    1. FlexibilityIndex = NumRotatableBonds / HeavyAtomCount
    2. SOL = Hildebrand solubility parameter (Van Krevelen GC method)
    3. HBondDensity = (NumHDonors + NumHAcceptors) / HeavyAtomCount
    4. PolarityIndex = (nO + nN + nS) / HeavyAtomCount

Public API:
    compute_afsordeh_4(smiles) -> dict of 4 features
    afsordeh_feature_names() -> list of 4 feature names
    afsordeh_vector(smiles) -> list of 4 float values
"""

from typing import Dict, List

from rdkit import Chem
from rdkit.Chem import Descriptors

from src.features.solubility_param import compute_solubility_param


# Feature names in fixed order
FEATURE_NAMES = ["FlexibilityIndex", "HBondDensity", "PolarityIndex", "SOL"]


def compute_afsordeh_4(smiles: str) -> Dict[str, float]:
    """Compute Afsordeh's 4 physical features from repeat unit SMILES.
    计算 Afsordeh 的 4 个物理特征。

    Args:
        smiles: Polymer repeat unit SMILES (may contain * for attachment points).

    Returns:
        Dict with FlexibilityIndex, SOL, HBondDensity, PolarityIndex.
        Returns NaN values if SMILES parsing fails.
    """
    clean = smiles.replace("*", "[H]")
    mol = Chem.MolFromSmiles(clean)

    if mol is None:
        return {name: float("nan") for name in FEATURE_NAMES}

    heavy = Descriptors.HeavyAtomCount(mol)
    if heavy == 0:
        heavy = 1  # prevent division by zero

    n_rot = Descriptors.NumRotatableBonds(mol)
    n_hbd = Descriptors.NumHDonors(mol)
    n_hba = Descriptors.NumHAcceptors(mol)

    # Heteroatom counts for polarity
    n_o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    n_s = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)

    return {
        "FlexibilityIndex": n_rot / heavy,
        "HBondDensity": (n_hbd + n_hba) / heavy,
        "PolarityIndex": (n_o + n_n + n_s) / heavy,
        "SOL": compute_solubility_param(mol),
    }


def afsordeh_feature_names() -> List[str]:
    """Return feature names in alphabetical order.
    返回按字母顺序排列的特征名称。
    """
    return list(FEATURE_NAMES)


def afsordeh_vector(smiles: str) -> List[float]:
    """Compute Afsordeh features as a fixed-order vector.
    计算 Afsordeh 特征并返回固定顺序的向量。

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        List of 4 float values in alphabetical feature name order.
    """
    features = compute_afsordeh_4(smiles)
    return [features[name] for name in FEATURE_NAMES]
