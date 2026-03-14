"""
Unified Feature Pipeline for Tg Prediction
统一特征管线 -- Tg 预测

Layered descriptor strategy (低成本计算描述符调研.md S9):
    L0: Afsordeh 4 physical features (4 dim)
    L1: L0 + RDKit 2D 15 descriptors (19 dim)
    L2: L1 + Morgan 1024 + fragments 15 + polymer descriptors 14 (~1068 dim)

Public API:
    compute_features(smiles, bigsmiles, layer, morgan_bits) -> np.ndarray
    build_dataset_v2(layer, morgan_bits) -> (X, y, names, feature_names)
    get_feature_names(layer, morgan_bits) -> list of feature names
"""

from typing import List, Optional, Tuple

import numpy as np

from src.features.afsordeh_features import (
    afsordeh_feature_names,
    afsordeh_vector,
)
from src.features.rdkit_descriptors import (
    l1_descriptor_names,
    l1_descriptor_vector,
)
from src.features.hbond_features import (
    compute_hbond_features,
    hbond_feature_names,
)
from src.features.physical_proxy import (
    ppf_feature_names,
    ppf_vector,
)
from src.features.virtual_polymerization import (
    vpd_feature_names,
    vpd_vector,
)
from src.bigsmiles.fingerprint import (
    morgan_fingerprint,
    fragment_vector,
    fragment_names,
    descriptor_vector,
    descriptor_names,
)
from src.data.bicerano_tg_dataset import BICERANO_DATA


# ---------------------------------------------------------------------------
# Layer configurations
# ---------------------------------------------------------------------------

LAYER_COMPONENTS = {
    "L0": ["afsordeh"],
    "L1": ["afsordeh", "rdkit_2d"],
    "L1H": ["afsordeh", "rdkit_2d", "hbond"],
    "L2": ["afsordeh", "rdkit_2d", "morgan", "fragments", "polymer_desc"],
    "L2H": ["afsordeh", "rdkit_2d", "hbond", "morgan", "fragments", "polymer_desc"],
    # M2M layers — Physical Proxy + Virtual Polymerization
    "M2M": ["afsordeh", "rdkit_2d", "hbond", "ppf", "vpd"],        # 56-dim
    "M2M-P": ["afsordeh", "rdkit_2d", "hbond", "ppf"],              # 44-dim (PPF only)
    "M2M-V": ["afsordeh", "rdkit_2d", "hbond", "vpd"],              # 46-dim (VPD only)
    "M2M-PV": ["ppf", "vpd"],                                        # 22-dim (pure physics)
}


def get_feature_names(
    layer: str = "L1",
    morgan_bits: int = 1024,
) -> List[str]:
    """Return feature names for the specified layer.
    返回指定层级的特征名称列表。

    Args:
        layer: One of 'L0', 'L1', 'L2'.
        morgan_bits: Number of Morgan fingerprint bits (for L2).

    Returns:
        List of feature name strings.
    """
    if layer not in LAYER_COMPONENTS:
        raise ValueError(f"Unknown layer: {layer}. Available: {list(LAYER_COMPONENTS.keys())}")

    components = LAYER_COMPONENTS[layer]
    names: List[str] = []

    if "afsordeh" in components:
        names.extend(f"L0_{n}" for n in afsordeh_feature_names())

    if "rdkit_2d" in components:
        names.extend(f"L1_{n}" for n in l1_descriptor_names())

    if "hbond" in components:
        names.extend(hbond_feature_names())

    if "morgan" in components:
        names.extend(f"morgan_{i}" for i in range(morgan_bits))

    if "fragments" in components:
        names.extend(f"frag_{n}" for n in fragment_names())

    if "polymer_desc" in components:
        names.extend(descriptor_names())

    if "ppf" in components:
        names.extend(f"PPF_{n}" for n in ppf_feature_names())

    if "vpd" in components:
        names.extend(f"VPD_{n}" for n in vpd_feature_names())

    return names


def compute_features(
    smiles: str,
    bigsmiles: Optional[str] = None,
    layer: str = "L1",
    morgan_bits: int = 1024,
) -> np.ndarray:
    """Compute feature vector for a single polymer at the specified layer.
    计算指定层级的单个聚合物特征向量。

    Args:
        smiles: Polymer repeat unit SMILES.
        bigsmiles: BigSMILES string (needed for L2 polymer descriptors).
        layer: One of 'L0', 'L1', 'L2'.
        morgan_bits: Number of Morgan fingerprint bits.

    Returns:
        1D numpy array of features.
    """
    if layer not in LAYER_COMPONENTS:
        raise ValueError(f"Unknown layer: {layer}. Available: {list(LAYER_COMPONENTS.keys())}")

    components = LAYER_COMPONENTS[layer]
    features: List[float] = []

    if "afsordeh" in components:
        features.extend(afsordeh_vector(smiles))

    if "rdkit_2d" in components:
        features.extend(l1_descriptor_vector(smiles))

    if "hbond" in components:
        features.extend(float(x) for x in compute_hbond_features(smiles))

    if "morgan" in components:
        fp = morgan_fingerprint(smiles, radius=2, n_bits=morgan_bits)
        features.extend(float(x) for x in fp)

    if "fragments" in components:
        features.extend(float(x) for x in fragment_vector(smiles))

    if "polymer_desc" in components:
        features.extend(descriptor_vector(smiles, bigsmiles or ""))

    if "ppf" in components:
        features.extend(ppf_vector(smiles))

    if "vpd" in components:
        features.extend(vpd_vector(smiles))

    return np.array(features, dtype=float)


def build_dataset_v2(
    layer: str = "L1",
    morgan_bits: int = 1024,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build feature matrix and target vector from Bicerano dataset.
    从 Bicerano 数据集构建特征矩阵和目标向量。

    Args:
        layer: Feature layer ('L0', 'L1', 'L2').
        morgan_bits: Number of Morgan fingerprint bits (for L2).
        verbose: Print dataset info.

    Returns:
        (X, y, names, feature_names) where:
            X: np.ndarray of shape (n_samples, n_features)
            y: np.ndarray of shape (n_samples,)
            names: list of polymer names
            feature_names: list of feature name strings
    """
    feat_names = get_feature_names(layer, morgan_bits)

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    names: List[str] = []
    skipped = 0

    for name, smiles, bigsmiles, tg_k in BICERANO_DATA:
        try:
            x = compute_features(smiles, bigsmiles, layer, morgan_bits)
            if np.any(np.isnan(x)):
                skipped += 1
                continue
            X_list.append(x)
            y_list.append(float(tg_k))
            names.append(name)
        except Exception:
            skipped += 1

    X = np.array(X_list)
    y = np.array(y_list)

    if verbose:
        print(f"  Dataset [{layer}]: {X.shape[0]} samples, {X.shape[1]} features")
        if skipped > 0:
            print(f"  Skipped: {skipped} polymers (feature extraction errors)")

    return X, y, names, feat_names
