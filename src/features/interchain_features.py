"""
Interchain Interaction Features — 链间相互作用特征 (Phase B2)
8 features capturing electrostatic, dipole, and hydrophobic interactions:

Electrostatic (3d):
    MaxPartialCharge:    Gasteiger 最大正偏电荷
    MinPartialCharge:    Gasteiger 最大负偏电荷 (绝对值最大的负电荷)
    MaxAbsPartialCharge: 偏电荷绝对值最大值 → 极性强度

Dipole (3d):
    dipole_moment:       重复单元偶极矩 |μ| (单构象 MMFF + Gasteiger)
    MolMR:               Wildman-Crippen 摩尔折射率 → 极化率代理
    polar_bond_fraction: 极性键 (C-O, C-N, C-F, C-Cl) / 总键数

Hydrophobic (2d):
    hydrophobic_ratio:   疏水表面积 / 总表面积
    hydrophilic_ratio:   亲水表面积 / 总表面积

Public API:
    interchain_feature_names() -> list[str]
    interchain_vector(smiles) -> list[float]
"""

import math
import warnings
from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MolSurf

FEATURE_NAMES = [
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "dipole_moment",
    "MolMR",
    "polar_bond_fraction",
    "hydrophobic_ratio",
    "hydrophilic_ratio",
]

_NAN_RESULT = {name: float("nan") for name in FEATURE_NAMES}

# Polar bond types: (atomic_num_a, atomic_num_b)
_POLAR_PAIRS = frozenset({
    (6, 8),   # C-O
    (6, 7),   # C-N
    (6, 9),   # C-F
    (6, 17),  # C-Cl
})


def interchain_feature_names() -> List[str]:
    """Return interchain feature names in fixed order."""
    return list(FEATURE_NAMES)


def _clean_smiles(smiles: str) -> str:
    """Replace * attachment points with H for RDKit processing."""
    return smiles.replace("*", "[H]")


def _compute_partial_charges(mol: Chem.Mol) -> Dict[str, float]:
    """Compute Gasteiger partial charge features (3d)."""
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        q = atom.GetDoubleProp("_GasteigerCharge")
        if math.isfinite(q):
            charges.append(q)

    if not charges:
        return {
            "MaxPartialCharge": float("nan"),
            "MinPartialCharge": float("nan"),
            "MaxAbsPartialCharge": float("nan"),
        }

    return {
        "MaxPartialCharge": max(charges),
        "MinPartialCharge": min(charges),
        "MaxAbsPartialCharge": max(abs(q) for q in charges),
    }


def _compute_dipole_moment(mol: Chem.Mol) -> float:
    """Compute dipole moment from single MMFF-optimized conformer + Gasteiger charges.

    Returns |μ| in Debye-like units (charge * Angstrom).
    """
    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    cid = AllChem.EmbedMolecule(mol_3d, params)
    if cid < 0:
        # Fallback: basic embedding
        cid = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        if cid < 0:
            return float("nan")

    try:
        AllChem.MMFFOptimizeMolecule(mol_3d, confId=cid, maxIters=200)
    except Exception:
        pass  # use unoptimized conformer

    AllChem.ComputeGasteigerCharges(mol_3d)
    conf = mol_3d.GetConformer(cid)

    mu = np.zeros(3)
    for atom in mol_3d.GetAtoms():
        q = atom.GetDoubleProp("_GasteigerCharge")
        if not math.isfinite(q):
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        mu += q * np.array([pos.x, pos.y, pos.z])

    return float(np.linalg.norm(mu))


def _compute_polar_bond_fraction(mol: Chem.Mol) -> float:
    """Fraction of bonds that are polar (C-O, C-N, C-F, C-Cl)."""
    total = mol.GetNumBonds()
    if total == 0:
        return 0.0

    polar_count = 0
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetAtomicNum()
        a2 = bond.GetEndAtom().GetAtomicNum()
        pair = (min(a1, a2), max(a1, a2))
        if pair in _POLAR_PAIRS:
            polar_count += 1

    return polar_count / total


def _compute_hydrophobic_ratios(mol: Chem.Mol) -> Dict[str, float]:
    """Compute hydrophobic/hydrophilic surface area ratios.

    Uses SlogP_VSA descriptors: bins of Van der Waals surface area
    partitioned by Crippen logP contribution.
    Negative logP bins → hydrophilic, positive → hydrophobic.
    """
    # SlogP_VSA1-12: 12 bins of surface area by logP contribution
    # Bins 1-5 (logP < 0): hydrophilic
    # Bins 6-12 (logP >= 0): hydrophobic
    slogp_vsa = MolSurf.SlogP_VSA_(mol)  # tuple of 12 values

    hydrophilic = sum(slogp_vsa[:5])   # bins 1-5 (negative logP)
    hydrophobic = sum(slogp_vsa[5:])   # bins 6-12 (positive logP)
    total = hydrophilic + hydrophobic

    if total < 1e-6:
        return {"hydrophobic_ratio": float("nan"), "hydrophilic_ratio": float("nan")}

    return {
        "hydrophobic_ratio": hydrophobic / total,
        "hydrophilic_ratio": hydrophilic / total,
    }


def compute_interchain(smiles: str) -> Dict[str, float]:
    """Compute all 8 interchain interaction features.

    Args:
        smiles: Polymer repeat unit SMILES (may contain * attachment points).

    Returns:
        Dict with 8 features. NaN for failed computations.
    """
    clean = _clean_smiles(smiles)
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return dict(_NAN_RESULT)

    result: Dict[str, float] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Electrostatic (3d)
        result.update(_compute_partial_charges(mol))

        # Dipole moment (needs 3D conformer)
        result["dipole_moment"] = _compute_dipole_moment(mol)

        # Molar refractivity
        result["MolMR"] = Descriptors.MolMR(mol)

        # Polar bond fraction
        result["polar_bond_fraction"] = _compute_polar_bond_fraction(mol)

        # Hydrophobic ratios (2d)
        result.update(_compute_hydrophobic_ratios(mol))

    return result


def interchain_vector(smiles: str) -> List[float]:
    """Compute interchain features as fixed-order numeric vector.

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        List of 8 float values.
    """
    features = compute_interchain(smiles)
    return [features[name] for name in FEATURE_NAMES]
