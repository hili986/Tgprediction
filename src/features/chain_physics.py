"""
Chain Segment Physics Features — N_eff + Cn_proxy from 3-mer conformational sampling
链段物理特征 — 3-mer 构象采样得到 Boltzmann N_eff + 链刚性代理

8 features from a single 3-mer + 50 conformers computation:
  Neff_300K:      Boltzmann effective conformer count at 300K (Gibbs-DiMarzio)
  Neff_500K:      Boltzmann effective conformer count at 500K
  Neff_ratio:     Neff_500K / Neff_300K (energy landscape steepness)
  conf_strain:    mean energy - min energy (conformational strain, kcal/mol)
  Cn_proxy:       characteristic ratio approximation from 3-mer R²/(n·l²)
  curl_ratio:     end-to-end distance / contour length (chain curl)
  curl_variance:  std(R_ee) / mean(R_ee) (conformational plasticity)
  oligomer_level: meta-feature (0=failed, 3=3-mer success)

Physical predictions:
  Neff_300K vs Tg:  negative correlation (flexible → many conformers → low Tg)
  Cn_proxy vs Tg:   positive correlation (stiff → high Cn → high Tg)

Source: 方案待选-物理驱动多尺度算法重构.md §2.2, §2.5

Public API:
    compute_3mer_physics(smiles, n_confs=50) -> dict (8 features)
    chain_physics_feature_names() -> list[str]
"""

from typing import Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops

from src.features.virtual_polymerization import build_oligomer


FEATURE_NAMES = [
    "Neff_300K",
    "Neff_500K",
    "Neff_ratio",
    "conf_strain",
    "Cn_proxy",
    "curl_ratio",
    "curl_variance",
    "oligomer_level",
]

_NAN_RESULT = {name: float("nan") for name in FEATURE_NAMES}
_NAN_RESULT["oligomer_level"] = 0.0


def chain_physics_feature_names() -> List[str]:
    """Return chain physics feature names in fixed order."""
    return list(FEATURE_NAMES)


def _boltzmann_neff(energies: np.ndarray, T: float) -> float:
    """Compute Boltzmann-weighted effective conformer count at temperature T.

    N_eff = exp(S_conf) where S_conf = -Σ pᵢ ln(pᵢ)  (Shannon entropy)
    pᵢ = exp(-ΔEᵢ/RT) / Z  (Boltzmann probability)
    """
    RT_kcal = 1.987e-3 * T  # gas constant in kcal/(mol·K)
    dE = energies - energies.min()
    boltz = np.exp(-dE / RT_kcal)
    Z = boltz.sum()
    p = boltz / Z
    S = -np.sum(p * np.log(p + 1e-30))
    return float(np.exp(S))


def compute_3mer_physics(
    smiles: str,
    n_confs: int = 50,
) -> Dict[str, float]:
    """Compute chain-segment physics from 3-mer conformational sampling.

    Single function that computes BOTH N_eff and Cn_proxy from one
    round of oligomer building + conformer embedding + MMFF optimization.

    Args:
        smiles: Polymer repeat unit SMILES (with * attachment points).
        n_confs: Number of conformers to generate (default 50).

    Returns:
        Dict with 8 features. NaN for all (except oligomer_level=0)
        if computation fails at any stage.
    """
    # --- Stage 1: Build 3-mer oligomer ---
    oligomer_smi = build_oligomer(smiles, n=3)
    if oligomer_smi is None:
        return dict(_NAN_RESULT)

    mol = Chem.MolFromSmiles(oligomer_smi)
    if mol is None:
        return dict(_NAN_RESULT)

    # --- Stage 2: Find chain ends BEFORE AddHs ---
    # Use graph diameter (topologically most distant heavy atom pair)
    # Must be done before AddHs because AddHs changes degree of terminal atoms
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy_idx) < 4:
        return dict(_NAN_RESULT)

    dist_matrix = rdmolops.GetDistanceMatrix(mol)
    max_dist = 0
    first_atom, last_atom = heavy_idx[0], heavy_idx[-1]
    for i in heavy_idx:
        for j in heavy_idx:
            if i < j and dist_matrix[i][j] > max_dist:
                max_dist = dist_matrix[i][j]
                first_atom, last_atom = i, j

    n_backbone_bonds = int(max_dist)
    if n_backbone_bonds < 3:
        return dict(_NAN_RESULT)

    # --- Stage 3: Generate 3D conformers ---
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if len(cids) < 5:
        return dict(_NAN_RESULT)

    # --- Stage 4: MMFF optimize and collect energies ---
    energies = []
    valid_cids = []
    for cid in cids:
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                if ff is not None:
                    energies.append(ff.CalcEnergy())
                    valid_cids.append(cid)
        except Exception:
            continue

    if len(energies) < 5:
        return dict(_NAN_RESULT)

    energies = np.array(energies)

    # --- Stage 5: Compute N_eff at two temperatures ---
    neff_300 = _boltzmann_neff(energies, 300.0)
    neff_500 = _boltzmann_neff(energies, 500.0)
    neff_ratio = neff_500 / max(neff_300, 1.0)
    conf_strain = float(np.mean(energies) - energies.min())

    # --- Stage 6: Compute Cn_proxy and curl metrics ---
    # Read actual bond lengths from first conformer along backbone path
    conf0 = mol.GetConformer(valid_cids[0])
    path = rdmolops.GetShortestPath(mol, first_atom, last_atom)

    bond_lengths_sq = []
    for k in range(len(path) - 1):
        p_a = conf0.GetAtomPosition(path[k])
        p_b = conf0.GetAtomPosition(path[k + 1])
        bl2 = (p_a.x - p_b.x)**2 + (p_a.y - p_b.y)**2 + (p_a.z - p_b.z)**2
        bond_lengths_sq.append(bl2)

    if not bond_lengths_sq:
        return dict(_NAN_RESULT)

    mean_l_sq = np.mean(bond_lengths_sq)
    n_path_bonds = len(bond_lengths_sq)
    contour_length = sum(np.sqrt(bl) for bl in bond_lengths_sq)

    # End-to-end distance² for all conformers
    r_ee_sq = []
    for cid in valid_cids:
        conf = mol.GetConformer(cid)
        p1 = conf.GetAtomPosition(first_atom)
        p2 = conf.GetAtomPosition(last_atom)
        r2 = (p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2
        r_ee_sq.append(r2)

    r_ee_sq = np.array(r_ee_sq)
    r_ee_mean = np.sqrt(np.mean(r_ee_sq))

    # Cn = <R²> / (n × <l²>)
    cn_proxy = float(np.mean(r_ee_sq) / (n_path_bonds * mean_l_sq))

    # curl_ratio: actual distance / max stretch (0=coiled, 1=extended)
    curl_ratio = float(r_ee_mean / max(contour_length, 1e-6))

    # curl_variance: conformational plasticity
    r_ee_vals = np.sqrt(r_ee_sq)
    curl_var = float(np.std(r_ee_vals) / max(r_ee_mean, 1e-6))

    return {
        "Neff_300K": neff_300,
        "Neff_500K": neff_500,
        "Neff_ratio": neff_ratio,
        "conf_strain": conf_strain,
        "Cn_proxy": cn_proxy,
        "curl_ratio": curl_ratio,
        "curl_variance": curl_var,
        "oligomer_level": 3.0,
    }
