"""
Chain Segment Physics Features — N_eff + Cn_proxy from 3-mer conformational sampling
链段物理特征 — 3-mer 构象采样得到 Boltzmann N_eff + 链刚性代理

Energy backend:
  Default: MMFF94 (CPU, RDKit built-in)
  Opt-in: AIMNet2 (GPU, pip install aimnet2calc) — set _BACKEND = None to auto-detect

8 features from a single 3-mer + 50 conformers computation:
  Neff_300K:      Boltzmann effective conformer count at 300K (Gibbs-DiMarzio)
  Neff_500K:      Boltzmann effective conformer count at 500K
  Neff_ratio:     Neff_500K / Neff_300K (energy landscape steepness)
  conf_strain:    mean energy - min energy (conformational strain, kcal/mol)
  Cn_proxy:       characteristic ratio approximation from 3-mer R²/(n·l²)
  curl_ratio:     end-to-end distance / contour length (chain curl)
  curl_variance:  std(R_ee) / mean(R_ee) (conformational plasticity)
  oligomer_level: meta-feature (0=failed, 3=3-mer success)

Public API:
    compute_3mer_physics(smiles, n_confs=50) -> dict (8 features)
    chain_physics_feature_names() -> list[str]
"""

import warnings
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Energy backend: MMFF94 (CPU) default, AIMNet2 (GPU) opt-in
# ---------------------------------------------------------------------------

_AIMNET2_MODEL = None  # lazy-loaded singleton
_BACKEND = "mmff"      # default CPU; set to None to auto-detect AIMNet2


def _get_backend() -> str:
    """Detect available energy backend. MMFF94 default."""
    global _BACKEND, _AIMNET2_MODEL
    if _BACKEND is not None:
        return _BACKEND

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU")

        import aimnet2calc
        _AIMNET2_MODEL = aimnet2calc.AIMNet2Calculator("aimnet2")
        _BACKEND = "aimnet2"
        print(f"[chain_physics] Using AIMNet2 backend (GPU: {torch.cuda.get_device_name(0)})")
    except Exception as e:
        _BACKEND = "mmff"
        warnings.warn(
            f"[chain_physics] AIMNet2 unavailable ({e}), falling back to MMFF94 (CPU). "
            f"Install: pip install aimnet2calc"
        )
    return _BACKEND


def _compute_energies_aimnet2(
    mol: Chem.Mol, cids: list
) -> Tuple[np.ndarray, list]:
    """Compute conformer energies using AIMNet2 on GPU.

    AIMNet2 supports batch computation — all conformers in one GPU call.
    Returns energies in kcal/mol and list of valid conformer IDs.
    """
    import torch
    from aimnet2calc import AIMNet2Calculator

    global _AIMNET2_MODEL
    if _AIMNET2_MODEL is None:
        _AIMNET2_MODEL = AIMNet2Calculator("aimnet2")

    calc = _AIMNET2_MODEL
    energies = []
    valid_cids = []

    # AIMNet2 works with ASE atoms objects or coordinates + atomic numbers
    for cid in cids:
        try:
            conf = mol.GetConformer(cid)
            coords = np.array(conf.GetPositions())  # (N, 3) in Angstrom
            atomic_nums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

            # AIMNet2 input: dict with 'coord' and 'numbers'
            inp = {
                "coord": torch.tensor(coords, dtype=torch.float32).unsqueeze(0),
                "numbers": torch.tensor(atomic_nums, dtype=torch.long).unsqueeze(0),
            }
            result = calc(inp)
            # Energy in Hartree → convert to kcal/mol
            e_hartree = result["energy"].item()
            energies.append(e_hartree * 627.509)  # 1 Hartree = 627.509 kcal/mol
            valid_cids.append(cid)
        except Exception:
            continue

    return np.array(energies) if energies else np.array([]), valid_cids


def _compute_energies_mmff(
    mol: Chem.Mol, cids: list
) -> Tuple[np.ndarray, list]:
    """Compute conformer energies using MMFF94 (CPU fallback).

    Optimizes each conformer then computes energy.
    Returns energies in kcal/mol and list of valid conformer IDs.
    """
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

    return np.array(energies) if energies else np.array([]), valid_cids


def _compute_energies(
    mol: Chem.Mol, cids: list
) -> Tuple[np.ndarray, list]:
    """Compute conformer energies using best available backend."""
    backend = _get_backend()
    if backend == "aimnet2":
        energies, valid = _compute_energies_aimnet2(mol, cids)
        if len(energies) >= 5:
            return energies, valid
        # AIMNet2 failed for this molecule → fallback to MMFF
        return _compute_energies_mmff(mol, cids)
    return _compute_energies_mmff(mol, cids)


# ---------------------------------------------------------------------------
# Core physics computation
# ---------------------------------------------------------------------------

def chain_physics_feature_names() -> List[str]:
    """Return chain physics feature names in fixed order."""
    return list(FEATURE_NAMES)


def _boltzmann_neff(energies: np.ndarray, T: float) -> float:
    """Compute Boltzmann-weighted effective conformer count at temperature T.

    N_eff = exp(S_conf) where S_conf = -Σ pᵢ ln(pᵢ)  (Shannon entropy)
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
    round of oligomer building + conformer embedding + energy computation.

    Energy backend: MMFF94 (CPU) by default.

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

    # --- Stage 3: Generate 3D conformers (CPU, ETKDGv3) ---
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if len(cids) < 5:
        return dict(_NAN_RESULT)

    # --- Stage 4: Compute energies (MMFF94 CPU default) ---
    energies, valid_cids = _compute_energies(mol, list(cids))

    if len(energies) < 5:
        return dict(_NAN_RESULT)

    # --- Stage 5: Compute N_eff at two temperatures ---
    neff_300 = _boltzmann_neff(energies, 300.0)
    neff_500 = _boltzmann_neff(energies, 500.0)
    neff_ratio = neff_500 / max(neff_300, 1.0)
    conf_strain = float(np.mean(energies) - energies.min())

    # --- Stage 6: Compute Cn_proxy and curl metrics ---
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

    r_ee_sq = []
    for cid in valid_cids:
        conf = mol.GetConformer(cid)
        p1 = conf.GetAtomPosition(first_atom)
        p2 = conf.GetAtomPosition(last_atom)
        r2 = (p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2
        r_ee_sq.append(r2)

    r_ee_sq = np.array(r_ee_sq)
    r_ee_mean = np.sqrt(np.mean(r_ee_sq))

    cn_proxy = float(np.mean(r_ee_sq) / (n_path_bonds * mean_l_sq))
    curl_ratio = float(r_ee_mean / max(contour_length, 1e-6))

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
