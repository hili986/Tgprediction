"""
Physical Proxy Features (PPF) for Tg Prediction
物理代理特征 — 从单体 SMILES 推断材料级物理行为

10 features bridging monomer structure to material behavior:
    PPF_1  M_per_f              — mass per flexible bond (Gibbs-DiMarzio)
    PPF_2  CED_estimate         — cohesive energy density (Van Krevelen GC)
    PPF_3  Vf_estimate          — free volume fraction estimate
    PPF_4  backbone_rigidity    — fraction of rigid backbone bonds
    PPF_5  steric_volume        — side group Van der Waals volume
    PPF_6  flexible_bond_density— flexible bonds / heavy atom count
    PPF_7  symmetry_index       — structural symmetry score [0, 1]
    PPF_8  side_chain_ratio     — side chain atoms / backbone atoms
    PPF_9  CED_hbond_frac       — H-bond contribution to CED
    PPF_10 ring_strain_proxy    — 3-4 membered ring count / total ring count

Public API:
    compute_ppf(smiles) -> dict
    ppf_feature_names() -> list[str]
    ppf_vector(smiles) -> list[float]
"""

import math
from typing import Dict, List, Set

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from src.features.solubility_param import (
    compute_solubility_param,
    GC_GROUPS,
    _COMPILED_PATTERNS,
)


FEATURE_NAMES = [
    "M_per_f",
    "CED_estimate",
    "Vf_estimate",
    "backbone_rigidity",
    "steric_volume",
    "flexible_bond_density",
    "symmetry_index",
    "side_chain_ratio",
    "CED_hbond_frac",
    "ring_strain_proxy",
]

# SMARTS keys known to form H-bonds
_HBOND_GC_KEYS = {"[OX2H]", "[NX3H2]", "[NX3H1]", "[NX3]([CX3]=O)"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_terminal_substituent(atom) -> bool:
    """Check if atom is a true terminal substituent whose rotation is trivial.

    Only single-atom terminals: halogens (-F, -Cl, -Br, -I) and -OH.
    Does NOT include -CH3 or other carbon terminals, because in polymer
    SMILES the * attachment points become -CH3 after H-capping, and
    those bonds are real backbone bonds, not trivial terminals.
    """
    if atom.GetDegree() != 1:
        return False
    sym = atom.GetSymbol()
    if sym in ("F", "Cl", "Br", "I"):
        return True
    if sym == "O" and atom.GetTotalNumHs() >= 1:
        # Si-O backbone bond from siloxane H-capping (*[Si]...O* → O has deg=1)
        # This is NOT a terminal -OH, it's a real backbone bond
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == "Si":
                return False  # siloxane backbone, not -OH
        return True
    return False


def _count_flexible_bonds(mol: Chem.Mol) -> float:
    """Count flexible bonds with type-specific weights (Schneider-DiMarzio rules).

    Rules (Schneider & DiMarzio 2006, PMC2203329):
      - Non-ring single bonds only
      - Terminal atoms (degree=1): 0 — rotation of -OH, -CH3, -F etc.
        does not change molecular shape (Schneider-DiMarzio principle)
      - Amide-adjacent C-N: 0 (resonance restricts rotation)
      - Si-O: 1.5 (organosilicon super-flexible, low barrier ~1 kJ/mol)
      - Aromatic-aromatic single bond: 0.5 (restricted rotation)
      - C-C / C-O / C-N / C-S non-ring single: 1.0
      - Ring bonds / double / triple: 0
    """
    f_total = 0.0
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue

        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        s1, s2 = a1.GetSymbol(), a2.GetSymbol()

        # Rule: amide C-N → 0 (conjugation)
        if _is_amide_adjacent(a1, a2, bond):
            continue

        # Rule: true terminal single-atom substituents → 0
        # -F, -Cl, -Br, -I (single atom, degree=1): rotation is identity
        # -OH oxygen (degree=1, has H): H too small to change shape
        # NOTE: degree=1 carbons are NOT skipped — after * → [H] replacement,
        # backbone carbons become CH3 with degree=1, but they represent
        # real polymer backbone bonds, not true terminals
        if _is_terminal_substituent(a1) or _is_terminal_substituent(a2):
            continue

        if a1.GetIsAromatic() and a2.GetIsAromatic():
            f_total += 0.5
            continue

        if (s1 == "Si" and s2 == "O") or (s1 == "O" and s2 == "Si"):
            f_total += 1.5
            continue

        if {s1, s2} <= {"C", "O", "N", "S"}:
            f_total += 1.0

    return max(f_total, 0.1)  # lowered from 0.5 to preserve rigidity differences


def _is_amide_adjacent(a1, a2, bond) -> bool:
    """Check if C-N bond is adjacent to C=O (amide-like, rotation restricted)."""
    for atom in [a1, a2]:
        if atom.GetSymbol() == "N":
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() == bond.GetOtherAtomIdx(atom.GetIdx()):
                    continue
                if nbr.GetSymbol() == "C":
                    for b in nbr.GetBonds():
                        other = b.GetOtherAtom(nbr)
                        if (other.GetSymbol() == "O"
                                and b.GetBondType() == Chem.BondType.DOUBLE):
                            return True
    return False


_USE_3D_VOLUME = True   # Set False to use fast Labute approximation
_MAX_HEAVY_FOR_3D = 80  # Skip 3D for molecules larger than this


def _compute_vdw_volume(mol: Chem.Mol) -> float:
    """Compute van der Waals volume.

    Primary: RDKit 3D embedding + ComputeMolVolume (accurate but slow).
    Fallback: Labute ASA → spherical volume approximation (fast).

    Returns:
        V_vdW in Angstrom^3, or NaN if computation fails.
    """
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return float("nan")

    # Fast mode or molecule too large for 3D
    if not _USE_3D_VOLUME or n_heavy > _MAX_HEAVY_FOR_3D:
        return _vdw_volume_fast(mol)

    # 3D embedding + volume
    mol_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol_3d, params) < 0:
        return _vdw_volume_fast(mol)  # fallback
    AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=100)
    try:
        return AllChem.ComputeMolVolume(mol_3d)
    except Exception:
        return _vdw_volume_fast(mol)


def _vdw_volume_fast(mol: Chem.Mol) -> float:
    """Fast vdW volume estimate from Labute ASA (no 3D needed).

    Uses spherical approximation: V ≈ (4/3)π × (ASA/4π)^(3/2)
    This gives V ∝ ASA^1.5, normalized to match 3D volumes empirically.
    """
    asa = Descriptors.LabuteASA(mol)
    if asa <= 0:
        return float("nan")
    # Empirical scale: calibrated against ComputeMolVolume on PE/PP/PS/PMMA/PVC/PET
    # ASA in Å², volume in Å³. Factor 0.47 from regression against 3D volumes.
    return 0.47 * (asa ** 1.5)


def _estimate_molar_volume(mol: Chem.Mol) -> float:
    """Estimate molar volume V_m using GC method.

    GC_GROUPS Vw values are molar volumes (cm³/mol) from VK Table 4.5,
    NOT van der Waals volumes. Sum directly gives V_m.
    Fallback: V_m = Mw / 1.1  (density ≈ 1.1 g/cm³)

    Returns:
        V_m in cm³/mol, or NaN if estimation fails.
    """
    total_vm = 0.0
    for smarts_str, (name, ecoh, vm) in GC_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts_str)
        if pattern is not None:
            count = len(mol.GetSubstructMatches(pattern))
            total_vm += vm * count
    if total_vm > 0:
        return total_vm
    # Fallback: density ≈ 1.1 g/cm³
    mw = Descriptors.MolWt(mol)
    return mw / 1.1 if mw > 0 else float("nan")


def compute_free_volume(mol: Chem.Mol) -> Dict[str, float]:
    """Compute corrected free volume fraction using 3D vdW volume.

    Van Krevelen formula: f = 1 - 1.3 × V_vdW / V_m
      V_vdW = van der Waals volume (cm³/mol) from 3D conformer
      V_m   = molar volume (cm³/mol) from GC method

    Returns NaN if 3D embedding fails (NaN = signal, not error).

    Returns:
        Dict with FV_fraction and FV_packing.
    """
    v_vdw_a3 = _compute_vdw_volume(mol)
    if math.isnan(v_vdw_a3):
        return {"FV_fraction": float("nan"), "FV_packing": float("nan")}

    v_m = _estimate_molar_volume(mol)
    if math.isnan(v_m) or v_m <= 0:
        return {"FV_fraction": float("nan"), "FV_packing": float("nan")}

    # Unit conversion: Å³ → cm³/mol (×Avogadro/10²⁴ = ×0.6022)
    v_vdw_cm3 = v_vdw_a3 * 0.6022

    packing = v_vdw_cm3 / v_m
    f = 1.0 - 1.3 * packing
    f = max(0.0, min(0.6, f))

    return {"FV_fraction": f, "FV_packing": packing}


def _find_backbone_atoms(mol: Chem.Mol, smiles: str) -> Set[int]:
    """Find backbone atoms as shortest path between * attachment points.

    Strategy:
      1. Parse original SMILES to find * (dummy atom) neighbors
      2. Shortest path between the two attachment points = backbone
      3. Fallback: all heavy atoms
    """
    if "*" in smiles:
        raw_mol = Chem.MolFromSmiles(smiles)
        if raw_mol is not None:
            dummy_nbrs = []
            for atom in raw_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    for nbr in atom.GetNeighbors():
                        dummy_nbrs.append(nbr.GetIdx())

            if len(dummy_nbrs) >= 2:
                # In the clean mol (with [H] replacing *), dummy atoms are
                # removed, so indices shift. Use shortest path between first
                # and last heavy atoms as approximation.
                n_atoms = mol.GetNumAtoms()
                if n_atoms >= 2:
                    path = Chem.rdmolops.GetShortestPath(
                        mol, 0, n_atoms - 1
                    )
                    return set(path)

    # Fallback: all heavy atoms are considered backbone
    return set(range(mol.GetNumHeavyAtoms()))


def _compute_symmetry_index(mol: Chem.Mol, smiles: str) -> float:
    """Compute structural symmetry score [0, 1].

    For each substituted backbone carbon, compare substituents on both sides.
    More identical substituents -> higher symmetry.

    Examples:
      PE (*CC*):           1.0 (no substituents = fully symmetric)
      PP (*CC(C)*):        0.0 (one methyl = asymmetric)
      PIB (*CC(C)(C)*):    0.5 (two identical substituents = symmetric at that C)
    """
    backbone = _find_backbone_atoms(mol, smiles)
    n_substituted = 0
    n_symmetric = 0

    for idx in backbone:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != "C":
            continue

        subs = []
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in backbone and nbr.GetAtomicNum() > 1:
                subs.append(nbr.GetAtomicNum())

        if not subs:
            continue

        n_substituted += 1
        if len(subs) >= 2 and subs[0] == subs[1]:
            n_symmetric += 1

    if n_substituted == 0:
        return 1.0  # no substituents = fully symmetric (e.g., PE)

    return n_symmetric / n_substituted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ppf(smiles: str) -> Dict[str, float]:
    """Compute all 10 Physical Proxy Features from repeat unit SMILES.

    Args:
        smiles: Polymer repeat unit SMILES (may contain * for attachment).

    Returns:
        Dict with 10 feature values. Returns NaN for all if parsing fails.
    """
    clean = smiles.replace("*", "[H]")
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        return {name: float("nan") for name in FEATURE_NAMES}

    mw = Descriptors.MolWt(mol)
    heavy = Descriptors.HeavyAtomCount(mol) or 1

    # PPF_1: M/f — mass per flexible bond (Gibbs-DiMarzio)
    f = _count_flexible_bonds(mol)
    m_per_f = mw / f

    # PPF_2: CED — cohesive energy density = delta^2
    sol = compute_solubility_param(mol)
    ced = sol ** 2

    # PPF_3: Vf — free volume fraction (corrected: 3D vdW volume + GC V_m)
    fv = compute_free_volume(mol)
    vf = fv["FV_fraction"]  # NaN if 3D embedding fails (NaN = signal)

    # PPF_4: backbone_rigidity — rigid bonds / total backbone bonds
    backbone = _find_backbone_atoms(mol, smiles)
    rigid_bonds = 0
    total_backbone_bonds = 0
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in backbone and a2 in backbone:
            total_backbone_bonds += 1
            if (bond.GetIsAromatic()
                    or bond.GetBondType() == Chem.BondType.DOUBLE):
                rigid_bonds += 1
    br = rigid_bonds / max(total_backbone_bonds, 1)

    # PPF_5: steric_volume — side chain heavy atoms * ~15 A^3 each
    n_backbone = len(backbone) or 1
    n_sidechain = max(heavy - n_backbone, 0)
    steric = n_sidechain * 15.0

    # PPF_6: flexible_bond_density — flexible bonds / heavy atoms
    fbd = f / heavy

    # PPF_7: symmetry_index
    sym = _compute_symmetry_index(mol, smiles)

    # PPF_8: side_chain_ratio — side chain atoms / backbone atoms
    sc_ratio = n_sidechain / n_backbone

    # PPF_9: CED_hbond_frac — H-bond group CED / total CED
    hbond_ecoh = 0.0
    total_ecoh = 0.0
    for pattern, name, ecoh, vw in _COMPILED_PATTERNS:
        matches = mol.GetSubstructMatches(pattern)
        count = len(matches)
        if count == 0:
            continue
        total_ecoh += ecoh * count
        # Check if this GC group is H-bond capable
        for smarts_key, (gname, gecoh, gvw) in GC_GROUPS.items():
            if gecoh == ecoh and smarts_key in _HBOND_GC_KEYS:
                hbond_ecoh += ecoh * count
                break
    ced_hbond_frac = hbond_ecoh / max(total_ecoh, 1.0)

    # PPF_10: ring_strain_proxy — 3-4 membered rings / total rings
    ring_info = mol.GetRingInfo()
    total_rings = ring_info.NumRings()
    strained_rings = 0
    if total_rings > 0:
        for ring in ring_info.AtomRings():
            if len(ring) <= 4:
                strained_rings += 1
        ring_strain = strained_rings / total_rings
    else:
        ring_strain = 0.0

    return {
        "M_per_f": m_per_f,
        "CED_estimate": ced,
        "Vf_estimate": vf,
        "backbone_rigidity": br,
        "steric_volume": steric,
        "flexible_bond_density": fbd,
        "symmetry_index": sym,
        "side_chain_ratio": sc_ratio,
        "CED_hbond_frac": ced_hbond_frac,
        "ring_strain_proxy": ring_strain,
    }


def ppf_feature_names() -> List[str]:
    """Return PPF feature names in fixed order."""
    return list(FEATURE_NAMES)


def ppf_vector(smiles: str) -> List[float]:
    """Compute PPF as a fixed-order numeric vector.

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        List of 10 float values.
    """
    features = compute_ppf(smiles)
    return [features[name] for name in FEATURE_NAMES]
