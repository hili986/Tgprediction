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

from typing import Dict, List, Set

from rdkit import Chem
from rdkit.Chem import Descriptors

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

def _count_flexible_bonds(mol: Chem.Mol) -> float:
    """Count flexible bonds with type-specific weights.

    Rules:
      - Non-ring single bonds only
      - C-C / C-O / C-N / C-S non-ring single: 1.0
      - Si-O: 1.5 (organosilicon super-flexible)
      - Amide-adjacent C-N (neighbor has C=O): 0 (conjugation restricts rotation)
      - Aromatic-aromatic single bond: 0.5
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

        if _is_amide_adjacent(a1, a2, bond):
            continue

        if a1.GetIsAromatic() and a2.GetIsAromatic():
            f_total += 0.5
            continue

        if (s1 == "Si" and s2 == "O") or (s1 == "O" and s2 == "Si"):
            f_total += 1.5
            continue

        if {s1, s2} <= {"C", "O", "N", "S"}:
            f_total += 1.0

    return max(f_total, 0.5)  # min 0.5 to prevent division by zero


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

    # PPF_3: Vf — free volume fraction estimate
    # Vf = 1 - 1.3 * V_vdW / V_m, using LabuteASA as V_vdW proxy
    v_vdw = Descriptors.LabuteASA(mol)
    v_m = max(mw, 1.0)  # rough V_m ~ MW (density ~ 1 g/cm3)
    vf = 1.0 - 1.3 * v_vdw / v_m
    vf = max(0.0, min(1.0, vf))

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
