"""
Virtual Polymerization Descriptors (VPD) for Tg Prediction
虚拟聚合描述符 — 在寡聚体上计算描述符捕获聚合效应

GRIN theory guarantees 3-RU oligomer captures all non-redundant local
chemical environments. VPD bridges monomer→polymer gap.

12 features in 3 groups:
    Core (6):  per-RU normalized descriptors on 3-mer
    Delta (4): polymerization effect = (dimer/2) - monomer
    Junction (2): attachment point local features

Public API:
    build_oligomer(repeat_smiles, n=3) -> Optional[str]
    compute_vpd(smiles) -> dict
    vpd_feature_names() -> list[str]
    vpd_vector(smiles) -> list[float]
"""

from typing import Dict, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors


FEATURE_NAMES = [
    # Core (6) — trimer per-RU
    "MolWt_per_RU",
    "TPSA_per_RU",
    "MolLogP_per_RU",
    "HeavyAtom_per_RU",
    "RotBonds_per_RU",
    "RingCount_per_RU",
    # Delta (4) — polymerization effect
    "MolWt_delta",
    "TPSA_delta",
    "LogP_delta",
    "RotBonds_delta",
    # Junction (2) — attachment point features
    "junction_hbond_count",
    "junction_flex_ratio",
]

# Core descriptors to compute on oligomers
_CORE_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "TPSA": Descriptors.TPSA,
    "MolLogP": Descriptors.MolLogP,
    "HeavyAtom": Descriptors.HeavyAtomCount,
    "RotBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
}


# ---------------------------------------------------------------------------
# Oligomer assembly
# ---------------------------------------------------------------------------

def build_oligomer(repeat_smiles: str, n: int = 3,
                   _suppress_log: bool = True) -> Optional[str]:
    """Assemble n copies of repeat unit into an oligomer SMILES.

    Algorithm:
    1. Parse SMILES, find exactly 2 dummy atoms (*)
    2. Record attachment point neighbors and bond types
    3. CombineMols n copies, connect adjacent copies
    4. Remove dummy atoms, sanitize, return canonical SMILES

    Args:
        repeat_smiles: SMILES with * marking attachment points.
        n: Number of repeat units (default 3).
        _suppress_log: Internal flag. Set False when caller already manages
            RDKit log suppression (avoids nested Enable undoing outer Disable).

    Returns:
        Canonical SMILES of the oligomer, or None if assembly fails.
    """
    # NOTE: DisableLog/EnableLog is process-global, not thread-safe.
    # If parallel VPD computation is needed, use a lock + refcount pattern.
    if _suppress_log:
        RDLogger.DisableLog('rdApp.*')
    try:
        return _build_oligomer_impl(repeat_smiles, n)
    finally:
        if _suppress_log:
            RDLogger.EnableLog('rdApp.*')


def _build_oligomer_impl(repeat_smiles: str, n: int) -> Optional[str]:
    """Internal implementation of build_oligomer (no log suppression)."""
    mol = Chem.MolFromSmiles(repeat_smiles)
    if mol is None:
        return _fallback_oligomer(repeat_smiles, n)

    dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return _fallback_oligomer(repeat_smiles, n)

    left_info = _get_attachment_info(mol, dummies[0])
    right_info = _get_attachment_info(mol, dummies[1])

    if left_info is None or right_info is None:
        return _fallback_oligomer(repeat_smiles, n)

    try:
        combined = Chem.RWMol(mol)
        offsets = [0]
        for i in range(1, n):
            combined = Chem.RWMol(Chem.CombineMols(combined, mol))
            offsets.append(offsets[-1] + mol.GetNumAtoms())

        # Connect adjacent copies: right-* neighbor of [i] -> left-* neighbor of [i+1]
        for i in range(n - 1):
            right_nbr = offsets[i] + right_info["neighbor_idx"]
            left_nbr = offsets[i + 1] + left_info["neighbor_idx"]
            combined.AddBond(right_nbr, left_nbr, right_info["bond_type"])

        # Remove all dummy atoms (descending index order)
        all_dummies = sorted(
            [a.GetIdx() for a in combined.GetAtoms() if a.GetAtomicNum() == 0],
            reverse=True,
        )
        for d_idx in all_dummies:
            combined.RemoveAtom(d_idx)

        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)

    except Exception:
        return _fallback_oligomer(repeat_smiles, n)


def _get_attachment_info(mol, dummy_idx):
    """Get neighbor info for a dummy atom."""
    atom = mol.GetAtomWithIdx(dummy_idx)
    neighbors = atom.GetNeighbors()
    if len(neighbors) != 1:
        return None
    nbr = neighbors[0]
    bond = mol.GetBondBetweenAtoms(dummy_idx, nbr.GetIdx())
    return {
        "neighbor_idx": nbr.GetIdx(),
        "bond_type": bond.GetBondType(),
        "neighbor_symbol": nbr.GetSymbol(),
    }


def _fallback_oligomer(smiles: str, n: int) -> Optional[str]:
    """Fallback: hydrogen-capped monomer repeat for failed assembly.

    Strategy (ordered by quality):
      1. Replace * with [H] (proper capping) and repeat n times
      2. If that fails, try single capped monomer (n=1)
      3. Return None as last resort
    """
    # Strategy 1: H-cap + repeat
    capped = smiles.replace("[*]", "[H]").replace("*", "[H]")
    if not capped or capped == smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(capped)
        if mol is not None:
            # For n>1, combine n copies via CombineMols
            # CombineMols returns disconnected fragments; MolToSmiles handles directly
            combined = mol
            for _ in range(n - 1):
                combined = Chem.CombineMols(combined, mol)
            return Chem.MolToSmiles(combined)
    except Exception:
        pass

    # Strategy 2: single capped monomer (still better than None)
    try:
        mol = Chem.MolFromSmiles(capped)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Descriptor computation helpers
# ---------------------------------------------------------------------------

def _compute_core_descs(smiles_or_mol) -> Dict[str, float]:
    """Compute 6 core descriptors on a molecule."""
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        mol = smiles_or_mol

    if mol is None:
        return {k: float("nan") for k in _CORE_FUNCS}

    result = {}
    for name, func in _CORE_FUNCS.items():
        try:
            result[name] = float(func(mol))
        except Exception:
            result[name] = float("nan")
    return result


def _compute_junction_features(smiles: str) -> Dict[str, float]:
    """Extract H-bond and flexibility features at attachment points.

    junction_hbond_count: number of H-bond donors/acceptors near * points
    junction_flex_ratio: fraction of flexible (non-ring single) bonds near * points
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"junction_hbond_count": 0.0, "junction_flex_ratio": 0.0}

    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) < 2:
        return {"junction_hbond_count": 0.0, "junction_flex_ratio": 0.0}

    hbond_count = 0
    flex_bonds = 0
    total_bonds = 0

    for dummy in dummies:
        for nbr in dummy.GetNeighbors():
            # Check H-bond capability of neighbor
            sym = nbr.GetSymbol()
            if sym in ("N", "O"):
                hbond_count += 1
            # Check if neighbor has H-bond donors/acceptors in vicinity
            for nbr2 in nbr.GetNeighbors():
                if nbr2.GetAtomicNum() == 0:
                    continue
                if nbr2.GetSymbol() in ("N", "O"):
                    hbond_count += 1

            # Count bonds around attachment atom for flexibility
            for bond in nbr.GetBonds():
                other = bond.GetOtherAtom(nbr)
                if other.GetAtomicNum() == 0:
                    continue
                total_bonds += 1
                if (bond.GetBondType() == Chem.BondType.SINGLE
                        and not bond.IsInRing()):
                    flex_bonds += 1

    flex_ratio = flex_bonds / max(total_bonds, 1)

    return {
        "junction_hbond_count": float(hbond_count),
        "junction_flex_ratio": float(flex_ratio),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_vpd(smiles: str) -> Dict[str, float]:
    """Compute all 12 Virtual Polymerization Descriptors.

    Args:
        smiles: Polymer repeat unit SMILES with * attachment points.

    Returns:
        Dict with 12 feature values.
    """
    RDLogger.DisableLog('rdApp.*')
    try:
        return _compute_vpd_impl(smiles)
    finally:
        RDLogger.EnableLog('rdApp.*')


def _compute_vpd_impl(smiles: str) -> Dict[str, float]:
    """Internal VPD computation (called with RDKit logging suppressed)."""
    # Monomer descriptors (capped with [H])
    mono_smiles = smiles.replace("[*]", "[H]").replace("*", "[H]")
    mono_descs = _compute_core_descs(mono_smiles)

    # Dimer for delta computation (suppress_log=False: caller already manages)
    dimer_smiles = build_oligomer(smiles, n=2, _suppress_log=False)
    dimer_descs = (
        _compute_core_descs(dimer_smiles) if dimer_smiles else mono_descs
    )

    # Trimer for per-RU computation (suppress_log=False: caller already manages)
    trimer_smiles = build_oligomer(smiles, n=3, _suppress_log=False)
    trimer_descs = (
        _compute_core_descs(trimer_smiles) if trimer_smiles else mono_descs
    )

    # Junction features
    junc = _compute_junction_features(smiles)

    result = {}

    # Core (6): trimer normalized per repeat unit
    for key in ["MolWt", "TPSA", "MolLogP", "HeavyAtom", "RotBonds", "RingCount"]:
        result[f"{key}_per_RU"] = trimer_descs[key] / 3.0

    # Delta (4): (dimer/2) - monomer = polymerization effect
    result["MolWt_delta"] = (dimer_descs["MolWt"] / 2.0) - mono_descs["MolWt"]
    result["TPSA_delta"] = (dimer_descs["TPSA"] / 2.0) - mono_descs["TPSA"]
    result["LogP_delta"] = (dimer_descs["MolLogP"] / 2.0) - mono_descs["MolLogP"]
    result["RotBonds_delta"] = (
        (dimer_descs["RotBonds"] / 2.0) - mono_descs["RotBonds"]
    )

    # Junction (2)
    result["junction_hbond_count"] = junc["junction_hbond_count"]
    result["junction_flex_ratio"] = junc["junction_flex_ratio"]

    return result


def vpd_feature_names() -> List[str]:
    """Return VPD feature names in fixed order."""
    return list(FEATURE_NAMES)


def vpd_vector(smiles: str) -> List[float]:
    """Compute VPD as a fixed-order numeric vector.

    Args:
        smiles: Polymer repeat unit SMILES.

    Returns:
        List of 12 float values.
    """
    features = compute_vpd(smiles)
    return [features[name] for name in FEATURE_NAMES]
