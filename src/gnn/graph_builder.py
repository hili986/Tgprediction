"""
Polymer SMILES -> PyG Data Graph Builder
聚合物 SMILES → PyG 分子图转换

Builds molecular graphs from repeat-unit SMILES with physics-enhanced
atom/edge features and repeat-unit masks for GRIN pooling.

Atom features (25-dim):
    Standard (15): element_one_hot[11] + formal_charge + degree + is_aromatic + is_in_ring
    Physics (10): is_flexible_bond_neighbor + gc_group[5] + is_backbone + is_sidechain
                  + local_steric + is_sp3

Edge features (6-dim):
    bond_type_one_hot[4] + is_conjugated + is_in_ring

Public API:
    smiles_to_graph(smiles, n_repeat=3, physics_features=True) -> Data
    batch_smiles_to_graphs(smiles_list, ...) -> Tuple[List[Data], List[int]]
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import torch
    from torch_geometric.data import Data

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from rdkit import Chem

from src.features.virtual_polymerization import build_oligomer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Element vocabulary (covers 99%+ of polymer atoms)
ELEMENT_VOCAB = ["C", "N", "O", "S", "F", "Cl", "Br", "Si", "P", "H", "OTHER"]
_ELEM_TO_IDX = {e: i for i, e in enumerate(ELEMENT_VOCAB)}

# Group contribution categories for CED estimation
_GC_GROUPS = {
    "aromatic": lambda a: a.GetIsAromatic(),
    "carbonyl": lambda a: (
        a.GetSymbol() == "C"
        and any(
            b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(a).GetSymbol() == "O"
            for b in a.GetBonds()
        )
    ),
    "hydroxyl_amine": lambda a: a.GetSymbol() in ("O", "N") and a.GetTotalNumHs() > 0,
    "halogen": lambda a: a.GetSymbol() in ("F", "Cl", "Br", "I"),
    "heteroatom_other": lambda a: a.GetAtomicNum() not in (1, 6) and not a.GetIsAromatic(),
}

ATOM_FEAT_DIM = 25
EDGE_FEAT_DIM = 6


# ---------------------------------------------------------------------------
# Atom feature computation
# ---------------------------------------------------------------------------

def _compute_atom_features(
    mol: Chem.Mol,
    backbone_atoms: Set[int],
    physics_features: bool = True,
) -> np.ndarray:
    """Compute atom feature matrix (n_atoms, 25).

    Args:
        mol: RDKit molecule.
        backbone_atoms: Set of backbone atom indices.
        physics_features: Whether to include physics-enhanced features.

    Returns:
        Feature matrix of shape (n_atoms, 25).
    """
    n_atoms = mol.GetNumAtoms()
    feat_dim = ATOM_FEAT_DIM if physics_features else 15
    features = np.zeros((n_atoms, feat_dim), dtype=np.float32)

    ring_info = mol.GetRingInfo()

    for idx in range(n_atoms):
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        offset = 0

        # --- Standard features (15) ---
        # Element one-hot (11)
        elem_idx = _ELEM_TO_IDX.get(symbol, _ELEM_TO_IDX["OTHER"])
        features[idx, elem_idx] = 1.0
        offset = 11

        # Formal charge (normalized)
        features[idx, offset] = atom.GetFormalCharge() / 2.0
        offset += 1

        # Degree (normalized)
        features[idx, offset] = atom.GetDegree() / 4.0
        offset += 1

        # Is aromatic
        features[idx, offset] = 1.0 if atom.GetIsAromatic() else 0.0
        offset += 1

        # Is in ring
        features[idx, offset] = 1.0 if ring_info.NumAtomRings(idx) > 0 else 0.0
        offset += 1

        if not physics_features:
            continue

        # --- Physics features (10) ---
        # Is flexible bond neighbor
        is_flex_nbr = 0.0
        for bond in atom.GetBonds():
            if (
                bond.GetBondType() == Chem.BondType.SINGLE
                and not bond.IsInRing()
                and not bond.GetIsConjugated()
            ):
                is_flex_nbr = 1.0
                break
        features[idx, offset] = is_flex_nbr
        offset += 1

        # GC group one-hot (5)
        for group_name, group_fn in _GC_GROUPS.items():
            features[idx, offset] = 1.0 if group_fn(atom) else 0.0
            offset += 1

        # Is backbone
        features[idx, offset] = 1.0 if idx in backbone_atoms else 0.0
        offset += 1

        # Is sidechain
        features[idx, offset] = 1.0 if idx not in backbone_atoms else 0.0
        offset += 1

        # Local steric (number of heavy neighbors / 4)
        heavy_nbrs = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
        features[idx, offset] = heavy_nbrs / 4.0
        offset += 1

        # Is sp3
        features[idx, offset] = (
            1.0 if atom.GetHybridization() == Chem.HybridizationType.SP3 else 0.0
        )

    return features


# ---------------------------------------------------------------------------
# Edge feature computation
# ---------------------------------------------------------------------------

def _compute_edge_features(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """Compute edge index and edge feature matrix.

    Args:
        mol: RDKit molecule.

    Returns:
        Tuple of (edge_index [2, n_edges*2], edge_attr [n_edges*2, 6]).
    """
    bond_type_map = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    edges_src, edges_dst = [], []
    edge_attrs = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond_type_map.get(bond.GetBondType(), 0)

        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
        feat[bt] = 1.0  # Bond type one-hot (4)
        feat[4] = 1.0 if bond.GetIsConjugated() else 0.0
        feat[5] = 1.0 if bond.IsInRing() else 0.0

        # Undirected: add both directions
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        edge_attrs.extend([feat, feat.copy()])

    if not edges_src:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, EDGE_FEAT_DIM), dtype=np.float32),
        )

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr = np.array(edge_attrs, dtype=np.float32)
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Repeat unit mask
# ---------------------------------------------------------------------------

def _compute_repeat_unit_mask(
    mol: Chem.Mol,
    n_repeat: int,
) -> np.ndarray:
    """Assign each atom to a repeat unit (0, 1, ..., n_repeat-1).

    For GRIN pooling, the middle unit (mask==1 for n=3) is the key.
    Uses simple heuristic: divide atoms evenly.

    Args:
        mol: RDKit molecule (oligomer).
        n_repeat: Number of repeat units.

    Returns:
        Integer array of shape (n_atoms,) with values in [0, n_repeat-1].
    """
    n_atoms = mol.GetNumAtoms()
    atoms_per_unit = n_atoms // n_repeat
    remainder = n_atoms % n_repeat

    mask = np.zeros(n_atoms, dtype=np.int64)
    start = 0
    for unit_idx in range(n_repeat):
        end = start + atoms_per_unit + (1 if unit_idx < remainder else 0)
        mask[start:end] = unit_idx
        start = end

    return mask


# ---------------------------------------------------------------------------
# Backbone detection
# ---------------------------------------------------------------------------

def _find_backbone_in_oligomer(mol: Chem.Mol) -> Set[int]:
    """Find backbone atoms in oligomer using longest carbon chain heuristic.

    For oligomers without * markers, use the longest chain as backbone.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return set(range(n_atoms))

    # Try to find longest path between terminal atoms
    # Terminal = degree 1 heavy atoms
    terminals = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetDegree() == 1 and a.GetAtomicNum() > 1
    ]

    if len(terminals) >= 2:
        # Try shortest path between first and last terminal
        path = Chem.rdmolops.GetShortestPath(mol, terminals[0], terminals[-1])
        if len(path) > n_atoms * 0.3:
            return set(path)

    # Fallback: all atoms on the longest simple path from atom 0
    path = Chem.rdmolops.GetShortestPath(mol, 0, n_atoms - 1)
    return set(path)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def smiles_to_graph(
    smiles: str,
    n_repeat: int = 3,
    physics_features: bool = True,
    y: Optional[float] = None,
) -> Optional["Data"]:
    """Convert polymer repeat-unit SMILES to a PyG Data graph.

    Pipeline:
      1. build_oligomer(smiles, n_repeat) -> oligomer SMILES
      2. RDKit parse -> Mol
      3. Compute atom features (25-dim)
      4. Compute edge features (6-dim)
      5. Compute repeat-unit mask
      6. Package as PyG Data

    Args:
        smiles: Repeat-unit SMILES with * markers.
        n_repeat: Number of repeat units for oligomer.
        physics_features: Include physics-enhanced atom features.
        y: Optional target value (Tg).

    Returns:
        PyG Data object, or None if construction fails.
    """
    if not _HAS_PYG:
        raise ImportError(
            "PyTorch Geometric is required for graph building. "
            "Install: pip install torch torch-geometric"
        )

    # Step 1: Build oligomer
    oligomer_smi = build_oligomer(smiles, n=n_repeat)
    if oligomer_smi is None:
        return None

    # Step 2: Parse
    mol = Chem.MolFromSmiles(oligomer_smi)
    if mol is None:
        return None

    if mol.GetNumAtoms() == 0:
        return None

    # Step 3: Backbone detection
    backbone = _find_backbone_in_oligomer(mol)

    # Step 4: Atom features
    atom_feats = _compute_atom_features(mol, backbone, physics_features)

    # Step 5: Edge features
    edge_index, edge_attr = _compute_edge_features(mol)

    # Step 6: Repeat unit mask
    ru_mask = _compute_repeat_unit_mask(mol, n_repeat)

    # Step 7: Package as PyG Data
    data = Data(
        x=torch.tensor(atom_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        repeat_unit_mask=torch.tensor(ru_mask, dtype=torch.long),
        smiles=smiles,
    )

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data


def batch_smiles_to_graphs(
    smiles_list: List[str],
    y_list: Optional[List[float]] = None,
    n_repeat: int = 3,
    physics_features: bool = True,
) -> Tuple[List["Data"], List[int]]:
    """Convert a list of SMILES to a list of PyG Data objects.

    Skips SMILES that fail to convert, and returns the valid indices
    so callers can align other arrays (e.g. tabular features).

    Args:
        smiles_list: List of repeat-unit SMILES.
        y_list: Optional list of target values.
        n_repeat: Number of repeat units.
        physics_features: Include physics features.

    Returns:
        Tuple of (graphs, valid_indices):
            graphs: List of successfully created Data objects.
            valid_indices: List of original indices that succeeded.
    """
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        y_val = y_list[i] if y_list is not None else None
        data = smiles_to_graph(smi, n_repeat, physics_features, y_val)
        if data is not None:
            graphs.append(data)
            valid_indices.append(i)
    if len(graphs) < len(smiles_list):
        n_failed = len(smiles_list) - len(graphs)
        print(f"Warning: {n_failed}/{len(smiles_list)} SMILES failed graph conversion")
    return graphs, valid_indices
