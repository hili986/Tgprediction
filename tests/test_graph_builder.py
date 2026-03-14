"""Tests for GNN graph builder."""

import unittest

import numpy as np

from src.gnn.graph_builder import (
    ATOM_FEAT_DIM,
    EDGE_FEAT_DIM,
    _compute_atom_features,
    _compute_edge_features,
    _compute_repeat_unit_mask,
    _find_backbone_in_oligomer,
)
from src.features.virtual_polymerization import build_oligomer
from rdkit import Chem


class TestComputeAtomFeatures(unittest.TestCase):
    """Test atom feature computation."""

    def _make_mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        self.assertIsNotNone(mol, f"Failed to parse: {smiles}")
        return mol

    def test_feature_dim_with_physics(self):
        mol = self._make_mol("CCCC")
        backbone = set(range(mol.GetNumAtoms()))
        feats = _compute_atom_features(mol, backbone, physics_features=True)
        self.assertEqual(feats.shape, (mol.GetNumAtoms(), ATOM_FEAT_DIM))
        self.assertEqual(feats.shape[1], 25)

    def test_feature_dim_without_physics(self):
        mol = self._make_mol("CCCC")
        backbone = set()
        feats = _compute_atom_features(mol, backbone, physics_features=False)
        self.assertEqual(feats.shape, (mol.GetNumAtoms(), 15))

    def test_no_nan(self):
        mol = self._make_mol("c1ccccc1")
        backbone = {0, 1, 2}
        feats = _compute_atom_features(mol, backbone, physics_features=True)
        self.assertFalse(np.isnan(feats).any(), "NaN in atom features")

    def test_element_one_hot(self):
        mol = self._make_mol("CO")  # C and O
        backbone = set()
        feats = _compute_atom_features(mol, backbone, physics_features=True)
        # C is index 0, O is index 2
        self.assertEqual(feats[0, 0], 1.0)  # C
        self.assertEqual(feats[1, 2], 1.0)  # O

    def test_aromatic_flag(self):
        mol = self._make_mol("c1ccccc1")
        backbone = set()
        feats = _compute_atom_features(mol, backbone, physics_features=True)
        # offset=13 is is_aromatic
        for i in range(mol.GetNumAtoms()):
            self.assertEqual(feats[i, 13], 1.0, f"Atom {i} should be aromatic")


class TestComputeEdgeFeatures(unittest.TestCase):
    """Test edge feature computation."""

    def test_edge_dim(self):
        mol = Chem.MolFromSmiles("CCCC")
        edge_index, edge_attr = _compute_edge_features(mol)
        n_bonds = mol.GetNumBonds()
        self.assertEqual(edge_index.shape, (2, n_bonds * 2))
        self.assertEqual(edge_attr.shape, (n_bonds * 2, EDGE_FEAT_DIM))
        self.assertEqual(edge_attr.shape[1], 6)

    def test_undirected(self):
        mol = Chem.MolFromSmiles("CC")
        edge_index, edge_attr = _compute_edge_features(mol)
        # 1 bond -> 2 directed edges
        self.assertEqual(edge_index.shape[1], 2)
        # Both directions
        self.assertTrue(0 in edge_index[0] and 1 in edge_index[0])

    def test_no_bonds(self):
        mol = Chem.MolFromSmiles("[Na]")
        edge_index, edge_attr = _compute_edge_features(mol)
        self.assertEqual(edge_index.shape, (2, 0))
        self.assertEqual(edge_attr.shape, (0, EDGE_FEAT_DIM))

    def test_aromatic_bond(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        edge_index, edge_attr = _compute_edge_features(mol)
        # Aromatic bonds should have index 3 set
        has_aromatic = any(edge_attr[i, 3] == 1.0 for i in range(edge_attr.shape[0]))
        self.assertTrue(has_aromatic, "Benzene should have aromatic bonds")


class TestRepeatUnitMask(unittest.TestCase):
    """Test repeat unit mask assignment."""

    def test_mask_values(self):
        mol = Chem.MolFromSmiles("CCCCCCCCCC")  # 10 atoms
        mask = _compute_repeat_unit_mask(mol, n_repeat=3)
        self.assertEqual(len(mask), 10)
        unique_vals = set(mask)
        self.assertTrue(unique_vals.issubset({0, 1, 2}))

    def test_mask_contains_all_units(self):
        mol = Chem.MolFromSmiles("CCCCCC")  # 6 atoms
        mask = _compute_repeat_unit_mask(mol, n_repeat=3)
        self.assertIn(0, mask)
        self.assertIn(1, mask)
        self.assertIn(2, mask)

    def test_mask_covers_all_atoms(self):
        mol = Chem.MolFromSmiles("CCCCCCCCC")  # 9 atoms, 3 per unit
        mask = _compute_repeat_unit_mask(mol, n_repeat=3)
        self.assertEqual(len(mask), 9)
        # 3 atoms each
        self.assertEqual(np.sum(mask == 0), 3)
        self.assertEqual(np.sum(mask == 1), 3)
        self.assertEqual(np.sum(mask == 2), 3)


class TestFindBackbone(unittest.TestCase):
    """Test backbone detection in oligomers."""

    def test_linear_chain(self):
        mol = Chem.MolFromSmiles("CCCCCCCC")
        backbone = _find_backbone_in_oligomer(mol)
        self.assertIsInstance(backbone, set)
        self.assertTrue(len(backbone) > 0)

    def test_single_atom(self):
        mol = Chem.MolFromSmiles("[C]")
        backbone = _find_backbone_in_oligomer(mol)
        self.assertEqual(backbone, {0})


class TestSmilesToGraphIntegration(unittest.TestCase):
    """Integration test: full pipeline (requires PyG)."""

    def setUp(self):
        try:
            import torch
            from torch_geometric.data import Data
            self.has_pyg = True
        except ImportError:
            self.has_pyg = False

    def test_polyethylene(self):
        if not self.has_pyg:
            self.skipTest("PyTorch Geometric not installed")
        from src.gnn.graph_builder import smiles_to_graph
        data = smiles_to_graph("*CC*", n_repeat=3)
        self.assertIsNotNone(data)
        self.assertEqual(data.x.shape[1], 25)
        self.assertEqual(data.edge_attr.shape[1], 6)
        # mask should contain 0, 1, 2
        mask_vals = set(data.repeat_unit_mask.numpy().tolist())
        self.assertTrue(mask_vals.issubset({0, 1, 2}))

    def test_polystyrene(self):
        if not self.has_pyg:
            self.skipTest("PyTorch Geometric not installed")
        from src.gnn.graph_builder import smiles_to_graph
        data = smiles_to_graph("*CC(c1ccccc1)*", n_repeat=3)
        self.assertIsNotNone(data)
        self.assertEqual(data.x.shape[1], 25)

    def test_with_target(self):
        if not self.has_pyg:
            self.skipTest("PyTorch Geometric not installed")
        from src.gnn.graph_builder import smiles_to_graph
        data = smiles_to_graph("*CC*", y=373.0)
        self.assertIsNotNone(data)
        self.assertAlmostEqual(data.y.item(), 373.0)

    def test_invalid_smiles(self):
        if not self.has_pyg:
            self.skipTest("PyTorch Geometric not installed")
        from src.gnn.graph_builder import smiles_to_graph
        data = smiles_to_graph("INVALID_SMILES_XYZ")
        self.assertIsNone(data)

    def test_batch_conversion(self):
        if not self.has_pyg:
            self.skipTest("PyTorch Geometric not installed")
        from src.gnn.graph_builder import batch_smiles_to_graphs
        smiles_list = ["*CC*", "*CC(C)*", "*CC(c1ccccc1)*"]
        graphs = batch_smiles_to_graphs(smiles_list)
        self.assertGreaterEqual(len(graphs), 2)


if __name__ == "__main__":
    unittest.main()
