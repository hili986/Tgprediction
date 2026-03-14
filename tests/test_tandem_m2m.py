"""Tests for Tandem M2M and PhysicsGAT (requires PyG)."""

import unittest

try:
    import torch
    from torch_geometric.data import Data, Batch

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


def _make_dummy_graph(n_atoms=10, n_bonds=12):
    """Create a minimal PyG Data for testing."""
    x = torch.randn(n_atoms, 25)
    edge_index = torch.randint(0, n_atoms, (2, n_bonds))
    edge_attr = torch.randn(n_bonds, 6)
    repeat_unit_mask = torch.tensor([0]*3 + [1]*4 + [2]*3)[:n_atoms]
    y = torch.tensor([350.0])
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        repeat_unit_mask=repeat_unit_mask, y=y,
    )


@unittest.skipUnless(_HAS_PYG, "PyTorch Geometric not installed")
class TestPhysicsGAT(unittest.TestCase):
    """Test PhysicsGAT forward pass."""

    def setUp(self):
        from src.gnn.physics_gat import PhysicsGAT
        self.model = PhysicsGAT(in_dim=25, hidden_dim=128, out_dim=64, heads=4)
        self.model.eval()

    def test_single_graph(self):
        data = _make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            out = self.model(data)
        self.assertEqual(out.shape, (1, 64))

    def test_batched_graphs(self):
        graphs = [_make_dummy_graph() for _ in range(4)]
        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            out = self.model(batch)
        self.assertEqual(out.shape, (4, 64))

    def test_node_embeddings(self):
        data = _make_dummy_graph(n_atoms=8, n_bonds=10)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            node_out = self.model.get_node_embeddings(data)
        self.assertEqual(node_out.shape, (8, 64))


@unittest.skipUnless(_HAS_PYG, "PyTorch Geometric not installed")
class TestTandemM2M(unittest.TestCase):
    """Test TandemM2M architecture."""

    def setUp(self):
        from src.gnn.tandem_m2m import TandemM2M
        self.model = TandemM2M(
            in_dim=25, tabular_dim=56, gnn_hidden=128, gnn_out=64,
        )
        self.model.eval()

    def test_forward_shape(self):
        data = _make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        tabular = torch.randn(1, 56)
        with torch.no_grad():
            out = self.model(data, tabular)
        self.assertEqual(out.shape, (1, 1))

    def test_forward_batch(self):
        graphs = [_make_dummy_graph() for _ in range(3)]
        batch = Batch.from_data_list(graphs)
        tabular = torch.randn(3, 56)
        with torch.no_grad():
            out = self.model(batch, tabular)
        self.assertEqual(out.shape, (3, 1))

    def test_alpha_gradient(self):
        """Alpha parameter should have gradient."""
        from src.gnn.tandem_m2m import TandemM2M
        model = TandemM2M(in_dim=25, tabular_dim=56)
        data = _make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        tabular = torch.randn(1, 56)
        out = model(data, tabular)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.alpha.grad)

    def test_get_embedding(self):
        data = _make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            emb = self.model.get_embedding(data)
        self.assertEqual(emb.shape, (1, 64))

    def test_freeze_unfreeze(self):
        self.model.freeze_gnn_layers(2)
        params = self.model.count_parameters()
        self.assertGreater(params["frozen"], 0)

        self.model.unfreeze_all()
        params = self.model.count_parameters()
        self.assertEqual(params["frozen"], 0)

    def test_with_baseline(self):
        from src.gnn.tandem_m2m import TandemM2M
        model = TandemM2M(in_dim=25, tabular_dim=56, use_baseline=True)
        model.eval()
        data = _make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        tabular = torch.randn(1, 56)
        baseline = torch.tensor([[400.0]])
        with torch.no_grad():
            out = model(data, tabular, baseline)
        self.assertEqual(out.shape, (1, 1))


@unittest.skipUnless(_HAS_PYG, "PyTorch Geometric not installed")
class TestMultiTaskModel(unittest.TestCase):
    """Test MultiTaskTgModel."""

    def test_forward(self):
        from src.gnn.multitask import MultiTaskTgModel
        model = MultiTaskTgModel(in_dim=25)
        model.eval()
        data = _make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        with torch.no_grad():
            preds = model(data)
        self.assertIn("tg", preds)
        self.assertIn("density", preds)
        self.assertEqual(preds["tg"].shape, (1, 1))

    def test_compute_loss(self):
        from src.gnn.multitask import MultiTaskTgModel
        model = MultiTaskTgModel(in_dim=25)
        data = _make_dummy_graph()
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        preds = model(data)
        targets = {
            "tg": torch.tensor([350.0]),
            "density": torch.tensor([1.2]),
        }
        loss = model.compute_loss(preds, targets)
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
