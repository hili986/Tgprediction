import unittest

import numpy as np

from scripts.predict_tg_tabpfn_186d import (
    BestTgPredictor,
    GNN_DIM,
    PBERT_PCA_DIM,
    PHY_C_LIGHT_DIM,
    _build_precomputed_component_lookup,
)


class TestPrecomputedLookup(unittest.TestCase):
    def test_build_lookup_keeps_only_finite_rows(self):
        smiles = np.array(["*CC(*)", "*CO(*)"])
        x_phyc = np.vstack([np.ones(PHY_C_LIGHT_DIM), np.full(PHY_C_LIGHT_DIM, np.nan)])
        x_gnn = np.ones((2, GNN_DIM))
        x_pbert = np.ones((2, PBERT_PCA_DIM))

        lookup = _build_precomputed_component_lookup(smiles, x_phyc, x_gnn, x_pbert)

        self.assertEqual(set(lookup), {"*CC(*)"})
        self.assertEqual(lookup["*CC(*)"]["chain_physics_source"], "precomputed")
        self.assertEqual(lookup["*CC(*)"]["phyc"].shape[0], PHY_C_LIGHT_DIM)
        self.assertEqual(lookup["*CC(*)"]["gnn"].shape[0], GNN_DIM)
        self.assertEqual(lookup["*CC(*)"]["pbert"].shape[0], PBERT_PCA_DIM)

    def test_featurize_component_uses_precomputed_lookup_before_recompute(self):
        predictor = BestTgPredictor.__new__(BestTgPredictor)
        predictor._component_cache = {}
        predictor._precomputed_component_lookup = {
            "*CC(*)": {
                "smiles": "*CC(*)",
                "phyc": np.ones(PHY_C_LIGHT_DIM),
                "gnn": np.ones(GNN_DIM),
                "pbert": np.ones(PBERT_PCA_DIM),
                "chain_physics_source": "precomputed",
            }
        }

        def _boom(*args, **kwargs):
            raise AssertionError("should not recompute")

        predictor._compute_phyc_light = _boom
        predictor._compute_gnn_embedding = _boom
        predictor._compute_polybert_pca = _boom

        result = predictor.featurize_component("*CC(*)")

        self.assertEqual(result["chain_physics_source"], "precomputed")
        self.assertIn("*CC(*)", predictor._component_cache)


if __name__ == "__main__":
    unittest.main()
