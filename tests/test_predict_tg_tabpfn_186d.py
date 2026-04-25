import unittest

import numpy as np

from scripts.predict_tg_tabpfn_186d import (
    BestTgPredictor,
    GNN_DIM,
    PBERT_PCA_DIM,
    PHY_C_LIGHT_DIM,
    _canonical_repeat_unit_key,
    _build_precomputed_component_lookup,
)


class TestPrecomputedLookup(unittest.TestCase):
    def test_canonical_repeat_unit_key_returns_none_for_non_repeat_unit(self):
        self.assertIsNone(_canonical_repeat_unit_key("CCO"))

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
        predictor._component_error_cache = {}
        predictor._canonical_component_lookup = {}
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

    def test_featurize_component_uses_canonical_precomputed_lookup(self):
        predictor = BestTgPredictor.__new__(BestTgPredictor)
        predictor._component_cache = {}
        predictor._component_error_cache = {}
        predictor._precomputed_component_lookup = {}
        predictor._canonical_component_lookup = {
            "*CC(*)C#N": {
                "smiles": "*CC(*)C#N",
                "phyc": np.ones(PHY_C_LIGHT_DIM),
                "gnn": np.ones(GNN_DIM),
                "pbert": np.ones(PBERT_PCA_DIM),
                "chain_physics_source": "precomputed_canonical",
            }
        }

        def _boom(*args, **kwargs):
            raise AssertionError("should not recompute")

        predictor._compute_phyc_light = _boom
        predictor._compute_gnn_embedding = _boom
        predictor._compute_polybert_pca = _boom

        result = predictor.featurize_component("*C(C#N)C*")

        self.assertEqual(result["smiles"], "*CC(*)C#N")
        self.assertEqual(result["chain_physics_source"], "precomputed_canonical")

    def test_precomputed_component_match_reports_source(self):
        predictor = BestTgPredictor.__new__(BestTgPredictor)
        predictor._precomputed_component_lookup = {"*CC(*)": {"smiles": "*CC(*)"}}
        predictor._canonical_component_lookup = {
            "*CC(*)C#N": {"smiles": "*CC(*)C#N"}
        }

        self.assertEqual(predictor.precomputed_component_match("*CC(*)"), ("exact", "*CC(*)"))
        self.assertEqual(
            predictor.precomputed_component_match("*C(C#N)C*"),
            ("canonical", "*CC(*)C#N"),
        )
        self.assertEqual(predictor.precomputed_component_match("*CO(*)"), ("miss", None))
        self.assertEqual(predictor.precomputed_component_match("CCO"), ("invalid", None))

    def test_component_homopolymer_prediction_is_cached(self):
        predictor = BestTgPredictor.__new__(BestTgPredictor)
        predictor._homopolymer_tg_cache = {}
        calls = []

        def _fake_predict(matrix):
            calls.append(np.asarray(matrix).shape)
            return np.array([321.0])

        predictor._predict_from_full_matrix = _fake_predict
        component = {
            "smiles": "*CC(*)",
            "phyc": np.ones(PHY_C_LIGHT_DIM),
            "gnn": np.ones(GNN_DIM),
            "pbert": np.ones(PBERT_PCA_DIM),
            "chain_physics_source": "precomputed",
        }

        self.assertEqual(predictor._predict_component_homopolymer_k(component), 321.0)
        self.assertEqual(predictor._predict_component_homopolymer_k(component), 321.0)
        self.assertEqual(len(calls), 1)

    def test_multicomponent_batch_predicts_descriptor_rows_together(self):
        predictor = BestTgPredictor.__new__(BestTgPredictor)
        predictor._component_cache = {}
        predictor._homopolymer_tg_cache = {}
        predictor._component_error_cache = {}
        predictor._canonical_component_lookup = {}
        predictor._precomputed_component_lookup = {
            smiles: {
                "smiles": smiles,
                "phyc": np.ones(PHY_C_LIGHT_DIM) * idx,
                "gnn": np.ones(GNN_DIM) * idx,
                "pbert": np.ones(PBERT_PCA_DIM) * idx,
                "chain_physics_source": "precomputed",
            }
            for idx, smiles in enumerate(["*CC(*)", "*CO(*)", "*CN(*)"], start=1)
        }
        calls = []

        def _fake_predict(matrix):
            n_rows = np.asarray(matrix).shape[0]
            calls.append(n_rows)
            if len(calls) == 1:
                return np.arange(n_rows, dtype=float) + 300.0
            return np.arange(n_rows, dtype=float) + 350.0

        predictor._predict_from_full_matrix = _fake_predict

        results = predictor.predict_multicomponent_batch(
            [
                (["*CC(*)", "*CO(*)"], [0.5, 0.5], "random"),
                (["*CC(*)", "*CN(*)"], [0.5, 0.5], "random"),
            ]
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(calls, [3, 2])
        self.assertEqual(set(predictor._homopolymer_tg_cache), {"*CC(*)", "*CO(*)", "*CN(*)"})
        self.assertEqual(results[0]["tg_k_pred"], 350.0)
        self.assertEqual(results[1]["tg_k_pred"], 351.0)

    def test_featurize_component_caches_failures(self):
        predictor = BestTgPredictor.__new__(BestTgPredictor)
        predictor._component_cache = {}
        predictor._component_error_cache = {}
        predictor._precomputed_component_lookup = {}
        predictor._canonical_component_lookup = {}
        calls = []

        def _fail(smiles):
            calls.append(smiles)
            raise ValueError("PHY-B2 feature extraction failed.")

        predictor._compute_phyc_light = _fail
        predictor._compute_gnn_embedding = lambda smiles: np.ones(GNN_DIM)
        predictor._compute_polybert_pca = lambda smiles: np.ones(PBERT_PCA_DIM)

        with self.assertRaisesRegex(ValueError, "Component featurization failed"):
            predictor.featurize_component("*CC(*)")
        with self.assertRaisesRegex(ValueError, "Component featurization failed"):
            predictor.featurize_component("*CC(*)")

        self.assertEqual(calls, ["*CC(*)"])
        self.assertIn("*CC(*)", predictor._component_error_cache)


if __name__ == "__main__":
    unittest.main()
