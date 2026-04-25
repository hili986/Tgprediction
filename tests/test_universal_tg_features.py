import math
import unittest

import numpy as np

from src.ml.universal_tg_features import (
    ComponentRecord,
    PolymerRecord,
    fox_tg_c,
    normalize_weights,
    polymer_record_to_features,
)


class TestUniversalTgFeatures(unittest.TestCase):
    def test_normalize_weights_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "sum to a positive"):
            normalize_weights([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_weights([0.5, -0.5])

    def test_normalize_weights_returns_unit_sum(self):
        weights = normalize_weights([2.0, 3.0])
        self.assertTrue(np.allclose(weights, [0.4, 0.6]))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_fox_tg_c_uses_kelvin_harmonic_mix(self):
        pred = fox_tg_c([0.0, 100.0], [0.5, 0.5])
        expected_k = 1.0 / (0.5 / 273.15 + 0.5 / 373.15)
        self.assertAlmostEqual(pred, expected_k - 273.15, places=6)

    def test_polymer_record_to_features_is_permutation_invariant_for_weighted_mean(self):
        a = ComponentRecord(
            smiles="A",
            vector=np.array([1.0, 3.0]),
            endpoint_tg_c=10.0,
            endpoint_source="measured",
        )
        b = ComponentRecord(
            smiles="B",
            vector=np.array([5.0, 7.0]),
            endpoint_tg_c=90.0,
            endpoint_source="measured",
        )
        rec1 = PolymerRecord(
            sample_id="ab",
            source="unit",
            architecture="random",
            components=[a, b],
            weights=[0.25, 0.75],
            target_tg_c=50.0,
        )
        rec2 = PolymerRecord(
            sample_id="ba",
            source="unit",
            architecture="random",
            components=[b, a],
            weights=[0.75, 0.25],
            target_tg_c=50.0,
        )
        row1 = polymer_record_to_features(rec1)
        row2 = polymer_record_to_features(rec2)
        self.assertAlmostEqual(row1["emb_mean_000"], row2["emb_mean_000"])
        self.assertAlmostEqual(row1["emb_mean_001"], row2["emb_mean_001"])
        self.assertAlmostEqual(row1["endpoint_tg_weighted_mean_c"], row2["endpoint_tg_weighted_mean_c"])
        self.assertEqual(row1["n_components"], 2)
        self.assertEqual(row1["is_random"], 1.0)
        self.assertEqual(row1["is_homopolymer"], 0.0)

    def test_missing_endpoint_sets_indicator_and_nan_priors(self):
        rec = PolymerRecord(
            sample_id="x",
            source="unit",
            architecture="homo",
            components=[
                ComponentRecord(
                    smiles="X",
                    vector=np.array([2.0]),
                    endpoint_tg_c=None,
                    endpoint_source="missing",
                )
            ],
            weights=[1.0],
            target_tg_c=20.0,
        )
        row = polymer_record_to_features(rec)
        self.assertEqual(row["endpoint_missing_count"], 1.0)
        self.assertTrue(math.isnan(row["endpoint_tg_fox_c"]))


if __name__ == "__main__":
    unittest.main()
