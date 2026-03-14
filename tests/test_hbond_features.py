"""Tests for H-bond SMARTS features (15-dim)."""

import unittest

import numpy as np

from src.features.hbond_features import (
    HBOND_SMARTS,
    compute_hbond_features,
    count_hbond_groups,
    hbond_density,
    hbond_feature_names,
    ced_weighted_sum,
)


class TestCountHbondGroups(unittest.TestCase):
    """Test L1: SMARTS match counts."""

    def test_amide_in_nylon(self):
        counts = count_hbond_groups("*C(=O)NCCCCC*")
        self.assertEqual(counts["amide"], 1)

    def test_urea_detected(self):
        counts = count_hbond_groups("*NCCCCCCNC(=O)N*")
        self.assertGreater(counts["urea"], 0)

    def test_urethane_detected(self):
        counts = count_hbond_groups("*OC(=O)NCCCCCCNC(=O)OCC*")
        self.assertGreater(counts["urethane"], 0)

    def test_hydroxyl_detected(self):
        counts = count_hbond_groups("*CC(O)*")
        self.assertGreater(counts["hydroxyl"], 0)

    def test_phosphoester_detected(self):
        counts = count_hbond_groups("*CCOP(=O)(OC)O*")
        self.assertGreater(counts["phosphoester"], 0)

    def test_benzimidazole_detected(self):
        counts = count_hbond_groups("*c1ccc2[nH]c(nc2c1)c1nc2cc(*)ccc2[nH]1")
        self.assertGreater(counts["benzimidazole"], 0)

    def test_invalid_smiles_returns_zeros(self):
        counts = count_hbond_groups("NOT_A_SMILES")
        self.assertTrue(all(v == 0 for v in counts.values()))

    def test_returns_all_keys(self):
        counts = count_hbond_groups("CC")
        self.assertEqual(set(counts.keys()), set(HBOND_SMARTS.keys()))


class TestHbondDensity(unittest.TestCase):
    """Test L2: density indicators."""

    def test_density_keys(self):
        d = hbond_density("*C(=O)NCCCCC*")
        expected = {"total_hbond_density", "strong_hbond_density",
                    "nucleic_relevance", "aromatic_hbond_density"}
        self.assertEqual(set(d.keys()), expected)

    def test_density_positive_for_amide(self):
        d = hbond_density("*C(=O)NCCCCC*")
        self.assertGreater(d["total_hbond_density"], 0)
        self.assertGreater(d["strong_hbond_density"], 0)

    def test_density_zero_for_no_hbond(self):
        d = hbond_density("CCCCCCCC")
        self.assertEqual(d["total_hbond_density"], 0)

    def test_invalid_smiles_returns_zeros(self):
        d = hbond_density("INVALID")
        self.assertTrue(all(v == 0.0 for v in d.values()))


class TestCedWeightedSum(unittest.TestCase):
    """Test L3: CED-weighted sum."""

    def test_amide_ced(self):
        ced = ced_weighted_sum("*C(=O)NCCCCC*")
        self.assertGreater(ced, 0)

    def test_stronger_hbond_higher_ced(self):
        # Urea > amide in CED contribution
        ced_urea = ced_weighted_sum("*NCCNC(=O)N*")
        ced_simple = ced_weighted_sum("CCCCCCCCCC")
        self.assertGreater(ced_urea, ced_simple)

    def test_invalid_smiles_returns_zero(self):
        self.assertEqual(ced_weighted_sum("INVALID"), 0.0)


class TestComputeHbondFeatures(unittest.TestCase):
    """Test combined 15-dim feature vector."""

    def test_output_shape(self):
        feat = compute_hbond_features("*C(=O)NCCCCC*")
        self.assertEqual(feat.shape, (15,))

    def test_output_dtype(self):
        feat = compute_hbond_features("CC")
        self.assertEqual(feat.dtype, float)

    def test_no_nan(self):
        feat = compute_hbond_features("*C(=O)NCCCCC*")
        self.assertFalse(np.any(np.isnan(feat)))

    def test_invalid_smiles_no_nan(self):
        feat = compute_hbond_features("INVALID")
        self.assertFalse(np.any(np.isnan(feat)))
        np.testing.assert_array_equal(feat, np.zeros(15))


class TestFeatureNames(unittest.TestCase):
    """Test feature name list."""

    def test_length(self):
        self.assertEqual(len(hbond_feature_names()), 15)

    def test_starts_with_counts(self):
        names = hbond_feature_names()
        self.assertTrue(names[0].startswith("hbond_count_"))

    def test_ends_with_ced(self):
        names = hbond_feature_names()
        self.assertEqual(names[-1], "ced_weighted_sum")


if __name__ == "__main__":
    unittest.main()
