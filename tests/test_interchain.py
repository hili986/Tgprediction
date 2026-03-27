"""Tests for Interchain Interaction Features (Phase B2)."""

import math
import unittest

from src.features.interchain_features import (
    FEATURE_NAMES,
    compute_interchain,
    interchain_feature_names,
    interchain_vector,
)


TEST_SMILES = {
    "PE": "*CC*",                          # Polyethylene (non-polar)
    "PS": "*CC(c1ccccc1)*",                # Polystyrene
    "PMMA": "*CC(C)(C(=O)OC)*",            # PMMA (polar)
    "PVC": "*CC(Cl)*",                     # PVC (very polar)
    "PDMS": "*[Si](C)(C)O*",              # PDMS (silicone)
}


class TestInterchainBasic(unittest.TestCase):
    """Basic interchain feature computation tests."""

    def test_feature_names_count(self):
        """Should have exactly 8 features."""
        self.assertEqual(len(interchain_feature_names()), 8)
        self.assertEqual(len(FEATURE_NAMES), 8)

    def test_feature_names_order(self):
        """Feature names should match FEATURE_NAMES constant."""
        self.assertEqual(interchain_feature_names(), FEATURE_NAMES)

    def test_vector_length(self):
        """interchain_vector should return 8 values."""
        for name, smi in TEST_SMILES.items():
            vec = interchain_vector(smi)
            self.assertEqual(len(vec), 8, f"Failed for {name}: {smi}")

    def test_no_nan_values(self):
        """All 5 test polymers should produce no NaN."""
        for name, smi in TEST_SMILES.items():
            vec = interchain_vector(smi)
            for i, v in enumerate(vec):
                self.assertFalse(
                    math.isnan(v),
                    f"NaN at index {i} ({FEATURE_NAMES[i]}) for {name}",
                )

    def test_dict_keys_match_names(self):
        """compute_interchain dict keys should match FEATURE_NAMES."""
        result = compute_interchain("*CC*")
        self.assertEqual(sorted(result.keys()), sorted(FEATURE_NAMES))

    def test_invalid_smiles(self):
        """Invalid SMILES should return all NaN (except as dict)."""
        result = compute_interchain("INVALID_SMILES_XXX")
        for name in FEATURE_NAMES:
            self.assertTrue(math.isnan(result[name]), f"{name} should be NaN")


class TestInterchainRanges(unittest.TestCase):
    """Value range sanity checks."""

    def test_partial_charges_range(self):
        """Gasteiger charges should be in reasonable range."""
        # PE is too small (ethane) — skip charge sign checks for it
        for name, smi in TEST_SMILES.items():
            result = compute_interchain(smi)
            self.assertGreater(result["MaxAbsPartialCharge"], 0,
                               f"MaxAbs should be positive for {name}")
        # Polar molecules should have positive max charge
        for name in ("PMMA", "PVC", "PDMS"):
            result = compute_interchain(TEST_SMILES[name])
            self.assertGreater(result["MaxPartialCharge"], 0,
                               f"Max charge should be positive for {name}")
            self.assertLess(result["MinPartialCharge"], 0,
                            f"Min charge should be negative for {name}")

    def test_dipole_moment_positive(self):
        """Dipole moment should be non-negative."""
        for name, smi in TEST_SMILES.items():
            result = compute_interchain(smi)
            self.assertGreaterEqual(result["dipole_moment"], 0,
                                    f"Dipole should be >= 0 for {name}")

    def test_polar_bond_fraction_range(self):
        """Polar bond fraction should be in [0, 1]."""
        for name, smi in TEST_SMILES.items():
            result = compute_interchain(smi)
            self.assertGreaterEqual(result["polar_bond_fraction"], 0)
            self.assertLessEqual(result["polar_bond_fraction"], 1)

    def test_hydrophobic_ratios_sum_to_one(self):
        """Hydrophobic + hydrophilic ratios should sum to ~1."""
        for name, smi in TEST_SMILES.items():
            result = compute_interchain(smi)
            total = result["hydrophobic_ratio"] + result["hydrophilic_ratio"]
            self.assertAlmostEqual(total, 1.0, places=5,
                                   msg=f"Ratios don't sum to 1 for {name}")

    def test_pe_less_polar_than_pvc(self):
        """PE should have lower polar_bond_fraction than PVC."""
        pe = compute_interchain(TEST_SMILES["PE"])
        pvc = compute_interchain(TEST_SMILES["PVC"])
        self.assertLess(pe["polar_bond_fraction"], pvc["polar_bond_fraction"])

    def test_ps_more_hydrophobic_than_pmma(self):
        """PS (aromatic, non-polar) should have higher hydrophobic_ratio than PMMA."""
        ps = compute_interchain(TEST_SMILES["PS"])
        pmma = compute_interchain(TEST_SMILES["PMMA"])
        self.assertGreater(ps["hydrophobic_ratio"], pmma["hydrophobic_ratio"])


class TestPipelineIntegration(unittest.TestCase):
    """Test integration with feature_pipeline."""

    def test_phy_b2_layer_exists(self):
        """PHY-B2 layer should be registered."""
        from src.features.feature_pipeline import LAYER_COMPONENTS
        self.assertIn("PHY-B2", LAYER_COMPONENTS)
        self.assertIn("interchain", LAYER_COMPONENTS["PHY-B2"])

    def test_phy_b2_feature_count(self):
        """PHY-B2 should produce 56 features (48 PHY + 8 interchain)."""
        from src.features.feature_pipeline import get_feature_names
        names = get_feature_names("PHY-B2")
        self.assertEqual(len(names), 56, f"Expected 56, got {len(names)}: {names}")

    def test_phy_b2_compute(self):
        """PHY-B2 compute_features should return 56-dim vector."""
        from src.features.feature_pipeline import compute_features
        x = compute_features("*CC*", layer="PHY-B2")
        self.assertEqual(len(x), 56)


if __name__ == "__main__":
    unittest.main()
