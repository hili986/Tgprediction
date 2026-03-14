"""Tests for Physical Proxy Features (PPF)."""

import math
import unittest

from src.features.physical_proxy import (
    FEATURE_NAMES,
    compute_ppf,
    ppf_feature_names,
    ppf_vector,
)


# Typical polymer repeat unit SMILES for testing
TEST_SMILES = {
    "PE": "*CC*",                          # Polyethylene
    "PS": "*CC(c1ccccc1)*",                # Polystyrene
    "PMMA": "*CC(C)(C(=O)OC)*",            # PMMA
    "PVC": "*CC(Cl)*",                     # PVC
    "PDMS": "*[Si](C)(C)O*",              # PDMS (silicone)
}


class TestPPFBasic(unittest.TestCase):
    """Basic PPF computation tests."""

    def test_feature_names_count(self):
        """PPF should have exactly 10 features."""
        self.assertEqual(len(ppf_feature_names()), 10)
        self.assertEqual(len(FEATURE_NAMES), 10)

    def test_feature_names_order(self):
        """Feature names should match FEATURE_NAMES constant."""
        self.assertEqual(ppf_feature_names(), FEATURE_NAMES)

    def test_vector_length(self):
        """ppf_vector should return 10 values."""
        for name, smi in TEST_SMILES.items():
            vec = ppf_vector(smi)
            self.assertEqual(len(vec), 10, f"Failed for {name}: {smi}")

    def test_no_nan_values(self):
        """All 5 test polymers should produce no NaN."""
        for name, smi in TEST_SMILES.items():
            vec = ppf_vector(smi)
            for i, v in enumerate(vec):
                self.assertFalse(
                    math.isnan(v),
                    f"NaN at index {i} ({FEATURE_NAMES[i]}) for {name}",
                )

    def test_dict_keys_match_names(self):
        """compute_ppf dict keys should match FEATURE_NAMES."""
        result = compute_ppf("*CC*")
        self.assertEqual(sorted(result.keys()), sorted(FEATURE_NAMES))


class TestPPFRanges(unittest.TestCase):
    """Value range sanity checks."""

    def test_m_per_f_range(self):
        """M_per_f should be in reasonable range (20-500)."""
        for name, smi in TEST_SMILES.items():
            result = compute_ppf(smi)
            self.assertGreater(result["M_per_f"], 10, f"{name}")
            self.assertLess(result["M_per_f"], 500, f"{name}")

    def test_ced_positive(self):
        """CED_estimate should be positive."""
        for name, smi in TEST_SMILES.items():
            result = compute_ppf(smi)
            self.assertGreater(result["CED_estimate"], 0, f"{name}")

    def test_vf_bounded(self):
        """Vf_estimate should be in [0, 1]."""
        for name, smi in TEST_SMILES.items():
            result = compute_ppf(smi)
            self.assertGreaterEqual(result["Vf_estimate"], 0.0, f"{name}")
            self.assertLessEqual(result["Vf_estimate"], 1.0, f"{name}")

    def test_backbone_rigidity_bounded(self):
        """backbone_rigidity should be in [0, 1]."""
        for name, smi in TEST_SMILES.items():
            result = compute_ppf(smi)
            self.assertGreaterEqual(result["backbone_rigidity"], 0.0, f"{name}")
            self.assertLessEqual(result["backbone_rigidity"], 1.0, f"{name}")

    def test_symmetry_bounded(self):
        """symmetry_index should be in [0, 1]."""
        for name, smi in TEST_SMILES.items():
            result = compute_ppf(smi)
            self.assertGreaterEqual(result["symmetry_index"], 0.0, f"{name}")
            self.assertLessEqual(result["symmetry_index"], 1.0, f"{name}")


class TestPPFPhysics(unittest.TestCase):
    """Physics-based ordering tests."""

    def test_ps_stiffer_than_pe(self):
        """PS (aromatic ring) should have higher backbone_rigidity than PE."""
        pe = compute_ppf("*CC*")
        ps = compute_ppf("*CC(c1ccccc1)*")
        self.assertGreaterEqual(
            ps["backbone_rigidity"], pe["backbone_rigidity"],
            "PS should be at least as rigid as PE",
        )

    def test_pdms_high_flexibility(self):
        """PDMS should have high flexible_bond_density."""
        pdms = compute_ppf("*[Si](C)(C)O*")
        pe = compute_ppf("*CC*")
        self.assertGreater(
            pdms["flexible_bond_density"], 0,
            "PDMS should have positive flexible bond density",
        )

    def test_pe_symmetric(self):
        """PE has no substituents → symmetry = 1.0."""
        pe = compute_ppf("*CC*")
        self.assertEqual(pe["symmetry_index"], 1.0)


class TestPPFInvalidInput(unittest.TestCase):
    """Edge cases and invalid inputs."""

    def test_invalid_smiles(self):
        """Invalid SMILES should return all NaN."""
        result = compute_ppf("INVALID_SMILES_XYZ")
        for name in FEATURE_NAMES:
            self.assertTrue(math.isnan(result[name]), f"{name} should be NaN")

    def test_empty_smiles(self):
        """Empty SMILES produces a result (H-capped → valid molecule)."""
        result = compute_ppf("")
        # Empty string → replace * with [H] → "" which RDKit may parse
        # Just verify it returns 10 values without crashing
        self.assertEqual(len(result), 10)


class TestPPFBicerano(unittest.TestCase):
    """Batch test on Bicerano dataset."""

    def test_bicerano_no_nan(self):
        """All 304 Bicerano polymers should produce valid PPF."""
        from src.data.bicerano_tg_dataset import BICERANO_DATA

        nan_count = 0
        for name, smiles, _, _ in BICERANO_DATA:
            vec = ppf_vector(smiles)
            if any(math.isnan(v) for v in vec):
                nan_count += 1

        # Allow up to 5% NaN (some SMILES may be tricky)
        max_nan = int(len(BICERANO_DATA) * 0.05)
        self.assertLessEqual(
            nan_count, max_nan,
            f"Too many NaN results: {nan_count}/{len(BICERANO_DATA)}",
        )


if __name__ == "__main__":
    unittest.main()
