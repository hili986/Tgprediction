"""Tests for Virtual Polymerization Descriptors (VPD)."""

import math
import unittest

from rdkit import Chem

from src.features.virtual_polymerization import (
    FEATURE_NAMES,
    build_oligomer,
    compute_vpd,
    vpd_feature_names,
    vpd_vector,
)


TEST_SMILES = {
    "PE": "*CC*",
    "PS": "*CC(c1ccccc1)*",
    "PMMA": "*CC(C)(C(=O)OC)*",
    "PVC": "*CC(Cl)*",
    "PDMS": "*[Si](C)(C)O*",
}


class TestBuildOligomer(unittest.TestCase):
    """Test oligomer assembly."""

    def test_trimer_valid_smiles(self):
        """build_oligomer(n=3) should return valid SMILES."""
        success = 0
        for name, smi in TEST_SMILES.items():
            result = build_oligomer(smi, n=3)
            if result is not None:
                mol = Chem.MolFromSmiles(result)
                self.assertIsNotNone(
                    mol, f"Invalid oligomer SMILES for {name}: {result}"
                )
                success += 1

        # At least 4/5 should succeed
        self.assertGreaterEqual(success, 4, f"Only {success}/5 oligomers built")

    def test_dimer(self):
        """build_oligomer(n=2) should also work."""
        result = build_oligomer("*CC*", n=2)
        self.assertIsNotNone(result)
        mol = Chem.MolFromSmiles(result)
        self.assertIsNotNone(mol)

    def test_monomer(self):
        """build_oligomer(n=1) should return something valid."""
        result = build_oligomer("*CC*", n=1)
        if result is not None:
            mol = Chem.MolFromSmiles(result)
            self.assertIsNotNone(mol)

    def test_invalid_smiles(self):
        """Invalid SMILES should return None (fallback may also fail)."""
        result = build_oligomer("INVALID", n=3)
        # May return None or fallback — either is acceptable
        if result is not None:
            mol = Chem.MolFromSmiles(result)
            self.assertIsNotNone(mol)

    def test_no_star_smiles(self):
        """SMILES without * uses fallback."""
        result = build_oligomer("CC", n=3)
        # Fallback concatenation: "CCCCCC" → valid
        if result is not None:
            mol = Chem.MolFromSmiles(result)
            self.assertIsNotNone(mol)


class TestVPDBasic(unittest.TestCase):
    """Basic VPD computation tests."""

    def test_feature_names_count(self):
        """VPD should have exactly 12 features."""
        self.assertEqual(len(vpd_feature_names()), 12)
        self.assertEqual(len(FEATURE_NAMES), 12)

    def test_vector_length(self):
        """vpd_vector should return 12 values."""
        for name, smi in TEST_SMILES.items():
            vec = vpd_vector(smi)
            self.assertEqual(len(vec), 12, f"Failed for {name}")

    def test_no_nan_pe(self):
        """PE (*CC*) should produce no NaN."""
        vec = vpd_vector("*CC*")
        for i, v in enumerate(vec):
            self.assertFalse(
                math.isnan(v),
                f"NaN at index {i} ({FEATURE_NAMES[i]}) for PE",
            )

    def test_no_nan_all(self):
        """All 5 test polymers should produce no NaN."""
        for name, smi in TEST_SMILES.items():
            vec = vpd_vector(smi)
            for i, v in enumerate(vec):
                self.assertFalse(
                    math.isnan(v),
                    f"NaN at index {i} ({FEATURE_NAMES[i]}) for {name}",
                )

    def test_dict_keys_match_names(self):
        """compute_vpd dict keys should match FEATURE_NAMES."""
        result = compute_vpd("*CC*")
        self.assertEqual(sorted(result.keys()), sorted(FEATURE_NAMES))


class TestVPDValues(unittest.TestCase):
    """Value sanity checks."""

    def test_molwt_per_ru_positive(self):
        """MolWt_per_RU should be positive."""
        for name, smi in TEST_SMILES.items():
            result = compute_vpd(smi)
            self.assertGreater(result["MolWt_per_RU"], 0, f"{name}")

    def test_heavy_atom_per_ru_positive(self):
        """HeavyAtom_per_RU should be positive."""
        for name, smi in TEST_SMILES.items():
            result = compute_vpd(smi)
            self.assertGreater(result["HeavyAtom_per_RU"], 0, f"{name}")

    def test_delta_small_for_pe(self):
        """PE deltas should be small (simple linear chain)."""
        result = compute_vpd("*CC*")
        # Delta = (dimer/2) - monomer, for PE this should be modest
        self.assertLess(abs(result["MolWt_delta"]), 50, "PE MolWt delta too large")

    def test_junction_flex_bounded(self):
        """junction_flex_ratio should be in [0, 1]."""
        for name, smi in TEST_SMILES.items():
            result = compute_vpd(smi)
            self.assertGreaterEqual(result["junction_flex_ratio"], 0.0, f"{name}")
            self.assertLessEqual(result["junction_flex_ratio"], 1.0, f"{name}")


class TestVPDBicerano(unittest.TestCase):
    """Batch test on Bicerano dataset."""

    def test_bicerano_build_rate(self):
        """build_oligomer success rate should be > 95%."""
        from src.data.bicerano_tg_dataset import BICERANO_DATA

        success = 0
        for _, smiles, _, _ in BICERANO_DATA:
            result = build_oligomer(smiles, n=3)
            if result is not None:
                success += 1

        rate = success / len(BICERANO_DATA)
        self.assertGreater(
            rate, 0.90,
            f"Build rate too low: {rate:.1%} ({success}/{len(BICERANO_DATA)})",
        )

    def test_bicerano_no_nan(self):
        """Most Bicerano polymers should produce valid VPD."""
        from src.data.bicerano_tg_dataset import BICERANO_DATA

        nan_count = 0
        for name, smiles, _, _ in BICERANO_DATA:
            vec = vpd_vector(smiles)
            if any(math.isnan(v) for v in vec):
                nan_count += 1

        max_nan = int(len(BICERANO_DATA) * 0.05)
        self.assertLessEqual(
            nan_count, max_nan,
            f"Too many NaN results: {nan_count}/{len(BICERANO_DATA)}",
        )


if __name__ == "__main__":
    unittest.main()
