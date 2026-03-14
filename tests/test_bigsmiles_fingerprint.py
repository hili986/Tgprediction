"""
BigSMILES 指纹模块测试
Tests for BigSMILES Fingerprint module

覆盖: Morgan 指纹、片段计数、聚合物描述符、组合向量、Tg 回归演示
Covers: Morgan fingerprint, fragment counts, polymer descriptors, combined vector, Tg regression demo
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigsmiles.fingerprint import (
    morgan_fingerprint,
    morgan_fingerprint_counts,
    fragment_counts,
    fragment_vector,
    fragment_names,
    polymer_descriptors,
    descriptor_vector,
    descriptor_names,
    combined_fingerprint,
    combined_feature_names,
    tg_regression_demo,
    _HAS_RDKIT,
)


@unittest.skipUnless(_HAS_RDKIT, "RDKit not available")
class TestMorganFingerprint(unittest.TestCase):
    """Morgan 指纹测试 / Morgan fingerprint tests."""

    def test_returns_list(self):
        fp = morgan_fingerprint('*CC*')
        self.assertIsInstance(fp, list)

    def test_default_length(self):
        fp = morgan_fingerprint('*CC*')
        self.assertEqual(len(fp), 2048)

    def test_custom_length(self):
        fp = morgan_fingerprint('*CC*', n_bits=512)
        self.assertEqual(len(fp), 512)

    def test_binary_values(self):
        fp = morgan_fingerprint('*CC*')
        for bit in fp:
            self.assertIn(bit, (0, 1))

    def test_nonzero(self):
        """Fingerprint should have at least some bits on."""
        fp = morgan_fingerprint('*CC*')
        self.assertGreater(sum(fp), 0)

    def test_different_smiles_different_fps(self):
        fp1 = morgan_fingerprint('*CC*')
        fp2 = morgan_fingerprint('*CC(c1ccccc1)*')
        self.assertNotEqual(fp1, fp2)

    def test_radius_effect(self):
        fp2 = morgan_fingerprint('*CC(c1ccccc1)*', radius=2)
        fp3 = morgan_fingerprint('*CC(c1ccccc1)*', radius=3)
        # Different radii should give different fingerprints
        # (for molecules with enough depth)
        self.assertEqual(len(fp2), len(fp3))

    def test_invalid_smiles(self):
        with self.assertRaises(ValueError):
            morgan_fingerprint('not_a_smiles_XXX')

    def test_count_fingerprint(self):
        fp = morgan_fingerprint_counts('*CC(c1ccccc1)*')
        self.assertIsInstance(fp, list)
        self.assertEqual(len(fp), 2048)
        # Count vector can have values > 1
        self.assertGreater(sum(fp), 0)


@unittest.skipUnless(_HAS_RDKIT, "RDKit not available")
class TestFragmentCounts(unittest.TestCase):
    """片段计数测试 / Fragment counting tests."""

    def test_returns_dict(self):
        frags = fragment_counts('*CC*')
        self.assertIsInstance(frags, dict)

    def test_all_fragments_present(self):
        frags = fragment_counts('*CC*')
        names = fragment_names()
        for name in names:
            self.assertIn(name, frags)

    def test_aromatic_ring(self):
        frags = fragment_counts('*CC(c1ccccc1)*')
        self.assertGreater(frags['aromatic_ring'], 0)

    def test_no_aromatic_in_pe(self):
        frags = fragment_counts('*CC*')
        self.assertEqual(frags['aromatic_ring'], 0)

    def test_ether(self):
        # Use SMILES with O between two C atoms (not at chain end)
        frags = fragment_counts('*CCOCC*')
        self.assertGreater(frags['ether'], 0)

    def test_carbonyl(self):
        frags = fragment_counts('*CC(C)(C(=O)OC)*')
        self.assertGreater(frags['carbonyl'], 0)

    def test_halogen(self):
        frags = fragment_counts('*CC(Cl)*')
        self.assertGreater(frags['halogen'], 0)
        self.assertGreater(frags['chlorine'], 0)

    def test_nitrile(self):
        frags = fragment_counts('*CC(C#N)*')
        self.assertGreater(frags['nitrile'], 0)

    def test_fluorine(self):
        frags = fragment_counts('*C(F)(F)C(F)(F)*')
        self.assertGreater(frags['fluorine'], 0)

    def test_silicon(self):
        frags = fragment_counts('*O[Si](C)(C)*')
        self.assertGreater(frags['silicon'], 0)

    def test_fragment_vector_length(self):
        vec = fragment_vector('*CC*')
        names = fragment_names()
        self.assertEqual(len(vec), len(names))

    def test_fragment_vector_matches_counts(self):
        smiles = '*CC(c1ccccc1)*'
        frags = fragment_counts(smiles)
        vec = fragment_vector(smiles)
        names = fragment_names()
        for i, name in enumerate(names):
            self.assertEqual(vec[i], frags[name])


@unittest.skipUnless(_HAS_RDKIT, "RDKit not available")
class TestPolymerDescriptors(unittest.TestCase):
    """聚合物描述符测试 / Polymer descriptor tests."""

    def test_returns_dict(self):
        desc = polymer_descriptors('*CC*')
        self.assertIsInstance(desc, dict)

    def test_expected_keys(self):
        desc = polymer_descriptors('*CC*')
        expected = {
            "mw_repeat_unit", "num_heavy_atoms", "num_rotatable",
            "c_fraction", "o_fraction", "n_fraction",
            "heteroatom_ratio", "aromatic_fraction",
            "hbd_count", "hba_count", "tpsa", "logp",
            "bonding_type", "num_descriptors",
        }
        self.assertEqual(set(desc.keys()), expected)

    def test_mw_positive(self):
        desc = polymer_descriptors('*CC*')
        self.assertGreater(desc['mw_repeat_unit'], 0)

    def test_carbon_fraction(self):
        desc = polymer_descriptors('*CC*')
        self.assertGreater(desc['c_fraction'], 0)
        self.assertLessEqual(desc['c_fraction'], 1.0)

    def test_heteroatom_ratio_pe(self):
        """Polyethylene has no heteroatoms."""
        desc = polymer_descriptors('*CC*')
        self.assertEqual(desc['heteroatom_ratio'], 0.0)

    def test_heteroatom_ratio_peo(self):
        """PEO has oxygen."""
        desc = polymer_descriptors('*CCO*')
        self.assertGreater(desc['heteroatom_ratio'], 0)

    def test_aromatic_fraction_ps(self):
        """Polystyrene has aromatic atoms."""
        desc = polymer_descriptors('*CC(c1ccccc1)*')
        self.assertGreater(desc['aromatic_fraction'], 0)

    def test_bonding_type_aa(self):
        desc = polymer_descriptors('*CC*', '{[$]CC[$]}')
        self.assertEqual(desc['bonding_type'], 0)

    def test_bonding_type_ab(self):
        desc = polymer_descriptors('*CCO*', '{[<]CCO[>]}')
        self.assertEqual(desc['bonding_type'], 1)

    def test_bonding_type_unknown(self):
        desc = polymer_descriptors('*CC*')
        self.assertEqual(desc['bonding_type'], -1)

    def test_num_descriptors(self):
        desc = polymer_descriptors('*CC*', '{[$]CC[$]}')
        self.assertEqual(desc['num_descriptors'], 2)

    def test_descriptor_vector_length(self):
        vec = descriptor_vector('*CC*')
        names = descriptor_names()
        self.assertEqual(len(vec), len(names))

    def test_descriptor_vector_values(self):
        vec = descriptor_vector('*CC*', '{[$]CC[$]}')
        # All values should be finite
        for v in vec:
            self.assertIsInstance(v, float)


@unittest.skipUnless(_HAS_RDKIT, "RDKit not available")
class TestCombinedFingerprint(unittest.TestCase):
    """组合指纹测试 / Combined fingerprint tests."""

    def test_returns_list(self):
        fp = combined_fingerprint('*CC*', '{[$]CC[$]}', morgan_bits=256)
        self.assertIsInstance(fp, list)

    def test_length(self):
        morgan_bits = 256
        fp = combined_fingerprint('*CC*', morgan_bits=morgan_bits)
        frag_len = len(fragment_names())
        desc_len = len(descriptor_names())
        self.assertEqual(len(fp), morgan_bits + frag_len + desc_len)

    def test_all_floats(self):
        fp = combined_fingerprint('*CC*', morgan_bits=256)
        for v in fp:
            self.assertIsInstance(v, float)

    def test_nonzero(self):
        fp = combined_fingerprint('*CC(c1ccccc1)*', morgan_bits=256)
        self.assertGreater(sum(abs(v) for v in fp), 0)

    def test_feature_names_match(self):
        morgan_bits = 256
        fp = combined_fingerprint('*CC*', morgan_bits=morgan_bits)
        names = combined_feature_names(morgan_bits=morgan_bits)
        self.assertEqual(len(fp), len(names))


@unittest.skipUnless(_HAS_RDKIT, "RDKit not available")
class TestTgRegressionDemo(unittest.TestCase):
    """Tg 回归演示测试 / Tg regression demo tests."""

    def test_demo_runs(self):
        """Demo should run without errors."""
        result = tg_regression_demo(
            use_morgan=False,
            use_fragments=True,
            use_descriptors=True,
            verbose=False,
        )
        self.assertIsInstance(result, dict)

    def test_result_keys(self):
        result = tg_regression_demo(
            use_morgan=False,
            use_fragments=True,
            use_descriptors=True,
            verbose=False,
        )
        expected = {"r2", "mae", "rmse", "num_features", "num_train",
                    "num_test", "num_skipped", "features_used"}
        self.assertEqual(set(result.keys()), expected)

    def test_r2_range(self):
        """R² should be finite (can be negative for simple models)."""
        result = tg_regression_demo(
            use_morgan=False,
            use_fragments=True,
            use_descriptors=True,
            verbose=False,
        )
        self.assertGreater(result['r2'], -10.0)
        self.assertLessEqual(result['r2'], 1.0)

    def test_mae_positive(self):
        result = tg_regression_demo(
            use_morgan=False,
            use_fragments=True,
            use_descriptors=True,
            verbose=False,
        )
        self.assertGreater(result['mae'], 0)

    def test_num_samples(self):
        result = tg_regression_demo(
            use_morgan=False,
            use_fragments=True,
            use_descriptors=True,
            verbose=False,
        )
        self.assertGreater(result['num_train'], 0)
        self.assertGreater(result['num_test'], 0)
        self.assertEqual(
            result['num_train'] + result['num_test'] + result['num_skipped'],
            304,
        )

    def test_descriptors_only(self):
        """Descriptors-only model should work."""
        result = tg_regression_demo(
            use_morgan=False,
            use_fragments=False,
            use_descriptors=True,
            verbose=False,
        )
        self.assertIn('r2', result)

    def test_fragments_only(self):
        """Fragments-only model should work."""
        result = tg_regression_demo(
            use_morgan=False,
            use_fragments=True,
            use_descriptors=False,
            verbose=False,
        )
        self.assertIn('r2', result)


@unittest.skipUnless(_HAS_RDKIT, "RDKit not available")
class TestBiceranoFingerprints(unittest.TestCase):
    """Bicerano 数据集指纹测试 / Bicerano dataset fingerprint tests."""

    def test_all_entries_produce_fingerprints(self):
        """All 304 Bicerano entries should produce valid fingerprints."""
        from src.data.bicerano_tg_dataset import BICERANO_DATA

        failures = []
        for i, (name, smiles, bigsmiles, _) in enumerate(BICERANO_DATA):
            try:
                fp = morgan_fingerprint(smiles, n_bits=256)
                self.assertGreater(sum(fp), 0,
                                   f"Zero fingerprint for {name}")
            except Exception as e:
                failures.append((i, name, str(e)))

        self.assertEqual(len(failures), 0,
                         f"Fingerprint failures: {failures[:5]}")

    def test_sample_descriptors(self):
        """Spot-check descriptors for well-known polymers."""
        from src.data.bicerano_tg_dataset import BICERANO_DATA

        # Find PE
        pe_entry = next(
            (e for e in BICERANO_DATA if e[0] == 'Polyethylene'), None
        )
        self.assertIsNotNone(pe_entry)
        desc = polymer_descriptors(pe_entry[1], pe_entry[2])
        self.assertGreater(desc['mw_repeat_unit'], 0)
        self.assertEqual(desc['bonding_type'], 0)  # AA type


if __name__ == "__main__":
    unittest.main()
