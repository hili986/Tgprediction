"""
Bicerano Tg 数据集测试
Tests for Bicerano Tg Dataset module

覆盖: 数据完整性、API 函数、导出、校验、简写转换器
Covers: data integrity, API functions, export, validation, shorthand converter
"""

import unittest
import os
import sys
import json
import csv
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.bicerano_tg_dataset import (
    BICERANO_DATA,
    load_dataset,
    get_names,
    get_smiles,
    get_bigsmiles,
    get_tg_values,
    to_csv,
    to_json,
    validate_all,
    summary,
    shorthand_to_bracket,
)


class TestDataIntegrity(unittest.TestCase):
    """数据完整性测试 / Data integrity tests."""

    def test_dataset_count(self):
        self.assertEqual(len(BICERANO_DATA), 304)

    def test_entry_structure(self):
        for i, entry in enumerate(BICERANO_DATA):
            with self.subTest(index=i):
                self.assertEqual(len(entry), 4)
                name, smiles, bigsmiles, tg_k = entry
                self.assertIsInstance(name, str)
                self.assertIsInstance(smiles, str)
                self.assertIsInstance(bigsmiles, str)
                self.assertIsInstance(tg_k, int)

    def test_names_non_empty(self):
        for i, (name, _, _, _) in enumerate(BICERANO_DATA):
            with self.subTest(index=i):
                self.assertGreater(len(name), 0)

    def test_smiles_non_empty(self):
        for i, (_, smiles, _, _) in enumerate(BICERANO_DATA):
            with self.subTest(index=i):
                self.assertGreater(len(smiles), 0)

    def test_bigsmiles_non_empty(self):
        for i, (_, _, bigsmiles, _) in enumerate(BICERANO_DATA):
            with self.subTest(index=i):
                self.assertGreater(len(bigsmiles), 0)

    def test_tg_positive(self):
        for i, (name, _, _, tg_k) in enumerate(BICERANO_DATA):
            with self.subTest(index=i, name=name):
                self.assertGreater(tg_k, 0, f"Tg must be positive: {name}")

    def test_tg_range(self):
        """Tg values should be in a physically reasonable range (50-900 K)."""
        for i, (name, _, _, tg_k) in enumerate(BICERANO_DATA):
            with self.subTest(index=i, name=name):
                self.assertGreaterEqual(tg_k, 50)
                self.assertLessEqual(tg_k, 900)

    def test_bigsmiles_bracket_notation(self):
        """All BigSMILES strings should use bracket notation."""
        for i, (name, _, bigsmiles, _) in enumerate(BICERANO_DATA):
            with self.subTest(index=i, name=name):
                self.assertIn('{', bigsmiles)
                self.assertIn('}', bigsmiles)
                # Should contain bracket descriptors
                has_bracket_desc = (
                    '[$]' in bigsmiles or
                    '[<]' in bigsmiles or
                    '[>]' in bigsmiles
                )
                self.assertTrue(has_bracket_desc,
                                f"Missing bracket descriptor: {bigsmiles}")

    def test_known_polymers(self):
        """Spot-check well-known polymers."""
        data = {name: (smiles, bigsmiles, tg) for name, smiles, bigsmiles, tg in BICERANO_DATA}

        # Polyethylene
        self.assertIn('Polyethylene', data)
        self.assertEqual(data['Polyethylene'][2], 195)

        # Polystyrene
        self.assertIn('Polystyrene', data)

    def test_no_duplicate_entries(self):
        """No exact duplicate (name, smiles, bigsmiles, tg) tuples."""
        self.assertEqual(len(set(BICERANO_DATA)), len(BICERANO_DATA))


class TestLoadDataset(unittest.TestCase):
    """load_dataset() 测试 / load_dataset() tests."""

    def test_returns_list(self):
        data = load_dataset()
        self.assertIsInstance(data, list)

    def test_count_matches(self):
        data = load_dataset()
        self.assertEqual(len(data), 304)

    def test_dict_keys(self):
        data = load_dataset()
        expected_keys = {"name", "smiles", "bigsmiles", "tg_k"}
        for entry in data:
            self.assertEqual(set(entry.keys()), expected_keys)

    def test_first_entry(self):
        data = load_dataset()
        first = data[0]
        self.assertIsInstance(first["name"], str)
        self.assertIsInstance(first["tg_k"], int)


class TestGetters(unittest.TestCase):
    """Getter 函数测试 / Getter function tests."""

    def test_get_names(self):
        names = get_names()
        self.assertEqual(len(names), 304)
        self.assertIsInstance(names[0], str)

    def test_get_smiles(self):
        smiles = get_smiles()
        self.assertEqual(len(smiles), 304)
        # SMILES should contain * (attachment points)
        for s in smiles:
            self.assertIn('*', s)

    def test_get_bigsmiles(self):
        bs = get_bigsmiles()
        self.assertEqual(len(bs), 304)
        for b in bs:
            self.assertIn('{', b)

    def test_get_tg_values(self):
        tgs = get_tg_values()
        self.assertEqual(len(tgs), 304)
        for tg in tgs:
            self.assertIsInstance(tg, int)
            self.assertGreater(tg, 0)


class TestExportCSV(unittest.TestCase):
    """CSV 导出测试 / CSV export tests."""

    def test_to_csv(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False, encoding='utf-8') as f:
            path = f.name
        try:
            to_csv(path)
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ['name', 'smiles', 'bigsmiles', 'tg_k'])
                rows = list(reader)
                self.assertEqual(len(rows), 304)
        finally:
            os.unlink(path)

    def test_csv_roundtrip(self):
        """CSV export should preserve data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                         delete=False, encoding='utf-8') as f:
            path = f.name
        try:
            to_csv(path)
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for i, row in enumerate(reader):
                    name, smiles, bigsmiles, tg_str = row
                    orig = BICERANO_DATA[i]
                    self.assertEqual(name, orig[0])
                    self.assertEqual(smiles, orig[1])
                    self.assertEqual(bigsmiles, orig[2])
                    self.assertEqual(int(tg_str), orig[3])
        finally:
            os.unlink(path)


class TestExportJSON(unittest.TestCase):
    """JSON 导出测试 / JSON export tests."""

    def test_to_json(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False, encoding='utf-8') as f:
            path = f.name
        try:
            to_json(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.assertEqual(len(data), 304)
            self.assertIn('name', data[0])
            self.assertIn('tg_k', data[0])
        finally:
            os.unlink(path)

    def test_json_values(self):
        """JSON export should preserve data types."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False, encoding='utf-8') as f:
            path = f.name
        try:
            to_json(path)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for i, entry in enumerate(data):
                orig = BICERANO_DATA[i]
                self.assertEqual(entry['name'], orig[0])
                self.assertEqual(entry['tg_k'], orig[3])
        finally:
            os.unlink(path)


class TestSummary(unittest.TestCase):
    """summary() 测试 / summary() tests."""

    def test_summary_keys(self):
        s = summary()
        expected = {"total_entries", "tg_min_k", "tg_max_k", "tg_mean_k",
                    "unique_smiles", "source"}
        self.assertEqual(set(s.keys()), expected)

    def test_summary_values(self):
        s = summary()
        self.assertEqual(s["total_entries"], 304)
        self.assertGreater(s["tg_min_k"], 0)
        self.assertGreater(s["tg_max_k"], s["tg_min_k"])
        self.assertIsInstance(s["tg_mean_k"], float)
        self.assertGreater(s["unique_smiles"], 0)

    def test_summary_source(self):
        s = summary()
        self.assertIn("Choi", s["source"])


class TestShorthandConverter(unittest.TestCase):
    """简写转换器测试 / Shorthand converter tests."""

    def test_dollar_simple(self):
        self.assertEqual(shorthand_to_bracket('{$CC$}'), '{[$]CC[$]}')

    def test_angle_simple(self):
        self.assertEqual(shorthand_to_bracket('{<CCO>}'), '{[<]CCO[>]}')

    def test_already_bracket(self):
        """Already bracket notation should pass through unchanged."""
        s = '{[$]CC[$]}'
        self.assertEqual(shorthand_to_bracket(s), s)

    def test_already_bracket_angle(self):
        s = '{[<]CCO[>]}'
        self.assertEqual(shorthand_to_bracket(s), s)

    def test_mixed_content(self):
        """Descriptors outside stochastic objects should not be converted."""
        self.assertEqual(shorthand_to_bracket('CC'), 'CC')

    def test_comma_separated(self):
        result = shorthand_to_bracket('{<CC>,$CC$}')
        self.assertEqual(result, '{[<]CC[>],[$]CC[$]}')

    def test_no_stochastic(self):
        """Plain SMILES should pass through unchanged."""
        self.assertEqual(shorthand_to_bracket('CCO'), 'CCO')

    def test_empty_string(self):
        self.assertEqual(shorthand_to_bracket(''), '')

    def test_nested_brackets(self):
        """SMILES brackets like [Si] should not be modified."""
        result = shorthand_to_bracket('{<O[Si](C)(C)>}')
        self.assertEqual(result, '{[<]O[Si](C)(C)[>]}')


class TestValidation(unittest.TestCase):
    """BigSMILES 校验测试 / BigSMILES validation tests."""

    def test_validate_all_pass(self):
        """All 304 entries should pass BigSMILES validation."""
        failures = validate_all()
        self.assertEqual(len(failures), 0,
                         f"Validation failures: {failures[:5]}")

    def test_validate_returns_list(self):
        failures = validate_all()
        self.assertIsInstance(failures, list)

    def test_validate_sample(self):
        """Spot-check a few well-known BigSMILES strings."""
        from src.bigsmiles.checker import check_bigsmiles
        samples = [
            '{[$]CC[$]}',           # PE
            '{[<]CCO[>]}',          # PEO
            '{[$]CC(c1ccccc1)[$]}', # PS
        ]
        for s in samples:
            with self.subTest(bigsmiles=s):
                self.assertTrue(check_bigsmiles(s))


if __name__ == "__main__":
    unittest.main()
