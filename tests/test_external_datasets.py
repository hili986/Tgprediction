"""
External Dataset Loader 测试
Tests for external_datasets module (v2 — canonical SMILES dedup)

覆盖: SMILES 处理, 质量过滤, 各 loader, canonical 去重, 冲突解决,
       load_all_external, load_copolymer_data, build_extended_dataset
"""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.external_datasets import (
    _normalize_psmiles,
    _to_star_format,
    _canonical_smiles,
    _is_polymer_smiles,
    _has_predicted_tg,
    _read_csv,
    _canonical_dedup,
    _load_polymetrix,
    _load_neurips_opp,
    _load_openpoly,
    _load_conjugated_polymer,
    _load_pilania_pha,
    HOMOPOLYMER_LOADERS,
    COPOLYMER_LOADERS,
    LOADERS,
    DEFAULT_SOURCES,
    load_all_external,
    load_copolymer_data,
    build_extended_dataset,
    DATA_DIR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, header, rows, encoding="utf-8-sig"):
    """Write a small CSV file for testing."""
    with open(path, "w", newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


EXPECTED_DICT_KEYS = {"smiles", "tg_k", "source", "reliability",
                      "polymer_class", "name"}


# ---------------------------------------------------------------------------
# SMILES processing functions
# ---------------------------------------------------------------------------

class TestNormalizePsmiles(unittest.TestCase):
    """_normalize_psmiles() 测试 — bare * → [*]"""

    def test_bare_star_to_bracket(self):
        """Bare * should be converted to [*]."""
        self.assertEqual(_normalize_psmiles("*CC*"), "[*]CC[*]")

    def test_bracket_star_unchanged(self):
        """[*] should remain [*]."""
        self.assertEqual(_normalize_psmiles("[*]CC[*]"), "[*]CC[*]")

    def test_strip_whitespace(self):
        self.assertEqual(_normalize_psmiles("  *CC*  "), "[*]CC[*]")

    def test_mixed_stars(self):
        """Mix of bare * and [*] should normalize to all [*]."""
        self.assertEqual(_normalize_psmiles("[*]CC*"), "[*]CC[*]")

    def test_empty_string(self):
        self.assertEqual(_normalize_psmiles(""), "")

    def test_no_stars(self):
        self.assertEqual(_normalize_psmiles("CCO"), "CCO")

    def test_element_brackets_preserved(self):
        """[Si], [N] etc. should not be affected."""
        result = _normalize_psmiles("*O[Si](C)(C)*")
        self.assertIn("[Si]", result)
        self.assertIn("[*]", result)


class TestToStarFormat(unittest.TestCase):
    """_to_star_format() 测试 — [*] → bare *"""

    def test_bracket_to_bare(self):
        self.assertEqual(_to_star_format("[*]CC[*]"), "*CC*")

    def test_no_brackets(self):
        self.assertEqual(_to_star_format("*CC*"), "*CC*")

    def test_element_brackets_preserved(self):
        self.assertIn("[Si]", _to_star_format("[*]O[Si](C)(C)[*]"))


class TestCanonicalSmiles(unittest.TestCase):
    """_canonical_smiles() 测试"""

    def test_returns_string(self):
        result = _canonical_smiles("*CC*")
        self.assertIsInstance(result, str)

    def test_same_molecule_same_canon(self):
        """Different SMILES representations → same canonical form."""
        c1 = _canonical_smiles("*CC*")
        c2 = _canonical_smiles("[*]CC[*]")
        self.assertEqual(c1, c2)

    def test_invalid_smiles_returns_none(self):
        result = _canonical_smiles("NOT_A_SMILES((((")
        self.assertIsNone(result)


class TestIsPolymerSmiles(unittest.TestCase):
    """_is_polymer_smiles() 测试"""

    def test_with_star(self):
        self.assertTrue(_is_polymer_smiles("*CC*"))

    def test_with_bracket_star(self):
        self.assertTrue(_is_polymer_smiles("[*]CC[*]"))

    def test_no_star(self):
        self.assertFalse(_is_polymer_smiles("CCCCCC"))

    def test_empty(self):
        self.assertFalse(_is_polymer_smiles(""))


class TestHasPredictedTg(unittest.TestCase):
    """_has_predicted_tg() 测试 — >2 decimal places = likely predicted"""

    def test_integer(self):
        self.assertFalse(_has_predicted_tg("123"))

    def test_one_decimal(self):
        self.assertFalse(_has_predicted_tg("123.4"))

    def test_two_decimals(self):
        self.assertFalse(_has_predicted_tg("123.45"))

    def test_three_decimals(self):
        self.assertTrue(_has_predicted_tg("123.456"))

    def test_many_decimals(self):
        self.assertTrue(_has_predicted_tg("123.456789"))

    def test_trailing_zeros(self):
        """123.400 has 1 significant decimal → not predicted."""
        self.assertFalse(_has_predicted_tg("123.400"))

    def test_invalid(self):
        self.assertFalse(_has_predicted_tg("not_a_number"))


# ---------------------------------------------------------------------------
# _read_csv
# ---------------------------------------------------------------------------

class TestReadCsv(unittest.TestCase):
    """_read_csv() 测试"""

    def test_reads_utf8_sig(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False,
            encoding="utf-8-sig", newline=""
        ) as f:
            path = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["col_a", "col_b"])
            writer.writerow(["x", "1"])
            writer.writerow(["y", "2"])
        try:
            rows = _read_csv(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["col_a"], "x")
            self.assertEqual(rows[1]["col_b"], "2")
        finally:
            os.unlink(path)

    def test_empty_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False,
            encoding="utf-8-sig", newline=""
        ) as f:
            path = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["col_a", "col_b"])
        try:
            rows = _read_csv(path)
            self.assertEqual(rows, [])
        finally:
            os.unlink(path)

    def test_returns_list_of_dicts(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False,
            encoding="utf-8-sig", newline=""
        ) as f:
            path = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(["name", "value"])
            writer.writerow(["alpha", "100"])
        try:
            rows = _read_csv(path)
            self.assertIsInstance(rows, list)
            self.assertIn("name", rows[0])
        finally:
            os.unlink(path)

    def test_latin1_fallback(self):
        """Should fall back to latin-1 on UTF-8 decode errors."""
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".csv", delete=False,
        ) as f:
            path = Path(f.name)
            f.write(b"name,value\n")
            f.write(b"test\xa5,100\n")  # \xa5 is invalid UTF-8
        try:
            rows = _read_csv(path)
            self.assertEqual(len(rows), 1)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Canonical deduplication and conflict resolution
# ---------------------------------------------------------------------------

class TestCanonicalDedup(unittest.TestCase):
    """_canonical_dedup() 测试"""

    def _entry(self, smiles, tg_k, source="test"):
        return {
            "smiles": smiles, "tg_k": tg_k, "source": source,
            "reliability": "", "polymer_class": "", "name": "",
        }

    def test_no_duplicates(self):
        data = [self._entry("*CC*", 300), self._entry("*CCC*", 400)]
        result = _canonical_dedup(data, verbose=False)
        self.assertEqual(len(result), 2)

    def test_removes_duplicates(self):
        data = [
            self._entry("*CC*", 300, "a"),
            self._entry("[*]CC[*]", 310, "b"),  # same molecule
        ]
        result = _canonical_dedup(data, resolve_conflicts="first",
                                  verbose=False)
        self.assertEqual(len(result), 1)

    def test_median_conflict_resolution(self):
        data = [
            self._entry("*CC*", 300, "a"),
            self._entry("[*]CC[*]", 400, "b"),
        ]
        result = _canonical_dedup(data, resolve_conflicts="median",
                                  verbose=False)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["tg_k"], 350.0)

    def test_exclude_openpoly_median(self):
        data = [
            self._entry("*CC*", 300, "polymetrix"),
            self._entry("[*]CC[*]", 600, "openpoly"),  # outlier
            self._entry("*CC*", 320, "neurips_opp"),
        ]
        result = _canonical_dedup(data,
                                  resolve_conflicts="exclude_openpoly_median",
                                  verbose=False)
        self.assertEqual(len(result), 1)
        # Median of [300, 320] = 310, NOT [300, 600, 320]
        self.assertAlmostEqual(result[0]["tg_k"], 310.0)

    def test_merged_source_names(self):
        data = [
            self._entry("*CC*", 300, "a"),
            self._entry("[*]CC[*]", 300, "b"),
        ]
        result = _canonical_dedup(data, verbose=False)
        self.assertIn("+", result[0]["source"])
        self.assertIn("a", result[0]["source"])
        self.assertIn("b", result[0]["source"])


# ---------------------------------------------------------------------------
# Individual Loaders — file-not-found path
# ---------------------------------------------------------------------------

class TestLoadersFileNotFound(unittest.TestCase):
    """All loaders should return [] when their CSV file is missing."""

    def test_polymetrix_missing(self):
        with patch("src.data.external_datasets.DATA_DIR",
                   Path(tempfile.mkdtemp())):
            self.assertEqual(_load_polymetrix(), [])

    def test_neurips_opp_missing(self):
        with patch("src.data.external_datasets.DATA_DIR",
                   Path(tempfile.mkdtemp())):
            self.assertEqual(_load_neurips_opp(), [])

    def test_openpoly_missing(self):
        with patch("src.data.external_datasets.DATA_DIR",
                   Path(tempfile.mkdtemp())):
            self.assertEqual(_load_openpoly(), [])

    def test_conjugated_missing(self):
        with patch("src.data.external_datasets.DATA_DIR",
                   Path(tempfile.mkdtemp())):
            self.assertEqual(_load_conjugated_polymer(), [])

    def test_pilania_pha_missing(self):
        with patch("src.data.external_datasets.DATA_DIR",
                   Path(tempfile.mkdtemp())):
            self.assertEqual(_load_pilania_pha(), [])


# ---------------------------------------------------------------------------
# Individual Loaders — with data
# ---------------------------------------------------------------------------

class TestLoadPolymetrix(unittest.TestCase):
    """_load_polymetrix() 测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv_path = self.tmpdir / "polymetrix_tg.csv"

    def tearDown(self):
        if self.csv_path.exists():
            os.unlink(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_valid_data(self):
        _write_csv(self.csv_path,
                   ["PSMILES", "Tg_K", "reliability", "polymer_class",
                    "polymer_name"],
                   [["[*]CC[*]", "373.0", "green", "Vinyl", "PE"]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_polymetrix()
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0].keys()), EXPECTED_DICT_KEYS)
        self.assertEqual(result[0]["smiles"], "*CC*")
        self.assertAlmostEqual(result[0]["tg_k"], 373.0)
        self.assertEqual(result[0]["source"], "polymetrix")
        self.assertEqual(result[0]["reliability"], "green")
        self.assertEqual(result[0]["name"], "PE")

    def test_skips_empty_smiles(self):
        _write_csv(self.csv_path,
                   ["PSMILES", "Tg_K", "reliability", "polymer_class",
                    "polymer_name"],
                   [["", "373.0", "", "", ""]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_polymetrix()
        self.assertEqual(result, [])

    def test_skips_invalid_tg(self):
        _write_csv(self.csv_path,
                   ["PSMILES", "Tg_K", "reliability", "polymer_class",
                    "polymer_name"],
                   [["*CC*", "not_a_number", "", "", ""]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_polymetrix()
        self.assertEqual(result, [])


class TestLoadNeuripsOpp(unittest.TestCase):
    """_load_neurips_opp() 测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv_path = self.tmpdir / "neurips_opp_tg.csv"

    def tearDown(self):
        if self.csv_path.exists():
            os.unlink(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_celsius_to_kelvin(self):
        _write_csv(self.csv_path,
                   ["SMILES", "Tg"],
                   [["*CC*", "100.0"]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_neurips_opp(quality_filter=False)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["tg_k"], 373.15)

    def test_quality_filter_removes_non_polymer(self):
        _write_csv(self.csv_path,
                   ["SMILES", "Tg"],
                   [["*CC*", "100.0"],     # polymer — keep
                    ["CCCCCC", "50.0"]])    # non-polymer — filter
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_neurips_opp(quality_filter=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["smiles"], "*CC*")

    def test_quality_filter_removes_predicted(self):
        _write_csv(self.csv_path,
                   ["SMILES", "Tg"],
                   [["*CC*", "100.0"],         # experimental — keep
                    ["*CCC*", "123.456789"]])   # predicted — filter
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_neurips_opp(quality_filter=True)
        self.assertEqual(len(result), 1)

    def test_quality_filter_disabled(self):
        _write_csv(self.csv_path,
                   ["SMILES", "Tg"],
                   [["CCCCCC", "50.0"]])    # non-polymer
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_neurips_opp(quality_filter=False)
        self.assertEqual(len(result), 1)


class TestLoadOpenpoly(unittest.TestCase):
    """_load_openpoly() 测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv_path = self.tmpdir / "openpoly_properties.csv"

    def tearDown(self):
        if self.csv_path.exists():
            os.unlink(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_valid_data_with_low_reliability(self):
        _write_csv(self.csv_path,
                   ["PSMILES", "Tg (K)", "Name"],
                   [["[*]CCO[*]", "350.0", "PEO"]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_openpoly()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "PEO")
        self.assertEqual(result[0]["source"], "openpoly")
        self.assertEqual(result[0]["reliability"], "low")

    def test_skips_whitespace_only_tg(self):
        _write_csv(self.csv_path,
                   ["PSMILES", "Tg (K)", "Name"],
                   [["*CC*", "  ", "test"]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_openpoly()
        self.assertEqual(result, [])


class TestLoadConjugatedPolymer(unittest.TestCase):
    """_load_conjugated_polymer() 测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv_path = self.tmpdir / "conjugated_polymer_tg.csv"

    def tearDown(self):
        if self.csv_path.exists():
            os.unlink(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_valid_data(self):
        _write_csv(self.csv_path,
                   ["psmiles", "tg_c", "tg_k", "source"],
                   [["C1=C(SC(=C1)[*])[*]", "215.0", "488.15",
                     "conjugated_polymer_32"]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_conjugated_polymer()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["tg_k"], 488.15)
        self.assertEqual(result[0]["source"], "conjugated_32")
        self.assertEqual(result[0]["polymer_class"], "Conjugated")


class TestLoadPilaniaPha(unittest.TestCase):
    """_load_pilania_pha() 测试"""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.csv_path = self.tmpdir / "pilania_pha_tg.csv"

    def tearDown(self):
        if self.csv_path.exists():
            os.unlink(self.csv_path)
        os.rmdir(self.tmpdir)

    def test_copolymer_entry(self):
        _write_csv(self.csv_path,
                   ["smiles_1", "smiles_2", "monomer_ratio", "type",
                    "tg_k", "tg_c", "source"],
                   [["*OC(C)CC(*)=O", "*OC(CC)CC(*)=O", "50.0", "R",
                     "260.0", "-13.15", "Pilania_PHA_2019"]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_pilania_pha()
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["is_copolymer"])
        self.assertEqual(result[0]["source"], "pilania_pha")
        self.assertAlmostEqual(result[0]["monomer_ratio"], 50.0)

    def test_homopolymer_entry(self):
        """monomer_ratio=100 with smiles_2='C' → homopolymer."""
        _write_csv(self.csv_path,
                   ["smiles_1", "smiles_2", "monomer_ratio", "type",
                    "tg_k", "tg_c", "source"],
                   [["*OC(C)CC(*)=O", "C", "100.0", "R",
                     "276.15", "3.0", "Pilania_PHA_2019"]])
        with patch("src.data.external_datasets.DATA_DIR", self.tmpdir):
            result = _load_pilania_pha()
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["is_copolymer"])


# ---------------------------------------------------------------------------
# Source registries
# ---------------------------------------------------------------------------

class TestSourceRegistries(unittest.TestCase):
    """Source registry 测试"""

    def test_homopolymer_sources(self):
        expected = {"polymetrix", "neurips_opp", "conjugated_32", "openpoly"}
        self.assertEqual(set(HOMOPOLYMER_LOADERS.keys()), expected)

    def test_copolymer_sources(self):
        expected = {"pilania_pha", "kuenneth_copolymer"}
        self.assertEqual(set(COPOLYMER_LOADERS.keys()), expected)

    def test_default_sources_exclude_openpoly(self):
        self.assertNotIn("openpoly", DEFAULT_SOURCES)
        self.assertIn("polymetrix", DEFAULT_SOURCES)
        self.assertIn("neurips_opp", DEFAULT_SOURCES)

    def test_loaders_backward_compat(self):
        """LOADERS dict should contain both homopolymer and copolymer."""
        self.assertIn("polymetrix", LOADERS)
        self.assertIn("pilania_pha", LOADERS)

    def test_all_values_callable(self):
        for name, loader in LOADERS.items():
            with self.subTest(source=name):
                self.assertTrue(callable(loader))


# ---------------------------------------------------------------------------
# load_all_external
# ---------------------------------------------------------------------------

class TestLoadAllExternal(unittest.TestCase):
    """load_all_external() 测试"""

    def _entry(self, smiles, tg_k, source="test"):
        return {
            "smiles": smiles, "tg_k": tg_k, "source": source,
            "reliability": "", "polymer_class": "", "name": "",
        }

    def test_filters_by_tg_range(self):
        fake_data = [
            self._entry("*A*", 50.0),   # below min
            self._entry("*B*", 500.0),   # in range
            self._entry("*C*", 950.0),   # above max
        ]
        with patch.dict(HOMOPOLYMER_LOADERS,
                        {"fake": lambda: fake_data}, clear=True):
            result = load_all_external(
                sources=["fake"], min_tg_k=100, max_tg_k=900,
                deduplicate=False, verbose=False
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["smiles"], "*B*")

    def test_tg_boundary_inclusive(self):
        fake_data = [
            self._entry("*A*", 100.0),
            self._entry("*B*", 900.0),
        ]
        with patch.dict(HOMOPOLYMER_LOADERS,
                        {"fake": lambda: fake_data}, clear=True):
            result = load_all_external(
                sources=["fake"], min_tg_k=100, max_tg_k=900,
                deduplicate=False, verbose=False
            )
        self.assertEqual(len(result), 2)

    def test_canonical_dedup(self):
        """Same molecule in different SMILES forms → deduplicated."""
        fake_data = [
            self._entry("*CC*", 300.0, "a"),
            self._entry("[*]CC[*]", 320.0, "b"),  # same canonical SMILES
        ]
        with patch.dict(HOMOPOLYMER_LOADERS,
                        {"fake": lambda: fake_data}, clear=True):
            result = load_all_external(
                sources=["fake"], deduplicate=True, verbose=False
            )
        self.assertEqual(len(result), 1)

    def test_deduplication_disabled(self):
        fake_data = [
            self._entry("*CC*", 300.0),
            self._entry("*CC*", 350.0),
        ]
        with patch.dict(HOMOPOLYMER_LOADERS,
                        {"fake": lambda: fake_data}, clear=True):
            result = load_all_external(
                sources=["fake"], deduplicate=False, verbose=False
            )
        self.assertEqual(len(result), 2)

    def test_unknown_source_skipped(self):
        with patch.dict(HOMOPOLYMER_LOADERS, {}, clear=True):
            result = load_all_external(
                sources=["nonexistent"], verbose=False
            )
        self.assertEqual(result, [])

    def test_empty_result(self):
        with patch.dict(HOMOPOLYMER_LOADERS,
                        {"empty": lambda: []}, clear=True):
            result = load_all_external(
                sources=["empty"], verbose=False
            )
        self.assertEqual(result, [])

    def test_dict_keys_present(self):
        fake_data = [self._entry("*CC*", 400.0)]
        with patch.dict(HOMOPOLYMER_LOADERS,
                        {"fake": lambda: fake_data}, clear=True):
            result = load_all_external(
                sources=["fake"], verbose=False, deduplicate=False
            )
        self.assertEqual(set(result[0].keys()), EXPECTED_DICT_KEYS)

    def test_multiple_sources_merged(self):
        def loader_x():
            return [self._entry("*X*", 300.0, "x")]

        def loader_y():
            return [self._entry("*Y*", 400.0, "y")]

        with patch.dict(HOMOPOLYMER_LOADERS,
                        {"x": loader_x, "y": loader_y}, clear=True):
            result = load_all_external(
                sources=["x", "y"], deduplicate=False, verbose=False
            )
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# load_copolymer_data
# ---------------------------------------------------------------------------

class TestLoadCopolymerData(unittest.TestCase):
    """load_copolymer_data() 测试"""

    def test_loads_available_sources(self):
        fake = [{
            "smiles_1": "*OC(C)CC(*)=O",
            "smiles_2": "*OC(CC)CC(*)=O",
            "monomer_ratio": 50.0,
            "type": "R",
            "tg_k": 260.0,
            "source": "fake",
            "is_copolymer": True,
        }]
        with patch.dict(COPOLYMER_LOADERS,
                        {"fake": lambda: fake}, clear=True):
            result = load_copolymer_data(sources=["fake"], verbose=False)
        self.assertEqual(len(result), 1)

    def test_filters_by_tg_range(self):
        fake = [
            {"smiles_1": "*A*", "smiles_2": "*B*", "monomer_ratio": 50.0,
             "type": "R", "tg_k": 50.0, "source": "f", "is_copolymer": True},
            {"smiles_1": "*C*", "smiles_2": "*D*", "monomer_ratio": 50.0,
             "type": "R", "tg_k": 300.0, "source": "f", "is_copolymer": True},
        ]
        with patch.dict(COPOLYMER_LOADERS,
                        {"fake": lambda: fake}, clear=True):
            result = load_copolymer_data(
                sources=["fake"], min_tg_k=100, verbose=False)
        self.assertEqual(len(result), 1)

    def test_empty_when_no_data(self):
        with patch.dict(COPOLYMER_LOADERS,
                        {"empty": lambda: []}, clear=True):
            result = load_copolymer_data(sources=["empty"], verbose=False)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# build_extended_dataset — smoke test with mocks
# ---------------------------------------------------------------------------

class TestBuildExtendedDataset(unittest.TestCase):
    """build_extended_dataset() 烟雾测试 (mocked dependencies)."""

    @patch("src.data.external_datasets.load_all_external")
    def test_smoke_returns_correct_shape(self, mock_load_ext):
        mock_load_ext.return_value = [
            {"smiles": "*AA*", "tg_k": 400.0, "source": "mock",
             "reliability": "", "polymer_class": "", "name": "TestA"},
        ]
        mock_compute = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        mock_feat_names = MagicMock(return_value=["f1", "f2", "f3"])

        with patch.dict("sys.modules", {
            "src.features.feature_pipeline": MagicMock(
                compute_features=mock_compute,
                get_feature_names=mock_feat_names,
            ),
            "src.data.bicerano_tg_dataset": MagicMock(
                BICERANO_DATA=[
                    ("PE", "*CC*", "{[$]CC[$]}", 195),
                ],
            ),
        }):
            X, y, names, feat_names = build_extended_dataset(
                layer="L1", verbose=False, include_bicerano=True
            )

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(feat_names), 3)
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(len(y), len(names))

    @patch("src.data.external_datasets.load_all_external")
    def test_smoke_no_bicerano(self, mock_load_ext):
        mock_load_ext.return_value = [
            {"smiles": "*XX*", "tg_k": 350.0, "source": "mock",
             "reliability": "", "polymer_class": "", "name": "X"},
        ]
        mock_compute = MagicMock(return_value=np.array([1.0, 2.0]))
        mock_feat_names = MagicMock(return_value=["f1", "f2"])

        with patch.dict("sys.modules", {
            "src.features.feature_pipeline": MagicMock(
                compute_features=mock_compute,
                get_feature_names=mock_feat_names,
            ),
        }):
            X, y, names, feat_names = build_extended_dataset(
                layer="L0", verbose=False, include_bicerano=False
            )

        self.assertIsInstance(X, np.ndarray)
        self.assertEqual(len(y), len(names))


# ---------------------------------------------------------------------------
# DATA_DIR constant
# ---------------------------------------------------------------------------

class TestDataDir(unittest.TestCase):
    """DATA_DIR 路径测试"""

    def test_data_dir_is_path(self):
        self.assertIsInstance(DATA_DIR, Path)

    def test_data_dir_ends_with_external(self):
        self.assertEqual(DATA_DIR.name, "external")
        self.assertEqual(DATA_DIR.parent.name, "data")


if __name__ == "__main__":
    unittest.main()
