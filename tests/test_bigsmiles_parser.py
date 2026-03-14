"""
BigSMILES 解析器与生成器测试
Tests for BigSMILES Parser & Generator API

覆盖: 解析、生成、往返一致性、提取 API、拓扑分析
Covers: parsing, generation, round-trip, extraction API, topology analysis
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigsmiles.parser import BigSMILESParser, Generator, get_repeat_units, get_bonding_descriptors, get_topology
from src.bigsmiles.checker import (
    Tokenizer, Parser, BigSMILESString, SMILESSegment, StochasticObject,
    TokenizerError, ParserError,
)


class TestParserBasic(unittest.TestCase):
    """基本解析测试 / Basic parsing tests."""

    def setUp(self):
        self.bp = BigSMILESParser()

    def test_parse_simple_homopolymer(self):
        ast = self.bp.parse("{[$]CC[$]}")
        self.assertIsInstance(ast, BigSMILESString)
        self.assertEqual(len(ast.segments), 1)
        self.assertIsInstance(ast.segments[0], StochasticObject)

    def test_parse_ab_type(self):
        ast = self.bp.parse("{[>]CCO[<]}")
        stoch = ast.segments[0]
        self.assertEqual(len(stoch.repeat_units), 1)

    def test_parse_copolymer(self):
        ast = self.bp.parse("{[$]CC[$],[$]CC(CC)[$]}")
        stoch = ast.segments[0]
        self.assertEqual(len(stoch.repeat_units), 2)

    def test_parse_block_copolymer(self):
        ast = self.bp.parse("{[$]CC(c1ccccc1)[$]}{[$]CC(C)(C(=O)OC)[$]}")
        self.assertEqual(len(ast.segments), 2)
        for seg in ast.segments:
            self.assertIsInstance(seg, StochasticObject)

    def test_parse_with_end_groups(self):
        ast = self.bp.parse("{[$]CC(c1ccccc1)[$];[$]CC(C)(C#N)CC(C)(C#N),[$]Cl}")
        stoch = ast.segments[0]
        self.assertEqual(len(stoch.end_groups), 2)

    def test_parse_with_smiles_flanks(self):
        ast = self.bp.parse("CC{[$]CC[$]}CC")
        self.assertEqual(len(ast.segments), 3)
        self.assertIsInstance(ast.segments[0], SMILESSegment)
        self.assertIsInstance(ast.segments[1], StochasticObject)
        self.assertIsInstance(ast.segments[2], SMILESSegment)

    def test_parse_nested_stochastic(self):
        ast = self.bp.parse("{[$]CC(c1ccccc1)[$],[$]CC(c1ccc(O{[>]CCO[<]})cc1)[$]}")
        stoch = ast.segments[0]
        self.assertTrue(any(ru.nested_objects for ru in stoch.repeat_units))

    def test_parse_error_unmatched(self):
        with self.assertRaises((TokenizerError, ParserError)):
            self.bp.parse("{[$]CC[$]")

    def test_parse_empty_stochastic(self):
        """Empty stochastic object — parser accepts it (no repeat units)."""
        ast = self.bp.parse("{}")
        self.assertIsInstance(ast, BigSMILESString)


class TestGenerator(unittest.TestCase):
    """生成器测试 / Generator tests."""

    def setUp(self):
        self.bp = BigSMILESParser()

    def test_generate_simple(self):
        ast = self.bp.parse("{[$]CC[$]}")
        result = self.bp.generate(ast)
        self.assertEqual(result, "{[$]CC[$]}")

    def test_generate_ab_type(self):
        ast = self.bp.parse("{[>]CCO[<]}")
        result = self.bp.generate(ast)
        self.assertEqual(result, "{[>]CCO[<]}")

    def test_generate_copolymer(self):
        ast = self.bp.parse("{[$]CC[$],[$]CC(CC)[$]}")
        result = self.bp.generate(ast)
        self.assertEqual(result, "{[$]CC[$],[$]CC(CC)[$]}")

    def test_generate_with_end_groups(self):
        s = "{[$]CC(c1ccccc1)[$];[$]CC(C)(C#N)CC(C)(C#N),[$]Cl}"
        ast = self.bp.parse(s)
        result = self.bp.generate(ast)
        self.assertEqual(result, s)

    def test_generate_block_copolymer(self):
        s = "{[$]CC(c1ccccc1)[$]}{[$]CC(C)(C(=O)OC)[$]}"
        ast = self.bp.parse(s)
        result = self.bp.generate(ast)
        self.assertEqual(result, s)

    def test_generate_with_flanks(self):
        s = "CC(C)(C#N){[$]CC(c1ccccc1)[$]}CC(C)(C#N)"
        ast = self.bp.parse(s)
        result = self.bp.generate(ast)
        self.assertEqual(result, s)


class TestRoundTrip(unittest.TestCase):
    """往返一致性测试 / Round-trip consistency tests.

    Parse → generate should reproduce the original string exactly.
    Then re-parsing should yield a structurally equivalent AST.
    """

    def setUp(self):
        self.bp = BigSMILESParser()

    # Test strings covering various BigSMILES features
    ROUND_TRIP_STRINGS = [
        "{[$]CC[$]}",                              # PE
        "{[>]CCO[<]}",                              # PEO
        "{[$]CC(c1ccccc1)[$]}",                     # PS
        "{[$]CC(C)[$]}",                            # PP
        "{[$]CC(Cl)[$]}",                           # PVC
        "{[$]CC(C)(C(=O)OC)[$]}",                   # PMMA
        "{[$]C(F)(F)C(F)(F)[$]}",                   # PTFE
        "{[$]CC(C#N)[$]}",                          # PAN
        "{[$]CC[$],[$]CC(CC)[$]}",                  # random copoly
        "{[>]C(=O)CCCCC(=O)NCCCCCCN[<]}",          # Nylon-66
        "{[>]OC(C)C(=O)[<]}",                      # PLA
        "{[>]O[Si](C)(C)[<]}",                      # PDMS
        # Block copolymer
        "{[$]CC(c1ccccc1)[$]}{[$]CC(C)(C(=O)OC)[$]}",
        # Triblock
        "{[>]CCO[<]}{[>]CC(C)O[<]}{[>]CCO[<]}",
        # With deterministic end groups
        "CC(C)(C#N){[$]CC(c1ccccc1)[$]}CC(C)(C#N)",
        # With stochastic end groups
        "{[$]CC(c1ccccc1)[$];[$]CC(C)(C#N)CC(C)(C#N),[$]Cl}",
        # Nested stochastic object (graft copolymer)
        "{[$]CC(c1ccccc1)[$],[$]CC(c1ccc(O{[>]CCO[<]})cc1)[$]}",
        # Branched (LDPE)
        "{[$]CC[$],[$]CC([$])[$]}",
    ]

    def test_round_trip_all(self):
        for s in self.ROUND_TRIP_STRINGS:
            with self.subTest(bigsmiles=s):
                result = self.bp.round_trip(s)
                self.assertEqual(result, s, f"Round-trip failed for: {s}")

    def test_double_round_trip(self):
        """Parse → generate → parse → generate should be stable."""
        for s in self.ROUND_TRIP_STRINGS:
            with self.subTest(bigsmiles=s):
                first = self.bp.round_trip(s)
                second = self.bp.round_trip(first)
                self.assertEqual(first, second)


class TestRepeatUnitExtraction(unittest.TestCase):
    """重复单元提取测试 / Repeat unit extraction tests."""

    def setUp(self):
        self.bp = BigSMILESParser()

    def test_single_repeat_unit(self):
        units = self.bp.get_repeat_units("{[$]CC[$]}")
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["smiles"], "*CC*")
        self.assertEqual(units[0]["depth"], 0)

    def test_copolymer_units(self):
        units = self.bp.get_repeat_units("{[$]CC[$],[$]CC(CC)[$]}")
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["unit_index"], 0)
        self.assertEqual(units[1]["unit_index"], 1)

    def test_block_copolymer_units(self):
        units = self.bp.get_repeat_units(
            "{[$]CC(c1ccccc1)[$]}{[$]CC(C)(C(=O)OC)[$]}"
        )
        self.assertEqual(len(units), 2)
        # Different stochastic object indices
        self.assertNotEqual(units[0]["stoch_index"], units[1]["stoch_index"])

    def test_nested_units(self):
        units = self.bp.get_repeat_units(
            "{[$]CC(c1ccccc1)[$],[$]CC(c1ccc(O{[>]CCO[<]})cc1)[$]}"
        )
        # 2 top-level repeat units + 1 nested
        self.assertEqual(len(units), 3)
        nested = [u for u in units if u["depth"] > 0]
        self.assertEqual(len(nested), 1)
        self.assertEqual(nested[0]["depth"], 1)

    def test_descriptor_info(self):
        units = self.bp.get_repeat_units("{[>]CCO[<]}")
        self.assertEqual(len(units), 1)
        descs = units[0]["descriptors"]
        self.assertEqual(len(descs), 2)
        types = {d["type"] for d in descs}
        self.assertEqual(types, {">", "<"})


class TestBondingDescriptorExtraction(unittest.TestCase):
    """键连接描述符提取测试 / Bonding descriptor extraction tests."""

    def setUp(self):
        self.bp = BigSMILESParser()

    def test_aa_descriptors(self):
        descs = self.bp.get_bonding_descriptors("{[$]CC[$]}")
        self.assertEqual(len(descs), 2)
        for d in descs:
            self.assertEqual(d["descriptor_type"], "$")

    def test_ab_descriptors(self):
        descs = self.bp.get_bonding_descriptors("{[>]CCO[<]}")
        types = [d["descriptor_type"] for d in descs]
        self.assertIn(">", types)
        self.assertIn("<", types)

    def test_numbered_descriptors(self):
        descs = self.bp.get_bonding_descriptors("{[>1]CCO[<1]}")
        for d in descs:
            self.assertEqual(d["numeric_id"], "1")

    def test_end_group_descriptors(self):
        descs = self.bp.get_bonding_descriptors(
            "{[$]CC(c1ccccc1)[$];[$]Cl}"
        )
        # 2 from repeat unit + 1 from end group
        self.assertEqual(len(descs), 3)


class TestTopologyAnalysis(unittest.TestCase):
    """拓扑分析测试 / Topology analysis tests."""

    def setUp(self):
        self.bp = BigSMILESParser()

    def test_small_molecule(self):
        topo = self.bp.get_topology("CCO")
        self.assertEqual(topo["topology"], "small_molecule")
        self.assertEqual(topo["num_stochastic_objects"], 0)

    def test_linear_homopolymer(self):
        topo = self.bp.get_topology("{[$]CC[$]}")
        self.assertEqual(topo["topology"], "linear_homopolymer")
        self.assertEqual(topo["num_repeat_units"], 1)
        self.assertFalse(topo["has_crosslinks"])

    def test_random_copolymer(self):
        topo = self.bp.get_topology("{[$]CC[$],[$]CC(CC)[$]}")
        self.assertEqual(topo["topology"], "random_copolymer")
        self.assertEqual(topo["num_repeat_units"], 2)

    def test_block_copolymer(self):
        topo = self.bp.get_topology(
            "{[$]CC(c1ccccc1)[$]}{[$]CC(C)(C(=O)OC)[$]}"
        )
        self.assertEqual(topo["topology"], "block_copolymer")
        self.assertEqual(topo["num_stochastic_objects"], 2)

    def test_triblock(self):
        topo = self.bp.get_topology(
            "{[>]CCO[<]}{[>]CC(C)O[<]}{[>]CCO[<]}"
        )
        self.assertEqual(topo["topology"], "block_copolymer")
        self.assertEqual(topo["num_stochastic_objects"], 3)

    def test_graft_copolymer(self):
        topo = self.bp.get_topology(
            "{[$]CC(c1ccccc1)[$],[$]CC(c1ccc(O{[>]CCO[<]})cc1)[$]}"
        )
        self.assertEqual(topo["topology"], "graft_copolymer")
        self.assertTrue(topo["has_nesting"])

    def test_branched(self):
        topo = self.bp.get_topology("{[$]CC[$],[$]CC([$])[$]}")
        self.assertEqual(topo["topology"], "branched")
        self.assertTrue(topo["has_crosslinks"])

    def test_with_end_groups(self):
        topo = self.bp.get_topology(
            "{[$]CC(c1ccccc1)[$];[$]Cl}"
        )
        self.assertTrue(topo["has_end_groups"])

    def test_with_smiles_flanks(self):
        topo = self.bp.get_topology("CC{[$]CC[$]}CC")
        self.assertTrue(topo["has_smiles_segments"])

    def test_bonding_types_aa(self):
        topo = self.bp.get_topology("{[$]CC[$]}")
        self.assertEqual(topo["bonding_types"], ["$"])

    def test_bonding_types_ab(self):
        topo = self.bp.get_topology("{[>]CCO[<]}")
        self.assertEqual(topo["bonding_types"], ["<", ">"])


class TestValidation(unittest.TestCase):
    """校验接口测试 / Validation interface tests."""

    def setUp(self):
        self.bp = BigSMILESParser()

    def test_valid_string(self):
        errors = self.bp.validate("{[$]CC[$]}")
        self.assertEqual(len(errors), 0)

    def test_invalid_tokenizer_error(self):
        errors = self.bp.validate("{[$]CC[$]")
        self.assertGreater(len(errors), 0)

    def test_invalid_parser_error(self):
        errors = self.bp.validate("}")
        self.assertGreater(len(errors), 0)


class TestExamplesRoundTrip(unittest.TestCase):
    """使用 bigsmiles_examples.py 中的示例进行往返测试。
    Round-trip test using examples from bigsmiles_examples.py.
    """

    def setUp(self):
        self.bp = BigSMILESParser()

    def test_all_examples_round_trip(self):
        try:
            from bigsmiles_examples import EXAMPLES
        except ImportError:
            self.skipTest("bigsmiles_examples.py not available")

        for ex in EXAMPLES:
            s = ex["bigsmiles"]
            with self.subTest(example_id=ex["id"], name=ex["name_en"]):
                try:
                    result = self.bp.round_trip(s)
                    self.assertEqual(result, s,
                                     f"Round-trip failed for {ex['id']} {ex['name_en']}")
                except (TokenizerError, ParserError) as e:
                    self.fail(f"Parse error for {ex['id']} {ex['name_en']}: {e}")


if __name__ == "__main__":
    unittest.main()
