"""
BigSMILES 测试套件
BigSMILES Test Suite

- 正向测试: 39 个示例库条目必须全部通过
- 负向测试: ~10 个非法输入必须被拒绝
- 边界测试: ~6 个边界情况

运行方式 / Run: python -m unittest test_bigsmiles -v
"""

import unittest
import json
import os
import sys

# 确保项目根目录在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigsmiles.checker import (
    check_bigsmiles, Tokenizer, Parser, Validator,
    TokenType, TokenizerError, ParserError
)
from src.bigsmiles.examples import EXAMPLES, to_json, get_examples


class TestPositive(unittest.TestCase):
    """正向测试 — 示例库全部 39 个条目必须通过语法检查。"""

    def _check(self, example):
        result = check_bigsmiles(example["bigsmiles"], verbose=False)
        self.assertTrue(
            result,
            f"Example {example['id']} ({example['name_en']}) failed: "
            f"{example['bigsmiles']}"
        )

    def test_1_1_polyethylene(self):
        self._check(EXAMPLES[0])

    def test_1_2_peg(self):
        self._check(EXAMPLES[1])

    def test_1_3_polystyrene(self):
        self._check(EXAMPLES[2])

    def test_1_4_polypropylene(self):
        self._check(EXAMPLES[3])

    def test_1_5_pvc(self):
        self._check(EXAMPLES[4])

    def test_1_6_pmma(self):
        self._check(EXAMPLES[5])

    def test_1_7_ptfe(self):
        self._check(EXAMPLES[6])

    def test_1_8_pan(self):
        self._check(EXAMPLES[7])

    def test_2_1_cis_14_polyisoprene(self):
        self._check(EXAMPLES[8])

    def test_2_2_trans_14_polyisoprene(self):
        self._check(EXAMPLES[9])

    def test_2_3_34_polyisoprene(self):
        self._check(EXAMPLES[10])

    def test_2_4_mixed_polyisoprene(self):
        self._check(EXAMPLES[11])

    def test_3_1_pe_co_pb(self):
        self._check(EXAMPLES[12])

    def test_3_2_ps_co_pmma(self):
        self._check(EXAMPLES[13])

    def test_3_3_sbr(self):
        self._check(EXAMPLES[14])

    def test_4_1_nylon66(self):
        self._check(EXAMPLES[15])

    def test_4_2_pet(self):
        self._check(EXAMPLES[16])

    def test_4_3_bpa_pc(self):
        self._check(EXAMPLES[17])

    def test_4_4_pla(self):
        self._check(EXAMPLES[18])

    def test_5_1_ps_b_pmma(self):
        self._check(EXAMPLES[19])

    def test_5_2_pluronic(self):
        self._check(EXAMPLES[20])

    def test_5_3_sbs(self):
        self._check(EXAMPLES[21])

    def test_6_1_pib_g_pmma(self):
        self._check(EXAMPLES[22])

    def test_6_2_ps_g_peg(self):
        self._check(EXAMPLES[23])

    def test_7_1_ldpe(self):
        self._check(EXAMPLES[24])

    def test_7_2_hyperbranched_polyester(self):
        self._check(EXAMPLES[25])

    def test_8_1_vulcanized_rubber(self):
        self._check(EXAMPLES[26])

    def test_8_2_epoxy_amine_network(self):
        self._check(EXAMPLES[27])

    def test_9_1_cyclic_ps(self):
        self._check(EXAMPLES[28])

    def test_9_2_cyclic_peg(self):
        self._check(EXAMPLES[29])

    def test_10_1_aibn_ps_deterministic(self):
        self._check(EXAMPLES[30])

    def test_10_2_ps_stochastic_endgroups(self):
        self._check(EXAMPLES[31])

    def test_11_1_polyurethane_nested(self):
        self._check(EXAMPLES[32])

    def test_12_1_nylon6(self):
        self._check(EXAMPLES[33])

    def test_12_2_pcl(self):
        self._check(EXAMPLES[34])

    def test_12_3_pai(self):
        self._check(EXAMPLES[35])

    def test_12_4_pdms(self):
        self._check(EXAMPLES[36])

    def test_13_1_ssdna_pdl1_aptamer(self):
        self._check(EXAMPLES[37])

    def test_13_2_ssrna(self):
        self._check(EXAMPLES[38])

    def test_all_39_examples_count(self):
        """确认示例库恰好包含 39 个条目。"""
        self.assertEqual(len(EXAMPLES), 39)


class TestNegative(unittest.TestCase):
    """负向测试 — 非法输入必须被拒绝。"""

    def test_unmatched_open_brace(self):
        """未闭合的花括号。"""
        self.assertFalse(check_bigsmiles("{[$]CC[$]", verbose=False))

    def test_unmatched_close_brace(self):
        """多余的闭合花括号。"""
        self.assertFalse(check_bigsmiles("{[$]CC[$]}}",  verbose=False))

    def test_unmatched_open_bracket(self):
        """未闭合的方括号。"""
        self.assertFalse(check_bigsmiles("{[$CC[$]}", verbose=False))

    def test_mixed_descriptors(self):
        """混用 $ 和 <> 描述符。"""
        self.assertFalse(check_bigsmiles("{[$]CC[>]}", verbose=False))

    def test_empty_stochastic_object(self):
        """空的随机对象（无重复单元）。"""
        self.assertFalse(check_bigsmiles("{}", verbose=False))

    def test_repeat_unit_too_few_descriptors(self):
        """重复单元只有一个描述符。"""
        self.assertFalse(check_bigsmiles("{[$]CC}", verbose=False))

    def test_endgroup_wrong_descriptor_count(self):
        """端基有两个描述符（应恰好 1 个）。"""
        self.assertFalse(check_bigsmiles("{[$]CC[$];[$]C[$]}", verbose=False))

    def test_unmatched_ab_id(self):
        """AB 型 ID 不配对。"""
        self.assertFalse(check_bigsmiles("{[>1]CC[<2]}", verbose=False))

    def test_invalid_smiles_fragment(self):
        """无效的 SMILES 片段。"""
        self.assertFalse(check_bigsmiles("{[$]XYZ[$]}", verbose=False))

    def test_unmatched_parentheses(self):
        """圆括号不匹配。"""
        self.assertFalse(check_bigsmiles("{[$]CC(C[$]}", verbose=False))


class TestBoundary(unittest.TestCase):
    """边界测试 — 嵌套、环形、立体化学、端基等边界情况。"""

    def test_nested_stochastic_object(self):
        """嵌套随机对象。"""
        self.assertTrue(check_bigsmiles(
            "{[>]C(=O)Nc1ccc(NC(=O){[>]CCO[<]})cc1[<]}",
            verbose=False
        ))

    def test_deeply_nested(self):
        """两层嵌套随机对象。"""
        self.assertTrue(check_bigsmiles(
            "{[$]CC(c1ccc(O{[>]CC(O{[>]CCO[<]})O[<]})cc1)[$]}",
            verbose=False
        ))

    def test_multiple_repeat_units(self):
        """多个重复单元（逗号分隔）。"""
        self.assertTrue(check_bigsmiles(
            "{[$]CC[$],[$]CC(CC)[$],[$]CC(C)[$]}",
            verbose=False
        ))

    def test_stereochemistry(self):
        """含立体化学的 BigSMILES。"""
        self.assertTrue(check_bigsmiles(
            r"{[$]C/C=C(\C)C[$]}",
            verbose=False
        ))

    def test_endgroups_with_separator(self):
        """包含端基分隔符的完整结构。"""
        self.assertTrue(check_bigsmiles(
            "{[$]CC(c1ccccc1)[$];[$]Cl}",
            verbose=False
        ))

    def test_block_copolymer_adjacent_objects(self):
        """相邻随机对象（嵌段共聚物）。"""
        self.assertTrue(check_bigsmiles(
            "{[$]CC[$]}{[$]CC(C)[$]}",
            verbose=False
        ))


class TestTokenizer(unittest.TestCase):
    """分词器单元测试。"""

    def test_simple_pe(self):
        tokens = Tokenizer("{[$]CC[$]}").tokenize()
        types = [t.type for t in tokens]
        self.assertEqual(types[0], TokenType.STOCH_OPEN)
        self.assertEqual(types[1], TokenType.BOND_DESC)
        self.assertEqual(types[2], TokenType.ATOM_ORGANIC)
        self.assertEqual(types[3], TokenType.ATOM_ORGANIC)
        self.assertEqual(types[4], TokenType.BOND_DESC)
        self.assertEqual(types[5], TokenType.STOCH_CLOSE)
        self.assertEqual(types[6], TokenType.EOF)

    def test_bond_descriptor_variants(self):
        """各种描述符格式。"""
        for desc in ["[$]", "[>]", "[<]", "[$1]", "[>2]", "[<3]", "[=$]"]:
            tokens = Tokenizer(desc).tokenize()
            self.assertEqual(tokens[0].type, TokenType.BOND_DESC,
                             f"Failed for {desc}")

    def test_bracket_atom(self):
        """方括号原子不被误判为描述符。"""
        tokens = Tokenizer("[NH]").tokenize()
        self.assertEqual(tokens[0].type, TokenType.ATOM_BRACKET)

    def test_empty_terminal(self):
        """空终端描述符 []。"""
        tokens = Tokenizer("[]").tokenize()
        self.assertEqual(tokens[0].type, TokenType.TERM_DESC_EMPTY)

    def test_comma_outside_stoch(self):
        """花括号外的逗号是不可识别字符。"""
        with self.assertRaises(TokenizerError):
            Tokenizer(",").tokenize()


class TestExamplesLibrary(unittest.TestCase):
    """示例库功能测试。"""

    def test_example_count(self):
        self.assertEqual(len(get_examples()), 39)

    def test_all_have_required_fields(self):
        required_fields = [
            "id", "category_cn", "category_en", "name_cn", "name_en",
            "mechanism_cn", "bonding_type", "bigsmiles", "smiles_repeat_unit",
            "structure_ascii", "explanation_cn", "explanation_en", "source"
        ]
        for ex in EXAMPLES:
            for field in required_fields:
                self.assertIn(field, ex, f"Example {ex.get('id', '?')} missing field: {field}")

    def test_13_categories(self):
        categories = set(e["category_cn"] for e in EXAMPLES)
        self.assertEqual(len(categories), 13)

    def test_to_json(self):
        output_path = os.path.join(os.path.dirname(__file__), "output", "test_output.json")
        to_json(output_path)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 39)
        # 清理
        os.remove(output_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
