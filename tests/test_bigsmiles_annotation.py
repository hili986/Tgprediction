"""
Unit tests for bigsmiles_annotation.py — BigSMILES Property Annotation Extension.
bigsmiles_annotation.py 单元测试 — BigSMILES 属性注释扩展。

Tests cover:
    1. Parsing annotated strings
    2. Creating annotations (add, merge, remove)
    3. Validation
    4. AnnotatedBigSMILES data model
    5. Batch operations
    6. Property schema
    7. Edge cases
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.bigsmiles.annotation import (
    parse_annotation, add_annotation, remove_annotation,
    merge_annotations, validate_annotation,
    annotate_dataset, parse_dataset_annotations,
    AnnotatedBigSMILES, PROPERTY_SCHEMA,
)


# ===========================================================================
# 1. Parsing Tests / 解析测试
# ===========================================================================

class TestParseAnnotation(unittest.TestCase):
    """Tests for parse_annotation()."""

    def test_parse_with_annotation(self):
        result = parse_annotation("{[$]CC[$]}|Tg=373K;Mn=50000|")
        self.assertEqual(result.bigsmiles, "{[$]CC[$]}")
        self.assertEqual(result.properties["tg"], "373K")
        self.assertEqual(result.properties["mn"], "50000")

    def test_parse_without_annotation(self):
        result = parse_annotation("{[$]CC[$]}")
        self.assertEqual(result.bigsmiles, "{[$]CC[$]}")
        self.assertEqual(result.properties, {})

    def test_parse_single_property(self):
        result = parse_annotation("{[$]CC[$]}|Tg=373K|")
        self.assertEqual(len(result.properties), 1)
        self.assertEqual(result.properties["tg"], "373K")

    def test_parse_trailing_semicolon(self):
        result = parse_annotation("{[$]CC[$]}|Tg=373K;|")
        self.assertEqual(result.properties["tg"], "373K")
        self.assertEqual(len(result.properties), 1)

    def test_parse_whitespace_tolerance(self):
        result = parse_annotation("  {[$]CC[$]}  | Tg = 373K ; Mn = 50000 |  ")
        self.assertEqual(result.bigsmiles, "{[$]CC[$]}")
        self.assertEqual(result.properties["tg"], "373K")
        self.assertEqual(result.properties["mn"], "50000")

    def test_parse_string_value(self):
        result = parse_annotation("{[$]CC[$]}|source=Bicerano2018|")
        self.assertEqual(result.properties["source"], "Bicerano2018")

    def test_parse_preserves_raw(self):
        raw = "{[$]CC[$]}|Tg=373K|"
        result = parse_annotation(raw)
        self.assertEqual(result.raw, raw)

    def test_parse_multiple_properties(self):
        result = parse_annotation(
            "{[$]CC[$]}|Tg=373K;Tm=500K;density=1.05g/cm3;method=DSC|"
        )
        self.assertEqual(len(result.properties), 4)
        self.assertEqual(result.properties["density"], "1.05g/cm3")
        self.assertEqual(result.properties["method"], "DSC")

    def test_parse_empty_string(self):
        result = parse_annotation("")
        self.assertEqual(result.bigsmiles, "")
        self.assertEqual(result.properties, {})


# ===========================================================================
# 2. Construction Tests / 构建测试
# ===========================================================================

class TestAddAnnotation(unittest.TestCase):
    """Tests for add_annotation()."""

    def test_add_to_clean_bigsmiles(self):
        result = add_annotation("{[$]CC[$]}", Tg="373K")
        self.assertIn("tg=373K", result)
        self.assertTrue(result.startswith("{[$]CC[$]}"))

    def test_add_multiple_properties(self):
        result = add_annotation("{[$]CC[$]}", Tg="373K", Mn="50000")
        parsed = parse_annotation(result)
        self.assertEqual(parsed.properties["tg"], "373K")
        self.assertEqual(parsed.properties["mn"], "50000")

    def test_add_no_properties(self):
        result = add_annotation("{[$]CC[$]}")
        self.assertEqual(result, "{[$]CC[$]}")

    def test_add_overwrites_existing(self):
        """Adding to already-annotated string merges properties."""
        annotated = "{[$]CC[$]}|Tg=373K|"
        result = add_annotation(annotated, Tg="400K")
        parsed = parse_annotation(result)
        self.assertEqual(parsed.properties["tg"], "400K")

    def test_add_merges_with_existing(self):
        annotated = "{[$]CC[$]}|Tg=373K|"
        result = add_annotation(annotated, Mn="50000")
        parsed = parse_annotation(result)
        self.assertEqual(parsed.properties["tg"], "373K")
        self.assertEqual(parsed.properties["mn"], "50000")


class TestRemoveAnnotation(unittest.TestCase):
    """Tests for remove_annotation()."""

    def test_remove_from_annotated(self):
        result = remove_annotation("{[$]CC[$]}|Tg=373K|")
        self.assertEqual(result, "{[$]CC[$]}")

    def test_remove_from_clean(self):
        result = remove_annotation("{[$]CC[$]}")
        self.assertEqual(result, "{[$]CC[$]}")


class TestMergeAnnotations(unittest.TestCase):
    """Tests for merge_annotations()."""

    def test_merge_adds_new_property(self):
        result = merge_annotations("{[$]CC[$]}|Tg=373K|", Tm="500K")
        parsed = parse_annotation(result)
        self.assertEqual(parsed.properties["tg"], "373K")
        self.assertEqual(parsed.properties["tm"], "500K")

    def test_merge_overwrites_existing(self):
        result = merge_annotations("{[$]CC[$]}|Tg=373K|", Tg="400K")
        parsed = parse_annotation(result)
        self.assertEqual(parsed.properties["tg"], "400K")


# ===========================================================================
# 3. Validation Tests / 验证测试
# ===========================================================================

class TestValidateAnnotation(unittest.TestCase):
    """Tests for validate_annotation()."""

    def test_valid_annotation(self):
        valid, errors = validate_annotation("{[$]CC[$]}|Tg=373K;Mn=50000|")
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_valid_without_annotation(self):
        valid, errors = validate_annotation("{[$]CC[$]}")
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    def test_invalid_pair_syntax(self):
        valid, errors = validate_annotation("{[$]CC[$]}|badpair|")
        self.assertFalse(valid)
        self.assertTrue(any("Invalid" in e or "无效" in e for e in errors))

    def test_duplicate_key(self):
        valid, errors = validate_annotation("{[$]CC[$]}|Tg=373K;Tg=400K|")
        self.assertFalse(valid)
        self.assertTrue(any("Duplicate" in e or "重复" in e for e in errors))

    def test_non_numeric_for_float_property(self):
        valid, errors = validate_annotation("{[$]CC[$]}|Tg=notanumber|")
        self.assertFalse(valid)
        self.assertTrue(any("numeric" in e.lower() or "数字" in e for e in errors))

    def test_valid_with_units(self):
        valid, errors = validate_annotation("{[$]CC[$]}|Tg=373K;density=1.05g/cm3|")
        self.assertTrue(valid)

    def test_empty_bigsmiles_part(self):
        valid, errors = validate_annotation("|Tg=373K|")
        self.assertFalse(valid)
        self.assertTrue(any("Empty" in e or "为空" in e for e in errors))


# ===========================================================================
# 4. Data Model Tests / 数据模型测试
# ===========================================================================

class TestAnnotatedBigSMILES(unittest.TestCase):
    """Tests for AnnotatedBigSMILES class."""

    def test_get_float(self):
        obj = AnnotatedBigSMILES(
            bigsmiles="{[$]CC[$]}",
            properties={"tg": "373K", "mn": "50000"},
        )
        self.assertAlmostEqual(obj.get_float("Tg"), 373.0)
        self.assertAlmostEqual(obj.get_float("Mn"), 50000.0)

    def test_get_float_without_unit(self):
        obj = AnnotatedBigSMILES(
            bigsmiles="{[$]CC[$]}",
            properties={"tg": "373"},
        )
        self.assertAlmostEqual(obj.get_float("Tg"), 373.0)

    def test_get_float_missing(self):
        obj = AnnotatedBigSMILES(bigsmiles="{[$]CC[$]}", properties={})
        self.assertIsNone(obj.get_float("Tg"))

    def test_get_float_non_numeric(self):
        obj = AnnotatedBigSMILES(
            bigsmiles="{[$]CC[$]}",
            properties={"source": "Bicerano"},
        )
        self.assertIsNone(obj.get_float("source"))

    def test_get_str(self):
        obj = AnnotatedBigSMILES(
            bigsmiles="{[$]CC[$]}",
            properties={"source": "Bicerano2018"},
        )
        self.assertEqual(obj.get_str("source"), "Bicerano2018")

    def test_get_str_missing(self):
        obj = AnnotatedBigSMILES(bigsmiles="{[$]CC[$]}", properties={})
        self.assertIsNone(obj.get_str("source"))

    def test_to_string(self):
        obj = AnnotatedBigSMILES(
            bigsmiles="{[$]CC[$]}",
            properties={"tg": "373K", "mn": "50000"},
        )
        result = obj.to_string()
        self.assertTrue(result.startswith("{[$]CC[$]}|"))
        self.assertTrue(result.endswith("|"))
        self.assertIn("tg=373K", result)
        self.assertIn("mn=50000", result)

    def test_to_string_no_props(self):
        obj = AnnotatedBigSMILES(bigsmiles="{[$]CC[$]}", properties={})
        self.assertEqual(obj.to_string(), "{[$]CC[$]}")

    def test_has_property(self):
        obj = AnnotatedBigSMILES(
            bigsmiles="{[$]CC[$]}",
            properties={"tg": "373K"},
        )
        self.assertTrue(obj.has_property("Tg"))
        self.assertTrue(obj.has_property("tg"))
        self.assertFalse(obj.has_property("Tm"))

    def test_round_trip(self):
        """Parse → to_string → parse should preserve properties."""
        original = "{[$]CC[$]}|tg=373K;mn=50000|"
        parsed = parse_annotation(original)
        rebuilt = parsed.to_string()
        reparsed = parse_annotation(rebuilt)
        self.assertEqual(parsed.bigsmiles, reparsed.bigsmiles)
        self.assertEqual(parsed.properties, reparsed.properties)


# ===========================================================================
# 5. Batch Operations Tests / 批量操作测试
# ===========================================================================

class TestBatchOperations(unittest.TestCase):
    """Tests for batch annotation operations."""

    def test_annotate_dataset(self):
        data = [
            ("PE", "*CC*", "{[$]CC[$]}", 250.0),
            ("PS", "*CC(c1ccccc1)*", "{[$]CC(c1ccccc1)[$]}", 373.0),
        ]
        results = annotate_dataset(data)
        self.assertEqual(len(results), 2)
        parsed0 = parse_annotation(results[0])
        self.assertEqual(parsed0.get_float("Tg"), 250.0)
        self.assertEqual(parsed0.get_str("name"), "PE")

    def test_parse_dataset_annotations(self):
        annotated = [
            "{[$]CC[$]}|tg=250.0K;name=PE|",
            "{[$]CC(c1ccccc1)[$]}|tg=373.0K;name=PS|",
        ]
        results = parse_dataset_annotations(annotated)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].get_float("Tg"), 250.0)
        self.assertEqual(results[1].get_str("name"), "PS")

    def test_annotate_custom_property(self):
        data = [("PE", "*CC*", "{[$]CC[$]}", 1.05)]
        results = annotate_dataset(data, property_key="density", property_unit="g/cm3")
        parsed = parse_annotation(results[0])
        self.assertAlmostEqual(parsed.get_float("density"), 1.05)


# ===========================================================================
# 6. Schema Tests / Schema 测试
# ===========================================================================

class TestPropertySchema(unittest.TestCase):
    """Tests for property schema."""

    def test_known_properties_exist(self):
        self.assertIn("tg", PROPERTY_SCHEMA)
        self.assertIn("tm", PROPERTY_SCHEMA)
        self.assertIn("mn", PROPERTY_SCHEMA)
        self.assertIn("source", PROPERTY_SCHEMA)

    def test_alias_resolution(self):
        """Aliases should map to the same PropertyDef."""
        tg_def = PROPERTY_SCHEMA["tg"]
        glass_def = PROPERTY_SCHEMA.get("glass_transition")
        self.assertIsNotNone(glass_def)
        self.assertEqual(tg_def.key, glass_def.key)

    def test_schema_has_description(self):
        for key, pdef in PROPERTY_SCHEMA.items():
            self.assertTrue(len(pdef.description) > 0,
                            f"Missing description for {key}")


# ===========================================================================
# 7. Edge Cases / 边界情况
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_pipe_in_bigsmiles_value(self):
        """BigSMILES shouldn't contain pipes — this tests graceful handling."""
        result = parse_annotation("CC|Tg=373K|")
        self.assertEqual(result.bigsmiles, "CC")
        self.assertEqual(result.properties["tg"], "373K")

    def test_numeric_value_no_unit(self):
        result = parse_annotation("{[$]CC[$]}|Tg=373|")
        self.assertAlmostEqual(result.get_float("Tg"), 373.0)

    def test_value_with_decimal(self):
        result = parse_annotation("{[$]CC[$]}|density=1.05|")
        self.assertAlmostEqual(result.get_float("density"), 1.05)

    def test_case_insensitive_keys(self):
        result = parse_annotation("{[$]CC[$]}|TG=373K;mn=50000|")
        self.assertIn("tg", result.properties)
        self.assertIn("mn", result.properties)

    def test_special_characters_in_value(self):
        """DOI contains special characters."""
        result = parse_annotation("{[$]CC[$]}|doi=10.1021/acs.jcim.123|")
        self.assertEqual(result.properties["doi"], "10.1021/acs.jcim.123")

    def test_underscore_in_key(self):
        result = parse_annotation("{[$]CC[$]}|tensile_strength=50MPa|")
        self.assertEqual(result.properties["tensile_strength"], "50MPa")


if __name__ == "__main__":
    unittest.main()
