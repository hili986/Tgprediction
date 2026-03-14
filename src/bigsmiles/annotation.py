"""
BigSMILES Property Annotation Extension
BigSMILES 属性注释扩展

Extends BigSMILES strings with property annotations using pipe-delimited syntax:
    {[$]CC[$]}|Tg=373K;Mn=50000;source=Bicerano2018|

Syntax:  <BigSMILES> | key=value; key2=value2 |
    - Keys: alphanumeric + underscore (case-insensitive for lookup)
    - Values: numbers (with optional units), strings, ranges (e.g., 350-400K)
    - Semicolons separate key-value pairs; trailing semicolons tolerated

Provides:
    1. AnnotatedBigSMILES dataclass: parsed representation
    2. parse_annotation(string): extract annotation from BigSMILES string
    3. add_annotation(bigsmiles, **props): attach properties to BigSMILES
    4. validate_annotation(string): check syntax correctness
    5. Property schema: known polymer property keys with units/types

Public API / 公共 API:
    parse_annotation(string) -> AnnotatedBigSMILES
    add_annotation(bigsmiles, **props) -> str
    remove_annotation(string) -> str
    validate_annotation(string) -> (bool, list[str])
    merge_annotations(string, **props) -> str
    PROPERTY_SCHEMA — known property definitions
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# 1. Property schema / 属性 schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropertyDef:
    """Definition of a known polymer property.
    已知聚合物属性的定义。
    """
    key: str
    description: str
    unit: str = ""
    value_type: str = "float"  # float, int, str, range
    aliases: Tuple[str, ...] = ()


# Known polymer properties (extensible)
_PROPERTY_DEFS: List[PropertyDef] = [
    PropertyDef("Tg", "Glass transition temperature / 玻璃化转变温度", "K", "float", ("tg", "glass_transition")),
    PropertyDef("Tm", "Melting temperature / 熔点", "K", "float", ("tm", "melting_point")),
    PropertyDef("Td", "Decomposition temperature / 分解温度", "K", "float", ("td",)),
    PropertyDef("Mn", "Number-average molecular weight / 数均分子量", "g/mol", "float", ("mn",)),
    PropertyDef("Mw", "Weight-average molecular weight / 重均分子量", "g/mol", "float", ("mw",)),
    PropertyDef("PDI", "Polydispersity index / 分散度", "", "float", ("pdi", "dispersity")),
    PropertyDef("density", "Density / 密度", "g/cm3", "float", ("rho",)),
    PropertyDef("CTE", "Coefficient of thermal expansion / 热膨胀系数", "1/K", "float", ("cte",)),
    PropertyDef("modulus", "Young's modulus / 杨氏模量", "GPa", "float", ("E", "youngs_modulus")),
    PropertyDef("tensile_strength", "Tensile strength / 抗拉强度", "MPa", "float"),
    PropertyDef("elongation", "Elongation at break / 断裂伸长率", "%", "float"),
    PropertyDef("source", "Data source / 数据来源", "", "str"),
    PropertyDef("method", "Measurement method / 测量方法", "", "str"),
    PropertyDef("doi", "DOI reference / 文献 DOI", "", "str"),
    PropertyDef("name", "Polymer name / 聚合物名称", "", "str"),
]

PROPERTY_SCHEMA: Dict[str, PropertyDef] = {}
for _pdef in _PROPERTY_DEFS:
    PROPERTY_SCHEMA[_pdef.key.lower()] = _pdef
    for alias in _pdef.aliases:
        PROPERTY_SCHEMA[alias.lower()] = _pdef


# ---------------------------------------------------------------------------
# 2. Data model / 数据模型
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedBigSMILES:
    """Parsed BigSMILES string with property annotations.
    带属性注释的 BigSMILES 解析结果。
    """
    bigsmiles: str
    properties: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

    def get_float(self, key: str) -> Optional[float]:
        """Get property value as float, stripping unit suffixes.
        获取浮点数属性值，自动去除单位后缀。
        """
        val = self.properties.get(key.lower())
        if val is None:
            # Try schema aliases
            pdef = PROPERTY_SCHEMA.get(key.lower())
            if pdef:
                val = self.properties.get(pdef.key.lower())
            if val is None:
                return None
        # Extract leading number (int or float, optional sign)
        num_match = re.match(r'^[+-]?(?:\d+\.?\d*|\.\d+)', val.strip())
        if not num_match:
            return None
        try:
            return float(num_match.group())
        except ValueError:
            return None

    def get_str(self, key: str) -> Optional[str]:
        """Get property value as string.
        获取字符串属性值。
        """
        val = self.properties.get(key.lower())
        if val is None:
            pdef = PROPERTY_SCHEMA.get(key.lower())
            if pdef:
                val = self.properties.get(pdef.key.lower())
        return val

    def to_string(self) -> str:
        """Reconstruct annotated BigSMILES string.
        重建带注释的 BigSMILES 字符串。
        """
        if not self.properties:
            return self.bigsmiles
        pairs = [f"{k}={v}" for k, v in self.properties.items()]
        return f"{self.bigsmiles}|{';'.join(pairs)}|"

    def has_property(self, key: str) -> bool:
        """Check if a property exists."""
        lower = key.lower()
        if lower in self.properties:
            return True
        pdef = PROPERTY_SCHEMA.get(lower)
        if pdef and pdef.key.lower() in self.properties:
            return True
        return False


# ---------------------------------------------------------------------------
# 3. Parsing / 解析
# ---------------------------------------------------------------------------

# Regex for annotation block: |key=val;key2=val2|
_ANNOTATION_PATTERN = re.compile(
    r'^(.*?)\|([^|]+)\|\s*$'
)

# Regex for a single key=value pair
_KV_PATTERN = re.compile(
    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$'
)


def parse_annotation(string: str) -> AnnotatedBigSMILES:
    """Parse an annotated BigSMILES string.
    解析带注释的 BigSMILES 字符串。

    Examples:
        >>> parse_annotation("{[$]CC[$]}|Tg=373K;Mn=50000|")
        AnnotatedBigSMILES(bigsmiles='{[$]CC[$]}', properties={'tg': '373K', 'mn': '50000'})

        >>> parse_annotation("{[$]CC[$]}")
        AnnotatedBigSMILES(bigsmiles='{[$]CC[$]}', properties={})
    """
    string = string.strip()
    match = _ANNOTATION_PATTERN.match(string)

    if not match:
        return AnnotatedBigSMILES(bigsmiles=string, raw=string)

    bigsmiles_part = match.group(1).strip()
    annotation_part = match.group(2).strip()

    properties: Dict[str, str] = {}
    pairs = annotation_part.split(';')
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        kv_match = _KV_PATTERN.match(pair)
        if kv_match:
            key = kv_match.group(1).lower()
            value = kv_match.group(2)
            properties[key] = value

    return AnnotatedBigSMILES(
        bigsmiles=bigsmiles_part,
        properties=properties,
        raw=string,
    )


# ---------------------------------------------------------------------------
# 4. Construction / 构建
# ---------------------------------------------------------------------------

def add_annotation(bigsmiles: str, **props: Any) -> str:
    """Add property annotations to a BigSMILES string.
    向 BigSMILES 字符串添加属性注释。

    Args:
        bigsmiles: Base BigSMILES string.
        **props: Property key=value pairs.

    Returns:
        Annotated BigSMILES string.

    Example:
        >>> add_annotation("{[$]CC[$]}", Tg="373K", source="Bicerano")
        '{[$]CC[$]}|tg=373K;source=Bicerano|'
    """
    # Strip any existing annotation
    existing = parse_annotation(bigsmiles)
    base = existing.bigsmiles

    merged = dict(existing.properties)
    for k, v in props.items():
        merged[k.lower()] = str(v)

    if not merged:
        return base

    pairs = [f"{k}={v}" for k, v in merged.items()]
    return f"{base}|{';'.join(pairs)}|"


def remove_annotation(string: str) -> str:
    """Remove annotation from a BigSMILES string.
    从 BigSMILES 字符串中移除注释。
    """
    parsed = parse_annotation(string)
    return parsed.bigsmiles


def merge_annotations(string: str, **props: Any) -> str:
    """Merge new properties into existing annotation.
    将新属性合并到已有注释中。
    """
    return add_annotation(string, **props)


# ---------------------------------------------------------------------------
# 5. Validation / 验证
# ---------------------------------------------------------------------------

def validate_annotation(string: str) -> Tuple[bool, List[str]]:
    """Validate an annotated BigSMILES string.
    验证带注释的 BigSMILES 字符串。

    Returns:
        (is_valid, list_of_warnings_or_errors)
    """
    errors: List[str] = []
    string = string.strip()

    # Check for annotation
    match = _ANNOTATION_PATTERN.match(string)
    if not match:
        # No annotation — valid BigSMILES without annotation
        return True, []

    bigsmiles_part = match.group(1).strip()
    annotation_part = match.group(2).strip()

    if not bigsmiles_part:
        errors.append("Empty BigSMILES part before annotation / 注释前 BigSMILES 为空")

    # Parse key-value pairs
    pairs = annotation_part.split(';')
    seen_keys = set()
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue

        kv_match = _KV_PATTERN.match(pair)
        if not kv_match:
            errors.append(
                f"Invalid key=value pair: '{pair}' / 无效的键值对: '{pair}'"
            )
            continue

        key = kv_match.group(1).lower()
        value = kv_match.group(2)

        # Duplicate check
        if key in seen_keys:
            errors.append(f"Duplicate key: '{key}' / 重复的键: '{key}'")
        seen_keys.add(key)

        # Empty value check
        if not value.strip():
            errors.append(f"Empty value for key '{key}' / 键 '{key}' 的值为空")

        # Schema validation (warning, not error)
        pdef = PROPERTY_SCHEMA.get(key)
        if pdef and pdef.value_type == "float":
            num_match = re.match(r'^[+-]?(?:\d+\.?\d*|\.\d+)', value.strip())
            if not num_match:
                errors.append(
                    f"Non-numeric value for '{key}': '{value}' "
                    f"(expected {pdef.value_type}) / "
                    f"键 '{key}' 的值 '{value}' 不是数字"
                )
            else:
                try:
                    float(num_match.group())
                except ValueError:
                    errors.append(
                        f"Non-numeric value for '{key}': '{value}' "
                        f"(expected {pdef.value_type}) / "
                        f"键 '{key}' 的值 '{value}' 不是数字"
                    )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# 6. Batch operations / 批量操作
# ---------------------------------------------------------------------------

def annotate_dataset(
    data: List[Tuple[str, str, str, float]],
    property_key: str = "Tg",
    property_unit: str = "K",
) -> List[str]:
    """Add annotations to a list of (name, smiles, bigsmiles, value) entries.
    批量为数据集条目添加注释。

    Args:
        data: List of (name, smiles, bigsmiles, value) tuples.
        property_key: Property name to annotate.
        property_unit: Unit suffix.

    Returns:
        List of annotated BigSMILES strings.
    """
    results = []
    for name, smiles, bigsmiles, value in data:
        annotated = add_annotation(
            bigsmiles,
            **{
                property_key: f"{value}{property_unit}",
                "name": name,
            }
        )
        results.append(annotated)
    return results


def parse_dataset_annotations(
    annotated_list: List[str],
) -> List[AnnotatedBigSMILES]:
    """Parse a list of annotated BigSMILES strings.
    解析注释后的 BigSMILES 字符串列表。
    """
    return [parse_annotation(s) for s in annotated_list]


# ---------------------------------------------------------------------------
# CLI entry / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv:
        print("Usage: python bigsmiles_annotation.py [options]")
        print("  --demo       Run demo with Bicerano data")
        print("  --validate   Validate a string: --validate '<string>'")
        print("  --schema     Print known property schema")
        sys.exit(0)

    if "--schema" in sys.argv:
        print("Known property schema / 已知属性 schema:")
        seen = set()
        for pdef in _PROPERTY_DEFS:
            if pdef.key not in seen:
                aliases = ", ".join(pdef.aliases) if pdef.aliases else "-"
                print(f"  {pdef.key:20s} [{pdef.value_type:5s}] "
                      f"{pdef.unit:8s} {pdef.description}  (aliases: {aliases})")
                seen.add(pdef.key)
        sys.exit(0)

    if "--validate" in sys.argv:
        idx = sys.argv.index("--validate")
        if idx + 1 < len(sys.argv):
            test_str = sys.argv[idx + 1]
            valid, msgs = validate_annotation(test_str)
            print(f"Valid: {valid}")
            for msg in msgs:
                print(f"  - {msg}")
        sys.exit(0)

    # Demo
    print("=== BigSMILES Annotation Demo ===\n")

    # Example 1: Create annotation
    bs = "{[$]CC[$]}"
    annotated = add_annotation(bs, Tg="373K", Mn="50000", source="Bicerano2018")
    print(f"Original:  {bs}")
    print(f"Annotated: {annotated}")

    # Example 2: Parse annotation
    parsed = parse_annotation(annotated)
    print(f"\nParsed BigSMILES: {parsed.bigsmiles}")
    print(f"Properties: {parsed.properties}")
    print(f"Tg (float): {parsed.get_float('Tg')}")
    print(f"Source:     {parsed.get_str('source')}")

    # Example 3: Validate
    valid, errors = validate_annotation(annotated)
    print(f"\nValidation: {'PASS' if valid else 'FAIL'}")
    for e in errors:
        print(f"  - {e}")

    # Example 4: Merge
    merged = merge_annotations(annotated, Tm="500K", method="DSC")
    print(f"\nMerged: {merged}")

    # Example 5: Round-trip
    rt_parsed = parse_annotation(merged)
    rt_string = rt_parsed.to_string()
    print(f"Round-trip: {rt_string}")

    # Example 6: Batch with Bicerano
    print("\n=== Batch Demo (first 5 from Bicerano) ===")
    try:
        from src.data.bicerano_tg_dataset import BICERANO_DATA
        batch = annotate_dataset(BICERANO_DATA[:5])
        for s in batch:
            print(f"  {s}")
    except ImportError:
        print("  (Bicerano dataset not available)")
