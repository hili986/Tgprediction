"""
BigSMILES 解析器与生成器 API — 抽象语法树访问与序列化
BigSMILES Parser & Generator API — AST Access and Serialization

基于 bigsmiles_checker.py 的 Tokenizer 和 Parser，提供：
Built on bigsmiles_checker.py's Tokenizer and Parser, provides:
  1. parse(s) → AST: 解析 BigSMILES 字符串
  2. generate(ast) → str: 从 AST 重建 BigSMILES 字符串
  3. get_repeat_units(ast): 提取所有重复单元
  4. get_bonding_descriptors(ast): 提取所有键连接描述符
  5. get_topology(ast): 分析拓扑结构

公共 API / Public API:
    BigSMILESParser — 统一外观类 / Unified facade class
"""

from typing import List, Dict, Any

from src.bigsmiles.checker import (
    Tokenizer, TokenizerError,
    Parser, ParserError,
    BigSMILESString, SMILESSegment, StochasticObject,
    RepeatUnit, EndGroup, BondingDescriptor,
    ValidationError, Validator,
)


# ---------------------------------------------------------------------------
# Generator: AST → BigSMILES string
# ---------------------------------------------------------------------------

class Generator:
    """将 BigSMILES AST 转化为字符串。/ Generate BigSMILES string from AST."""

    def generate(self, ast: BigSMILESString) -> str:
        """从 AST 生成 BigSMILES 字符串。/ Generate BigSMILES string from AST."""
        parts = []
        for segment in ast.segments:
            if isinstance(segment, SMILESSegment):
                parts.append(segment.raw_text)
            elif isinstance(segment, StochasticObject):
                parts.append(self._generate_stochastic(segment))
        return "".join(parts)

    def _generate_stochastic(self, stoch: StochasticObject) -> str:
        """Generate string for a StochasticObject."""
        parts = ["{"]

        if stoch.left_term is not None:
            parts.append(stoch.left_term.value)

        for i, ru in enumerate(stoch.repeat_units):
            if i > 0:
                parts.append(",")
            parts.append(self._generate_unit(ru))

        if stoch.end_groups:
            parts.append(";")
            for i, eg in enumerate(stoch.end_groups):
                if i > 0:
                    parts.append(",")
                parts.append(self._generate_unit(eg))

        if stoch.right_term is not None:
            parts.append(stoch.right_term.value)

        parts.append("}")
        return "".join(parts)

    def _generate_unit(self, unit) -> str:
        """Generate string for a RepeatUnit or EndGroup.

        Interleaves tokens and nested stochastic objects by position
        to reconstruct the original ordering.
        """
        items = []
        for tok in unit.tokens:
            items.append((tok.position, tok.value))
        for nested in unit.nested_objects:
            items.append((nested.open_pos, self._generate_stochastic(nested)))
        items.sort(key=lambda x: x[0])
        return "".join(value for _, value in items)


# ---------------------------------------------------------------------------
# Extraction helpers / 提取辅助函数
# ---------------------------------------------------------------------------

def _collect_repeat_units(stoch: StochasticObject, results: list,
                          stoch_index: int, depth: int = 0):
    """Recursively collect repeat units from a StochasticObject."""
    for j, ru in enumerate(stoch.repeat_units):
        smiles_text = _unit_smiles(ru)
        results.append({
            "raw_text": ru.raw_text,
            "smiles": smiles_text,
            "descriptors": [
                {"type": bd.descriptor_type, "id": bd.numeric_id, "raw": bd.raw}
                for bd in ru.descriptors
            ],
            "stoch_index": stoch_index,
            "unit_index": j,
            "depth": depth,
        })
        for nested in ru.nested_objects:
            _collect_repeat_units(nested, results, stoch_index, depth + 1)


def _collect_descriptors(stoch: StochasticObject, results: list):
    """Recursively collect bonding descriptors."""
    for ru in stoch.repeat_units:
        for bd in ru.descriptors:
            results.append({
                "bond_type": bd.bond_type,
                "descriptor_type": bd.descriptor_type,
                "numeric_id": bd.numeric_id,
                "raw": bd.raw,
                "position": bd.position,
            })
        for nested in ru.nested_objects:
            _collect_descriptors(nested, results)
    for eg in stoch.end_groups:
        for bd in eg.descriptors:
            results.append({
                "bond_type": bd.bond_type,
                "descriptor_type": bd.descriptor_type,
                "numeric_id": bd.numeric_id,
                "raw": bd.raw,
                "position": bd.position,
            })


def _unit_smiles(unit) -> str:
    """Convert a RepeatUnit/EndGroup to SMILES by replacing descriptors with *."""
    parts = []
    for tok in unit.tokens:
        if tok.type.name in ("BOND_DESC", "TERM_DESC_EMPTY"):
            parts.append("*")
        else:
            parts.append(tok.value)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public extraction API / 公共提取 API
# ---------------------------------------------------------------------------

def get_repeat_units(ast: BigSMILESString) -> List[Dict[str, Any]]:
    """提取所有重复单元及元数据。/ Extract all repeat units with metadata.

    Returns list of dicts with keys:
        raw_text, smiles, descriptors, stoch_index, unit_index, depth
    """
    results = []
    for i, seg in enumerate(ast.segments):
        if isinstance(seg, StochasticObject):
            _collect_repeat_units(seg, results, stoch_index=i)
    return results


def get_bonding_descriptors(ast: BigSMILESString) -> List[Dict[str, Any]]:
    """提取所有键连接描述符。/ Extract all bonding descriptors.

    Returns list of dicts with keys:
        bond_type, descriptor_type, numeric_id, raw, position
    """
    results = []
    for seg in ast.segments:
        if isinstance(seg, StochasticObject):
            _collect_descriptors(seg, results)
    return results


def get_topology(ast: BigSMILESString) -> Dict[str, Any]:
    """分析 BigSMILES 结构的拓扑类型。/ Analyze the topology of a BigSMILES structure.

    Topology categories:
        small_molecule, linear_homopolymer, random_copolymer,
        block_copolymer, graft_copolymer, branched, network
    """
    stoch_objects = [s for s in ast.segments if isinstance(s, StochasticObject)]
    smiles_segs = [s for s in ast.segments if isinstance(s, SMILESSegment)]

    num_stoch = len(stoch_objects)
    total_ru = 0
    has_nesting = False
    max_descriptors = 0
    bonding_types = set()

    for stoch in stoch_objects:
        for ru in stoch.repeat_units:
            total_ru += 1
            max_descriptors = max(max_descriptors, len(ru.descriptors))
            for bd in ru.descriptors:
                bonding_types.add(bd.descriptor_type)
            if ru.nested_objects:
                has_nesting = True

    # Determine topology
    if num_stoch == 0:
        topology = "small_molecule"
    elif num_stoch == 1:
        stoch = stoch_objects[0]
        if has_nesting:
            topology = "graft_copolymer"
        elif max_descriptors > 2:
            topology = "branched"
        elif len(stoch.repeat_units) == 1:
            topology = "linear_homopolymer"
        else:
            topology = "random_copolymer"
    else:
        if has_nesting:
            topology = "graft_block_copolymer"
        else:
            topology = "block_copolymer"

    return {
        "topology": topology,
        "num_stochastic_objects": num_stoch,
        "num_repeat_units": total_ru,
        "has_nesting": has_nesting,
        "has_crosslinks": max_descriptors > 2,
        "bonding_types": sorted(bonding_types),
        "has_end_groups": any(st.end_groups for st in stoch_objects),
        "has_smiles_segments": len(smiles_segs) > 0,
    }


# ---------------------------------------------------------------------------
# Facade class / 外观类
# ---------------------------------------------------------------------------

class BigSMILESParser:
    """统一外观类 / Unified facade for BigSMILES parsing, generation, extraction.

    Usage:
        parser = BigSMILESParser()
        ast = parser.parse("{[$]CC[$]}")
        s = parser.generate(ast)           # "{[$]CC[$]}"
        units = parser.get_repeat_units(ast)
        topo = parser.get_topology(ast)
    """

    def __init__(self):
        self._generator = Generator()

    def parse(self, s: str) -> BigSMILESString:
        """解析 BigSMILES 字符串为 AST。/ Parse BigSMILES string into AST."""
        tokenizer = Tokenizer(s)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def generate(self, ast: BigSMILESString) -> str:
        """从 AST 生成 BigSMILES 字符串。/ Generate BigSMILES string from AST."""
        return self._generator.generate(ast)

    def validate(self, s: str) -> List[ValidationError]:
        """校验 BigSMILES 字符串。/ Validate a BigSMILES string.

        Returns empty list if valid, list of ValidationError otherwise.
        """
        try:
            tokenizer = Tokenizer(s)
            tokens = tokenizer.tokenize()
        except TokenizerError as e:
            return [ValidationError(e.message_en, e.message_cn, e.position)]
        try:
            parser = Parser(tokens)
            ast = parser.parse()
        except ParserError as e:
            return [ValidationError(e.message_en, e.message_cn, e.position)]
        validator = Validator(s, ast)
        return validator.validate()

    def round_trip(self, s: str) -> str:
        """Parse → generate round trip. Result should equal input."""
        return self.generate(self.parse(s))

    def get_repeat_units(self, ast_or_str) -> List[Dict[str, Any]]:
        """提取重复单元。/ Extract repeat units.

        Args:
            ast_or_str: BigSMILESString AST or a raw string to parse first.
        """
        ast = self._ensure_ast(ast_or_str)
        return get_repeat_units(ast)

    def get_bonding_descriptors(self, ast_or_str) -> List[Dict[str, Any]]:
        """提取键连接描述符。/ Extract bonding descriptors."""
        ast = self._ensure_ast(ast_or_str)
        return get_bonding_descriptors(ast)

    def get_topology(self, ast_or_str) -> Dict[str, Any]:
        """分析拓扑结构。/ Analyze topology."""
        ast = self._ensure_ast(ast_or_str)
        return get_topology(ast)

    def _ensure_ast(self, ast_or_str):
        if isinstance(ast_or_str, str):
            return self.parse(ast_or_str)
        return ast_or_str


# ---------------------------------------------------------------------------
# CLI entry / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python bigsmiles_parser.py <bigsmiles_string> [--topology] [--units] [--descriptors]")
        sys.exit(1)

    input_str = sys.argv[1]
    flags = set(sys.argv[2:])
    bp = BigSMILESParser()

    try:
        ast = bp.parse(input_str)
    except (TokenizerError, ParserError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    regenerated = bp.generate(ast)
    print(f"Input:       {input_str}")
    print(f"Round-trip:  {regenerated}")
    print(f"Match:       {input_str == regenerated}")

    if "--topology" in flags or not flags:
        topo = bp.get_topology(ast)
        print(f"\nTopology:    {json.dumps(topo, indent=2)}")

    if "--units" in flags:
        units = bp.get_repeat_units(ast)
        print(f"\nRepeat units ({len(units)}):")
        for u in units:
            print(f"  [{u['stoch_index']}.{u['unit_index']}] {u['raw_text']}  SMILES: {u['smiles']}")

    if "--descriptors" in flags:
        descs = bp.get_bonding_descriptors(ast)
        print(f"\nBonding descriptors ({len(descs)}):")
        for d in descs:
            print(f"  {d['raw']}  type={d['descriptor_type']}  id={d['numeric_id']}")
