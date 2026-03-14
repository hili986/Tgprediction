"""
BigSMILES 语法检查器 — 三阶段流水线架构
BigSMILES Syntax Checker — Three-stage Pipeline Architecture

Stage 1: 分词器 (Tokenizer) — 将输入字符串转化为 Token 流
Stage 2: 递归下降解析器 (Parser) — 将 Token 流转化为 AST
Stage 3: 语义校验器 (Validator) — 对 AST 执行 7 项语义检查

公共 API / Public API:
    check_bigsmiles(s: str, verbose=True) -> bool
"""

import re
import sys
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Stage 1: 分词器 / Tokenizer
# ---------------------------------------------------------------------------

class TokenType(Enum):
    """Token 类型枚举"""
    STOCH_OPEN = auto()       # {
    STOCH_CLOSE = auto()      # }
    BOND_DESC = auto()         # [$] [$1] [>] [<] [>1] [<2] etc.
    TERM_DESC_EMPTY = auto()   # [] — 空终端描述符（隐式键连接）
    REPEAT_SEP = auto()        # , (inside stochastic object, between repeat units)
    ENDGROUP_SEP = auto()      # ; (inside stochastic object, before end groups)
    ATOM_ORGANIC = auto()      # B C N O P S F Cl Br I (organic subset)
    ATOM_BRACKET = auto()      # [NH] [C@@H] [Si] etc.
    BOND = auto()              # - = # $ ~ / backslash
    BRANCH_OPEN = auto()       # (
    BRANCH_CLOSE = auto()      # )
    RING_DIGIT = auto()        # 1-9 or %nn
    DOT = auto()               # .
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    position: int  # 在原始字符串中的起始位置


class TokenizerError(Exception):
    def __init__(self, message_en, message_cn, position):
        self.message_en = message_en
        self.message_cn = message_cn
        self.position = position
        super().__init__(f"Position {position}: {message_en}")


class Tokenizer:
    """BigSMILES 分词器 / Tokenizer"""

    # 有机子集原子（SMILES标准）
    ORGANIC_ATOMS = {"B", "C", "N", "O", "P", "S", "F", "I"}
    ORGANIC_TWO_CHAR = {"Cl", "Br"}
    BOND_CHARS = {"-", "=", "#", "~", "/", "\\"}
    # 键连接描述符正则: [$] [$1] [>] [<] [>1] [<2] [=$ ] etc.
    BOND_DESC_RE = re.compile(r'^\[([=#/\\]?)([\$<>])(\d*)\]$')

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.stoch_depth = 0  # {} 嵌套深度
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """对整个输入进行分词，返回 Token 列表。"""
        self.tokens = []
        self.pos = 0
        self.stoch_depth = 0

        while self.pos < len(self.text):
            self._next_token()

        self.tokens.append(Token(TokenType.EOF, "", self.pos))
        return self.tokens

    def _peek(self, offset=0):
        idx = self.pos + offset
        if idx < len(self.text):
            return self.text[idx]
        return None

    def _next_token(self):
        ch = self.text[self.pos]

        # 花括号 — 随机对象边界
        if ch == '{':
            self.tokens.append(Token(TokenType.STOCH_OPEN, '{', self.pos))
            self.stoch_depth += 1
            self.pos += 1
            return

        if ch == '}':
            self.stoch_depth -= 1
            self.tokens.append(Token(TokenType.STOCH_CLOSE, '}', self.pos))
            self.pos += 1
            return

        # 方括号 — 需要区分三种情况
        if ch == '[':
            self._read_bracket()
            return

        # 逗号和分号 — 只在随机对象内有特殊意义
        if ch == ',' and self.stoch_depth > 0:
            self.tokens.append(Token(TokenType.REPEAT_SEP, ',', self.pos))
            self.pos += 1
            return

        if ch == ';' and self.stoch_depth > 0:
            self.tokens.append(Token(TokenType.ENDGROUP_SEP, ';', self.pos))
            self.pos += 1
            return

        # 分支括号
        if ch == '(':
            self.tokens.append(Token(TokenType.BRANCH_OPEN, '(', self.pos))
            self.pos += 1
            return

        if ch == ')':
            self.tokens.append(Token(TokenType.BRANCH_CLOSE, ')', self.pos))
            self.pos += 1
            return

        # 键符号
        if ch in self.BOND_CHARS:
            self.tokens.append(Token(TokenType.BOND, ch, self.pos))
            self.pos += 1
            return

        # 点（断开的片段）
        if ch == '.':
            self.tokens.append(Token(TokenType.DOT, '.', self.pos))
            self.pos += 1
            return

        # 环闭合数字
        if ch == '%':
            # %nn 格式
            start = self.pos
            self.pos += 1
            digits = ""
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                digits += self.text[self.pos]
                self.pos += 1
            if len(digits) < 2:
                raise TokenizerError(
                    f"Expected two digits after %",
                    f"% 后需要两个数字",
                    start
                )
            self.tokens.append(Token(TokenType.RING_DIGIT, '%' + digits, start))
            return

        if ch.isdigit():
            self.tokens.append(Token(TokenType.RING_DIGIT, ch, self.pos))
            self.pos += 1
            return

        # 有机原子（两字符优先）
        two_char = self.text[self.pos:self.pos+2]
        if two_char in self.ORGANIC_TWO_CHAR:
            self.tokens.append(Token(TokenType.ATOM_ORGANIC, two_char, self.pos))
            self.pos += 2
            return

        if ch in self.ORGANIC_ATOMS:
            # 小写字母紧跟 → 芳香原子 (c, n, o, s, etc.)
            self.tokens.append(Token(TokenType.ATOM_ORGANIC, ch, self.pos))
            self.pos += 1
            return

        # 小写字母芳香原子
        if ch in ('c', 'n', 'o', 's', 'p', 'b'):
            self.tokens.append(Token(TokenType.ATOM_ORGANIC, ch, self.pos))
            self.pos += 1
            return

        # 空白字符跳过
        if ch in (' ', '\t', '\n', '\r'):
            self.pos += 1
            return

        raise TokenizerError(
            f"Unexpected character: '{ch}'",
            f"意外字符: '{ch}'",
            self.pos
        )

    def _read_bracket(self):
        """读取方括号内容并分类。"""
        start = self.pos
        # 找到匹配的 ]
        end = self.text.find(']', self.pos + 1)
        if end == -1:
            raise TokenizerError(
                "Unmatched '[' — missing ']'",
                "方括号 '[' 未闭合 — 缺少 ']'",
                start
            )

        full = self.text[start:end+1]  # 包含 [ 和 ]
        inner = self.text[start+1:end]  # 括号内内容

        # 空终端描述符 []
        if inner == "":
            self.tokens.append(Token(TokenType.TERM_DESC_EMPTY, full, start))
            self.pos = end + 1
            return

        # 检查是否为键连接描述符
        if self.BOND_DESC_RE.match(full):
            self.tokens.append(Token(TokenType.BOND_DESC, full, start))
            self.pos = end + 1
            return

        # 否则为 SMILES 方括号原子
        self.tokens.append(Token(TokenType.ATOM_BRACKET, full, start))
        self.pos = end + 1


# ---------------------------------------------------------------------------
# Stage 2: 递归下降解析器 / Recursive Descent Parser (AST)
# ---------------------------------------------------------------------------

@dataclass
class BondingDescriptor:
    """键连接描述符节点"""
    bond_type: str            # "" or "=" or "#" or "/" or "\\"
    descriptor_type: str      # "$" or ">" or "<"
    numeric_id: str           # "" or "1" or "2" etc.
    raw: str                  # 原始文本 e.g. "[$]", "[>1]"
    position: int


@dataclass
class RepeatUnit:
    """重复单元节点"""
    tokens: List[Token]       # 包含的所有 token（不含分隔符）
    descriptors: List[BondingDescriptor]
    nested_objects: list       # 嵌套的 StochasticObject 列表
    raw_text: str = ""


@dataclass
class EndGroup:
    """端基节点"""
    tokens: List[Token]
    descriptors: List[BondingDescriptor]
    nested_objects: list
    raw_text: str = ""


@dataclass
class StochasticObject:
    """随机对象节点"""
    left_term: Optional[Token]     # 左终端描述符 token（BOND_DESC or TERM_DESC_EMPTY）
    repeat_units: List[RepeatUnit]
    end_groups: List[EndGroup]
    right_term: Optional[Token]    # 右终端描述符 token
    open_pos: int = 0
    close_pos: int = 0


@dataclass
class SMILESSegment:
    """SMILES 片段节点（非聚合物部分）"""
    tokens: List[Token]
    raw_text: str = ""


@dataclass
class BigSMILESString:
    """BigSMILES 根节点"""
    segments: list  # List of SMILESSegment | StochasticObject


class ParserError(Exception):
    def __init__(self, message_en, message_cn, position):
        self.message_en = message_en
        self.message_cn = message_cn
        self.position = position
        super().__init__(f"Position {position}: {message_en}")


class Parser:
    """BigSMILES 递归下降解析器"""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def _current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, "", -1)

    def _advance(self) -> Token:
        tok = self._current()
        self.pos += 1
        return tok

    def _expect(self, token_type: TokenType) -> Token:
        tok = self._current()
        if tok.type != token_type:
            raise ParserError(
                f"Expected {token_type.name}, got {tok.type.name} ('{tok.value}')",
                f"期望 {token_type.name}，得到 {tok.type.name} ('{tok.value}')",
                tok.position
            )
        return self._advance()

    def parse(self) -> BigSMILESString:
        """解析为 BigSMILESString AST。"""
        segments = []
        while self._current().type != TokenType.EOF:
            if self._current().type == TokenType.STOCH_OPEN:
                segments.append(self._parse_stochastic_object())
            elif self._current().type == TokenType.STOCH_CLOSE:
                # 多余的 } — 报错
                raise ParserError(
                    f"Unexpected '}}' at position {self._current().position}",
                    f"位置 {self._current().position} 处出现多余的 '}}'",
                    self._current().position
                )
            else:
                seg = self._parse_smiles_segment()
                if seg.tokens:
                    segments.append(seg)
                else:
                    # 无法识别的 token，避免死循环
                    tok = self._current()
                    raise ParserError(
                        f"Unexpected token '{tok.value}' at position {tok.position}",
                        f"位置 {tok.position} 处出现无法识别的标记 '{tok.value}'",
                        tok.position
                    )
        return BigSMILESString(segments=segments)

    def _parse_smiles_segment(self) -> SMILESSegment:
        """解析 SMILES 片段（直到遇到 { 或 EOF）。"""
        tokens = []
        smiles_types = {
            TokenType.ATOM_ORGANIC, TokenType.ATOM_BRACKET,
            TokenType.BOND, TokenType.BRANCH_OPEN, TokenType.BRANCH_CLOSE,
            TokenType.RING_DIGIT, TokenType.DOT,
            TokenType.BOND_DESC, TokenType.TERM_DESC_EMPTY,
        }
        while self._current().type in smiles_types:
            tokens.append(self._advance())
        raw = "".join(t.value for t in tokens)
        return SMILESSegment(tokens=tokens, raw_text=raw)

    def _parse_stochastic_object(self) -> StochasticObject:
        """解析随机对象 { ... }"""
        open_tok = self._expect(TokenType.STOCH_OPEN)
        open_pos = open_tok.position

        # 左终端描述符 — 仅 [] (空描述符) 作为显式终端，
        # [$] [>] [<] 属于重复单元内容
        left_term = None
        if self._current().type == TokenType.TERM_DESC_EMPTY:
            left_term = self._advance()

        # 解析重复单元（逗号分隔）
        repeat_units = []
        ru = self._parse_repeat_unit()
        repeat_units.append(ru)
        while self._current().type == TokenType.REPEAT_SEP:
            self._advance()  # consume ','
            ru = self._parse_repeat_unit()
            repeat_units.append(ru)

        # 端基（分号后）
        end_groups = []
        if self._current().type == TokenType.ENDGROUP_SEP:
            self._advance()  # consume ';'
            eg = self._parse_end_group()
            end_groups.append(eg)
            while self._current().type == TokenType.REPEAT_SEP:
                self._advance()  # consume ','
                eg = self._parse_end_group()
                end_groups.append(eg)

        # 右终端描述符 — 同上，仅 []
        right_term = None
        if self._current().type == TokenType.TERM_DESC_EMPTY:
            right_term = self._advance()

        close_tok = self._expect(TokenType.STOCH_CLOSE)

        return StochasticObject(
            left_term=left_term,
            repeat_units=repeat_units,
            end_groups=end_groups,
            right_term=right_term,
            open_pos=open_pos,
            close_pos=close_tok.position,
        )

    def _is_unit_token(self) -> bool:
        """判断当前 token 是否属于重复单元/端基的内容。"""
        t = self._current().type
        return t in {
            TokenType.ATOM_ORGANIC, TokenType.ATOM_BRACKET,
            TokenType.BOND, TokenType.BRANCH_OPEN, TokenType.BRANCH_CLOSE,
            TokenType.RING_DIGIT, TokenType.DOT,
            TokenType.BOND_DESC, TokenType.TERM_DESC_EMPTY,
            TokenType.STOCH_OPEN,  # 嵌套随机对象
        }

    def _parse_repeat_unit(self) -> RepeatUnit:
        """解析一个重复单元。"""
        tokens = []
        descriptors = []
        nested = []

        while self._is_unit_token():
            if self._current().type == TokenType.STOCH_OPEN:
                nested.append(self._parse_stochastic_object())
            elif self._current().type == TokenType.BOND_DESC:
                tok = self._advance()
                bd = self._parse_bond_descriptor(tok)
                descriptors.append(bd)
                tokens.append(tok)
            else:
                tokens.append(self._advance())

        raw = "".join(t.value for t in tokens)
        return RepeatUnit(tokens=tokens, descriptors=descriptors,
                          nested_objects=nested, raw_text=raw)

    def _parse_end_group(self) -> EndGroup:
        """解析一个端基。"""
        tokens = []
        descriptors = []
        nested = []

        while self._is_unit_token():
            if self._current().type == TokenType.STOCH_OPEN:
                nested.append(self._parse_stochastic_object())
            elif self._current().type == TokenType.BOND_DESC:
                tok = self._advance()
                bd = self._parse_bond_descriptor(tok)
                descriptors.append(bd)
                tokens.append(tok)
            else:
                tokens.append(self._advance())

        raw = "".join(t.value for t in tokens)
        return EndGroup(tokens=tokens, descriptors=descriptors,
                        nested_objects=nested, raw_text=raw)

    def _parse_bond_descriptor(self, tok: Token) -> BondingDescriptor:
        """从 Token 解析 BondingDescriptor。"""
        m = Tokenizer.BOND_DESC_RE.match(tok.value)
        if not m:
            raise ParserError(
                f"Invalid bonding descriptor: {tok.value}",
                f"无效键连接描述符: {tok.value}",
                tok.position
            )
        return BondingDescriptor(
            bond_type=m.group(1),
            descriptor_type=m.group(2),
            numeric_id=m.group(3),
            raw=tok.value,
            position=tok.position,
        )


# ---------------------------------------------------------------------------
# Stage 3: 语义校验器 / Semantic Validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """校验错误条目"""
    message_en: str
    message_cn: str
    position: int = -1


class Validator:
    """BigSMILES 语义校验器 — 7 项检查"""

    def __init__(self, original_text: str, ast: BigSMILESString):
        self.text = original_text
        self.ast = ast
        self.errors: List[ValidationError] = []

    def validate(self) -> List[ValidationError]:
        """执行全部 7 项检查。"""
        self.errors = []
        self._check_brackets()          # 1. 括号匹配
        self._check_descriptor_syntax()  # 2. 描述符语法
        self._check_stoch_structure()    # 3. 随机对象结构
        self._check_terminal_pairing()   # 4. 终端描述符配对
        self._check_smiles_validity()    # 5. SMILES 有效性
        self._check_descriptor_consistency()  # 6. 描述符一致性
        self._check_min_descriptors()    # 7. 最少描述符数
        return self.errors

    # --- Check 1: 括号匹配 ---
    def _check_brackets(self):
        stack = []
        pairs = {'{': '}', '[': ']', '(': ')'}
        close_to_open = {v: k for k, v in pairs.items()}
        names_cn = {'{': '花括号', '[': '方括号', '(': '圆括号'}

        for i, ch in enumerate(self.text):
            if ch in pairs:
                stack.append((ch, i))
            elif ch in close_to_open:
                expected_open = close_to_open[ch]
                if not stack:
                    self.errors.append(ValidationError(
                        f"Unmatched closing '{ch}' at position {i}",
                        f"位置 {i} 处的 '{ch}' 没有匹配的开括号",
                        i
                    ))
                elif stack[-1][0] != expected_open:
                    self.errors.append(ValidationError(
                        f"Mismatched brackets: expected closing for '{stack[-1][0]}' "
                        f"(pos {stack[-1][1]}), got '{ch}' (pos {i})",
                        f"括号不匹配: 期望 '{stack[-1][0]}' (位置 {stack[-1][1]}) "
                        f"的闭合括号，得到 '{ch}' (位置 {i})",
                        i
                    ))
                else:
                    stack.pop()

        for ch, pos in stack:
            self.errors.append(ValidationError(
                f"Unmatched opening '{ch}' at position {pos}",
                f"位置 {pos} 处的 '{ch}' 未闭合",
                pos
            ))

    # --- Check 2: 描述符语法 ---
    def _check_descriptor_syntax(self):
        """检查所有描述符的语法正确性。"""
        for seg in self.ast.segments:
            if isinstance(seg, StochasticObject):
                self._check_descriptors_in_stoch(seg)

    def _check_descriptors_in_stoch(self, stoch: StochasticObject):
        for ru in stoch.repeat_units:
            for bd in ru.descriptors:
                self._validate_single_descriptor(bd)
            for nested in ru.nested_objects:
                self._check_descriptors_in_stoch(nested)
        for eg in stoch.end_groups:
            for bd in eg.descriptors:
                self._validate_single_descriptor(bd)

    def _validate_single_descriptor(self, bd: BondingDescriptor):
        if bd.descriptor_type not in ('$', '<', '>'):
            self.errors.append(ValidationError(
                f"Invalid descriptor type '{bd.descriptor_type}' in {bd.raw}",
                f"描述符类型无效 '{bd.descriptor_type}' in {bd.raw}",
                bd.position
            ))

    # --- Check 3: 随机对象结构 ---
    def _check_stoch_structure(self):
        """每个随机对象至少 1 个重复单元。"""
        for seg in self.ast.segments:
            if isinstance(seg, StochasticObject):
                self._check_single_stoch_structure(seg)

    def _check_single_stoch_structure(self, stoch: StochasticObject):
        if len(stoch.repeat_units) == 0:
            self.errors.append(ValidationError(
                f"Stochastic object at position {stoch.open_pos} has no repeat units",
                f"位置 {stoch.open_pos} 的随机对象没有重复单元",
                stoch.open_pos
            ))
        # 检查重复单元非空
        for ru in stoch.repeat_units:
            if not ru.tokens and not ru.nested_objects:
                self.errors.append(ValidationError(
                    f"Empty repeat unit in stochastic object at position {stoch.open_pos}",
                    f"位置 {stoch.open_pos} 的随机对象中存在空重复单元",
                    stoch.open_pos
                ))
            for nested in ru.nested_objects:
                self._check_single_stoch_structure(nested)
        for eg in stoch.end_groups:
            for nested in eg.nested_objects:
                self._check_single_stoch_structure(nested)

    # --- Check 4: 终端描述符配对 ---
    def _check_terminal_pairing(self):
        """检查 [>n] 与 [<n] 的 ID 匹配。"""
        for seg in self.ast.segments:
            if isinstance(seg, StochasticObject):
                self._check_pairing_in_stoch(seg)

    def _check_pairing_in_stoch(self, stoch: StochasticObject):
        # 收集此随机对象内所有描述符
        all_descs = []
        for ru in stoch.repeat_units:
            all_descs.extend(ru.descriptors)
        for eg in stoch.end_groups:
            all_descs.extend(eg.descriptors)

        # 检查 < > 配对（当有数字 ID 时）
        gt_ids = {}
        lt_ids = {}
        for bd in all_descs:
            if bd.descriptor_type == '>' and bd.numeric_id:
                gt_ids[bd.numeric_id] = bd
            elif bd.descriptor_type == '<' and bd.numeric_id:
                lt_ids[bd.numeric_id] = bd

        for nid, bd in gt_ids.items():
            if nid not in lt_ids:
                self.errors.append(ValidationError(
                    f"Bonding descriptor [>{nid}] has no matching [<{nid}]",
                    f"键连接描述符 [>{nid}] 没有匹配的 [<{nid}]",
                    bd.position
                ))

        for nid, bd in lt_ids.items():
            if nid not in gt_ids:
                self.errors.append(ValidationError(
                    f"Bonding descriptor [<{nid}] has no matching [>{nid}]",
                    f"键连接描述符 [<{nid}] 没有匹配的 [>{nid}]",
                    bd.position
                ))

        # 递归检查嵌套
        for ru in stoch.repeat_units:
            for nested in ru.nested_objects:
                self._check_pairing_in_stoch(nested)

    # --- Check 5: SMILES 有效性 ---
    def _check_smiles_validity(self):
        """用 RDKit 校验 SMILES 片段（描述符替换为 *）。"""
        try:
            from rdkit import Chem
            has_rdkit = True
        except ImportError:
            has_rdkit = False

        if not has_rdkit:
            return  # 无 RDKit，跳过此检查

        for seg in self.ast.segments:
            if isinstance(seg, SMILESSegment):
                self._validate_smiles_fragment(seg.raw_text, seg.tokens[0].position if seg.tokens else 0)
            elif isinstance(seg, StochasticObject):
                self._check_smiles_in_stoch(seg)

    def _check_smiles_in_stoch(self, stoch: StochasticObject):
        from rdkit import Chem
        for ru in stoch.repeat_units:
            smiles_text = self._tokens_to_smiles(ru.tokens)
            if smiles_text.strip():
                pos = ru.tokens[0].position if ru.tokens else stoch.open_pos
                self._validate_smiles_fragment(smiles_text, pos)
            for nested in ru.nested_objects:
                self._check_smiles_in_stoch(nested)
        for eg in stoch.end_groups:
            smiles_text = self._tokens_to_smiles(eg.tokens)
            if smiles_text.strip():
                pos = eg.tokens[0].position if eg.tokens else stoch.open_pos
                self._validate_smiles_fragment(smiles_text, pos)

    def _tokens_to_smiles(self, tokens: List[Token]) -> str:
        """将 token 列表转化为 SMILES 字符串，描述符替换为 *。"""
        parts = []
        for tok in tokens:
            if tok.type == TokenType.BOND_DESC:
                parts.append("*")
            elif tok.type == TokenType.TERM_DESC_EMPTY:
                parts.append("*")
            else:
                parts.append(tok.value)
        return "".join(parts)

    def _validate_smiles_fragment(self, raw_text: str, position: int):
        from rdkit import Chem
        # 替换描述符为 *
        smiles = re.sub(r'\[[=#/\\]?[\$<>]\d*\]', '*', raw_text)
        if not smiles.strip():
            return
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            self.errors.append(ValidationError(
                f"Invalid SMILES fragment: '{smiles}' (from '{raw_text}')",
                f"无效的 SMILES 片段: '{smiles}' (来自 '{raw_text}')",
                position
            ))

    # --- Check 6: 描述符一致性 ---
    def _check_descriptor_consistency(self):
        """同一 {} 内不混用 $ 和 <>。"""
        for seg in self.ast.segments:
            if isinstance(seg, StochasticObject):
                self._check_consistency_in_stoch(seg)

    def _check_consistency_in_stoch(self, stoch: StochasticObject):
        all_descs = []
        for ru in stoch.repeat_units:
            all_descs.extend(ru.descriptors)
        for eg in stoch.end_groups:
            all_descs.extend(eg.descriptors)

        # 也检查终端描述符
        for term in (stoch.left_term, stoch.right_term):
            if term and term.type == TokenType.BOND_DESC:
                m = Tokenizer.BOND_DESC_RE.match(term.value)
                if m:
                    all_descs.append(BondingDescriptor(
                        bond_type=m.group(1),
                        descriptor_type=m.group(2),
                        numeric_id=m.group(3),
                        raw=term.value,
                        position=term.position,
                    ))

        has_dollar = any(bd.descriptor_type == '$' for bd in all_descs)
        has_angle = any(bd.descriptor_type in ('<', '>') for bd in all_descs)

        if has_dollar and has_angle:
            self.errors.append(ValidationError(
                f"Mixed AA-type ($) and AB-type (<>) descriptors in same "
                f"stochastic object at position {stoch.open_pos}",
                f"位置 {stoch.open_pos} 的随机对象中混用了 AA 型($)和 AB 型(<>)描述符",
                stoch.open_pos
            ))

        # 递归嵌套
        for ru in stoch.repeat_units:
            for nested in ru.nested_objects:
                self._check_consistency_in_stoch(nested)

    # --- Check 7: 最少描述符数 ---
    def _check_min_descriptors(self):
        """重复单元 >= 2 个描述符，端基 = 1 个描述符。"""
        for seg in self.ast.segments:
            if isinstance(seg, StochasticObject):
                self._check_min_desc_in_stoch(seg)

    def _check_min_desc_in_stoch(self, stoch: StochasticObject):
        for ru in stoch.repeat_units:
            # 对于包含嵌套随机对象的重复单元，嵌套对象本身消耗描述符连接
            # 所以放宽对含嵌套对象的重复单元的描述符数量要求
            min_expected = 2
            if len(ru.descriptors) < min_expected and not ru.nested_objects:
                self.errors.append(ValidationError(
                    f"Repeat unit has {len(ru.descriptors)} descriptor(s), "
                    f"expected >= {min_expected}",
                    f"重复单元有 {len(ru.descriptors)} 个描述符，"
                    f"期望 >= {min_expected} 个",
                    ru.tokens[0].position if ru.tokens else stoch.open_pos
                ))
            for nested in ru.nested_objects:
                self._check_min_desc_in_stoch(nested)

        for eg in stoch.end_groups:
            if len(eg.descriptors) != 1 and not eg.nested_objects:
                self.errors.append(ValidationError(
                    f"End group has {len(eg.descriptors)} descriptor(s), expected exactly 1",
                    f"端基有 {len(eg.descriptors)} 个描述符，期望恰好 1 个",
                    eg.tokens[0].position if eg.tokens else stoch.open_pos
                ))

        for ru in stoch.repeat_units:
            for nested in ru.nested_objects:
                self._check_min_desc_in_stoch(nested)


# ---------------------------------------------------------------------------
# 公共 API / Public API
# ---------------------------------------------------------------------------

def check_bigsmiles(s: str, verbose: bool = True) -> bool:
    """
    校验 BigSMILES 字符串。
    Validate a BigSMILES string.

    Args:
        s: BigSMILES 字符串
        verbose: 是否打印详细错误信息

    Returns:
        True if valid, False if errors found.
    """
    errors = []

    # Stage 1: 分词
    try:
        tokenizer = Tokenizer(s)
        tokens = tokenizer.tokenize()
    except TokenizerError as e:
        errors.append(ValidationError(e.message_en, e.message_cn, e.position))
        if verbose:
            _print_errors(s, errors)
        return False

    # Stage 2: 解析
    try:
        parser = Parser(tokens)
        ast = parser.parse()
    except ParserError as e:
        errors.append(ValidationError(e.message_en, e.message_cn, e.position))
        if verbose:
            _print_errors(s, errors)
        return False

    # Stage 3: 语义校验
    validator = Validator(s, ast)
    errors = validator.validate()

    if verbose and errors:
        _print_errors(s, errors)

    if verbose and not errors:
        print(f"[OK] BigSMILES 语法正确 / Valid BigSMILES: {s}")

    return len(errors) == 0


def _print_errors(s: str, errors: List[ValidationError]):
    """格式化打印错误信息。"""
    print(f"\n  Input: {s}")
    for err in errors:
        pos_str = f"position {err.position}" if err.position >= 0 else "unknown position"
        print(f"\n  [ERROR] (位置/position {err.position})")
        print(f"    EN: {err.message_en}")
        print(f"    CN: {err.message_cn}")
    print()


# ---------------------------------------------------------------------------
# 命令行入口 / CLI Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bigsmiles_checker.py <bigsmiles_string>")
        print('Example: python bigsmiles_checker.py "{[$]CC[$]}"')
        sys.exit(1)

    input_str = sys.argv[1]
    result = check_bigsmiles(input_str, verbose=True)
    sys.exit(0 if result else 1)
