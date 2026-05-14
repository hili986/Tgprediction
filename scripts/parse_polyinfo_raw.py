"""Parse raw text copied from PoLyInfo into structured CSV.

Usage:
    python scripts/parse_polyinfo_raw.py \
        --raw raw_data.txt [raw_data1.txt ...] \
        --out data/external/polyinfo_copolymer_tg.csv

The script handles two types of input:
  1. Sample data (from copolymer detail pages)
  2. Monomer lookup data (from constitutional unit pages)

It auto-detects which type each block is and processes accordingly.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Monomer:
    """A constitutional unit (monomer) from PoLyInfo."""
    cuid: str = ""           # e.g. CU040001
    pid: str = ""            # e.g. P040001
    name: str = ""           # IUPAC source based name
    abbreviation: str = ""
    cu_formula: str = ""     # e.g. C3H4O2
    fw: str = ""             # formula weight
    smiles: str = ""


@dataclass
class Sample:
    """A single copolymer Tg measurement from PoLyInfo."""
    sample_id: str = ""
    polymer_id: str = ""     # COID, e.g. P900001
    name: str = ""
    copolymer_type: str = "U"  # R/B/A/G/U
    cuid_1: str = ""
    cuid_2: str = ""
    cu_formula_1: str = ""
    cu_formula_2: str = ""
    fw_1: str = ""
    fw_2: str = ""
    # Composition: cuid -> value, unit stored separately
    comp_cuid_order: list = field(default_factory=list)
    comp_values: list = field(default_factory=list)
    comp_unit: str = ""      # mol% or wt%
    tg_c: str = ""
    tg_method: str = ""
    mn: str = ""
    mw: str = ""
    reference: str = ""
    # Related info table rows
    related_rows: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Monomer lookup parsing
# ---------------------------------------------------------------------------

def parse_monomer_blocks(text: str) -> list[Monomer]:
    """Parse monomer info blocks from raw_data1.txt style text."""
    monomers = []
    current = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Polymer ID(PID):"):
            if current and current.pid:
                monomers.append(current)
            current = Monomer()
            current.pid = line.split("\t")[-1].strip()
        elif current is None:
            continue
        elif line.startswith("IUPAC source based name:"):
            current.name = line.split("\t")[-1].strip()
        elif line.startswith("Abbreviation:"):
            current.abbreviation = line.split("\t")[-1].strip()
        elif line.startswith("CU formula:"):
            current.cu_formula = line.split("\t")[-1].strip()
        elif line.startswith("Formula weight(FW):"):
            current.fw = line.split("\t")[-1].strip()
        elif line.startswith("Smiles:"):
            current.smiles = line.split("\t")[-1].strip()

    if current and current.pid:
        monomers.append(current)
    return monomers


def build_smiles_lookup(monomers: list[Monomer]) -> tuple[dict, dict, dict]:
    """Build CUID->SMILES, formula->SMILES, CUID->Monomer lookups.

    CUID-based lookup is the primary method (unambiguous).
    Formula-based is a fallback (fails on isomers like PMMA/PEA, both C5H8O2).
    """
    cuid_to_smiles: dict[str, str] = {}
    formula_to_smiles: dict[str, str] = {}
    cuid_to_monomer: dict[str, Monomer] = {}
    for m in monomers:
        if m.pid and m.smiles:
            cuid = pid_to_cuid(m.pid)
            cuid_to_smiles[cuid] = m.smiles
            cuid_to_monomer[cuid] = m
        if m.cu_formula and m.smiles:
            # Only set if not already present (first occurrence wins for isomers)
            formula_to_smiles.setdefault(m.cu_formula, m.smiles)
    return cuid_to_smiles, formula_to_smiles, cuid_to_monomer


# ---------------------------------------------------------------------------
# CUID extraction helper
# ---------------------------------------------------------------------------

# PoLyInfo sometimes uses PID (P040001) in monomer detail pages
# and CUID (CU040001) in component sections. Map between them.
PID_TO_CUID = {}  # populated during parsing


def pid_to_cuid(pid: str) -> str:
    """Convert PID like P040001 to CUID like CU040001."""
    if pid.startswith("CU"):
        return pid
    # P040001 -> CU040001
    m = re.match(r'P(\d+)', pid)
    if m:
        return f"CU{m.group(1)}"
    return pid


# ---------------------------------------------------------------------------
# Sample data parsing
# ---------------------------------------------------------------------------

def parse_tg_value(raw: str) -> tuple[str, bool]:
    """Extract numeric Tg from strings like '-42[C]', 'ca15[C]', etc.

    Returns (value_str, is_approximate).
    """
    raw = raw.strip()
    is_approx = raw.startswith("ca") or raw.startswith(">=") or raw.startswith("<=") \
        or raw.startswith(">") or raw.startswith("<")
    # Remove 'ca' prefix
    cleaned = re.sub(r'^ca\s*', '', raw)
    # Remove inequality prefixes: >=, <=, >, <
    cleaned = re.sub(r'^[<>]=?\s*', '', cleaned)
    # Remove [C] suffix
    cleaned = re.sub(r'\[C\]$', '', cleaned)
    # Remove any trailing whitespace
    cleaned = cleaned.strip()
    # Validate it parses as a number
    try:
        float(cleaned)
    except ValueError:
        return "", is_approx
    return cleaned, is_approx


_NON_MONOMER_TOKENS = frozenset({
    # Physical units that may appear in X/Y form
    "G", "KG", "MG", "CM", "CM2", "CM3", "MM", "M", "ML", "L",
    "PA", "KPA", "MPA", "GPA", "TORR",
    "DEGREE", "K", "C", "F",
    "MOL", "MOLE", "WT", "VOL",
    "HZ", "KHZ", "MHZ", "GHZ",
    "DYN", "J", "CAL", "KCAL", "N", "KN",
    "PPM", "PPB", "%",
    "S", "MIN", "H", "HR", "DAY",
})


def _extract_abbrev_pair(text: str) -> set[str]:
    """Extract a monomer abbreviation pair like 'AA/BA' from a Related table
    title or header. Returns a set of uppercase tokens, or empty set.

    Handles digit-starting abbreviations (2-EHA) and avoids physical units.

    Examples:
      "Compn. (AA/BA)[mol%]"          -> {"AA", "BA"}
      "in 2-EHA/AA copolymer"         -> {"2-EHA", "AA"}
      "Compn.(ethylene/vinyl acetate)" -> {"ETHYLENE", "VINYL ACETATE"}
    """
    # Token pattern: letters/digits, may have dash, 1-20 chars
    # Allows digit-starting tokens like "2-EHA"
    TOKEN = r'[A-Za-z0-9][A-Za-z\-0-9 ]{0,19}'

    # Try parenthesized pair: (X/Y) — strongest signal, allows full names
    m = re.search(rf'\(\s*({TOKEN}?)/({TOKEN}?)\s*\)', text)
    if m:
        t1 = m.group(1).strip().upper()
        t2 = m.group(2).strip().upper()
        if t1 and t2:
            tokens = {t1, t2}
            if not tokens & _NON_MONOMER_TOKENS:
                return tokens

    # "X/Y ratio" pattern
    SHORT = r'[A-Za-z0-9][A-Za-z\-0-9]{0,5}'
    m = re.search(rf'\b({SHORT})/({SHORT})\s+ratio\b', text)
    if m:
        tokens = {m.group(1).upper(), m.group(2).upper()}
        if not tokens & _NON_MONOMER_TOKENS:
            return tokens

    # "in X/Y copolymer" or "in X-Y copolymer" — title pattern
    m = re.search(rf'\bin\s+({TOKEN}?)[/-]({TOKEN}?)\s+copolymer',
                  text, re.IGNORECASE)
    if m:
        t1 = m.group(1).strip().upper()
        t2 = m.group(2).strip().upper()
        if t1 and t2:
            tokens = {t1, t2}
            if not tokens & _NON_MONOMER_TOKENS:
                return tokens

    # "of X-Y copolymer" variant
    m = re.search(rf'\bof\s+({TOKEN}?)[/-]({TOKEN}?)\s+copolymer',
                  text, re.IGNORECASE)
    if m:
        t1 = m.group(1).strip().upper()
        t2 = m.group(2).strip().upper()
        if t1 and t2:
            tokens = {t1, t2}
            if not tokens & _NON_MONOMER_TOKENS:
                return tokens

    return set()


def _strip_poly_prefix(ab: str) -> str:
    """Strip 'P' or 'PA' prefix from polymer abbreviation to get monomer abbrev.

    PAA -> AA, PBA -> BA, PMMA -> MMA, PS -> S, PE -> E, PAN -> AN.
    For nylons: N6/PA 6/PA6 -> kept as-is (N6/PA6).
    """
    ab = ab.upper().strip()
    if not ab:
        return ""
    # Nylons: keep N6, N66, N12 as is
    if re.match(r'^N\d+$', ab) or re.match(r'^PA\s*\d+$', ab):
        return ab.replace(" ", "")
    # Strip leading P if result is still >= 1 char
    if ab.startswith("P") and len(ab) > 1:
        return ab[1:]
    return ab


def _parse_float_safe(s: str) -> float | None:
    """Parse a number from messy string: '>=−11.5[C]', 'ca-20', '3/97' -> 3."""
    if not s:
        return None
    s = s.strip()
    # Handle "X/Y" ratio: take first value
    if "/" in s:
        s = s.split("/")[0].strip()
    # Strip ca, inequality, [C] suffix
    s = re.sub(r'^ca\s*', '', s)
    s = re.sub(r'^[<>]=?\s*', '', s)
    s = re.sub(r'\[C\]$', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def _verify_table_by_data(
    sample_ratio: float | None,
    sample_tg: float | None,
    header: str,
    data_lines: list[str],
    ratio_tol: float = 1.0,
    tg_tol: float = 2.0,
) -> bool | None:
    """Verify a Related table by checking if sample's (ratio, Tg) appears in the
    table data rows.

    Returns:
        True  — sample's (ratio, Tg) found in table (table belongs to this sample)
        False — not found (table is about a different copolymer)
        None  — cannot verify (missing data)
    """
    if sample_ratio is None or sample_tg is None:
        return None

    # Locate Tg column in header
    h_parts = header.split("\t")
    tg_col = None
    for ci, col in enumerate(h_parts):
        if "Tg" in col:
            tg_col = ci
            break
    if tg_col is None:
        return None

    # Scan data rows for matching (ratio, Tg)
    for data_line in data_lines:
        d_parts = data_line.split("\t")
        if len(d_parts) <= tg_col:
            continue

        # Parse ratio — first column may be "X/Y" or "X"
        ratio_cell = d_parts[0].strip()
        r1 = _parse_float_safe(ratio_cell)
        r2 = None
        if "/" in ratio_cell:
            parts = ratio_cell.split("/")
            if len(parts) == 2:
                r2 = _parse_float_safe(parts[1])

        # Parse Tg
        tg_val = _parse_float_safe(d_parts[tg_col])
        if tg_val is None:
            continue

        # Tg must match (strong signal)
        if abs(tg_val - sample_tg) > tg_tol:
            continue

        # Ratio must match either r1 or r2 (order-invariant)
        if r1 is not None and abs(r1 - sample_ratio) < ratio_tol:
            return True
        if r2 is not None and abs(r2 - sample_ratio) < ratio_tol:
            return True

    return False


def _abbrevs_match(header_abbrevs: set[str], sample_abbrevs: set[str]) -> bool:
    """Check if header abbreviations match sample abbreviations.

    Uses prefix tolerance: VA matches VAC (both vinyl acetate),
    AN matches ANY in {AN, PAN}, etc. Min 2 chars to avoid spurious matches.
    """
    for h in header_abbrevs:
        matched = False
        for s in sample_abbrevs:
            if h == s:
                matched = True
                break
            # Allow prefix match if longer token contains shorter as prefix
            # and the shorter is at least 2 chars (avoids "S" matching everything)
            short, long_ = (h, s) if len(h) <= len(s) else (s, h)
            if len(short) >= 2 and long_.startswith(short):
                matched = True
                break
        if not matched:
            return False
    return True


_ETHENE_SYNONYMS = {"ETHENE", "ETHYLENE", "E"}


def _build_sample_abbrev_set(sample: 'Sample', cuid_to_monomer: dict) -> set[str]:
    """Build set of possible abbreviations + full names for sample's monomers.

    Includes:
      - Abbreviation with/without P prefix (PAA -> {PAA, AA})
      - Full monomer name from IUPAC source-based name (poly-prefix stripped)
      - Known synonyms (ethene <-> ethylene)
    """
    tokens: set[str] = set()
    for cuid in (sample.cuid_1, sample.cuid_2):
        m = cuid_to_monomer.get(cuid)
        if not m:
            continue
        # Abbreviations
        if m.abbreviation:
            for ab in m.abbreviation.split():
                ab = ab.strip().upper()
                if ab:
                    tokens.add(ab)
                    tokens.add(_strip_poly_prefix(ab))
        # Monomer full name: strip poly prefix from polymer name
        if m.name:
            name = m.name.upper().strip()
            # "poly(acrylic acid)" -> "acrylic acid", "polyethene" -> "ethene"
            monomer_name = re.sub(r'^POLY[\(\[]?', '', name)
            monomer_name = re.sub(r'[\)\]]$', '', monomer_name).strip()
            if monomer_name:
                tokens.add(monomer_name)
                # Add synonyms
                if monomer_name in _ETHENE_SYNONYMS:
                    tokens |= _ETHENE_SYNONYMS
    tokens.discard("")
    return tokens


def _normalize_unit(line: str) -> str:
    """Extract and normalize composition unit from PoLyInfo text.

    Variants seen in data:
      [mol%], [wt%]          -> canonical
      [mol ratio], [wt ratio] -> treat as mol%/wt% (semantically equivalent)
      [%]                     -> ambiguous, default to mol%
      [unknown]               -> empty
    """
    m = re.search(r'\[([^\]]+)\]', line)
    if not m:
        return ""
    raw = m.group(1).strip().lower()
    if raw == "unknown":
        return ""
    if "mol" in raw:
        return "mol%"
    if "wt" in raw:
        return "wt%"
    if raw == "%":
        return "mol%"  # assume mol% when bare %
    return raw


def _is_related_header(line: str) -> bool:
    """Detect if a line is a Related Information table header.

    Must contain "Tg" AND the first column must indicate copolymer composition.
    Excludes:
      - Blend tables
      - Additive/filler content tables (e.g., "Organophilic MMT clay content[wt%]")
    """
    if "Tg" not in line:
        return False
    if "\t" not in line:
        return False  # must be tab-delimited table header
    line_lower = line.lower()
    if "blend" in line_lower:
        return False

    # First column (before first tab) must be a composition column
    first_col = line.split("\t")[0].lower()

    # Reject additive/filler content columns
    additive_indicators = [
        "clay content", "filler content", "additive", "loading",
        "organophilic", "content of ", "dosage",
    ]
    if any(ind in first_col for ind in additive_indicators):
        return False

    # Accept if first column indicates composition
    composition_indicators = ["compn.", "composition", "ratio"]
    if any(ind in first_col for ind in composition_indicators):
        return True

    # Fallback: first column has "X/Y" pattern with unit (e.g., "E/P[wt%]")
    if "/" in first_col and ("mol%" in first_col or "wt%" in first_col):
        return True

    return False


def parse_sample_blocks(text: str) -> list[Sample]:
    """Parse sample info blocks from raw PoLyInfo text."""
    samples = []
    current: Optional[Sample] = None
    in_component = False
    in_composition = False
    in_related = False
    related_header = ""
    related_title = ""  # the description line just above the header
    prev_line = ""      # track previous non-empty line as potential title
    component_row_idx = 0
    composition_row_idx = 0

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # Skip monomer-only blocks (from raw_data1.txt mixed in)
        if line.startswith("Polymer ID(PID):"):
            continue

        # New sample block
        if line.startswith("Sample ID:"):
            if current and current.sample_id:
                samples.append(current)
            current = Sample()
            current.sample_id = line.split("\t")[-1].strip()
            in_component = False
            in_composition = False
            in_related = False
            continue

        if current is None:
            # Check if this is a related info line from a previous sample
            continue

        # Basic fields
        if line.startswith("Polymer ID:"):
            current.polymer_id = line.split("\t")[-1].strip()
        elif line.startswith("Name:"):
            current.name = line.split("\t")[-1].strip()
        elif line.startswith("Copolymer Type:"):
            ct = line.split("\t")[-1].strip().lower()
            type_map = {
                "random": "R", "block": "B", "alternating": "A",
                "graft": "G", "unspecified": "U",
            }
            current.copolymer_type = type_map.get(ct, "U")

        # Molecular weights
        elif line.startswith("Mn:"):
            val = line.split(":")[-1].strip().split()[0]
            current.mn = re.sub(r'[^\d.]', '', val)
        elif line.startswith("Mw:"):
            val = line.split(":")[-1].strip().split()[0]
            current.mw = re.sub(r'[^\d.]', '', val)

        # Reference
        elif line.startswith("Reference:"):
            current.reference = line.split("\t")[-1].strip()

        # Component section
        elif line.startswith("Component:"):
            in_component = True
            in_composition = False
            in_related = False
            component_row_idx = 0
        elif in_component and line.startswith("CUID of Component"):
            parts = line.split("\t")
            if len(parts) >= 3:
                current.cuid_1 = parts[1].strip()
                current.cuid_2 = parts[2].strip()
        elif in_component and line.startswith("CU formula"):
            parts = line.split("\t")
            if len(parts) >= 3:
                current.cu_formula_1 = parts[1].strip()
                current.cu_formula_2 = parts[2].strip()
        elif in_component and line.startswith("Formula weight"):
            parts = line.split("\t")
            if len(parts) >= 3:
                current.fw_1 = parts[1].strip()
                current.fw_2 = parts[2].strip()
            in_component = False

        # Composition section
        elif line.startswith("Composition:"):
            in_composition = True
            in_component = False
            in_related = False
            composition_row_idx = 0
        elif in_composition and line.startswith("CUID of Component"):
            parts = line.split("\t")
            current.comp_cuid_order = [p.strip() for p in parts[1:] if p.strip()]
        elif in_composition and line.startswith("Compotision"):
            # Variants: [mol%], [wt%], [mol ratio], [wt ratio], [%], [unknown]
            current.comp_unit = _normalize_unit(line)
            parts = line.split("\t")
            current.comp_values = [p.strip() for p in parts[1:] if p.strip()]
            in_composition = False

        # Property: Glass transition temperature
        elif line.startswith("Glass transition temperature") and not line.startswith("Glass transition temperatures"):
            # Could be "Glass transition temperature\t-42[C]"
            parts = line.split("\t")
            if len(parts) >= 2:
                val_str = parts[-1].strip()
                if "[C]" in val_str or re.match(r'^ca?-?\d', val_str):
                    tg_val, _ = parse_tg_value(val_str)
                    if tg_val and not current.tg_c:
                        current.tg_c = tg_val
        elif line.startswith("Measurement methods") and current.tg_c and not current.tg_method:
            current.tg_method = line.split("\t")[-1].strip()

        # Related Information table
        elif line.startswith("Related Information:") or line.startswith("Related information"):
            in_related = True
            in_component = False
            in_composition = False
        elif in_related and _is_related_header(line):
            # Related info table header, e.g.:
            # "Compn. (AA/BA)[mol%]\tMn\tMw\tTg[C]"
            # "E/AA ratio[wt%]\tTm[C]\t...\tTg[C]\t..."
            related_header = line
            related_title = prev_line  # the line just above is the description
        elif in_related and related_header and re.match(r'^[\d./]+\t', line):
            # Data row in related info table (store title + header + data)
            current.related_rows.append((related_title, related_header, line))

        # Section transitions
        elif line.startswith("Information:"):
            # Start of a new block without "Sample ID:" on same line
            # Save current and reset
            if current and current.sample_id:
                samples.append(current)
                current = None
            in_component = False
            in_composition = False
            in_related = False
        elif line.startswith("Property:") or line.startswith("Dynamic"):
            in_component = False
            in_composition = False
            # Don't reset in_related — properties and related can coexist

        # Track previous line for Related table title detection
        prev_line = line

    # Don't forget last sample
    if current and current.sample_id:
        samples.append(current)

    return samples


# ---------------------------------------------------------------------------
# Resolve composition to ratio_1 for CUID_1
# ---------------------------------------------------------------------------

def resolve_ratio(sample: Sample) -> tuple[str, str]:
    """Return (ratio_1_value, ratio_unit) for cuid_1.

    PoLyInfo composition section lists CUIDs in arbitrary order,
    not necessarily matching the component order.
    """
    if not sample.comp_cuid_order or not sample.comp_values:
        return "", ""

    # Find which position corresponds to cuid_1
    for idx, cuid in enumerate(sample.comp_cuid_order):
        if cuid == sample.cuid_1 and idx < len(sample.comp_values):
            return sample.comp_values[idx], sample.comp_unit

    # If cuid_1 not found in composition order, return first value
    if sample.comp_values:
        return sample.comp_values[0], sample.comp_unit
    return "", ""


# ---------------------------------------------------------------------------
# Convert to CSV rows
# ---------------------------------------------------------------------------

def _lookup_smiles(cuid: str, cu_formula: str,
                   cuid_map: dict, formula_map: dict) -> str:
    """Look up SMILES, preferring CUID (unambiguous) over formula (fails on isomers)."""
    if cuid and cuid in cuid_map:
        return cuid_map[cuid]
    return formula_map.get(cu_formula, "")


def samples_to_rows(
    samples: list[Sample],
    cuid_to_smiles: dict,
    formula_to_smiles: dict,
) -> list[dict]:
    """Convert parsed samples to CSV row dicts."""
    rows = []

    for s in samples:
        if not s.tg_c:
            continue  # skip samples without Tg

        smiles_1 = _lookup_smiles(s.cuid_1, s.cu_formula_1, cuid_to_smiles, formula_to_smiles)
        smiles_2 = _lookup_smiles(s.cuid_2, s.cu_formula_2, cuid_to_smiles, formula_to_smiles)
        ratio_1, ratio_unit = resolve_ratio(s)

        # Determine monomer names from formula lookup or leave blank
        rows.append({
            "COID": s.polymer_id,
            "name": s.name,
            "copolymer_type": s.copolymer_type,
            "CUID_1": s.cuid_1,
            "monomer_1_name": "",  # filled later from lookup
            "SMILES_1": smiles_1,
            "CUID_2": s.cuid_2,
            "monomer_2_name": "",
            "SMILES_2": smiles_2,
            "ratio_1": ratio_1,
            "ratio_unit": ratio_unit,
            "Tg_C": s.tg_c,
            "Tg_method": s.tg_method,
            "Mn": s.mn,
            "Mw": s.mw,
            "reference": s.reference,
            "sample_id": s.sample_id,
            "notes": "",
        })

    return rows


def related_to_rows(
    samples: list[Sample],
    cuid_to_smiles: dict,
    formula_to_smiles: dict,
    cuid_to_monomer: dict,
) -> tuple[list[dict], list[str]]:
    """Extract rows from Related Information tables.

    Skips tables that belong to a DIFFERENT copolymer system (e.g., a
    sample page may list Related tables for other copolymers from the
    same reference — those must not be attributed to the current sample).

    Returns (rows, skip_log).
    """
    rows = []
    seen = set()  # dedup by (polymer_id, ratio, tg)
    skip_log: list[str] = []

    for s in samples:
        if not s.related_rows:
            continue

        smiles_1 = _lookup_smiles(s.cuid_1, s.cu_formula_1, cuid_to_smiles, formula_to_smiles)
        smiles_2 = _lookup_smiles(s.cuid_2, s.cu_formula_2, cuid_to_smiles, formula_to_smiles)
        sample_abbrevs = _build_sample_abbrev_set(s, cuid_to_monomer)

        # Get sample's own (ratio, Tg) for data-anchor matching
        sample_ratio_str, _ = resolve_ratio(s)
        sample_ratio = _parse_float_safe(sample_ratio_str)
        sample_tg = _parse_float_safe(s.tg_c)

        # Group rows by (title, header) — verify each table once, not per-row
        from collections import defaultdict
        tables: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in s.related_rows:
            if len(row) == 3:
                title, header, data_line = row
            else:
                title, header, data_line = "", row[0], row[1]
            tables[(title, header)].append(data_line)

        # Verify each table and decide keep/skip
        table_verdicts: dict[tuple[str, str], bool] = {}
        for (title, header), data_lines in tables.items():
            # Primary: data-anchor check (sample's (ratio, Tg) in table?)
            data_verdict = _verify_table_by_data(sample_ratio, sample_tg, header, data_lines)

            if data_verdict is True:
                table_verdicts[(title, header)] = True
                continue
            if data_verdict is False:
                table_verdicts[(title, header)] = False
                skip_log.append(
                    f"  [SKIP] sample {s.sample_id}: table '{title[:40]}' — "
                    f"sample (ratio={sample_ratio},Tg={sample_tg}) not found in table"
                )
                continue

            # Fallback: abbreviation matching when data-anchor unavailable
            if sample_abbrevs:
                hdr_abbrevs = _extract_abbrev_pair(header) or _extract_abbrev_pair(title)
                if hdr_abbrevs and not _abbrevs_match(hdr_abbrevs, sample_abbrevs):
                    table_verdicts[(title, header)] = False
                    skip_log.append(
                        f"  [SKIP-fallback] sample {s.sample_id} ({'/'.join(sample_abbrevs)}): "
                        f"Related table for {'/'.join(hdr_abbrevs)} — different copolymer"
                    )
                    continue
            table_verdicts[(title, header)] = True  # conservative: keep if can't disprove

        # Emit only rows from tables verdict=True
        for row in s.related_rows:
            if len(row) == 3:
                title, header, data_line = row
            else:
                title, header, data_line = "", row[0], row[1]
            if not table_verdicts.get((title, header), True):
                continue
            # Parse header to find column positions
            # e.g. "Compn. (AA/BA)[mol%]\tMn\tMw\tTg[C]"
            h_parts = header.split("\t")
            d_parts = data_line.split("\t")

            # Find ratio unit from header — use normalize for robustness
            ratio_unit = _normalize_unit(h_parts[0])

            # Find Tg column(s) — prefer DSC over DMA over generic
            tg_dsc_col = None
            tg_dma_col = None
            tg_generic_col = None
            mn_col = None
            mw_col = None
            tg_method = ""
            for ci, col in enumerate(h_parts):
                col_stripped = col.strip()
                if "Tg" in col_stripped:
                    if "DSC" in col_stripped:
                        tg_dsc_col = ci
                    elif "DMA" in col_stripped:
                        tg_dma_col = ci
                    else:
                        tg_generic_col = ci
                elif col_stripped == "Mn":
                    mn_col = ci
                elif col_stripped == "Mw":
                    mw_col = ci

            # Priority: DSC > generic > DMA
            if tg_dsc_col is not None:
                tg_col = tg_dsc_col
                tg_method = "DSC"
            elif tg_generic_col is not None:
                tg_col = tg_generic_col
                tg_method = ""
            elif tg_dma_col is not None:
                tg_col = tg_dma_col
                tg_method = "DMA"
            else:
                tg_col = None

            if tg_col is None or tg_col >= len(d_parts):
                continue

            tg_raw = d_parts[tg_col].strip()
            if not tg_raw or tg_raw == "[ND]":
                # Fallback: try DMA column if DSC was empty
                if tg_method == "DSC" and tg_dma_col is not None and tg_dma_col < len(d_parts):
                    tg_raw = d_parts[tg_dma_col].strip()
                    tg_method = "DMA"
                if not tg_raw or tg_raw == "[ND]":
                    continue

            tg_val, _ = parse_tg_value(tg_raw)
            if not tg_val:
                continue

            # Parse ratio from first column: "5/95" or "0/100"
            ratio_raw = d_parts[0].strip()
            ratio_parts = ratio_raw.split("/")
            ratio_1 = ratio_parts[0].strip() if ratio_parts else ""

            # Dedup
            key = (s.polymer_id, ratio_raw, tg_val)
            if key in seen:
                continue
            seen.add(key)

            mn_val = ""
            mw_val = ""
            if mn_col is not None and mn_col < len(d_parts):
                mn_val = d_parts[mn_col].strip()
            if mw_col is not None and mw_col < len(d_parts):
                mw_val = d_parts[mw_col].strip()

            rows.append({
                "COID": s.polymer_id,
                "name": s.name,
                "copolymer_type": s.copolymer_type,
                "CUID_1": s.cuid_1,
                "monomer_1_name": "",
                "SMILES_1": smiles_1,
                "CUID_2": s.cuid_2,
                "monomer_2_name": "",
                "SMILES_2": smiles_2,
                "ratio_1": ratio_1,
                "ratio_unit": ratio_unit,
                "Tg_C": tg_val,
                "Tg_method": tg_method,
                "Mn": mn_val,
                "Mw": mw_val,
                "reference": s.reference,
                "sample_id": "",
                "notes": "from_related_table",
            })

    return rows, skip_log


# ---------------------------------------------------------------------------
# Fill monomer names from lookup
# ---------------------------------------------------------------------------

CUID_TO_NAME = {}  # populated from monomer data


def fill_monomer_names(rows: list[dict], monomers: list[Monomer]) -> None:
    """Fill monomer_1_name and monomer_2_name from parsed monomers."""
    # Build CUID -> name mapping
    for m in monomers:
        cuid = pid_to_cuid(m.pid)
        name = m.abbreviation or m.name
        if cuid and name:
            CUID_TO_NAME[cuid] = name

    for row in rows:
        if not row["monomer_1_name"] and row["CUID_1"] in CUID_TO_NAME:
            row["monomer_1_name"] = CUID_TO_NAME[row["CUID_1"]]
        if not row["monomer_2_name"] and row["CUID_2"] in CUID_TO_NAME:
            row["monomer_2_name"] = CUID_TO_NAME[row["CUID_2"]]


# ---------------------------------------------------------------------------
# Dedup: prefer sample-level rows over related table rows
# ---------------------------------------------------------------------------

def dedup_rows(sample_rows: list[dict], related_rows: list[dict]) -> list[dict]:
    """Merge and deduplicate. Sample-level rows take priority.

    Dedup uses two keys:
      1. sample_id (if present) — exact duplicate detection
      2. (COID, ratio_1, Tg_C) — semantic duplicate detection
    """
    seen_sample_ids: set[str] = set()
    seen_keys: set[tuple] = set()
    result = []

    for row in sample_rows:
        sid = row["sample_id"]
        if sid and sid in seen_sample_ids:
            continue  # exact duplicate
        key = (row["COID"], row["ratio_1"], row["Tg_C"])
        if key in seen_keys:
            continue  # semantic duplicate
        if sid:
            seen_sample_ids.add(sid)
        seen_keys.add(key)
        result.append(row)

    for row in related_rows:
        key = (row["COID"], row["ratio_1"], row["Tg_C"])
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(row)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "COID", "name", "copolymer_type",
    "CUID_1", "monomer_1_name", "SMILES_1",
    "CUID_2", "monomer_2_name", "SMILES_2",
    "ratio_1", "ratio_unit", "Tg_C", "Tg_method",
    "Mn", "Mw", "reference", "sample_id", "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse raw PoLyInfo text into structured CSV"
    )
    parser.add_argument(
        "--raw", nargs="+", required=True,
        help="Raw text files copied from PoLyInfo",
    )
    parser.add_argument(
        "--out", default="data/external/polyinfo_copolymer_tg.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing CSV instead of overwriting",
    )
    args = parser.parse_args()

    all_monomers: list[Monomer] = []
    all_samples: list[Sample] = []

    for fpath in args.raw:
        text = Path(fpath).read_text(encoding="utf-8")

        # Auto-detect: monomer lookup vs sample data
        if "Polymer ID(PID):" in text:
            monomers = parse_monomer_blocks(text)
            all_monomers.extend(monomers)
            print(f"  [单体] {fpath}: 解析到 {len(monomers)} 种单体")

        if "Sample ID:" in text:
            samples = parse_sample_blocks(text)
            all_samples.extend(samples)
            print(f"  [样本] {fpath}: 解析到 {len(samples)} 条样本")

    # Build SMILES lookup
    cuid_to_smiles, formula_to_smiles, cuid_to_monomer = build_smiles_lookup(all_monomers)
    print(f"\nCUID SMILES mapping: {len(cuid_to_smiles)} monomers")
    for cuid, smiles in cuid_to_smiles.items():
        print(f"  {cuid} -> {smiles}")

    # Convert to rows
    sample_rows = samples_to_rows(all_samples, cuid_to_smiles, formula_to_smiles)
    related_rows, skip_log = related_to_rows(
        all_samples, cuid_to_smiles, formula_to_smiles, cuid_to_monomer
    )

    if skip_log:
        print(f"\n[SKIPPED] {len(skip_log)} Related tables from different copolymer systems:")
        for msg in skip_log[:10]:
            print(msg)
        if len(skip_log) > 10:
            print(f"  ... and {len(skip_log) - 10} more")

    # Fill monomer names
    all_rows = dedup_rows(sample_rows, related_rows)
    fill_monomer_names(all_rows, all_monomers)

    # Statistics
    has_tg = [r for r in all_rows if r["Tg_C"]]
    has_smiles = [r for r in all_rows if r["SMILES_1"] and r["SMILES_2"]]
    has_ratio = [r for r in all_rows if r["ratio_1"]]
    missing_smiles_formulas = set()
    for r in all_rows:
        if not r["SMILES_1"]:
            # Find the formula
            for s in all_samples:
                if s.cuid_1 == r["CUID_1"]:
                    missing_smiles_formulas.add(s.cu_formula_1)
        if not r["SMILES_2"]:
            for s in all_samples:
                if s.cuid_2 == r["CUID_2"]:
                    missing_smiles_formulas.add(s.cu_formula_2)

    print(f"\n=== 解析结果 ===")
    print(f"总行数: {len(all_rows)}")
    print(f"  有 Tg: {len(has_tg)}")
    print(f"  有 SMILES (双): {len(has_smiles)}")
    print(f"  有组成比例: {len(has_ratio)}")
    print(f"  来自 Related Table: {sum(1 for r in all_rows if r['notes'] == 'from_related_table')}")

    if missing_smiles_formulas:
        print(f"\n[WARNING] missing SMILES for formulas: {missing_smiles_formulas}")
        print(f"  please add these monomers to monomer lookup file")

    # Write CSV
    out_path = Path(args.out)
    mode = "a" if args.append and out_path.exists() else "w"
    write_header = mode == "w" or not out_path.exists()

    existing_keys = set()
    if args.append and out_path.exists():
        with open(out_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(
                    (row.get("COID", ""), row.get("ratio_1", ""), row.get("Tg_C", ""))
                )

    with open(out_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        written = 0
        skipped = 0
        for row in all_rows:
            key = (row["COID"], row["ratio_1"], row["Tg_C"])
            if key in existing_keys:
                skipped += 1
                continue
            writer.writerow(row)
            written += 1

    print(f"\n写入: {out_path}")
    print(f"  新增: {written} 行")
    if skipped:
        print(f"  跳过（已存在）: {skipped} 行")


if __name__ == "__main__":
    main()
