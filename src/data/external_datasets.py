"""
External Dataset Loader — Unified loading for polymer Tg datasets.

Loads, normalizes, deduplicates, and merges external Tg datasets.
Uses canonical SMILES deduplication via RDKit for true molecular identity.

Data source priority (high → low):
    1. PolyMetriX (7,367 entries, 62.1% exclusive, reliability graded)
    2. NeurIPS OPP 2025 (17,330 raw → ~14,000 after quality filter)
       Superset of: POINT2, PolyInfo, Qiu_Polymer, Qiu_PI, JCIM_662
    3. Conjugated Polymer 32 (32 entries, from Polymer_Tg_ repo)
    4. OpenPoly (443 entries, LOW reliability — 84.3% of cross-source conflicts)

Copolymer sources (separate API):
    - Pilania PHA 2019 (131 entries: 104 copolymers + 27 homopolymers)
    - Kuenneth PolyInfo (reserved — requires NIMS account)

Removed sources (100% contained in NeurIPS OPP):
    - POINT2 (≡ PolyInfo, 100% bidirectional overlap)
    - Qiu_Polymer (314 entries, 100% ⊂ NeurIPS)
    - Qiu_PI (372 entries, 100% ⊂ NeurIPS)

Public API:
    load_all_external(...)        -> List[Dict]  (homopolymers)
    load_copolymer_data(...)      -> List[Dict]  (copolymers)
    build_extended_dataset(...)   -> (X, y, names, feature_names)
"""

import csv
import re
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "external"


# ---------------------------------------------------------------------------
# SMILES normalization and canonicalization
# ---------------------------------------------------------------------------

def _normalize_psmiles(psmiles: str) -> str:
    """Normalize PSMILES: strip whitespace, ensure bare * → [*]."""
    s = psmiles.strip()
    # Convert bare * to [*] for RDKit compatibility
    s = re.sub(r'(?<!\[)\*(?!\])', '[*]', s)
    return s


def _canonical_smiles(psmiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES, or None if parse fails.

    Uses [*] notation for polymer end-groups.
    """
    if not HAS_RDKIT:
        return psmiles  # fallback: raw string comparison

    normalized = _normalize_psmiles(psmiles)
    mol = Chem.MolFromSmiles(normalized)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _to_star_format(psmiles: str) -> str:
    """Convert [*] back to bare * for compatibility with feature pipeline."""
    return psmiles.replace("[*]", "*")


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_csv(path: Path, encoding: str = "utf-8-sig") -> List[Dict[str, str]]:
    """Read CSV file with flexible encoding."""
    try:
        with open(path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f)
            return list(reader)
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            reader = csv.DictReader(f)
            return list(reader)


# ---------------------------------------------------------------------------
# NeurIPS OPP quality filters
# ---------------------------------------------------------------------------

def _is_polymer_smiles(smiles: str) -> bool:
    """Check if SMILES contains polymer end-group markers (* or [*])."""
    return "*" in smiles


def _has_predicted_tg(tg_str: str) -> bool:
    """Detect likely ML-predicted Tg values (>2 decimal places).

    Experimental Tg is typically reported as integer or 1 decimal.
    Values like 123.456789 suggest ML model predictions mixed into data.
    """
    try:
        val = float(tg_str)
    except (ValueError, TypeError):
        return False

    # Check decimal places
    if "." in tg_str:
        decimals = len(tg_str.rstrip("0").split(".")[-1])
        return decimals > 2
    return False


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def _load_polymetrix() -> List[Dict]:
    """Load PolyMetriX dataset (7,367 entries with reliability grading).

    62.1% exclusive molecules not found in other sources.
    """
    path = DATA_DIR / "polymetrix_tg.csv"
    if not path.exists():
        return []

    rows = _read_csv(path)
    results = []
    for r in rows:
        smiles = r.get("PSMILES", "").strip()
        tg_k_str = r.get("Tg_K", "")
        if not smiles or not tg_k_str:
            continue
        try:
            tg_k = float(tg_k_str)
        except ValueError:
            continue

        results.append({
            "smiles": _to_star_format(_normalize_psmiles(smiles)),
            "tg_k": tg_k,
            "source": "polymetrix",
            "reliability": r.get("reliability", ""),
            "polymer_class": r.get("polymer_class", ""),
            "name": r.get("polymer_name", ""),
        })
    return results


def _load_neurips_opp(quality_filter: bool = True) -> List[Dict]:
    """Load NeurIPS OPP 2025 dataset with quality filtering.

    Raw: 17,330 entries (Tg in Celsius).
    Quality issues found in investigation:
        - 6.2% (1,072) are non-polymer small molecules (no * markers)
        - 11.1% (1,920) have >2 decimal places (likely ML predictions)
        - 70 entries with Tg < 100K (physically implausible)

    Args:
        quality_filter: If True, remove non-polymer and predicted entries.
    """
    path = DATA_DIR / "neurips_opp_tg.csv"
    if not path.exists():
        return []

    rows = _read_csv(path)
    results = []
    filtered_counts = {"non_polymer": 0, "predicted": 0, "parse_fail": 0}

    for r in rows:
        smiles_raw = r.get("SMILES", "").strip()
        tg_c_str = r.get("Tg", "")
        if not smiles_raw or not tg_c_str:
            continue

        if quality_filter:
            if not _is_polymer_smiles(smiles_raw):
                filtered_counts["non_polymer"] += 1
                continue
            if _has_predicted_tg(tg_c_str):
                filtered_counts["predicted"] += 1
                continue

        try:
            tg_c = float(tg_c_str)
            tg_k = tg_c + 273.15
        except ValueError:
            filtered_counts["parse_fail"] += 1
            continue

        results.append({
            "smiles": _to_star_format(_normalize_psmiles(smiles_raw)),
            "tg_k": tg_k,
            "source": "neurips_opp",
            "reliability": "",
            "polymer_class": "",
            "name": "",
        })
    return results


def _load_openpoly() -> List[Dict]:
    """Load OpenPoly dataset (443 Tg entries, LOW reliability).

    WARNING: 84.3% of cross-source Tg conflicts involve OpenPoly.
    Internal contradictions up to 321K observed.
    Loaded with reliability='low' flag.
    """
    path = DATA_DIR / "openpoly_properties.csv"
    if not path.exists():
        return []

    rows = _read_csv(path)
    results = []
    for r in rows:
        smiles = r.get("PSMILES", "").strip()
        tg_k_str = r.get("Tg (K)", "")
        if not smiles or not tg_k_str or not tg_k_str.strip():
            continue
        try:
            tg_k = float(tg_k_str)
        except ValueError:
            continue

        results.append({
            "smiles": _to_star_format(_normalize_psmiles(smiles)),
            "tg_k": tg_k,
            "source": "openpoly",
            "reliability": "low",
            "polymer_class": "",
            "name": r.get("Name", ""),
        })
    return results


def _load_conjugated_polymer() -> List[Dict]:
    """Load conjugated polymer dataset (32 entries, Tg in Celsius).

    From figotj/Polymer_Tg_ repo, 32 conjugated polymers.
    Tg range: -30°C to 215°C (243K to 488K).
    ~88% overlap with PolyMetriX, ~4 exclusive molecules.
    """
    path = DATA_DIR / "conjugated_polymer_tg.csv"
    if not path.exists():
        return []

    rows = _read_csv(path)
    results = []
    for r in rows:
        smiles = r.get("psmiles", "").strip()
        tg_k_str = r.get("tg_k", "")
        if not smiles or not tg_k_str:
            continue
        try:
            tg_k = float(tg_k_str)
        except ValueError:
            continue

        results.append({
            "smiles": _to_star_format(_normalize_psmiles(smiles)),
            "tg_k": tg_k,
            "source": "conjugated_32",
            "reliability": "",
            "polymer_class": "Conjugated",
            "name": "",
        })
    return results


# ---------------------------------------------------------------------------
# Copolymer loaders
# ---------------------------------------------------------------------------

def _load_pilania_pha() -> List[Dict]:
    """Load Pilania PHA copolymer dataset (131 entries).

    104 copolymers + 27 homopolymers, poly-hydroxyalkanoate system.
    Tg range: 218.2-288.9K. Source: Pilania et al. 2019.

    Returns dicts with keys: smiles_1, smiles_2, monomer_ratio, type,
    tg_k, source, is_copolymer.
    """
    path = DATA_DIR / "pilania_pha_tg.csv"
    if not path.exists():
        return []

    rows = _read_csv(path)
    results = []
    for r in rows:
        smiles_1 = r.get("smiles_1", "").strip()
        smiles_2 = r.get("smiles_2", "").strip()
        ratio_str = r.get("monomer_ratio", "")
        tg_k_str = r.get("tg_k", "")
        poly_type = r.get("type", "R")

        if not smiles_1 or not tg_k_str:
            continue
        try:
            tg_k = float(tg_k_str)
            ratio = float(ratio_str) if ratio_str else 100.0
        except ValueError:
            continue

        is_copolymer = ratio < 100.0 and smiles_2 and smiles_2 != "C"

        results.append({
            "smiles_1": _to_star_format(_normalize_psmiles(smiles_1)),
            "smiles_2": smiles_2 if not is_copolymer else _to_star_format(
                _normalize_psmiles(smiles_2)),
            "monomer_ratio": ratio,
            "type": poly_type,
            "tg_k": tg_k,
            "source": "pilania_pha",
            "is_copolymer": is_copolymer,
        })
    return results


def _load_kuenneth_copolymer() -> List[Dict]:
    """Load Kuenneth copolymer dataset (RESERVED — not yet available).

    Requires NIMS PolyInfo academic account registration.
    Expected: 7,774 copolymer Tg entries with two SMILES + ratio.
    Paper: Kuenneth & Ramprasad, Macromolecules 2021.

    Returns empty list until data is obtained.
    """
    path = DATA_DIR / "kuenneth_copolymer_tg.csv"
    if not path.exists():
        return []

    # TODO: Implement when user obtains PolyInfo data
    # Expected columns: SMILES_1, SMILES_2, ratio, Tg_K, type (R/B)
    rows = _read_csv(path)
    results = []
    for r in rows:
        smiles_1 = r.get("SMILES_1", "").strip()
        smiles_2 = r.get("SMILES_2", "").strip()
        ratio_str = r.get("ratio", "")
        tg_k_str = r.get("Tg_K", "")

        if not smiles_1 or not smiles_2 or not tg_k_str:
            continue
        try:
            tg_k = float(tg_k_str)
            ratio = float(ratio_str) if ratio_str else 50.0
        except ValueError:
            continue

        results.append({
            "smiles_1": _to_star_format(_normalize_psmiles(smiles_1)),
            "smiles_2": _to_star_format(_normalize_psmiles(smiles_2)),
            "monomer_ratio": ratio,
            "type": r.get("type", "R"),
            "tg_k": tg_k,
            "source": "kuenneth_copolymer",
            "is_copolymer": True,
        })
    return results


# ---------------------------------------------------------------------------
# Deduplication and conflict resolution
# ---------------------------------------------------------------------------

def _canonical_dedup(
    data: List[Dict],
    resolve_conflicts: str = "median",
    verbose: bool = True,
) -> List[Dict]:
    """Deduplicate by canonical SMILES with Tg conflict resolution.

    Args:
        data: List of dicts with 'smiles' and 'tg_k' keys.
        resolve_conflicts: How to handle Tg conflicts for same molecule.
            'median' — take median Tg across sources.
            'first' — keep first occurrence (priority by load order).
            'exclude_openpoly_median' — median excluding OpenPoly values
                (recommended, since 84.3% of conflicts involve OpenPoly).
        verbose: Print dedup statistics.

    Returns:
        Deduplicated list with one entry per canonical SMILES.
    """
    # Group by canonical SMILES
    groups: Dict[str, List[Dict]] = {}
    canon_fail = 0

    for d in data:
        canon = _canonical_smiles(d["smiles"])
        if canon is None:
            canon_fail += 1
            canon = d["smiles"]  # fallback to raw string
        if canon not in groups:
            groups[canon] = []
        groups[canon].append(d)

    if verbose and canon_fail > 0:
        print(f"  Canonicalization failed for {canon_fail} entries "
              f"(using raw SMILES)")

    # Resolve each group
    results = []
    conflict_count = 0

    for canon_smi, entries in groups.items():
        if len(entries) == 1:
            results.append(entries[0])
            continue

        # Multiple entries for same molecule
        tg_values = [e["tg_k"] for e in entries]

        if resolve_conflicts == "median":
            resolved_tg = median(tg_values)
        elif resolve_conflicts == "exclude_openpoly_median":
            non_openpoly = [e["tg_k"] for e in entries
                           if e["source"] != "openpoly"]
            resolved_tg = median(non_openpoly) if non_openpoly else median(
                tg_values)
        else:  # 'first'
            resolved_tg = entries[0]["tg_k"]

        # Check if there's a real conflict (>10K difference)
        tg_range = max(tg_values) - min(tg_values)
        if tg_range > 10.0:
            conflict_count += 1

        # Keep first entry's metadata, update Tg
        best = entries[0].copy()
        best["tg_k"] = resolved_tg
        if len(set(e["source"] for e in entries)) > 1:
            best["source"] = "+".join(
                sorted(set(e["source"] for e in entries)))
        results.append(best)

    if verbose:
        print(f"  Canonical dedup: {len(data)} -> {len(results)} "
              f"({len(data) - len(results)} duplicates merged)")
        if conflict_count > 0:
            print(f"  Tg conflicts (>10K): {conflict_count} molecules")

    return results


# ---------------------------------------------------------------------------
# Public API — Homopolymer data
# ---------------------------------------------------------------------------

HOMOPOLYMER_LOADERS = {
    "polymetrix": _load_polymetrix,
    "neurips_opp": _load_neurips_opp,
    "conjugated_32": _load_conjugated_polymer,
    "openpoly": _load_openpoly,
}

# Default: exclude OpenPoly due to low reliability
DEFAULT_SOURCES = ["polymetrix", "neurips_opp", "conjugated_32"]


def load_all_external(
    sources: Optional[List[str]] = None,
    min_tg_k: float = 100.0,
    max_tg_k: float = 900.0,
    deduplicate: bool = True,
    conflict_resolution: str = "exclude_openpoly_median",
    neurips_quality_filter: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """Load and merge external homopolymer Tg datasets.

    Args:
        sources: Source names to load. None = DEFAULT_SOURCES
            (excludes OpenPoly by default due to low reliability).
            Use sources=list(HOMOPOLYMER_LOADERS.keys()) for all sources.
        min_tg_k: Minimum Tg in Kelvin (filter outliers).
        max_tg_k: Maximum Tg in Kelvin (filter outliers).
        deduplicate: If True, canonical SMILES dedup with conflict resolution.
        conflict_resolution: 'median', 'first', or 'exclude_openpoly_median'.
        neurips_quality_filter: Apply NeurIPS quality filters.
        verbose: Print loading statistics.

    Returns:
        List of dicts with keys: smiles, tg_k, source, reliability,
        polymer_class, name.
    """
    if sources is None:
        sources = list(DEFAULT_SOURCES)

    all_data: List[Dict] = []
    for src in sources:
        if src not in HOMOPOLYMER_LOADERS:
            if verbose:
                print(f"  WARNING: Unknown source '{src}', skipping")
            continue

        if src == "neurips_opp":
            entries = _load_neurips_opp(quality_filter=neurips_quality_filter)
        else:
            entries = HOMOPOLYMER_LOADERS[src]()
        if verbose:
            print(f"  Loaded {src}: {len(entries)} entries")
        all_data.extend(entries)

    if verbose:
        print(f"  Total raw: {len(all_data)}")

    # Filter by Tg range
    before = len(all_data)
    all_data = [d for d in all_data if min_tg_k <= d["tg_k"] <= max_tg_k]
    if verbose and before != len(all_data):
        print(f"  Filtered Tg range [{min_tg_k}, {max_tg_k}]K: "
              f"{before} -> {len(all_data)}")

    # Canonical SMILES deduplication with Tg conflict resolution
    if deduplicate:
        all_data = _canonical_dedup(
            all_data,
            resolve_conflicts=conflict_resolution,
            verbose=verbose,
        )

    return all_data


# ---------------------------------------------------------------------------
# Public API — Copolymer data
# ---------------------------------------------------------------------------

COPOLYMER_LOADERS = {
    "pilania_pha": _load_pilania_pha,
    "kuenneth_copolymer": _load_kuenneth_copolymer,
}


def load_copolymer_data(
    sources: Optional[List[str]] = None,
    min_tg_k: float = 100.0,
    max_tg_k: float = 900.0,
    verbose: bool = True,
) -> List[Dict]:
    """Load copolymer Tg datasets (separate from homopolymers).

    Copolymer entries have two monomer SMILES + ratio, incompatible with
    the standard single-SMILES feature pipeline.

    Args:
        sources: Copolymer source names. None = all available.
        min_tg_k: Minimum Tg filter.
        max_tg_k: Maximum Tg filter.
        verbose: Print statistics.

    Returns:
        List of dicts with keys: smiles_1, smiles_2, monomer_ratio,
        type (R/B), tg_k, source, is_copolymer.
    """
    if sources is None:
        sources = list(COPOLYMER_LOADERS.keys())

    all_data: List[Dict] = []
    for src in sources:
        if src not in COPOLYMER_LOADERS:
            if verbose:
                print(f"  WARNING: Unknown copolymer source '{src}', skipping")
            continue
        entries = COPOLYMER_LOADERS[src]()
        if verbose:
            print(f"  Loaded {src}: {len(entries)} entries")
        all_data.extend(entries)

    # Filter by Tg range
    before = len(all_data)
    all_data = [d for d in all_data if min_tg_k <= d["tg_k"] <= max_tg_k]
    if verbose and before != len(all_data):
        print(f"  Filtered Tg range [{min_tg_k}, {max_tg_k}]K: "
              f"{before} -> {len(all_data)}")

    if verbose:
        n_copoly = sum(1 for d in all_data if d.get("is_copolymer", False))
        n_homo = len(all_data) - n_copoly
        print(f"  Copolymers: {n_copoly}, Homopolymers: {n_homo}")

    return all_data


# ---------------------------------------------------------------------------
# Public API — Feature matrix builder (homopolymers only)
# ---------------------------------------------------------------------------

def build_extended_dataset(
    layer: str = "L1",
    morgan_bits: int = 1024,
    sources: Optional[List[str]] = None,
    min_tg_k: float = 100.0,
    max_tg_k: float = 900.0,
    include_bicerano: bool = True,
    include_openpoly: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Build extended feature matrix from external datasets + Bicerano.

    Same return format as build_dataset_v2() for drop-in compatibility.

    Args:
        layer: Feature layer ('L0', 'L1', 'L1H', 'L2', 'L2H').
        morgan_bits: Number of Morgan fingerprint bits (for L2).
        sources: External sources to include. None = DEFAULT_SOURCES.
        min_tg_k: Minimum Tg filter.
        max_tg_k: Maximum Tg filter.
        include_bicerano: Whether to include the original Bicerano dataset.
        include_openpoly: Whether to include OpenPoly (excluded by default
            due to 84.3% cross-source Tg conflict rate).
        verbose: Print dataset info.

    Returns:
        (X, y, names, feature_names)
    """
    from src.features.feature_pipeline import (
        compute_features,
        get_feature_names,
    )

    feat_names = get_feature_names(layer, morgan_bits)

    # Determine sources
    effective_sources = list(sources) if sources else list(DEFAULT_SOURCES)
    if include_openpoly and "openpoly" not in effective_sources:
        effective_sources.append("openpoly")

    # Collect all entries
    entries: List[Tuple[str, str, float]] = []  # (name, smiles, tg_k)

    # Load Bicerano first (highest priority for dedup)
    bicerano_canonical = set()
    if include_bicerano:
        from src.data.bicerano_tg_dataset import BICERANO_DATA
        for name, smiles, _, tg_k in BICERANO_DATA:
            tg_val = float(tg_k)
            if not (min_tg_k <= tg_val <= max_tg_k):
                continue
            entries.append((name, smiles, tg_val))
            canon = _canonical_smiles(smiles)
            if canon:
                bicerano_canonical.add(canon)
        if verbose:
            print(f"  Bicerano: {len(BICERANO_DATA)} entries")

    # Load external data
    external = load_all_external(
        sources=effective_sources,
        min_tg_k=min_tg_k,
        max_tg_k=max_tg_k,
        deduplicate=True,
        verbose=verbose,
    )

    # Deduplicate external against Bicerano using canonical SMILES
    added = 0
    for d in external:
        canon = _canonical_smiles(d["smiles"])
        if canon and canon in bicerano_canonical:
            continue
        if canon:
            bicerano_canonical.add(canon)
        label = d["name"] or f"{d['source']}_{added}"
        entries.append((label, d["smiles"], d["tg_k"]))
        added += 1

    if verbose:
        print(f"  External (new): {added} entries")
        print(f"  Total entries: {len(entries)}")

    # Compute features
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    names: List[str] = []
    skipped = 0

    for name, smiles, tg_k in entries:
        try:
            x = compute_features(smiles, layer=layer, morgan_bits=morgan_bits)
            if np.any(np.isnan(x)):
                skipped += 1
                continue
            X_list.append(x)
            y_list.append(tg_k)
            names.append(name)
        except Exception as e:
            skipped += 1
            if verbose and skipped <= 3:
                print(f"  WARNING: Feature extraction failed for "
                      f"'{smiles[:60]}': {e}")

    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
    else:
        X = np.empty((0, len(feat_names)))
        y = np.array([])

    if verbose:
        print(f"  Dataset [{layer}]: {X.shape[0]} samples, "
              f"{X.shape[1]} features")
        if skipped > 0:
            print(f"  Skipped: {skipped} polymers (feature extraction errors)")

    return X, y, names, feat_names


# ---------------------------------------------------------------------------
# Backward compatibility — LOADERS dict for legacy code
# ---------------------------------------------------------------------------

# Backward compatibility — NOT used by load_all_external() internally.
# Legacy code that patches LOADERS will NOT affect load_all_external().
LOADERS = {**HOMOPOLYMER_LOADERS, **COPOLYMER_LOADERS}
