"""
Chain Physics Feature Cache — SMILES-based lookup for precomputed 3D conformational features.

chain_physics features are expensive to compute (3D conformer generation + MMFF energy),
so they are precomputed and stored in a parquet file. This module provides a SMILES-indexed
cache for fast lookup during feature pipeline execution.

Public API:
    chain_physics_cache_names() -> list[str]  (8 feature names)
    chain_physics_cache_vector(smiles, cache) -> list[float]
    load_chain_physics_cache(parquet_path) -> dict[str, dict[str, float]]
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Same 8 features as chain_physics.py, but loaded from cache
FEATURE_NAMES = [
    "Neff_300K",
    "Neff_500K",
    "Neff_ratio",
    "conf_strain",
    "Cn_proxy",
    "curl_ratio",
    "curl_variance",
    "oligomer_level",
]

_NAN_VECTOR = [float("nan")] * len(FEATURE_NAMES)

# Singleton cache
_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def chain_physics_cache_names() -> List[str]:
    """Return chain physics feature names in fixed order."""
    return list(FEATURE_NAMES)


def load_chain_physics_cache(
    parquet_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Load chain_physics_features.parquet into SMILES-indexed dict.

    Args:
        parquet_path: Path to parquet. If None, auto-detect from project root.

    Returns:
        Dict mapping SMILES string -> {feature_name: float value}.
    """
    global _CACHE

    if parquet_path is None:
        parquet_path = str(
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "chain_physics_features.parquet"
        )

    p = Path(parquet_path)
    if not p.exists():
        return {}

    df = pd.read_parquet(parquet_path)
    if "smiles" not in df.columns:
        return {}

    cache: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        smi = str(row["smiles"])
        entry = {}
        for name in FEATURE_NAMES:
            if name in row.index:
                val = row[name]
                entry[name] = float(val) if pd.notna(val) else float("nan")
            else:
                entry[name] = float("nan")
        cache[smi] = entry

    _CACHE = cache
    return cache


def get_cache() -> Dict[str, Dict[str, float]]:
    """Get or lazily load the singleton cache."""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_chain_physics_cache()
    return _CACHE


def chain_physics_cache_vector(
    smiles: str,
    cache: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[float]:
    """Look up chain physics features for a SMILES string.

    Returns NaN vector if SMILES not found in cache.

    Args:
        smiles: Polymer repeat unit SMILES.
        cache: Preloaded cache dict. If None, uses singleton.

    Returns:
        List of 8 float values in FEATURE_NAMES order.
    """
    if cache is None:
        cache = get_cache()

    entry = cache.get(smiles)
    if entry is None:
        return list(_NAN_VECTOR)

    return [entry.get(name, float("nan")) for name in FEATURE_NAMES]
