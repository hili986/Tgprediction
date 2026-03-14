"""
Copolymer Tg Virtual Data Generator
共聚物 Tg 虚拟数据生成器

Equations (increasing fidelity):
    F0: Fox equation — simplest, no interaction parameter
    F1: Gordon-Taylor — volume effect parameter k
    F2: Kwei — hydrogen bond contribution q (Phase 3)

Data generation strategy:
    C(304,2) = 46,056 monomer pairs
    × 9 weight fractions = ~414,504 entries
    Sampling: diversity-based down to 10,000-50,000

Source: 虚拟合成数据增强Tg预测调研.md

Public API:
    fox_equation(tg1, tg2, w1) -> float
    gordon_taylor(tg1, tg2, w1, k) -> float
    estimate_k(tg1, tg2) -> float
    generate_copolymer_data(homopolymers, ...) -> list of dicts
    load_homopolymers() -> list of dicts from Bicerano dataset
"""

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np

from src.data.bicerano_tg_dataset import BICERANO_DATA


# ---------------------------------------------------------------------------
# 1. Mixing rules / 混合方程
# ---------------------------------------------------------------------------

def fox_equation(tg1: float, tg2: float, w1: float) -> float:
    """Fox equation: 1/Tg = w1/Tg1 + w2/Tg2.

    Args:
        tg1: Tg of polymer 1 (K).
        tg2: Tg of polymer 2 (K).
        w1: Weight fraction of polymer 1 (0-1).

    Returns:
        Predicted copolymer Tg (K), or NaN if invalid.
    """
    w2 = 1.0 - w1
    if tg1 <= 0 or tg2 <= 0:
        return float("nan")
    return 1.0 / (w1 / tg1 + w2 / tg2)


def gordon_taylor(tg1: float, tg2: float, w1: float, k: float = 1.0) -> float:
    """Gordon-Taylor equation: Tg = (w1*Tg1 + k*w2*Tg2) / (w1 + k*w2).

    Args:
        tg1: Tg of polymer 1 (K).
        tg2: Tg of polymer 2 (K).
        w1: Weight fraction of polymer 1.
        k: Interaction parameter (typically 0.5-2.0).

    Returns:
        Predicted copolymer Tg (K), or NaN if invalid.
    """
    w2 = 1.0 - w1
    denom = w1 + k * w2
    if abs(denom) < 1e-10:
        return float("nan")
    return (w1 * tg1 + k * w2 * tg2) / denom


def estimate_k(tg1: float, tg2: float) -> float:
    """Estimate k parameter using Simha-Boyer rule: k ~ Tg1/Tg2.

    Args:
        tg1: Tg of polymer 1 (K).
        tg2: Tg of polymer 2 (K).

    Returns:
        Estimated k value, clamped to [0.3, 3.0].
    """
    if tg2 <= 0:
        return 1.0
    k = tg1 / tg2
    return max(0.3, min(3.0, k))


# ---------------------------------------------------------------------------
# 2. Data loading / 数据加载
# ---------------------------------------------------------------------------

def load_homopolymers() -> List[Dict]:
    """Load homopolymer data from Bicerano dataset.

    Returns:
        List of dicts with keys: name, smiles, bigsmiles, tg.
    """
    result = []
    for name, smiles, bigsmiles, tg_k in BICERANO_DATA:
        if tg_k > 0:
            result.append({
                "name": name,
                "smiles": smiles,
                "bigsmiles": bigsmiles,
                "tg": float(tg_k),
            })
    return result


# ---------------------------------------------------------------------------
# 3. Copolymer data generation / 共聚物数据生成
# ---------------------------------------------------------------------------

def generate_copolymer_data(
    homopolymers: Optional[List[Dict]] = None,
    weight_fractions: Optional[List[float]] = None,
    max_samples: int = 50000,
    fidelity: str = "F0",
    random_state: int = 42,
) -> List[Dict]:
    """Generate virtual copolymer Tg data using mixing rules.

    Args:
        homopolymers: List of dicts with 'smiles', 'tg', 'name' keys.
            If None, loads from Bicerano dataset.
        weight_fractions: List of w1 values to generate.
            Default: [0.1, 0.2, ..., 0.9]
        max_samples: Maximum number of virtual samples.
        fidelity: 'F0' (Fox) or 'F1' (Gordon-Taylor).
        random_state: Random seed for sampling.

    Returns:
        List of dicts with keys:
            smiles1, smiles2, bigsmiles1, bigsmiles2,
            name1, name2, w1, tg_virtual, fidelity
    """
    if homopolymers is None:
        homopolymers = load_homopolymers()

    if weight_fractions is None:
        weight_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    n_poly = len(homopolymers)
    pairs = list(combinations(range(n_poly), 2))
    total_possible = len(pairs) * len(weight_fractions)

    # If too many, randomly sample pairs
    rng = np.random.RandomState(random_state)
    if total_possible > max_samples:
        n_pairs_needed = max_samples // len(weight_fractions)
        indices = rng.choice(len(pairs), size=n_pairs_needed, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]

    results: List[Dict] = []

    for i, j in pairs:
        p1, p2 = homopolymers[i], homopolymers[j]
        tg1, tg2 = p1["tg"], p2["tg"]

        for w1 in weight_fractions:
            if fidelity == "F0":
                tg_virtual = fox_equation(tg1, tg2, w1)
            elif fidelity == "F1":
                k = estimate_k(tg1, tg2)
                tg_virtual = gordon_taylor(tg1, tg2, w1, k)
            else:
                raise ValueError(f"Unknown fidelity: {fidelity}")

            if not np.isnan(tg_virtual) and tg_virtual > 0:
                results.append({
                    "smiles1": p1["smiles"],
                    "smiles2": p2["smiles"],
                    "bigsmiles1": p1.get("bigsmiles", ""),
                    "bigsmiles2": p2.get("bigsmiles", ""),
                    "name1": p1.get("name", ""),
                    "name2": p2.get("name", ""),
                    "w1": w1,
                    "tg_virtual": round(tg_virtual, 2),
                    "fidelity": fidelity,
                })

    return results


# ---------------------------------------------------------------------------
# 4. Copolymer feature computation / 共聚物特征计算
# ---------------------------------------------------------------------------

def compute_copolymer_features(
    smiles1: str,
    smiles2: str,
    w1: float,
    bigsmiles1: str = "",
    bigsmiles2: str = "",
    layer: str = "L1",
    morgan_bits: int = 1024,
) -> np.ndarray:
    """Compute weight-averaged features for a copolymer.

    For physical/chemical descriptors: feature = w1*f(s1) + w2*f(s2)
    For binary fingerprints (Morgan): bitwise OR then scale by max(w1,w2)

    Args:
        smiles1, smiles2: SMILES of the two monomers.
        w1: Weight fraction of monomer 1.
        bigsmiles1, bigsmiles2: BigSMILES strings.
        layer: Feature layer ('L0', 'L1', 'L2').
        morgan_bits: Morgan fingerprint bits.

    Returns:
        1D numpy array of copolymer features.
    """
    from src.features.feature_pipeline import compute_features, LAYER_COMPONENTS

    w2 = 1.0 - w1

    f1 = compute_features(smiles1, bigsmiles1, layer, morgan_bits)
    f2 = compute_features(smiles2, bigsmiles2, layer, morgan_bits)

    if np.any(np.isnan(f1)) or np.any(np.isnan(f2)):
        return np.full_like(f1, np.nan)

    components = LAYER_COMPONENTS[layer]

    # Determine split points for different feature types
    # L0 Afsordeh: 4, L1 RDKit: 15, Morgan: morgan_bits, Fragments: 15, Poly desc: 14
    idx = 0
    result_parts = []

    if "afsordeh" in components:
        n = 4
        result_parts.append(w1 * f1[idx:idx + n] + w2 * f2[idx:idx + n])
        idx += n

    if "rdkit_2d" in components:
        n = 15
        result_parts.append(w1 * f1[idx:idx + n] + w2 * f2[idx:idx + n])
        idx += n

    if "morgan" in components:
        n = morgan_bits
        # For binary fingerprints: use weighted average
        result_parts.append(w1 * f1[idx:idx + n] + w2 * f2[idx:idx + n])
        idx += n

    if "fragments" in components:
        n = 15
        result_parts.append(w1 * f1[idx:idx + n] + w2 * f2[idx:idx + n])
        idx += n

    if "polymer_desc" in components:
        n = len(f1) - idx
        result_parts.append(w1 * f1[idx:idx + n] + w2 * f2[idx:idx + n])

    return np.concatenate(result_parts)


def build_copolymer_dataset(
    copolymer_data: Optional[List[Dict]] = None,
    layer: str = "L1",
    morgan_bits: int = 1024,
    max_samples: int = 50000,
    fidelity: str = "F0",
    verbose: bool = True,
) -> tuple:
    """Build feature matrix for virtual copolymer data.

    Args:
        copolymer_data: Pre-generated copolymer data (or None to generate).
        layer: Feature layer.
        morgan_bits: Morgan fingerprint bits.
        max_samples: Max samples if generating new data.
        fidelity: Mixing rule fidelity.
        verbose: Print progress.

    Returns:
        (X, y, metadata) where:
            X: np.ndarray (n_samples, n_features)
            y: np.ndarray (n_samples,) — virtual Tg
            metadata: list of dicts with copolymer info
    """
    if copolymer_data is None:
        if verbose:
            print("  Generating copolymer data...")
        copolymer_data = generate_copolymer_data(
            max_samples=max_samples, fidelity=fidelity,
        )

    if verbose:
        print(f"  Computing features for {len(copolymer_data)} copolymers...")

    X_list = []
    y_list = []
    meta_list = []
    skipped = 0

    for i, entry in enumerate(copolymer_data):
        try:
            x = compute_copolymer_features(
                entry["smiles1"], entry["smiles2"], entry["w1"],
                entry.get("bigsmiles1", ""), entry.get("bigsmiles2", ""),
                layer=layer, morgan_bits=morgan_bits,
            )
            if np.any(np.isnan(x)):
                skipped += 1
                continue
            X_list.append(x)
            y_list.append(entry["tg_virtual"])
            meta_list.append(entry)
        except Exception:
            skipped += 1

        if verbose and (i + 1) % 5000 == 0:
            print(f"    Processed {i + 1}/{len(copolymer_data)}...")

    X = np.array(X_list)
    y = np.array(y_list)

    if verbose:
        print(f"  Copolymer dataset [{layer}]: {X.shape[0]} samples, {X.shape[1]} features")
        if skipped > 0:
            print(f"  Skipped: {skipped} entries")

    return X, y, meta_list
