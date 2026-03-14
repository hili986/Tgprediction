"""
BigSMILES Structural Fingerprints — Bridge BigSMILES to ML Feature Vectors
BigSMILES 结构指纹 — 将 BigSMILES 表示映射到机器学习特征向量

Provides:
    1. Morgan/ECFP fingerprints from repeat unit SMILES (via RDKit)
    2. Fragment counting (functional groups, bonding patterns)
    3. Polymer-level descriptors (heteroatom ratios, molecular weight proxy)
    4. Combined fingerprint vector for ML
    5. Simple Tg regression demo using Bicerano dataset

Public API / 公共 API:
    morgan_fingerprint(smiles, radius, n_bits)  — Morgan/ECFP bit vector
    fragment_counts(smiles)                     — functional group counts
    polymer_descriptors(smiles, bigsmiles)      — polymer-level features
    combined_fingerprint(smiles, bigsmiles)      — full feature vector
    tg_regression_demo()                         — linear regression on Bicerano data

Requires: RDKit (optional — functions raise ImportError with clear message if missing)
"""

from typing import List, Dict, Any, Optional, Tuple
import re

# RDKit imports with graceful degradation
_HAS_RDKIT = False
_HAS_MORGAN_GEN = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    _HAS_RDKIT = True
    try:
        from rdkit.Chem import rdFingerprintGenerator
        _HAS_MORGAN_GEN = True
    except ImportError:
        pass
except ImportError:
    pass


def _require_rdkit():
    """Raise ImportError if RDKit is not available."""
    if not _HAS_RDKIT:
        raise ImportError(
            "RDKit is required for fingerprint generation. "
            "Install via: conda install -c conda-forge rdkit"
        )


# ---------------------------------------------------------------------------
# 1. Morgan/ECFP fingerprints / Morgan 指纹
# ---------------------------------------------------------------------------

def morgan_fingerprint(smiles: str, radius: int = 2,
                       n_bits: int = 2048) -> List[int]:
    """Generate Morgan/ECFP fingerprint from repeat unit SMILES.
    从重复单元 SMILES 生成 Morgan/ECFP 指纹。

    Args:
        smiles: Repeat unit SMILES (with * as attachment points).
        radius: Morgan radius (2 = ECFP4, 3 = ECFP6).
        n_bits: Length of bit vector.

    Returns:
        List of 0/1 integers (length = n_bits).
    """
    _require_rdkit()
    # Replace * with [H] for RDKit compatibility
    clean = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if _HAS_MORGAN_GEN:
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        fp = gen.GetFingerprintAsNumPy(mol)
        return [int(x) for x in fp]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)


def morgan_fingerprint_counts(smiles: str, radius: int = 2,
                              n_bits: int = 2048) -> List[int]:
    """Generate Morgan fingerprint as count vector (not just bits).
    生成 Morgan 指纹计数向量。

    Returns:
        List of non-negative integers (length = n_bits).
    """
    _require_rdkit()
    clean = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if _HAS_MORGAN_GEN:
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        fp = gen.GetCountFingerprintAsNumPy(mol)
        return [int(x) for x in fp]
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    result = [0] * n_bits
    for idx, count in fp.GetNonzeroElements().items():
        result[idx % n_bits] = count
    return result


# ---------------------------------------------------------------------------
# 2. Fragment counting / 片段计数
# ---------------------------------------------------------------------------

# Functional group SMARTS patterns
_FRAGMENT_SMARTS = {
    "aromatic_ring":   "c1ccccc1",
    "carbonyl":        "[CX3]=[OX1]",
    "ester":           "[CX3](=[OX1])[OX2]",
    "amide":           "[CX3](=[OX1])[NX3]",
    "ether":           "[OD2]([#6])[#6]",
    "hydroxyl":        "[OX2H]",
    "amine":           "[NX3;H2,H1,H0;!$(NC=O)]",
    "nitrile":         "[CX2]#[NX1]",
    "halogen":         "[F,Cl,Br,I]",
    "fluorine":        "[F]",
    "chlorine":        "[Cl]",
    "double_bond":     "[CX3]=[CX3]",
    "silicon":         "[Si]",
    "sulfone":         "[SX4](=[OX1])(=[OX1])",
    "imide":           "[CX3](=[OX1])[NX3][CX3](=[OX1])",
}


def fragment_counts(smiles: str) -> Dict[str, int]:
    """Count functional group fragments in a repeat unit SMILES.
    统计重复单元 SMILES 中的官能团片段数。

    Args:
        smiles: Repeat unit SMILES (with * as attachment points).

    Returns:
        Dict mapping fragment name to count.
    """
    _require_rdkit()
    clean = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    counts = {}
    for name, smarts in _FRAGMENT_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            counts[name] = len(matches)
        else:
            counts[name] = 0
    return counts


def fragment_vector(smiles: str) -> List[int]:
    """Fragment counts as ordered vector.
    片段计数向量。

    Returns:
        List of integers in alphabetical fragment name order.
    """
    counts = fragment_counts(smiles)
    return [counts[k] for k in sorted(counts.keys())]


def fragment_names() -> List[str]:
    """Get fragment names in the same order as fragment_vector.
    获取片段名称。
    """
    return sorted(_FRAGMENT_SMARTS.keys())


# ---------------------------------------------------------------------------
# 3. Polymer-level descriptors / 聚合物级描述符
# ---------------------------------------------------------------------------

def polymer_descriptors(smiles: str,
                        bigsmiles: Optional[str] = None) -> Dict[str, float]:
    """Compute polymer-level descriptors from repeat unit SMILES and BigSMILES.
    从重复单元 SMILES 和 BigSMILES 计算聚合物级描述符。

    Descriptors:
        mw_repeat_unit:   Molecular weight of repeat unit
        num_heavy_atoms:  Number of non-hydrogen atoms
        num_rotatable:    Number of rotatable bonds
        c_fraction:       Carbon atom fraction
        o_fraction:       Oxygen atom fraction
        n_fraction:       Nitrogen atom fraction
        heteroatom_ratio: (O+N+S+F+Cl+Br+Si) / heavy atoms
        aromatic_fraction: Aromatic atoms / heavy atoms
        hbd_count:        Hydrogen bond donors
        hba_count:        Hydrogen bond acceptors
        tpsa:             Topological polar surface area
        logp:             Estimated logP (Wildman-Crippen)
        bonding_type:     0=$ (AA), 1=<> (AB), -1=unknown
        num_descriptors:  Number of bonding descriptors in BigSMILES
    """
    _require_rdkit()
    clean = smiles.replace('*', '[H]')
    mol = Chem.MolFromSmiles(clean)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        heavy = 1  # avoid division by zero

    # Atom counts
    atom_counts: Dict[str, int] = {}
    aromatic_count = 0
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
        if atom.GetIsAromatic():
            aromatic_count += 1

    c_count = atom_counts.get('C', 0)
    o_count = atom_counts.get('O', 0)
    n_count = atom_counts.get('N', 0)
    s_count = atom_counts.get('S', 0)
    f_count = atom_counts.get('F', 0)
    cl_count = atom_counts.get('Cl', 0)
    br_count = atom_counts.get('Br', 0)
    si_count = atom_counts.get('Si', 0)
    hetero = o_count + n_count + s_count + f_count + cl_count + br_count + si_count

    # Bonding type from BigSMILES
    bonding_type = -1
    num_bd = 0
    if bigsmiles:
        if '[$]' in bigsmiles:
            bonding_type = 0
        elif '[<]' in bigsmiles or '[>]' in bigsmiles:
            bonding_type = 1
        num_bd = len(re.findall(r'\[\$\]|\[<\]|\[>\]', bigsmiles))

    return {
        "mw_repeat_unit": round(Descriptors.MolWt(mol), 2),
        "num_heavy_atoms": heavy,
        "num_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "c_fraction": round(c_count / heavy, 4),
        "o_fraction": round(o_count / heavy, 4),
        "n_fraction": round(n_count / heavy, 4),
        "heteroatom_ratio": round(hetero / heavy, 4),
        "aromatic_fraction": round(aromatic_count / heavy, 4),
        "hbd_count": rdMolDescriptors.CalcNumHBD(mol),
        "hba_count": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 4),
        "bonding_type": bonding_type,
        "num_descriptors": num_bd,
    }


def descriptor_vector(smiles: str,
                      bigsmiles: Optional[str] = None) -> List[float]:
    """Polymer descriptors as ordered vector.
    聚合物描述符向量。

    Returns:
        List of floats in alphabetical descriptor name order.
    """
    desc = polymer_descriptors(smiles, bigsmiles)
    return [float(desc[k]) for k in sorted(desc.keys())]


def descriptor_names() -> List[str]:
    """Get descriptor names in the same order as descriptor_vector.
    获取描述符名称。
    """
    return sorted([
        "mw_repeat_unit", "num_heavy_atoms", "num_rotatable",
        "c_fraction", "o_fraction", "n_fraction",
        "heteroatom_ratio", "aromatic_fraction",
        "hbd_count", "hba_count", "tpsa", "logp",
        "bonding_type", "num_descriptors",
    ])


# ---------------------------------------------------------------------------
# 4. Combined fingerprint vector / 组合指纹向量
# ---------------------------------------------------------------------------

def combined_fingerprint(smiles: str,
                         bigsmiles: Optional[str] = None,
                         morgan_radius: int = 2,
                         morgan_bits: int = 1024) -> List[float]:
    """Build combined feature vector: Morgan bits + fragments + descriptors.
    构建组合特征向量: Morgan 位 + 片段计数 + 聚合物描述符。

    Returns:
        List of floats (length = morgan_bits + len(fragments) + len(descriptors)).
    """
    morgan = morgan_fingerprint(smiles, radius=morgan_radius, n_bits=morgan_bits)
    frags = fragment_vector(smiles)
    desc = descriptor_vector(smiles, bigsmiles)
    return [float(x) for x in morgan] + [float(x) for x in frags] + desc


def combined_feature_names(morgan_bits: int = 1024) -> List[str]:
    """Get feature names for combined_fingerprint.
    获取组合指纹的特征名称。
    """
    morgan_names = [f"morgan_{i}" for i in range(morgan_bits)]
    frag_names = [f"frag_{n}" for n in fragment_names()]
    desc = descriptor_names()
    return morgan_names + frag_names + desc


# ---------------------------------------------------------------------------
# 5. Tg regression demo / Tg 回归演示
# ---------------------------------------------------------------------------

def tg_regression_demo(use_morgan: bool = True,
                       use_fragments: bool = True,
                       use_descriptors: bool = True,
                       morgan_bits: int = 256,
                       verbose: bool = True) -> Dict[str, Any]:
    """Simple linear regression demo: predict Tg from structural fingerprints.
    简单线性回归演示: 从结构指纹预测 Tg。

    Uses Bicerano dataset (304 polymers) with sklearn-free ridge regression.
    No external ML library required — uses numpy-free manual implementation.

    Args:
        use_morgan: Include Morgan fingerprint bits.
        use_fragments: Include fragment counts.
        use_descriptors: Include polymer descriptors.
        morgan_bits: Number of Morgan fingerprint bits.
        verbose: Print results.

    Returns:
        Dict with r2, mae, rmse, num_features, num_samples.
    """
    _require_rdkit()

    from src.data.bicerano_tg_dataset import BICERANO_DATA

    # Build feature matrix and target vector
    X_rows = []
    y_vals = []
    skipped = 0

    for name, smiles, bigsmiles, tg_k in BICERANO_DATA:
        try:
            features = []
            if use_morgan:
                features.extend(
                    morgan_fingerprint(smiles, radius=2, n_bits=morgan_bits)
                )
            if use_fragments:
                features.extend(fragment_vector(smiles))
            if use_descriptors:
                features.extend(descriptor_vector(smiles, bigsmiles))
            X_rows.append(features)
            y_vals.append(float(tg_k))
        except (ValueError, Exception):
            skipped += 1

    if len(X_rows) < 10:
        raise RuntimeError(
            f"Too few valid samples ({len(X_rows)}). Need at least 10."
        )

    # Manual ridge regression (no numpy/sklearn required)
    n = len(X_rows)
    p = len(X_rows[0])

    # Split: 80% train, 20% test (deterministic split by index)
    split = int(n * 0.8)
    X_train, X_test = X_rows[:split], X_rows[split:]
    y_train, y_test = y_vals[:split], y_vals[split:]

    # Compute mean and std for normalization
    means = [0.0] * p
    for row in X_train:
        for j in range(p):
            means[j] += row[j]
    means = [m / len(X_train) for m in means]

    stds = [0.0] * p
    for row in X_train:
        for j in range(p):
            stds[j] += (row[j] - means[j]) ** 2
    stds = [(s / len(X_train)) ** 0.5 for s in stds]
    # Avoid division by zero
    stds = [s if s > 1e-10 else 1.0 for s in stds]

    # Normalize
    X_train_n = [
        [(row[j] - means[j]) / stds[j] for j in range(p)]
        for row in X_train
    ]
    X_test_n = [
        [(row[j] - means[j]) / stds[j] for j in range(p)]
        for row in X_test
    ]

    # Ridge regression: w = (X^T X + alpha I)^{-1} X^T y
    # Use gradient descent for simplicity (no matrix inversion needed)
    alpha = 0.1  # regularization
    lr = 0.01    # learning rate
    n_iter = 2000
    w = [0.0] * p
    b = sum(y_train) / len(y_train)  # bias initialized to mean

    for _ in range(n_iter):
        # Compute predictions
        grad_w = [0.0] * p
        grad_b = 0.0
        for i_sample in range(len(X_train_n)):
            pred = sum(w[j] * X_train_n[i_sample][j] for j in range(p)) + b
            err = pred - y_train[i_sample]
            for j in range(p):
                grad_w[j] += err * X_train_n[i_sample][j]
            grad_b += err

        # Update with L2 regularization
        n_train = len(X_train_n)
        w = [
            w[j] - lr * (grad_w[j] / n_train + alpha * w[j] / n_train)
            for j in range(p)
        ]
        b = b - lr * (grad_b / n_train)

    # Evaluate on test set
    y_pred = []
    for row in X_test_n:
        pred = sum(w[j] * row[j] for j in range(p)) + b
        y_pred.append(pred)

    # Metrics
    y_test_mean = sum(y_test) / len(y_test)
    ss_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
    ss_tot = sum((y_test[i] - y_test_mean) ** 2 for i in range(len(y_test)))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = sum(abs(y_test[i] - y_pred[i]) for i in range(len(y_test))) / len(y_test)
    rmse = (ss_res / len(y_test)) ** 0.5

    result = {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "num_features": p,
        "num_train": len(X_train),
        "num_test": len(X_test),
        "num_skipped": skipped,
        "features_used": {
            "morgan": use_morgan,
            "fragments": use_fragments,
            "descriptors": use_descriptors,
        },
    }

    if verbose:
        print("=== Tg Regression Demo ===")
        print(f"  Train: {result['num_train']}, Test: {result['num_test']}, "
              f"Skipped: {result['num_skipped']}")
        print(f"  Features: {result['num_features']}")
        print(f"  R2:   {result['r2']}")
        print(f"  MAE:  {result['mae']} K")
        print(f"  RMSE: {result['rmse']} K")

    return result


# ---------------------------------------------------------------------------
# CLI entry / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bigsmiles_fingerprint.py <smiles> [--morgan] [--fragments] [--descriptors] [--demo]")
        print("       python bigsmiles_fingerprint.py --demo")
        sys.exit(0)

    if "--demo" in sys.argv:
        tg_regression_demo(verbose=True)
        sys.exit(0)

    smiles_input = sys.argv[1]
    flags = set(sys.argv[2:])

    if "--morgan" in flags or not flags:
        fp = morgan_fingerprint(smiles_input)
        on_bits = sum(fp)
        print(f"Morgan fingerprint: {on_bits}/{len(fp)} bits on")

    if "--fragments" in flags or not flags:
        frags = fragment_counts(smiles_input)
        print(f"\nFragment counts:")
        for name, count in sorted(frags.items()):
            if count > 0:
                print(f"  {name}: {count}")

    if "--descriptors" in flags or not flags:
        desc = polymer_descriptors(smiles_input)
        print(f"\nPolymer descriptors:")
        for name, val in sorted(desc.items()):
            print(f"  {name}: {val}")
