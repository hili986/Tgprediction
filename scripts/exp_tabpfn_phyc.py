"""
TabPFN v2 on PHY-C-light 58d — 快速验证 CatBoost 天花板是否可突破

Usage:
    python scripts/exp_tabpfn_phyc.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.evaluation import nested_cv_no_tuning
from src.ml.sklearn_models import build_preprocessing
from src.features.feature_pipeline import get_feature_names

CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c"

# PHY-C-light: drop 6 junk/dupes
DROP = [
    "CP_oligomer_level", "CP_Cn_proxy",
    "L1_RingCount", "L1_HeavyAtomCount", "L1_MolWt",
    "IC_hydrophilic_ratio",
]


def main():
    # Load cached matrix
    if not CACHE.exists():
        print(f"ERROR: Cache not found: {CACHE}")
        sys.exit(1)

    df = pd.read_parquet(CACHE)
    all_names = get_feature_names("PHY-C")
    y = df["tg_k"].values
    X_full = df[all_names].values

    # Drop all-NaN rows
    valid = ~np.all(np.isnan(X_full), axis=1)
    X_full = X_full[valid]
    y = y[valid]

    # Apply light pruning
    keep_names = [f for f in all_names if f not in DROP]
    keep_idx = [all_names.index(f) for f in keep_names]
    X = X_full[:, keep_idx]

    # Preprocessing (TabPFN benefits from scaled input)
    pp = build_preprocessing()
    X_pp = pp.fit_transform(X)

    print(f"PHY-C-light 58d: {X_pp.shape[0]} samples, {X_pp.shape[1]} features")

    # TabPFN
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("TabPFN 未安装。运行: pip install tabpfn")
        sys.exit(1)

    estimator = TabPFNRegressor()

    print(f"\nRunning TabPFN v2 Nested CV (5x3 folds)...")
    result = nested_cv_no_tuning(
        X_pp, y, estimator,
        outer_splits=5, outer_repeats=3, verbose=True,
    )

    metrics = result["metrics"]
    print(f"\n{'='*60}")
    print(f"  TabPFN v2 + PHY-C-light (58d) Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    print(f"{'='*60}")

    # Compare with CatBoost results
    print(f"\n  Comparison:")
    print(f"  {'Model + Features':<35s} {'R2':>12s} {'MAE':>10s}")
    print(f"  {'-'*60}")

    refs = [
        ("CatBoost PHY-B2 56d", "ablation_PHY-B2.json"),
        ("CatBoost PHY-C-light 58d", "select_PHY-C-light-v2.json"),
        ("CatBoost PHY-C-full 64d", "select_PHY-C-full.json"),
    ]
    for label, fname in refs:
        p = RESULT_DIR / fname
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            m = d["metrics"]
            print(f"  {label:<35s} {m['R2_mean']:.4f}+-{m['R2_std']:.4f} {m['MAE_mean']:.1f}+-{m['MAE_std']:.1f}K")

    print(f"  {'TabPFN PHY-C-light 58d':<35s} {metrics['R2_mean']:.4f}+-{metrics['R2_std']:.4f} {metrics['MAE_mean']:.1f}+-{metrics['MAE_std']:.1f}K")

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "tabpfn_PHY-C-light.json"
    save_data = {
        "experiment": "TabPFN_PHY-C-light",
        "model": "TabPFN_v2",
        "n_features": len(keep_names),
        "n_samples": len(y),
        "feature_names": keep_names,
        "dropped": DROP,
        "metrics": metrics,
        "outer_cv": "5x3",
    }
    with open(out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
