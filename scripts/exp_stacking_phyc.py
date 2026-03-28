"""
Stacking (TabPFN + CatBoost + LightGBM + ExtraTrees → Ridge) on PHY-C-light 58d

Usage:
    python scripts/exp_stacking_phyc.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import get_feature_names
from src.ml.sklearn_models import build_preprocessing

CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c"

DROP = [
    "CP_oligomer_level", "CP_Cn_proxy",
    "L1_RingCount", "L1_HeavyAtomCount", "L1_MolWt",
    "IC_hydrophilic_ratio",
]


def build_stacking():
    """Build stacking with TabPFN as a base learner."""
    from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from tabpfn import TabPFNRegressor

    base_estimators = [
        ("tabpfn", TabPFNRegressor()),
        ("catboost", CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, random_seed=42, verbose=0,
        )),
        ("lightgbm", LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )),
        ("extratrees", ExtraTreesRegressor(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        )),
    ]

    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=1,  # TabPFN uses GPU, avoid conflicts
    )


def main():
    from sklearn.model_selection import RepeatedKFold

    if not CACHE.exists():
        print(f"ERROR: Cache not found: {CACHE}")
        sys.exit(1)

    df = pd.read_parquet(CACHE)
    all_names = get_feature_names("PHY-C")
    y = df["tg_k"].values
    X_full = df[all_names].values

    valid = ~np.all(np.isnan(X_full), axis=1)
    X_full = X_full[valid]
    y = y[valid]

    keep_names = [f for f in all_names if f not in DROP]
    keep_idx = [all_names.index(f) for f in keep_names]
    X = X_full[:, keep_idx]

    # NaN → median for stacking (LightGBM/ExtraTrees don't all handle NaN)
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    pp = build_preprocessing()
    X_pp = pp.fit_transform(X)

    print(f"PHY-C-light 58d: {X_pp.shape[0]} samples, {X_pp.shape[1]} features")

    # Nested CV (outer only, stacking has internal CV)
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    r2s, maes, rmses = [], [], []
    t0 = time.time()
    fold = 0

    print(f"\nRunning Stacking Nested CV (5x3 folds)...")
    for train_idx, test_idx in rkf.split(X_pp):
        stacker = build_stacking()
        stacker.fit(X_pp[train_idx], y[train_idx])
        pred = stacker.predict(X_pp[test_idx])

        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
        r2s.append(1 - ss_res / ss_tot)
        maes.append(np.mean(np.abs(y[test_idx] - pred)))
        rmses.append(np.sqrt(np.mean((y[test_idx] - pred) ** 2)))
        fold += 1

        if fold % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Repeat {fold//5}/3 done (R2={np.mean(r2s[-5:]):.4f}, MAE={np.mean(maes[-5:]):.1f}K, time={elapsed:.0f}s)")

    elapsed = time.time() - t0
    metrics = {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
    }

    print(f"\n{'='*60}")
    print(f"  Stacking (TabPFN+CatBoost+LightGBM+ET→Ridge) Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    # Compare
    print(f"\n  Comparison:")
    print(f"  {'Model':<40s} {'R2':>12s} {'MAE':>10s}")
    print(f"  {'-'*65}")

    refs = [
        ("CatBoost PHY-C-light 58d", "select_PHY-C-light-v2.json"),
        ("TabPFN PHY-C-light 58d", "tabpfn_PHY-C-light.json"),
    ]
    for label, fname in refs:
        p = RESULT_DIR / fname
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            m = d["metrics"]
            print(f"  {label:<40s} {m['R2_mean']:.4f}+-{m['R2_std']:.4f} {m['MAE_mean']:.1f}+-{m['MAE_std']:.1f}K")

    print(f"  {'Stacking PHY-C-light 58d':<40s} {metrics['R2_mean']:.4f}+-{metrics['R2_std']:.4f} {metrics['MAE_mean']:.1f}+-{metrics['MAE_std']:.1f}K")

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "stacking_PHY-C-light.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": "Stacking_PHY-C-light",
            "model": "Stacking(TabPFN+CatBoost+LightGBM+ET→Ridge)",
            "n_features": len(keep_names),
            "n_samples": len(y),
            "metrics": metrics,
            "outer_cv": "5x3",
            "elapsed_seconds": elapsed,
        }, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
