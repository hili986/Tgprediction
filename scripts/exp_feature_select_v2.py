"""
特征筛选 v2: 轻度修剪只去 6 个 (保留 ring_strain_proxy + hbond_network_potential)

Usage:
    python scripts/exp_feature_select_v2.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c"

# 只去 6 个: 常量/方向错/完全重复
DROP_FEATURES = [
    "CP_oligomer_level",       # 常量, 零方差
    "CP_Cn_proxy",             # |r|=0.01, 物理方向错
    "L1_RingCount",            # = VPD_RingCount_per_RU
    "L1_HeavyAtomCount",       # = VPD_HeavyAtom_per_RU
    "L1_MolWt",                # = VPD_MolWt_per_RU
    "IC_hydrophilic_ratio",    # = -IC_hydrophobic_ratio
]


def main():
    import pandas as pd
    from src.features.feature_pipeline import get_feature_names

    # Load cache
    cache = CACHE_DIR / "feature_matrix_PHY-C.parquet"
    if not cache.exists():
        print(f"ERROR: Cache not found: {cache}")
        sys.exit(1)

    df = pd.read_parquet(cache)
    all_names = get_feature_names("PHY-C")
    y = df["tg_k"].values
    X_full = df[all_names].values

    valid = ~np.all(np.isnan(X_full), axis=1)
    X_full = X_full[valid]
    y = y[valid]

    # Light pruning v2: only drop 6
    pruned_names = [f for f in all_names if f not in DROP_FEATURES]
    pruned_idx = [all_names.index(f) for f in pruned_names]
    X_pruned = X_full[:, pruned_idx]

    print(f"PHY-C-light-v2: {len(pruned_names)}d (dropped {len(DROP_FEATURES)})")
    print(f"  Dropped: {DROP_FEATURES}")
    print(f"  Kept: PPF_ring_strain_proxy, hbond_network_potential")

    # CatBoost Nested CV
    from catboost import CatBoostRegressor
    from sklearn.model_selection import RepeatedKFold

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    r2s, maes = [], []
    t0 = time.time()
    fold = 0

    print(f"\n  Running CatBoost Nested CV (5x3 folds)...")
    for train_idx, test_idx in rkf.split(X_pruned):
        model = CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, random_seed=42, verbose=0,
        )
        model.fit(X_pruned[train_idx], y[train_idx],
                  eval_set=(X_pruned[test_idx], y[test_idx]),
                  early_stopping_rounds=50, verbose=0)
        pred = model.predict(X_pruned[test_idx])

        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
        r2s.append(1 - ss_res / ss_tot)
        maes.append(np.mean(np.abs(y[test_idx] - pred)))
        fold += 1
        if fold % 5 == 0:
            print(f"  Repeat {fold//5}/3 done (R2={np.mean(r2s[-5:]):.4f}, MAE={np.mean(maes[-5:]):.1f}K, time={time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    metrics = {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
    }

    print(f"\n  {'='*55}")
    print(f"  PHY-C-light-v2 ({len(pruned_names)}d) Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  {'='*55}")

    # Compare with previous results
    print(f"\n  Comparison:")
    print(f"  {'Layer':<20s} {'Dim':>4s} {'R2':>12s} {'MAE':>10s}")
    print(f"  {'-'*50}")

    refs = [
        ("PHY-B2 (ref)", "ablation_PHY-B2.json"),
        ("PHY-C-light-v1", "select_PHY-C-light.json"),
        ("PHY-C-full", "select_PHY-C-full.json"),
    ]
    for label, fname in refs:
        p = RESULT_DIR / fname
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            m = d["metrics"]
            print(f"  {label:<20s} {d['n_features']:>4d} {m['R2_mean']:.4f}+-{m['R2_std']:.4f} {m['MAE_mean']:.1f}+-{m['MAE_std']:.1f}K")

    print(f"  {'PHY-C-light-v2':<20s} {len(pruned_names):>4d} {metrics['R2_mean']:.4f}+-{metrics['R2_std']:.4f} {metrics['MAE_mean']:.1f}+-{metrics['MAE_std']:.1f}K")

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "PHY-C-light-v2",
        "n_features": len(pruned_names),
        "n_samples": len(y),
        "feature_names": pruned_names,
        "dropped": DROP_FEATURES,
        "metrics": metrics,
        "model": "CatBoost",
        "outer_cv": "5x3",
        "elapsed_seconds": elapsed,
    }
    out = RESULT_DIR / "select_PHY-C-light-v2.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
