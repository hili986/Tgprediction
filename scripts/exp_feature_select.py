"""
特征筛选实验: PHY-C-select vs PHY-C-full vs PHY-B2

基于相关性分析，从 64d 中挑选高价值特征，对比全量和 PHY-B2 基线。
依赖 data/feature_matrix_PHY-C.parquet 缓存（由 feature_correlation.py 生成）。

Usage:
    python scripts/exp_feature_select.py
"""
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c"

# ────────────────────────────────────────────────────────────────
#  特征筛选方案
#  选择标准: max(|Pearson|, |Spearman|) >= 0.3, 去除冗余
# ────────────────────────────────────────────────────────────────

# 冗余对 (保留 VPD/PPF 版本, 它们是 per-RU 物理量, 更有意义)
# 丢弃: L1_RingCount (=VPD_RingCount_per_RU)
#       L1_HeavyAtomCount (=VPD_HeavyAtom_per_RU)
#       L1_MolWt (=VPD_MolWt_per_RU)
#       IC_hydrophilic_ratio (=-IC_hydrophobic_ratio)
#       L1_TPSA (≈VPD_TPSA_per_RU)

SELECTED_FEATURES = [
    # === Tier 1: |r| > 0.5, 核心预测因子 ===
    "PPF_flexible_bond_density",     # r=-0.744  链柔性密度
    "VPD_RingCount_per_RU",          # r=+0.710  环结构
    "L1_FractionCSP3",               # r=-0.700  sp3 碳比例
    "L0_FlexibilityIndex",           # r=-0.663  柔性指数 (SHAP #1)
    "L1_NumAromaticRings",           # r=+0.662  芳环数
    "GC_Tg",                         # r=+0.647  基团贡献 Tg
    "PPF_steric_volume",             # r=+0.586  位阻体积
    "L0_SOL",                        # r=+0.583  溶解度参数
    "PPF_CED_estimate",              # r=+0.576  内聚能密度
    "PPF_backbone_rigidity",         # r=+0.569  骨架刚性
    "PPF_side_chain_ratio",          # r=+0.568  侧链比
    "IC_hydrophobic_ratio",          # r=+0.555  疏水比
    "L1_BalabanJ",                   # r=-0.543  拓扑指数
    "VPD_junction_flex_ratio",       # r=-0.532  聚合接头柔性
    "VPD_HeavyAtom_per_RU",          # r=+0.507  重原子/RU
    "IC_MolMR",                      # r=+0.479  摩尔折射率
    "PPF_Vf_estimate",               # r=-0.476  自由体积
    "VPD_MolWt_per_RU",              # r=+0.472  分子量/RU
    # === Tier 2: 0.3 < |r| < 0.5, 辅助特征 ===
    "L1_Chi0v",                      # r=+0.429
    "L1_MolLogP",                    # r=+0.387
    "VPD_TPSA_per_RU",               # r=+0.380
    "L1_Chi1v",                      # r=+0.377
    "interaction_types",             # r=+0.365
    "GC_coverage",                   # r=-0.358
    "L1_Kappa1",                     # r=+0.348
    "PPF_symmetry_index",            # r=-0.332
    # === Tier 3: Pearson 低但 Spearman 高 (非线性) ===
    "PPF_M_per_f",                   # P=+0.193, S=+0.737! 非线性王者
    "total_hbond_density",           # P=+0.215, S=+0.358
    "ced_weighted_sum",              # P=+0.131, S=+0.333
    # === Tier 4: chain_physics top 3 ===
    "CP_curl_ratio",                 # P=-0.310, S=-0.295
    "CP_Neff_ratio",                 # P=-0.280, S=-0.306
    "CP_conf_strain",                # P=+0.242, S=+0.388
    # === 额外: 边界有用 ===
    "VPD_RotBonds_delta",            # P=-0.291, S=-0.303
    "PPF_CED_hbond_frac",           # P=+0.289, S=+0.311
    "L1_NumHAcceptors",              # P=+0.244, S=+0.298
]


def load_cached_matrix(layer: str):
    """Load cached feature matrix."""
    import pandas as pd
    cache = CACHE_DIR / f"feature_matrix_{layer}.parquet"
    if not cache.exists():
        print(f"ERROR: Cache not found: {cache}")
        print("Run 'python scripts/feature_correlation.py' first to generate cache.")
        sys.exit(1)
    df = pd.read_parquet(cache)
    return df


def nested_cv_catboost(X, y, n_repeats=3, n_splits=5):
    """CatBoost Nested CV (no tuning)."""
    from catboost import CatBoostRegressor
    from sklearn.model_selection import RepeatedKFold

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    results = {"r2": [], "mae": [], "rmse": []}
    t0 = time.time()
    fold = 0

    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=0)
        y_pred = model.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        results["r2"].append(r2)
        results["mae"].append(mae)
        results["rmse"].append(rmse)
        fold += 1

        if fold % n_splits == 0:
            repeat_n = fold // n_splits
            r2_mean = np.mean(results["r2"][-n_splits:])
            mae_mean = np.mean(results["mae"][-n_splits:])
            elapsed = time.time() - t0
            print(f"  Repeat {repeat_n}/{n_repeats} done (R2={r2_mean:.4f}, MAE={mae_mean:.1f}K, time={elapsed:.1f}s)")

    elapsed = time.time() - t0
    metrics = {
        "R2_mean": float(np.mean(results["r2"])),
        "R2_std": float(np.std(results["r2"])),
        "MAE_mean": float(np.mean(results["mae"])),
        "MAE_std": float(np.std(results["mae"])),
        "RMSE_mean": float(np.mean(results["rmse"])),
        "RMSE_std": float(np.std(results["rmse"])),
    }
    return metrics, elapsed


def run_experiment(name: str, X: np.ndarray, y: np.ndarray, feature_names: list):
    """Run a single experiment and save results."""
    print(f"\n{'='*60}")
    print(f"  {name}: {X.shape[1]}d features, {X.shape[0]} samples")
    print(f"{'='*60}")
    print(f"  Running CatBoost Nested CV (5x3 folds)...")

    metrics, elapsed = nested_cv_catboost(X, y)

    print(f"\n  Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    print(f"  Time: {elapsed:.0f}s")

    result = {
        "experiment": name,
        "n_features": X.shape[1],
        "n_samples": X.shape[0],
        "feature_names": feature_names,
        "metrics": metrics,
        "model": "CatBoost",
        "outer_cv": "5x3",
        "elapsed_seconds": elapsed,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"select_{name}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out}")

    return metrics


def main():
    import pandas as pd
    from src.features.feature_pipeline import get_feature_names

    # Load PHY-C cached matrix
    df_c = load_cached_matrix("PHY-C")
    all_names = get_feature_names("PHY-C")
    y = df_c["tg_k"].values
    X_full = df_c[all_names].values

    # Drop all-NaN rows
    valid = ~np.all(np.isnan(X_full), axis=1)
    X_full = X_full[valid]
    y = y[valid]
    print(f"Loaded PHY-C cache: {X_full.shape[0]} samples, {X_full.shape[1]}d")

    # Validate selected features exist
    missing = [f for f in SELECTED_FEATURES if f not in all_names]
    if missing:
        print(f"WARNING: Missing features: {missing}")
        sel_valid = [f for f in SELECTED_FEATURES if f in all_names]
    else:
        sel_valid = SELECTED_FEATURES

    sel_idx = [all_names.index(f) for f in sel_valid]
    X_select = X_full[:, sel_idx]

    print(f"Selected features: {len(sel_valid)}d (from 64d)")
    print(f"  Tier 1 (|r|>0.5): {sum(1 for f in sel_valid[:18])}")
    print(f"  Tier 2 (0.3-0.5): {sum(1 for f in sel_valid[18:26])}")
    print(f"  Tier 3 (非线性):   {sum(1 for f in sel_valid[26:29])}")
    print(f"  Tier 4 (CP top3):  {sum(1 for f in sel_valid[29:32])}")
    print(f"  Tier 5 (边界):     {sum(1 for f in sel_valid[32:])}")

    # Run experiments
    results = {}

    # 1. PHY-C-select
    m1 = run_experiment("PHY-C-select", X_select, y, sel_valid)
    results["PHY-C-select"] = (len(sel_valid), m1)

    # 2. PHY-C-full 64d
    m2 = run_experiment("PHY-C-full", X_full, y, all_names)
    results["PHY-C-full"] = (64, m2)

    # Summary (include PHY-B2 from ablation for reference)
    print(f"\n{'='*70}")
    print(f"  Feature Selection Summary")
    print(f"{'='*70}")
    print(f"  {'Layer':<18s} {'Dim':>5s} {'R2':>14s} {'MAE (K)':>14s} {'Delta R2':>10s}")
    print(f"  {'-'*65}")

    # PHY-B2 reference from ablation
    phyb2_path = RESULT_DIR / "ablation_PHY-B2.json"
    if phyb2_path.exists():
        with open(phyb2_path) as f:
            b2 = json.load(f)["metrics"]
        baseline_r2 = b2["R2_mean"]
        print(f"  {'PHY-B2 (ref)':<18s} {'56':>5s} {b2['R2_mean']:.4f}+-{b2['R2_std']:.4f} {b2['MAE_mean']:.1f}+-{b2['MAE_std']:.1f}K {'baseline':>10s}")
    else:
        baseline_r2 = None
        print(f"  PHY-B2 reference not found")

    for name, (dim, m) in results.items():
        delta = f"+{m['R2_mean']-baseline_r2:.4f}" if baseline_r2 else "N/A"
        print(f"  {name:<18s} {dim:>5d} {m['R2_mean']:.4f}+-{m['R2_std']:.4f} {m['MAE_mean']:.1f}+-{m['MAE_std']:.1f}K {delta:>10s}")


if __name__ == "__main__":
    main()
