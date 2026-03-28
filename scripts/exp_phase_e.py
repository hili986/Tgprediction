"""
Phase E: 三方案对比 — CatBoost 专家委员会 vs TabPFN 专家委员会 vs TabPFN 直接

方案 A: CatBoost 专家委员会 (含单调约束) + ElasticNet 元学习器
方案 B: TabPFN 专家委员会 + ElasticNet 元学习器
方案 C: TabPFN 直接跑 186d (零泄漏)

Usage:
    python scripts/exp_phase_e.py
    python scripts/exp_phase_e.py --plan A B   # 只跑指定方案
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RepeatedKFold, KFold

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_pipeline import get_feature_names
from src.ml.sklearn_models import build_preprocessing

PHYC_CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
GNN_CACHE = PROJECT_ROOT / "data" / "gnn_embeddings_64d.parquet"
PBERT_CACHE = PROJECT_ROOT / "data" / "polybert_embeddings.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_e"

DROP = [
    "CP_oligomer_level", "CP_Cn_proxy",
    "L1_RingCount", "L1_HeavyAtomCount", "L1_MolWt",
    "IC_hydrophilic_ratio",
]

POLYBERT_PCA_DIM = 64

# ── 专家特征定义 ─────────────────────────────────────────────

EXPERT_FEATURES = {
    "E1_flexibility": [
        "L0_FlexibilityIndex", "L1_FractionCSP3", "L1_NumRotatableBonds",
        "PPF_flexible_bond_density", "PPF_backbone_rigidity",
        "VPD_RotBonds_per_RU", "VPD_RotBonds_delta", "VPD_junction_flex_ratio",
        "CP_curl_ratio", "CP_Neff_ratio", "CP_conf_strain", "CP_curl_variance",
    ],
    "E2_intermolecular": [
        "L0_HBondDensity", "L0_PolarityIndex", "L0_SOL",
        "PPF_CED_estimate", "PPF_Vf_estimate", "PPF_CED_hbond_frac",
        "ced_weighted_sum", "total_hbond_density", "hbond_network_potential",
        "polar_fraction", "interaction_types",
        "IC_hydrophobic_ratio", "IC_dipole_moment", "IC_MolMR",
        "IC_polar_bond_fraction", "IC_MaxPartialCharge", "IC_MinPartialCharge",
        "IC_MaxAbsPartialCharge",
    ],
    "E3_physics_baseline": [
        "GC_Tg", "GC_coverage", "PPF_M_per_f",
    ],
    # E4 uses ALL features (defined dynamically)
    # E5 uses GNN + anchors (defined dynamically)
    # E6 uses polyBERT PCA + anchor (defined dynamically)
}

# 元学习器额外特征
META_FEATURES = ["GC_Tg", "L0_FlexibilityIndex", "CP_conf_strain"]


# ── 数据加载 ─────────────────────────────────────────────────

def load_all_data() -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Load PHY-C 58d + GNN 64d + polyBERT 600d raw.

    Returns:
        X_phyc (N, 58), y (N,), phyc_names, X_gnn (N, 64), X_pbert_raw (N, 600)
    """
    # PHY-C-light
    df_phyc = pd.read_parquet(PHYC_CACHE)
    all_names = get_feature_names("PHY-C")
    keep_names = [f for f in all_names if f not in DROP]
    X_phyc = df_phyc[keep_names].values
    y = df_phyc["tg_k"].values

    # GNN 64d
    df_gnn = pd.read_parquet(GNN_CACHE)
    gnn_cols = sorted([c for c in df_gnn.columns if c.startswith("GNN_")])
    X_gnn = df_gnn[gnn_cols].values

    # polyBERT 600d raw
    df_pbert = pd.read_parquet(PBERT_CACHE)
    pbert_cols = sorted([c for c in df_pbert.columns if c.startswith("pBERT_")])
    X_pbert = df_pbert[pbert_cols].values

    print(f"Loaded: PHY-C {X_phyc.shape[1]}d + GNN {X_gnn.shape[1]}d + polyBERT {X_pbert.shape[1]}d")
    print(f"Samples: {len(y)}")

    return X_phyc, y, keep_names, X_gnn, X_pbert


def build_fold_features(
    X_phyc, X_gnn, X_pbert_raw, train_idx, test_idx,
    phyc_names, gnn_names, pbert_pca_names,
):
    """Build per-fold feature matrices with polyBERT PCA fitted on train only."""
    # PCA on train, transform both
    pca = PCA(n_components=POLYBERT_PCA_DIM, random_state=42)
    pbert_valid_train = ~np.any(np.isnan(X_pbert_raw[train_idx]), axis=1)
    pca.fit(X_pbert_raw[train_idx][pbert_valid_train])

    X_pbert_pca = np.full((len(X_pbert_raw), POLYBERT_PCA_DIM), np.nan)
    valid_all = ~np.any(np.isnan(X_pbert_raw), axis=1)
    X_pbert_pca[valid_all] = pca.transform(X_pbert_raw[valid_all])

    # Full feature matrix
    X_full = np.hstack([X_phyc, X_gnn, X_pbert_pca])
    all_names = phyc_names + gnn_names + pbert_pca_names

    # Preprocessing
    pp = build_preprocessing()
    X_train_pp = pp.fit_transform(X_full[train_idx])
    X_test_pp = pp.transform(X_full[test_idx])

    return X_train_pp, X_test_pp, all_names


def get_expert_indices(all_names: List[str]) -> Dict[str, List[int]]:
    """Map expert names to column indices in the full feature matrix."""
    gnn_names = [f"GNN_{i}" for i in range(64)]
    pbert_names = [f"pBERT_PCA_{i}" for i in range(POLYBERT_PCA_DIM)]

    experts = {}

    # E1-E3: predefined feature lists
    for ename, feat_list in EXPERT_FEATURES.items():
        idx = [all_names.index(f) for f in feat_list if f in all_names]
        experts[ename] = idx

    # E4: all features
    experts["E4_full"] = list(range(len(all_names)))

    # E5: GNN + anchors
    e5_feats = gnn_names + ["GC_Tg", "L0_FlexibilityIndex"]
    experts["E5_graph"] = [all_names.index(f) for f in e5_feats if f in all_names]

    # E6: polyBERT PCA + anchor
    e6_feats = pbert_names + ["GC_Tg"]
    experts["E6_language"] = [all_names.index(f) for f in e6_feats if f in all_names]

    # Meta features
    experts["_meta"] = [all_names.index(f) for f in META_FEATURES if f in all_names]

    for k, v in experts.items():
        if not k.startswith("_"):
            print(f"  {k}: {len(v)}d")

    return experts


def make_expert_model(expert_name: str, model_type: str):
    """Create model for a given expert."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    if expert_name == "E3_physics_baseline":
        # Ridge can't handle NaN, wrap with imputer
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge", Ridge(alpha=1.0)),
        ])

    if model_type == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, random_seed=42, verbose=0,
        )
    elif model_type == "tabpfn":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── 方案 A/B: 专家委员会 CV ──────────────────────────────────

def expert_committee_cv(
    X_phyc, y, X_gnn, X_pbert_raw, phyc_names,
    model_type: str = "catboost",
    outer_splits: int = 5, outer_repeats: int = 3,
):
    """Run expert committee with Nested CV."""
    gnn_names = [f"GNN_{i}" for i in range(64)]
    pbert_pca_names = [f"pBERT_PCA_{i}" for i in range(POLYBERT_PCA_DIM)]

    label = f"Expert Committee ({model_type})"
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    # Dry run to get feature indices
    all_names = phyc_names + gnn_names + pbert_pca_names
    expert_idx = get_expert_indices(all_names)
    expert_names = [k for k in expert_idx if not k.startswith("_")]

    rkf = RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=42)
    r2s, maes = [], []
    t0 = time.time()
    fold = 0

    for train_idx, test_idx in rkf.split(X_phyc):
        # Build per-fold features (polyBERT PCA on train)
        X_train, X_test, _ = build_fold_features(
            X_phyc, X_gnn, X_pbert_raw, train_idx, test_idx,
            phyc_names, gnn_names, pbert_pca_names,
        )
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for OOF predictions (meta-learner training data)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(train_idx), len(expert_names)))

        for ei, ename in enumerate(expert_names):
            idx = expert_idx[ename]
            for inner_train, inner_val in inner_cv.split(X_train):
                model = make_expert_model(ename, model_type)
                X_it = X_train[inner_train][:, idx]
                X_iv = X_train[inner_val][:, idx]

                if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                    model.fit(X_it, y_train[inner_train],
                              eval_set=(X_iv, y_train[inner_val]),
                              early_stopping_rounds=50, verbose=0)
                else:
                    model.fit(X_it, y_train[inner_train])

                oof_preds[inner_val, ei] = model.predict(X_iv)

        # Train meta-learner on OOF predictions + meta features
        meta_idx = expert_idx["_meta"]
        X_meta_train = np.hstack([oof_preds, X_train[:, meta_idx]])
        meta_learner = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        meta_learner.fit(X_meta_train, y_train)

        # Test: each expert predicts on test set
        test_preds = np.zeros((len(test_idx), len(expert_names)))
        for ei, ename in enumerate(expert_names):
            idx = expert_idx[ename]
            model = make_expert_model(ename, model_type)

            if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
                # Use full train for final expert
                model.fit(X_train[:, idx], y_train, verbose=0)
            else:
                model.fit(X_train[:, idx], y_train)

            test_preds[:, ei] = model.predict(X_test[:, idx])

        # Meta-learner predicts
        X_meta_test = np.hstack([test_preds, X_test[:, meta_idx]])
        y_pred = meta_learner.predict(X_meta_test)

        # Metrics
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2s.append(1 - ss_res / ss_tot)
        maes.append(np.mean(np.abs(y_test - y_pred)))

        fold += 1
        if fold % outer_splits == 0:
            rep = fold // outer_splits
            print(f"  Repeat {rep}/{outer_repeats} done "
                  f"(R2={np.mean(r2s[-outer_splits:]):.4f}, "
                  f"MAE={np.mean(maes[-outer_splits:]):.1f}K, "
                  f"time={time.time()-t0:.0f}s)")

    metrics = {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
    }
    print(f"\n  {label} Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    return metrics


# ── 方案 C: TabPFN 直接 ──────────────────────────────────────

def tabpfn_direct_cv(
    X_phyc, y, X_gnn, X_pbert_raw, phyc_names,
    outer_splits: int = 5, outer_repeats: int = 3,
):
    """TabPFN on full 186d with per-fold PCA (zero leakage)."""
    from tabpfn import TabPFNRegressor

    gnn_names = [f"GNN_{i}" for i in range(64)]
    pbert_pca_names = [f"pBERT_PCA_{i}" for i in range(POLYBERT_PCA_DIM)]

    print(f"\n{'='*65}")
    print(f"  TabPFN Direct 186d (zero leakage)")
    print(f"{'='*65}")

    rkf = RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats, random_state=42)
    r2s, maes = [], []
    t0 = time.time()
    fold = 0

    for train_idx, test_idx in rkf.split(X_phyc):
        X_train, X_test, _ = build_fold_features(
            X_phyc, X_gnn, X_pbert_raw, train_idx, test_idx,
            phyc_names, gnn_names, pbert_pca_names,
        )
        y_train, y_test = y[train_idx], y[test_idx]

        model = TabPFNRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2s.append(1 - ss_res / ss_tot)
        maes.append(np.mean(np.abs(y_test - y_pred)))

        fold += 1
        if fold % outer_splits == 0:
            rep = fold // outer_splits
            print(f"  Repeat {rep}/{outer_repeats} done "
                  f"(R2={np.mean(r2s[-outer_splits:]):.4f}, "
                  f"MAE={np.mean(maes[-outer_splits:]):.1f}K, "
                  f"time={time.time()-t0:.0f}s)")

    metrics = {
        "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
    }
    print(f"\n  TabPFN Direct Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    return metrics


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase E: Three-way comparison")
    parser.add_argument("--plan", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"], help="Which plans to run")
    args = parser.parse_args()

    X_phyc, y, phyc_names, X_gnn, X_pbert_raw = load_all_data()

    results = {}

    if "A" in args.plan:
        results["A_CatBoost_Committee"] = expert_committee_cv(
            X_phyc, y, X_gnn, X_pbert_raw, phyc_names, model_type="catboost",
        )

    if "B" in args.plan:
        results["B_TabPFN_Committee"] = expert_committee_cv(
            X_phyc, y, X_gnn, X_pbert_raw, phyc_names, model_type="tabpfn",
        )

    if "C" in args.plan:
        results["C_TabPFN_Direct"] = tabpfn_direct_cv(
            X_phyc, y, X_gnn, X_pbert_raw, phyc_names,
        )

    # Summary
    print(f"\n{'='*70}")
    print(f"  Phase E Final Comparison")
    print(f"{'='*70}")
    print(f"  {'Plan':<30s} {'R2':>14s} {'MAE (K)':>14s}")
    print(f"  {'-'*60}")
    for name, m in results.items():
        print(f"  {name:<30s} {m['R2_mean']:.4f}+-{m['R2_std']:.4f} {m['MAE_mean']:.1f}+-{m['MAE_std']:.1f}K")

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "phase_e_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
