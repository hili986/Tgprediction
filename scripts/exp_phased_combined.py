"""
Phase D Combined: PHY-C-light 58d + GNN 64d + polyBERT 64d = 186d → TabPFN

Usage:
    python scripts/exp_phased_combined.py
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

from src.features.feature_pipeline import get_feature_names
from src.gnn.polybert_embedder import polybert_pca
from src.ml.sklearn_models import build_preprocessing
from src.ml.evaluation import nested_cv_no_tuning

PHYC_CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
GNN_CACHE = PROJECT_ROOT / "data" / "gnn_embeddings_64d.parquet"
PBERT_CACHE = PROJECT_ROOT / "data" / "polybert_embeddings.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_d"

DROP = [
    "CP_oligomer_level", "CP_Cn_proxy",
    "L1_RingCount", "L1_HeavyAtomCount", "L1_MolWt",
    "IC_hydrophilic_ratio",
]


def main():
    # Load PHY-C-light 58d
    df_phyc = pd.read_parquet(PHYC_CACHE)
    all_names = get_feature_names("PHY-C")
    keep_names = [f for f in all_names if f not in DROP]
    X_phyc = df_phyc[keep_names].values
    y = df_phyc["tg_k"].values

    # Load GNN 64d
    df_gnn = pd.read_parquet(GNN_CACHE)
    gnn_cols = [c for c in df_gnn.columns if c.startswith("GNN_")]
    X_gnn = df_gnn[gnn_cols].values

    # Load polyBERT 600d → PCA 64d
    df_pbert = pd.read_parquet(PBERT_CACHE)
    pbert_cols = [c for c in df_pbert.columns if c.startswith("pBERT_")]
    X_pbert_raw = df_pbert[pbert_cols].values
    X_pbert = polybert_pca(X_pbert_raw, target_dim=64)

    # Concatenate all
    X_combined = np.hstack([X_phyc, X_gnn, X_pbert])
    print(f"Combined: {X_combined.shape[1]}d (PHY-C {X_phyc.shape[1]}d + GNN {X_gnn.shape[1]}d + polyBERT 64d)")

    # Drop all-NaN rows
    valid = ~np.all(np.isnan(X_combined), axis=1)
    X_combined = X_combined[valid]
    y_valid = y[valid]
    print(f"Valid samples: {len(y_valid)}")

    # Preprocessing
    pp = build_preprocessing()
    X_pp = pp.fit_transform(X_combined)

    # TabPFN
    from tabpfn import TabPFNRegressor
    estimator = TabPFNRegressor()

    print(f"\nRunning TabPFN Nested CV (5x3 folds) on {X_pp.shape[1]}d...")
    result = nested_cv_no_tuning(X_pp, y_valid, estimator, outer_splits=5, outer_repeats=3, verbose=True)

    metrics = result["metrics"]
    print(f"\n{'='*70}")
    print(f"  TabPFN + PHY-C + GNN + polyBERT ({X_pp.shape[1]}d) Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    print(f"{'='*70}")

    # Full comparison
    print(f"\n  Phase D Summary:")
    print(f"  {'Model':<50s} {'R2':>12s} {'MAE':>10s}")
    print(f"  {'-'*75}")

    refs = [
        ("TabPFN PHY-C-light 58d", "phase_c/tabpfn_PHY-C-light.json"),
        ("TabPFN PHY-C + GNN 122d", "phase_d/tabpfn_phyc_gnn.json"),
        ("TabPFN PHY-C + polyBERT 122d", "phase_d/tabpfn_phyc_polybert.json"),
    ]
    for label, path in refs:
        p = PROJECT_ROOT / "results" / path
        if p.exists():
            with open(p) as f:
                m = json.load(f)["metrics"]
            print(f"  {label:<50s} {m['R2_mean']:.4f}+-{m['R2_std']:.4f} {m['MAE_mean']:.1f}+-{m['MAE_std']:.1f}K")

    print(f"  {'TabPFN PHY-C + GNN + polyBERT ' + str(X_pp.shape[1]) + 'd':<50s} {metrics['R2_mean']:.4f}+-{metrics['R2_std']:.4f} {metrics['MAE_mean']:.1f}+-{metrics['MAE_std']:.1f}K")

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "tabpfn_phyc_gnn_polybert.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": "TabPFN_PHY-C_GNN_polyBERT",
            "model": "TabPFN_v2",
            "n_features": X_pp.shape[1],
            "n_samples": len(y_valid),
            "feature_composition": {
                "PHY-C-light": X_phyc.shape[1],
                "GNN_embed": X_gnn.shape[1],
                "polyBERT_PCA": 64,
            },
            "metrics": metrics,
            "outer_cv": "5x3",
        }, f, indent=2)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
