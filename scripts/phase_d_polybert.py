"""
Phase D Track B: polyBERT 嵌入提取 + 缓存 + 快速验证

Usage:
    python scripts/phase_d_polybert.py                # 提取 + 缓存 + TabPFN 验证
    python scripts/phase_d_polybert.py --extract-only  # 只提取不验证
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"
POLYBERT_CACHE = PROJECT_ROOT / "data" / "polybert_embeddings.parquet"
PHYC_CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_d"

DROP = [
    "CP_oligomer_level", "CP_Cn_proxy",
    "L1_RingCount", "L1_HeavyAtomCount", "L1_MolWt",
    "IC_hydrophilic_ratio",
]


def extract_and_cache(local_model: str = None):
    """Extract polyBERT embeddings and save to parquet."""
    from src.gnn.polybert_embedder import extract_polybert_embeddings

    df = pd.read_parquet(DATA_PATH)
    smiles_list = df["smiles"].tolist()
    print(f"Extracting polyBERT embeddings for {len(smiles_list)} SMILES...")

    t0 = time.time()
    embeddings = extract_polybert_embeddings(
        smiles_list, model_path=local_model, batch_size=64, device="cuda",
    )
    elapsed = time.time() - t0
    print(f"  Extraction done in {elapsed:.0f}s")

    # Save raw embeddings (600d for polyBERT)
    embed_dim = embeddings.shape[1]
    col_names = [f"pBERT_{i}" for i in range(embed_dim)]
    df_embed = pd.DataFrame(embeddings, columns=col_names)
    df_embed["smiles"] = smiles_list
    df_embed.to_parquet(POLYBERT_CACHE, index=False)
    print(f"  Saved: {POLYBERT_CACHE}")

    return embeddings, smiles_list


def load_cached():
    """Load cached polyBERT embeddings."""
    df = pd.read_parquet(POLYBERT_CACHE)
    col_names = [c for c in df.columns if c.startswith("pBERT_")]
    return df[col_names].values, df["smiles"].tolist()


def run_tabpfn_test(embeddings_768: np.ndarray):
    """Quick TabPFN test: PHY-C-light 58d + polyBERT PCA 64d."""
    from src.gnn.polybert_embedder import polybert_pca
    from src.features.feature_pipeline import get_feature_names
    from src.ml.sklearn_models import build_preprocessing
    from src.ml.evaluation import nested_cv_no_tuning

    # Load PHY-C-light features
    df_phyc = pd.read_parquet(PHYC_CACHE)
    all_names = get_feature_names("PHY-C")
    keep_names = [f for f in all_names if f not in DROP]
    X_phyc = df_phyc[[f for f in keep_names]].values
    y = df_phyc["tg_k"].values

    # PCA on full set (for quick test; proper Nested CV PCA in Phase E)
    pca_names = [f"pBERT_PCA_{i}" for i in range(64)]
    X_pca = polybert_pca(embeddings_768, target_dim=64)

    # Concatenate
    X_combined = np.hstack([X_phyc, X_pca])
    combined_names = keep_names + pca_names
    print(f"\nCombined features: {X_combined.shape[1]}d (PHY-C 58d + polyBERT 64d)")

    # Drop all-NaN rows
    valid = ~np.all(np.isnan(X_combined), axis=1)
    X_combined = X_combined[valid]
    y_valid = y[valid]
    print(f"  Valid samples: {len(y_valid)}")

    # Preprocessing
    pp = build_preprocessing()
    X_pp = pp.fit_transform(X_combined)

    # TabPFN
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("TabPFN not installed. Skipping verification.")
        return

    estimator = TabPFNRegressor()
    print(f"\nRunning TabPFN Nested CV (5x3 folds) on {X_pp.shape[1]}d...")
    result = nested_cv_no_tuning(X_pp, y_valid, estimator, outer_splits=5, outer_repeats=3, verbose=True)

    metrics = result["metrics"]
    print(f"\n{'='*65}")
    print(f"  TabPFN + PHY-C-light + polyBERT ({X_pp.shape[1]}d) Results:")
    print(f"  R2  = {metrics['R2_mean']:.4f} +/- {metrics['R2_std']:.4f}")
    print(f"  MAE = {metrics['MAE_mean']:.1f}K +/- {metrics['MAE_std']:.1f}K")
    print(f"{'='*65}")

    # Compare
    print(f"\n  Comparison:")
    print(f"  {'Model':<45s} {'R2':>12s} {'MAE':>10s}")
    print(f"  {'-'*70}")

    ref = RESULT_DIR.parent / "phase_c" / "tabpfn_PHY-C-light.json"
    if ref.exists():
        with open(ref) as f:
            m = json.load(f)["metrics"]
        print(f"  {'TabPFN PHY-C-light 58d':<45s} {m['R2_mean']:.4f}+-{m['R2_std']:.4f} {m['MAE_mean']:.1f}+-{m['MAE_std']:.1f}K")

    print(f"  {'TabPFN PHY-C-light+polyBERT ' + str(X_pp.shape[1]) + 'd':<45s} {metrics['R2_mean']:.4f}+-{metrics['R2_std']:.4f} {metrics['MAE_mean']:.1f}+-{metrics['MAE_std']:.1f}K")

    # Save
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "tabpfn_phyc_polybert.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": "TabPFN_PHY-C-light_polyBERT",
            "model": "TabPFN_v2",
            "n_features": X_pp.shape[1],
            "n_samples": len(y_valid),
            "feature_composition": {"PHY-C-light": 58, "polyBERT_PCA": 64},
            "metrics": metrics,
            "outer_cv": "5x3",
        }, f, indent=2)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--local-model", type=str, default=None,
                        help="Path to locally downloaded polyBERT model directory")
    args = parser.parse_args()

    # Step 1: Extract or load
    if POLYBERT_CACHE.exists():
        print(f"Loading cached polyBERT embeddings: {POLYBERT_CACHE}")
        embeddings, smiles = load_cached()
        print(f"  Loaded: {embeddings.shape}")
    else:
        embeddings, smiles = extract_and_cache(local_model=args.local_model)

    if args.extract_only:
        print("Done (extract only).")
        return

    # Step 2: TabPFN verification
    run_tabpfn_test(embeddings)


if __name__ == "__main__":
    main()
