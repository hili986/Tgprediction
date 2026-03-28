"""
PHY-C 64d 特征与 Tg 相关性分析

Usage:
    python scripts/feature_correlation.py --quick     # 仅 chain_physics 8d (秒出)
    python scripts/feature_correlation.py             # 全部 64d (首次 ~2h, 之后秒出)
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"
CP_PATH = PROJECT_ROOT / "data" / "chain_physics_features.parquet"
MATRIX_CACHE = PROJECT_ROOT / "data" / "feature_matrix_PHY-C.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c"


def correlate(X: np.ndarray, y: np.ndarray, names: list) -> pd.DataFrame:
    """Compute Pearson + Spearman for each feature vs y."""
    rows = []
    for i, name in enumerate(names):
        col = X[:, i]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        if n_valid < 30:
            rows.append((name, np.nan, np.nan, n_valid, n_valid / len(y)))
            continue
        pr, _ = stats.pearsonr(col[valid], y[valid])
        sr, _ = stats.spearmanr(col[valid], y[valid])
        rows.append((name, pr, sr, n_valid, n_valid / len(y)))

    df = pd.DataFrame(rows, columns=["feature", "pearson_r", "spearman_r", "n_valid", "coverage"])
    df["abs_pearson"] = df["pearson_r"].abs()
    return df.sort_values("abs_pearson", ascending=False)


def print_table(result: pd.DataFrame, title: str):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(f"  {'#':>3s}  {'Feature':<30s} {'Pearson':>9s} {'Spearman':>9s} {'Coverage':>9s}")
    print(f"  {'-'*65}")
    for rank, (_, r) in enumerate(result.iterrows(), 1):
        cov = f"{r['coverage']:.1%}"
        pr = f"{r['pearson_r']:+.4f}" if pd.notna(r['pearson_r']) else "N/A"
        sr = f"{r['spearman_r']:+.4f}" if pd.notna(r['spearman_r']) else "N/A"
        print(f"  {rank:3d}  {r['feature']:<30s} {pr:>9s} {sr:>9s} {cov:>9s}")


def quick_chain_physics():
    """Instant correlation: chain_physics 8d from cached parquet."""
    print("Loading unified_tg + chain_physics cache...")
    df_tg = pd.read_parquet(DATA_PATH)[["smiles", "tg_k"]]
    df_cp = pd.read_parquet(CP_PATH)

    merged = df_tg.merge(df_cp, on="smiles", how="inner")
    print(f"  Matched: {len(merged)} / {len(df_tg)} samples")

    cp_cols = [c for c in df_cp.columns if c != "smiles"]
    X = merged[cp_cols].values
    y = merged["tg_k"].values
    names = [f"CP_{c}" for c in cp_cols]

    result = correlate(X, y, names)
    print_table(result, "Chain Physics 8d Correlations with Tg")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_DIR / "correlation_chain_physics.csv", index=False)
    print(f"\n  Saved: {RESULT_DIR / 'correlation_chain_physics.csv'}")
    return result


def full_64d():
    """Full PHY-C 64d correlation (uses cached matrix if available)."""
    from src.data.external_datasets import load_unified_dataset
    from src.features.feature_pipeline import compute_features, get_feature_names

    layer = "PHY-C"
    feat_names = get_feature_names(layer)

    if MATRIX_CACHE.exists():
        print(f"Loading cached feature matrix: {MATRIX_CACHE}")
        df_cache = pd.read_parquet(MATRIX_CACHE)
        X = df_cache[feat_names].values
        y = df_cache["tg_k"].values
        print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        print(f"No cache found. Computing {layer} features (this takes ~2h, saved for reuse)...")
        df = load_unified_dataset(str(DATA_PATH), split=None)
        X_list, y_list, smi_list = [], [], []
        t0 = time.time()
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f"  [{i+1}/{total}] elapsed={elapsed:.0f}s, ETA={eta:.0f}s")
            try:
                x = compute_features(row["smiles"], layer=layer)
                X_list.append(x)
                y_list.append(float(row["tg_k"]))
                smi_list.append(row["smiles"])
            except Exception:
                pass

        X = np.array(X_list)
        y = np.array(y_list)
        print(f"  Computed in {time.time()-t0:.0f}s. Saving cache...")

        df_save = pd.DataFrame(X, columns=feat_names)
        df_save["tg_k"] = y
        df_save["smiles"] = smi_list
        df_save.to_parquet(MATRIX_CACHE, index=False)
        print(f"  Cached: {MATRIX_CACHE}")

    result = correlate(X, y, feat_names)
    print_table(result, f"{layer} {len(feat_names)}d — All Feature Correlations with Tg")

    # Top 20
    print(f"\n  {'='*60}")
    print(f"  Top 20 (by |Pearson|):")
    print(f"  {'='*60}")
    for rank, (_, r) in enumerate(result.head(20).iterrows(), 1):
        direction = "+" if r["pearson_r"] > 0 else "-"
        print(f"  {rank:2d}. {r['feature']:<30s} r={r['pearson_r']:+.4f}  (Tg{direction})")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULT_DIR / "correlation_full_64d.csv", index=False)
    print(f"\n  Saved: {RESULT_DIR / 'correlation_full_64d.csv'}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Only chain_physics 8d (instant)")
    args = parser.parse_args()

    if args.quick:
        quick_chain_physics()
    else:
        # Always show chain_physics first (instant)
        quick_chain_physics()
        print("\n" + "~" * 78)
        full_64d()


if __name__ == "__main__":
    main()
