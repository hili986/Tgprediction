"""
Phase C Ablation: PHY (48d) vs PHY-B2 (56d) vs PHY-C (64d) CatBoost Nested CV
Phase C 消融实验：对比基线、链间特征、链段物理的增量贡献

Usage:
    python scripts/exp_phase_c_ablation.py
    python scripts/exp_phase_c_ablation.py --quick
    python scripts/exp_phase_c_ablation.py --layer PHY PHY-B2 PHY-C
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message=".*wmic.*", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external_datasets import load_unified_dataset
from src.features.feature_pipeline import compute_features, get_feature_names
from src.ml.sklearn_models import build_preprocessing, get_estimator
from src.ml.evaluation import nested_cv_no_tuning, save_result


DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c"


def _cache_path(layer: str) -> Path:
    """Return path for cached feature matrix."""
    return PROJECT_ROOT / "data" / f"feature_matrix_{layer}.parquet"


def load_and_featurize(layer: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load unified dataset and compute features (with parquet cache)."""
    import pandas as pd

    feature_names = get_feature_names(layer)
    cache = _cache_path(layer)

    # Try loading from cache
    if cache.exists():
        print(f"\n  Loading cached {layer} features: {cache}")
        df_cache = pd.read_parquet(cache)
        X = df_cache[feature_names].values
        y = df_cache["tg_k"].values
        print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features (cached)")
    else:
        print(f"\n  Loading data and computing {layer} features...")
        df = load_unified_dataset(str(DATA_PATH), split=None)
        print(f"  Total samples: {len(df)}")

        X_list, y_list, smi_list = [], [], []
        skipped = 0
        t0 = time.time()
        total = len(df)

        for i, (idx, row) in enumerate(df.iterrows()):
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
                skipped += 1

        elapsed = time.time() - t0
        X = np.array(X_list)
        y = np.array(y_list)
        print(f"  Features: {X.shape[1]}d, extracted in {elapsed:.1f}s, skipped: {skipped}")

        # Save cache for future runs
        df_save = pd.DataFrame(X, columns=feature_names)
        df_save["tg_k"] = y
        df_save["smiles"] = smi_list
        cache.parent.mkdir(parents=True, exist_ok=True)
        df_save.to_parquet(cache, index=False)
        print(f"  Cached: {cache}")

    nan_mask = np.any(np.isnan(X), axis=1)
    n_nan = nan_mask.sum()
    print(f"  NaN rows: {n_nan}")

    # CatBoost handles NaN natively — only drop all-NaN rows
    all_nan = np.all(np.isnan(X), axis=1)
    if all_nan.sum() > 0:
        print(f"  Dropping {all_nan.sum()} all-NaN rows")
        keep = ~all_nan
        X = X[keep]
        y = y[keep]

    print(f"  Final: {len(X)} samples, {X.shape[1]} features")

    # Report per-component NaN rates for chain_physics
    if "chain_physics" in layer.upper() or layer == "PHY-C":
        cp_cols = [i for i, n in enumerate(feature_names) if n.startswith("CP_")]
        if cp_cols:
            cp_nan = np.isnan(X[:, cp_cols]).any(axis=1).sum()
            print(f"  chain_physics NaN: {cp_nan}/{len(X)} ({100*cp_nan/len(X):.1f}%)")

    return X, y, feature_names


def run_experiment(layer: str, quick: bool = False):
    """Run CatBoost Nested CV on a feature layer."""
    X, y, feat_names = load_and_featurize(layer)

    pp = build_preprocessing()
    X_pp = pp.fit_transform(X)

    estimator = get_estimator("CatBoost")
    outer_splits = 3 if quick else 5
    outer_repeats = 1 if quick else 3

    print(f"\n  Running CatBoost Nested CV ({outer_splits}x{outer_repeats} folds)...")
    t0 = time.time()
    result = nested_cv_no_tuning(
        X_pp, y, estimator,
        outer_splits=outer_splits,
        outer_repeats=outer_repeats,
        verbose=True,
    )
    elapsed = time.time() - t0

    metrics = result["metrics"]
    r2 = metrics["R2_mean"]
    r2_std = metrics["R2_std"]
    mae = metrics["MAE_mean"]
    mae_std = metrics["MAE_std"]

    print(f"\n  {'='*50}")
    print(f"  {layer} ({len(feat_names)}d) CatBoost Results:")
    print(f"  R2  = {r2:.4f} +/- {r2_std:.4f}")
    print(f"  MAE = {mae:.1f}K +/- {mae_std:.1f}K")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  {'='*50}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_result(
        {
            "experiment": f"PhaseC_{layer}_CatBoost",
            "layer": layer,
            "n_features": len(feat_names),
            "n_samples": len(X),
            "feature_names": feat_names,
            "metrics": metrics,
            "model": "CatBoost",
            "outer_cv": f"{outer_splits}x{outer_repeats}",
            "elapsed_seconds": elapsed,
        },
        str(RESULT_DIR / f"ablation_{layer}.json"),
    )

    return r2, r2_std, mae, mae_std


def main():
    parser = argparse.ArgumentParser(description="Phase C ablation study")
    parser.add_argument(
        "--layer", nargs="+", default=["PHY", "PHY-B2", "PHY-C"],
        help="Feature layers to compare (default: PHY PHY-B2 PHY-C)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 3x1 CV instead of 5x3",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase C Ablation: PHY → PHY-B2 → PHY-C")
    print("=" * 60)

    results = {}
    for layer in args.layer:
        r2, r2_std, mae, mae_std = run_experiment(layer, quick=args.quick)
        results[layer] = {
            "R2": r2, "R2_std": r2_std,
            "MAE": mae, "MAE_std": mae_std,
        }

    # Summary table
    print(f"\n{'='*65}")
    print("  Ablation Summary")
    print(f"{'='*65}")
    print(f"  {'Layer':12s} {'Dim':>4s} {'R2':>12s} {'MAE (K)':>14s} {'Delta R2':>10s}")
    print(f"  {'-'*55}")

    baseline_r2 = None
    for layer in args.layer:
        m = results[layer]
        dim = len(get_feature_names(layer))
        delta = ""
        if baseline_r2 is not None:
            d = m["R2"] - baseline_r2
            delta = f"{d:+.4f}"
        else:
            baseline_r2 = m["R2"]
            delta = "baseline"
        print(
            f"  {layer:12s} {dim:4d} "
            f"{m['R2']:.4f}+-{m['R2_std']:.4f} "
            f"{m['MAE']:.1f}+-{m['MAE_std']:.1f}K "
            f"{delta:>10s}"
        )


if __name__ == "__main__":
    main()
