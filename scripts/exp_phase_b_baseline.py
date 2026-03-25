"""
Phase B Baseline Test: PHY (48d) vs M2M-V (46d) Feature Comparison
Phase B 基线测试：物理增强特征 vs 旧特征的 Nested CV 对比

Usage:
    python scripts/exp_phase_b_baseline.py
    python scripts/exp_phase_b_baseline.py --layer PHY
    python scripts/exp_phase_b_baseline.py --layer M2M-V PHY --quick
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
RESULT_DIR = PROJECT_ROOT / "results" / "phase_b"


def load_and_featurize(layer: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load unified dataset and compute features (train+test combined for Nested CV)."""
    print(f"\n  Loading data and computing {layer} features...")
    df = load_unified_dataset(str(DATA_PATH), split=None)
    print(f"  Total samples: {len(df)}")

    feature_names = get_feature_names(layer)
    X_list, y_list = [], []
    skipped = 0
    t0 = time.time()

    for idx, row in df.iterrows():
        try:
            x = compute_features(row["smiles"], layer=layer)
            X_list.append(x)
            y_list.append(float(row["tg_k"]))
        except Exception:
            skipped += 1

    elapsed = time.time() - t0
    X = np.array(X_list)
    y = np.array(y_list)

    # Count NaN
    nan_mask = np.any(np.isnan(X), axis=1)
    n_nan = nan_mask.sum()
    print(f"  Features: {X.shape[1]}d, extracted in {elapsed:.1f}s")
    print(f"  Skipped: {skipped}, NaN rows: {n_nan}")

    # For non-CatBoost models, drop NaN rows
    if n_nan > 0:
        print(f"  Dropping {n_nan} NaN rows for preprocessing compatibility")
        keep = ~nan_mask
        X = X[keep]
        y = y[keep]

    print(f"  Final: {len(X)} samples, {X.shape[1]} features")
    return X, y, feature_names


def run_baseline(layer: str, quick: bool = False):
    """Run CatBoost Nested CV on a feature layer."""
    X, y, feat_names = load_and_featurize(layer)

    # Preprocess
    pp = build_preprocessing()
    X_pp = pp.fit_transform(X)

    # CatBoost with default params
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

    # Save result
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_result(
        {
            "experiment": f"PhaseB_{layer}_CatBoost",
            "layer": layer,
            "n_features": len(feat_names),
            "n_samples": len(X),
            "feature_names": feat_names,
            "metrics": metrics,
            "model": "CatBoost",
            "outer_cv": f"{outer_splits}x{outer_repeats}",
            "elapsed_seconds": elapsed,
        },
        str(RESULT_DIR / f"baseline_{layer}.json"),
    )

    return r2, mae


def main():
    parser = argparse.ArgumentParser(description="Phase B baseline comparison")
    parser.add_argument("--layer", nargs="+", default=["M2M-V", "PHY"],
                        help="Feature layers to compare")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3x1 CV instead of 5x3")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase B Baseline Test: Feature Set Comparison")
    print("=" * 60)

    results = {}
    for layer in args.layer:
        r2, mae = run_baseline(layer, quick=args.quick)
        results[layer] = {"R2": r2, "MAE": mae}

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  Comparison Summary")
        print(f"{'='*60}")
        print(f"  {'Layer':15s} {'Dim':>5s} {'R2':>8s} {'MAE':>8s}")
        print(f"  {'-'*40}")
        for layer, m in results.items():
            dim = len(get_feature_names(layer))
            print(f"  {layer:15s} {dim:5d} {m['R2']:8.4f} {m['MAE']:7.1f}K")


if __name__ == "__main__":
    main()
