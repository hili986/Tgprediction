"""
Phase 4 M2M 消融实验 — E1-E8

Experiments:
    E1: L1H 34d + GBR          → 基线复现
    E2: M2M-P 44d + GBR        → PPF 增益
    E3: M2M-V 46d + GBR        → VPD 增益
    E4: M2M 56d + GBR          → PPF+VPD 联合
    E5: M2M-PV 22d + GBR       → 纯物理特征
    E6: M2M 56d + constrained  → 单调约束增益
    E7: M2M 56d + HRL          → 层级残差学习
    E8: M2M 56d + HRL+MAPIE    → 不确定性量化 (if MAPIE available)

Usage:
    python scripts/exp_phase4_m2m.py
    python scripts/exp_phase4_m2m.py --exp E4    # Run single experiment
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.feature_pipeline import build_dataset_v2, get_feature_names
from src.ml.evaluation import nested_cv, compute_metrics
from src.ml.constrained_gbr import build_constrained_gbr, get_constraint_summary
from src.ml.hierarchical_model import (
    HierarchicalTgPredictor,
    nested_cv_hrl,
)


RESULT_DIR = Path("results/phase4")


def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def _save_result(exp_id: str, result: dict):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULT_DIR / f"exp_{exp_id}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=convert, ensure_ascii=False)

    _safe_print(f"  -> Saved: {filepath}")


# ---------------------------------------------------------------------------
# GBR param space (for nested CV tuning)
# ---------------------------------------------------------------------------

GBR_PARAM_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "min_samples_leaf": [3, 5, 7, 10],
}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_e1():
    """E1: L1H 34d + GBR baseline."""
    _safe_print("\n=== E1: L1H 34d + GBR (baseline) ===")
    X, y, names, feat_names, _ = build_dataset_v2(layer="L1H")

    result = nested_cv(
        X, y,
        estimator=GradientBoostingRegressor(random_state=42),
        param_space=GBR_PARAM_SPACE,
    )
    result["experiment"] = "E1"
    result["layer"] = "L1H"
    result["n_features"] = len(feat_names)
    result["n_samples"] = X.shape[0]
    _save_result("E1_L1H_GBR", result)
    return result


def run_e2():
    """E2: M2M-P 44d + GBR (PPF only)."""
    _safe_print("\n=== E2: M2M-P 44d + GBR (PPF only) ===")
    X, y, names, feat_names, _ = build_dataset_v2(layer="M2M-P")

    result = nested_cv(
        X, y,
        estimator=GradientBoostingRegressor(random_state=42),
        param_space=GBR_PARAM_SPACE,
    )
    result["experiment"] = "E2"
    result["layer"] = "M2M-P"
    result["n_features"] = len(feat_names)
    result["n_samples"] = X.shape[0]
    _save_result("E2_M2M-P_GBR", result)
    return result


def run_e3():
    """E3: M2M-V 46d + GBR (VPD only)."""
    _safe_print("\n=== E3: M2M-V 46d + GBR (VPD only) ===")
    X, y, names, feat_names, _ = build_dataset_v2(layer="M2M-V")

    result = nested_cv(
        X, y,
        estimator=GradientBoostingRegressor(random_state=42),
        param_space=GBR_PARAM_SPACE,
    )
    result["experiment"] = "E3"
    result["layer"] = "M2M-V"
    result["n_features"] = len(feat_names)
    result["n_samples"] = X.shape[0]
    _save_result("E3_M2M-V_GBR", result)
    return result


def run_e4():
    """E4: M2M 56d + GBR (PPF + VPD combined)."""
    _safe_print("\n=== E4: M2M 56d + GBR (full M2M) ===")
    X, y, names, feat_names, _ = build_dataset_v2(layer="M2M")

    result = nested_cv(
        X, y,
        estimator=GradientBoostingRegressor(random_state=42),
        param_space=GBR_PARAM_SPACE,
    )
    result["experiment"] = "E4"
    result["layer"] = "M2M"
    result["n_features"] = len(feat_names)
    result["n_samples"] = X.shape[0]
    _save_result("E4_M2M_GBR", result)
    return result


def run_e5():
    """E5: M2M-PV 22d + GBR (pure physics features)."""
    _safe_print("\n=== E5: M2M-PV 22d + GBR (pure physics) ===")
    X, y, names, feat_names, _ = build_dataset_v2(layer="M2M-PV")

    result = nested_cv(
        X, y,
        estimator=GradientBoostingRegressor(random_state=42),
        param_space=GBR_PARAM_SPACE,
    )
    result["experiment"] = "E5"
    result["layer"] = "M2M-PV"
    result["n_features"] = len(feat_names)
    result["n_samples"] = X.shape[0]
    _save_result("E5_M2M-PV_GBR", result)
    return result


def run_e6():
    """E6: M2M 56d + constrained GBR."""
    _safe_print("\n=== E6: M2M 56d + Constrained GBR ===")
    X, y, names, feat_names, _ = build_dataset_v2(layer="M2M")

    # Build constrained model
    model = build_constrained_gbr(feat_names)
    summary = get_constraint_summary(feat_names)
    _safe_print(f"  Constrained features: +1={len(summary['positive'])}, "
                f"-1={len(summary['negative'])}, "
                f"0={len(summary['unconstrained'])}")

    # Constrained GBR has fixed hyperparams, use simple nested CV
    from sklearn.model_selection import RepeatedKFold
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    fold_results = []
    y_true_all, y_pred_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        m = build_constrained_gbr(feat_names)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        fold_results.append(metrics)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    r2_scores = [f["R2"] for f in fold_results]
    mae_scores = [f["MAE"] for f in fold_results]
    overall = compute_metrics(np.array(y_true_all), np.array(y_pred_all))

    result = {
        "experiment": "E6",
        "model": "ConstrainedGBR",
        "layer": "M2M",
        "n_features": len(feat_names),
        "n_samples": X.shape[0],
        "constraint_summary": summary,
        "metrics": {
            "R2_mean": round(float(np.mean(r2_scores)), 4),
            "R2_std": round(float(np.std(r2_scores)), 4),
            "MAE_mean": round(float(np.mean(mae_scores)), 2),
            "MAE_std": round(float(np.std(mae_scores)), 2),
            "R2_overall": overall["R2"],
            "MAE_overall": overall["MAE"],
        },
        "fold_results": fold_results,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }
    _safe_print(f"  E6 Result: R2={result['metrics']['R2_mean']:.4f} "
                f"+/- {result['metrics']['R2_std']:.4f}")
    _save_result("E6_M2M_ConstrainedGBR", result)
    return result


def run_e7():
    """E7: M2M 56d + HRL."""
    _safe_print("\n=== E7: M2M 56d + HRL ===")
    X, y, names, feat_names, _ = build_dataset_v2(layer="M2M")

    result = nested_cv_hrl(X, y, feat_names)
    result["experiment"] = "E7"
    result["layer"] = "M2M"
    result["n_features"] = len(feat_names)
    result["n_samples"] = X.shape[0]
    _save_result("E7_M2M_HRL", result)
    return result


def run_e8():
    """E8: M2M 56d + HRL + MAPIE (if available)."""
    _safe_print("\n=== E8: M2M 56d + HRL + MAPIE ===")
    try:
        from mapie.regression import SplitConformalRegressor
    except ImportError:
        _safe_print("  MAPIE not installed, skipping E8. pip install mapie")
        return None

    X, y, names, feat_names, _ = build_dataset_v2(layer="M2M")

    from sklearn.model_selection import RepeatedKFold, train_test_split
    outer_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

    fold_results = []
    coverage_90 = []
    y_true_all, y_pred_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        # Split training into fit + calibration (80/20)
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )

        # Fit HRL on training portion
        hrl = HierarchicalTgPredictor(random_state=42)
        hrl.fit(X_train, y_train, feature_names=feat_names)

        # Conformal prediction with prefit estimator
        mapie = SplitConformalRegressor(
            estimator=hrl, prefit=True, confidence_level=0.9
        )
        mapie.conformalize(X_cal, y_cal)

        y_pred, y_pi = mapie.predict_interval(X_test)

        metrics = compute_metrics(y_test, y_pred)
        fold_results.append(metrics)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        # Coverage: fraction of true values within 90% PI
        in_pi = ((y_test >= y_pi[:, 0, 0]) & (y_test <= y_pi[:, 1, 0]))
        coverage_90.append(float(np.mean(in_pi)))

    r2_scores = [f["R2"] for f in fold_results]
    mae_scores = [f["MAE"] for f in fold_results]
    overall = compute_metrics(np.array(y_true_all), np.array(y_pred_all))

    result = {
        "experiment": "E8",
        "model": "HRL+MAPIE",
        "layer": "M2M",
        "n_features": len(feat_names),
        "n_samples": X.shape[0],
        "metrics": {
            "R2_mean": round(float(np.mean(r2_scores)), 4),
            "R2_std": round(float(np.std(r2_scores)), 4),
            "MAE_mean": round(float(np.mean(mae_scores)), 2),
            "MAE_std": round(float(np.std(mae_scores)), 2),
            "R2_overall": overall["R2"],
            "MAE_overall": overall["MAE"],
            "coverage_90_mean": round(float(np.mean(coverage_90)), 4),
        },
        "fold_results": fold_results,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }
    _safe_print(f"  E8 Result: R2={result['metrics']['R2_mean']:.4f}, "
                f"90% coverage={result['metrics']['coverage_90_mean']:.1%}")
    _save_result("E8_M2M_HRL_MAPIE", result)
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    """Print comparison table."""
    _safe_print("\n" + "=" * 70)
    _safe_print("Phase 4 M2M Ablation Summary")
    _safe_print("=" * 70)
    _safe_print(f"{'Exp':>4} | {'Layer':<8} | {'Dim':>4} | {'Model':<16} | "
                f"{'R2 mean':>8} | {'R2 std':>7} | {'MAE':>6}")
    _safe_print("-" * 70)

    for exp_id, r in sorted(results.items()):
        if r is None:
            continue
        m = r.get("metrics", {})
        layer = r.get("layer", "?")
        n_feat = r.get("n_features", "?")
        model = r.get("model", "GBR")
        r2_mean = m.get("R2_mean", m.get("R2", "?"))
        r2_std = m.get("R2_std", "?")
        mae = m.get("MAE_mean", m.get("MAE", "?"))

        _safe_print(f"{exp_id:>4} | {layer:<8} | {n_feat:>4} | {model:<16} | "
                    f"{r2_mean:>8} | {r2_std:>7} | {mae:>6}")

    _safe_print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "E1": run_e1,
    "E2": run_e2,
    "E3": run_e3,
    "E4": run_e4,
    "E5": run_e5,
    "E6": run_e6,
    "E7": run_e7,
    "E8": run_e8,
}


def main():
    parser = argparse.ArgumentParser(description="Phase 4 M2M Ablation Experiments")
    parser.add_argument("--exp", type=str, default=None,
                        help="Run single experiment (E1-E8)")
    args = parser.parse_args()

    _safe_print("Phase 4: M2M-Deep Feature Engineering + Constrained Modeling")
    _safe_print(f"Output: {RESULT_DIR}")

    t0 = time.time()

    if args.exp:
        exp_id = args.exp.upper()
        if exp_id not in EXPERIMENTS:
            _safe_print(f"Unknown experiment: {exp_id}. Available: {list(EXPERIMENTS.keys())}")
            return
        result = EXPERIMENTS[exp_id]()
        if result:
            print_summary({exp_id: result})
    else:
        results = {}
        for exp_id, func in EXPERIMENTS.items():
            result = func()
            results[exp_id] = result

        print_summary(results)

    elapsed = time.time() - t0
    _safe_print(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
