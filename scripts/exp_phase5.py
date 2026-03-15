"""
Phase 5 Experiment Script: Algorithm Restructure on Unified Dataset (7,486 samples)
Phase 5 实验脚本：统一数据集上的算法重构实验 (E16-E24)

Experiment Matrix:
    E16: GBR baseline (Phase 4 → Phase 5 迁移)
    E17: CatBoost (ordered boosting, primary model)
    E18: LightGBM (histogram-based GBDT)
    E19: TabPFN v2 (zero-tuning DL baseline) — OPTIONAL
    E20: ExtraTrees (Phase 4 主模型对照)
    E21: Stacking v2 (CatBoost + LightGBM + ExtraTrees + XGBoost → Ridge)
    E22: Best model + MAPIE CQR uncertainty quantification
    E23: Best model + feature selection (Boruta-SHAP → Top-K)
    E24: XGBoost (mature GBDT, strong baseline)

Usage:
    python scripts/exp_phase5.py                     # 运行全部实验
    python scripts/exp_phase5.py --exp E17           # 运行单个实验
    python scripts/exp_phase5.py --exp E16 E17 E18   # 运行多个实验
    python scripts/exp_phase5.py --no-optuna         # 跳过 Optuna 调参 (快速模式)
    python scripts/exp_phase5.py --n-trials 20       # 自定义 Optuna 试验次数
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Silence joblib/loky wmic deprecation warning on Windows 11
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message=".*wmic.*", category=UserWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.external_datasets import load_unified_dataset
from src.features.feature_pipeline import compute_features, get_feature_names
from src.ml.sklearn_models import (
    build_preprocessing,
    build_stacking_v2,
    get_estimator,
)
from src.ml.evaluation import (
    holdout_evaluate,
    nested_cv_no_tuning,
    nested_cv_optuna,
    save_result,
    simple_cv,
)
from src.ml.uncertainty import (
    run_cqr_evaluation,
    run_cross_conformal_evaluation,
    get_quantile_estimator,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_LAYER = "M2M-V"  # 46-dim: Afsordeh(4) + RDKit(15) + HBond(15) + VPD(12)
RESULT_DIR = PROJECT_ROOT / "results" / "phase5"
DATA_PATH = PROJECT_ROOT / "data" / "unified_tg.parquet"

# Optuna defaults
DEFAULT_N_TRIALS = 50
OUTER_SPLITS = 5
OUTER_REPEATS = 3


# ---------------------------------------------------------------------------
# Data loading and feature extraction
# ---------------------------------------------------------------------------

def load_and_featurize(
    layer: str = FEATURE_LAYER,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load unified dataset and compute features.

    Returns:
        (X_train, y_train, X_test, y_test, feature_names)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  加载统一数据集: {DATA_PATH}")
        print(f"  特征层级: {layer}")
        print(f"{'='*60}")

    df_train = load_unified_dataset(str(DATA_PATH), split="train")
    df_test = load_unified_dataset(str(DATA_PATH), split="test")

    if verbose:
        print(f"  训练集: {len(df_train)} 条")
        print(f"  测试集: {len(df_test)} 条")

    feature_names = get_feature_names(layer)

    def extract_features(df, label: str) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        skipped = 0
        t0 = time.time()

        for idx, row in df.iterrows():
            try:
                x = compute_features(row["smiles"], layer=layer)
                if np.any(np.isnan(x)):
                    skipped += 1
                    continue
                X_list.append(x)
                y_list.append(float(row["tg_k"]))
            except Exception:
                skipped += 1

        elapsed = time.time() - t0
        if verbose:
            print(f"  {label}: {len(X_list)} 条特征提取完成 "
                  f"(跳过 {skipped}, 耗时 {elapsed:.1f}s)")

        return np.array(X_list), np.array(y_list)

    X_train, y_train = extract_features(df_train, "训练集")
    X_test, y_test = extract_features(df_test, "测试集")

    if verbose:
        print(f"  特征维度: {X_train.shape[1]} ({layer})")
        print(f"  Tg 范围: [{y_train.min():.0f}, {y_train.max():.0f}] K (train)")

    return X_train, y_train, X_test, y_test, feature_names


def preprocess(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply PowerTransformer + MinMaxScaler preprocessing."""
    pp = build_preprocessing()
    X_train_pp = pp.fit_transform(X_train)
    X_test_pp = pp.transform(X_test)
    return X_train_pp, X_test_pp


# ---------------------------------------------------------------------------
# Helper: extract best estimator from Optuna CV results
# ---------------------------------------------------------------------------

def _best_estimator_from_cv(cv_result: Dict, model_name: str) -> Any:
    """Extract best Optuna params from nested CV and create tuned estimator.

    Picks the fold with highest R2, uses its Optuna best_params to create
    a fresh estimator via get_estimator(). Note: study.best_params contains
    only Optuna-searched params; get_estimator supplies remaining defaults
    from the model zoo.
    """
    best_fold_idx = max(
        range(len(cv_result["fold_results"])),
        key=lambda i: cv_result["fold_results"][i]["R2"],
    )
    best_params = cv_result["best_params"][best_fold_idx]
    return get_estimator(model_name, **best_params)


# ---------------------------------------------------------------------------
# Individual experiment functions
# ---------------------------------------------------------------------------

def run_e16_gbr(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    use_optuna: bool = True,
) -> Dict[str, Any]:
    """E16: GBR baseline — Phase 4 → Phase 5 migration benchmark."""
    print(f"\n{'='*60}")
    print("  E16: GBR 基线 (Phase 4 → Phase 5 迁移)")
    print(f"{'='*60}")

    estimator = get_estimator("GBR")

    # Nested CV with Optuna on training set
    if use_optuna:
        cv_result = nested_cv_optuna(
            X_train, y_train, "GBR",
            outer_splits=OUTER_SPLITS, outer_repeats=OUTER_REPEATS,
            n_trials=n_trials,
        )
        estimator = _best_estimator_from_cv(cv_result, "GBR")
    else:
        cv_result = nested_cv_no_tuning(X_train, y_train, estimator)

    # Holdout evaluation on test set
    holdout_result = holdout_evaluate(X_train, y_train, X_test, y_test, estimator)

    result = {
        "experiment": "E16",
        "model": "GBR",
        "layer": FEATURE_LAYER,
        "n_features": X_train.shape[1],
        "cv": cv_result,
        "holdout": holdout_result,
    }
    save_result(result, str(RESULT_DIR / "exp_E16_gbr.json"))
    return result


def run_e17_catboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    use_optuna: bool = True,
) -> Dict[str, Any]:
    """E17: CatBoost — ordered boosting, primary model candidate."""
    print(f"\n{'='*60}")
    print("  E17: CatBoost (ordered boosting, 主模型候选)")
    print(f"{'='*60}")

    estimator = get_estimator("CatBoost")

    if use_optuna:
        cv_result = nested_cv_optuna(
            X_train, y_train, "CatBoost",
            outer_splits=OUTER_SPLITS, outer_repeats=OUTER_REPEATS,
            n_trials=n_trials,
        )
        estimator = _best_estimator_from_cv(cv_result, "CatBoost")
    else:
        cv_result = nested_cv_no_tuning(X_train, y_train, estimator)

    holdout_result = holdout_evaluate(X_train, y_train, X_test, y_test, estimator)

    result = {
        "experiment": "E17",
        "model": "CatBoost",
        "layer": FEATURE_LAYER,
        "n_features": X_train.shape[1],
        "cv": cv_result,
        "holdout": holdout_result,
    }
    save_result(result, str(RESULT_DIR / "exp_E17_catboost.json"))
    return result


def run_e18_lightgbm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    use_optuna: bool = True,
) -> Dict[str, Any]:
    """E18: LightGBM — histogram-based GBDT."""
    print(f"\n{'='*60}")
    print("  E18: LightGBM (直方图 GBDT)")
    print(f"{'='*60}")

    estimator = get_estimator("LightGBM")

    if use_optuna:
        cv_result = nested_cv_optuna(
            X_train, y_train, "LightGBM",
            outer_splits=OUTER_SPLITS, outer_repeats=OUTER_REPEATS,
            n_trials=n_trials,
        )
        estimator = _best_estimator_from_cv(cv_result, "LightGBM")
    else:
        cv_result = nested_cv_no_tuning(X_train, y_train, estimator)

    holdout_result = holdout_evaluate(X_train, y_train, X_test, y_test, estimator)

    result = {
        "experiment": "E18",
        "model": "LightGBM",
        "layer": FEATURE_LAYER,
        "n_features": X_train.shape[1],
        "cv": cv_result,
        "holdout": holdout_result,
    }
    save_result(result, str(RESULT_DIR / "exp_E18_lightgbm.json"))
    return result


def run_e19_tabpfn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """E19: TabPFN v2 — zero-tuning DL baseline.

    Requires: pip install tabpfn
    May fail if not installed — returns None in that case.
    """
    print(f"\n{'='*60}")
    print("  E19: TabPFN v2 (零调参 DL 基线)")
    print(f"{'='*60}")

    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("  [!] TabPFN 未安装。跳过 E19。")
        print("  安装: python -m pip install tabpfn")
        return None

    try:
        estimator = TabPFNRegressor()

        # TabPFN is zero-tuning: use nested_cv_no_tuning
        cv_result = nested_cv_no_tuning(X_train, y_train, estimator)

        holdout_result = holdout_evaluate(
            X_train, y_train, X_test, y_test, estimator
        )

        result = {
            "experiment": "E19",
            "model": "TabPFN_v2",
            "layer": FEATURE_LAYER,
            "n_features": X_train.shape[1],
            "cv": cv_result,
            "holdout": holdout_result,
        }
        save_result(result, str(RESULT_DIR / "exp_E19_tabpfn.json"))
        return result

    except Exception as e:
        print(f"  [!] TabPFN 运行失败: {e}")
        return None


def run_e20_extratrees(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    use_optuna: bool = True,
) -> Dict[str, Any]:
    """E20: ExtraTrees — Phase 4 primary model comparison."""
    print(f"\n{'='*60}")
    print("  E20: ExtraTrees (Phase 4 主模型对照)")
    print(f"{'='*60}")

    estimator = get_estimator("ExtraTrees")

    if use_optuna:
        cv_result = nested_cv_optuna(
            X_train, y_train, "ExtraTrees",
            outer_splits=OUTER_SPLITS, outer_repeats=OUTER_REPEATS,
            n_trials=n_trials,
        )
        estimator = _best_estimator_from_cv(cv_result, "ExtraTrees")
    else:
        cv_result = nested_cv_no_tuning(X_train, y_train, estimator)

    holdout_result = holdout_evaluate(X_train, y_train, X_test, y_test, estimator)

    result = {
        "experiment": "E20",
        "model": "ExtraTrees",
        "layer": FEATURE_LAYER,
        "n_features": X_train.shape[1],
        "cv": cv_result,
        "holdout": holdout_result,
    }
    save_result(result, str(RESULT_DIR / "exp_E20_extratrees.json"))
    return result


def run_e21_stacking(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict[str, Any]:
    """E21: Stacking v2 — CatBoost + LightGBM + ExtraTrees + XGBoost → Ridge."""
    print(f"\n{'='*60}")
    print("  E21: Stacking v2 (CatBoost+LightGBM+ExtraTrees+XGBoost → Ridge)")
    print(f"{'='*60}")

    stacking = build_stacking_v2()

    # Stacking is slow → use simple_cv (no inner Optuna loop)
    cv_result = simple_cv(
        X_train, y_train, stacking,
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS,
    )

    holdout_result = holdout_evaluate(
        X_train, y_train, X_test, y_test, stacking
    )

    result = {
        "experiment": "E21",
        "model": "Stacking_v2",
        "base_models": ["CatBoost", "LightGBM", "ExtraTrees", "XGBoost"],
        "meta_learner": "Ridge",
        "layer": FEATURE_LAYER,
        "n_features": X_train.shape[1],
        "cv": cv_result,
        "holdout": holdout_result,
    }
    save_result(result, str(RESULT_DIR / "exp_E21_stacking.json"))
    return result


def run_e22_uncertainty(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict[str, Any]:
    """E22: MAPIE CQR/CrossConformal uncertainty quantification."""
    print(f"\n{'='*60}")
    print("  E22: 不确定性量化 (MAPIE CQR + CrossConformal)")
    print(f"{'='*60}")

    results = {}

    # CQR with GBR (quantile loss)
    print("\n  --- CQR (GBR quantile) ---")
    gbr_q = get_quantile_estimator("GBR")
    cqr_result = run_cqr_evaluation(
        gbr_q, X_train, y_train, X_test, y_test,
        confidence_level=0.9,
        calib_fraction=0.2,
    )
    results["cqr_gbr"] = cqr_result

    # CQR with LightGBM (quantile loss)
    print("\n  --- CQR (LightGBM quantile) ---")
    lgbm_q = get_quantile_estimator("LightGBM")
    cqr_lgbm_result = run_cqr_evaluation(
        lgbm_q, X_train, y_train, X_test, y_test,
        confidence_level=0.9,
        calib_fraction=0.2,
    )
    results["cqr_lgbm"] = cqr_lgbm_result

    # CrossConformal with CatBoost (any model)
    print("\n  --- CrossConformal (CatBoost) ---")
    cb = get_estimator("CatBoost")
    cc_result = run_cross_conformal_evaluation(
        cb, X_train, y_train, X_test, y_test,
        confidence_level=0.9,
        cv=5,
    )
    results["cross_conformal_catboost"] = cc_result

    # CrossConformal with ExtraTrees
    print("\n  --- CrossConformal (ExtraTrees) ---")
    et = get_estimator("ExtraTrees")
    cc_et_result = run_cross_conformal_evaluation(
        et, X_train, y_train, X_test, y_test,
        confidence_level=0.9,
        cv=5,
    )
    results["cross_conformal_et"] = cc_et_result

    result = {
        "experiment": "E22",
        "description": "Uncertainty quantification with MAPIE 1.3.0",
        "layer": FEATURE_LAYER,
        "n_features": X_train.shape[1],
        "methods": results,
    }
    save_result(result, str(RESULT_DIR / "exp_E22_uncertainty.json"))
    return result


def run_e23_feature_selection(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """E23: Feature selection via SHAP importance → Top-K features."""
    print(f"\n{'='*60}")
    print("  E23: 特征选择 (SHAP 重要性 → Top-K)")
    print(f"{'='*60}")

    import shap

    # Train CatBoost to get SHAP values
    print("  训练 CatBoost 计算 SHAP 值...")
    cb = get_estimator("CatBoost")
    cb.fit(X_train, y_train)

    explainer = shap.TreeExplainer(cb)
    shap_values = explainer.shap_values(X_train)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Rank features by SHAP importance
    importance_order = np.argsort(mean_abs_shap)[::-1]
    ranked_features = [
        (feature_names[i], float(mean_abs_shap[i]))
        for i in importance_order
    ]

    print("  Top-10 特征:")
    for rank, (name, score) in enumerate(ranked_features[:10], 1):
        print(f"    {rank}. {name}: {score:.2f}")

    # Evaluate with Top-K subsets
    results_by_k = {}
    for k in [10, 15, 20, 30, X_train.shape[1]]:
        if k > X_train.shape[1]:
            continue

        top_k_idx = importance_order[:k]
        X_train_k = X_train[:, top_k_idx]
        X_test_k = X_test[:, top_k_idx]

        print(f"\n  --- Top-{k} 特征 ---")
        cb_k = get_estimator("CatBoost")
        holdout_k = holdout_evaluate(X_train_k, y_train, X_test_k, y_test, cb_k)

        cv_k = simple_cv(X_train_k, y_train, cb_k, n_splits=5, n_repeats=1)

        results_by_k[f"top_{k}"] = {
            "k": k,
            "features": [feature_names[i] for i in top_k_idx],
            "holdout": holdout_k,
            "cv": cv_k,
        }

    result = {
        "experiment": "E23",
        "model": "CatBoost",
        "method": "SHAP_importance_ranking",
        "layer": FEATURE_LAYER,
        "n_features_original": X_train.shape[1],
        "feature_ranking": ranked_features,
        "results_by_k": results_by_k,
    }
    save_result(result, str(RESULT_DIR / "exp_E23_feature_selection.json"))
    return result


def run_e24_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_trials: int = DEFAULT_N_TRIALS,
    use_optuna: bool = True,
) -> Dict[str, Any]:
    """E24: XGBoost — mature GBDT baseline."""
    print(f"\n{'='*60}")
    print("  E24: XGBoost (成熟 GBDT 基线)")
    print(f"{'='*60}")

    estimator = get_estimator("XGBoost")

    if use_optuna:
        cv_result = nested_cv_optuna(
            X_train, y_train, "XGBoost",
            outer_splits=OUTER_SPLITS, outer_repeats=OUTER_REPEATS,
            n_trials=n_trials,
        )
        estimator = _best_estimator_from_cv(cv_result, "XGBoost")
    else:
        cv_result = nested_cv_no_tuning(X_train, y_train, estimator)

    holdout_result = holdout_evaluate(X_train, y_train, X_test, y_test, estimator)

    result = {
        "experiment": "E24",
        "model": "XGBoost",
        "layer": FEATURE_LAYER,
        "n_features": X_train.shape[1],
        "cv": cv_result,
        "holdout": holdout_result,
    }
    save_result(result, str(RESULT_DIR / "exp_E24_xgboost.json"))
    return result


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(all_results: Dict[str, Any]) -> str:
    """Generate markdown summary of all experiment results."""
    lines = [
        "# Phase 5 实验结果汇总",
        "",
        f"特征层级: {FEATURE_LAYER}",
        f"数据集: unified_tg.parquet",
        "",
        "## 模型对比 (Nested CV / Simple CV on train)",
        "",
        "| 实验 | 模型 | CV R² | CV MAE (K) | Holdout R² | Holdout MAE (K) |",
        "|------|------|-------|-----------|-----------|----------------|",
    ]

    for exp_id in ["E16", "E17", "E18", "E19", "E20", "E21", "E24"]:
        r = all_results.get(exp_id)
        if r is None:
            continue

        model = r.get("model", "?")

        cv = r.get("cv", {})
        cv_metrics = cv.get("metrics", {})
        cv_r2 = cv_metrics.get("R2_mean", float("nan"))
        cv_mae = cv_metrics.get("MAE_mean", float("nan"))

        ho = r.get("holdout", {})
        ho_test = ho.get("test_metrics", {})
        ho_r2 = ho_test.get("R2", float("nan"))
        ho_mae = ho_test.get("MAE", float("nan"))

        lines.append(
            f"| {exp_id} | {model} | {cv_r2:.4f} | {cv_mae:.1f} | "
            f"{ho_r2:.4f} | {ho_mae:.1f} |"
        )

    # UQ summary
    if "E22" in all_results:
        lines.extend(["", "## 不确定性量化 (E22)", ""])
        lines.append("| 方法 | 覆盖率 | 平均区间宽度 (K) | 相对宽度 | 点预测 R² |")
        lines.append("|------|--------|-----------------|---------|----------|")

        e22 = all_results["E22"]
        for method_name, method_result in e22.get("methods", {}).items():
            cov = method_result.get("coverage", {})
            pm = method_result.get("point_metrics", {})
            lines.append(
                f"| {method_name} | {cov.get('coverage', 0):.3f} | "
                f"{cov.get('avg_width', 0):.1f} | "
                f"{cov.get('avg_width_relative', 0):.3f} | "
                f"{pm.get('R2', 0):.4f} |"
            )

    # Feature selection summary
    if "E23" in all_results:
        lines.extend(["", "## 特征选择 (E23)", ""])
        lines.append("| Top-K | Holdout R² | CV R² |")
        lines.append("|-------|-----------|-------|")

        e23 = all_results["E23"]
        for k_key, k_result in sorted(
            e23.get("results_by_k", {}).items(),
            key=lambda x: x[1].get("k", 0)
        ):
            ho = k_result.get("holdout", {}).get("test_metrics", {})
            cv = k_result.get("cv", {}).get("metrics", {})
            lines.append(
                f"| {k_result.get('k', '?')} | "
                f"{ho.get('R2', 0):.4f} | "
                f"{cv.get('R2_mean', 0):.4f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5 experiments")
    parser.add_argument(
        "--exp", nargs="*", default=None,
        help="Experiment IDs to run (e.g., E16 E17). Default: all."
    )
    parser.add_argument(
        "--no-optuna", action="store_true",
        help="Skip Optuna tuning (quick mode)."
    )
    parser.add_argument(
        "--n-trials", type=int, default=DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials (default: {DEFAULT_N_TRIALS})."
    )
    parser.add_argument(
        "--skip-tabpfn", action="store_true",
        help="Skip TabPFN experiment."
    )
    args = parser.parse_args()

    # Determine which experiments to run
    all_exp_ids = ["E16", "E17", "E18", "E19", "E20", "E21", "E22", "E23", "E24"]
    if args.exp:
        run_ids = [e.upper() for e in args.exp]
        invalid = [e for e in run_ids if e not in all_exp_ids]
        if invalid:
            print(f"未知实验: {invalid}. 可用: {all_exp_ids}")
            sys.exit(1)
    else:
        run_ids = all_exp_ids

    if args.skip_tabpfn and "E19" in run_ids:
        run_ids.remove("E19")

    use_optuna = not args.no_optuna
    n_trials = args.n_trials

    print(f"\n{'#'*60}")
    print(f"  Phase 5: 算法重构实验")
    print(f"  实验: {run_ids}")
    print(f"  Optuna: {'开启' if use_optuna else '关闭'} (n_trials={n_trials})")
    print(f"{'#'*60}")

    # Suppress known harmless warnings (keep targeted, not blanket)
    warnings.filterwarnings("ignore", message=".*wmic.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*deprecated.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*PYTORCH_CUDA_ALLOC_CONF.*")

    # Load data and extract features (shared across all experiments)
    t_start = time.time()
    X_train, y_train, X_test, y_test, feature_names = load_and_featurize()

    # Apply preprocessing
    print("\n  应用预处理 (PowerTransformer + MinMaxScaler)...")
    X_train_pp, X_test_pp = preprocess(X_train, X_test)

    all_results: Dict[str, Any] = {}

    # --- Run experiments ---
    exp_dispatch = {
        "E16": lambda: run_e16_gbr(
            X_train_pp, y_train, X_test_pp, y_test, n_trials, use_optuna),
        "E17": lambda: run_e17_catboost(
            X_train_pp, y_train, X_test_pp, y_test, n_trials, use_optuna),
        "E18": lambda: run_e18_lightgbm(
            X_train_pp, y_train, X_test_pp, y_test, n_trials, use_optuna),
        "E19": lambda: run_e19_tabpfn(
            X_train_pp, y_train, X_test_pp, y_test),
        "E20": lambda: run_e20_extratrees(
            X_train_pp, y_train, X_test_pp, y_test, n_trials, use_optuna),
        "E21": lambda: run_e21_stacking(
            X_train_pp, y_train, X_test_pp, y_test),
        "E22": lambda: run_e22_uncertainty(
            X_train_pp, y_train, X_test_pp, y_test),
        "E23": lambda: run_e23_feature_selection(
            X_train_pp, y_train, X_test_pp, y_test, feature_names),
        "E24": lambda: run_e24_xgboost(
            X_train_pp, y_train, X_test_pp, y_test, n_trials, use_optuna),
    }

    for exp_id in run_ids:
        try:
            result = exp_dispatch[exp_id]()
            if result is not None:
                all_results[exp_id] = result
        except Exception as e:
            print(f"\n  [X] {exp_id} 失败: {e}")
            import traceback
            traceback.print_exc()

    # --- Generate summary ---
    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  全部实验完成! 总耗时: {t_total:.0f}s ({t_total/60:.1f}min)")
    print(f"{'='*60}")

    if all_results:
        summary = generate_summary(all_results)
        summary_path = RESULT_DIR / "phase5_summary.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n  汇总已保存: {summary_path}")
        # Safe print for Windows GBK terminals
        print(summary.encode("gbk", errors="replace").decode("gbk"))


if __name__ == "__main__":
    main()
