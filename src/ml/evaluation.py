"""
Nested Cross-Validation for Tg Prediction (Phase 5 — Algorithm Restructure)
嵌套交叉验证 — Tg 预测评估框架（Phase 5 — 算法重构）

Provides:
    1. Nested CV with RepeatedKFold outer + KFold inner (RandomizedSearchCV)
    2. Nested CV with Optuna TPE inner loop (Phase 5 new)
    3. Simple CV without tuning (for TabPFN v2 / fixed-param models)
    4. Holdout evaluation (for unified dataset train/test split)
    5. Metrics: R², MAE, RMSE, MAPE
    6. Result aggregation and reporting

Reference: Xu et al. (2024) — Nested CV best practice for small datasets.

Public API:
    nested_cv(X, y, estimator, param_space, ...)       → dict (RandomizedSearchCV inner)
    nested_cv_optuna(X, y, model_name, ...)             → dict (Optuna TPE inner)
    nested_cv_no_tuning(X, y, estimator, ...)           → dict (no inner loop)
    holdout_evaluate(X_train, y_train, X_test, y_test)  → dict (single split)
    simple_cv(X, y, estimator, ...)                     → dict (no tuning)
    compute_metrics(y_true, y_pred)                     → dict of metric values
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import (
    RepeatedKFold,
    KFold,
    cross_val_predict,
    RandomizedSearchCV,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import clone


# ---------------------------------------------------------------------------
# 1. Metrics / 评估指标
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: R², MAE, RMSE, MAPE.
    计算回归指标：R²、MAE、RMSE、MAPE。

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with R2, MAE, RMSE, MAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE — skip zeros
    mask = np.abs(y_true) > 1e-10
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {
        "R2": round(float(r2), 4),
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2),
    }


# ---------------------------------------------------------------------------
# 2. Nested CV / 嵌套交叉验证
# ---------------------------------------------------------------------------

def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    param_space: Dict[str, Any],
    outer_splits: int = 5,
    outer_repeats: int = 3,
    inner_splits: int = 3,
    n_iter: int = 15,
    scoring: str = "r2",
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run nested cross-validation with hyperparameter tuning.
    运行嵌套交叉验证（含超参数调优）。

    Outer loop: RepeatedKFold(n_splits=outer_splits, n_repeats=outer_repeats)
    Inner loop: RandomizedSearchCV with KFold(n_splits=inner_splits)

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        estimator: sklearn estimator (will be cloned).
        param_space: Hyperparameter search space for RandomizedSearchCV.
        outer_splits: Number of outer CV folds.
        outer_repeats: Number of outer CV repeats.
        inner_splits: Number of inner CV folds.
        n_iter: Number of random search iterations per outer fold.
        scoring: Scoring metric for inner CV.
        random_state: Random seed.
        verbose: Print progress.

    Returns:
        Dict with:
            - metrics: aggregated R², MAE, RMSE, MAPE (mean ± std)
            - fold_results: per-fold metrics
            - best_params: list of best params per outer fold
            - y_true_all, y_pred_all: concatenated predictions
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    outer_cv = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_state,
    )
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    fold_results: List[Dict[str, float]] = []
    best_params_list: List[Dict[str, Any]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    total_folds = outer_splits * outer_repeats
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner loop: hyperparameter tuning
        search = RandomizedSearchCV(
            estimator=clone(estimator),
            param_distributions=param_space,
            n_iter=min(n_iter, _count_combinations(param_space)),
            cv=inner_cv,
            scoring=scoring,
            random_state=random_state + fold_idx,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        # Predict on outer test fold
        y_pred = search.predict(X_test)

        # Compute metrics
        fold_metrics = compute_metrics(y_test, y_pred)
        fold_results.append(fold_metrics)
        best_params_list.append(search.best_params_)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose and (fold_idx + 1) % outer_splits == 0:
            repeat_num = (fold_idx + 1) // outer_splits
            elapsed = time.time() - t0
            print(
                f"  Repeat {repeat_num}/{outer_repeats} done "
                f"(R2={fold_metrics['R2']:.4f}, "
                f"MAE={fold_metrics['MAE']:.1f}K, "
                f"time={elapsed:.1f}s)"
            )

    # Aggregate metrics
    metrics_agg = _aggregate_metrics(fold_results)
    overall = compute_metrics(y_true_all, y_pred_all)

    elapsed_total = time.time() - t0

    result = {
        "metrics": metrics_agg,
        "overall": overall,
        "fold_results": fold_results,
        "best_params": best_params_list,
        "n_folds": total_folds,
        "time_sec": round(elapsed_total, 1),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }

    if verbose:
        print(f"\n  === Nested CV Result ({total_folds} folds, {elapsed_total:.1f}s) ===")
        print(f"  R2:   {metrics_agg['R2_mean']:.4f} +/- {metrics_agg['R2_std']:.4f}")
        print(f"  MAE:  {metrics_agg['MAE_mean']:.1f} +/- {metrics_agg['MAE_std']:.1f} K")
        print(f"  RMSE: {metrics_agg['RMSE_mean']:.1f} +/- {metrics_agg['RMSE_std']:.1f} K")
        print(f"  Overall: R2={overall['R2']:.4f}, MAE={overall['MAE']:.1f}K")

    return result


# ---------------------------------------------------------------------------
# 3. Nested CV with Optuna / Optuna 内层调参嵌套交叉验证
# ---------------------------------------------------------------------------

def nested_cv_optuna(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    outer_splits: int = 5,
    outer_repeats: int = 3,
    inner_splits: int = 3,
    n_trials: int = 50,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run nested CV with Optuna TPE sampler for inner-loop tuning.
    使用 Optuna TPE 采样器进行内层调参的嵌套交叉验证。

    Outer loop: RepeatedKFold(n_splits, n_repeats)
    Inner loop: Optuna TPE with MedianPruner (n_trials per fold)

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        model_name: Model name from sklearn_models (CatBoost, LightGBM, etc.)
        outer_splits: Number of outer CV folds.
        outer_repeats: Number of outer CV repeats.
        inner_splits: Number of inner CV folds for Optuna objective.
        n_trials: Number of Optuna trials per outer fold.
        random_state: Random seed.
        verbose: Print progress.

    Returns:
        Dict with metrics, fold_results, best_params, y_true_all, y_pred_all.
    """
    import optuna
    from src.ml.sklearn_models import get_estimator, suggest_optuna_params

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    outer_cv = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_state,
    )

    fold_results: List[Dict[str, float]] = []
    best_params_list: List[Dict[str, Any]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    total_folds = outer_splits * outer_repeats
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner loop: Optuna objective
        inner_cv = KFold(
            n_splits=inner_splits, shuffle=True,
            random_state=random_state + fold_idx,
        )

        def objective(trial):
            params = suggest_optuna_params(trial, model_name)
            model = get_estimator(model_name, **params)

            scores = []
            for tr_idx, va_idx in inner_cv.split(X_train, y_train):
                X_tr, X_va = X_train[tr_idx], X_train[va_idx]
                y_tr, y_va = y_train[tr_idx], y_train[va_idx]

                model_clone = clone(model)
                model_clone.fit(X_tr, y_tr)
                y_va_pred = model_clone.predict(X_va)
                scores.append(r2_score(y_va, y_va_pred))

            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state + fold_idx),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Refit with best params on full outer train set
        best_params = suggest_optuna_params(study.best_trial, model_name)
        best_model = get_estimator(model_name, **best_params)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        fold_metrics = compute_metrics(y_test, y_pred)
        fold_results.append(fold_metrics)
        best_params_list.append(study.best_params)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose and (fold_idx + 1) % outer_splits == 0:
            repeat_num = (fold_idx + 1) // outer_splits
            elapsed = time.time() - t0
            print(
                f"  Repeat {repeat_num}/{outer_repeats} done "
                f"(R2={fold_metrics['R2']:.4f}, "
                f"MAE={fold_metrics['MAE']:.1f}K, "
                f"time={elapsed:.1f}s)"
            )

    metrics_agg = _aggregate_metrics(fold_results)
    overall = compute_metrics(y_true_all, y_pred_all)
    elapsed_total = time.time() - t0

    result = {
        "metrics": metrics_agg,
        "overall": overall,
        "fold_results": fold_results,
        "best_params": best_params_list,
        "n_folds": total_folds,
        "n_trials_per_fold": n_trials,
        "time_sec": round(elapsed_total, 1),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }

    if verbose:
        print(f"\n  === Nested CV + Optuna ({total_folds} folds, "
              f"{n_trials} trials/fold, {elapsed_total:.1f}s) ===")
        print(f"  R2:   {metrics_agg['R2_mean']:.4f} +/- {metrics_agg['R2_std']:.4f}")
        print(f"  MAE:  {metrics_agg['MAE_mean']:.1f} +/- {metrics_agg['MAE_std']:.1f} K")
        print(f"  RMSE: {metrics_agg['RMSE_mean']:.1f} +/- {metrics_agg['RMSE_std']:.1f} K")
        print(f"  Overall: R2={overall['R2']:.4f}, MAE={overall['MAE']:.1f}K")

    return result


# ---------------------------------------------------------------------------
# 4. Nested CV without tuning / 无调参嵌套交叉验证（TabPFN 等）
# ---------------------------------------------------------------------------

def nested_cv_no_tuning(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    outer_splits: int = 5,
    outer_repeats: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run nested CV without inner-loop tuning (for zero-tuning models like TabPFN).
    无内层调参的嵌套交叉验证（适用于 TabPFN v2 等零调参模型）。

    Output format matches nested_cv() / nested_cv_optuna() for consistency.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        estimator: sklearn-compatible estimator (will be cloned per fold).
        outer_splits: Number of CV folds.
        outer_repeats: Number of repeats.
        random_state: Random seed.
        verbose: Print progress.

    Returns:
        Dict with metrics, fold_results, y_true_all, y_pred_all.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    outer_cv = RepeatedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_state,
    )

    fold_results: List[Dict[str, float]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    total_folds = outer_splits * outer_repeats
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = compute_metrics(y_test, y_pred)
        fold_results.append(fold_metrics)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose and (fold_idx + 1) % outer_splits == 0:
            repeat_num = (fold_idx + 1) // outer_splits
            elapsed = time.time() - t0
            print(
                f"  Repeat {repeat_num}/{outer_repeats} done "
                f"(R2={fold_metrics['R2']:.4f}, "
                f"MAE={fold_metrics['MAE']:.1f}K, "
                f"time={elapsed:.1f}s)"
            )

    metrics_agg = _aggregate_metrics(fold_results)
    overall = compute_metrics(y_true_all, y_pred_all)
    elapsed_total = time.time() - t0

    result = {
        "metrics": metrics_agg,
        "overall": overall,
        "fold_results": fold_results,
        "best_params": None,  # No tuning
        "n_folds": total_folds,
        "time_sec": round(elapsed_total, 1),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }

    if verbose:
        print(f"\n  === CV No-Tuning ({total_folds} folds, {elapsed_total:.1f}s) ===")
        print(f"  R2:   {metrics_agg['R2_mean']:.4f} +/- {metrics_agg['R2_std']:.4f}")
        print(f"  MAE:  {metrics_agg['MAE_mean']:.1f} +/- {metrics_agg['MAE_std']:.1f} K")
        print(f"  RMSE: {metrics_agg['RMSE_mean']:.1f} +/- {metrics_agg['RMSE_std']:.1f} K")
        print(f"  Overall: R2={overall['R2']:.4f}, MAE={overall['MAE']:.1f}K")

    return result


# ---------------------------------------------------------------------------
# 5. Holdout evaluation / Holdout 评估（统一数据集 train/test split）
# ---------------------------------------------------------------------------

def holdout_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimator: Any,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate on a fixed holdout split (for unified dataset's 80/20 split).
    在固定的 holdout 集上评估（适用于统一数据集的 80/20 划分）。

    Args:
        X_train: Training feature matrix.
        y_train: Training targets.
        X_test: Test feature matrix.
        y_test: Test targets.
        estimator: sklearn-compatible estimator (will be cloned).
        verbose: Print results.

    Returns:
        Dict with train_metrics, test_metrics, model (fitted).
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    t0 = time.time()

    model = clone(estimator)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    elapsed = time.time() - t0

    result = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "time_sec": round(elapsed, 1),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "y_test_true": y_test.tolist(),
        "y_test_pred": y_test_pred.tolist(),
    }

    if verbose:
        print(f"\n  === Holdout Evaluation ({len(y_train)} train / {len(y_test)} test, "
              f"{elapsed:.1f}s) ===")
        print(f"  Train: R2={train_metrics['R2']:.4f}, MAE={train_metrics['MAE']:.1f}K")
        print(f"  Test:  R2={test_metrics['R2']:.4f}, MAE={test_metrics['MAE']:.1f}K, "
              f"RMSE={test_metrics['RMSE']:.1f}K")

    return result


# ---------------------------------------------------------------------------
# 6. Simple CV (no tuning) / 简单交叉验证（无调参）
# ---------------------------------------------------------------------------

def simple_cv(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run simple repeated K-fold CV without hyperparameter tuning.
    运行简单的重复 K 折交叉验证（无超参数调优）。

    Args:
        X: Feature matrix.
        y: Target vector.
        estimator: sklearn estimator (will be cloned per fold).
        n_splits: Number of CV folds.
        n_repeats: Number of repeats.
        random_state: Random seed.
        verbose: Print progress.

    Returns:
        Dict with metrics, fold_results, y_true_all, y_pred_all.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_results: List[Dict[str, float]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    t0 = time.time()

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = clone(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics = compute_metrics(y_test, y_pred)
        fold_results.append(fold_metrics)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    metrics_agg = _aggregate_metrics(fold_results)
    overall = compute_metrics(y_true_all, y_pred_all)
    elapsed = time.time() - t0

    result = {
        "metrics": metrics_agg,
        "overall": overall,
        "fold_results": fold_results,
        "n_folds": n_splits * n_repeats,
        "time_sec": round(elapsed, 1),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }

    if verbose:
        total = n_splits * n_repeats
        print(f"\n  === Simple CV Result ({total} folds, {elapsed:.1f}s) ===")
        print(f"  R2:   {metrics_agg['R2_mean']:.4f} +/- {metrics_agg['R2_std']:.4f}")
        print(f"  MAE:  {metrics_agg['MAE_mean']:.1f} +/- {metrics_agg['MAE_std']:.1f} K")
        print(f"  RMSE: {metrics_agg['RMSE_mean']:.1f} +/- {metrics_agg['RMSE_std']:.1f} K")

    return result


# ---------------------------------------------------------------------------
# 7. Result I/O / 结果保存
# ---------------------------------------------------------------------------

def save_result(result: Dict[str, Any], filepath: str) -> None:
    """Save experiment result to JSON file.
    保存实验结果到 JSON 文件。

    Args:
        result: Experiment result dict.
        filepath: Output file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    serializable = _make_serializable(result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"  Result saved: {filepath}")


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# 8. Internal helpers / 内部辅助函数
# ---------------------------------------------------------------------------

def _aggregate_metrics(fold_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute mean and std for each metric across folds.
    计算各指标在各折上的均值和标准差。
    """
    keys = fold_results[0].keys()
    agg = {}
    for key in keys:
        values = [f[key] for f in fold_results]
        agg[f"{key}_mean"] = round(float(np.mean(values)), 4)
        agg[f"{key}_std"] = round(float(np.std(values)), 4)
    return agg


def _count_combinations(param_space: Dict[str, Any]) -> int:
    """Count total combinations in a parameter space."""
    total = 1
    for values in param_space.values():
        if isinstance(values, list):
            total *= len(values)
        else:
            total *= 10  # estimate for distributions
    return total
