"""
Uncertainty Quantification for Tg Prediction (Phase 5)
不确定性量化 — 基于 MAPIE 1.3.0 的保形预测区间

Two approaches:
    1. CQR (Conformalized Quantile Regression):
       For models supporting quantile loss (GBR, LightGBM).
       Produces asymmetric, adaptive intervals. Best quality.

    2. CrossConformal:
       For ANY sklearn-compatible model (CatBoost, ExtraTrees, etc.).
       Uses cross-conformal + absolute residual scores. Symmetric intervals.

Public API:
    fit_cqr(estimator, X_train, y_train, X_calib, y_calib, ...)
    fit_cross_conformal(estimator, X, y, ...)
    predict_interval(model, X_test) -> (y_pred, y_lower, y_upper)
    evaluate_coverage(y_true, y_lower, y_upper) -> dict
    get_quantile_estimator(model_name, **kwargs) -> estimator

Reference: Romano et al. (2019), Conformalized Quantile Regression.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.base import RegressorMixin


# ---------------------------------------------------------------------------
# 1. Quantile estimator factory
# ---------------------------------------------------------------------------

# Models that natively support quantile loss for CQR
_QUANTILE_CONFIGS = {
    "GBR": {
        "class": "sklearn.ensemble.GradientBoostingRegressor",
        "quantile_param": {"loss": "quantile"},
        "defaults": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    },
    "LightGBM": {
        "class": "lightgbm.LGBMRegressor",
        "quantile_param": {"objective": "quantile", "alpha": 0.5},
        "defaults": {"n_estimators": 200, "verbose": -1},
    },
}


def get_quantile_estimator(
    model_name: str = "GBR",
    random_state: int = 42,
    **kwargs,
) -> RegressorMixin:
    """Create an estimator configured for quantile regression (CQR-compatible).

    Args:
        model_name: "GBR" or "LightGBM".
        random_state: Random seed.
        **kwargs: Override default hyperparameters.

    Returns:
        Estimator with quantile loss pre-configured.

    Raises:
        ValueError: If model_name not supported for quantile regression.
    """
    if model_name not in _QUANTILE_CONFIGS:
        supported = ", ".join(_QUANTILE_CONFIGS.keys())
        raise ValueError(
            f"CQR 不支持 '{model_name}'。支持的模型: {supported}。"
            f"其他模型请使用 fit_cross_conformal()。"
        )

    config = _QUANTILE_CONFIGS[model_name]
    params = {**config["defaults"], **config["quantile_param"], "random_state": random_state}
    params.update(kwargs)

    # Lazy import to avoid hard dependency
    module_path, class_name = config["class"].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    return cls(**params)


# ---------------------------------------------------------------------------
# 2. CQR fitting (quantile-capable models)
# ---------------------------------------------------------------------------

def fit_cqr(
    estimator: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    confidence_level: float = 0.9,
) -> Any:
    """Train + conformalize a CQR model.

    Uses MAPIE ConformalizedQuantileRegressor: fits 3 quantile models
    (lower, upper, median) on training data, then calibrates on a
    separate conformalization set.

    Args:
        estimator: Regressor with quantile loss (use get_quantile_estimator()).
        X_train: Training features.
        y_train: Training targets.
        X_calib: Calibration features (held-out from training).
        y_calib: Calibration targets.
        confidence_level: Target coverage probability (default 0.9 = 90%).

    Returns:
        Fitted ConformalizedQuantileRegressor object.
    """
    from mapie.regression import ConformalizedQuantileRegressor

    cqr = ConformalizedQuantileRegressor(
        estimator=estimator,
        confidence_level=confidence_level,
    )
    cqr.fit(X_train, y_train)
    cqr.conformalize(X_calib, y_calib)
    return cqr


# ---------------------------------------------------------------------------
# 3. CrossConformal fitting (any model)
# ---------------------------------------------------------------------------

def fit_cross_conformal(
    estimator: RegressorMixin,
    X: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.9,
    cv: int = 5,
    random_state: int = 42,
) -> Any:
    """Train a CrossConformal model for any sklearn-compatible estimator.

    Uses MAPIE CrossConformalRegressor with absolute residual scores.
    Does not require a separate calibration set (uses cross-validation).

    Args:
        estimator: Any sklearn-compatible regressor (CatBoost, ExtraTrees, etc.).
        X: Features (full training set).
        y: Targets.
        confidence_level: Target coverage probability (default 0.9).
        cv: Number of cross-validation folds for conformalization.
        random_state: Random seed.

    Returns:
        Fitted CrossConformalRegressor object.
    """
    from mapie.regression import CrossConformalRegressor

    ccr = CrossConformalRegressor(
        estimator=estimator,
        confidence_level=confidence_level,
        conformity_score="absolute",
        method="plus",
        cv=cv,
        random_state=random_state,
    )
    ccr.fit_conformalize(X, y)
    return ccr


# ---------------------------------------------------------------------------
# 4. Unified prediction
# ---------------------------------------------------------------------------

def predict_interval(
    model: Any,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict point estimates and confidence intervals.

    Works with both ConformalizedQuantileRegressor and CrossConformalRegressor.

    Args:
        model: Fitted MAPIE model (from fit_cqr or fit_cross_conformal).
        X_test: Test features.

    Returns:
        Tuple of (y_pred, y_lower, y_upper), each shape (n_samples,).
    """
    y_pred, y_intervals = model.predict_interval(X_test)

    # MAPIE returns intervals with shape (n_samples, 2, n_alpha)
    # For single confidence level: (n_samples, 2, 1) — squeeze last dim
    if y_intervals.ndim == 3:
        y_intervals = y_intervals[:, :, 0]

    y_lower = y_intervals[:, 0]
    y_upper = y_intervals[:, 1]

    return y_pred, y_lower, y_upper


# ---------------------------------------------------------------------------
# 5. Coverage evaluation
# ---------------------------------------------------------------------------

def evaluate_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> Dict[str, float]:
    """Evaluate prediction interval quality.

    Args:
        y_true: Ground truth targets.
        y_lower: Lower bounds of prediction intervals.
        y_upper: Upper bounds of prediction intervals.

    Returns:
        Dict with:
            - coverage: fraction of y_true within [y_lower, y_upper]
            - avg_width: mean interval width (K)
            - median_width: median interval width (K)
            - avg_width_relative: mean width / mean |y_true| (relative)
            - n_samples: number of test samples
    """
    y_true = np.asarray(y_true)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)

    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    widths = y_upper - y_lower

    coverage = float(np.mean(in_interval))
    avg_width = float(np.mean(widths))
    median_width = float(np.median(widths))

    mean_abs_y = float(np.mean(np.abs(y_true)))
    avg_width_relative = avg_width / mean_abs_y if mean_abs_y > 0 else float("inf")

    return {
        "coverage": coverage,
        "avg_width": avg_width,
        "median_width": median_width,
        "avg_width_relative": avg_width_relative,
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# 6. Convenience: full pipeline for experiment scripts
# ---------------------------------------------------------------------------

def run_cqr_evaluation(
    estimator: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    confidence_level: float = 0.9,
    calib_fraction: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end CQR evaluation: split calibration, fit, predict, evaluate.

    Splits X_train into train/calibration, fits CQR, predicts on X_test,
    and computes coverage metrics.

    Args:
        estimator: Quantile-capable estimator (use get_quantile_estimator()).
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        confidence_level: Target coverage (default 0.9).
        calib_fraction: Fraction of training data for calibration (default 0.2).
        random_state: Random seed.
        verbose: Print results.

    Returns:
        Dict with coverage metrics + predictions.
    """
    from sklearn.model_selection import train_test_split

    # Split training data into fit/calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train,
        test_size=calib_fraction,
        random_state=random_state,
    )

    if verbose:
        print(f"  CQR: 训练 {len(X_fit)} 条, 校准 {len(X_calib)} 条, 测试 {len(X_test)} 条")

    # Fit CQR
    model = fit_cqr(
        estimator, X_fit, y_fit, X_calib, y_calib,
        confidence_level=confidence_level,
    )

    # Predict intervals
    y_pred, y_lower, y_upper = predict_interval(model, X_test)

    # Evaluate
    coverage_metrics = evaluate_coverage(y_test, y_lower, y_upper)

    # Point prediction metrics
    from src.ml.evaluation import compute_metrics
    point_metrics = compute_metrics(y_test, y_pred)

    if verbose:
        print(f"  覆盖率: {coverage_metrics['coverage']:.3f} (目标: {confidence_level:.2f})")
        print(f"  平均区间宽度: {coverage_metrics['avg_width']:.1f} K")
        print(f"  相对区间宽度: {coverage_metrics['avg_width_relative']:.3f}")
        print(f"  点预测 R2: {point_metrics['R2']:.4f}")

    return {
        "confidence_level": confidence_level,
        "coverage": coverage_metrics,
        "point_metrics": point_metrics,
        "y_pred": y_pred.tolist(),
        "y_lower": y_lower.tolist(),
        "y_upper": y_upper.tolist(),
        "n_train": len(X_fit),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
    }


def run_cross_conformal_evaluation(
    estimator: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    confidence_level: float = 0.9,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end CrossConformal evaluation for any model.

    Args:
        estimator: Any sklearn-compatible regressor.
        X_train: Training features (full, no calibration split needed).
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        confidence_level: Target coverage (default 0.9).
        cv: CV folds for conformalization.
        random_state: Random seed.
        verbose: Print results.

    Returns:
        Dict with coverage metrics + predictions.
    """
    if verbose:
        print(f"  CrossConformal: 训练 {len(X_train)} 条 ({cv}-fold CV), 测试 {len(X_test)} 条")

    model = fit_cross_conformal(
        estimator, X_train, y_train,
        confidence_level=confidence_level,
        cv=cv,
        random_state=random_state,
    )

    y_pred, y_lower, y_upper = predict_interval(model, X_test)

    coverage_metrics = evaluate_coverage(y_test, y_lower, y_upper)

    from src.ml.evaluation import compute_metrics
    point_metrics = compute_metrics(y_test, y_pred)

    if verbose:
        print(f"  覆盖率: {coverage_metrics['coverage']:.3f} (目标: {confidence_level:.2f})")
        print(f"  平均区间宽度: {coverage_metrics['avg_width']:.1f} K")
        print(f"  相对区间宽度: {coverage_metrics['avg_width_relative']:.3f}")
        print(f"  点预测 R2: {point_metrics['R2']:.4f}")

    return {
        "confidence_level": confidence_level,
        "coverage": coverage_metrics,
        "point_metrics": point_metrics,
        "y_pred": y_pred.tolist(),
        "y_lower": y_lower.tolist(),
        "y_upper": y_upper.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "cv_folds": cv,
    }
