"""
Sklearn-based ML Models for Tg Prediction
基于 sklearn 的 Tg 预测模型库

Provides:
    1. Model zoo: ExtraTrees, GBR, SVR, KRR, Ridge
    2. Preprocessing pipeline: PowerTransformer + MinMaxScaler
    3. Stacking ensemble: ExtraTrees + GBR + SVR → Ridge
    4. Hyperparameter search space definitions

Public API:
    get_model_zoo()          → dict of name → (estimator, param_grid)
    build_preprocessing()    → sklearn Pipeline (PowerTransformer + MinMaxScaler)
    build_stacking_model()   → StackingRegressor
    get_search_space(name)   → dict of param distributions
"""

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# 1. Preprocessing / 预处理
# ---------------------------------------------------------------------------

def build_preprocessing() -> Pipeline:
    """Build preprocessing pipeline: Yeo-Johnson PowerTransformer + MinMaxScaler.
    构建预处理管线：Yeo-Johnson 幂变换 + MinMax 缩放 (Xu 2024 最佳实践)。

    Returns:
        sklearn Pipeline with two steps.
    """
    return Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", MinMaxScaler()),
    ])


# ---------------------------------------------------------------------------
# 2. Model zoo / 模型动物园
# ---------------------------------------------------------------------------

def get_model_zoo() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Return dict of model_name → (estimator_instance, default_params).
    返回模型名 → (估计器实例, 默认参数) 的字典。

    Models included (ranked by literature for ~300 sample datasets):
        1. ExtraTrees — SOTA for small polymer datasets
        2. GBR — strong baseline, good interpretability
        3. SVR — kernel-based, good with proper scaling
        4. KRR — kernel ridge, alternative to SVR
        5. Ridge — linear baseline / meta-learner for stacking
    """
    return {
        "ExtraTrees": (
            ExtraTreesRegressor(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            {
                "n_estimators": [300, 500, 800],
                "max_features": ["sqrt", "log2", 0.3, 0.5],
                "min_samples_leaf": [1, 2, 3, 5],
                "max_depth": [None, 15, 20, 30],
            },
        ),
        "GBR": (
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            ),
            {
                "n_estimators": [200, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6],
                "min_samples_leaf": [3, 5, 8],
                "subsample": [0.7, 0.8, 0.9],
            },
        ),
        "SVR": (
            SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
            {
                "C": [10, 50, 100, 500],
                "gamma": ["scale", "auto", 0.01, 0.001],
                "epsilon": [0.05, 0.1, 0.2],
            },
        ),
        "KRR": (
            KernelRidge(alpha=1.0, kernel="rbf", gamma=None),
            {
                "alpha": [0.1, 1.0, 10.0],
                "gamma": [None, 0.01, 0.001, 0.0001],
                "kernel": ["rbf", "laplacian"],
            },
        ),
        "Ridge": (
            Ridge(alpha=1.0),
            {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        ),
    }


# ---------------------------------------------------------------------------
# 3. Stacking ensemble / 堆叠集成
# ---------------------------------------------------------------------------

def build_stacking_model(
    et_params: Dict[str, Any] | None = None,
    gbr_params: Dict[str, Any] | None = None,
    svr_params: Dict[str, Any] | None = None,
    meta_alpha: float = 1.0,
    cv: int = 3,
) -> StackingRegressor:
    """Build Stacking ensemble: ExtraTrees + GBR + SVR → Ridge meta-learner.
    构建 Stacking 集成：ExtraTrees + GBR + SVR → Ridge 元学习器。

    Args:
        et_params: Override ExtraTrees params.
        gbr_params: Override GBR params.
        svr_params: Override SVR params.
        meta_alpha: Ridge alpha for meta-learner.
        cv: Number of CV folds for stacking.

    Returns:
        StackingRegressor ready for fit/predict.
    """
    et_defaults = {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }
    gbr_defaults = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_leaf": 5,
        "subsample": 0.8,
        "random_state": 42,
    }
    svr_defaults = {
        "kernel": "rbf",
        "C": 100,
        "gamma": "scale",
        "epsilon": 0.1,
    }

    et = ExtraTreesRegressor(**(et_defaults | (et_params or {})))
    gbr = GradientBoostingRegressor(**(gbr_defaults | (gbr_params or {})))
    svr = SVR(**(svr_defaults | (svr_params or {})))

    return StackingRegressor(
        estimators=[
            ("et", et),
            ("gbr", gbr),
            ("svr", svr),
        ],
        final_estimator=Ridge(alpha=meta_alpha),
        cv=cv,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 4. Hyperparameter search spaces / 超参数搜索空间
# ---------------------------------------------------------------------------

def get_search_space(model_name: str) -> Dict[str, Any]:
    """Return hyperparameter search space for a given model.
    返回给定模型的超参数搜索空间。

    Args:
        model_name: One of 'ExtraTrees', 'GBR', 'SVR', 'KRR', 'Ridge'.

    Returns:
        Dict of param_name → list of values for GridSearchCV/RandomizedSearchCV.
    """
    zoo = get_model_zoo()
    if model_name not in zoo:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(zoo.keys())}"
        )
    return zoo[model_name][1]


def get_estimator(model_name: str, **overrides) -> Any:
    """Get a fresh estimator instance with optional param overrides.
    获取一个新的估计器实例（可选参数覆盖）。

    Args:
        model_name: Model name from the zoo.
        **overrides: Parameter overrides.

    Returns:
        sklearn estimator instance.
    """
    zoo = get_model_zoo()
    if model_name not in zoo:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(zoo.keys())}"
        )
    estimator = zoo[model_name][0]
    if overrides:
        estimator.set_params(**overrides)
    return estimator
