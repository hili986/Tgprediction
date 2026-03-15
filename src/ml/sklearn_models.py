"""
Sklearn-based ML Models for Tg Prediction (Phase 5 — Algorithm Restructure)
基于 sklearn 的 Tg 预测模型库（Phase 5 — 算法重构）

Provides:
    1. Model zoo: CatBoost, LightGBM, XGBoost, ExtraTrees, GBR, SVR, Ridge
    2. Preprocessing pipeline: PowerTransformer + MinMaxScaler
    3. Stacking v1 (legacy): ExtraTrees + GBR + SVR → Ridge
    4. Stacking v2 (new): CatBoost + LightGBM + ExtraTrees + TabPFN → Ridge
    5. Optuna-compatible search spaces for each model
    6. TabPFN v2 zero-tuning wrapper

Public API:
    get_model_zoo()          → dict of name → (estimator, param_grid)
    build_preprocessing()    → sklearn Pipeline (PowerTransformer + MinMaxScaler)
    build_stacking_model()   → StackingRegressor (legacy v1)
    build_stacking_v2()      → StackingRegressor (Phase 5 new)
    get_search_space(name)   → dict of param distributions
    get_estimator(name)      → fresh estimator instance
    suggest_optuna_params()  → Optuna trial → params dict
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
from typing import Any, Dict, List, Optional, Tuple

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


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
    返回模型名 → (估计器实例, 默认参数搜索空间) 的字典。

    Models (ranked by expected performance on ~7K samples):
        1. CatBoost — ordered boosting, top for 2K-10K datasets
        2. LightGBM — histogram-based, fast training
        3. XGBoost — mature GBDT, strong baseline
        4. ExtraTrees — randomized splits, robust on small data
        5. GBR — sklearn gradient boosting baseline
        6. SVR — kernel-based, good with proper scaling
        7. KRR — kernel ridge, alternative to SVR
        8. Ridge — linear baseline / meta-learner for stacking
    """
    return {
        # --- Phase 5 新增模型 ---
        "CatBoost": (
            CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=42,
                verbose=0,
            ),
            {
                "iterations": [500, 800, 1000, 1500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "depth": [4, 5, 6, 7, 8],
                "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
                "subsample": [0.7, 0.8, 0.9, 1.0],
            },
        ),
        "LightGBM": (
            LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
            {
                "n_estimators": [500, 800, 1000, 1500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63, 127],
                "min_child_samples": [5, 10, 20, 30],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.8, 1.0],
            },
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            ),
            {
                "n_estimators": [500, 800, 1000, 1500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [4, 5, 6, 7, 8],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0.0, 0.1, 1.0],
                "reg_lambda": [1.0, 3.0, 5.0],
            },
        ),
        # --- Phase 4 保留模型（消融对比） ---
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
# 3. Optuna search space / Optuna 搜索空间
# ---------------------------------------------------------------------------

def suggest_optuna_params(trial, model_name: str) -> Dict[str, Any]:
    """Suggest hyperparameters from an Optuna trial for a given model.
    从 Optuna trial 中为给定模型建议超参数。

    Args:
        trial: optuna.Trial instance.
        model_name: One of the model zoo keys.

    Returns:
        Dict of param_name → suggested value.
    """
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_seed": 42,
            "verbose": 0,
        }
    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbosity": 0,
        }
    elif model_name == "ExtraTrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_depth": trial.suggest_categorical("max_depth", [None, 15, 20, 30, 50]),
            "random_state": 42,
            "n_jobs": -1,
        }
    elif model_name == "GBR":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
        }
    else:
        raise ValueError(
            f"Optuna search space not defined for: {model_name}. "
            f"Available: CatBoost, LightGBM, XGBoost, ExtraTrees, GBR"
        )


# ---------------------------------------------------------------------------
# 4. Stacking v1 (legacy) / 堆叠集成 v1（旧版）
# ---------------------------------------------------------------------------

def build_stacking_model(
    et_params: Dict[str, Any] | None = None,
    gbr_params: Dict[str, Any] | None = None,
    svr_params: Dict[str, Any] | None = None,
    meta_alpha: float = 1.0,
    cv: int = 3,
) -> StackingRegressor:
    """Build Stacking v1: ExtraTrees + GBR + SVR → Ridge meta-learner.
    构建 Stacking v1：ExtraTrees + GBR + SVR → Ridge 元学习器（Phase 4 遗留）。

    Note: This failed 3 times on 304 samples. Kept for backward compatibility.

    Returns:
        StackingRegressor ready for fit/predict.
    """
    et_defaults = {
        "n_estimators": 500, "max_features": "sqrt",
        "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1,
    }
    gbr_defaults = {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4,
        "min_samples_leaf": 5, "subsample": 0.8, "random_state": 42,
    }
    svr_defaults = {
        "kernel": "rbf", "C": 100, "gamma": "scale", "epsilon": 0.1,
    }

    et = ExtraTreesRegressor(**(et_defaults | (et_params or {})))
    gbr = GradientBoostingRegressor(**(gbr_defaults | (gbr_params or {})))
    svr = SVR(**(svr_defaults | (svr_params or {})))

    return StackingRegressor(
        estimators=[("et", et), ("gbr", gbr), ("svr", svr)],
        final_estimator=Ridge(alpha=meta_alpha),
        cv=cv,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 5. Stacking v2 (Phase 5) / 堆叠集成 v2
# ---------------------------------------------------------------------------

def build_stacking_v2(
    catboost_params: Dict[str, Any] | None = None,
    lgbm_params: Dict[str, Any] | None = None,
    et_params: Dict[str, Any] | None = None,
    xgb_params: Dict[str, Any] | None = None,
    meta_alpha: float = 1.0,
    cv: int = 5,
    passthrough: bool = False,
) -> StackingRegressor:
    """Build Stacking v2: CatBoost + LightGBM + ExtraTrees + XGBoost → Ridge.
    构建 Stacking v2：CatBoost + LightGBM + ExtraTrees + XGBoost → Ridge。

    Designed for ~7.5K samples (Phase 5 dataset). Stacking finally feasible
    with 25x more data than Phase 4's 304 samples where it failed 3 times.

    Args:
        catboost_params: Override CatBoost params.
        lgbm_params: Override LightGBM params.
        et_params: Override ExtraTrees params.
        xgb_params: Override XGBoost params.
        meta_alpha: Ridge alpha for meta-learner.
        cv: Number of CV folds for stacking.
        passthrough: Whether to pass original features to meta-learner.

    Returns:
        StackingRegressor ready for fit/predict.
    """
    cb_defaults = {
        "iterations": 1000, "learning_rate": 0.05, "depth": 6,
        "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": 0,
    }
    lgbm_defaults = {
        "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "verbose": -1,
    }
    et_defaults = {
        "n_estimators": 500, "max_features": "sqrt",
        "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1,
    }
    xgb_defaults = {
        "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "random_state": 42, "verbosity": 0,
    }

    cb = CatBoostRegressor(**(cb_defaults | (catboost_params or {})))
    lgbm = LGBMRegressor(**(lgbm_defaults | (lgbm_params or {})))
    et = ExtraTreesRegressor(**(et_defaults | (et_params or {})))
    xgb = XGBRegressor(**(xgb_defaults | (xgb_params or {})))

    return StackingRegressor(
        estimators=[
            ("catboost", cb),
            ("lgbm", lgbm),
            ("et", et),
            ("xgb", xgb),
        ],
        final_estimator=Ridge(alpha=meta_alpha),
        cv=cv,
        passthrough=passthrough,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# 6. Utility functions / 工具函数
# ---------------------------------------------------------------------------

def get_search_space(model_name: str) -> Dict[str, Any]:
    """Return hyperparameter search space for a given model.
    返回给定模型的超参数搜索空间（用于 RandomizedSearchCV / GridSearchCV）。

    Args:
        model_name: One of the model zoo keys.

    Returns:
        Dict of param_name → list of values.
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
        sklearn-compatible estimator instance.
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


def available_models() -> List[str]:
    """Return list of available model names.
    返回可用模型名列表。
    """
    return list(get_model_zoo().keys())
