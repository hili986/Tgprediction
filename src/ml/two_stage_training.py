"""
Two-Stage Training for Multi-Fidelity Tg Prediction
两阶段训练 — 多保真度 Tg 预测

Strategy (locked decision):
    Stage 1: Pretrain on virtual copolymer data (Fox/GT)
    Stage 2: Finetune on experimental homopolymer data (warm_start)

For ExtraTrees with warm_start=True:
    - Stage 1 trains N1 trees on virtual data
    - Stage 2 increases n_estimators and trains additional trees on experimental data

Also implements sample_weight approach for comparison.

IMPORTANT: Leakage-free CV — virtual samples containing test homopolymers
are excluded from training in each fold.

Public API:
    two_stage_warm_start(X_virt, y_virt, X_exp, y_exp, ...) -> fitted model
    combined_weighted_training(X_virt, y_virt, X_exp, y_exp, ...) -> fitted model
    evaluate_two_stage(X_exp, y_exp, X_virt, y_virt, ...) -> dict of results
    evaluate_baseline(X_exp, y_exp, ...) -> dict of results
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

from src.ml.evaluation import compute_metrics, _aggregate_metrics


# ---------------------------------------------------------------------------
# 1. Two-stage warm_start training
# ---------------------------------------------------------------------------

def two_stage_warm_start(
    X_virtual: np.ndarray,
    y_virtual: np.ndarray,
    X_exp: np.ndarray,
    y_exp: np.ndarray,
    pretrain_trees: int = 100,
    total_trees: int = 500,
    random_state: int = 42,
    **et_params,
) -> ExtraTreesRegressor:
    """Two-stage training using ExtraTrees warm_start.

    Stage 1: Train `pretrain_trees` trees on virtual data.
    Stage 2: Add trees up to `total_trees` on experimental data.

    Args:
        X_virtual: Virtual copolymer features.
        y_virtual: Virtual copolymer Tg values.
        X_exp: Experimental homopolymer features.
        y_exp: Experimental homopolymer Tg values.
        pretrain_trees: Number of trees in pretrain stage.
        total_trees: Total number of trees after finetune.
        random_state: Random seed.
        **et_params: Additional ExtraTrees parameters.

    Returns:
        Fitted ExtraTreesRegressor.
    """
    defaults = {
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "n_jobs": -1,
    }
    defaults.update(et_params)

    model = ExtraTreesRegressor(
        n_estimators=pretrain_trees,
        warm_start=True,
        random_state=random_state,
        **defaults,
    )

    # Stage 1: Pretrain on virtual data
    model.fit(X_virtual, y_virtual)

    # Stage 2: Add more trees on experimental data
    model.n_estimators = total_trees
    model.fit(X_exp, y_exp)

    return model


# ---------------------------------------------------------------------------
# 2. Combined weighted training
# ---------------------------------------------------------------------------

def combined_weighted_training(
    X_virtual: np.ndarray,
    y_virtual: np.ndarray,
    X_exp: np.ndarray,
    y_exp: np.ndarray,
    virtual_weight: float = 0.1,
    n_estimators: int = 500,
    random_state: int = 42,
    **et_params,
) -> ExtraTreesRegressor:
    """Train on combined data with sample weights.

    Experimental data weight = 1.0, virtual data weight = virtual_weight.

    Args:
        X_virtual: Virtual copolymer features.
        y_virtual: Virtual copolymer Tg values.
        X_exp: Experimental features.
        y_exp: Experimental Tg values.
        virtual_weight: Weight for virtual data samples (0-1).
        n_estimators: Number of trees.
        random_state: Random seed.
        **et_params: Additional ExtraTrees parameters.

    Returns:
        Fitted ExtraTreesRegressor.
    """
    defaults = {
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "n_jobs": -1,
    }
    defaults.update(et_params)

    X_combined = np.vstack([X_exp, X_virtual])
    y_combined = np.concatenate([y_exp, y_virtual])

    weights = np.concatenate([
        np.ones(len(y_exp)),
        np.full(len(y_virtual), virtual_weight),
    ])

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        **defaults,
    )
    model.fit(X_combined, y_combined, sample_weight=weights)

    return model


# ---------------------------------------------------------------------------
# 3. Leakage-free virtual data filtering
# ---------------------------------------------------------------------------

def _filter_virtual_for_fold(
    virt_smiles_pairs: List[Tuple[str, str]],
    test_smiles: set,
) -> np.ndarray:
    """Return boolean mask: True for virtual samples safe to use in training.

    A virtual sample is excluded if either of its source homopolymers
    is in the test set (prevents data leakage).

    Args:
        virt_smiles_pairs: List of (smiles1, smiles2) for each virtual sample.
        test_smiles: Set of SMILES strings in the test fold.

    Returns:
        Boolean mask array.
    """
    mask = np.array([
        s1 not in test_smiles and s2 not in test_smiles
        for s1, s2 in virt_smiles_pairs
    ])
    return mask


# ---------------------------------------------------------------------------
# 4. Evaluation: Leakage-free CV on experimental data
# ---------------------------------------------------------------------------

def evaluate_two_stage(
    X_exp: np.ndarray,
    y_exp: np.ndarray,
    X_virtual: np.ndarray,
    y_virtual: np.ndarray,
    method: str = "warm_start",
    pretrain_trees: int = 100,
    total_trees: int = 500,
    virtual_weight: float = 0.1,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    preprocess: bool = True,
    verbose: bool = True,
    exp_smiles: Optional[List[str]] = None,
    virt_smiles_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Evaluate two-stage training with leakage-free CV.

    Virtual samples containing test homopolymers are excluded per fold.

    Args:
        X_exp: Experimental features.
        y_exp: Experimental targets.
        X_virtual: Virtual copolymer features.
        y_virtual: Virtual copolymer targets.
        method: 'warm_start' or 'weighted'.
        pretrain_trees: Trees for pretrain stage (warm_start method).
        total_trees: Total trees.
        virtual_weight: Weight for virtual data (weighted method).
        n_splits: CV folds.
        n_repeats: CV repeats.
        random_state: Random seed.
        preprocess: Apply PowerTransformer + MinMaxScaler.
        verbose: Print progress.
        exp_smiles: SMILES list for experimental data (for leakage filtering).
        virt_smiles_pairs: List of (smiles1, smiles2) for virtual data.

    Returns:
        Dict with metrics, fold_results, config.
    """
    leakage_free = exp_smiles is not None and virt_smiles_pairs is not None

    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_results: List[Dict[str, float]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []
    virt_used_counts: List[int] = []

    total_folds = n_splits * n_repeats

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_exp, y_exp)):
        X_train, X_test = X_exp[train_idx], X_exp[test_idx]
        y_train, y_test = y_exp[train_idx], y_exp[test_idx]

        # Filter virtual data to exclude test homopolymers
        if leakage_free:
            test_smiles_set = {exp_smiles[i] for i in test_idx}
            safe_mask = _filter_virtual_for_fold(virt_smiles_pairs, test_smiles_set)
            X_virt_fold = X_virtual[safe_mask]
            y_virt_fold = y_virtual[safe_mask]
            virt_used_counts.append(int(safe_mask.sum()))
        else:
            X_virt_fold = X_virtual
            y_virt_fold = y_virtual

        # Preprocessing: fit on train+virtual, transform all
        if preprocess:
            scaler = Pipeline([
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", MinMaxScaler()),
            ])
            X_combined_train = np.vstack([X_train, X_virt_fold])
            scaler.fit(X_combined_train)

            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            X_virt_s = scaler.transform(X_virt_fold)
        else:
            X_train_s = X_train
            X_test_s = X_test
            X_virt_s = X_virt_fold

        # Train model
        if method == "warm_start":
            model = two_stage_warm_start(
                X_virt_s, y_virt_fold,
                X_train_s, y_train,
                pretrain_trees=pretrain_trees,
                total_trees=total_trees,
                random_state=random_state + fold_idx,
            )
        elif method == "weighted":
            model = combined_weighted_training(
                X_virt_s, y_virt_fold,
                X_train_s, y_train,
                virtual_weight=virtual_weight,
                n_estimators=total_trees,
                random_state=random_state + fold_idx,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Predict on test (experimental only)
        y_pred = model.predict(X_test_s)

        fold_metrics = compute_metrics(y_test, y_pred)
        fold_results.append(fold_metrics)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    metrics_agg = _aggregate_metrics(fold_results)
    overall = compute_metrics(y_true_all, y_pred_all)

    config = {
        "method": method,
        "pretrain_trees": pretrain_trees if method == "warm_start" else None,
        "total_trees": total_trees,
        "virtual_weight": virtual_weight if method == "weighted" else None,
        "n_virtual_total": len(y_virtual),
        "n_experimental": len(y_exp),
        "leakage_free": leakage_free,
        "preprocess": preprocess,
    }

    if virt_used_counts:
        config["virt_used_mean"] = int(np.mean(virt_used_counts))
        config["virt_used_min"] = min(virt_used_counts)

    result = {
        "metrics": metrics_agg,
        "overall": overall,
        "fold_results": fold_results,
        "config": config,
        "n_folds": total_folds,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }

    if verbose:
        leak_tag = " leakage-free" if leakage_free else ""
        virt_info = ""
        if virt_used_counts:
            virt_info = f" (avg {int(np.mean(virt_used_counts))} used per fold)"
        _safe_print(
            f"\n  === Two-Stage CV ({method},{leak_tag} {total_folds} folds) ===\n"
            f"  Virtual: {len(y_virtual)} total{virt_info}, Exp: {len(y_exp)}\n"
            f"  R2:   {metrics_agg['R2_mean']:.4f} +/- {metrics_agg['R2_std']:.4f}\n"
            f"  MAE:  {metrics_agg['MAE_mean']:.1f} +/- {metrics_agg['MAE_std']:.1f} K\n"
            f"  Overall: R2={overall['R2']:.4f}, MAE={overall['MAE']:.1f}K"
        )

    return result


# ---------------------------------------------------------------------------
# 5. Baseline: experimental-only training (for comparison)
# ---------------------------------------------------------------------------

def evaluate_baseline(
    X_exp: np.ndarray,
    y_exp: np.ndarray,
    n_estimators: int = 500,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    preprocess: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Baseline: train only on experimental data (no virtual data).

    For fair comparison with two-stage methods.
    """
    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_results: List[Dict[str, float]] = []
    y_true_all: List[float] = []
    y_pred_all: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_exp, y_exp)):
        X_train, X_test = X_exp[train_idx], X_exp[test_idx]
        y_train, y_test = y_exp[train_idx], y_exp[test_idx]

        if preprocess:
            scaler = Pipeline([
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", MinMaxScaler()),
            ])
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
        else:
            X_train_s = X_train
            X_test_s = X_test

        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        fold_metrics = compute_metrics(y_test, y_pred)
        fold_results.append(fold_metrics)
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

    metrics_agg = _aggregate_metrics(fold_results)
    overall = compute_metrics(y_true_all, y_pred_all)

    result = {
        "metrics": metrics_agg,
        "overall": overall,
        "fold_results": fold_results,
        "config": {
            "method": "baseline",
            "n_estimators": n_estimators,
            "n_experimental": len(y_exp),
        },
        "n_folds": n_splits * n_repeats,
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
    }

    if verbose:
        total_folds = n_splits * n_repeats
        _safe_print(
            f"\n  === Baseline CV ({total_folds} folds) ===\n"
            f"  Exp: {len(y_exp)} samples\n"
            f"  R2:   {metrics_agg['R2_mean']:.4f} +/- {metrics_agg['R2_std']:.4f}\n"
            f"  MAE:  {metrics_agg['MAE_mean']:.1f} +/- {metrics_agg['MAE_std']:.1f} K\n"
            f"  Overall: R2={overall['R2']:.4f}, MAE={overall['MAE']:.1f}K"
        )

    return result


def _safe_print(msg: str) -> None:
    """Print with fallback encoding for Windows GBK console."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))
