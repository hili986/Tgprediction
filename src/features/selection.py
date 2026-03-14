"""
4-Stage Feature Selection Pipeline for Tg Prediction
4 阶段特征选择流水线 -- Tg 预测

Stage 1: VarianceThreshold(0.01) -- remove near-constant features
Stage 2: Boruta -- random forest statistical significance
Stage 3: mRMR -- max relevance min redundancy (sklearn MI-based)
Stage 4: SHAP -- model-based final ranking

Public API:
    run_selection_pipeline(X, y, feature_names, target_features) -> SelectionResult
    stage1_variance(X, threshold) -> (X_out, mask)
    stage2_boruta(X, y, max_iter) -> (X_out, mask)
    stage3_mrmr(X, y, feature_names, n_select) -> (X_out, selected_indices)
    stage4_shap_ranking(X, y, top_k) -> (X_out, selected_indices, importances)
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression


@dataclass
class SelectionResult:
    """Result of the 4-stage feature selection pipeline."""

    X_selected: np.ndarray
    selected_names: List[str]
    selected_indices: np.ndarray
    stage_log: List[str] = field(default_factory=list)
    shap_importances: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Stage 1: Variance Threshold
# ---------------------------------------------------------------------------

def stage1_variance(
    X: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove features with variance below threshold.

    Args:
        X: Feature matrix (n_samples, n_features).
        threshold: Minimum variance to keep.

    Returns:
        (X_filtered, boolean mask of kept features).
    """
    selector = VarianceThreshold(threshold=threshold)
    X_out = selector.fit_transform(X)
    mask = selector.get_support()
    return X_out, mask


# ---------------------------------------------------------------------------
# Stage 2: Boruta
# ---------------------------------------------------------------------------

def stage2_boruta(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Boruta feature selection using random forest.

    Args:
        X: Feature matrix.
        y: Target vector.
        max_iter: Maximum Boruta iterations.
        random_state: Random seed.

    Returns:
        (X_filtered, boolean mask of selected features).
    """
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    boruta = BorutaPy(
        rf,
        n_estimators="auto",
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )
    boruta.fit(X, y)

    # Include both confirmed and tentative features
    mask = boruta.support_ | boruta.support_weak_
    if mask.sum() == 0:
        # Fallback: keep all if Boruta rejects everything
        mask = np.ones(X.shape[1], dtype=bool)

    return X[:, mask], mask


# ---------------------------------------------------------------------------
# Stage 3: mRMR (Max Relevance Min Redundancy)
# ---------------------------------------------------------------------------

def stage3_mrmr(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_select: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """mRMR feature selection using sklearn mutual information.

    Greedy forward selection: at each step, select the feature that
    maximizes (relevance to y) - mean(redundancy with already selected).

    Args:
        X: Feature matrix.
        y: Target vector.
        feature_names: Names corresponding to columns of X.
        n_select: Number of features to select.
        random_state: Random seed.

    Returns:
        (X_filtered, integer indices of selected features).
    """
    n_features = X.shape[1]
    n_select = min(n_select, n_features)

    if n_features <= n_select:
        return X, np.arange(n_features)

    # Compute relevance: MI(feature_i, y)
    relevance = mutual_info_regression(X, y, random_state=random_state)

    # Greedy forward selection
    selected: List[int] = []
    remaining = set(range(n_features))

    # First feature: highest relevance
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.discard(first)

    for _ in range(n_select - 1):
        best_score = -np.inf
        best_idx = -1

        for idx in remaining:
            # Redundancy: mean MI with already selected features
            redundancy = 0.0
            for sel_idx in selected:
                # Approximate MI between features using correlation
                corr = np.abs(np.corrcoef(X[:, idx], X[:, sel_idx])[0, 1])
                redundancy += corr

            redundancy /= len(selected)
            score = relevance[idx] - redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)

    indices = np.array(selected)
    return X[:, indices], indices


# ---------------------------------------------------------------------------
# Stage 4: SHAP Ranking
# ---------------------------------------------------------------------------

def stage4_shap_ranking(
    X: np.ndarray,
    y: np.ndarray,
    top_k: int = 30,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SHAP-based feature importance ranking.

    Uses ExtraTreesRegressor + TreeExplainer for fast SHAP values.

    Args:
        X: Feature matrix.
        y: Target vector.
        top_k: Number of top features to keep.
        random_state: Random seed.

    Returns:
        (X_filtered, integer indices of top features, importance array).
    """
    import shap
    from sklearn.ensemble import ExtraTreesRegressor

    top_k = min(top_k, X.shape[1])

    model = ExtraTreesRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(importance)[::-1][:top_k]

    return X[:, top_indices], top_indices, importance


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_selection_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    target_features: int = 30,
    variance_threshold: float = 0.01,
    boruta_max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True,
) -> SelectionResult:
    """Run full 4-stage feature selection pipeline.

    Stage 1: VarianceThreshold -> remove near-constant
    Stage 2: Boruta -> statistical significance (skip if <100 features)
    Stage 3: mRMR -> max relevance min redundancy (skip if <target*2)
    Stage 4: SHAP -> final top-K ranking

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        feature_names: Feature name list matching X columns.
        target_features: Final number of features to keep.
        variance_threshold: Stage 1 threshold.
        boruta_max_iter: Stage 2 max iterations.
        random_state: Random seed.
        verbose: Print progress.

    Returns:
        SelectionResult with selected features, names, and metadata.
    """
    log: List[str] = []
    current_names = list(feature_names)
    global_indices = np.arange(X.shape[1])

    def _log(msg: str) -> None:
        log.append(msg)
        if verbose:
            _safe_print(msg)

    _log(f"  Input: {X.shape[1]} features, {X.shape[0]} samples")

    # Stage 1: Variance
    X1, mask1 = stage1_variance(X, threshold=variance_threshold)
    global_indices = global_indices[mask1]
    current_names = [current_names[i] for i, keep in enumerate(mask1) if keep]
    _log(f"  Stage 1 (Variance>{variance_threshold}): {X.shape[1]} -> {X1.shape[1]}")

    # Stage 2: Boruta (only if >100 features)
    if X1.shape[1] > 100:
        X2, mask2 = stage2_boruta(X1, y, max_iter=boruta_max_iter, random_state=random_state)
        global_indices = global_indices[mask2]
        current_names = [current_names[i] for i, keep in enumerate(mask2) if keep]
        _log(f"  Stage 2 (Boruta): {X1.shape[1]} -> {X2.shape[1]}")
    else:
        X2 = X1
        _log(f"  Stage 2 (Boruta): skipped (features <= 100)")

    # Stage 3: mRMR (only if > target * 2)
    if X2.shape[1] > target_features * 2:
        mrmr_target = target_features * 2
        X3, idx3 = stage3_mrmr(
            X2, y, current_names, n_select=mrmr_target, random_state=random_state,
        )
        global_indices = global_indices[idx3]
        current_names = [current_names[i] for i in idx3]
        _log(f"  Stage 3 (mRMR): {X2.shape[1]} -> {X3.shape[1]}")
    else:
        X3 = X2
        _log(f"  Stage 3 (mRMR): skipped (features <= {target_features * 2})")

    # Stage 4: SHAP
    X4, idx4, importances = stage4_shap_ranking(
        X3, y, top_k=target_features, random_state=random_state,
    )
    global_indices = global_indices[idx4]
    current_names = [current_names[i] for i in idx4]
    full_importances = importances  # importance for all Stage 3 features
    _log(f"  Stage 4 (SHAP): {X3.shape[1]} -> {X4.shape[1]}")

    _log(f"  Final: {X4.shape[1]} features selected")
    if verbose and len(current_names) <= 50:
        _safe_print(f"  Top features: {current_names[:10]}")

    return SelectionResult(
        X_selected=X4,
        selected_names=current_names,
        selected_indices=global_indices,
        stage_log=log,
        shap_importances=full_importances,
    )


def _safe_print(msg: str) -> None:
    """Print with fallback encoding for Windows GBK console."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))
