"""
Hierarchical Residual Learning (HRL) for Tg Prediction
层级残差学习 — 4 层物理分解

Each layer ONLY learns the residual from previous layers:
    L0: LinearRegression(M/f)      — Gibbs-DiMarzio backbone baseline
    L1: GBR(steric features)       — steric correction
    L2: GBR(polar features)        — polarity/H-bond correction
    L3: GBR(all 56-dim features)   — residual catch-all

Key distinction from Stacking: HRL layers come from physical priors,
not statistical. Each layer has physical meaning.

Public API:
    HierarchicalTgPredictor  — main class (fit, predict, diagnose)
    nested_cv_hrl()          — Nested CV compatible with evaluation.py output format
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

from src.ml.evaluation import compute_metrics


# ---------------------------------------------------------------------------
# Feature group definitions (physics-motivated)
# ---------------------------------------------------------------------------

HRL_FEATURE_GROUPS = {
    "L0_backbone": ["PPF_M_per_f"],
    "L1_steric": ["PPF_steric_volume", "PPF_symmetry_index", "PPF_side_chain_ratio"],
    "L2_polar": ["PPF_CED_estimate", "PPF_CED_hbond_frac", "L0_HBondDensity"],
    # L3 uses ALL features (indices set dynamically)
}


def _get_feature_indices(
    feature_names: List[str],
    group_names: List[str],
) -> List[int]:
    """Map feature group names to column indices."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    indices = []
    for name in group_names:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
    return indices


# ---------------------------------------------------------------------------
# HierarchicalTgPredictor
# ---------------------------------------------------------------------------

class HierarchicalTgPredictor(BaseEstimator, RegressorMixin):
    """4-layer Hierarchical Residual Learning predictor.

    Each layer learns the residual from all previous layers combined.
    """

    def __init__(
        self,
        l1_n_estimators: int = 50,
        l1_max_depth: int = 3,
        l2_n_estimators: int = 50,
        l2_max_depth: int = 3,
        l3_n_estimators: int = 200,
        l3_max_depth: int = 5,
        random_state: int = 42,
    ):
        self.l1_n_estimators = l1_n_estimators
        self.l1_max_depth = l1_max_depth
        self.l2_n_estimators = l2_n_estimators
        self.l2_max_depth = l2_max_depth
        self.l3_n_estimators = l3_n_estimators
        self.l3_max_depth = l3_max_depth
        self.random_state = random_state

        self._feature_names: List[str] = []
        self._l0_model = None
        self._l1_model = None
        self._l2_model = None
        self._l3_model = None
        self._l0_indices: List[int] = []
        self._l1_indices: List[int] = []
        self._l2_indices: List[int] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "HierarchicalTgPredictor":
        """Fit all 4 layers sequentially (each on previous residual).

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target Tg values.
            feature_names: Feature names for group mapping.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if feature_names is not None:
            self._feature_names = list(feature_names)
        else:
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]

        # Resolve feature indices for each group
        self._l0_indices = _get_feature_indices(
            self._feature_names, HRL_FEATURE_GROUPS["L0_backbone"]
        )
        self._l1_indices = _get_feature_indices(
            self._feature_names, HRL_FEATURE_GROUPS["L1_steric"]
        )
        self._l2_indices = _get_feature_indices(
            self._feature_names, HRL_FEATURE_GROUPS["L2_polar"]
        )

        # L0: Linear regression on M/f (backbone baseline)
        if self._l0_indices:
            self._l0_model = LinearRegression()
            self._l0_model.fit(X[:, self._l0_indices], y)
            residual = y - self._l0_model.predict(X[:, self._l0_indices])
        else:
            # Fallback: use mean as baseline
            self._l0_model = _MeanPredictor()
            self._l0_model.fit(X, y)
            residual = y - self._l0_model.predict(X)

        # L1: GBR on steric features (learn L0 residual)
        if self._l1_indices:
            self._l1_model = GradientBoostingRegressor(
                n_estimators=self.l1_n_estimators,
                max_depth=self.l1_max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            )
            self._l1_model.fit(X[:, self._l1_indices], residual)
            residual = residual - self._l1_model.predict(X[:, self._l1_indices])
        else:
            self._l1_model = _ZeroPredictor()

        # L2: GBR on polar features (learn L0+L1 residual)
        if self._l2_indices:
            self._l2_model = GradientBoostingRegressor(
                n_estimators=self.l2_n_estimators,
                max_depth=self.l2_max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            )
            self._l2_model.fit(X[:, self._l2_indices], residual)
            residual = residual - self._l2_model.predict(X[:, self._l2_indices])
        else:
            self._l2_model = _ZeroPredictor()

        # L3: GBR on ALL features (learn remaining residual)
        self._l3_model = GradientBoostingRegressor(
            n_estimators=self.l3_n_estimators,
            max_depth=self.l3_max_depth,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state,
        )
        self._l3_model.fit(X, residual)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict Tg as sum of all layer contributions."""
        X = np.asarray(X, dtype=float)
        pred = np.zeros(X.shape[0])

        # L0
        if self._l0_indices:
            pred += self._l0_model.predict(X[:, self._l0_indices])
        else:
            pred += self._l0_model.predict(X)

        # L1
        if self._l1_indices:
            pred += self._l1_model.predict(X[:, self._l1_indices])

        # L2
        if self._l2_indices:
            pred += self._l2_model.predict(X[:, self._l2_indices])

        # L3
        pred += self._l3_model.predict(X)

        return pred

    def get_layer_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get per-layer prediction contributions.

        Returns:
            Dict mapping layer name to contribution array.
        """
        X = np.asarray(X, dtype=float)

        contributions = {}

        if self._l0_indices:
            contributions["L0_backbone"] = self._l0_model.predict(X[:, self._l0_indices])
        else:
            contributions["L0_backbone"] = self._l0_model.predict(X)

        if self._l1_indices:
            contributions["L1_steric"] = self._l1_model.predict(X[:, self._l1_indices])
        else:
            contributions["L1_steric"] = np.zeros(X.shape[0])

        if self._l2_indices:
            contributions["L2_polar"] = self._l2_model.predict(X[:, self._l2_indices])
        else:
            contributions["L2_polar"] = np.zeros(X.shape[0])

        contributions["L3_residual"] = self._l3_model.predict(X)

        return contributions

    def diagnose(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Diagnose: show per-layer variance decomposition.

        Returns:
            Dict mapping layer name to metrics dict (cumulative R², MAE).
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        contributions = self.get_layer_contributions(X)
        cumulative = np.zeros(X.shape[0])
        result = {}

        for layer_name in ["L0_backbone", "L1_steric", "L2_polar", "L3_residual"]:
            cumulative = cumulative + contributions[layer_name]
            metrics = compute_metrics(y, cumulative)
            result[layer_name] = {
                "cumulative_R2": metrics["R2"],
                "cumulative_MAE": metrics["MAE"],
            }

        return result


# ---------------------------------------------------------------------------
# Helper predictors for missing feature groups
# ---------------------------------------------------------------------------

class _MeanPredictor:
    """Predict the training mean (fallback for L0 when M/f not available)."""

    def __init__(self):
        self._mean = 0.0

    def fit(self, X, y):
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0] if hasattr(X, 'shape') else len(X), self._mean)


class _ZeroPredictor:
    """Always predict zero (for skipped layers)."""

    def predict(self, X):
        return np.zeros(X.shape[0] if hasattr(X, 'shape') else len(X))


# ---------------------------------------------------------------------------
# Nested CV for HRL (compatible with evaluation.nested_cv output format)
# ---------------------------------------------------------------------------

def nested_cv_hrl(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    outer_splits: int = 5,
    outer_repeats: int = 3,
    random_state: int = 42,
    verbose: bool = True,
    **hrl_kwargs: Any,
) -> Dict[str, Any]:
    """Run Nested CV for HierarchicalTgPredictor.

    Output format is compatible with evaluation.nested_cv().

    Args:
        X: Feature matrix.
        y: Target vector.
        feature_names: Feature names for HRL group mapping.
        outer_splits: Number of outer CV folds.
        outer_repeats: Number of outer CV repeats.
        random_state: Random seed.
        verbose: Print progress.
        **hrl_kwargs: Passed to HierarchicalTgPredictor constructor.

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

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = HierarchicalTgPredictor(random_state=random_state, **hrl_kwargs)
        model.fit(X_train, y_train, feature_names=feature_names)

        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        fold_results.append(metrics)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        if verbose and (fold_idx + 1) % 5 == 0:
            print(f"  HRL Fold {fold_idx + 1}/{total_folds}: R2={metrics['R2']:.4f}")

    # Aggregate metrics
    r2_scores = [f["R2"] for f in fold_results]
    mae_scores = [f["MAE"] for f in fold_results]

    overall = compute_metrics(np.array(y_true_all), np.array(y_pred_all))

    result = {
        "model": "HierarchicalTgPredictor",
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

    if verbose:
        print(f"  HRL Nested CV: R2={result['metrics']['R2_mean']:.4f} "
              f"+/- {result['metrics']['R2_std']:.4f}, "
              f"MAE={result['metrics']['MAE_mean']:.1f} "
              f"+/- {result['metrics']['MAE_std']:.1f}")

    return result
