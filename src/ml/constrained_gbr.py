"""
Physics-Constrained Gradient Boosting for Tg Prediction
物理约束梯度提升 — 单调约束 GBR

9 monotone constraints from physical laws:
    +1 (positive): M_per_f, CED_estimate, backbone_rigidity, RingCount, NumAromaticRings
    -1 (negative): FlexibilityIndex, Vf_estimate, symmetry_index, flexible_bond_density
    0  (unconstrained): all others

Public API:
    build_constrained_gbr(feature_names, ...) -> HistGradientBoostingRegressor
    clip_predictions(y_pred, min_tg, max_tg) -> np.ndarray
    PHYSICS_MONOTONE -> dict of feature→constraint
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# Physics-derived monotone constraints
# ---------------------------------------------------------------------------

PHYSICS_MONOTONE: Dict[str, int] = {
    # Positive monotone (+1): higher value → higher Tg
    "PPF_M_per_f": 1,
    "PPF_CED_estimate": 1,
    "PPF_backbone_rigidity": 1,
    "L1_RingCount": 1,
    "L1_NumAromaticRings": 1,
    # Negative monotone (-1): higher value → lower Tg
    "L0_FlexibilityIndex": -1,
    "PPF_Vf_estimate": -1,
    "PPF_symmetry_index": -1,
    "PPF_flexible_bond_density": -1,
}

# Default HistGBR hyperparameters
_DEFAULT_GBR_PARAMS = {
    "max_iter": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
    "random_state": 42,
}


def build_constrained_gbr(
    feature_names: List[str],
    custom_constraints: Optional[Dict[str, int]] = None,
    **gbr_kwargs: Any,
) -> HistGradientBoostingRegressor:
    """Build a HistGBR with physics-derived monotone constraints.

    Args:
        feature_names: Ordered list of feature names matching X columns.
        custom_constraints: Override/extend default PHYSICS_MONOTONE.
        **gbr_kwargs: Additional GBR parameters (override defaults).

    Returns:
        Configured HistGradientBoostingRegressor (not yet fitted).
    """
    constraints = dict(PHYSICS_MONOTONE)
    if custom_constraints:
        constraints.update(custom_constraints)

    # Build constraint array matching feature order
    constraint_array = []
    for name in feature_names:
        constraint_array.append(constraints.get(name, 0))

    # Merge default params with user overrides
    params = dict(_DEFAULT_GBR_PARAMS)
    params.update(gbr_kwargs)

    return HistGradientBoostingRegressor(
        monotonic_cst=constraint_array,
        **params,
    )


def clip_predictions(
    y_pred: np.ndarray,
    min_tg: float = 100.0,
    max_tg: float = 700.0,
) -> np.ndarray:
    """Clip predictions to physically plausible Tg range.

    Args:
        y_pred: Raw predictions.
        min_tg: Minimum plausible Tg (K).
        max_tg: Maximum plausible Tg (K).

    Returns:
        Clipped predictions.
    """
    return np.clip(y_pred, min_tg, max_tg)


def get_constraint_summary(
    feature_names: List[str],
    custom_constraints: Optional[Dict[str, int]] = None,
) -> Dict[str, List[str]]:
    """Summarize which features are constrained and how.

    Returns:
        Dict with keys 'positive', 'negative', 'unconstrained'.
    """
    constraints = dict(PHYSICS_MONOTONE)
    if custom_constraints:
        constraints.update(custom_constraints)

    positive = []
    negative = []
    unconstrained = []

    for name in feature_names:
        c = constraints.get(name, 0)
        if c > 0:
            positive.append(name)
        elif c < 0:
            negative.append(name)
        else:
            unconstrained.append(name)

    return {
        "positive": positive,
        "negative": negative,
        "unconstrained": unconstrained,
    }
