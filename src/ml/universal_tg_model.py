from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is available in this project.
    pd = None


@dataclass
class PhysicsKernelDiagnostics:
    n_train: int
    n_landmarks: int
    gamma: float
    prior_columns: list[str]


class PhysicsResidualKernelRegressor:
    """Single Tg regressor with an explicit physical prior and learned kernel residual.

    The model is intentionally small and inspectable:

    1. Fit a weighted ridge prior on physical/composition columns.
    2. Fit an RBF-kernel residual model on the full universal feature vector.
    3. Predict Tg as `prior(X) + residual(X)`.

    This is still one regressor object and one final prediction path. The prior is not
    a route; it is the low-frequency component of the same fitted model.
    """

    def __init__(
        self,
        n_landmarks: int = 1200,
        gamma: Optional[float] = None,
        prior_lambda: float = 1.0,
        residual_lambda: float = 3.0,
        random_state: int = 42,
        prior_column_patterns: Optional[Sequence[str]] = None,
        kernel_scales: Sequence[float] = (1.0,),
    ) -> None:
        self.n_landmarks = int(n_landmarks)
        self.gamma = gamma
        self.prior_lambda = float(prior_lambda)
        self.residual_lambda = float(residual_lambda)
        self.random_state = int(random_state)
        self.kernel_scales = tuple(float(scale) for scale in kernel_scales if float(scale) > 0)
        if not self.kernel_scales:
            raise ValueError("kernel_scales must contain at least one positive value.")
        self.prior_column_patterns = list(
            prior_column_patterns
            or [
                "endpoint_tg_",
                "w_",
                "n_components",
                "is_homopolymer",
                "is_random",
                "is_block",
                "is_multicomponent",
            ]
        )

    def fit(self, X, y, sample_weight: Optional[Sequence[float]] = None):
        x_raw, feature_names = self._as_matrix_and_names(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_raw.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y length mismatch.")
        if x_raw.shape[0] < 2:
            raise ValueError("Need at least two rows to fit PhysicsResidualKernelRegressor.")

        weights = self._normalise_sample_weight(sample_weight, len(y_arr))
        self.feature_names_ = feature_names
        self.impute_values_ = self._weighted_nanmedian(x_raw, weights)
        x_imputed = self._impute(x_raw)
        self.x_mean_ = np.average(x_imputed, axis=0, weights=weights)
        centered = x_imputed - self.x_mean_
        self.x_scale_ = np.sqrt(np.average(centered * centered, axis=0, weights=weights))
        self.x_scale_[~np.isfinite(self.x_scale_) | (self.x_scale_ < 1e-12)] = 1.0
        x_scaled = self._scale(x_imputed)

        self.prior_indices_ = self._select_prior_indices(feature_names)
        prior_design = self._prior_design(x_scaled)
        self.prior_coef_ = self._weighted_ridge(prior_design, y_arr, weights, self.prior_lambda)
        prior_pred = prior_design @ self.prior_coef_
        residual = y_arr - prior_pred

        landmark_idx = self._select_landmarks(x_scaled, weights)
        self.landmarks_ = x_scaled[landmark_idx].copy()
        self.gamma_ = float(self.gamma) if self.gamma is not None else self._median_gamma(self.landmarks_)
        phi = self._rbf_features(x_scaled)
        self.residual_coef_ = self._weighted_ridge(phi, residual, weights, self.residual_lambda)
        self.diagnostics_ = PhysicsKernelDiagnostics(
            n_train=int(x_scaled.shape[0]),
            n_landmarks=int(self.landmarks_.shape[0]),
            gamma=float(self.gamma_),
            prior_columns=[feature_names[i] for i in self.prior_indices_],
        )
        return self

    def predict(self, X) -> np.ndarray:
        x_raw, _ = self._as_matrix_and_names(X)
        x_scaled = self._scale(self._impute(x_raw))
        return self._prior_design(x_scaled) @ self.prior_coef_ + self._rbf_features(x_scaled) @ self.residual_coef_

    def _as_matrix_and_names(self, X) -> tuple[np.ndarray, list[str]]:
        if pd is not None and isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float), [str(col) for col in X.columns]
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        names = getattr(self, "feature_names_", [f"x_{idx}" for idx in range(arr.shape[1])])
        return arr, list(names)

    def _normalise_sample_weight(self, sample_weight: Optional[Sequence[float]], n_rows: int) -> np.ndarray:
        if sample_weight is None:
            weights = np.ones(n_rows, dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float).reshape(-1)
            if weights.shape[0] != n_rows:
                raise ValueError("sample_weight length mismatch.")
            weights[~np.isfinite(weights) | (weights < 0)] = 0.0
            if float(weights.sum()) <= 0.0:
                weights = np.ones(n_rows, dtype=float)
        return weights / float(np.mean(weights))

    def _weighted_nanmedian(self, matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        values = np.empty(matrix.shape[1], dtype=float)
        for idx in range(matrix.shape[1]):
            col = matrix[:, idx]
            mask = np.isfinite(col) & (weights > 0)
            if not np.any(mask):
                values[idx] = 0.0
                continue
            order = np.argsort(col[mask])
            sorted_values = col[mask][order]
            sorted_weights = weights[mask][order]
            cutoff = 0.5 * float(sorted_weights.sum())
            values[idx] = float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])
        return values

    def _impute(self, matrix: np.ndarray) -> np.ndarray:
        out = np.asarray(matrix, dtype=float).copy()
        bad = ~np.isfinite(out)
        if np.any(bad):
            out[bad] = np.take(self.impute_values_, np.where(bad)[1])
        return out

    def _scale(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.x_mean_) / self.x_scale_

    def _select_prior_indices(self, feature_names: Sequence[str]) -> np.ndarray:
        indices: list[int] = []
        for idx, name in enumerate(feature_names):
            lower = str(name).lower()
            if any(pattern in lower for pattern in self.prior_column_patterns):
                indices.append(idx)
        return np.asarray(indices, dtype=int)

    def _prior_design(self, x_scaled: np.ndarray) -> np.ndarray:
        intercept = np.ones((x_scaled.shape[0], 1), dtype=float)
        if self.prior_indices_.size == 0:
            return intercept
        return np.hstack([intercept, x_scaled[:, self.prior_indices_]])

    def _weighted_ridge(
        self,
        design: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        penalty: float,
    ) -> np.ndarray:
        sqrt_w = np.sqrt(weights).reshape(-1, 1)
        xw = design * sqrt_w
        yw = target * sqrt_w.reshape(-1)
        reg = float(penalty) * np.eye(design.shape[1], dtype=float)
        reg[0, 0] = 0.0
        lhs = xw.T @ xw + reg
        rhs = xw.T @ yw
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def _select_landmarks(self, x_scaled: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n_rows = x_scaled.shape[0]
        n_landmarks = min(max(1, self.n_landmarks), n_rows)
        if n_landmarks == n_rows:
            return np.arange(n_rows)
        rng = np.random.default_rng(self.random_state)
        prob = np.asarray(weights, dtype=float).copy()
        prob[~np.isfinite(prob) | (prob < 0)] = 0.0
        if float(prob.sum()) <= 0:
            prob = np.ones(n_rows, dtype=float)
        prob = prob / float(prob.sum())
        return np.sort(rng.choice(n_rows, size=n_landmarks, replace=False, p=prob))

    def _median_gamma(self, landmarks: np.ndarray) -> float:
        if landmarks.shape[0] < 2:
            return 1.0 / max(landmarks.shape[1], 1)
        subset = landmarks
        if subset.shape[0] > 512:
            rng = np.random.default_rng(self.random_state)
            subset = subset[rng.choice(subset.shape[0], size=512, replace=False)]
        d2 = self._squared_distances(subset, subset)
        upper = d2[np.triu_indices_from(d2, k=1)]
        upper = upper[np.isfinite(upper) & (upper > 1e-12)]
        if upper.size == 0:
            return 1.0 / max(landmarks.shape[1], 1)
        median = float(np.median(upper))
        return 1.0 / max(2.0 * median, 1e-12)

    def _rbf_features(self, x_scaled: np.ndarray) -> np.ndarray:
        d2 = self._squared_distances(x_scaled, self.landmarks_)
        blocks = [np.exp(-(self.gamma_ * scale) * d2) for scale in self.kernel_scales]
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks)

    def _squared_distances(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_norm = np.sum(left * left, axis=1, keepdims=True)
        right_norm = np.sum(right * right, axis=1, keepdims=True).T
        d2 = left_norm + right_norm - 2.0 * (left @ right.T)
        return np.maximum(d2, 0.0)
