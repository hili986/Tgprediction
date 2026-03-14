"""
Pure-Python Machine Learning Models for Polymer Property Prediction
纯 Python 机器学习模型库 — 用于高分子性质预测

Provides:
    1. Math utilities: dot product, normalization, train/test split, k-fold
    2. Regression models with unified fit/predict interface:
       - RidgeRegression, LassoRegression, ElasticNetRegression
       - KNNRegressor
       - DecisionTreeRegressor, RandomForestRegressor
    3. Evaluation metrics: R2, MAE, RMSE, MAPE
    4. Optional sklearn adapters (auto-detected)

All models are implemented without numpy/sklearn dependency.
所有模型均不依赖 numpy/sklearn，纯 Python 标准库实现。

Public API / 公共 API:
    get_model(name, **kwargs)   — factory to create model by name
    r2_score, mae_score, rmse_score, mape_score — evaluation metrics
    normalize, train_test_split, k_fold_indices — data utilities
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Math utilities / 数学工具
# ---------------------------------------------------------------------------

def dot(a: List[float], b: List[float]) -> float:
    """Vector dot product. 向量点积。"""
    return sum(x * y for x, y in zip(a, b))


def vec_add(a: List[float], b: List[float]) -> List[float]:
    """Element-wise vector addition. 向量加法。"""
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: List[float], b: List[float]) -> List[float]:
    """Element-wise vector subtraction. 向量减法。"""
    return [x - y for x, y in zip(a, b)]


def vec_scale(a: List[float], s: float) -> List[float]:
    """Scalar multiplication. 标量乘法。"""
    return [x * s for x in a]


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors. 欧氏距离。"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan_distance(a: List[float], b: List[float]) -> float:
    """Manhattan distance between two vectors. 曼哈顿距离。"""
    return sum(abs(x - y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 2. Data utilities / 数据工具
# ---------------------------------------------------------------------------

def compute_mean_std(X: List[List[float]]) -> Tuple[List[float], List[float]]:
    """Compute column-wise mean and std.
    计算特征矩阵的列均值和标准差。

    Returns:
        (means, stds) — stds clamped to >= 1e-10 to avoid division by zero.
    """
    n = len(X)
    p = len(X[0])
    means = [0.0] * p
    for row in X:
        for j in range(p):
            means[j] += row[j]
    means = [m / n for m in means]

    stds = [0.0] * p
    for row in X:
        for j in range(p):
            stds[j] += (row[j] - means[j]) ** 2
    stds = [max((s / n) ** 0.5, 1e-10) for s in stds]
    return means, stds


def normalize(X_train: List[List[float]],
              X_test: List[List[float]]) -> Tuple[List[List[float]], List[List[float]],
                                                   List[float], List[float]]:
    """Normalize features using training set statistics.
    使用训练集统计量标准化特征。

    Returns:
        (X_train_norm, X_test_norm, means, stds)
    """
    means, stds = compute_mean_std(X_train)
    p = len(means)
    X_train_n = [[(row[j] - means[j]) / stds[j] for j in range(p)] for row in X_train]
    X_test_n = [[(row[j] - means[j]) / stds[j] for j in range(p)] for row in X_test]
    return X_train_n, X_test_n, means, stds


def train_test_split(X: List[List[float]], y: List[float],
                     test_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[List[List[float]], List[List[float]],
                                              List[float], List[float]]:
    """Deterministic train/test split by shuffling indices.
    确定性的训练/测试划分。

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(n * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def k_fold_indices(n: int, k: int = 5,
                   seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """Generate k-fold cross-validation train/test index splits.
    生成 K 折交叉验证的训练/测试索引。

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# 3. Evaluation metrics / 评估指标
# ---------------------------------------------------------------------------

def r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """R-squared (coefficient of determination). R² 决定系数。"""
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mae_score(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error. 平均绝对误差。"""
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse_score(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Squared Error. 均方根误差。"""
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def mape_score(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Percentage Error. 平均绝对百分比误差。"""
    total = 0.0
    count = 0
    for t, p in zip(y_true, y_pred):
        if abs(t) > 1e-10:
            total += abs((t - p) / t)
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# 4. Base model interface / 模型基类
# ---------------------------------------------------------------------------

class BaseRegressor:
    """Base class for regression models. 回归模型基类。"""

    def fit(self, X: List[List[float]], y: List[float]) -> 'BaseRegressor':
        raise NotImplementedError

    def predict(self, X: List[List[float]]) -> List[float]:
        raise NotImplementedError

    def fit_predict(self, X_train: List[List[float]], y_train: List[float],
                    X_test: List[List[float]]) -> List[float]:
        """Fit on training data, predict on test data. 训练并预测。"""
        self.fit(X_train, y_train)
        return self.predict(X_test)


# ---------------------------------------------------------------------------
# 5. Ridge Regression / 岭回归
# ---------------------------------------------------------------------------

class RidgeRegression(BaseRegressor):
    """Ridge regression via gradient descent. L2 正则化线性回归。

    Uses gradient descent (no matrix inversion) — works for any feature count.
    """

    def __init__(self, alpha: float = 0.1, lr: float = 0.01,
                 n_iter: int = 2000):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.w: List[float] = []
        self.b: float = 0.0
        self._means: List[float] = []
        self._stds: List[float] = []

    def fit(self, X: List[List[float]], y: List[float]) -> 'RidgeRegression':
        n = len(X)
        p = len(X[0])

        # Normalize
        self._means, self._stds = compute_mean_std(X)
        X_n = [[(X[i][j] - self._means[j]) / self._stds[j]
                for j in range(p)] for i in range(n)]

        # Initialize
        self.w = [0.0] * p
        self.b = sum(y) / n

        for _ in range(self.n_iter):
            grad_w = [0.0] * p
            grad_b = 0.0
            for i in range(n):
                pred = dot(self.w, X_n[i]) + self.b
                err = pred - y[i]
                for j in range(p):
                    grad_w[j] += err * X_n[i][j]
                grad_b += err

            self.w = [
                self.w[j] - self.lr * (grad_w[j] / n + self.alpha * self.w[j] / n)
                for j in range(p)
            ]
            self.b -= self.lr * grad_b / n

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        p = len(self.w)
        results = []
        for row in X:
            row_n = [(row[j] - self._means[j]) / self._stds[j] for j in range(p)]
            results.append(dot(self.w, row_n) + self.b)
        return results


# ---------------------------------------------------------------------------
# 6. Lasso Regression / Lasso 回归
# ---------------------------------------------------------------------------

class LassoRegression(BaseRegressor):
    """Lasso regression via proximal gradient descent. L1 正则化线性回归。"""

    def __init__(self, alpha: float = 0.1, lr: float = 0.01,
                 n_iter: int = 2000):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.w: List[float] = []
        self.b: float = 0.0
        self._means: List[float] = []
        self._stds: List[float] = []

    @staticmethod
    def _soft_threshold(x: float, threshold: float) -> float:
        """Soft thresholding operator for proximal gradient. 软阈值算子。"""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        return 0.0

    def fit(self, X: List[List[float]], y: List[float]) -> 'LassoRegression':
        n = len(X)
        p = len(X[0])

        self._means, self._stds = compute_mean_std(X)
        X_n = [[(X[i][j] - self._means[j]) / self._stds[j]
                for j in range(p)] for i in range(n)]

        self.w = [0.0] * p
        self.b = sum(y) / n

        for _ in range(self.n_iter):
            grad_w = [0.0] * p
            grad_b = 0.0
            for i in range(n):
                pred = dot(self.w, X_n[i]) + self.b
                err = pred - y[i]
                for j in range(p):
                    grad_w[j] += err * X_n[i][j]
                grad_b += err

            # Gradient step + proximal (soft thresholding)
            threshold = self.lr * self.alpha / n
            self.w = [
                self._soft_threshold(self.w[j] - self.lr * grad_w[j] / n, threshold)
                for j in range(p)
            ]
            self.b -= self.lr * grad_b / n

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        p = len(self.w)
        results = []
        for row in X:
            row_n = [(row[j] - self._means[j]) / self._stds[j] for j in range(p)]
            results.append(dot(self.w, row_n) + self.b)
        return results


# ---------------------------------------------------------------------------
# 7. Elastic Net / 弹性网
# ---------------------------------------------------------------------------

class ElasticNetRegression(BaseRegressor):
    """Elastic Net regression (L1 + L2). 弹性网回归。"""

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5,
                 lr: float = 0.01, n_iter: int = 2000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.n_iter = n_iter
        self.w: List[float] = []
        self.b: float = 0.0
        self._means: List[float] = []
        self._stds: List[float] = []

    def fit(self, X: List[List[float]], y: List[float]) -> 'ElasticNetRegression':
        n = len(X)
        p = len(X[0])

        self._means, self._stds = compute_mean_std(X)
        X_n = [[(X[i][j] - self._means[j]) / self._stds[j]
                for j in range(p)] for i in range(n)]

        self.w = [0.0] * p
        self.b = sum(y) / n
        l1_weight = self.alpha * self.l1_ratio
        l2_weight = self.alpha * (1 - self.l1_ratio)

        for _ in range(self.n_iter):
            grad_w = [0.0] * p
            grad_b = 0.0
            for i in range(n):
                pred = dot(self.w, X_n[i]) + self.b
                err = pred - y[i]
                for j in range(p):
                    grad_w[j] += err * X_n[i][j]
                grad_b += err

            threshold = self.lr * l1_weight / n
            self.w = [
                LassoRegression._soft_threshold(
                    self.w[j] - self.lr * (grad_w[j] / n + l2_weight * self.w[j] / n),
                    threshold
                )
                for j in range(p)
            ]
            self.b -= self.lr * grad_b / n

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        p = len(self.w)
        results = []
        for row in X:
            row_n = [(row[j] - self._means[j]) / self._stds[j] for j in range(p)]
            results.append(dot(self.w, row_n) + self.b)
        return results


# ---------------------------------------------------------------------------
# 8. KNN Regressor / K 近邻回归
# ---------------------------------------------------------------------------

class KNNRegressor(BaseRegressor):
    """K-Nearest Neighbors regression. K 近邻回归。

    Stores training data and predicts by averaging k nearest neighbors.
    """

    def __init__(self, k: int = 5, metric: str = "euclidean", n_neighbors: Optional[int] = None):
        self.k = n_neighbors if n_neighbors is not None else k
        self.metric = metric
        self._X_train: List[List[float]] = []
        self._y_train: List[float] = []
        self._means: List[float] = []
        self._stds: List[float] = []

    def _dist_fn(self, a: List[float], b: List[float]) -> float:
        if self.metric == "manhattan":
            return manhattan_distance(a, b)
        return euclidean_distance(a, b)

    def fit(self, X: List[List[float]], y: List[float]) -> 'KNNRegressor':
        self._means, self._stds = compute_mean_std(X)
        p = len(self._means)
        self._X_train = [
            [(X[i][j] - self._means[j]) / self._stds[j] for j in range(p)]
            for i in range(len(X))
        ]
        self._y_train = list(y)
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        p = len(self._means)
        results = []
        for row in X:
            row_n = [(row[j] - self._means[j]) / self._stds[j] for j in range(p)]
            distances = [
                (self._dist_fn(row_n, self._X_train[i]), i)
                for i in range(len(self._X_train))
            ]
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            pred = sum(self._y_train[idx] for _, idx in k_nearest) / len(k_nearest)
            results.append(pred)
        return results


# ---------------------------------------------------------------------------
# 9. Decision Tree Regressor / 决策树回归
# ---------------------------------------------------------------------------

@dataclass
class _TreeNode:
    """Internal node for decision tree. 决策树内部节点。"""
    feature: int = -1
    threshold: float = 0.0
    left: Optional['_TreeNode'] = None
    right: Optional['_TreeNode'] = None
    value: float = 0.0
    is_leaf: bool = False


class DecisionTreeRegressor(BaseRegressor):
    """CART Decision Tree for regression. CART 决策树回归。

    Uses variance reduction (MSE) as split criterion.
    """

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 5,
                 max_features: Optional[int] = None, seed: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self._root: Optional[_TreeNode] = None
        self._rng: Optional[random.Random] = None

    def fit(self, X: List[List[float]], y: List[float]) -> 'DecisionTreeRegressor':
        self._rng = random.Random(self.seed) if self.seed is not None else random.Random()
        indices = list(range(len(X)))
        self._root = self._build(X, y, indices, depth=0)
        return self

    def _build(self, X: List[List[float]], y: List[float],
               indices: List[int], depth: int) -> _TreeNode:
        values = [y[i] for i in indices]
        node_value = sum(values) / len(values)

        # Leaf conditions
        if (depth >= self.max_depth or
                len(indices) <= self.min_samples_leaf or
                len(set(values)) == 1):
            return _TreeNode(value=node_value, is_leaf=True)

        p = len(X[0])
        # Feature subset selection
        if self.max_features is not None and self.max_features < p:
            feature_candidates = self._rng.sample(range(p), self.max_features)
        else:
            feature_candidates = list(range(p))

        best_feature = -1
        best_threshold = 0.0
        best_mse = float('inf')
        best_left = []
        best_right = []

        for feat in feature_candidates:
            feat_values = sorted(set(X[i][feat] for i in indices))
            if len(feat_values) <= 1:
                continue

            # Try midpoints between unique values
            thresholds = [
                (feat_values[j] + feat_values[j + 1]) / 2
                for j in range(min(len(feat_values) - 1, 20))
            ]

            for thresh in thresholds:
                left_idx = [i for i in indices if X[i][feat] <= thresh]
                right_idx = [i for i in indices if X[i][feat] > thresh]

                if (len(left_idx) < self.min_samples_leaf or
                        len(right_idx) < self.min_samples_leaf):
                    continue

                left_vals = [y[i] for i in left_idx]
                right_vals = [y[i] for i in right_idx]
                left_mean = sum(left_vals) / len(left_vals)
                right_mean = sum(right_vals) / len(right_vals)

                mse = (sum((v - left_mean) ** 2 for v in left_vals) +
                       sum((v - right_mean) ** 2 for v in right_vals))

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feat
                    best_threshold = thresh
                    best_left = left_idx
                    best_right = right_idx

        # No valid split found
        if best_feature == -1:
            return _TreeNode(value=node_value, is_leaf=True)

        left_child = self._build(X, y, best_left, depth + 1)
        right_child = self._build(X, y, best_right, depth + 1)
        return _TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            value=node_value,
        )

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self._predict_one(row) for row in X]

    def _predict_one(self, row: List[float]) -> float:
        node = self._root
        while node is not None and not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value if node else 0.0


# ---------------------------------------------------------------------------
# 10. Random Forest Regressor / 随机森林回归
# ---------------------------------------------------------------------------

class RandomForestRegressor(BaseRegressor):
    """Random Forest regression (ensemble of decision trees).
    随机森林回归（决策树集成）。

    Uses bootstrap sampling and random feature subsets.
    """

    def __init__(self, n_trees: int = 50, max_depth: int = 8,
                 min_samples_leaf: int = 3, max_features: Optional[int] = None,
                 seed: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self._trees: List[DecisionTreeRegressor] = []

    def fit(self, X: List[List[float]], y: List[float]) -> 'RandomForestRegressor':
        n = len(X)
        p = len(X[0])
        max_feat = self.max_features or max(1, int(math.sqrt(p)))
        rng = random.Random(self.seed)
        self._trees = []

        for t in range(self.n_trees):
            # Bootstrap sample
            boot_idx = [rng.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in boot_idx]
            y_boot = [y[i] for i in boot_idx]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feat,
                seed=self.seed + t,
            )
            tree.fit(X_boot, y_boot)
            self._trees.append(tree)

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        all_preds = [tree.predict(X) for tree in self._trees]
        n = len(X)
        return [
            sum(all_preds[t][i] for t in range(len(self._trees))) / len(self._trees)
            for i in range(n)
        ]


# ---------------------------------------------------------------------------
# 11. Gradient Boosting Regressor / 梯度提升回归
# ---------------------------------------------------------------------------

class GradientBoostingRegressor(BaseRegressor):
    """Gradient Boosting regression with decision tree weak learners.
    梯度提升回归。
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_leaf: int = 5,
                 seed: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self._initial_pred: float = 0.0
        self._trees: List[DecisionTreeRegressor] = []

    def fit(self, X: List[List[float]], y: List[float]) -> 'GradientBoostingRegressor':
        n = len(X)
        self._initial_pred = sum(y) / n
        residuals = [y[i] - self._initial_pred for i in range(n)]
        self._trees = []

        for t in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                seed=self.seed + t,
            )
            tree.fit(X, residuals)
            preds = tree.predict(X)
            residuals = [
                residuals[i] - self.learning_rate * preds[i]
                for i in range(n)
            ]
            self._trees.append(tree)

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        n = len(X)
        results = [self._initial_pred] * n
        for tree in self._trees:
            preds = tree.predict(X)
            results = [results[i] + self.learning_rate * preds[i] for i in range(n)]
        return results


# ---------------------------------------------------------------------------
# 12. Model factory / 模型工厂
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, type] = {
    "ridge": RidgeRegression,
    "lasso": LassoRegression,
    "elasticnet": ElasticNetRegression,
    "knn": KNNRegressor,
    "tree": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
    "gbr": GradientBoostingRegressor,
}


def get_model(name: str, **kwargs) -> BaseRegressor:
    """Create a model by name. 按名称创建模型。

    Args:
        name: One of 'ridge', 'lasso', 'elasticnet', 'knn', 'tree', 'rf', 'gbr'.
        **kwargs: Model-specific parameters.

    Returns:
        Model instance with fit/predict interface.
    """
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name_lower](**kwargs)


def available_models() -> List[str]:
    """List available model names. 列出可用模型名称。"""
    return list(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 13. Cross-validation helper / 交叉验证辅助
# ---------------------------------------------------------------------------

def cross_validate(model_name: str, X: List[List[float]], y: List[float],
                   k: int = 5, seed: int = 42,
                   **model_kwargs) -> Dict[str, Any]:
    """Run k-fold cross-validation for a model.
    对模型运行 K 折交叉验证。

    Returns:
        Dict with r2_mean, r2_std, mae_mean, mae_std, rmse_mean, rmse_std,
        fold_results (list of per-fold metrics).
    """
    folds = k_fold_indices(len(X), k=k, seed=seed)
    fold_results = []

    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        model = get_model(model_name, **model_kwargs)
        y_pred = model.fit_predict(X_train, y_train, X_test)

        fold_results.append({
            "r2": r2_score(y_test, y_pred),
            "mae": mae_score(y_test, y_pred),
            "rmse": rmse_score(y_test, y_pred),
        })

    def _mean(vals):
        return sum(vals) / len(vals)

    def _std(vals):
        m = _mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    r2s = [f["r2"] for f in fold_results]
    maes = [f["mae"] for f in fold_results]
    rmses = [f["rmse"] for f in fold_results]

    return {
        "model": model_name,
        "k": k,
        "r2_mean": round(_mean(r2s), 4),
        "r2_std": round(_std(r2s), 4),
        "mae_mean": round(_mean(maes), 2),
        "mae_std": round(_std(maes), 2),
        "rmse_mean": round(_mean(rmses), 2),
        "rmse_std": round(_std(rmses), 2),
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# CLI entry / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ML Models Library — Available models:")
    for name in available_models():
        print(f"  {name}: {_MODEL_REGISTRY[name].__doc__.split(chr(10))[0]}")

    # Quick sanity check with synthetic data
    print("\n--- Synthetic data sanity check ---")
    rng = random.Random(42)
    X_syn = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(100)]
    y_syn = [2 * x[0] + 3 * x[1] + rng.gauss(0, 0.5) for x in X_syn]

    X_tr, X_te, y_tr, y_te = train_test_split(X_syn, y_syn, seed=42)
    for name in ["ridge", "knn", "tree", "rf"]:
        model = get_model(name)
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        mae = mae_score(y_te, y_pred)
        print(f"  {name:12s}  R2={r2:.4f}  MAE={mae:.4f}")
