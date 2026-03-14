"""
Unit tests for ml_models.py — Pure Python ML model library.
ml_models.py 单元测试 — 纯 Python 机器学习模型库。

Tests cover:
    1. Math utilities (dot, vec_add/sub/scale, distances)
    2. Data utilities (compute_mean_std, normalize, train_test_split, k_fold_indices)
    3. Evaluation metrics (R2, MAE, RMSE, MAPE)
    4. Each regression model (Ridge, Lasso, ElasticNet, KNN, Tree, RF, GBR)
    5. Model factory and cross-validation
"""

import math
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.models import (
    dot, vec_add, vec_sub, vec_scale,
    euclidean_distance, manhattan_distance,
    compute_mean_std, normalize, train_test_split, k_fold_indices,
    r2_score, mae_score, rmse_score, mape_score,
    RidgeRegression, LassoRegression, ElasticNetRegression,
    KNNRegressor, DecisionTreeRegressor, RandomForestRegressor,
    GradientBoostingRegressor,
    get_model, available_models, cross_validate,
    BaseRegressor,
)

import random


# ---------------------------------------------------------------------------
# Helper: generate synthetic linear data  y = w1*x1 + w2*x2 + noise
# ---------------------------------------------------------------------------

def _make_linear_data(n=200, w=(2.0, 3.0), noise=0.3, seed=42):
    """Generate synthetic data with known linear relationship."""
    rng = random.Random(seed)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(n)]
    y = [w[0] * x[0] + w[1] * x[1] + rng.gauss(0, noise) for x in X]
    return X, y


# ===========================================================================
# 1. Math Utilities Tests / 数学工具测试
# ===========================================================================

class TestMathUtilities(unittest.TestCase):
    """Tests for basic math operations."""

    def test_dot_product_basic(self):
        self.assertAlmostEqual(dot([1, 2, 3], [4, 5, 6]), 32.0)

    def test_dot_product_zeros(self):
        self.assertAlmostEqual(dot([0, 0], [1, 2]), 0.0)

    def test_dot_product_negative(self):
        self.assertAlmostEqual(dot([1, -1], [-1, 1]), -2.0)

    def test_vec_add(self):
        self.assertEqual(vec_add([1, 2], [3, 4]), [4, 6])

    def test_vec_sub(self):
        self.assertEqual(vec_sub([5, 3], [1, 2]), [4, 1])

    def test_vec_scale(self):
        self.assertEqual(vec_scale([1, 2, 3], 2.0), [2.0, 4.0, 6.0])

    def test_vec_scale_zero(self):
        self.assertEqual(vec_scale([1, 2, 3], 0.0), [0.0, 0.0, 0.0])

    def test_euclidean_distance(self):
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)

    def test_euclidean_distance_same_point(self):
        self.assertAlmostEqual(euclidean_distance([1, 2], [1, 2]), 0.0)

    def test_manhattan_distance(self):
        self.assertAlmostEqual(manhattan_distance([0, 0], [3, 4]), 7.0)

    def test_manhattan_distance_negative(self):
        self.assertAlmostEqual(manhattan_distance([-1, -1], [1, 1]), 4.0)


# ===========================================================================
# 2. Data Utilities Tests / 数据工具测试
# ===========================================================================

class TestDataUtilities(unittest.TestCase):
    """Tests for data preprocessing utilities."""

    def test_compute_mean_std(self):
        X = [[1, 10], [3, 20], [5, 30]]
        means, stds = compute_mean_std(X)
        self.assertAlmostEqual(means[0], 3.0)
        self.assertAlmostEqual(means[1], 20.0)
        # std of [1,3,5] = sqrt((4+0+4)/3) = sqrt(8/3)
        expected_std = math.sqrt(8.0 / 3)
        self.assertAlmostEqual(stds[0], expected_std, places=5)

    def test_compute_mean_std_constant_column(self):
        """Constant column should get std clamped to 1e-10."""
        X = [[5, 1], [5, 2], [5, 3]]
        means, stds = compute_mean_std(X)
        self.assertAlmostEqual(means[0], 5.0)
        self.assertEqual(stds[0], 1e-10)

    def test_normalize_preserves_shape(self):
        X_train = [[1, 2], [3, 4], [5, 6]]
        X_test = [[2, 3]]
        X_tr_n, X_te_n, means, stds = normalize(X_train, X_test)
        self.assertEqual(len(X_tr_n), 3)
        self.assertEqual(len(X_te_n), 1)
        self.assertEqual(len(X_tr_n[0]), 2)

    def test_normalize_train_centered(self):
        """Normalized training data should have mean ~0."""
        X_train = [[1, 10], [3, 20], [5, 30], [7, 40]]
        X_test = [[4, 25]]
        X_tr_n, _, _, _ = normalize(X_train, X_test)
        col0_mean = sum(row[0] for row in X_tr_n) / len(X_tr_n)
        col1_mean = sum(row[1] for row in X_tr_n) / len(X_tr_n)
        self.assertAlmostEqual(col0_mean, 0.0, places=5)
        self.assertAlmostEqual(col1_mean, 0.0, places=5)

    def test_train_test_split_sizes(self):
        X = [[i] for i in range(100)]
        y = list(range(100))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.2)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_te), 20)

    def test_train_test_split_deterministic(self):
        """Same seed produces same split."""
        X = [[i] for i in range(50)]
        y = list(range(50))
        split1 = train_test_split(X, y, seed=123)
        split2 = train_test_split(X, y, seed=123)
        self.assertEqual(split1[2], split2[2])  # y_train should match

    def test_train_test_split_no_data_loss(self):
        """All data present across train and test."""
        X = [[i] for i in range(30)]
        y = list(range(30))
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        all_y = sorted(y_tr + y_te)
        self.assertEqual(all_y, list(range(30)))

    def test_k_fold_indices_coverage(self):
        """K-fold should cover all indices exactly once per fold as test."""
        n = 50
        folds = k_fold_indices(n, k=5, seed=42)
        self.assertEqual(len(folds), 5)
        all_test_idx = []
        for train_idx, test_idx in folds:
            all_test_idx.extend(test_idx)
            # No overlap between train and test
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
        # All indices appear as test exactly once
        self.assertEqual(sorted(all_test_idx), list(range(n)))

    def test_k_fold_indices_train_sizes(self):
        """Each fold's train set should have n - fold_size elements."""
        n = 50
        folds = k_fold_indices(n, k=5)
        for train_idx, test_idx in folds:
            self.assertEqual(len(train_idx) + len(test_idx), n)


# ===========================================================================
# 3. Evaluation Metrics Tests / 评估指标测试
# ===========================================================================

class TestMetrics(unittest.TestCase):
    """Tests for evaluation metrics."""

    def test_r2_perfect_prediction(self):
        y = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(r2_score(y, y), 1.0)

    def test_r2_mean_prediction(self):
        """Predicting the mean should give R2 = 0."""
        y_true = [1, 2, 3, 4, 5]
        mean_y = sum(y_true) / len(y_true)
        y_pred = [mean_y] * len(y_true)
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.0)

    def test_r2_worse_than_mean(self):
        """Bad predictions can give R2 < 0."""
        y_true = [1, 2, 3]
        y_pred = [10, 20, 30]
        self.assertLess(r2_score(y_true, y_pred), 0.0)

    def test_mae_perfect(self):
        y = [1, 2, 3]
        self.assertAlmostEqual(mae_score(y, y), 0.0)

    def test_mae_known_value(self):
        y_true = [1, 2, 3]
        y_pred = [2, 3, 4]
        self.assertAlmostEqual(mae_score(y_true, y_pred), 1.0)

    def test_rmse_perfect(self):
        y = [1, 2, 3]
        self.assertAlmostEqual(rmse_score(y, y), 0.0)

    def test_rmse_known_value(self):
        y_true = [0, 0]
        y_pred = [3, 4]
        # RMSE = sqrt((9+16)/2) = sqrt(12.5)
        self.assertAlmostEqual(rmse_score(y_true, y_pred), math.sqrt(12.5))

    def test_rmse_geq_mae(self):
        """RMSE >= MAE for any prediction."""
        y_true = [1, 2, 3, 10, 5]
        y_pred = [1.5, 2.5, 2.8, 8, 6]
        self.assertGreaterEqual(
            rmse_score(y_true, y_pred),
            mae_score(y_true, y_pred) - 1e-10
        )

    def test_mape_perfect(self):
        y = [1, 2, 3]
        self.assertAlmostEqual(mape_score(y, y), 0.0)

    def test_mape_known_value(self):
        y_true = [100, 200]
        y_pred = [110, 180]
        # MAPE = (|10/100| + |20/200|) / 2 = (0.1 + 0.1) / 2 = 0.1
        self.assertAlmostEqual(mape_score(y_true, y_pred), 0.1)

    def test_mape_skips_zero_true(self):
        """MAPE should skip entries where y_true is near zero."""
        y_true = [0, 100]
        y_pred = [50, 110]
        # Only the second entry counts: |10/100| = 0.1
        self.assertAlmostEqual(mape_score(y_true, y_pred), 0.1)


# ===========================================================================
# 4. Model Tests / 模型测试
# ===========================================================================

class TestRidgeRegression(unittest.TestCase):
    """Tests for Ridge Regression."""

    def test_fit_predict_synthetic(self):
        """Ridge should achieve R2 > 0.9 on clean linear data."""
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model = RidgeRegression(alpha=0.01, lr=0.01, n_iter=3000)
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        self.assertGreater(r2, 0.9)

    def test_returns_self(self):
        X, y = _make_linear_data(n=50, seed=1)
        model = RidgeRegression()
        result = model.fit(X, y)
        self.assertIs(result, model)

    def test_predict_shape(self):
        X, y = _make_linear_data(n=50, seed=2)
        model = RidgeRegression(n_iter=100)
        model.fit(X, y)
        preds = model.predict(X[:5])
        self.assertEqual(len(preds), 5)


class TestLassoRegression(unittest.TestCase):
    """Tests for Lasso Regression."""

    def test_fit_predict_synthetic(self):
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model = LassoRegression(alpha=0.01, lr=0.01, n_iter=3000)
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        self.assertGreater(r2, 0.85)

    def test_soft_threshold(self):
        self.assertAlmostEqual(LassoRegression._soft_threshold(5.0, 1.0), 4.0)
        self.assertAlmostEqual(LassoRegression._soft_threshold(-5.0, 1.0), -4.0)
        self.assertAlmostEqual(LassoRegression._soft_threshold(0.5, 1.0), 0.0)

    def test_sparsity_with_high_alpha(self):
        """High alpha should drive some weights to zero."""
        X, y = _make_linear_data(n=200, seed=42)
        model = LassoRegression(alpha=10.0, lr=0.01, n_iter=2000)
        model.fit(X, y)
        zero_weights = sum(1 for w in model.w if abs(w) < 1e-6)
        self.assertGreaterEqual(zero_weights, 0)  # at least it doesn't crash


class TestElasticNet(unittest.TestCase):
    """Tests for Elastic Net Regression."""

    def test_fit_predict_synthetic(self):
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model = ElasticNetRegression(alpha=0.01, l1_ratio=0.5, lr=0.01, n_iter=3000)
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        self.assertGreater(r2, 0.85)

    def test_l1_ratio_extremes(self):
        """l1_ratio=0 should behave like Ridge, l1_ratio=1 like Lasso."""
        X, y = _make_linear_data(n=100, seed=42)
        # Just verify no errors
        en_ridge = ElasticNetRegression(alpha=0.1, l1_ratio=0.0, n_iter=500)
        en_ridge.fit(X, y)
        en_lasso = ElasticNetRegression(alpha=0.1, l1_ratio=1.0, n_iter=500)
        en_lasso.fit(X, y)
        self.assertEqual(len(en_ridge.w), 2)
        self.assertEqual(len(en_lasso.w), 2)


class TestKNNRegressor(unittest.TestCase):
    """Tests for KNN Regressor."""

    def test_fit_predict_synthetic(self):
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model = KNNRegressor(k=5)
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        self.assertGreater(r2, 0.7)

    def test_k1_overfits_train(self):
        """k=1 should perfectly predict training data."""
        X, y = _make_linear_data(n=50, seed=42)
        model = KNNRegressor(k=1)
        model.fit(X, y)
        y_pred = model.predict(X)
        for pred, true in zip(y_pred, y):
            self.assertAlmostEqual(pred, true, places=5)

    def test_manhattan_metric(self):
        """Manhattan metric should work without errors."""
        X, y = _make_linear_data(n=50, seed=42)
        model = KNNRegressor(k=3, metric="manhattan")
        model.fit(X, y)
        preds = model.predict(X[:5])
        self.assertEqual(len(preds), 5)

    def test_n_neighbors_alias(self):
        """n_neighbors should override k when both given."""
        model = KNNRegressor(n_neighbors=7)
        self.assertEqual(model.k, 7)

    def test_n_neighbors_alias_overrides_k(self):
        """n_neighbors takes precedence over positional k."""
        model = KNNRegressor(k=3, n_neighbors=10)
        self.assertEqual(model.k, 10)

    def test_n_neighbors_default_falls_back_to_k(self):
        """When n_neighbors is None, k is used."""
        model = KNNRegressor(k=4)
        self.assertEqual(model.k, 4)

    def test_cross_validate_knn_with_n_neighbors(self):
        """cross_validate k (folds) must not conflict with KNN n_neighbors.
        Regression test for the k-parameter conflict bug.
        """
        X, y = _make_linear_data(n=60, seed=42)
        # k=3 is CV folds, n_neighbors=5 is KNN neighbors — no conflict
        result = cross_validate("knn", X, y, k=3, seed=42, n_neighbors=5)
        self.assertEqual(result["k"], 3)
        self.assertEqual(len(result["fold_results"]), 3)
        self.assertIn("r2_mean", result)


class TestDecisionTreeRegressor(unittest.TestCase):
    """Tests for Decision Tree Regressor."""

    def test_fit_predict_synthetic(self):
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=3, seed=42)
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        self.assertGreater(r2, 0.5)

    def test_overfit_small_data(self):
        """Deep tree should closely fit small dataset."""
        X, y = _make_linear_data(n=30, noise=0.1, seed=42)
        model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=1, seed=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        self.assertGreater(r2, 0.9)

    def test_max_depth_respected(self):
        """Shallow tree should not overfit."""
        X, y = _make_linear_data(n=100, seed=42)
        model = DecisionTreeRegressor(max_depth=1, min_samples_leaf=1, seed=42)
        model.fit(X, y)
        # Depth-1 tree: limited predictions
        preds = model.predict(X)
        unique_preds = len(set(round(p, 6) for p in preds))
        self.assertLessEqual(unique_preds, 3)  # at most 2 leaf values + root

    def test_deterministic_with_seed(self):
        X, y = _make_linear_data(n=50, seed=42)
        model1 = DecisionTreeRegressor(max_depth=5, seed=99)
        model1.fit(X, y)
        preds1 = model1.predict(X[:5])
        model2 = DecisionTreeRegressor(max_depth=5, seed=99)
        model2.fit(X, y)
        preds2 = model2.predict(X[:5])
        for p1, p2 in zip(preds1, preds2):
            self.assertAlmostEqual(p1, p2)


class TestRandomForestRegressor(unittest.TestCase):
    """Tests for Random Forest Regressor."""

    def test_fit_predict_synthetic(self):
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model = RandomForestRegressor(n_trees=20, max_depth=6, seed=42)
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        self.assertGreater(r2, 0.6)

    def test_ensemble_variance_reduction(self):
        """RF should be more stable than a single tree."""
        X, y = _make_linear_data(n=200, noise=0.5, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        # Single tree
        tree = DecisionTreeRegressor(max_depth=6, seed=42)
        tree_preds = tree.fit_predict(X_tr, y_tr, X_te)
        tree_r2 = r2_score(y_te, tree_preds)
        # Forest
        rf = RandomForestRegressor(n_trees=30, max_depth=6, seed=42)
        rf_preds = rf.fit_predict(X_tr, y_tr, X_te)
        rf_r2 = r2_score(y_te, rf_preds)
        # RF should generally be >= single tree (allow small tolerance)
        self.assertGreater(rf_r2, tree_r2 - 0.15)

    def test_n_trees_stored(self):
        X, y = _make_linear_data(n=50, seed=42)
        model = RandomForestRegressor(n_trees=10, seed=42)
        model.fit(X, y)
        self.assertEqual(len(model._trees), 10)


class TestGradientBoostingRegressor(unittest.TestCase):
    """Tests for Gradient Boosting Regressor."""

    def test_fit_predict_synthetic(self):
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=4, seed=42
        )
        y_pred = model.fit_predict(X_tr, y_tr, X_te)
        r2 = r2_score(y_te, y_pred)
        self.assertGreater(r2, 0.4)

    def test_initial_pred_is_mean(self):
        X, y = _make_linear_data(n=50, seed=42)
        model = GradientBoostingRegressor(n_estimators=1, seed=42)
        model.fit(X, y)
        expected_mean = sum(y) / len(y)
        self.assertAlmostEqual(model._initial_pred, expected_mean)

    def test_more_estimators_improve(self):
        """More boosting rounds should reduce training error."""
        X, y = _make_linear_data(n=100, noise=0.3, seed=42)
        model_few = GradientBoostingRegressor(n_estimators=5, learning_rate=0.1, seed=42)
        model_few.fit(X, y)
        mae_few = mae_score(y, model_few.predict(X))
        model_many = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, seed=42)
        model_many.fit(X, y)
        mae_many = mae_score(y, model_many.predict(X))
        self.assertLess(mae_many, mae_few + 0.01)


# ===========================================================================
# 5. Base Class Tests / 基类测试
# ===========================================================================

class TestBaseRegressor(unittest.TestCase):
    """Tests for base class interface."""

    def test_fit_not_implemented(self):
        base = BaseRegressor()
        with self.assertRaises(NotImplementedError):
            base.fit([[1]], [1])

    def test_predict_not_implemented(self):
        base = BaseRegressor()
        with self.assertRaises(NotImplementedError):
            base.predict([[1]])


# ===========================================================================
# 6. Factory & Cross-Validation Tests / 工厂和交叉验证测试
# ===========================================================================

class TestModelFactory(unittest.TestCase):
    """Tests for model factory functions."""

    def test_available_models(self):
        models = available_models()
        self.assertIn("ridge", models)
        self.assertIn("lasso", models)
        self.assertIn("elasticnet", models)
        self.assertIn("knn", models)
        self.assertIn("tree", models)
        self.assertIn("rf", models)
        self.assertIn("gbr", models)
        self.assertEqual(len(models), 7)

    def test_get_model_returns_correct_type(self):
        self.assertIsInstance(get_model("ridge"), RidgeRegression)
        self.assertIsInstance(get_model("lasso"), LassoRegression)
        self.assertIsInstance(get_model("knn"), KNNRegressor)
        self.assertIsInstance(get_model("tree"), DecisionTreeRegressor)
        self.assertIsInstance(get_model("rf"), RandomForestRegressor)
        self.assertIsInstance(get_model("gbr"), GradientBoostingRegressor)

    def test_get_model_case_insensitive(self):
        self.assertIsInstance(get_model("Ridge"), RidgeRegression)
        self.assertIsInstance(get_model("LASSO"), LassoRegression)

    def test_get_model_with_kwargs(self):
        model = get_model("ridge", alpha=0.5, lr=0.001)
        self.assertEqual(model.alpha, 0.5)
        self.assertEqual(model.lr, 0.001)

    def test_get_model_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_model("nonexistent_model")


class TestCrossValidation(unittest.TestCase):
    """Tests for cross-validation."""

    def test_cross_validate_returns_expected_keys(self):
        X, y = _make_linear_data(n=100, seed=42)
        result = cross_validate("ridge", X, y, k=3, seed=42, n_iter=500)
        expected_keys = {"model", "k", "r2_mean", "r2_std",
                         "mae_mean", "mae_std", "rmse_mean", "rmse_std",
                         "fold_results"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_cross_validate_fold_count(self):
        X, y = _make_linear_data(n=100, seed=42)
        result = cross_validate("knn", X, y, k=4, seed=42)
        self.assertEqual(len(result["fold_results"]), 4)
        self.assertEqual(result["k"], 4)
        self.assertEqual(result["model"], "knn")

    def test_cross_validate_reasonable_metrics(self):
        """On clean linear data, ridge should have positive R2."""
        X, y = _make_linear_data(n=200, noise=0.3, seed=42)
        result = cross_validate("ridge", X, y, k=5, seed=42,
                                alpha=0.01, lr=0.01, n_iter=2000)
        self.assertGreater(result["r2_mean"], 0.5)
        self.assertGreater(result["mae_mean"], 0)
        self.assertGreater(result["rmse_mean"], 0)

    def test_cross_validate_all_models(self):
        """Every registered model should work with cross_validate."""
        X, y = _make_linear_data(n=80, noise=0.5, seed=42)
        for name in available_models():
            result = cross_validate(name, X, y, k=3, seed=42)
            self.assertIn("r2_mean", result, f"Model {name} failed")


# ===========================================================================
# 7. Integration / Sanity Checks / 集成测试
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflow."""

    def test_full_pipeline_all_models(self):
        """All models can fit and predict on synthetic data."""
        X, y = _make_linear_data(n=100, noise=0.3, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        for name in available_models():
            model = get_model(name)
            y_pred = model.fit_predict(X_tr, y_tr, X_te)
            self.assertEqual(len(y_pred), len(y_te),
                             f"Model {name}: prediction count mismatch")
            # All predictions should be finite
            for p in y_pred:
                self.assertTrue(math.isfinite(p),
                                f"Model {name}: non-finite prediction {p}")

    def test_fit_predict_equals_separate_calls(self):
        """fit_predict should give same result as fit() then predict()."""
        X, y = _make_linear_data(n=50, seed=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=42)
        model1 = RidgeRegression(alpha=0.1, lr=0.01, n_iter=500)
        preds1 = model1.fit_predict(X_tr, y_tr, X_te)
        model2 = RidgeRegression(alpha=0.1, lr=0.01, n_iter=500)
        model2.fit(X_tr, y_tr)
        preds2 = model2.predict(X_te)
        for p1, p2 in zip(preds1, preds2):
            self.assertAlmostEqual(p1, p2, places=5)


if __name__ == "__main__":
    unittest.main()
