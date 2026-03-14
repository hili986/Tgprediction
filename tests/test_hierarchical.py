"""Tests for Hierarchical Residual Learning (HRL) model."""

import unittest

import numpy as np

from src.ml.hierarchical_model import (
    HierarchicalTgPredictor,
    HRL_FEATURE_GROUPS,
    nested_cv_hrl,
)


class TestHRLBasic(unittest.TestCase):
    """Basic HRL functionality tests."""

    def setUp(self):
        """Create a simple synthetic dataset."""
        np.random.seed(42)
        n = 100
        # Simulate M2M-like features (56-dim)
        self.feature_names = [
            "L0_FlexibilityIndex", "L0_HBondDensity",
            "L0_PolarityIndex", "L0_SOL",
        ] + [f"L1_feat{i}" for i in range(15)] + [
            f"hbond_{i}" for i in range(15)
        ] + [
            "PPF_M_per_f", "PPF_CED_estimate", "PPF_Vf_estimate",
            "PPF_backbone_rigidity", "PPF_steric_volume",
            "PPF_flexible_bond_density", "PPF_symmetry_index",
            "PPF_side_chain_ratio", "PPF_CED_hbond_frac",
            "PPF_ring_strain_proxy",
        ] + [
            f"VPD_feat{i}" for i in range(12)
        ]
        n_features = len(self.feature_names)
        self.X = np.random.randn(n, n_features)
        # Tg roughly linear in M_per_f + noise
        m_per_f_idx = self.feature_names.index("PPF_M_per_f")
        self.y = 300 + 50 * self.X[:, m_per_f_idx] + np.random.randn(n) * 20

    def test_fit_predict_shape(self):
        """fit + predict should return correct shape."""
        model = HierarchicalTgPredictor(random_state=42)
        model.fit(self.X, self.y, feature_names=self.feature_names)
        pred = model.predict(self.X)
        self.assertEqual(pred.shape, (self.X.shape[0],))

    def test_predict_reasonable(self):
        """Predictions should be in a reasonable range."""
        model = HierarchicalTgPredictor(random_state=42)
        model.fit(self.X, self.y, feature_names=self.feature_names)
        pred = model.predict(self.X)
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_layer_contributions_sum(self):
        """Sum of layer contributions should equal predict output."""
        model = HierarchicalTgPredictor(random_state=42)
        model.fit(self.X, self.y, feature_names=self.feature_names)
        pred = model.predict(self.X)
        contributions = model.get_layer_contributions(self.X)

        total = sum(contributions.values())
        np.testing.assert_allclose(pred, total, atol=1e-6)

    def test_diagnose_output(self):
        """diagnose should return metrics for all 4 layers."""
        model = HierarchicalTgPredictor(random_state=42)
        model.fit(self.X, self.y, feature_names=self.feature_names)
        diag = model.diagnose(self.X, self.y)

        expected_layers = ["L0_backbone", "L1_steric", "L2_polar", "L3_residual"]
        for layer in expected_layers:
            self.assertIn(layer, diag)
            self.assertIn("cumulative_R2", diag[layer])
            self.assertIn("cumulative_MAE", diag[layer])

    def test_cumulative_r2_increases(self):
        """Cumulative R² should generally increase with more layers (on train)."""
        model = HierarchicalTgPredictor(random_state=42)
        model.fit(self.X, self.y, feature_names=self.feature_names)
        diag = model.diagnose(self.X, self.y)

        r2_l0 = diag["L0_backbone"]["cumulative_R2"]
        r2_l3 = diag["L3_residual"]["cumulative_R2"]
        # L3 (all features) should be better than L0 (single feature) on training data
        self.assertGreater(r2_l3, r2_l0)


class TestHRLFallback(unittest.TestCase):
    """Test HRL with missing feature groups."""

    def test_no_matching_features(self):
        """HRL should still work when feature names don't match groups."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50) * 50 + 300
        feature_names = [f"unknown_{i}" for i in range(10)]

        model = HierarchicalTgPredictor(random_state=42)
        model.fit(X, y, feature_names=feature_names)
        pred = model.predict(X)
        self.assertEqual(pred.shape, (50,))

    def test_no_feature_names(self):
        """HRL should work without feature names (auto-generate)."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50) * 50 + 300

        model = HierarchicalTgPredictor(random_state=42)
        model.fit(X, y)
        pred = model.predict(X)
        self.assertEqual(pred.shape, (50,))


class TestNestedCVHRL(unittest.TestCase):
    """Test the nested_cv_hrl function."""

    def test_output_format(self):
        """nested_cv_hrl output should match evaluation.nested_cv format."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randn(60) * 50 + 300
        feature_names = [f"feat_{i}" for i in range(10)]

        result = nested_cv_hrl(
            X, y, feature_names,
            outer_splits=3, outer_repeats=1,
            verbose=False,
        )

        self.assertIn("model", result)
        self.assertIn("metrics", result)
        self.assertIn("fold_results", result)
        self.assertIn("y_true_all", result)
        self.assertIn("y_pred_all", result)

        metrics = result["metrics"]
        self.assertIn("R2_mean", metrics)
        self.assertIn("R2_std", metrics)
        self.assertIn("MAE_mean", metrics)
        self.assertIn("MAE_std", metrics)

    def test_fold_count(self):
        """Number of folds should match outer_splits * outer_repeats."""
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = np.random.randn(60) * 50 + 300

        result = nested_cv_hrl(
            X, y, [f"f{i}" for i in range(10)],
            outer_splits=3, outer_repeats=2,
            verbose=False,
        )

        self.assertEqual(len(result["fold_results"]), 6)


if __name__ == "__main__":
    unittest.main()
