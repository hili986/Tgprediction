import unittest

import numpy as np
import pandas as pd

from src.ml.universal_tg_model import PhysicsResidualKernelRegressor


class TestPhysicsResidualKernelRegressor(unittest.TestCase):
    def test_fits_physics_plus_nonlinear_residual(self):
        rng = np.random.default_rng(7)
        x = np.linspace(-2.0, 2.0, 80)
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": 50.0 + 10.0 * x,
                "w_entropy": np.abs(x),
                "emb_mean_000": x,
                "emb_mean_001": x * x,
            }
        )
        y = frame["endpoint_tg_fox_c"].to_numpy() + 8.0 * np.sin(3.0 * x)
        y = y + rng.normal(0.0, 0.05, size=len(y))

        model = PhysicsResidualKernelRegressor(
            n_landmarks=80,
            prior_lambda=0.01,
            residual_lambda=0.001,
            random_state=3,
        )
        model.fit(frame, y)
        pred = model.predict(frame)

        self.assertLess(float(np.mean(np.abs(pred - y))), 0.6)
        self.assertIn("endpoint_tg_fox_c", model.diagnostics_.prior_columns)
        self.assertEqual(model.diagnostics_.n_landmarks, 80)

    def test_handles_missing_values_and_sample_weights(self):
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": [0.0, 10.0, np.nan, 30.0],
                "emb_mean_000": [0.0, 1.0, 2.0, np.nan],
            }
        )
        y = np.array([0.0, 10.0, 20.0, 30.0])
        weights = np.array([1.0, 1.0, 0.0, 10.0])
        model = PhysicsResidualKernelRegressor(n_landmarks=3, random_state=1)
        model.fit(frame, y, sample_weight=weights)
        pred = model.predict(frame)
        self.assertEqual(pred.shape, (4,))
        self.assertTrue(np.isfinite(pred).all())

    def test_multiscale_kernel_predicts(self):
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": np.linspace(0.0, 10.0, 12),
                "emb_mean_000": np.linspace(-1.0, 1.0, 12),
            }
        )
        y = np.linspace(0.0, 10.0, 12) ** 1.2
        model = PhysicsResidualKernelRegressor(
            n_landmarks=6,
            kernel_scales=(0.5, 1.0, 2.0),
            random_state=2,
        )
        model.fit(frame, y)
        pred = model.predict(frame)
        self.assertEqual(pred.shape, (12,))
        self.assertTrue(np.isfinite(pred).all())

    def test_local_residual_correction_predicts(self):
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": np.linspace(0.0, 10.0, 20),
                "emb_mean_000": np.linspace(-2.0, 2.0, 20),
            }
        )
        y = np.where(frame["emb_mean_000"].to_numpy() < 0, -5.0, 5.0)
        model = PhysicsResidualKernelRegressor(
            n_landmarks=10,
            local_k=4,
            local_weight=0.8,
            random_state=4,
        )
        model.fit(frame, y)
        pred = model.predict(frame)
        self.assertLess(float(np.mean(np.abs(pred - y))), 2.5)

    def test_final_calibration_uses_endpoint_fox_residual_correction(self):
        fox = np.linspace(-20.0, 80.0, 24)
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": fox,
                "w_entropy": np.linspace(0.1, 0.9, 24),
                "emb_mean_000": np.zeros(24),
            }
        )
        y = 12.0 + 0.75 * fox

        baseline = PhysicsResidualKernelRegressor(
            n_landmarks=4,
            prior_column_patterns=("not_present",),
            residual_lambda=1e6,
            random_state=11,
        )
        baseline.fit(frame, y)
        baseline_mae = float(np.mean(np.abs(baseline.predict(frame) - y)))

        model = PhysicsResidualKernelRegressor(
            n_landmarks=4,
            prior_column_patterns=("not_present",),
            residual_lambda=1e6,
            final_calibration_columns=("endpoint_tg_fox_c",),
            final_calibration_lambda=0.01,
            random_state=11,
        )
        model.fit(frame, y)
        calibrated_mae = float(np.mean(np.abs(model.predict(frame) - y)))

        self.assertLess(calibrated_mae, 1.0)
        self.assertLess(calibrated_mae, baseline_mae * 0.2)
        self.assertEqual(model.final_calibration_columns_, ["endpoint_tg_fox_c"])

    def test_final_calibration_can_be_gated_to_nonhomopolymer_rows(self):
        homo_fox = np.linspace(-20.0, 80.0, 36)
        copolymer_fox = np.linspace(-20.0, 80.0, 8)
        frame = pd.DataFrame(
            {
                "is_homopolymer": np.hstack([np.ones_like(homo_fox), np.zeros_like(copolymer_fox)]),
                "endpoint_tg_fox_c": np.hstack([homo_fox, copolymer_fox]),
                "emb_mean_000": np.zeros(len(homo_fox) + len(copolymer_fox)),
            }
        )
        y = np.hstack([np.zeros_like(homo_fox), 15.0 + 0.5 * copolymer_fox])

        model = PhysicsResidualKernelRegressor(
            n_landmarks=4,
            prior_column_patterns=("not_present",),
            residual_lambda=1e6,
            final_calibration_columns=("endpoint_tg_fox_c",),
            final_calibration_gate_column="is_homopolymer",
            final_calibration_gate_invert=True,
            final_calibration_lambda=0.01,
            random_state=12,
        )
        model.fit(frame, y)
        x_scaled = model._scale(model._impute(frame.to_numpy(dtype=float)))
        design = model._final_calibration_design(x_scaled)
        pred = model.predict(frame)

        self.assertTrue(np.allclose(design[: len(homo_fox)], 0.0))
        self.assertGreater(float(np.mean(np.abs(pred[: len(homo_fox)] - y[: len(homo_fox)]))), 1.0)
        self.assertLess(float(np.mean(np.abs(pred[len(homo_fox) :] - y[len(homo_fox) :]))), 1.0)
        self.assertEqual(model.final_calibration_gate_column_, "is_homopolymer")

    def test_final_calibration_can_use_prediction_delta_from_reference(self):
        frame = pd.DataFrame(
            {
                "is_homopolymer": [0.0, 0.0, 1.0],
                "endpoint_tg_fox_c": [-50.0, -20.0, -50.0],
                "emb_mean_000": [0.0, 1.0, 2.0],
            }
        )
        y = np.array([-45.0, -15.0, -45.0])

        model = PhysicsResidualKernelRegressor(
            n_landmarks=3,
            final_calibration_columns=("endpoint_tg_fox_c",),
            final_calibration_pred_delta_reference_column="endpoint_tg_fox_c",
            final_calibration_gate_column="is_homopolymer",
            final_calibration_gate_invert=True,
            random_state=14,
        )
        model.fit(frame, y)
        x_scaled = model._scale(model._impute(frame.to_numpy(dtype=float)))
        base_pred = frame["endpoint_tg_fox_c"].to_numpy(dtype=float) + np.array([10.0, 20.0, 30.0])
        design = model._final_calibration_design(x_scaled, base_pred)

        self.assertTrue(np.allclose(design[:2, -1], [10.0, 20.0]))
        self.assertEqual(design[2, -1], 0.0)
        self.assertEqual(model.final_calibration_pred_delta_reference_column_, "endpoint_tg_fox_c")

    def test_final_shrinkage_gates_low_fox_nonhomopolymer_rows(self):
        frame = pd.DataFrame(
            {
                "is_homopolymer": [0.0, 0.0, 1.0, 0.0],
                "endpoint_tg_fox_c": [-50.0, -30.0, -50.0, -40.0],
                "emb_mean_000": [0.0, 1.0, 2.0, 3.0],
            }
        )
        y = np.array([-45.0, -25.0, -45.0, -35.0])

        model = PhysicsResidualKernelRegressor(
            n_landmarks=4,
            final_shrinkage_reference_column="endpoint_tg_fox_c",
            final_shrinkage_scale=0.75,
            final_shrinkage_gate_column="endpoint_tg_fox_c",
            final_shrinkage_gate_threshold=-35.0,
            final_shrinkage_gate_less_than=True,
            final_shrinkage_exclude_homopolymer=True,
            random_state=13,
        )
        model.fit(frame, y)
        x_scaled = model._scale(model._impute(frame.to_numpy(dtype=float)))
        pre_shrink = frame["endpoint_tg_fox_c"].to_numpy(dtype=float) + 20.0
        post_shrink = model._apply_final_shrinkage(x_scaled, pre_shrink)

        self.assertTrue(np.allclose(post_shrink, [-35.0, -10.0, -30.0, -25.0]))
        self.assertEqual(model.final_shrinkage_reference_column_, "endpoint_tg_fox_c")
        self.assertEqual(model.final_shrinkage_gate_column_, "endpoint_tg_fox_c")

    def test_high_dim_kernel_weight_downweights_selected_embedding_dimensions(self):
        frame = pd.DataFrame(
            {
                "emb_mean_000": [0.0, 1.0, 2.0],
                "emb_mean_046": [10.0, 11.0, 12.0],
                "endpoint_tg_fox_c": [0.0, 1.0, 2.0],
            }
        )
        model = PhysicsResidualKernelRegressor(
            n_landmarks=3,
            high_dim_start=46,
            high_dim_end=232,
            high_dim_kernel_weight=0.25,
        )
        model.fit(frame, np.array([0.0, 1.0, 2.0]))
        idx = list(frame.columns).index("emb_mean_046")
        self.assertAlmostEqual(model.kernel_feature_weights_[idx], 0.25)

    def test_homopolymer_correction_is_gated(self):
        frame = pd.DataFrame(
            {
                "is_homopolymer": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                "endpoint_tg_fox_c": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "emb_mean_046": [0.0, 1.0, 2.0, 50.0, 51.0, 52.0],
            }
        )
        y = np.array([10.0, 11.0, 12.0, 0.0, 1.0, 2.0])
        model = PhysicsResidualKernelRegressor(
            n_landmarks=4,
            high_dim_start=46,
            high_dim_end=232,
            high_dim_kernel_weight=0.0,
            homo_correction=True,
            homo_correction_landmarks=3,
            random_state=5,
        )
        model.fit(frame, y)
        pred = model.predict(frame)
        self.assertLess(float(np.mean(np.abs(pred[:3] - y[:3]))), 4.0)
        self.assertLess(float(np.mean(np.abs(pred[3:] - y[3:]))), 4.0)

    def test_additive_kernel_groups_predict(self):
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": np.linspace(0.0, 1.0, 16),
                "w_entropy": np.linspace(1.0, 0.0, 16),
                "emb_mean_000": np.sin(np.linspace(0.0, 3.0, 16)),
                "emb_contrast_000": np.cos(np.linspace(0.0, 3.0, 16)),
            }
        )
        y = 5.0 * frame["endpoint_tg_fox_c"].to_numpy() + 2.0 * frame["emb_mean_000"].to_numpy()
        model = PhysicsResidualKernelRegressor(
            n_landmarks=8,
            additive_kernel_groups=(("endpoint_tg_", "w_"), ("emb_mean_", "emb_contrast_")),
            random_state=6,
        )
        model.fit(frame, y)
        pred = model.predict(frame)
        self.assertEqual(pred.shape, (16,))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(len(model.additive_group_indices_), 2)

    def test_additive_kernel_group_weights_are_applied(self):
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": np.linspace(0.0, 1.0, 10),
                "emb_mean_000": np.linspace(1.0, 2.0, 10),
            }
        )
        model = PhysicsResidualKernelRegressor(
            n_landmarks=5,
            additive_kernel_groups=(("endpoint_tg_",), ("emb_mean_",)),
            additive_kernel_group_weights=(0.25, 0.1),
            random_state=9,
        )
        model.fit(frame, np.linspace(0.0, 1.0, 10))
        self.assertEqual(model.additive_group_weights_, [0.25, 0.1])

    def test_combined_additive_kernel_keeps_landmark_width(self):
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": np.linspace(0.0, 1.0, 12),
                "emb_mean_000": np.linspace(1.0, 2.0, 12),
            }
        )
        model = PhysicsResidualKernelRegressor(
            n_landmarks=6,
            additive_kernel_groups=(("endpoint_tg_",), ("emb_mean_",)),
            additive_kernel_group_weights=(0.25, 0.1),
            combine_additive_kernels=True,
            random_state=10,
        )
        model.fit(frame, np.linspace(0.0, 1.0, 12))
        phi = model._rbf_features(model.train_kernel_[:2])
        self.assertEqual(phi.shape, (2, 6))

    def test_kernel_residual_penalizes_first_landmark(self):
        frame = pd.DataFrame(
            {
                "endpoint_tg_fox_c": [0.0, 0.0, 0.0],
                "emb_mean_000": [0.0, 1.0, 2.0],
            }
        )
        y = np.array([100.0, 0.0, 0.0])
        model = PhysicsResidualKernelRegressor(
            n_landmarks=3,
            prior_lambda=1e6,
            residual_lambda=1e6,
            random_state=8,
        )
        model.fit(frame, y)
        self.assertLess(abs(float(model.residual_coef_[0])), 1.0)


if __name__ == "__main__":
    unittest.main()
