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


if __name__ == "__main__":
    unittest.main()
