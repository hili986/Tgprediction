import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_universal_tg_single_regressor import (
    _normalise_limit,
    _fit_with_optional_weights,
    build_table_from_records,
    choose_model,
    compute_metrics,
    mask_hybrid186_for_nonhomopolymer,
    make_sample_weights,
)
from src.ml.universal_tg_features import ComponentRecord, PolymerRecord


class TestTrainUniversalSingleRegressor(unittest.TestCase):
    def test_make_sample_weights_uses_source_groups(self):
        frame = pd.DataFrame(
            {
                "source": [
                    "homopolymer_real",
                    "virtual_copolymer",
                    "polyinfo_real",
                    "nucleobase_real",
                    "unknown",
                ]
            }
        )
        weights = make_sample_weights(
            frame,
            homopolymer_weight=1.0,
            virtual_weight=0.2,
            copolymer_weight=10.0,
            nucleobase_weight=20.0,
        )
        self.assertTrue(np.allclose(weights, [1.0, 0.2, 10.0, 20.0, 1.0]))

    def test_compute_metrics_returns_standard_fields(self):
        metrics = compute_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.5, 2.0]))
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["mae"], 1.0 / 6.0)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

    def test_choose_model_returns_sklearn_fallback(self):
        model = choose_model("extratrees", random_state=7)
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))

    def test_choose_model_returns_custom_physics_kernel(self):
        model = choose_model("physics_kernel", random_state=7)
        self.assertEqual(model.__class__.__name__, "PhysicsResidualKernelRegressor")
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict"))

    def test_choose_model_returns_custom_physics_multikernel(self):
        model = choose_model("physics_multikernel", random_state=7)
        self.assertEqual(model.__class__.__name__, "PhysicsResidualKernelRegressor")
        self.assertEqual(model.kernel_scales, (0.25, 1.0, 4.0))

    def test_choose_model_returns_custom_physics_local(self):
        model = choose_model("physics_local", random_state=7)
        self.assertEqual(model.__class__.__name__, "PhysicsResidualKernelRegressor")
        self.assertEqual(model.local_k, 12)
        self.assertGreater(model.local_weight, 0.0)

    def test_choose_model_returns_custom_physics_local_light(self):
        model = choose_model("physics_local_light", random_state=7)
        self.assertEqual(model.__class__.__name__, "PhysicsResidualKernelRegressor")
        self.assertEqual(model.local_k, 8)
        self.assertAlmostEqual(model.local_weight, 0.15)

    def test_choose_model_returns_custom_physics_hybrid_balanced(self):
        model = choose_model("physics_hybrid_balanced", random_state=7)
        self.assertEqual(model.__class__.__name__, "PhysicsResidualKernelRegressor")
        self.assertEqual(model.high_dim_start, 46)
        self.assertEqual(model.high_dim_end, 233)
        self.assertAlmostEqual(model.high_dim_kernel_weight, 0.25)

    def test_choose_model_returns_custom_physics_homo_correction(self):
        model = choose_model("physics_homo_correction", random_state=7)
        self.assertEqual(model.__class__.__name__, "PhysicsResidualKernelRegressor")
        self.assertTrue(model.homo_correction)
        self.assertAlmostEqual(model.high_dim_kernel_weight, 0.0)

    def test_choose_model_returns_custom_physics_additive_kernel(self):
        model = choose_model("physics_additive_kernel", random_state=7)
        self.assertEqual(model.__class__.__name__, "PhysicsResidualKernelRegressor")
        self.assertEqual(len(model.additive_kernel_groups), 2)

    def test_normalise_limit_distinguishes_disable_from_unlimited(self):
        self.assertIsNone(_normalise_limit(-1))
        self.assertEqual(_normalise_limit(0), 0)
        self.assertEqual(_normalise_limit(10), 10)

    def test_fit_with_optional_weights_passes_direct_estimator_sample_weight(self):
        class DirectEstimator:
            def fit(self, x, y, sample_weight=None):
                self.sample_weight = sample_weight
                return self

        estimator = DirectEstimator()
        weights = np.array([1.0, 2.0])
        _fit_with_optional_weights(
            estimator,
            pd.DataFrame({"x": [0.0, 1.0]}),
            pd.Series([0.0, 1.0]),
            weights,
        )
        self.assertTrue(np.allclose(estimator.sample_weight, weights))

    def test_build_table_from_records_returns_numeric_features(self):
        records = [
            PolymerRecord(
                sample_id="h1",
                source="homopolymer_real",
                architecture="homo",
                components=[ComponentRecord("A", np.array([1.0, 2.0]), None, "missing")],
                weights=[1.0],
                target_tg_c=20.0,
            ),
            PolymerRecord(
                sample_id="c1",
                source="polyinfo_real",
                architecture="random",
                components=[
                    ComponentRecord("A", np.array([1.0, 2.0]), 20.0, "measured"),
                    ComponentRecord("B", np.array([3.0, 4.0]), 80.0, "measured"),
                ],
                weights=[0.4, 0.6],
                target_tg_c=55.0,
                split_group="P1",
            ),
        ]
        table = build_table_from_records(records)
        self.assertEqual(len(table), 2)
        self.assertIn("emb_mean_000", table.columns)
        self.assertIn("endpoint_tg_fox_c", table.columns)
        self.assertTrue(np.isfinite(table.loc[1, "endpoint_tg_fox_c"]))
        self.assertEqual(table.loc[1, "split_group"], "P1")

    def test_mask_hybrid186_for_nonhomopolymer_keeps_homopolymer_only(self):
        frame = pd.DataFrame(
            {
                "is_homopolymer": [1.0, 0.0],
                "emb_mean_045": [1.0, 2.0],
                "emb_mean_046": [3.0, 4.0],
                "emb_std_231": [5.0, 6.0],
            }
        )
        masked = mask_hybrid186_for_nonhomopolymer(frame)
        self.assertEqual(masked.loc[0, "emb_mean_046"], 3.0)
        self.assertTrue(np.isnan(masked.loc[1, "emb_mean_046"]))
        self.assertTrue(np.isnan(masked.loc[1, "emb_std_231"]))
        self.assertEqual(masked.loc[1, "emb_mean_045"], 2.0)

    def test_cli_trains_on_existing_feature_table(self):
        from scripts.train_universal_tg_single_regressor import main

        records = [
            PolymerRecord(
                sample_id=f"h{i}",
                source="homopolymer_real",
                architecture="homo",
                components=[ComponentRecord("A", np.array([float(i), 1.0]), None, "missing")],
                weights=[1.0],
                target_tg_c=float(i),
            )
            for i in range(8)
        ]
        table = build_table_from_records(records)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table_path = root / "table.parquet"
            out_dir = root / "out"
            table.to_parquet(table_path, index=False)
            code = main(
                [
                    "--table",
                    str(table_path),
                    "--output-dir",
                    str(out_dir),
                    "--model",
                    "extratrees",
                    "--test-size",
                    "0.25",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "model.joblib").exists())
            self.assertTrue((out_dir / "feature_columns.json").exists())
            self.assertTrue((out_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
