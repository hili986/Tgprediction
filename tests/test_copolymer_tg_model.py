import unittest

import numpy as np
import pandas as pd

from src.ml.copolymer_tg_model import (
    CopolymerRecord,
    build_copolymer_feature_vector,
    build_feature_matrix,
    fit_residual_corrector,
    predict_with_residual,
    parse_copolymer_records,
    residual_feature_matrix,
)


class TestCopolymerRecordParsing(unittest.TestCase):
    def test_parses_virtual_generator_rows(self):
        df = pd.DataFrame(
            [
                {
                    "components_serialized": "*CC(*)|*CO(*)",
                    "weights_serialized": "0.20000000|0.80000000",
                    "architecture": "random",
                    "tg_k_pred": 310.5,
                    "status": "ok",
                }
            ]
        )

        records = parse_copolymer_records(df, target_column="tg_k_pred")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].components, ("*CC(*)", "*CO(*)"))
        self.assertEqual(records[0].weights, (0.2, 0.8))
        self.assertEqual(records[0].architecture, "random")
        self.assertEqual(records[0].target_tg_k, 310.5)

    def test_parses_polyinfo_style_ratio_and_celsius_target(self):
        df = pd.DataFrame(
            [
                {
                    "SMILES_1": "*CC(*)",
                    "SMILES_2": "*CO(*)",
                    "ratio_1": 25,
                    "Tg_C": 42.0,
                    "copolymer_type": "U",
                }
            ]
        )

        records = parse_copolymer_records(df)

        self.assertEqual(records[0].components, ("*CC(*)", "*CO(*)"))
        self.assertEqual(records[0].weights, (0.25, 0.75))
        self.assertEqual(records[0].architecture, "random")
        self.assertAlmostEqual(records[0].target_tg_k, 315.15)


class TestCopolymerFeatureVector(unittest.TestCase):
    def test_builds_weighted_mix_dispersion_and_scalars(self):
        record = CopolymerRecord(
            components=("*A*", "*B*"),
            weights=(0.25, 0.75),
            architecture="block",
            target_tg_k=300.0,
            metadata={},
        )
        component_vectors = [np.array([1.0, 5.0]), np.array([3.0, 1.0])]

        names, values = build_copolymer_feature_vector(record, component_vectors)

        self.assertEqual(names[:2], ["mix_f000", "mix_f001"])
        self.assertTrue(np.allclose(values[:2], [2.5, 2.0]))
        self.assertEqual(names[2:4], ["disp_f000", "disp_f001"])
        self.assertTrue(np.allclose(values[2:4], [0.75, 1.5]))
        feature_map = dict(zip(names, values))
        self.assertEqual(feature_map["n_components"], 2.0)
        self.assertEqual(feature_map["architecture_block"], 1.0)
        self.assertEqual(feature_map["architecture_random"], 0.0)
        self.assertAlmostEqual(feature_map["max_weight"], 0.75)
        self.assertGreater(feature_map["weight_entropy"], 0.0)


class TestResidualFineTuning(unittest.TestCase):
    def test_residual_corrector_learns_bias_over_base_prediction(self):
        x_real = np.array([[0.0], [1.0], [2.0], [3.0]])
        base_pred = np.array([300.0, 301.0, 302.0, 303.0])
        y_real = base_pred + 5.0

        residual_model = fit_residual_corrector(x_real, y_real, base_pred)
        corrected = predict_with_residual(base_pred, residual_model, x_real)

        self.assertTrue(np.allclose(corrected, y_real, atol=1e-6))

    def test_residual_feature_matrix_appends_base_prediction(self):
        x_real = np.array([[1.0, 2.0], [3.0, 4.0]])
        base_pred = np.array([300.0, 310.0])

        residual_x = residual_feature_matrix(x_real, base_pred)

        self.assertEqual(residual_x.shape, (2, 3))
        self.assertTrue(np.allclose(residual_x[:, -1], base_pred))


class FakeFeaturePredictor:
    def __init__(self):
        self.components = {
            "*A*": {"phyc": np.array([1.0]), "gnn": np.array([2.0]), "pbert": np.array([3.0])},
            "*B*": {"phyc": np.array([3.0]), "gnn": np.array([4.0]), "pbert": np.array([5.0])},
        }

    def featurize_component(self, smiles):
        return self.components[smiles]

    def _component_full_vector(self, component):
        return np.hstack([component["phyc"], component["gnn"], component["pbert"]])

    def predict_multicomponent(self, smiles_list, weights, architecture="random"):
        return {
            "tg_k_pred": 333.0,
            "descriptor_mix_tg_k": 333.0,
            "fox_reference_tg_k": 320.0,
            "component_tg_window_k": [300.0, 340.0],
        }


class TestFeatureMatrixBuild(unittest.TestCase):
    def test_build_feature_matrix_uses_predictor_component_vectors(self):
        records = [
            CopolymerRecord(
                components=("*A*", "*B*"),
                weights=(0.5, 0.5),
                architecture="random",
                target_tg_k=310.0,
                metadata={},
            )
        ]

        result = build_feature_matrix(records, FakeFeaturePredictor(), include_teacher_scalars=True)

        self.assertEqual(result.X.shape[0], 1)
        self.assertEqual(result.y.tolist(), [310.0])
        self.assertIn("teacher_descriptor_mix_tg_k", result.feature_names)
        self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
