import unittest

import pandas as pd

from scripts.evaluate_nucleobase_copolymer_strategies import evaluate_strategies


class TestEvaluateNucleobaseCopolymerStrategies(unittest.TestCase):
    def test_evaluate_strategies_adds_physics_baselines(self):
        frame = pd.DataFrame(
            [
                {
                    "status": "predicted",
                    "Nucleobase": "none",
                    "Architecture": "homopolymer baseline",
                    "Polymer_mol_pct": 0.0,
                    "Tg_C_actual": -50.0,
                    "base_pred_tg_c": -48.0,
                    "pred_tg_c": -47.0,
                },
                {
                    "status": "predicted",
                    "Nucleobase": "A",
                    "Architecture": "homopolymer",
                    "Polymer_mol_pct": 100.0,
                    "Tg_C_actual": 70.0,
                    "base_pred_tg_c": 65.0,
                    "pred_tg_c": 64.0,
                },
                {
                    "status": "predicted",
                    "Nucleobase": "A",
                    "Architecture": "random copolymer",
                    "Polymer_mol_pct": 10.0,
                    "Tg_C_actual": -30.0,
                    "base_pred_tg_c": -25.0,
                    "pred_tg_c": -20.0,
                },
            ]
        )

        details, summary = evaluate_strategies(frame)

        self.assertEqual(len(details), 1)
        self.assertIn("fox_actual_endpoint_tg_c", details.columns)
        self.assertEqual(summary["n_random_rows"], 1)
        self.assertIn("base_pred_tg_c", summary["overall"])


if __name__ == "__main__":
    unittest.main()
