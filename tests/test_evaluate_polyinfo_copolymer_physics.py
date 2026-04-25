import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.evaluate_polyinfo_copolymer_physics import evaluate_polyinfo_physics


class TestEvaluatePolyinfoCopolymerPhysics(unittest.TestCase):
    def test_evaluate_polyinfo_physics_uses_7k_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_csv = root / "real.csv"
            unified_tg = root / "unified.parquet"
            pd.DataFrame(
                [
                    {"smiles": "*CC(*)", "tg_k": 250.0, "source": "test"},
                    {"smiles": "*CO(*)", "tg_k": 360.0, "source": "test"},
                ]
            ).to_parquet(unified_tg, index=False)
            pd.DataFrame(
                [
                    {
                        "COID": "P1",
                        "SMILES_1": "*CC(*)",
                        "SMILES_2": "*CO(*)",
                        "ratio_1": 75.0,
                        "ratio_unit": "wt%",
                        "Tg_C": 10.0,
                    },
                    {
                        "COID": "P1",
                        "SMILES_1": "*CC(*)",
                        "SMILES_2": "*CO(*)",
                        "ratio_1": 50.0,
                        "ratio_unit": "wt%",
                        "Tg_C": 25.0,
                    },
                ]
            ).to_csv(real_csv, index=False)

            details, summary = evaluate_polyinfo_physics(real_csv, unified_tg, "weight", "component1")

        self.assertEqual(summary["n_usable"], 2)
        self.assertIn("fox_endpoint_tg_c", summary["overall"])
        self.assertIn("linear_fox_leave_system_out_endpoint_tg_c", summary["overall"])
        self.assertIn("physics_ridge_loocv_endpoint_tg_c", details.columns)

    def test_auto_boundary_infers_related_table_ratio_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_csv = root / "real.csv"
            unified_tg = root / "unified.parquet"
            pd.DataFrame(
                [
                    {"smiles": "*CC(*)", "tg_k": 250.0, "source": "test"},
                    {"smiles": "*CO(*)", "tg_k": 360.0, "source": "test"},
                ]
            ).to_parquet(unified_tg, index=False)
            pd.DataFrame(
                [
                    {
                        "COID": "P1",
                        "SMILES_1": "*CC(*)",
                        "SMILES_2": "*CO(*)",
                        "ratio_1": 0.0,
                        "ratio_unit": "wt%",
                        "Tg_C": -23.15,
                        "notes": "from_related_table",
                    },
                    {
                        "COID": "P1",
                        "SMILES_1": "*CC(*)",
                        "SMILES_2": "*CO(*)",
                        "ratio_1": 100.0,
                        "ratio_unit": "wt%",
                        "Tg_C": 86.85,
                        "notes": "from_related_table",
                    },
                ]
            ).to_csv(real_csv, index=False)

            details, summary = evaluate_polyinfo_physics(real_csv, unified_tg, "weight", "auto-boundary")

        self.assertEqual(summary["orientation_by_group"]["P1|related"], "component2")
        self.assertTrue((details["ratio_orientation"] == "component2").all())


if __name__ == "__main__":
    unittest.main()
