import unittest

import pandas as pd

from scripts.predict_nucleobase_excel_with_copolymer_model import (
    NBA_REPEAT_SMILES,
    map_nucleobase_row,
)


class TestPredictNucleobaseExcelWithCopolymerModel(unittest.TestCase):
    def test_maps_random_nba_acrylic_nucleobase_row(self):
        row = pd.Series(
            {
                "Architecture": "random copolymer",
                "Monomer_or_Block": "acrylic adenine",
                "Polymer_System": "poly(nBA-co-acrylic adenine)",
                "Polymer_mol_pct": 7.0,
                "Tg_C": -25.0,
            }
        )

        mapped = map_nucleobase_row(row, 0)

        self.assertEqual(mapped.status, "mapped")
        self.assertEqual(mapped.record.components[0], NBA_REPEAT_SMILES)
        self.assertAlmostEqual(mapped.record.weights[1], 0.07)
        self.assertAlmostEqual(mapped.record.target_tg_k, 248.15)

    def test_skips_block_rows_without_composition(self):
        row = pd.Series(
            {
                "Architecture": "ABC triblock",
                "Monomer_or_Block": "ThA and AdA external blocks",
                "Polymer_System": "poly(ThA-b-nBA-b-AdA)",
                "Tg_C": 66.0,
            }
        )

        mapped = map_nucleobase_row(row, 0)

        self.assertEqual(mapped.status, "skipped")
        self.assertIsNone(mapped.record)
        self.assertIn("block", mapped.reason)


if __name__ == "__main__":
    unittest.main()
