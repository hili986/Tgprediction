import unittest

from scripts.train_copolymer_tg_residual import _filter_records_with_precomputed_components
from src.ml.copolymer_tg_model import CopolymerRecord


class _FakePredictor:
    def __init__(self, statuses):
        self.statuses = statuses

    def precomputed_component_match(self, smiles):
        status = self.statuses.get(smiles, "miss")
        mapped = smiles if status in {"exact", "canonical"} else None
        return status, mapped


class TestTrainCopolymerTgResidual(unittest.TestCase):
    def test_filter_records_with_precomputed_components_skips_any_miss(self):
        records = [
            CopolymerRecord(
                components=("*A*", "*B*"),
                weights=(0.5, 0.5),
                architecture="random",
                target_tg_k=300.0,
                metadata={},
            ),
            CopolymerRecord(
                components=("*A*", "*C*"),
                weights=(0.5, 0.5),
                architecture="random",
                target_tg_k=320.0,
                metadata={},
            ),
        ]
        predictor = _FakePredictor({"*A*": "exact", "*B*": "canonical", "*C*": "miss"})

        kept, skipped = _filter_records_with_precomputed_components(records, predictor)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].components, ("*A*", "*B*"))
        self.assertEqual(skipped.loc[0, "missing_components"], "*C*")
        self.assertEqual(skipped.loc[0, "missing_statuses"], "miss")


if __name__ == "__main__":
    unittest.main()
