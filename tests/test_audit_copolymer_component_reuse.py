import unittest

from scripts.audit_copolymer_component_reuse import build_reuse_report
from src.ml.copolymer_tg_model import CopolymerRecord


class TestAuditCopolymerComponentReuse(unittest.TestCase):
    def test_build_reuse_report_classifies_exact_canonical_and_miss(self):
        records = [
            CopolymerRecord(
                components=("*CC(*)", "*C(C#N)C*", "*CCCCCC(=O)N*"),
                weights=(0.2, 0.3, 0.5),
                architecture="random",
                target_tg_k=300.0,
                metadata={},
            )
        ]

        report = build_reuse_report(records, ["*CC(*)", "*CC(*)C#N"])
        by_component = {
            row["component_smiles"]: row
            for row in report.to_dict(orient="records")
        }

        self.assertEqual(by_component["*CC(*)"]["match_type"], "exact")
        self.assertEqual(by_component["*C(C#N)C*"]["match_type"], "canonical")
        self.assertEqual(by_component["*C(C#N)C*"]["mapped_smiles"], "*CC(*)C#N")
        self.assertEqual(by_component["*CCCCCC(=O)N*"]["match_type"], "miss")


if __name__ == "__main__":
    unittest.main()
