import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from scripts.predict_tg_universal_single_regressor import (
    load_model_bundle,
    predict_feature_frame,
)


class TestPredictUniversalSingleRegressor(unittest.TestCase):
    def test_load_model_bundle_and_predict_feature_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_columns = ["x1", "x2"]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", DummyRegressor(strategy="constant", constant=42.0)),
                ]
            )
            model.fit(pd.DataFrame({"x1": [1.0], "x2": [2.0]}), [42.0])
            joblib.dump(model, root / "model.joblib")
            (root / "feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")

            bundle = load_model_bundle(root)
            pred = predict_feature_frame(
                pd.DataFrame({"sample_id": ["a"], "x1": [1.0], "x2": [np.nan]}),
                bundle,
            )
            self.assertEqual(float(pred.loc[0, "tg_c_pred"]), 42.0)
            self.assertEqual(float(pred.loc[0, "tg_k_pred"]), 315.15)
            self.assertEqual(pred.loc[0, "model_dir"], str(root))


if __name__ == "__main__":
    unittest.main()
