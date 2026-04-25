# Universal Single-Regressor Tg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one trained Tg regressor that accepts homopolymers, general copolymers, and nucleobase copolymers through one unified feature schema.

**Architecture:** Add a focused feature module that converts component sets into fixed-length descriptors, then add a training CLI that builds a unified table and trains one final regressor. Add a prediction CLI that loads the saved feature schema/model and predicts Tg for any one-row or CSV query without task-specific prediction routes.

**Tech Stack:** Python, pandas, numpy, scikit-learn, joblib, optional CatBoost, RDKit through existing project helpers, existing cached 186d component features.

---

## File Structure

- Create: `src/ml/universal_tg_features.py`
  - Owns component-set parsing, weight normalization, composition features, endpoint physical priors, aggregation of per-component vectors, and feature column ordering.
- Create: `scripts/train_universal_tg_single_regressor.py`
  - Owns data-source loading, unified table assembly, model selection, sample weighting, split evaluation, artifact writing, and experiment summary.
- Create: `scripts/predict_tg_universal_single_regressor.py`
  - Owns CLI prediction against a saved single-regressor model directory.
- Create: `tests/test_universal_tg_features.py`
  - Fast unit tests for feature utilities, no heavy model artifacts.
- Create: `tests/test_train_universal_tg_single_regressor.py`
  - Fast tests for weighting, metrics, and model fallback with synthetic data.
- Create: `tests/test_predict_tg_universal_single_regressor.py`
  - Fast tests for prediction row parsing and model artifact loading with a tiny joblib model.
- Modify only if needed: `requirements.txt`
  - Do not require CatBoost; keep it optional to avoid breaking Python 3.10 installs.

## Task 1: Feature Utility Tests

**Files:**
- Create: `tests/test_universal_tg_features.py`
- Create later: `src/ml/universal_tg_features.py`

- [ ] **Step 1: Write failing tests for weights and composition features**

Create `tests/test_universal_tg_features.py` with:

```python
import math
import unittest

import numpy as np

from src.ml.universal_tg_features import (
    ComponentRecord,
    PolymerRecord,
    fox_tg_c,
    normalize_weights,
    polymer_record_to_features,
)


class TestUniversalTgFeatures(unittest.TestCase):
    def test_normalize_weights_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "sum to a positive"):
            normalize_weights([0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_weights([0.5, -0.5])

    def test_normalize_weights_returns_unit_sum(self):
        weights = normalize_weights([2.0, 3.0])
        self.assertTrue(np.allclose(weights, [0.4, 0.6]))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_fox_tg_c_uses_kelvin_harmonic_mix(self):
        pred = fox_tg_c([0.0, 100.0], [0.5, 0.5])
        expected_k = 1.0 / (0.5 / 273.15 + 0.5 / 373.15)
        self.assertAlmostEqual(pred, expected_k - 273.15, places=6)

    def test_polymer_record_to_features_is_permutation_invariant_for_weighted_mean(self):
        a = ComponentRecord(
            smiles="A",
            vector=np.array([1.0, 3.0]),
            endpoint_tg_c=10.0,
            endpoint_source="measured",
        )
        b = ComponentRecord(
            smiles="B",
            vector=np.array([5.0, 7.0]),
            endpoint_tg_c=90.0,
            endpoint_source="measured",
        )
        rec1 = PolymerRecord(
            sample_id="ab",
            source="unit",
            architecture="random",
            components=[a, b],
            weights=[0.25, 0.75],
            target_tg_c=50.0,
        )
        rec2 = PolymerRecord(
            sample_id="ba",
            source="unit",
            architecture="random",
            components=[b, a],
            weights=[0.75, 0.25],
            target_tg_c=50.0,
        )
        row1 = polymer_record_to_features(rec1)
        row2 = polymer_record_to_features(rec2)
        self.assertAlmostEqual(row1["emb_mean_000"], row2["emb_mean_000"])
        self.assertAlmostEqual(row1["emb_mean_001"], row2["emb_mean_001"])
        self.assertAlmostEqual(row1["endpoint_tg_weighted_mean_c"], row2["endpoint_tg_weighted_mean_c"])
        self.assertEqual(row1["n_components"], 2)
        self.assertEqual(row1["is_random"], 1.0)
        self.assertEqual(row1["is_homopolymer"], 0.0)

    def test_missing_endpoint_sets_indicator_and_nan_priors(self):
        rec = PolymerRecord(
            sample_id="x",
            source="unit",
            architecture="homo",
            components=[
                ComponentRecord(
                    smiles="X",
                    vector=np.array([2.0]),
                    endpoint_tg_c=None,
                    endpoint_source="missing",
                )
            ],
            weights=[1.0],
            target_tg_c=20.0,
        )
        row = polymer_record_to_features(rec)
        self.assertEqual(row["endpoint_missing_count"], 1.0)
        self.assertTrue(math.isnan(row["endpoint_tg_fox_c"]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m pytest tests/test_universal_tg_features.py -q
```

Expected: import failure for `src.ml.universal_tg_features`.

## Task 2: Feature Utility Implementation

**Files:**
- Create: `src/ml/universal_tg_features.py`
- Test: `tests/test_universal_tg_features.py`

- [ ] **Step 1: Implement dataclasses and feature helpers**

Create `src/ml/universal_tg_features.py` with these public objects:

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


SUPPORTED_ARCHITECTURES = {"homo", "random", "block", "multicomponent"}


@dataclass(frozen=True)
class ComponentRecord:
    smiles: str
    vector: np.ndarray
    endpoint_tg_c: Optional[float] = None
    endpoint_source: str = "missing"


@dataclass(frozen=True)
class PolymerRecord:
    sample_id: str
    source: str
    architecture: str
    components: Sequence[ComponentRecord]
    weights: Sequence[float]
    target_tg_c: Optional[float] = None
    metadata: Optional[Mapping[str, object]] = None


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence.")
    if not np.isfinite(arr).all() or np.any(arr < 0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return arr / total


def fox_tg_c(endpoint_tg_c: Sequence[float], weights: Sequence[float]) -> float:
    values_c = np.asarray(endpoint_tg_c, dtype=float)
    w = normalize_weights(weights)
    if values_c.shape != w.shape or not np.isfinite(values_c).all():
        return float("nan")
    values_k = values_c + 273.15
    if np.any(values_k <= 0):
        return float("nan")
    denom = float(np.sum(w / values_k))
    return float(1.0 / denom - 273.15) if denom > 0 else float("nan")


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def polymer_record_to_features(record: PolymerRecord) -> dict[str, float | str]:
    if not record.components:
        raise ValueError("record must contain at least one component.")
    architecture = str(record.architecture or "").strip().lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {record.architecture}")
    weights = normalize_weights(record.weights)
    if len(weights) != len(record.components):
        raise ValueError("weights length must match components length.")

    vectors = [np.asarray(component.vector, dtype=float).reshape(-1) for component in record.components]
    dim = int(vectors[0].shape[0])
    if any(vec.shape[0] != dim for vec in vectors):
        raise ValueError("all component vectors must have the same dimension.")
    matrix = np.vstack(vectors)
    weighted_mean = np.sum(matrix * weights[:, None], axis=0)
    weighted_var = np.sum(((matrix - weighted_mean) ** 2) * weights[:, None], axis=0)
    weighted_std = np.sqrt(np.maximum(weighted_var, 0.0))
    min_vec = np.min(matrix, axis=0)
    max_vec = np.max(matrix, axis=0)
    contrast = max_vec - min_vec

    row: dict[str, float | str] = {
        "sample_id": record.sample_id,
        "source": record.source,
        "architecture": architecture,
        "n_components": float(len(record.components)),
        "w_max": float(np.max(weights)),
        "w_min": float(np.min(weights)),
        "w_entropy": float(-np.sum([w * math.log(w) for w in weights if w > 0])),
        "w_herfindahl": float(np.sum(weights**2)),
        "is_homopolymer": 1.0 if len(record.components) == 1 or architecture == "homo" else 0.0,
        "is_random": 1.0 if architecture == "random" else 0.0,
        "is_block": 1.0 if architecture == "block" else 0.0,
        "is_multicomponent": 1.0 if len(record.components) > 2 or architecture == "multicomponent" else 0.0,
    }

    for idx in range(5):
        row[f"w_sorted_{idx + 1}"] = float(sorted(weights, reverse=True)[idx]) if idx < len(weights) else 0.0

    for idx in range(dim):
        row[f"emb_mean_{idx:03d}"] = float(weighted_mean[idx])
        row[f"emb_std_{idx:03d}"] = float(weighted_std[idx])
        row[f"emb_min_{idx:03d}"] = float(min_vec[idx])
        row[f"emb_max_{idx:03d}"] = float(max_vec[idx])
        row[f"emb_contrast_{idx:03d}"] = float(contrast[idx])

    endpoints = [_safe_float(component.endpoint_tg_c) for component in record.components]
    endpoint_arr = np.asarray(endpoints, dtype=float)
    finite_mask = np.isfinite(endpoint_arr)
    row["endpoint_missing_count"] = float(np.size(endpoint_arr) - int(np.sum(finite_mask)))
    row["endpoint_missing_fraction"] = float(row["endpoint_missing_count"] / len(endpoint_arr))
    if finite_mask.all():
        row["endpoint_tg_min_c"] = float(np.min(endpoint_arr))
        row["endpoint_tg_max_c"] = float(np.max(endpoint_arr))
        row["endpoint_tg_mean_c"] = float(np.mean(endpoint_arr))
        row["endpoint_tg_weighted_mean_c"] = float(np.sum(endpoint_arr * weights))
        row["endpoint_tg_delta_c"] = float(np.max(endpoint_arr) - np.min(endpoint_arr))
        row["endpoint_tg_fox_c"] = fox_tg_c(endpoint_arr, weights)
    else:
        for name in [
            "endpoint_tg_min_c",
            "endpoint_tg_max_c",
            "endpoint_tg_mean_c",
            "endpoint_tg_weighted_mean_c",
            "endpoint_tg_delta_c",
            "endpoint_tg_fox_c",
        ]:
            row[name] = float("nan")

    if record.target_tg_c is not None:
        row["target_tg_c"] = float(record.target_tg_c)
    return row


def numeric_feature_columns(frame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in {"sample_id", "source", "architecture", "target_tg_c", "split_group"}
        and np.issubdtype(frame[column].dtype, np.number)
    ]
```

- [ ] **Step 2: Run feature tests**

Run:

```bash
python -m pytest tests/test_universal_tg_features.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit feature utilities**

Run:

```bash
git add src/ml/universal_tg_features.py tests/test_universal_tg_features.py
git commit -m "feat: add universal Tg feature utilities"
```

## Task 3: Training Script Unit Tests

**Files:**
- Create: `tests/test_train_universal_tg_single_regressor.py`
- Create later: `scripts/train_universal_tg_single_regressor.py`

- [ ] **Step 1: Write failing tests for sample weights and metrics**

Create `tests/test_train_universal_tg_single_regressor.py` with:

```python
import unittest

import numpy as np
import pandas as pd

from scripts.train_universal_tg_single_regressor import (
    choose_model,
    compute_metrics,
    make_sample_weights,
)


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
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m pytest tests/test_train_universal_tg_single_regressor.py -q
```

Expected: import failure for `scripts.train_universal_tg_single_regressor`.

## Task 4: Training Script Implementation

**Files:**
- Create: `scripts/train_universal_tg_single_regressor.py`
- Test: `tests/test_train_universal_tg_single_regressor.py`

- [ ] **Step 1: Implement public helpers and CLI skeleton**

Create `scripts/train_universal_tg_single_regressor.py` with:

```python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.universal_tg_features import numeric_feature_columns


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan"),
    }


def choose_model(name: str, random_state: int = 42):
    key = str(name).lower()
    if key == "catboost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=1600,
                depth=6,
                learning_rate=0.03,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )
        except Exception:
            key = "extratrees"
    if key == "histgradient":
        return HistGradientBoostingRegressor(
            max_iter=800,
            learning_rate=0.035,
            l2_regularization=0.02,
            random_state=random_state,
        )
    if key == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=1,
            max_features=0.65,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"unsupported model: {name}")


def make_sample_weights(
    frame: pd.DataFrame,
    homopolymer_weight: float,
    virtual_weight: float,
    copolymer_weight: float,
    nucleobase_weight: float,
) -> np.ndarray:
    source = frame["source"].astype(str).str.lower()
    weights = np.ones(len(frame), dtype=float)
    weights[source.str.contains("homopolymer")] = homopolymer_weight
    weights[source.str.contains("virtual")] = virtual_weight
    weights[source.str.contains("polyinfo|copolymer_real")] = copolymer_weight
    weights[source.str.contains("nucleobase")] = nucleobase_weight
    return weights


def load_unified_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def fit_model(frame: pd.DataFrame, feature_columns: list[str], model_name: str, random_state: int):
    model = choose_model(model_name, random_state=random_state)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def evaluate_holdout(frame: pd.DataFrame, feature_columns: list[str], args: argparse.Namespace) -> tuple[object, dict, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=frame["source"] if frame["source"].nunique() > 1 and frame["source"].value_counts().min() >= 2 else None,
    )
    estimator = fit_model(frame, feature_columns, args.model, args.random_state)
    weights = make_sample_weights(
        train,
        args.homopolymer_weight,
        args.virtual_weight,
        args.copolymer_weight,
        args.nucleobase_weight,
    )
    estimator.fit(train[feature_columns], train["target_tg_c"], model__sample_weight=weights)
    pred = estimator.predict(test[feature_columns])
    details = test[["sample_id", "source", "architecture", "target_tg_c"]].copy()
    details["pred_tg_c"] = pred
    details["split"] = "holdout"
    summary = {"overall_holdout": compute_metrics(details["target_tg_c"], details["pred_tg_c"])}
    for source, group in details.groupby("source"):
        summary[f"holdout_{source}"] = compute_metrics(group["target_tg_c"], group["pred_tg_c"])
    return estimator, summary, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one universal Tg single-regressor model.")
    parser.add_argument("--table", default="results/universal_single_regressor/unified_training_table.parquet")
    parser.add_argument("--output-dir", default="results/universal_single_regressor/exp_default")
    parser.add_argument("--model", default="catboost", choices=["catboost", "extratrees", "histgradient"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--homopolymer-weight", type=float, default=1.0)
    parser.add_argument("--virtual-weight", type=float, default=0.2)
    parser.add_argument("--copolymer-weight", type=float, default=10.0)
    parser.add_argument("--nucleobase-weight", type=float, default=20.0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_unified_table(Path(args.table))
    feature_columns = numeric_feature_columns(frame)
    frame = frame[np.isfinite(pd.to_numeric(frame["target_tg_c"], errors="coerce"))].copy()
    estimator, summary, details = evaluate_holdout(frame, feature_columns, args)
    joblib.dump(estimator, out_dir / "model.joblib")
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    details.to_csv(out_dir / "predictions_by_split.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run training helper tests**

Run:

```bash
python -m pytest tests/test_train_universal_tg_single_regressor.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit training CLI skeleton**

Run:

```bash
git add scripts/train_universal_tg_single_regressor.py tests/test_train_universal_tg_single_regressor.py
git commit -m "feat: add universal Tg single-regressor training CLI"
```

## Task 5: Unified Table Builder

**Files:**
- Modify: `scripts/train_universal_tg_single_regressor.py`
- Test: `tests/test_train_universal_tg_single_regressor.py`

- [ ] **Step 1: Add synthetic table-builder test**

Append this test to `tests/test_train_universal_tg_single_regressor.py`:

```python
    def test_build_table_from_records_returns_numeric_features(self):
        from src.ml.universal_tg_features import ComponentRecord, PolymerRecord
        from scripts.train_universal_tg_single_regressor import build_table_from_records

        records = [
            PolymerRecord(
                sample_id="h1",
                source="homopolymer_real",
                architecture="homo",
                components=[ComponentRecord("A", np.array([1.0, 2.0]), 20.0, "measured")],
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
            ),
        ]
        table = build_table_from_records(records)
        self.assertEqual(len(table), 2)
        self.assertIn("emb_mean_000", table.columns)
        self.assertIn("endpoint_tg_fox_c", table.columns)
        self.assertTrue(np.isfinite(table.loc[1, "endpoint_tg_fox_c"]))
```

- [ ] **Step 2: Implement table builder helper**

Add to `scripts/train_universal_tg_single_regressor.py`:

```python
from src.ml.universal_tg_features import PolymerRecord, polymer_record_to_features


def build_table_from_records(records: list[PolymerRecord]) -> pd.DataFrame:
    rows = [polymer_record_to_features(record) for record in records]
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column not in {"sample_id", "source", "architecture"}:
            frame[column] = pd.to_numeric(frame[column], errors="ignore")
    return frame
```

- [ ] **Step 3: Run builder tests**

Run:

```bash
python -m pytest tests/test_train_universal_tg_single_regressor.py -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit table-builder helper**

Run:

```bash
git add scripts/train_universal_tg_single_regressor.py tests/test_train_universal_tg_single_regressor.py
git commit -m "feat: add universal Tg training table builder"
```

## Task 6: Prediction Script Tests

**Files:**
- Create: `tests/test_predict_tg_universal_single_regressor.py`
- Create later: `scripts/predict_tg_universal_single_regressor.py`

- [ ] **Step 1: Write failing prediction tests**

Create `tests/test_predict_tg_universal_single_regressor.py` with:

```python
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

from scripts.predict_tg_universal_single_regressor import load_model_bundle, predict_feature_frame


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
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m pytest tests/test_predict_tg_universal_single_regressor.py -q
```

Expected: import failure for `scripts.predict_tg_universal_single_regressor`.

## Task 7: Prediction Script Implementation

**Files:**
- Create: `scripts/predict_tg_universal_single_regressor.py`
- Test: `tests/test_predict_tg_universal_single_regressor.py`

- [ ] **Step 1: Implement model bundle loader and feature-frame prediction**

Create `scripts/predict_tg_universal_single_regressor.py` with:

```python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ModelBundle:
    model: object
    feature_columns: list[str]
    model_dir: Path


def load_model_bundle(model_dir: Path) -> ModelBundle:
    model_path = model_dir / "model.joblib"
    feature_path = model_dir / "feature_columns.json"
    if not model_path.exists():
        raise FileNotFoundError(f"missing model artifact: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"missing feature schema: {feature_path}")
    model = joblib.load(model_path)
    feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    return ModelBundle(model=model, feature_columns=list(feature_columns), model_dir=model_dir)


def predict_feature_frame(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = frame.copy()
    for column in bundle.feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    pred_c = bundle.model.predict(out[bundle.feature_columns])
    out["tg_c_pred"] = np.asarray(pred_c, dtype=float)
    out["tg_k_pred"] = out["tg_c_pred"] + 273.15
    out["model_dir"] = str(bundle.model_dir)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Tg with a saved universal single-regressor model.")
    parser.add_argument("--features-csv", required=True, help="CSV already using the saved feature schema.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = load_model_bundle(Path(args.model_dir))
    frame = pd.read_csv(args.features_csv)
    pred = predict_feature_frame(frame, bundle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run prediction tests**

Run:

```bash
python -m pytest tests/test_predict_tg_universal_single_regressor.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit prediction CLI skeleton**

Run:

```bash
git add scripts/predict_tg_universal_single_regressor.py tests/test_predict_tg_universal_single_regressor.py
git commit -m "feat: add universal Tg single-regressor prediction CLI"
```

## Task 8: Server Data Integration Experiment

**Files:**
- Modify: `scripts/train_universal_tg_single_regressor.py`
- Output on server: `results/universal_single_regressor/`

- [ ] **Step 1: Add source-loader functions behind CLI flags**

Extend the training script with optional loaders:

```text
--build-table
--homopolymer-data data/unified_tg.parquet
--polyinfo-details results/copolymer_residual_model/polyinfo_physics_details_clean.csv
--nucleobase-details results/copolymer_residual_model/nucleobase_strategy_details.csv
--virtual-csv results/virtual_data/bicerano_binary_random_100k_reuse.csv
```

Implementation rule:

```text
If optional source file is missing, continue and record it in summary["missing_sources"].
```

- [ ] **Step 2: Run smoke table build on server**

Run inside `~/Tgprediction`:

```bash
/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python scripts/train_universal_tg_single_regressor.py \
  --build-table \
  --output-dir results/universal_single_regressor/smoke \
  --model extratrees \
  --virtual-weight 0.1 \
  --copolymer-weight 10 \
  --nucleobase-weight 20
```

Expected:

```text
results/universal_single_regressor/smoke/unified_training_table.parquet
results/universal_single_regressor/smoke/model.joblib
results/universal_single_regressor/smoke/summary.json
```

- [ ] **Step 3: Review smoke metrics**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("results/universal_single_regressor/smoke/summary.json")
print(json.dumps(json.loads(p.read_text()), indent=2)[:4000])
PY
```

Expected: summary contains `overall_holdout` and per-source holdout metrics.

- [ ] **Step 4: Commit data integration**

Run:

```bash
git add scripts/train_universal_tg_single_regressor.py
git commit -m "feat: integrate universal Tg training data sources"
```

## Task 9: Iterative Experiments

**Files:**
- Output on server: `results/universal_single_regressor/`
- Update if useful: `docs/阶段性成果-统一高分子Tg预测路由-2026-04-25.md`

- [ ] **Step 1: Run baseline without virtual dominance**

Run:

```bash
/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python scripts/train_universal_tg_single_regressor.py \
  --build-table \
  --output-dir results/universal_single_regressor/exp01_real_weighted_extratrees \
  --model extratrees \
  --virtual-weight 0.0 \
  --copolymer-weight 10 \
  --nucleobase-weight 20
```

- [ ] **Step 2: Run virtual low-weight experiment**

Run:

```bash
/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python scripts/train_universal_tg_single_regressor.py \
  --build-table \
  --output-dir results/universal_single_regressor/exp02_virtual_010_extratrees \
  --model extratrees \
  --virtual-weight 0.1 \
  --copolymer-weight 10 \
  --nucleobase-weight 20
```

- [ ] **Step 3: Run stronger real-copolymer weighting experiment**

Run:

```bash
/home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python scripts/train_universal_tg_single_regressor.py \
  --build-table \
  --output-dir results/universal_single_regressor/exp03_real_copolymer_20_extratrees \
  --model extratrees \
  --virtual-weight 0.1 \
  --copolymer-weight 20 \
  --nucleobase-weight 20
```

- [ ] **Step 4: Summarize experiments**

Create `results/universal_single_regressor/experiment_log.md` with:

```markdown
# Universal Single-Regressor Tg Experiments

## Exp01

Hypothesis: real data with physical priors gives the strongest leakage-safe baseline.

Result: record MAE/RMSE/R2 for homopolymer, PolyInfo, and nucleobase.

## Exp02

Hypothesis: low-weight virtual copolymer data improves composition smoothness without hurting real subsets.

Result: record metric deltas versus Exp01.

## Exp03

Hypothesis: higher real-copolymer weighting improves general copolymer prediction.

Result: record metric deltas versus Exp02.
```

- [ ] **Step 5: Commit experiment script/doc updates only**

Run:

```bash
git add scripts/train_universal_tg_single_regressor.py docs/阶段性成果-统一高分子Tg预测路由-2026-04-25.md
git commit -m "docs: summarize universal single-regressor experiments"
```

Only include the docs file if it was intentionally updated.

## Task 10: Verification and Final Handoff

**Files:**
- Relevant changed source and tests

- [ ] **Step 1: Run fast local tests**

Run:

```bash
python -m pytest \
  tests/test_universal_tg_features.py \
  tests/test_train_universal_tg_single_regressor.py \
  tests/test_predict_tg_universal_single_regressor.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Check git status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing untracked files remain, or no changes if all task files are committed.

- [ ] **Step 3: Report metrics**

Final report must include:

```text
homopolymer heldout: MAE, RMSE, R2, n
general copolymer: MAE, RMSE, R2, n
nucleobase: MAE, RMSE, R2, n
best experiment directory
known limitations
```

## Self-Review

- Spec coverage: the plan covers unified table building, one final regressor, prediction with one saved model, per-family metrics, virtual-data weighting, and experiment summaries.
- Red-flag scan: no incomplete markers or unspecified implementation step is intentionally left.
- Type consistency: public objects are `ComponentRecord`, `PolymerRecord`, `polymer_record_to_features`, `numeric_feature_columns`, `choose_model`, `make_sample_weights`, `compute_metrics`, `load_model_bundle`, and `predict_feature_frame`.
