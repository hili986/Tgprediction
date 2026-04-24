from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CopolymerRecord:
    components: Tuple[str, ...]
    weights: Tuple[float, ...]
    architecture: str
    target_tg_k: Optional[float]
    metadata: Dict[str, object]


@dataclass(frozen=True)
class FeatureMatrixResult:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    records: List[CopolymerRecord]
    errors: List[str]


TARGET_COLUMN_CANDIDATES = (
    "tg_k",
    "Tg_K",
    "target_tg_k",
    "tg_k_exp",
    "tg_k_pred",
    "Tg",
    "tg_c",
    "Tg_C",
    "target_tg_c",
)


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _lookup(row: pd.Series, *names: str) -> object:
    lower_to_key = {str(key).lower(): key for key in row.index}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def normalize_weights(weights: Sequence[float]) -> Tuple[float, ...]:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("weights must be a non-empty 1D sequence.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value.")
    return tuple((values / total).astype(float))


def _parse_pipe_values(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _parse_components_field(text: str) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    components: List[str] = []
    weights: List[Optional[float]] = []
    for part in _parse_pipe_values(text):
        if "::" in part:
            component, weight = part.rsplit("::", 1)
            components.append(component.strip())
            weights.append(float(weight.strip()))
        else:
            components.append(part)
            weights.append(None)
    if not components:
        raise ValueError("No components found.")
    if all(weight is not None for weight in weights):
        return tuple(components), normalize_weights([float(weight) for weight in weights])
    return tuple(components), tuple([1.0 / len(components)] * len(components))


def _parse_components_and_weights(row: pd.Series) -> Tuple[Tuple[str, ...], Tuple[float, ...]]:
    serialized_components = _clean_text(_lookup(row, "components_serialized"))
    serialized_weights = _clean_text(_lookup(row, "weights_serialized"))
    if serialized_components:
        components = _parse_pipe_values(serialized_components)
        if serialized_weights:
            weights = normalize_weights([float(value) for value in _parse_pipe_values(serialized_weights)])
        else:
            weights = tuple([1.0 / len(components)] * len(components))
        return components, weights

    components_field = _clean_text(_lookup(row, "components"))
    if components_field:
        return _parse_components_field(components_field)

    indexed_components: Dict[int, str] = {}
    indexed_weights: Dict[int, float] = {}
    for key, value in row.items():
        key_text = str(key).lower()
        if key_text.startswith(("smiles_", "smiles", "component_", "component")):
            suffix = "".join(ch for ch in key_text if ch.isdigit())
            if suffix:
                text = _clean_text(value)
                if text:
                    indexed_components[int(suffix)] = text
        if key_text.startswith(("w_", "w", "weight_", "weight")):
            suffix = "".join(ch for ch in key_text if ch.isdigit())
            if suffix and _clean_text(value):
                indexed_weights[int(suffix)] = float(value)

    if not indexed_components:
        raise ValueError("No copolymer components found in row.")

    ordered = sorted(indexed_components)
    components = tuple(indexed_components[index] for index in ordered)
    if indexed_weights:
        weights = normalize_weights([indexed_weights.get(index, 0.0) for index in ordered])
        return components, weights

    ratio_1 = _lookup(row, "ratio_1", "ratio1")
    if len(components) == 2 and _clean_text(ratio_1):
        w1 = float(ratio_1)
        if w1 > 1.0:
            w1 /= 100.0
        return components, normalize_weights([w1, 1.0 - w1])

    return components, tuple([1.0 / len(components)] * len(components))


def _parse_architecture(row: pd.Series) -> str:
    raw = _clean_text(_lookup(row, "architecture"))
    if not raw:
        raw = _clean_text(_lookup(row, "copolymer_type"))
    text = raw.lower()
    if text in {"block", "b", "diblock", "triblock"}:
        return "block"
    return "random"


def _target_value_to_kelvin(column_name: str, value: object) -> Optional[float]:
    text = _clean_text(value)
    if not text:
        return None
    number = float(text)
    name = column_name.lower()
    if name.endswith("_c") or name in {"tg_c", "tgc", "target_tg_c"}:
        return number + 273.15
    return number


def _find_target(row: pd.Series, target_column: Optional[str]) -> Optional[float]:
    if target_column:
        value = _lookup(row, target_column)
        return _target_value_to_kelvin(target_column, value)
    for column in TARGET_COLUMN_CANDIDATES:
        value = _lookup(row, column)
        if _clean_text(value):
            return _target_value_to_kelvin(column, value)
    return None


def parse_copolymer_records(
    frame: pd.DataFrame,
    target_column: Optional[str] = None,
    keep_error_rows: bool = False,
) -> List[CopolymerRecord]:
    records: List[CopolymerRecord] = []
    for _, row in frame.iterrows():
        status = _clean_text(_lookup(row, "status")).lower()
        if status and status != "ok" and not keep_error_rows:
            continue
        try:
            components, weights = _parse_components_and_weights(row)
            target = _find_target(row, target_column)
        except Exception:
            continue
        if target is None or not np.isfinite(target):
            continue
        records.append(
            CopolymerRecord(
                components=tuple(components),
                weights=normalize_weights(weights),
                architecture=_parse_architecture(row),
                target_tg_k=float(target),
                metadata={str(key): row.get(key) for key in row.index},
            )
        )
    return records


def build_copolymer_feature_vector(
    record: CopolymerRecord,
    component_vectors: Sequence[np.ndarray],
    extra_scalars: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], np.ndarray]:
    if len(component_vectors) != len(record.weights):
        raise ValueError("component vector count must match weight count.")

    vectors = np.vstack([np.asarray(vector, dtype=float) for vector in component_vectors])
    weights = np.asarray(normalize_weights(record.weights), dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("component vectors must be a finite 2D matrix.")

    mix = np.sum(vectors * weights[:, None], axis=0)
    dispersion = np.sum(np.abs(vectors - mix[None, :]) * weights[:, None], axis=0)

    names = [f"mix_f{idx:03d}" for idx in range(vectors.shape[1])]
    names += [f"disp_f{idx:03d}" for idx in range(vectors.shape[1])]
    values = [*mix.tolist(), *dispersion.tolist()]

    entropy = -float(np.sum([w * math.log(max(w, 1e-12)) for w in weights]))
    scalars = {
        "n_components": float(len(weights)),
        "max_weight": float(np.max(weights)),
        "min_weight": float(np.min(weights)),
        "weight_entropy": entropy,
        "weight_herfindahl": float(np.sum(weights**2)),
        "architecture_random": 1.0 if record.architecture == "random" else 0.0,
        "architecture_block": 1.0 if record.architecture == "block" else 0.0,
    }
    if extra_scalars:
        for key, value in extra_scalars.items():
            if value is not None and np.isfinite(float(value)):
                scalars[str(key)] = float(value)

    for key in sorted(scalars):
        names.append(key)
        values.append(scalars[key])

    return names, np.asarray(values, dtype=float)


def _component_vector(predictor, smiles: str) -> np.ndarray:
    component = predictor.featurize_component(smiles)
    if hasattr(predictor, "_component_full_vector"):
        return np.asarray(predictor._component_full_vector(component), dtype=float)
    return np.hstack([component["phyc"], component["gnn"], component["pbert"]]).astype(float)


def _teacher_scalars(predictor, record: CopolymerRecord) -> Dict[str, float]:
    result = predictor.predict_multicomponent(
        list(record.components),
        list(record.weights),
        architecture=record.architecture,
    )
    window = result.get("component_tg_window_k") or [0.0, 0.0]
    window_min = float(window[0]) if window[0] is not None else 0.0
    window_max = float(window[1]) if window[1] is not None else 0.0
    fox = result.get("fox_reference_tg_k")
    return {
        "teacher_tg_k_pred": float(result.get("tg_k_pred", 0.0)),
        "teacher_descriptor_mix_tg_k": float(result.get("descriptor_mix_tg_k", result.get("tg_k_pred", 0.0))),
        "teacher_fox_reference_tg_k": 0.0 if fox is None else float(fox),
        "teacher_has_fox_reference": 0.0 if fox is None else 1.0,
        "teacher_component_window_min_k": window_min,
        "teacher_component_window_max_k": window_max,
        "teacher_component_window_span_k": window_max - window_min,
    }


def build_feature_matrix(
    records: Sequence[CopolymerRecord],
    predictor,
    include_teacher_scalars: bool = False,
) -> FeatureMatrixResult:
    rows: List[np.ndarray] = []
    y: List[float] = []
    kept_records: List[CopolymerRecord] = []
    errors: List[str] = []
    feature_names: Optional[List[str]] = None

    for index, record in enumerate(records):
        if record.target_tg_k is None or not np.isfinite(record.target_tg_k):
            errors.append(f"row {index}: missing finite target Tg.")
            continue
        try:
            component_vectors = [_component_vector(predictor, smiles) for smiles in record.components]
            extra_scalars = _teacher_scalars(predictor, record) if include_teacher_scalars else None
            names, values = build_copolymer_feature_vector(record, component_vectors, extra_scalars)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("Inconsistent feature names across copolymer records.")
            rows.append(values)
            y.append(float(record.target_tg_k))
            kept_records.append(record)
        except Exception as exc:
            errors.append(f"row {index}: {exc}")

    if not rows:
        raise ValueError("No valid copolymer records could be featurized.")

    return FeatureMatrixResult(
        X=np.vstack(rows),
        y=np.asarray(y, dtype=float),
        feature_names=feature_names or [],
        records=kept_records,
        errors=errors,
    )


def residual_feature_matrix(x: np.ndarray, base_predictions: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    base_predictions = np.asarray(base_predictions, dtype=float).reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("x must be a 2D feature matrix.")
    if len(x) != len(base_predictions):
        raise ValueError("x and base_predictions must have the same row count.")
    return np.hstack([x, base_predictions])


def fit_residual_corrector(x_real: np.ndarray, y_real: np.ndarray, base_predictions: np.ndarray):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_real = np.asarray(y_real, dtype=float)
    base_predictions = np.asarray(base_predictions, dtype=float)
    residual = y_real - base_predictions
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-6, 3, 10)),
    )
    model.fit(residual_feature_matrix(x_real, base_predictions), residual)
    return model


def predict_with_residual(
    base_predictions: np.ndarray,
    residual_model,
    x: np.ndarray,
) -> np.ndarray:
    base_predictions = np.asarray(base_predictions, dtype=float)
    if residual_model is None:
        return base_predictions
    residual = residual_model.predict(residual_feature_matrix(x, base_predictions))
    return base_predictions + np.asarray(residual, dtype=float)


def make_base_regressor(kind: str = "hgb", random_state: int = 42):
    clean_kind = str(kind).strip().lower()
    if clean_kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=random_state,
        )
    if clean_kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if clean_kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 4, 12)))
    raise ValueError("base model must be one of: hgb, extra_trees, ridge.")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def read_copolymer_records(path: Path, target_column: Optional[str] = None) -> List[CopolymerRecord]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    return parse_copolymer_records(frame, target_column=target_column)
