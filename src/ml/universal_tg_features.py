from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SUPPORTED_ARCHITECTURES = {"homo", "random", "block", "multicomponent"}
NON_FEATURE_COLUMNS = {
    "sample_id",
    "source",
    "architecture",
    "target_tg_c",
    "split_group",
    "component_smiles",
    "weights_serialized",
}


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
    split_group: Optional[str] = None
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


def _architecture_flags(architecture: str, n_components: int) -> dict[str, float]:
    architecture = str(architecture or "").strip().lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture}")
    return {
        "is_homopolymer": 1.0 if n_components == 1 or architecture == "homo" else 0.0,
        "is_random": 1.0 if architecture == "random" else 0.0,
        "is_block": 1.0 if architecture == "block" else 0.0,
        "is_multicomponent": 1.0 if n_components > 2 or architecture == "multicomponent" else 0.0,
    }


def polymer_record_to_features(record: PolymerRecord) -> dict[str, float | str]:
    if not record.components:
        raise ValueError("record must contain at least one component.")
    architecture = str(record.architecture or "").strip().lower()
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
        "sample_id": str(record.sample_id),
        "source": str(record.source),
        "architecture": architecture,
        "n_components": float(len(record.components)),
        "w_max": float(np.max(weights)),
        "w_min": float(np.min(weights)),
        "w_entropy": float(-np.sum([w * math.log(w) for w in weights if w > 0])),
        "w_herfindahl": float(np.sum(weights**2)),
        "component_smiles": "|".join(component.smiles for component in record.components),
        "weights_serialized": "|".join(f"{float(w):.8f}" for w in weights),
    }
    row.update(_architecture_flags(architecture, len(record.components)))
    if record.split_group is not None:
        row["split_group"] = str(record.split_group)

    sorted_weights = sorted((float(w) for w in weights), reverse=True)
    for idx in range(5):
        row[f"w_sorted_{idx + 1}"] = sorted_weights[idx] if idx < len(sorted_weights) else 0.0

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
    row["endpoint_all_measured"] = (
        1.0
        if all(str(component.endpoint_source).lower() == "measured" for component in record.components)
        else 0.0
    )

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

    if record.metadata:
        for key, value in record.metadata.items():
            if key in row:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[str(key)] = _safe_float(value)
            elif isinstance(value, str):
                row[str(key)] = value

    return row


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns
