"""
Evaluate endpoint-based physics baselines on parsed PoLyInfo copolymer Tg rows.

This script uses the deduplicated 7k homopolymer database only to retrieve
component endpoint Tg values. It does not load the expensive teacher model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_tg_tabpfn_186d import _canonical_repeat_unit_key
from src.ml.copolymer_physics import (
    BinaryCopolymerPoint,
    estimate_gordon_taylor_k,
    fit_kwei_k_q,
    fox_tg_k,
    gordon_taylor_binary_tg_k,
    kwei_binary_tg_k,
)
from src.ml.copolymer_tg_model import regression_metrics


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _component_tg_lookup(unified_tg: pd.DataFrame) -> Dict[str, float]:
    lookup: Dict[str, float] = {}
    for _, row in unified_tg.iterrows():
        tg_k = _finite_float(row.get("tg_k"))
        if tg_k is None:
            continue
        key = _canonical_repeat_unit_key(str(row.get("smiles", "")))
        if key is not None:
            lookup[key] = tg_k
    return lookup


def _repeat_unit_mw(smiles: object) -> Optional[float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mw = float(Descriptors.MolWt(mol))
    return mw if math.isfinite(mw) and mw > 0.0 else None


def _component1_fraction(row: pd.Series, composition_basis: str) -> Optional[float]:
    ratio_1 = _finite_float(row.get("ratio_1"))
    if ratio_1 is None:
        return None
    fraction = ratio_1 / 100.0
    if not 0.0 <= fraction <= 1.0:
        return None

    ratio_unit = str(row.get("ratio_unit", "")).strip().lower()
    if composition_basis == "weight" and ratio_unit == "mol%":
        mw1 = _repeat_unit_mw(row.get("SMILES_1"))
        mw2 = _repeat_unit_mw(row.get("SMILES_2"))
        if mw1 is not None and mw2 is not None:
            denom = fraction * mw1 + (1.0 - fraction) * mw2
            if denom > 0.0:
                fraction = fraction * mw1 / denom
    return fraction


def _load_points(
    real_csv: Path,
    unified_tg_path: Path,
    composition_basis: str,
) -> tuple[pd.DataFrame, list[BinaryCopolymerPoint], list[str]]:
    real = pd.read_csv(real_csv)
    unified = pd.read_parquet(unified_tg_path)
    endpoint_tg = _component_tg_lookup(unified)

    rows = []
    points: list[BinaryCopolymerPoint] = []
    labels: list[str] = []
    for index, row in real.iterrows():
        smiles1 = str(row.get("SMILES_1", ""))
        smiles2 = str(row.get("SMILES_2", ""))
        key1 = _canonical_repeat_unit_key(smiles1)
        key2 = _canonical_repeat_unit_key(smiles2)
        tg_c = _finite_float(row.get("Tg_C"))
        w1 = _component1_fraction(row, composition_basis)
        reason = ""
        if key1 is None or key2 is None:
            reason = "invalid component SMILES"
        elif key1 not in endpoint_tg or key2 not in endpoint_tg:
            reason = "missing 7k endpoint Tg"
        elif tg_c is None:
            reason = "missing target Tg"
        elif w1 is None:
            reason = "missing usable composition"

        base_row = dict(row)
        base_row.update(
            {
                "source_row_index": int(index),
                "composition_basis": composition_basis,
                "canonical_1": key1 or "",
                "canonical_2": key2 or "",
                "status": "skipped" if reason else "usable",
                "reason": reason,
            }
        )
        if reason:
            rows.append(base_row)
            continue

        point = BinaryCopolymerPoint(
            tg1_k=endpoint_tg[str(key1)],
            tg2_k=endpoint_tg[str(key2)],
            w1=float(w1),
            target_tg_k=float(tg_c) + 273.15,
        )
        base_row.update(
            {
                "w1_used": float(w1),
                "w2_used": 1.0 - float(w1),
                "endpoint_1_tg_c": point.tg1_k - 273.15,
                "endpoint_2_tg_c": point.tg2_k - 273.15,
            }
        )
        rows.append(base_row)
        points.append(point)
        labels.append(str(row.get("COID", "")))
    return pd.DataFrame(rows), points, labels


def _loocv_kwei(points: list[BinaryCopolymerPoint]) -> np.ndarray:
    preds = []
    for index, point in enumerate(points):
        train = [p for i, p in enumerate(points) if i != index]
        if len(train) >= 2:
            k, q = fit_kwei_k_q(train)
        else:
            k, q = estimate_gordon_taylor_k(point.tg1_k, point.tg2_k), 0.0
        preds.append(kwei_binary_tg_k(point.tg1_k, point.tg2_k, point.w1, k, q))
    return np.asarray(preds, dtype=float)


def _leave_group_out_kwei(points: list[BinaryCopolymerPoint], labels: Sequence[str]) -> np.ndarray:
    preds = []
    for index, point in enumerate(points):
        train = [p for i, p in enumerate(points) if labels[i] != labels[index]]
        if len(train) >= 2:
            k, q = fit_kwei_k_q(train)
        else:
            k, q = estimate_gordon_taylor_k(point.tg1_k, point.tg2_k), 0.0
        preds.append(kwei_binary_tg_k(point.tg1_k, point.tg2_k, point.w1, k, q))
    return np.asarray(preds, dtype=float)


def _same_group_preferred_kwei(points: list[BinaryCopolymerPoint], labels: Sequence[str]) -> np.ndarray:
    preds = []
    for index, point in enumerate(points):
        same_group = [p for i, p in enumerate(points) if labels[i] == labels[index] and i != index]
        train = same_group if len(same_group) >= 2 else [p for i, p in enumerate(points) if i != index]
        if len(train) >= 2:
            k, q = fit_kwei_k_q(train)
        else:
            k, q = estimate_gordon_taylor_k(point.tg1_k, point.tg2_k), 0.0
        preds.append(kwei_binary_tg_k(point.tg1_k, point.tg2_k, point.w1, k, q))
    return np.asarray(preds, dtype=float)


def _ridge_features(
    points: list[BinaryCopolymerPoint],
    labels: Sequence[str],
    categories: Sequence[str],
) -> np.ndarray:
    rows = []
    for point, label in zip(points, labels):
        w2 = 1.0 - point.w1
        interaction = point.w1 * w2
        fox = fox_tg_k([point.tg1_k, point.tg2_k], [point.w1, w2])
        rows.append(
            [
                fox,
                point.w1,
                w2,
                w2 * w2,
                interaction,
                point.tg2_k - point.tg1_k,
                *(interaction if label == category else 0.0 for category in categories),
            ]
        )
    return np.asarray(rows, dtype=float)


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if len(y_train) == 0:
        return np.full(len(x_test), float("nan"))
    if len(y_train) < 3:
        return np.full(len(x_test), float(np.mean(y_train)))

    from sklearn.linear_model import RidgeCV

    model = RidgeCV(alphas=np.logspace(-4, 4, 30))
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


def _ridge_cv_predictions(
    points: list[BinaryCopolymerPoint],
    labels: Sequence[str],
    mode: str,
) -> np.ndarray:
    categories = sorted(set(labels))
    x = _ridge_features(points, labels, categories)
    y = np.asarray([point.target_tg_k for point in points], dtype=float)
    labels_array = np.asarray(labels)
    preds = []
    for index in range(len(points)):
        if mode == "leave_group_out":
            train = np.flatnonzero(labels_array != labels_array[index])
        else:
            train = np.array([i for i in range(len(points)) if i != index], dtype=int)
        preds.append(_ridge_fit_predict(x[train], y[train], x[[index]])[0])
    return np.asarray(preds, dtype=float)


def _ridge_fit_predictions(points: list[BinaryCopolymerPoint], labels: Sequence[str]) -> np.ndarray:
    categories = sorted(set(labels))
    x = _ridge_features(points, labels, categories)
    y = np.asarray([point.target_tg_k for point in points], dtype=float)
    return _ridge_fit_predict(x, y, x)


def _metrics(y_true_k: np.ndarray, y_pred_k: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true_k) & np.isfinite(y_pred_k)
    if not valid.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return regression_metrics(y_true_k[valid], y_pred_k[valid])


def evaluate_polyinfo_physics(
    real_csv: Path,
    unified_tg_path: Path,
    composition_basis: str,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    details, points, labels = _load_points(real_csv, unified_tg_path, composition_basis)
    usable_index = details.index[details["status"].eq("usable")].to_numpy()
    if not points:
        return details, {
            "composition_basis": composition_basis,
            "n_rows": int(len(details)),
            "n_usable": 0,
            "status_counts": details["status"].value_counts(dropna=False).to_dict(),
            "overall": {},
        }

    y_true = np.asarray([point.target_tg_k for point in points], dtype=float)
    fox = np.asarray([
        fox_tg_k([point.tg1_k, point.tg2_k], [point.w1, 1.0 - point.w1])
        for point in points
    ])
    gt_est = np.asarray([
        gordon_taylor_binary_tg_k(
            point.tg1_k,
            point.tg2_k,
            point.w1,
            estimate_gordon_taylor_k(point.tg1_k, point.tg2_k),
        )
        for point in points
    ])
    predictions = {
        "fox_endpoint_tg_c": fox - 273.15,
        "gt_est_endpoint_tg_c": gt_est - 273.15,
        "kwei_loocv_endpoint_tg_c": _loocv_kwei(points) - 273.15,
        "kwei_leave_system_out_endpoint_tg_c": _leave_group_out_kwei(points, labels) - 273.15,
        "kwei_same_system_pref_loocv_endpoint_tg_c": _same_group_preferred_kwei(points, labels) - 273.15,
        "physics_ridge_loocv_endpoint_tg_c": _ridge_cv_predictions(points, labels, "loocv") - 273.15,
        "physics_ridge_leave_system_out_endpoint_tg_c": (
            _ridge_cv_predictions(points, labels, "leave_group_out") - 273.15
        ),
        "physics_ridge_fit_endpoint_tg_c": _ridge_fit_predictions(points, labels) - 273.15,
    }
    for name, values in predictions.items():
        details[name] = float("nan")
        details.loc[usable_index, name] = values

    methods = list(predictions.keys())
    overall = {
        method: _metrics(y_true, details.loc[usable_index, method].to_numpy(dtype=float) + 273.15)
        for method in methods
    }
    by_ratio_unit: Dict[str, Dict[str, Dict[str, float]]] = {}
    usable = details.loc[usable_index].copy()
    for ratio_unit, group in usable.groupby("ratio_unit", dropna=False):
        group_y = group["Tg_C"].to_numpy(dtype=float) + 273.15
        by_ratio_unit[str(ratio_unit)] = {
            method: _metrics(group_y, group[method].to_numpy(dtype=float) + 273.15)
            for method in methods
        }

    summary: Dict[str, object] = {
        "composition_basis": composition_basis,
        "n_rows": int(len(details)),
        "n_usable": int(len(points)),
        "n_systems": int(len(set(labels))),
        "status_counts": details["status"].value_counts(dropna=False).to_dict(),
        "ratio_unit_counts_usable": usable["ratio_unit"].value_counts(dropna=False).to_dict(),
        "overall": overall,
        "by_ratio_unit": by_ratio_unit,
    }
    return details, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PoLyInfo copolymer endpoint physics baselines.")
    parser.add_argument("--real-csv", default=_default_path("data", "external", "polyinfo_copolymer_tg.csv"))
    parser.add_argument("--unified-tg", default=_default_path("data", "unified_tg.parquet"))
    parser.add_argument(
        "--composition-basis",
        choices=["reported", "weight"],
        default="weight",
        help="Use reported ratio as-is, or convert mol%% rows to approximate weight fraction.",
    )
    parser.add_argument(
        "--output-csv",
        default=_default_path("results", "copolymer_residual_model", "polyinfo_physics_details.csv"),
    )
    parser.add_argument(
        "--summary-json",
        default=_default_path("results", "copolymer_residual_model", "polyinfo_physics_summary.json"),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    details, summary = evaluate_polyinfo_physics(
        Path(args.real_csv),
        Path(args.unified_tg),
        args.composition_basis,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved details: {output_csv}")
    print(f"Saved summary: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
