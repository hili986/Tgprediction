"""
Compare copolymer Tg strategies on the nucleobase acrylic workbook predictions.

Input is the CSV produced by predict_nucleobase_excel_with_copolymer_model.py.
The evaluation is intentionally cheap and does not load polyBERT/GNN artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.copolymer_physics import (
    BinaryCopolymerPoint,
    estimate_gordon_taylor_k,
    fit_gordon_taylor_k,
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


def _extract_endpoint_tg_c(predicted: pd.DataFrame, nucleobase: str) -> Optional[float]:
    if nucleobase == "none":
        subset = predicted[
            predicted["Architecture"].astype(str).str.contains("homopolymer baseline", case=False, na=False)
        ]
    else:
        subset = predicted[
            (predicted["Nucleobase"].astype(str) == nucleobase)
            & predicted["Architecture"].astype(str).str.contains("homopolymer", case=False, na=False)
        ]
    if subset.empty:
        return None
    return _finite_float(subset.iloc[0].get("Tg_C_actual"))


def _endpoint_maps(predicted: pd.DataFrame) -> tuple[Dict[str, float], Dict[str, float]]:
    actual: Dict[str, float] = {}
    model: Dict[str, float] = {}
    for base in ["none", "A", "T", "C", "G"]:
        actual_c = _extract_endpoint_tg_c(predicted, base)
        if actual_c is not None:
            actual[base] = actual_c + 273.15
        if base == "none":
            subset = predicted[
                predicted["Architecture"].astype(str).str.contains("homopolymer baseline", case=False, na=False)
            ]
        else:
            subset = predicted[
                (predicted["Nucleobase"].astype(str) == base)
                & predicted["Architecture"].astype(str).str.contains("homopolymer", case=False, na=False)
            ]
        if not subset.empty:
            pred_c = _finite_float(subset.iloc[0].get("base_pred_tg_c"))
            if pred_c is not None:
                model[base] = pred_c + 273.15
    return actual, model


def _binary_points(frame: pd.DataFrame, endpoints_k: Dict[str, float]) -> list[BinaryCopolymerPoint]:
    points = []
    for _, row in frame.iterrows():
        base = str(row.get("Nucleobase"))
        w_base = _finite_float(row.get("Polymer_mol_pct"))
        target_c = _finite_float(row.get("Tg_C_actual"))
        if base not in endpoints_k or "none" not in endpoints_k or w_base is None or target_c is None:
            continue
        points.append(
            BinaryCopolymerPoint(
                tg1_k=endpoints_k["none"],
                tg2_k=endpoints_k[base],
                w1=1.0 - w_base / 100.0,
                target_tg_k=target_c + 273.15,
            )
        )
    return points


def _loocv_gt(points: list[BinaryCopolymerPoint]) -> np.ndarray:
    preds = []
    for index, point in enumerate(points):
        train = [p for i, p in enumerate(points) if i != index]
        k = fit_gordon_taylor_k(train) if train else estimate_gordon_taylor_k(point.tg1_k, point.tg2_k)
        preds.append(gordon_taylor_binary_tg_k(point.tg1_k, point.tg2_k, point.w1, k))
    return np.asarray(preds, dtype=float)


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


def _metrics(y_true_k: np.ndarray, y_pred_k: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(y_true_k) & np.isfinite(y_pred_k)
    if not valid.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return regression_metrics(y_true_k[valid], y_pred_k[valid])


def evaluate_strategies(predictions: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    predicted = predictions[predictions["status"] == "predicted"].copy()
    random_rows = predicted[predicted["Architecture"].astype(str).eq("random copolymer")].copy()
    endpoints_actual, endpoints_model = _endpoint_maps(predicted)
    random_rows = random_rows.reset_index(drop=True)

    details = random_rows.copy()
    for endpoint_name, endpoints in [("actual_endpoint", endpoints_actual), ("model_endpoint", endpoints_model)]:
        fox_preds = []
        gt_est_preds = []
        for _, row in random_rows.iterrows():
            base = str(row.get("Nucleobase"))
            w_base = _finite_float(row.get("Polymer_mol_pct"))
            if base not in endpoints or "none" not in endpoints or w_base is None:
                fox_preds.append(float("nan"))
                gt_est_preds.append(float("nan"))
                continue
            w_nba = 1.0 - w_base / 100.0
            tg_nba = endpoints["none"]
            tg_base = endpoints[base]
            fox_preds.append(fox_tg_k([tg_nba, tg_base], [w_nba, 1.0 - w_nba]))
            k = estimate_gordon_taylor_k(tg_nba, tg_base)
            gt_est_preds.append(gordon_taylor_binary_tg_k(tg_nba, tg_base, w_nba, k))
        details[f"fox_{endpoint_name}_tg_c"] = np.asarray(fox_preds) - 273.15
        details[f"gt_est_{endpoint_name}_tg_c"] = np.asarray(gt_est_preds) - 273.15

    actual_points = _binary_points(random_rows, endpoints_actual)
    if len(actual_points) == len(random_rows) and actual_points:
        details["gt_loocv_actual_endpoint_tg_c"] = _loocv_gt(actual_points) - 273.15
        details["kwei_loocv_actual_endpoint_tg_c"] = _loocv_kwei(actual_points) - 273.15

        k_global, q_global = fit_kwei_k_q(actual_points)
        details["kwei_fit_actual_endpoint_tg_c"] = [
            kwei_binary_tg_k(point.tg1_k, point.tg2_k, point.w1, k_global, q_global) - 273.15
            for point in actual_points
        ]
    else:
        details["gt_loocv_actual_endpoint_tg_c"] = float("nan")
        details["kwei_loocv_actual_endpoint_tg_c"] = float("nan")
        details["kwei_fit_actual_endpoint_tg_c"] = float("nan")
        k_global, q_global = float("nan"), float("nan")

    methods = [
        "base_pred_tg_c",
        "pred_tg_c",
        "fox_actual_endpoint_tg_c",
        "gt_est_actual_endpoint_tg_c",
        "gt_loocv_actual_endpoint_tg_c",
        "kwei_loocv_actual_endpoint_tg_c",
        "kwei_fit_actual_endpoint_tg_c",
        "fox_model_endpoint_tg_c",
        "gt_est_model_endpoint_tg_c",
    ]
    y_true = details["Tg_C_actual"].to_numpy(dtype=float) + 273.15
    overall = {
        method: _metrics(y_true, details[method].to_numpy(dtype=float) + 273.15)
        for method in methods
        if method in details
    }

    by_base: Dict[str, Dict[str, Dict[str, float]]] = {}
    for base, group in details.groupby("Nucleobase", dropna=False):
        y_group = group["Tg_C_actual"].to_numpy(dtype=float) + 273.15
        by_base[str(base)] = {
            method: _metrics(y_group, group[method].to_numpy(dtype=float) + 273.15)
            for method in methods
            if method in group
        }

    summary = {
        "n_random_rows": int(len(details)),
        "endpoint_actual_tg_c": {key: value - 273.15 for key, value in endpoints_actual.items()},
        "endpoint_model_tg_c": {key: value - 273.15 for key, value in endpoints_model.items()},
        "kwei_fit_actual_endpoint": {"k": k_global, "q": q_global},
        "overall": overall,
        "by_nucleobase": by_base,
    }
    return details, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate nucleobase copolymer Tg strategies.")
    parser.add_argument(
        "--predictions-csv",
        default=_default_path("results", "copolymer_residual_model", "nucleobase_excel_predictions.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=_default_path("results", "copolymer_residual_model", "nucleobase_strategy_details.csv"),
    )
    parser.add_argument(
        "--summary-json",
        default=_default_path("results", "copolymer_residual_model", "nucleobase_strategy_summary.json"),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    predictions = pd.read_csv(args.predictions_csv)
    details, summary = evaluate_strategies(predictions)

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
