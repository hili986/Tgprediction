"""
Advanced PoLyInfo copolymer Tg calibration experiments.

This script builds on ``evaluate_polyinfo_copolymer_physics.py`` outputs.  It
does not recompute endpoints.  It evaluates a practical "known-system" mode:

    target prediction = global baseline + same-COID local residual correction

The correction is an inverse-distance-weighted average of residuals from nearby
composition points in the same copolymer system.  This is appropriate when a
copolymer family already has calibration data and the task is to predict a new
composition.  It is not a leave-system-out new-family predictor.
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

from src.ml.copolymer_tg_model import regression_metrics


DEFAULT_METHODS = [
    "fox_endpoint_tg_c",
    "linear_fox_leave_system_out_endpoint_tg_c",
    "physics_ridge_leave_system_out_endpoint_tg_c",
    "physics_ridge_loocv_endpoint_tg_c",
    "kwei_same_system_pref_loocv_endpoint_tg_c",
]


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _finite_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_metrics(y_c: Iterable[float], pred_c: Iterable[float]) -> Optional[Dict[str, float]]:
    y = np.asarray(list(y_c), dtype=float)
    pred = np.asarray(list(pred_c), dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) == 0:
        return None
    return regression_metrics(y[mask] + 273.15, pred[mask] + 273.15)


def _load_evaluation_frame(details_csv: Path, include_pure_endpoints: bool) -> pd.DataFrame:
    details = pd.read_csv(details_csv)
    numeric_cols = [
        "w1_used",
        "w2_used",
        "Tg_C",
        "ratio_1",
        *DEFAULT_METHODS,
    ]
    for col in numeric_cols:
        if col in details.columns:
            details[col] = pd.to_numeric(details[col], errors="coerce")

    usable = details[
        details["status"].astype(str).eq("usable")
        & np.isfinite(details["Tg_C"])
        & np.isfinite(details["w1_used"])
    ].copy()
    usable = usable.reset_index(drop=False).rename(columns={"index": "details_row_index"})

    if include_pure_endpoints:
        return usable.reset_index(drop=True)
    return usable[
        (usable["w1_used"].astype(float) > 1e-9)
        & (usable["w1_used"].astype(float) < 1.0 - 1e-9)
    ].copy().reset_index(drop=True)


def _mark_duplicate_conflicts(frame: pd.DataFrame, threshold_k: float) -> pd.DataFrame:
    out = frame.copy()
    grouped = (
        out.groupby(["COID", "w1_used"], dropna=False)["Tg_C"]
        .agg(["size", "mean", "std", "min", "max"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["duplicate_conflict"] = (grouped["size"] > 1) & (grouped["std"] > threshold_k)
    conflict_keys = {
        (row.COID, float(row.w1_used))
        for row in grouped[grouped["duplicate_conflict"]].itertuples(index=False)
    }
    out["duplicate_group_size"] = out.apply(
        lambda row: int(
            grouped[
                (grouped["COID"] == row["COID"]) & (grouped["w1_used"] == row["w1_used"])
            ]["size"].iloc[0]
        ),
        axis=1,
    )
    out["duplicate_group_tg_std_k"] = out.apply(
        lambda row: float(
            grouped[
                (grouped["COID"] == row["COID"]) & (grouped["w1_used"] == row["w1_used"])
            ]["std"].iloc[0]
        ),
        axis=1,
    )
    out["high_conflict_duplicate_group"] = out.apply(
        lambda row: (row["COID"], float(row["w1_used"])) in conflict_keys,
        axis=1,
    )
    return out


def _idw_residual(
    target_w: float,
    calibration_w: np.ndarray,
    residuals: np.ndarray,
    k: int,
    power: float,
) -> Optional[float]:
    mask = np.isfinite(calibration_w) & np.isfinite(residuals)
    calibration_w = calibration_w[mask]
    residuals = residuals[mask]
    if len(residuals) == 0:
        return None

    distance = np.abs(calibration_w - target_w)
    exact = distance < 1e-12
    if bool(np.any(exact)):
        return float(np.mean(residuals[exact]))

    order = np.argsort(distance)[: min(k, len(distance))]
    weights = 1.0 / (np.power(distance[order], power) + 1e-9)
    return float(np.sum(weights * residuals[order]) / np.sum(weights))


def _local_residual_predictions(
    frame: pd.DataFrame,
    base_col: str,
    k: int,
    power: float,
    exclude_same_composition: bool,
) -> list[float]:
    predictions: list[float] = []
    for _, row in frame.iterrows():
        base_value = _finite_float(row.get(base_col))
        if base_value is None:
            predictions.append(float("nan"))
            continue

        calibration = frame[frame["COID"].astype(str).eq(str(row["COID"]))].copy()
        calibration = calibration[
            calibration["details_row_index"].astype(int) != int(row["details_row_index"])
        ]
        if exclude_same_composition:
            calibration = calibration[
                np.abs(calibration["w1_used"].astype(float) - float(row["w1_used"])) > 1e-6
            ]

        calibration = calibration[np.isfinite(calibration[base_col])].copy()
        if calibration.empty:
            predictions.append(base_value)
            continue

        residuals = calibration["Tg_C"].to_numpy(dtype=float) - calibration[base_col].to_numpy(
            dtype=float
        )
        correction = _idw_residual(
            float(row["w1_used"]),
            calibration["w1_used"].to_numpy(dtype=float),
            residuals,
            k=k,
            power=power,
        )
        predictions.append(base_value if correction is None else base_value + correction)
    return predictions


def _metrics_block(frame: pd.DataFrame, methods: list[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method in methods:
        if method not in frame.columns:
            continue
        metrics = _finite_metrics(frame["Tg_C"], frame[method])
        if metrics is not None:
            out[method] = metrics
    return out


def evaluate_advanced(
    details_csv: Path,
    base_col: str,
    local_k: int,
    local_power: float,
    conflict_std_threshold_k: float,
    include_pure_endpoints: bool,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    frame = _load_evaluation_frame(details_csv, include_pure_endpoints)
    if base_col not in frame.columns:
        raise ValueError(f"Base column '{base_col}' not found in {details_csv}.")
    frame = _mark_duplicate_conflicts(frame, conflict_std_threshold_k)

    row_col = f"{base_col}_same_system_resid_idw_row_tg_c"
    composition_col = f"{base_col}_same_system_resid_idw_leave_composition_tg_c"
    frame[row_col] = _local_residual_predictions(
        frame,
        base_col=base_col,
        k=local_k,
        power=local_power,
        exclude_same_composition=False,
    )
    frame[composition_col] = _local_residual_predictions(
        frame,
        base_col=base_col,
        k=local_k,
        power=local_power,
        exclude_same_composition=True,
    )

    clean = frame[~frame["high_conflict_duplicate_group"]].copy()
    clean_row_col = f"{base_col}_same_system_resid_idw_row_clean_calibration_tg_c"
    clean_composition_col = (
        f"{base_col}_same_system_resid_idw_leave_composition_clean_calibration_tg_c"
    )
    clean[clean_row_col] = _local_residual_predictions(
        clean,
        base_col=base_col,
        k=local_k,
        power=local_power,
        exclude_same_composition=False,
    )
    clean[clean_composition_col] = _local_residual_predictions(
        clean,
        base_col=base_col,
        k=local_k,
        power=local_power,
        exclude_same_composition=True,
    )
    frame[clean_row_col] = np.nan
    frame[clean_composition_col] = np.nan
    frame.loc[clean.index, clean_row_col] = clean[clean_row_col]
    frame.loc[clean.index, clean_composition_col] = clean[clean_composition_col]

    raw_methods = [*DEFAULT_METHODS, row_col, composition_col]
    clean_methods = [*DEFAULT_METHODS, clean_row_col, clean_composition_col]
    conflict_groups = (
        frame[frame["high_conflict_duplicate_group"]]
        .groupby(["COID", "name", "w1_used"], dropna=False)
        .agg(
            n=("Tg_C", "size"),
            tg_mean_c=("Tg_C", "mean"),
            tg_std_k=("Tg_C", "std"),
            tg_min_c=("Tg_C", "min"),
            tg_max_c=("Tg_C", "max"),
        )
        .reset_index()
        .sort_values("tg_std_k", ascending=False)
    )

    summary: Dict[str, object] = {
        "details_csv": str(details_csv),
        "include_pure_endpoints": include_pure_endpoints,
        "base_col": base_col,
        "local_k": local_k,
        "local_power": local_power,
        "conflict_std_threshold_k": conflict_std_threshold_k,
        "n_evaluated_raw": int(len(frame)),
        "n_evaluated_clean": int(len(clean)),
        "n_high_conflict_rows": int(frame["high_conflict_duplicate_group"].sum()),
        "n_high_conflict_groups": int(len(conflict_groups)),
        "raw": _metrics_block(frame, raw_methods),
        "clean_drop_high_conflict_duplicate_groups": _metrics_block(clean, clean_methods),
        "high_conflict_groups": conflict_groups.to_dict(orient="records"),
        "notes": [
            "same_system_resid_idw_row excludes only the target row; duplicate same-composition measurements can calibrate each other.",
            "same_system_resid_idw_leave_composition also excludes same-composition rows and is the more conservative known-system interpolation metric.",
            "clean metrics drop composition groups whose duplicate Tg standard deviation exceeds conflict_std_threshold_k from both evaluation and local calibration.",
            "These local calibration metrics are not leave-system-out estimates for unseen copolymer families.",
        ],
    }
    return frame, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate advanced PoLyInfo copolymer calibration.")
    parser.add_argument(
        "--details-csv",
        default=_default_path("results", "copolymer_residual_model", "polyinfo_physics_details.csv"),
    )
    parser.add_argument(
        "--base-col",
        default="linear_fox_leave_system_out_endpoint_tg_c",
        help="Global baseline column to residual-correct with same-system local data.",
    )
    parser.add_argument("--local-k", type=int, default=3)
    parser.add_argument("--local-power", type=float, default=1.0)
    parser.add_argument("--conflict-std-threshold-k", type=float, default=10.0)
    parser.add_argument("--include-pure-endpoints", action="store_true")
    parser.add_argument(
        "--output-csv",
        default=_default_path(
            "results",
            "copolymer_residual_model",
            "polyinfo_advanced_calibration_details.csv",
        ),
    )
    parser.add_argument(
        "--summary-json",
        default=_default_path(
            "results",
            "copolymer_residual_model",
            "polyinfo_advanced_calibration_summary.json",
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    details, summary = evaluate_advanced(
        details_csv=Path(args.details_csv),
        base_col=args.base_col,
        local_k=args.local_k,
        local_power=args.local_power,
        conflict_std_threshold_k=args.conflict_std_threshold_k,
        include_pure_endpoints=args.include_pure_endpoints,
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
