"""
Train a copolymer Tg model from virtual pretraining data plus optional real-data residual fine-tuning.

Recommended workflow:
    1. Pretrain a base model on generated virtual copolymer rows.
    2. Fit a small residual corrector on manually collected real copolymer rows:
           Tg_real = base_virtual_model(features) + residual(features, base_prediction)

Supported input rows:
    - Virtual generator CSV: components_serialized, weights_serialized, architecture, tg_k_pred
    - Manual CSV: components column with "*A*::0.3|*B*::0.7"
    - Manual CSV: smiles1/w1/smiles2/w2 or SMILES_1/SMILES_2/ratio_1/Tg_C
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_virtual_copolymer_dataset import build_inference_paths
from scripts.predict_tg_tabpfn_186d import BestTgPredictor, _require_paths
from src.ml.copolymer_tg_model import (
    build_feature_matrix,
    fit_residual_corrector,
    make_base_regressor,
    predict_with_residual,
    read_copolymer_records,
    regression_metrics,
)


def _default_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def _progress(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain and residual-finetune a copolymer Tg model."
    )
    parser.add_argument("--virtual-csv", required=True, help="Virtual copolymer CSV from generator.")
    parser.add_argument("--real-csv", default=None, help="Optional manually collected real copolymer CSV.")
    parser.add_argument("--output-dir", default=_default_path("results", "copolymer_residual_model"))
    parser.add_argument("--virtual-target", default="tg_k_pred")
    parser.add_argument("--real-target", default=None)
    parser.add_argument("--base-model", choices=["hgb", "extra_trees", "ridge"], default="hgb")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--max-virtual-rows", type=int, default=None)
    parser.add_argument("--max-real-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-teacher-residual-scalars",
        action="store_true",
        help="Do not add current teacher/Fox scalar predictions to residual fine-tuning features.",
    )
    parser.add_argument(
        "--precomputed-real-components-only",
        action="store_true",
        help=(
            "For real-data fine-tuning, skip rows containing components that are not "
            "available through exact/canonical 7k precomputed feature reuse."
        ),
    )

    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--chain-physics-confs", type=int, default=50)
    parser.add_argument("--polybert-batch-size", type=int, default=64)
    parser.add_argument("--data-path", type=str, default=_default_path("data", "unified_tg.parquet"))
    parser.add_argument("--phyc-cache", type=str, default=_default_path("data", "feature_matrix_PHY-C.parquet"))
    parser.add_argument("--gnn-cache", type=str, default=_default_path("data", "gnn_embeddings_64d.parquet"))
    parser.add_argument("--pbert-cache", type=str, default=_default_path("data", "polybert_embeddings.parquet"))
    parser.add_argument(
        "--chain-physics-cache",
        type=str,
        default=_default_path("data", "chain_physics_features.parquet"),
    )
    parser.add_argument("--polybert-model-dir", type=str, default=_default_path("data", "polybert_model"))
    parser.add_argument("--gnn-checkpoint", type=str, default=_default_path("checkpoints", "gnn_pretrained.pt"))
    return parser


def _sample_records(records, max_rows: Optional[int], random_state: int):
    if max_rows is None or len(records) <= max_rows:
        return records
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(records), size=max_rows, replace=False))
    return [records[int(index)] for index in indices]


def _train_base_model(x: np.ndarray, y: np.ndarray, args) -> tuple[object, Dict[str, Dict[str, float]]]:
    from sklearn.model_selection import train_test_split

    model = make_base_regressor(args.base_model, random_state=args.random_state)
    metrics: Dict[str, Dict[str, float]] = {}
    if len(y) >= 10 and 0.0 < args.test_size < 0.5:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        model.fit(x_train, y_train)
        metrics["virtual_train"] = regression_metrics(y_train, model.predict(x_train))
        metrics["virtual_valid"] = regression_metrics(y_valid, model.predict(x_valid))
    else:
        model.fit(x, y)
        metrics["virtual_train"] = regression_metrics(y, model.predict(x))
    model.fit(x, y)
    metrics["virtual_full"] = regression_metrics(y, model.predict(x))
    return model, metrics


def _real_cv_metrics(
    residual_x: np.ndarray,
    y_real: np.ndarray,
    base_pred: np.ndarray,
    cv_splits: int,
    random_state: int,
) -> Optional[Dict[str, float]]:
    if cv_splits < 2 or len(y_real) < cv_splits:
        return None
    from sklearn.model_selection import KFold

    corrected = np.full(len(y_real), np.nan, dtype=float)
    splitter = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    for train_idx, valid_idx in splitter.split(residual_x):
        model = fit_residual_corrector(
            residual_x[train_idx],
            y_real[train_idx],
            base_pred[train_idx],
        )
        corrected[valid_idx] = predict_with_residual(
            base_pred[valid_idx],
            model,
            residual_x[valid_idx],
        )
    return regression_metrics(y_real, corrected)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _filter_records_with_precomputed_components(records, predictor) -> Tuple[list, pd.DataFrame]:
    kept = []
    skipped_rows: List[Dict[str, object]] = []
    reusable = {"exact", "canonical"}
    for index, record in enumerate(records):
        matches = [predictor.precomputed_component_match(component) for component in record.components]
        missing = [
            (component, status)
            for component, (status, _) in zip(record.components, matches)
            if status not in reusable
        ]
        if missing:
            skipped_rows.append(
                {
                    "record_index": index,
                    "components": "|".join(record.components),
                    "weights": "|".join(f"{weight:.8g}" for weight in record.weights),
                    "missing_components": "|".join(component for component, _ in missing),
                    "missing_statuses": "|".join(status for _, status in missing),
                    "target_tg_k": record.target_tg_k,
                }
            )
            continue
        kept.append(record)
    return kept, pd.DataFrame(skipped_rows)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = build_inference_paths(args)
    _require_paths(paths)

    _progress("Loading virtual copolymer records...")
    virtual_records = read_copolymer_records(Path(args.virtual_csv), target_column=args.virtual_target)
    virtual_records = _sample_records(virtual_records, args.max_virtual_rows, args.random_state)
    if not virtual_records:
        raise ValueError("No valid virtual copolymer rows found.")
    _progress(f"Virtual records: {len(virtual_records)}")

    predictor = BestTgPredictor(
        paths=paths,
        device=args.device,
        chain_physics_confs=args.chain_physics_confs,
        polybert_batch_size=args.polybert_batch_size,
    )
    predictor.fit()

    _progress("Building virtual base feature matrix...")
    virtual_features = build_feature_matrix(
        virtual_records,
        predictor,
        include_teacher_scalars=False,
    )
    if virtual_features.errors:
        _progress(f"Virtual featurization skipped {len(virtual_features.errors)} rows.")
    _progress(f"Virtual matrix: n={virtual_features.X.shape[0]}, d={virtual_features.X.shape[1]}")

    _progress(f"Training base model: {args.base_model}")
    base_model, metrics = _train_base_model(virtual_features.X, virtual_features.y, args)
    artifact: Dict[str, object] = {
        "base_model": base_model,
        "residual_model": None,
        "base_feature_names": virtual_features.feature_names,
        "residual_feature_names": None,
        "config": vars(args),
        "metrics": metrics,
    }

    real_predictions_path = None
    if args.real_csv:
        _progress("Loading real copolymer records...")
        real_records = read_copolymer_records(Path(args.real_csv), target_column=args.real_target)
        real_records = _sample_records(real_records, args.max_real_rows, args.random_state)
        if not real_records:
            raise ValueError("No valid real copolymer rows found.")
        _progress(f"Real records: {len(real_records)}")
        if args.precomputed_real_components_only:
            real_records, skipped_real = _filter_records_with_precomputed_components(
                real_records,
                predictor,
            )
            skipped_path = output_dir / "skipped_real_missing_precomputed_components.csv"
            if not skipped_real.empty:
                skipped_real.to_csv(skipped_path, index=False, encoding="utf-8-sig")
                _progress(f"Saved skipped real records: {skipped_path}")
            _progress(
                "Precomputed real component filter: "
                f"kept={len(real_records)}, skipped={len(skipped_real)}"
            )
            if not real_records:
                raise ValueError("No real copolymer rows remain after precomputed component filtering.")

        include_teacher = not args.no_teacher_residual_scalars
        _progress("Building real residual feature matrix...")
        residual_features = build_feature_matrix(
            real_records,
            predictor,
            include_teacher_scalars=include_teacher,
        )
        if residual_features.errors:
            _progress(f"Real residual featurization skipped {len(residual_features.errors)} rows.")

        _progress("Building aligned real base feature matrix...")
        real_base_features = build_feature_matrix(
            residual_features.records,
            predictor,
            include_teacher_scalars=False,
        )
        if real_base_features.feature_names != virtual_features.feature_names:
            raise ValueError("Real base feature names do not match virtual base feature names.")

        base_pred_real = base_model.predict(real_base_features.X)
        metrics["real_base"] = regression_metrics(real_base_features.y, base_pred_real)
        cv_metrics = _real_cv_metrics(
            residual_features.X,
            residual_features.y,
            base_pred_real,
            args.cv_splits,
            args.random_state,
        )
        if cv_metrics is not None:
            metrics["real_residual_cv"] = cv_metrics

        residual_model = fit_residual_corrector(
            residual_features.X,
            residual_features.y,
            base_pred_real,
        )
        corrected_real = predict_with_residual(base_pred_real, residual_model, residual_features.X)
        metrics["real_residual_train"] = regression_metrics(residual_features.y, corrected_real)
        artifact["residual_model"] = residual_model
        artifact["residual_feature_names"] = residual_features.feature_names

        prediction_frame = pd.DataFrame(
            {
                "target_tg_k": residual_features.y,
                "target_tg_c": residual_features.y - 273.15,
                "base_pred_tg_k": base_pred_real,
                "base_pred_tg_c": base_pred_real - 273.15,
                "residual_pred_tg_k": corrected_real,
                "residual_pred_tg_c": corrected_real - 273.15,
                "residual_k": corrected_real - base_pred_real,
            }
        )
        real_predictions_path = output_dir / "real_predictions.csv"
        prediction_frame.to_csv(real_predictions_path, index=False, encoding="utf-8-sig")

    model_path = output_dir / "copolymer_tg_residual_model.joblib"
    metrics_path = output_dir / "metrics.json"
    joblib.dump(artifact, model_path)
    _write_json(metrics_path, metrics)

    _progress(f"Saved model: {model_path}")
    _progress(f"Saved metrics: {metrics_path}")
    if real_predictions_path:
        _progress(f"Saved real predictions: {real_predictions_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
