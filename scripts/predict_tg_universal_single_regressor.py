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

from scripts.train_universal_tg_single_regressor import (
    ComponentFeatureFactory,
    _finite_float,
    _parse_components_and_weights,
    build_table_from_records,
)
from src.ml.universal_tg_features import ComponentRecord, PolymerRecord


@dataclass(frozen=True)
class ModelBundle:
    model: object
    feature_columns: list[str]
    model_dir: Path
    summary: dict


def load_model_bundle(model_dir: Path) -> ModelBundle:
    model_path = model_dir / "model.joblib"
    feature_path = model_dir / "feature_columns.json"
    summary_path = model_dir / "summary.json"
    if not model_path.exists():
        raise FileNotFoundError(f"missing model artifact: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"missing feature schema: {feature_path}")
    model = joblib.load(model_path)
    feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return ModelBundle(
        model=model,
        feature_columns=list(feature_columns),
        model_dir=model_dir,
        summary=summary,
    )


def predict_feature_frame(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = frame.copy()
    for column in bundle.feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    pred_c = bundle.model.predict(out[bundle.feature_columns])
    out["tg_c_pred"] = np.asarray(pred_c, dtype=float)
    out["tg_k_pred"] = out["tg_c_pred"] + 273.15
    out["model_dir"] = str(bundle.model_dir)
    out["model_type"] = str(bundle.summary.get("model", "unknown"))
    return out


def _endpoint_for_row(row: pd.Series, index: int) -> tuple[Optional[float], str]:
    keys = [
        f"endpoint_{index}_tg_c",
        f"endpoint_tg_c_{index}",
        f"tg_c_{index}",
    ]
    for key in keys:
        value = _finite_float(row.get(key))
        if value is not None:
            return value, "provided"
    return None, "missing"


def build_feature_frame_from_query(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    feature_layer = str(bundle.summary.get("feature_layer", "M2M-V"))
    morgan_bits = int(bundle.summary.get("morgan_bits", 256))
    factory = ComponentFeatureFactory(layer=feature_layer, morgan_bits=morgan_bits)
    records: list[PolymerRecord] = []
    errors: list[dict[str, object]] = []

    for idx, row in frame.iterrows():
        sample_id = str(row.get("sample_id") or row.get("case_id") or idx)
        try:
            components, weights = _parse_components_and_weights(row)
            if not components and str(row.get("smiles", "")).strip():
                components, weights = [str(row.get("smiles")).strip()], [1.0]
            if not components:
                raise ValueError("no components found")
            architecture = str(row.get("architecture", "homo" if len(components) == 1 else "random")).strip().lower()
            if architecture == "homopolymer":
                architecture = "homo"
            if architecture not in {"homo", "random", "block", "multicomponent"}:
                architecture = "random" if len(components) == 2 else "multicomponent"
            comp_records = []
            for comp_idx, smiles in enumerate(components, start=1):
                endpoint, endpoint_source = _endpoint_for_row(row, comp_idx)
                comp_records.append(
                    ComponentRecord(
                        smiles=smiles,
                        vector=factory.vector(smiles),
                        endpoint_tg_c=endpoint,
                        endpoint_source=endpoint_source,
                    )
                )
            records.append(
                PolymerRecord(
                    sample_id=sample_id,
                    source=str(row.get("source", "query")),
                    architecture=architecture,
                    components=comp_records,
                    weights=weights,
                    target_tg_c=_finite_float(row.get("target_tg_c")),
                    split_group=str(row.get("split_group", "")) or None,
                )
            )
        except Exception as exc:
            errors.append({"sample_id": sample_id, "status": "error", "error": str(exc)})

    feature_frame = build_table_from_records(records) if records else pd.DataFrame()
    if errors:
        error_frame = pd.DataFrame(errors)
        if feature_frame.empty:
            return error_frame
        feature_frame = pd.concat([feature_frame, error_frame], ignore_index=True, sort=False)
    return feature_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Tg with a saved universal single-regressor model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features-csv", help="CSV already using the saved feature schema.")
    parser.add_argument("--input-csv", help="CSV with smiles/components and weights; features are built on the fly.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.features_csv and not args.input_csv:
        raise SystemExit("Provide either --features-csv or --input-csv.")
    bundle = load_model_bundle(Path(args.model_dir))
    if args.features_csv:
        feature_frame = pd.read_csv(args.features_csv)
    else:
        query = pd.read_csv(args.input_csv)
        feature_frame = build_feature_frame_from_query(query, bundle)

    ok = feature_frame[~feature_frame.get("status", pd.Series(index=feature_frame.index, dtype=str)).astype(str).eq("error")].copy()
    pred = predict_feature_frame(ok, bundle) if not ok.empty else pd.DataFrame()
    errors = feature_frame[feature_frame.get("status", pd.Series(index=feature_frame.index, dtype=str)).astype(str).eq("error")].copy()
    output_frame = pd.concat([pred, errors], ignore_index=True, sort=False) if not errors.empty else pred
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
